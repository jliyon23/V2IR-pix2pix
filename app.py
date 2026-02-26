import io
import base64
import json
import logging
import queue
import threading
import uuid
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import torch
from flask import Flask, Response, jsonify, render_template, request, send_file
from PIL import Image

import config
from model.generator import build_generator
from services.detection import run_detection
from services.video import (
    extract_frames,
    translate_frames,
    detect_frames,
    build_video,
)
from utils.transforms import postprocess_to_pil, preprocess_to_pil, preprocess_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_generator(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Place final.pth inside the models/ folder."
        )

    netG = build_generator(ngf=config.NGF, use_dropout=True).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("netG", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = netG.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Missing keys in checkpoint: %s", missing)
    if unexpected:
        log.warning("Unexpected keys in checkpoint: %s", unexpected)
    netG.eval()
    log.info("Generator loaded  →  %s  on %s", checkpoint_path.name, device)
    return netG


netG = _load_generator(config.CHECKPOINT_PATH, DEVICE)

# ──────────────────────────────────────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_VIDEO_MB * 1024 * 1024  # Use video limit (larger)

# In-memory job store: {job_id: {"q": Queue, "status": str}}
JOB_STORE: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _validate_file(file) -> Optional[str]:
    if not file or not file.filename:
        return "No file selected."
    ext = Path(file.filename).suffix.lower().split('?')[0]
    if ext not in config.ALLOWED_EXTS:
        return f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(config.ALLOWED_EXTS))}"
    return None


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event string."""
    return f"data: {json.dumps(data)}\n\n"


# ──────────────────────────────────────────────────────────────────────────────
# Image routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    image_url = request.form.get("image_url")
    file = request.files.get("image")

    if image_url:
        try:
            req = urllib.request.Request(
                image_url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                image_bytes = response.read()
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                filename = image_url.split('/')[-1].split('?')[0] or "url_image"
            except Exception as exc:
                log.error("Failed to open downloaded image: %s", exc)
                return jsonify({"error": "Downloaded file is not a valid image."}), 400
        except urllib.error.URLError as e:
            log.error("Failed to fetch image URL: %s", e)
            return jsonify({"error": f"Could not fetch image from URL: {e.reason}"}), 400
        except Exception as e:
            log.error("Unexpected error fetching URL: %s", e)
            return jsonify({"error": "An error occurred while fetching the image URL."}), 400
    else:
        err = _validate_file(file)
        if err:
            return jsonify({"error": err}), 400
        try:
            img = Image.open(file.stream).convert("RGB")
            filename = file.filename
        except Exception as exc:
            log.error("Failed to open image: %s", exc)
            return jsonify({"error": "Could not read the uploaded image."}), 400

    try:
        tensor = (
            preprocess_to_tensor(img, crop_size=config.CROP_SIZE)
            .unsqueeze(0)
            .to(DEVICE)
        )
        with torch.inference_mode():
            fake = netG(tensor)

        output_pil = postprocess_to_pil(fake.squeeze(0).cpu())
        input_pil  = preprocess_to_pil(img, crop_size=config.CROP_SIZE)

        annotated_pil = output_pil
        detections    = []
        det_error     = None

        enable_det = request.form.get("enable_detection", "true") == "true"
        if enable_det:
            annotated_pil, detections, det_error = run_detection(
                image    = output_pil,
                api_url  = config.ROBOFLOW_API_URL,
                api_key  = config.ROBOFLOW_API_KEY,
                model_id = config.ROBOFLOW_MODEL_ID,
            )
            if det_error:
                log.warning("Detection skipped — %s", det_error)

        log.info(
            "Request OK  |  file=%-20s  device=%s  det=%s  detections=%d",
            filename, DEVICE, enable_det, len(detections),
        )

        return jsonify({
            "input":           _pil_to_base64(input_pil),
            "output":          _pil_to_base64(annotated_pil),
            "predictions":     [d.to_dict() for d in detections],
            "detection_count": len(detections),
            "detection_error": det_error,
        })

    except Exception as exc:
        log.exception("Inference failed: %s", exc)
        return jsonify({"error": f"Inference failed: {exc}"}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Video routes
# ──────────────────────────────────────────────────────────────────────────────

def _video_worker(
    job_id: str,
    input_path: Path,
    output_ir_path: Path,
    output_input_path: Path,
    enable_detection: bool,
):
    """Background thread: runs the full video pipeline and emits SSE events."""
    q: queue.Queue = JOB_STORE[job_id]["q"]

    def emit(data: dict):
        q.put(data)

    try:
        # ── Stage 1: Extract frames ──────────────────────────────────────────
        emit({"type": "stage", "stage": "Extracting frames..."})
        frames, fps = extract_frames(input_path, fps_cap=config.VIDEO_FPS_CAP)
        total = len(frames)

        if total == 0:
            emit({"type": "error", "message": "No frames could be extracted from the video."})
            return

        emit({"type": "total", "total": total})

        # ── Stage 2: IR Translation ──────────────────────────────────────────
        emit({"type": "stage", "stage": "Translating frames to IR..."})

        def translation_progress(done, tot):
            emit({"type": "progress", "phase": "translate", "frame": done, "total": tot})

        input_pils, ir_pils = translate_frames(
            frames, netG, DEVICE,
            crop_size   = config.CROP_SIZE,
            progress_cb = translation_progress,
        )

        # ── Stage 3: Detection (optional) ────────────────────────────────────
        output_frames = ir_pils
        if enable_detection:
            emit({"type": "stage", "stage": "Running object detection..."})

            def det_progress(done, tot):
                emit({"type": "progress", "phase": "detect", "frame": done, "total": tot})

            output_frames = detect_frames(
                ir_pils,
                api_url     = config.ROBOFLOW_API_URL,
                api_key     = config.ROBOFLOW_API_KEY,
                model_id    = config.ROBOFLOW_MODEL_ID,
                delay_secs  = config.DETECTION_DELAY,
                progress_cb = det_progress,
            )

        # ── Stage 4: Build output videos ─────────────────────────────────────
        emit({"type": "stage", "stage": "Encoding output video..."})
        build_video(output_frames, fps, output_ir_path)
        build_video(input_pils, fps, output_input_path)

        log.info("Video job %s done — %d frames, fps=%.1f", job_id, total, fps)
        emit({"type": "done", "job_id": job_id})

    except Exception as exc:
        log.exception("Video job %s failed: %s", job_id, exc)
        emit({"type": "error", "message": str(exc)})
    finally:
        JOB_STORE[job_id]["status"] = "done"
        q.put(None)  # Sentinel to close the SSE stream


@app.post("/predict_video")
def predict_video():
    file = request.files.get("video")

    if not file or not file.filename:
        return jsonify({"error": "No video file provided."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXTS:
        return jsonify({
            "error": f"Unsupported video format '{ext}'. Allowed: {', '.join(sorted(config.ALLOWED_VIDEO_EXTS))}"
        }), 400

    job_id = str(uuid.uuid4())

    # Save uploaded video to tmp dir
    config.VIDEO_TMP_DIR.mkdir(parents=True, exist_ok=True)
    input_path        = config.VIDEO_TMP_DIR / f"{job_id}_input{ext}"
    output_ir_path    = config.VIDEO_TMP_DIR / f"{job_id}_ir.mp4"
    output_input_path = config.VIDEO_TMP_DIR / f"{job_id}_input_processed.mp4"

    file.save(str(input_path))

    enable_detection = request.form.get("enable_detection", "true") == "true"

    JOB_STORE[job_id] = {
        "q":      queue.Queue(),
        "status": "running",
        "input_path":  input_path,
        "ir_path":     output_ir_path,
        "inp_path":    output_input_path,
    }

    t = threading.Thread(
        target  = _video_worker,
        args    = (job_id, input_path, output_ir_path, output_input_path, enable_detection),
        daemon  = True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.get("/video_status/<job_id>")
def video_status(job_id: str):
    if job_id not in JOB_STORE:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        q: queue.Queue = JOB_STORE[job_id]["q"]
        while True:
            try:
                event = q.get(timeout=60)
            except queue.Empty:
                yield _sse({"type": "ping"})
                continue

            if event is None:          # Sentinel — stream is done
                break

            yield _sse(event)

    return Response(
        generate(),
        mimetype   = "text/event-stream",
        headers    = {
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/video_result/<job_id>/<which>")
def video_result(job_id: str, which: str):
    """Serve either the 'ir' or 'input' result video."""
    if job_id not in JOB_STORE:
        return jsonify({"error": "Job not found"}), 404

    if which == "ir":
        path = JOB_STORE[job_id]["ir_path"]
    elif which == "input":
        path = JOB_STORE[job_id]["inp_path"]
    else:
        return jsonify({"error": "Use 'ir' or 'input'"}), 400

    if not path.exists():
        return jsonify({"error": "Video not ready yet"}), 404

    return send_file(str(path), mimetype="video/mp4")


@app.delete("/video_cleanup/<job_id>")
def video_cleanup(job_id: str):
    """Clean up temp files for a completed job."""
    if job_id not in JOB_STORE:
        return jsonify({"ok": True})

    job = JOB_STORE.pop(job_id)
    for key in ("input_path", "ir_path", "inp_path"):
        p: Path = job.get(key)
        if p and p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    return jsonify({"ok": True})


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting server  →  http://127.0.0.1:%d", config.PORT)
    app.run(host=config.HOST, port=config.PORT, debug=True)
