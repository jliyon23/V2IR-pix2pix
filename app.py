import io
import base64
import logging
from pathlib import Path
from typing import Optional

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

import config
from model.generator import build_generator
from services.detection import run_detection
from utils.transforms import postprocess_to_pil, preprocess_to_pil, preprocess_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _validate_file(file) -> Optional[str]:
    if not file or not file.filename:
        return "No file selected."
    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTS:
        return f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(config.ALLOWED_EXTS))}"
    return None

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    file = request.files.get("image")
    err  = _validate_file(file)
    if err:
        return jsonify({"error": err}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
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

        annotated_pil, detections, det_error = run_detection(
            image    = output_pil,
            api_url  = config.ROBOFLOW_API_URL,
            api_key  = config.ROBOFLOW_API_KEY,
            model_id = config.ROBOFLOW_MODEL_ID,
        )

        if det_error:
            log.warning("Detection skipped — %s", det_error)

        log.info(
            "Request OK  |  file=%-20s  device=%s  detections=%d",
            file.filename, DEVICE, len(detections),
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

if __name__ == "__main__":
    log.info("Starting server  →  http://127.0.0.1:%d", config.PORT)
    app.run(host=config.HOST, port=config.PORT, debug=True)
