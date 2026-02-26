"""
services/video.py
-----------------
Video processing pipeline for pix2pix IR translation.

Responsibilities:
  - Frame extraction & FPS downsampling  (extract_frames)
  - Frame-by-frame pix2pix translation   (translate_frames)
  - Detection with rate limiting         (detect_frames)
  - MP4 reconstruction                   (build_video)
"""

import logging
import time
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

import cv2
import torch
from PIL import Image

from services.detection import run_detection
from utils.transforms import postprocess_to_pil, preprocess_to_pil, preprocess_to_tensor

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Frame Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: Path,
    fps_cap: float = 10.0,
) -> Tuple[List[Image.Image], float]:
    """
    Read a video file and return (frames, effective_fps).

    Frames are downsampled so the output FPS ≤ fps_cap.
    Each frame is returned as a PIL RGB Image.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate how many source frames to skip between kept frames
    effective_fps = min(src_fps, fps_cap)
    step = max(1, int(round(src_fps / effective_fps)))

    frames: List[Image.Image] = []
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        frame_idx += 1

    cap.release()

    log.info(
        "Extracted %d frames from %s  (src_fps=%.1f → out_fps=%.1f, step=%d)",
        len(frames), video_path.name, src_fps, effective_fps, step,
    )
    return frames, effective_fps


# ──────────────────────────────────────────────────────────────────────────────
# 2. IR Translation
# ──────────────────────────────────────────────────────────────────────────────

def translate_frames(
    frames: List[Image.Image],
    netG: torch.nn.Module,
    device: torch.device,
    crop_size: int = 256,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Run pix2pix on each frame.

    Returns (input_pils, ir_pils):
      - input_pils: preprocessed (cropped/resized) source frames
      - ir_pils: generated IR frames
    """
    input_pils: List[Image.Image] = []
    ir_pils: List[Image.Image] = []
    total = len(frames)

    for i, frame in enumerate(frames):
        tensor = (
            preprocess_to_tensor(frame, crop_size=crop_size)
            .unsqueeze(0)
            .to(device)
        )

        with torch.inference_mode():
            fake = netG(tensor)

        input_pil = preprocess_to_pil(frame, crop_size=crop_size)
        ir_pil    = postprocess_to_pil(fake.squeeze(0).cpu())

        input_pils.append(input_pil)
        ir_pils.append(ir_pil)

        if progress_cb:
            progress_cb(i + 1, total)

    log.info("Translated %d frames", total)
    return input_pils, ir_pils


# ──────────────────────────────────────────────────────────────────────────────
# 3. Object Detection with Rate Limiting
# ──────────────────────────────────────────────────────────────────────────────

def detect_frames(
    ir_frames: List[Image.Image],
    api_url: str,
    api_key: str,
    model_id: str,
    delay_secs: float = 0.5,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Image.Image]:
    """
    Run object detection on each IR frame with rate limiting.

    Returns annotated frames. If detection fails for a frame the raw IR frame
    is used instead (never crashes the whole pipeline).
    """
    annotated: List[Image.Image] = []
    total = len(ir_frames)

    for i, frame in enumerate(ir_frames):
        annotated_frame, _, error = run_detection(
            image    = frame,
            api_url  = api_url,
            api_key  = api_key,
            model_id = model_id,
        )
        if error:
            log.warning("Detection frame %d failed: %s — using raw IR frame", i, error)
            annotated_frame = frame

        annotated.append(annotated_frame)

        if progress_cb:
            progress_cb(i + 1, total)

        # Rate limiting: pause between API calls
        if i < total - 1:
            time.sleep(delay_secs)

    return annotated


# ──────────────────────────────────────────────────────────────────────────────
# 4. Video Reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def build_video(
    frames: List[Image.Image],
    fps: float,
    output_path: Path,
) -> None:
    """
    Encode a list of PIL RGB Images into a browser-compatible H.264 MP4.

    Strategy:
      1. imageio + ffmpeg (libx264 — proper H.264, best browser compat)
      2. OpenCV avc1/H264 codec
      3. Last resort: OpenCV mp4v
    """
    import numpy as np

    if not frames:
        raise ValueError("Cannot build video: no frames provided")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np_frames = [np.array(f.convert("RGB"), dtype=np.uint8) for f in frames]
    w, h = frames[0].size

    # ── Strategy 1: imageio + ffmpeg ──────────────────────────────────────
    try:
        import imageio.v3 as iio
        iio.imwrite(
            str(output_path),
            np_frames,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",   # Required for browser playback
            output_params=["-preset", "fast", "-crf", "23"],
        )
        log.info("Built video via imageio/libx264: %s  (%d frames @ %.1f fps)",
                 output_path.name, len(np_frames), fps)
        return
    except Exception as e:
        log.warning("imageio/libx264 failed (%s) — trying OpenCV", e)

    # ── Strategy 2/3: OpenCV fallback ─────────────────────────────────────
    for fourcc_str in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        if writer.isOpened():
            for rgb in np_frames:
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            writer.release()
            log.info("Built video via OpenCV/%s: %s  (%d frames @ %.1f fps)",
                     fourcc_str, output_path.name, len(np_frames), fps)
            return
        writer.release()

    raise RuntimeError("All video encoding strategies failed.")

