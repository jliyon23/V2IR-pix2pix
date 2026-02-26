from pathlib import Path

BASE_DIR = Path(__file__).parent

CHECKPOINT_PATH = BASE_DIR.parent / "models" / "final.pth"
NGF             = 64
CROP_SIZE       = 256

HOST           = "0.0.0.0"
PORT           = 5000
MAX_UPLOAD_MB  = 16
ALLOWED_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Video settings
MAX_VIDEO_MB      = 200
VIDEO_FPS_CAP     = 10          # Downsample source video to at most this FPS
DETECTION_DELAY   = 0.5        # Seconds between Roboflow calls to avoid rate limiting
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VIDEO_TMP_DIR      = BASE_DIR / "tmp_videos"  # Temp dir for input/output video files

ROBOFLOW_API_URL  = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY  = "mSVHECzK6nJeSSQmZXj2"
ROBOFLOW_MODEL_ID = "flir-data-set/22"
