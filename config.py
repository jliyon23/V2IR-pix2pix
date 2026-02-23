from pathlib import Path

BASE_DIR = Path(__file__).parent

CHECKPOINT_PATH = BASE_DIR.parent / "models" / "final.pth"
NGF             = 64
CROP_SIZE       = 256

HOST           = "0.0.0.0"
PORT           = 5000
MAX_UPLOAD_MB  = 16
ALLOWED_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

ROBOFLOW_API_URL  = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY  = "mSVHECzK6nJeSSQmZXj2"
ROBOFLOW_MODEL_ID = "flir-data-set/22"
