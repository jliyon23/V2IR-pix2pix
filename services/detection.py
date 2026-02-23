import base64
import io
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

_INFER_URL = "https://serverless.roboflow.com/{model_id}"
_TIMEOUT   = 15

_CLASS_COLORS: dict = {
    "person":   (255,  55, 110),
    "car":      ( 55, 175, 255),
    "bicycle":  ( 80, 240, 120),
    "dog":      (255, 200,  55),
    "truck":    (180,  55, 255),
    "bus":      (255, 145,   0),
}
_DEFAULT_COLOR: Tuple[int, int, int] = (255, 107, 53)

_BOX_THICKNESS = 2
_FONT_SIZE     = 13
_LABEL_PAD     = 3

@dataclass
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str
    class_id: int
    detection_id: str

    @property
    def x1(self) -> int:
        return max(0, int(self.x - self.width  / 2))

    @property
    def y1(self) -> int:
        return max(0, int(self.y - self.height / 2))

    @property
    def x2(self) -> int:
        return int(self.x + self.width  / 2)

    @property
    def y2(self) -> int:
        return int(self.y + self.height / 2)

    def to_dict(self) -> dict:
        return {
            "x1":         self.x1,
            "y1":         self.y1,
            "x2":         self.x2,
            "y2":         self.y2,
            "width":      int(self.width),
            "height":     int(self.height),
            "confidence": round(self.confidence, 4),
            "class_name": self.class_name,
            "class_id":   self.class_id,
        }

def _class_color(name: str) -> Tuple[int, int, int]:
    return _CLASS_COLORS.get(name.lower(), _DEFAULT_COLOR)

def _load_font(size: int):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()

def _annotate(image: Image.Image, detections: List[Detection]) -> Image.Image:
    out  = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _load_font(_FONT_SIZE)

    for det in detections:
        c = _class_color(det.class_name)

        for t in range(_BOX_THICKNESS):
            draw.rectangle(
                [det.x1 + t, det.y1 + t, det.x2 - t, det.y2 - t],
                outline=c,
            )

        label = f"{det.class_name}  {det.confidence:.0%}"
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)

        lx1 = det.x1
        ly1 = max(0, det.y1 - th - _LABEL_PAD * 2)
        lx2 = min(out.width, lx1 + tw + _LABEL_PAD * 2)
        ly2 = ly1 + th + _LABEL_PAD * 2

        draw.rectangle([lx1, ly1, lx2, ly2], fill=c)
        draw.text((lx1 + _LABEL_PAD, ly1 + _LABEL_PAD),
                  label, fill=(255, 255, 255), font=font)

    return out

def run_detection(
    image: Image.Image,
    api_url: str,
    api_key: str,
    model_id: str,
) -> Tuple[Image.Image, List[Detection], Optional[str]]:
    try:

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        url = _INFER_URL.format(model_id=model_id)
        response = requests.post(
            url,
            params={"api_key": api_key},
            data=b64_image,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        raw = response.json()

        detections = [
            Detection(
                x            = float(p["x"]),
                y            = float(p["y"]),
                width        = float(p["width"]),
                height       = float(p["height"]),
                confidence   = float(p["confidence"]),
                class_name   = str(p.get("class", "unknown")),
                class_id     = int(p.get("class_id", -1)),
                detection_id = str(p.get("detection_id", "")),
            )
            for p in raw.get("predictions", [])
        ]

        log.info("Detection OK  |  model=%-20s  objects=%d", model_id, len(detections))
        return _annotate(image, detections), detections, None

    except Exception as exc:
        log.exception("Roboflow detection failed: %s", exc)
        return image, [], str(exc)
