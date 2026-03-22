# V2IR Project Architecture
## Visible-to-Infrared Translation using Pix2Pix GAN + YOLOv8 Detection

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Pix2Pix GAN — Deep Dive](#3-pix2pix-gan--deep-dive)
   - 3.1 [What is a Conditional GAN?](#31-what-is-a-conditional-gan)
   - 3.2 [U-Net 256 Generator](#32-u-net-256-generator)
   - 3.3 [70×70 PatchGAN Discriminator](#33-7070-patchgan-discriminator)
   - 3.4 [Loss Functions](#34-loss-functions)
   - 3.5 [Training Strategy](#35-training-strategy)
   - 3.6 [Implementation in This Project](#36-implementation-in-this-project)
4. [YOLOv8 Object Detection](#4-yolov8-object-detection)
   - 4.1 [YOLOv8 Architecture](#41-yolov8-architecture)
   - 4.2 [Integration via Roboflow API](#42-integration-via-roboflow-api)
5. [Web Application Architecture](#5-web-application-architecture)
   - 5.1 [Backend: Flask Server](#51-backend-flask-server)
   - 5.2 [Image Pipeline](#52-image-pipeline)
   - 5.3 [Video Pipeline](#53-video-pipeline)
   - 5.4 [Frontend](#54-frontend)
6. [Data & Preprocessing](#6-data--preprocessing)
7. [Module File Map](#7-module-file-map)
8. [End-to-End Data Flow Diagrams](#8-end-to-end-data-flow-diagrams)

---

## 1. Project Overview

This project implements a **Visible-to-Infrared (V2IR) image translation** system using a **Pix2Pix Generative Adversarial Network** trained on the **LLVIP dataset**. The translated infrared output is then passed through a **YOLOv8-based object detector** (hosted via Roboflow) to detect humans, vehicles, and other objects — tasks that are significantly more reliable in thermal/infrared images compared to RGB images.

**Key capabilities:**
- Translate any visible-light image or video to a synthetic infrared representation
- Run person/vehicle/object detection on the IR output
- Stream real-time progress during video processing (SSE)
- Accept uploads, file URLs, and direct image files

---

## 2. High-Level System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         Web Browser (Client)                      │
│       HTML/CSS/JS Frontend  ←──────────────────────────────────── │
│       Image Upload / URL        SSE Progress Stream               │
└──────────────────────────────┬────────────────────────────────────┘
                               │ HTTP/POST
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Flask Web Server (app.py)                   │
│                                                                  │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐   │
│  │  /predict    │   │/predict_video │   │ /video_status    │   │
│  │  (image)     │   │  (video)      │   │ (SSE stream)     │   │
│  └──────┬───────┘   └───────┬───────┘   └────────┬─────────┘   │
│         │                   │                     │              │
│         ▼                   ▼                     │              │
│  ┌────────────────────────────────────────────┐   │              │
│  │          Inference Pipeline                │   │              │
│  │  ┌──────────────────────────────────────┐ │   │              │
│  │  │    Pix2Pix Generator (UNet-256)      │ │◄──┘              │
│  │  │      model/generator.py              │ │                  │
│  │  └──────────────────────────────────────┘ │                  │
│  │                  ▼                         │                  │
│  │  ┌──────────────────────────────────────┐ │                  │
│  │  │   YOLOv8 Detection (Roboflow API)    │ │                  │
│  │  │      services/detection.py           │ │                  │
│  │  └──────────────────────────────────────┘ │                  │
│  └────────────────────────────────────────────┘                  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │         Video Service (services/video.py)                  │  │
│  │  extract_frames → translate_frames → detect_frames →       │  │
│  │  build_video                                               │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                ┌──────────────────────────┐
                │    Roboflow Cloud (API)  │
                │  YOLOv8 (flir-data-set) │
                └──────────────────────────┘
```

---

## 3. Pix2Pix GAN — Deep Dive

### 3.1 What is a Conditional GAN?

Standard GANs generate images from random noise. A **Conditional GAN (cGAN)** additionally conditions generation on an input signal — in this case, a visible-light image. The task becomes **image-to-image translation**: learn a mapping `G: X → Y` where X = visible domain and Y = infrared domain.

Pix2Pix is the canonical image-to-image cGAN, introduced by Isola et al. (2017).

```
Visible Image (x) ─────────────────────────────────────────────────────┐
        │                                                               │
        ▼                                                               ▼
  ┌───────────┐     Fake IR (G(x))    ┌──────────────────────────────────┐
  │ Generator │ ──────────────────►   │     Discriminator D(x, G(x))     │
  │  (U-Net)  │                       │   "Is this pair real or fake?"   │
  └───────────┘                       └──────────────────────────────────┘
        ▲                                          │
        │               Real IR (y) ──────────────►│
        │                                          │
        └──────── Adversarial + L1 Loss ───────────┘
                    (backpropagation)
```

**Key insight:** The discriminator receives the *pair* `(input, output)` — not just the output alone. This forces the generator to produce outputs that are both realistic *and* consistent with the input.

---

### 3.2 U-Net 256 Generator

The generator is a **U-Net** — an encoder-decoder with **skip connections** between symmetric layers. The "256" refers to the input/output resolution: 256×256 pixels.

#### Why U-Net?
A plain encoder-decoder causes information bottleneck at the latent code (compressed representation). Skip connections allow low-level spatial details (edges, textures) to bypass the bottleneck and be reused directly during decoding. For visible→IR translation, this is crucial because many structural details (building edges, body outlines) appear in both domains.

#### Architecture Detail (8-level depth, `ngf=64`)

```
INPUT (3×256×256)  ← RGB visible image
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  ENCODER (Downsampling Path) — all using Conv4×4, stride=2     │
├────────────┬────────────┬────────────────────────────────────── │
│  Block     │  Channels  │  Output Size      │  Notes            │
├────────────┼────────────┼───────────────────┼─────────────────── │
│  Down-1    │   3 → 64  │ 64×128×128        │  Conv only        │
│  Down-2    │  64 → 128  │ 128×64×64        │  LReLU+Conv+BN    │
│  Down-3    │ 128 → 256  │ 256×32×32        │  LReLU+Conv+BN    │
│  Down-4    │ 256 → 512  │ 512×16×16        │  LReLU+Conv+BN    │
│  Down-5    │ 512 → 512  │ 512×8×8          │  LReLU+Conv+BN    │
│  Down-6    │ 512 → 512  │ 512×4×4          │  LReLU+Conv+BN    │
│  Down-7    │ 512 → 512  │ 512×2×2          │  LReLU+Conv+BN    │
│  Down-8    │ 512 → 512  │ 512×1×1          │  Innermost (bottleneck) │
└────────────┴────────────┴───────────────────┴─────────────────── │

              BOTTLENECK: 512×1×1  ← most compressed representation

┌─────────────────────────────────────────────────────────────────┐
│  DECODER (Upsampling Path) — ConvTranspose4×4, stride=2        │
│  Skip connection: cat([decoder_output, encoder_feature])        │
├────────────┬────────────────────────┬──────────────────────── ─ │
│  Block     │  Channels (after cat)  │  Output Size              │
├────────────┼────────────────────────┼───────────────────────── ─ │
│  Up-8      │ 512 → 512              │ 512×2×2     + Dropout(0.5) │
│  Up-7      │ 1024 → 512             │ 512×4×4     + Dropout(0.5) │
│  Up-6      │ 1024 → 512             │ 512×8×8     + Dropout(0.5) │
│  Up-5      │ 1024 → 512             │ 512×16×16                  │
│  Up-4      │ 1024 → 256             │ 256×32×32                  │
│  Up-3      │ 512 → 128              │ 128×64×64                  │
│  Up-2      │ 256 → 64               │ 64×128×128                 │
│  Up-1      │ 128 → 3                │  3×256×256  + Tanh()       │
└────────────┴────────────────────────┴───────────────────────── ─ │

OUTPUT (3×256×256)  ← Synthetic IR image (values in [-1, 1])
```

**Activation functions:**
- **Encoder**: `LeakyReLU(0.2)` — allows small negative gradients, good for discriminator-like downsampling
- **Decoder**: `ReLU` — standard for upsampling
- **Output**: `Tanh()` — maps output to `[-1, 1]` range, normalized to match training targets

**Normalization**: `BatchNorm2d` after every layer except the outermost block (Conv-1) and innermost block (bottleneck).

**Dropout**: Applied at 0.5 probability to the top 3 decoder layers during both *training and inference* — this introduces stochasticity and acts as a form of regularization that the original pix2pix paper used.

#### Skip Connection Mechanism (in code)

```python
# UnetSkipConnectionBlock.forward()
def forward(self, x):
    if self.outermost:
        return self.model(x)           # No skip at outermost
    return torch.cat([x, self.model(x)], dim=1)  # Skip: concat input & decoded
```

The encoder features `x` are concatenated channel-wise with the decoded output before passing to the next decoder block. This doubling of channels is why the decoder ConvTranspose layers take `inner_nc * 2` as input.

---

### 3.3 70×70 PatchGAN Discriminator

Rather than classifying the whole image as real or fake, the PatchGAN discriminator outputs an **N×N grid of predictions**, where each prediction covers a 70×70 pixel receptive field of the image.

**Why PatchGAN?**
- Focuses on high-frequency texture structure
- Locally stationary assumption: statistics of patches repeat across image
- Fewer parameters than full-image discriminator
- Can handle arbitrary input sizes at test time

**Architecture:**

```
Input: cat([visible_x, IR_y]) → 6 channels (3 + 3)
        │
        ▼
  Conv4×4 stride=2 → 64 ch  (NO BN)    LeakyReLU(0.2)
        │
  Conv4×4 stride=2 → 128 ch  + BN      LeakyReLU(0.2)
        │
  Conv4×4 stride=2 → 256 ch  + BN      LeakyReLU(0.2)
        │
  Conv4×4 stride=1 → 512 ch  + BN      LeakyReLU(0.2)
        │
  Conv4×4 stride=1 → 1 ch              (No activation)
        │
        ▼
  Output: 30×30 grid of real/fake scores
          Each score covers ~70×70px receptive field
```

---

### 3.4 Loss Functions

Pix2Pix uses a combination of two losses:

#### Adversarial Loss (GAN Loss)
```
L_GAN(G, D) = E[log D(x, y)] + E[log(1 - D(x, G(x)))]
```
Using **LSGAN** (Least Squares GAN) variant in practice:
- D minimizes: `(D(x,y) - 1)² + (D(x,G(x)) - 0)²`
- G minimizes: `(D(x,G(x)) - 1)²`

This replaces log loss with squared error, producing more stable training gradients.

#### L1 Reconstruction Loss
```
L_L1(G) = E[||y - G(x)||₁]
```
L1 loss penalizes pixel-level differences between the generated IR image and the real IR ground truth. Unlike L2, L1 does not over-smooth outputs.

#### Total Generator Loss
```
L_G = L_GAN + λ × L_L1    (λ = 100 in LLVIP training)
```

The large λ=100 weight means the L1 loss dominates initially, ensuring the generator produces outputs structurally close to real IR. The GAN loss then refines textures and realism.

---

### 3.5 Training Strategy

| Parameter | Value |
|---|---|
| Dataset | LLVIP (15,488 co-registered visible/IR image pairs) |
| Resolution | 256×256 (scale_width_and_crop preprocessing) |
| Epochs | 200 total (100 constant LR + 100 linear LR decay) |
| Batch size | Typically 1 or 4 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| Learning rate | 0.0002 initial → 0 at epoch 200 |
| L1 weight (λ) | 100 |
| Generator filters (ngf) | 64 |

**Training schedule:**
```
Epochs 1–100:   LR = 0.0002 (constant)
Epochs 101–200: LR linearly decays: 0.0002 → 0
```

The checkpoint is saved as `final.pth`, containing:
```python
{
    "netG": <state_dict of UnetGenerator>,
    "netD": <state_dict of NLayerDiscriminator>,  # optional
    "epoch": 200
}
```

---

### 3.6 Implementation in This Project

#### `model/generator.py`

The `build_generator()` factory function creates the production generator:

```python
def build_generator(ngf=64, use_dropout=True) -> UnetGenerator:
    return UnetGenerator(
        input_nc=3,               # RGB input
        output_nc=3,              # RGB output (grayscale-like IR)
        num_downs=8,              # 8 downsampling levels
        ngf=64,                   # 64 base filters
        norm_layer=nn.BatchNorm2d,
        use_dropout=True,         # Kept on in inference (stochastic)
    )
```

The `UnetGenerator` class recursively nests `UnetSkipConnectionBlock` modules:

```
UnetSkipConnectionBlock (outermost)
  └── UnetSkipConnectionBlock (level 2)
        └── UnetSkipConnectionBlock (level 3)
              └── UnetSkipConnectionBlock (level 4)
                    └── UnetSkipConnectionBlock (level 5, dropout)
                          └── UnetSkipConnectionBlock (level 6, dropout)
                                └── UnetSkipConnectionBlock (level 7, dropout)
                                      └── UnetSkipConnectionBlock (innermost)
```

Each non-outermost block returns `torch.cat([x, self.model(x)], dim=1)` — implementing the skip connection inline.

#### Checkpoint Loading (`app.py`)

```python
netG = build_generator(ngf=64, use_dropout=True).to(device)
ckpt = torch.load("models/final.pth", map_location=device)
state_dict = ckpt.get("netG", ckpt)  # Support both dict and raw state_dict
netG.load_state_dict(state_dict, strict=False)
netG.eval()  # Disables training-mode BatchNorm, keeps Dropout active
```

#### Image Preprocessing & Postprocessing (`utils/transforms.py`)

```python
# Input preprocessing
def preprocess_to_tensor(img, crop_size=256):
    img = _scale_to_fit(img, crop_size)       # Maintain aspect ratio, min side = 256
    img = TF.center_crop(img, (256, 256))      # Center crop to exact 256×256
    tensor = TF.to_tensor(img)                 # [0.0, 1.0]
    tensor = tensor * 2.0 - 1.0               # Normalize to [-1.0, 1.0]
    return tensor                              # Shape: (3, 256, 256)

# Output postprocessing
def postprocess_to_pil(tensor):
    tensor = tensor.clamp(-1.0, 1.0)          # Safety clamp
    tensor = (tensor + 1.0) * 0.5             # Remap to [0.0, 1.0]
    return TF.to_pil_image(tensor)            # Convert to PIL Image
```

---

## 4. YOLOv8 Object Detection

### 4.1 YOLOv8 Architecture

YOLOv8 (You Only Look Once v8) by Ultralytics is a **single-stage, anchor-free** object detector. It processes the entire image in a single forward pass and predicts bounding boxes + class labels simultaneously.

#### Core Components

```
Input Image (640×640 normalized)
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  BACKBONE: CSP-Darknet with C2f blocks              │
│  (Cross Stage Partial + bottleneck feature reuse)   │
│                                                     │
│  Stem Conv → Stage1 → Stage2 → Stage3 → Stage4     │
│              (P1)     (P2)    (P3/P4)  (P5)        │
│              ↓        ↓       ↓        ↓            │
│           8×down   16×down  32×down  64×down        │
└──────────────────┬──────────────────────────────────┘
                   │ Multi-scale feature maps
                   ▼
┌─────────────────────────────────────────────────────┐
│  NECK: PAN-FPN (Path Aggregation Network)           │
│                                                     │
│  Top-down path:    P5 → P4 → P3  (high res detail) │
│  Bottom-up path:   P3 → P4 → P5  (semantic context) │
└──────────────────┬──────────────────────────────────┘
                   │ Fused feature maps at 3 scales
                   ▼
┌─────────────────────────────────────────────────────┐
│  HEAD: Decoupled Detection Head (Anchor-Free)       │
│                                                     │
│  Per scale:                                         │
│   ├── Regression branch → x, y, w, h (bbox)        │
│   └── Classification branch → class probabilities  │
│                                                     │
│  NMS: Non-Maximum Suppression (post-processing)     │
└─────────────────────────────────────────────────────┘
          │
          ▼
  Bounding boxes + class + confidence score
```

#### C2f Block (Core Innovation)

```
Input
  │
  ├──────────────────────┐
  │                      │
  ▼                      │
Conv1×1 (half ch)        │ (shortcut)
  │                      │
  ▼                      │
Bottleneck × n           │
  │                      │
  ▼                      │
Concat all branches      │
  │◄─────────────────────┘
  ▼
Conv1×1 (fuse)
  │
Output
```

The C2f block enhances gradient flow by providing multiple shortcut connections, similar to DenseNet but more efficient.

#### Key Differences from YOLOv5

| Feature | YOLOv5 | YOLOv8 |
|---|---|---|
| Anchors | Anchor-based | **Anchor-free** |
| Head | Coupled | **Decoupled** |
| Loss | BCE + IoU | **DFL + CIoU** |
| Backbone | C3 blocks | **C2f blocks** |
| Neck | FPN | **PAN-FPN** |

---

### 4.2 Integration via Roboflow API

The project uses the **Roboflow Serverless API** to run YOLOv8 inference on the translated IR images, avoiding the need to host the model locally.

**Model used:** `flir-data-set/22` — a YOLOv8 model trained on FLIR thermal imagery, which is semantically compatible with our pix2pix-generated pseudo-IR images.

**Service layer (`services/detection.py`):**

```python
def run_detection(image, api_url, api_key, model_id):
    # 1. Encode image to base64 JPEG
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 2. POST to Roboflow serverless endpoint
    url = f"https://serverless.roboflow.com/{model_id}"
    response = requests.post(url, params={"api_key": api_key}, data=b64_image)

    # 3. Parse predictions
    detections = [Detection(x, y, w, h, confidence, class_name) for p in response.json()["predictions"]]

    # 4. Annotate image with colored bounding boxes
    return _annotate(image, detections), detections, None
```

**Detection classes with color coding:**

| Class | Color (RGB) |
|---|---|
| person | (255, 55, 110) — Red-pink |
| car | (55, 175, 255) — Blue |
| bicycle | (80, 240, 120) — Green |
| dog | (255, 200, 55) — Yellow |
| truck | (180, 55, 255) — Purple |
| bus | (255, 145, 0) — Orange |

---

## 5. Web Application Architecture

### 5.1 Backend: Flask Server (`app.py`)

The Flask server serves as the API layer between the browser client and the ML pipeline.

**Routes:**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serve the frontend HTML |
| POST | `/predict` | Single image translation + detection |
| POST | `/predict_video` | Submit video job, returns `job_id` |
| GET | `/video_status/<job_id>` | SSE stream: real-time video progress |
| GET | `/video_result/<job_id>/<which>` | Download `ir` or `input` video |
| DELETE | `/video_cleanup/<job_id>` | Remove temp files |

**Startup sequence:**
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = build_generator(ngf=64, use_dropout=True).to(DEVICE)
ckpt = torch.load("models/final.pth", map_location=DEVICE)
netG.load_state_dict(ckpt.get("netG", ckpt), strict=False)
netG.eval()
app.run(host="0.0.0.0", port=5000, debug=True)
```

The generator is loaded **once at startup** and reused for all requests — no model reloading overhead per request.

---

### 5.2 Image Pipeline

```
Client POST /predict (image file or URL)
        │
        ▼
  Validate file type/size
        │
        ▼
  PIL.Image.open → convert("RGB")
        │
        ▼
  preprocess_to_tensor()        ← utils/transforms.py
  [Scale to 256 → center crop → normalize to [-1,1]]
        │
        ▼
  tensor.unsqueeze(0).to(DEVICE)   ← add batch dim, move to GPU/CPU
        │
        ▼
  netG(tensor)                  ← UNet forward pass
        │
        ▼
  postprocess_to_pil()          ← denormalize, clamp, to PIL
        │
        ▼
  run_detection() (if enabled)  ← services/detection.py
        │
        ▼
  base64 encode both images
        │
        ▼
  JSON response: {input, output, predictions, detection_count}
```

---

### 5.3 Video Pipeline

The video pipeline runs in a **background thread** and emits progress via **Server-Sent Events (SSE)**.

```
Client POST /predict_video (video file)
        │
        ▼
  Validate extension (.mp4/.avi/.mov/.mkv/.webm)
  Save to tmp_videos/{job_id}_input.ext
  Create Job: { q: Queue, status: "running" }
  Start Background Thread (_video_worker)
        │
        ▼ (returns job_id immediately to client)

BACKGROUND THREAD:
        │
   Stage 1: extract_frames()
   │   cv2.VideoCapture → downsample to FPS ≤ 10
   │   Returns: [PIL.Image, ...], effective_fps
        │
   Stage 2: translate_frames()
   │   For each frame:
   │     preprocess → netG forward pass → postprocess
   │   Returns: input_pils, ir_pils
        │
   Stage 3: detect_frames() (optional)
   │   For each IR frame:
   │     run_detection() via Roboflow API
   │     sleep(0.5) between calls (rate limiting)
   │   Returns: annotated_pils
        │
   Stage 4: build_video()
   │   imageio + libx264 (H.264/MP4)
   │   Fallback: OpenCV avc1 or mp4v
   │   Produces: {job_id}_ir.mp4 and {job_id}_input_processed.mp4
        │
        └── Queue.put({"type": "done", "job_id": ...})

SSE STREAM (/video_status/<job_id>):
  Continuously reads from Queue and yields SSE events:
  • {"type": "stage", "stage": "Extracting frames..."}
  • {"type": "total", "total": 120}
  • {"type": "progress", "phase": "translate", "frame": 45, "total": 120}
  • {"type": "done", "job_id": "..."}
```

---

### 5.4 Frontend

The frontend (`templates/index.html`) is a single-page application that:
- Accepts image uploads via drag-and-drop or file picker
- Accepts image URLs
- Displays side-by-side input/output comparison
- Shows detection bounding box overlays
- Handles video uploads with a progress bar (driven by SSE)
- Shows side-by-side video playback after processing

---

## 6. Data & Preprocessing

### LLVIP Dataset

The **Large-scale Visible-Infrared Paired** (LLVIP) dataset contains **15,488 strictly aligned image pairs** of visible and infrared images, captured simultaneously with co-registered cameras. Images depict pedestrian scenes.

**Preprocessing pipeline:**
1. **Scale Width and Crop**: Resize so width ≥ 256, then randomly (or center) crop to 256×256
2. **Normalization**: `pixel = pixel * 2 - 1` → maps `[0, 1]` to `[-1, 1]`
3. **Augmentation** (training only): random horizontal flip

### `config.py` Settings

```python
CHECKPOINT_PATH = BASE_DIR.parent / "models" / "final.pth"
NGF             = 64       # Generator base filters 
CROP_SIZE       = 256      # Input/output resolution

MAX_UPLOAD_MB   = 16       # Image upload size limit
MAX_VIDEO_MB    = 200      # Video upload size limit
VIDEO_FPS_CAP   = 10       # Downsample videos to max 10 FPS
DETECTION_DELAY = 0.5      # Seconds between Roboflow API calls

ROBOFLOW_MODEL_ID = "flir-data-set/22"  # YOLOv8 thermal model
```

---

## 7. Module File Map

```
webapp/
├── app.py                    ← Flask server, routes, SSE, job management
├── config.py                 ← All configurable constants
├── inference.py              ← Standalone batch inference script (CLI)
│
├── model/
│   ├── __init__.py
│   └── generator.py          ← UnetGenerator + UnetSkipConnectionBlock
│
├── models/
│   └── final.pth             ← Trained generator weights (218 MB)
│
├── services/
│   ├── detection.py          ← Roboflow API client + bounding box drawing
│   └── video.py              ← extract_frames, translate_frames,
│                                detect_frames, build_video
│
├── utils/
│   ├── __init__.py
│   └── transforms.py         ← preprocess_to_tensor, postprocess_to_pil
│
├── templates/
│   └── index.html            ← Frontend SPA
│
├── static/                   ← CSS, JS, assets
├── tmp_videos/               ← Temp storage for video pipeline
├── input/                    ← Test input images
├── output/                   ← Inference output images
└── output-yolo/              ← Detection output images
```

---

## 8. End-to-End Data Flow Diagrams

### Image Inference Flow

```
Visible Photo (any size, RGB)
        │
        ▼  scale so min_side ≥ 256 (bicubic)
[H' × W' × 3]
        │
        ▼  center crop
[256 × 256 × 3]  (PIL Image)
        │
        ▼  to_tensor + normalize × 2 - 1
[3 × 256 × 256]  float tensor, range [-1, 1]
        │
        ▼  unsqueeze(0)
[1 × 3 × 256 × 256]  batch tensor
        │
        ▼  UNet-256 Forward Pass (GPU/CPU)
[1 × 3 × 256 × 256]  fake IR tensor, range [-1, 1]
        │
        ▼  squeeze + clamp(-1,1) + (t+1)*0.5
[3 × 256 × 256]  float tensor, range [0, 1]
        │
        ▼  to_pil_image()
[256 × 256 × 3]  PIL Image — Synthetic IR
        │
        ▼  JPEG encode + base64 → Roboflow API
[30×30 predictions grid]  YOLOv8 output
        │
        ▼  parse + annotate bounding boxes
[256 × 256 × 3]  PIL Image — Annotated IR
        │
        ▼  base64 encode → JSON response
Client displays side-by-side
```

### GAN Training Flow (Reference)

```
Batch: (x_real, y_real) = (visible, infrared) paired images
        │
        ▼
  ── DISCRIMINATOR STEP ──
  D_real = D(x_real, y_real)         → target: 1 (real)
  y_fake = G(x_real)
  D_fake = D(x_real, y_fake.detach()) → target: 0 (fake)
  loss_D = MSE(D_real, 1) + MSE(D_fake, 0)
  loss_D.backward() → optimizer_D.step()

  ── GENERATOR STEP ──
  y_fake = G(x_real)
  D_fake = D(x_real, y_fake)         → fool D: target 1
  loss_GAN = MSE(D_fake, 1)
  loss_L1  = L1(y_fake, y_real) × 100
  loss_G   = loss_GAN + loss_L1
  loss_G.backward() → optimizer_G.step()
```

---

*Document generated for project: **V2IR — Visible to Infrared Translation with Pix2Pix + YOLOv8***
*Training dataset: LLVIP | Generator: U-Net 256 (200 epochs) | Detector: YOLOv8 via Roboflow flir-data-set/22*
