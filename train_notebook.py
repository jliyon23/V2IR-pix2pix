# %% [markdown]
# # 🔥 LLVIP Pix2Pix Baseline — Visible → Infrared Translation
#
# A **standalone Kaggle notebook** that faithfully replicates the pix2pixGAN
# training from the [LLVIP repository](https://github.com/bupt-ai-cz/LLVIP/tree/main/pix2pixGAN).
#
# **Architecture (exactly as in the paper/repo):**
# - **Generator:** U-Net 256 (8 downsamples, skip connections, BatchNorm, dropout=0.5 in inner 3 layers)
# - **Discriminator:** 70×70 PatchGAN (3-layer, BatchNorm)
# - **Loss:** Vanilla GAN (BCEWithLogits) + L1 (λ=100)
# - **Optimizer:** Adam (lr=2e-4, β₁=0.5, β₂=0.999)
# - **Schedule:** 100 epochs constant + 100 epochs linear decay to 0
# - **Direction:** AtoB (Visible → Infrared)
# - **Preprocess:** scale_width_and_crop, load_size=320, crop_size=256

# %% [markdown]
# ---
# ## Cell 1: Setup & Configuration

# %%
# ============================================================================
# CELL 1: SETUP & CONFIGURATION
# ============================================================================

import os
import glob
import random
import time
import math
import functools
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🎮  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

# ── Hyperparameters (matching LLVIP repo defaults) ───────────────────────────
BATCH_SIZE     = 8
LOAD_SIZE      = 320      # resize shorter side to this
CROP_SIZE      = 256      # random crop to this
N_EPOCHS       = 100      # epochs with constant LR
N_EPOCHS_DECAY = 100      # epochs with linearly decaying LR
LR             = 2e-4
BETA1          = 0.5
LAMBDA_L1      = 100.0
NGF            = 64       # generator filters
NDF            = 64       # discriminator filters
NUM_WORKERS    = 2
DIRECTION      = "AtoB"   # Visible(A) → Infrared(B)

# Checkpoint settings
SAVE_EPOCH_FREQ = 5       # save checkpoint every N epochs
VIS_EPOCH_FREQ  = 5       # save visualization every N epochs
MAX_CKPTS       = 3       # keep top-K recent periodic checkpoints

TOTAL_EPOCHS = N_EPOCHS + N_EPOCHS_DECAY

# ── Continuation settings ────────────────────────────────────────────────────
# Use these to resume from a final.pth that has NO scheduler/optimizer state.
# Set CONTINUE_FROM = "" to disable (normal fresh-start or latest.pth resume).
CONTINUE_FROM     = "models/final.pth"  # path to netG+netD checkpoint
CONTINUE_AT_EPOCH = 100                 # epoch the saved model reached
CONTINUE_N_EPOCHS = 50                  # constant-LR epochs for this continuation
CONTINUE_N_DECAY  = 50                  # linear-decay epochs for this continuation

# ── Dynamic Path Hunting ─────────────────────────────────────────────────────
def find_llvip_dataset():
    """Search for LLVIP dataset across Kaggle and local paths."""
    search_patterns = [
        "/kaggle/input/datasets/afradhossain/llvip-dataset/LLVIP",
        "/kaggle/input/datasets/afradhossain/llvip-dataset",
        "/kaggle/input/LLVIP*/LLVIP",
        "/kaggle/input/LLVIP*",
        "/kaggle/input/*llvip*/LLVIP",
        "/kaggle/input/*llvip*",
        "./LLVIP",
        "../LLVIP",
        os.path.expanduser("~/Desktop/llvip/LLVIP"),
        "/content/drive/MyDrive/LLVIP",
    ]
    for pattern in search_patterns:
        for m in glob.glob(pattern):
            if os.path.isdir(m):
                # Check for split format (infrared/ + visible/)
                if os.path.isdir(os.path.join(m, "infrared")) and os.path.isdir(os.path.join(m, "visible")):
                    print(f"📂 Found LLVIP (split format): {m}")
                    return m, "split"
                # Check for combined format (train/ with side-by-side images)
                if os.path.isdir(os.path.join(m, "train")):
                    print(f"📂 Found LLVIP (combined format): {m}")
                    return m, "combined"
    raise FileNotFoundError("❌ LLVIP dataset not found! Add it as a Kaggle input or place in ./LLVIP")


DATA_ROOT, DATA_LAYOUT = find_llvip_dataset()

# ── Output dirs ──────────────────────────────────────────────────────────────
if os.path.exists("/kaggle/working"):
    CKPT_DIR   = "/kaggle/working/checkpoints"
    OUTPUT_DIR = "/kaggle/working/samples"
else:
    CKPT_DIR   = os.path.join(os.path.dirname(DATA_ROOT), "checkpoints")
    OUTPUT_DIR = os.path.join(os.path.dirname(DATA_ROOT), "samples")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"💾 Checkpoints: {CKPT_DIR}")
print(f"🖼️  Samples:     {OUTPUT_DIR}")


# %% [markdown]
# ---
# ## Cell 2: Data Pipeline (AlignedDataset)

# %%
# ============================================================================
# CELL 2: DATA PIPELINE
# ============================================================================
# Replicates the LLVIP repo's AlignedDataset with scale_width_and_crop
# preprocessing and synchronized transforms for paired images.
# ============================================================================

class LLVIPAlignedDataset(Dataset):
    """
    Paired dataset for pix2pix training on LLVIP.

    Supports two layouts:
      - 'split':    infrared/{split}/ + visible/{split}/ (separate directories)
      - 'combined': {split}/ with side-by-side A|B concatenated images

    Preprocessing matches the repo:
      --preprocess scale_width_and_crop --load_size 320 --crop_size 256
    """

    def __init__(self, root, split="train", layout="split",
                 load_size=320, crop_size=256, direction="AtoB", is_train=True):
        super().__init__()
        self.load_size = load_size
        self.crop_size = crop_size
        self.direction = direction
        self.is_train = is_train
        self.layout = layout

        if layout == "split":
            ir_dir  = os.path.join(root, "infrared", split)
            vis_dir = os.path.join(root, "visible", split)
            self.ir_paths  = sorted(glob.glob(os.path.join(ir_dir, "*.*")))
            self.vis_paths = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
            assert len(self.ir_paths) == len(self.vis_paths), \
                f"Mismatch: {len(self.ir_paths)} IR vs {len(self.vis_paths)} VIS images"
            self.size = len(self.ir_paths)
        else:
            combined_dir = os.path.join(root, split)
            self.combined_paths = sorted(glob.glob(os.path.join(combined_dir, "*.*")))
            self.size = len(self.combined_paths)

        print(f"  📊 {split}: {self.size} image pairs (layout={layout}, direction={direction})")

    def __len__(self):
        return self.size

    def _load_pair(self, idx):
        """Load A (visible) and B (infrared) images."""
        if self.layout == "split":
            A = Image.open(self.vis_paths[idx]).convert("RGB")  # Visible
            B = Image.open(self.ir_paths[idx]).convert("RGB")   # Infrared
        else:
            AB = Image.open(self.combined_paths[idx]).convert("RGB")
            w, h = AB.size
            w2 = w // 2
            A = AB.crop((0, 0, w2, h))       # Left = A (Visible)
            B = AB.crop((w2, 0, w, h))        # Right = B (Infrared)
        return A, B

    def _transform(self, A, B):
        """
        Apply synchronized transforms matching the LLVIP repo:
        scale_width_and_crop → random_crop → flip → to_tensor → normalize
        """
        # ── Scale width and adjust height proportionally ─────────────────
        ow, oh = A.size
        new_w = self.load_size
        new_h = int(self.load_size * oh / ow)

        A = A.resize((new_w, new_h), Image.BICUBIC)
        B = B.resize((new_w, new_h), Image.BICUBIC)

        if self.is_train:
            # ── Random crop (synchronized) ───────────────────────────────
            i, j, h, w = T.RandomCrop.get_params(A, (self.crop_size, self.crop_size))
            A = TF.crop(A, i, j, h, w)
            B = TF.crop(B, i, j, h, w)

            # ── Random horizontal flip ───────────────────────────────────
            if random.random() > 0.5:
                A = TF.hflip(A)
                B = TF.hflip(B)
        else:
            # ── Center crop for test ─────────────────────────────────────
            A = TF.center_crop(A, [self.crop_size, self.crop_size])
            B = TF.center_crop(B, [self.crop_size, self.crop_size])

        # ── To tensor + normalize to [-1, 1] ────────────────────────────
        A = TF.to_tensor(A) * 2.0 - 1.0
        B = TF.to_tensor(B) * 2.0 - 1.0

        return A, B

    def __getitem__(self, idx):
        A, B = self._load_pair(idx)
        A, B = self._transform(A, B)

        # Apply direction
        if self.direction == "AtoB":
            return A, B   # input=Visible, target=Infrared
        else:
            return B, A   # input=Infrared, target=Visible


# ── Build DataLoaders ────────────────────────────────────────────────────────
print("\n── Building DataLoaders ──")
train_dataset = LLVIPAlignedDataset(
    DATA_ROOT, "train", DATA_LAYOUT,
    load_size=LOAD_SIZE, crop_size=CROP_SIZE,
    direction=DIRECTION, is_train=True
)
test_dataset = LLVIPAlignedDataset(
    DATA_ROOT, "test", DATA_LAYOUT,
    load_size=CROP_SIZE, crop_size=CROP_SIZE,
    direction=DIRECTION, is_train=False
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=4, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

print(f"  Train: {len(train_dataset)} images → {len(train_loader)} batches")
print(f"  Test:  {len(test_dataset)} images → {len(test_loader)} batches")

# Quick shape check
sample_A, sample_B = next(iter(train_loader))
print(f"  Batch shapes: A={sample_A.shape}, B={sample_B.shape}")
del sample_A, sample_B


# %% [markdown]
# ---
# ## Cell 3: Model Architecture (U-Net Generator + PatchGAN Discriminator)

# %%
# ============================================================================
# CELL 3: MODELS — Exact replicas from pytorch-CycleGAN-and-pix2pix
# ============================================================================

# ═══════════════════════════════════════════════════════════════════════════════
# 3A. U-Net Generator (recursive UnetSkipConnectionBlock — same as the repo)
# ═══════════════════════════════════════════════════════════════════════════════

class UnetSkipConnectionBlock(nn.Module):
    """
    Defines the Unet submodule with skip connection.
        X ------- identity -------
        |-- down -- |submodule| -- up --|

    Exactly as in the LLVIP/pix2pixGAN repo (from pytorch-CycleGAN-and-pix2pix).
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super().__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)  # skip connection


class UnetGenerator(nn.Module):
    """
    U-Net 256 generator built recursively from UnetSkipConnectionBlocks.

    num_downs=8 for 256×256 input:
      outermost(3→64) → 64→128 → 128→256 → 256→512 → 512→512 → 512→512(dropout)
      → 512→512(dropout) → innermost(512→512) → back up with skip connections
    """

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        # Build from innermost to outermost
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None,
            norm_layer=norm_layer, innermost=True
        )

        # Add intermediate layers with ngf * 8 filters (with dropout)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                norm_layer=norm_layer, use_dropout=use_dropout
            )

        # Gradually reduce filters: 512→256→128→64
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )

        # Outermost layer
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block,
            outermost=True, norm_layer=norm_layer
        )

    def forward(self, x):
        return self.model(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 3B. PatchGAN Discriminator (70×70) — NLayerDiscriminator
# ═══════════════════════════════════════════════════════════════════════════════

class NLayerDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator (3 layers).
    Input: concatenation of (input, target/generated) → 6 channels.
    Output: patch map of real/fake predictions.

    Exactly as in the LLVIP/pix2pixGAN repo.
    """

    def __init__(self, input_nc=6, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 3C. GAN Loss (vanilla — same as repo)
# ═══════════════════════════════════════════════════════════════════════════════

class GANLoss(nn.Module):
    """Vanilla GAN loss using BCEWithLogitsLoss (same as repo)."""

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


# ═══════════════════════════════════════════════════════════════════════════════
# 3D. Weight Initialization (Gaussian, mean=0, std=0.02 — same as repo)
# ═══════════════════════════════════════════════════════════════════════════════

def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights same as the original pix2pix repo."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


# ── Instantiate & print summary ──────────────────────────────────────────────
netG = init_weights(UnetGenerator(3, 3, num_downs=8, ngf=NGF, use_dropout=True)).to(DEVICE)
netD = init_weights(NLayerDiscriminator(input_nc=6, ndf=NDF, n_layers=3)).to(DEVICE)

g_params = sum(p.numel() for p in netG.parameters()) / 1e6
d_params = sum(p.numel() for p in netD.parameters()) / 1e6
print(f"\n🏗️  Generator (UNet256):     {g_params:.2f}M params")
print(f"🏗️  Discriminator (PatchGAN): {d_params:.2f}M params")


# %% [markdown]
# ---
# ## Cell 4: Training Loop

# %%
# ============================================================================
# CELL 4: TRAINING — Pix2Pix GAN (Visible → Infrared)
# ============================================================================
# Replicates the exact training procedure from the LLVIP pix2pixGAN repo:
#   - GAN loss (vanilla) + L1 loss (λ=100)
#   - 100 epochs constant LR + 100 epochs linear decay
#   - Adam optimizer (lr=2e-4, β₁=0.5, β₂=0.999)
# ============================================================================

# ── Loss functions ───────────────────────────────────────────────────────────
criterionGAN = GANLoss().to(DEVICE)
criterionL1  = nn.L1Loss()

# ── Optimizers ───────────────────────────────────────────────────────────────
optimizer_G = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))


# ── LR Scheduler (linear decay after N_EPOCHS) ──────────────────────────────
def get_scheduler(optimizer):
    """Same linear decay schedule as the repo: constant for N_EPOCHS, then linearly decay."""
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - N_EPOCHS) / float(N_EPOCHS_DECAY + 1)
        return lr_l
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


scheduler_G = get_scheduler(optimizer_G)
scheduler_D = get_scheduler(optimizer_D)


# ── Checkpoint utilities ─────────────────────────────────────────────────────
def save_ckpt(state, filepath, max_keep=3):
    """Save checkpoint, auto-clean old periodic saves."""
    torch.save(state, filepath)
    basename = os.path.basename(filepath)
    print(f"  💾 Saved: {basename}")

    # Clean old periodic checkpoints
    ckpt_dir = os.path.dirname(filepath)
    periodic = sorted(
        [f for f in glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
         if "latest" not in f],
        key=os.path.getmtime
    )
    while len(periodic) > max_keep:
        old = periodic.pop(0)
        os.remove(old)
        print(f"  🗑️  Removed: {os.path.basename(old)}")


MAX_SAMPLES = 5  # keep only the latest 5 sample images

def save_samples(epoch, netG, loader, output_dir, n=4):
    """Save a visualization grid: Input | Generated | Ground Truth. Keep only latest MAX_SAMPLES."""
    netG.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        real_A = batch[0][:n].to(DEVICE)
        real_B = batch[1][:n].to(DEVICE)
        fake_B = netG(real_A)

        # Concatenate: row1=input, row2=generated, row3=ground truth
        vis = torch.cat([real_A, fake_B, real_B], dim=0)
        grid = torchvision.utils.make_grid(vis, nrow=n, normalize=True, value_range=(-1, 1))
        img = TF.to_pil_image(grid.cpu())
        path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
        img.save(path)
        print(f"  🖼️  Saved: epoch_{epoch:03d}.png (top: vis | mid: gen | bot: IR)")

    # Auto-clean: keep only latest MAX_SAMPLES images
    existing = sorted(glob.glob(os.path.join(output_dir, "epoch_*.png")), key=os.path.getmtime)
    while len(existing) > MAX_SAMPLES:
        old = existing.pop(0)
        os.remove(old)
        print(f"  🗑️  Removed old sample: {os.path.basename(old)}")

    netG.train()


# ── Auto-resume ──────────────────────────────────────────────────────────────
latest_path = os.path.join(CKPT_DIR, "latest.pth")
start_epoch = 0

if os.path.exists(latest_path):
    # ── Full resume (weights + optimizer + scheduler state) ──────────────
    print(f"🔄 Resuming from latest checkpoint...")
    ckpt = torch.load(latest_path, map_location=DEVICE, weights_only=False)
    netG.load_state_dict(ckpt["netG"])
    netD.load_state_dict(ckpt["netD"])
    optimizer_G.load_state_dict(ckpt["optimizer_G"])
    optimizer_D.load_state_dict(ckpt["optimizer_D"])
    scheduler_G.load_state_dict(ckpt["scheduler_G"])
    scheduler_D.load_state_dict(ckpt["scheduler_D"])
    start_epoch = ckpt["epoch"]
    print(f"  Resumed at epoch {start_epoch}")
    del ckpt

elif CONTINUE_FROM and os.path.exists(CONTINUE_FROM):
    # ── Continuation from a weights-only checkpoint (e.g. final.pth) ─────
    # Loads netG + netD, then resets optimizers and builds a fresh LR
    # schedule sized for exactly CONTINUE_N_EPOCHS + CONTINUE_N_DECAY steps.
    print(f"📦 Loading pretrained weights for continuation: {CONTINUE_FROM}")
    ckpt = torch.load(CONTINUE_FROM, map_location=DEVICE, weights_only=False)
    netG.load_state_dict(ckpt["netG"])
    netD.load_state_dict(ckpt["netD"])
    del ckpt

    start_epoch = CONTINUE_AT_EPOCH

    # Override schedule lengths — the lambda closure reads N_EPOCHS /
    # N_EPOCHS_DECAY by name at call-time, so updating them here is enough.
    N_EPOCHS       = CONTINUE_N_EPOCHS
    N_EPOCHS_DECAY = CONTINUE_N_DECAY
    TOTAL_EPOCHS   = CONTINUE_AT_EPOCH + CONTINUE_N_EPOCHS + CONTINUE_N_DECAY

    # Fresh schedulers: internal counter starts at 0 and runs for
    # CONTINUE_N_EPOCHS (constant LR) + CONTINUE_N_DECAY (linear decay).
    scheduler_G = get_scheduler(optimizer_G)
    scheduler_D = get_scheduler(optimizer_D)

    print(f"  Weights loaded from epoch {CONTINUE_AT_EPOCH}")
    print(f"  New schedule: {CONTINUE_N_EPOCHS} constant + {CONTINUE_N_DECAY} decay")
    print(f"  Running epochs {start_epoch + 1} → {TOTAL_EPOCHS}")

# ── Print training config ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PIX2PIX TRAINING — VISIBLE → INFRARED")
print("=" * 70)
print(f"  📋 Epochs: {start_epoch} → {TOTAL_EPOCHS} (const={N_EPOCHS}, decay={N_EPOCHS_DECAY})")
print(f"  📋 Batch size: {BATCH_SIZE}, Batches/epoch: {len(train_loader)}")
print(f"  📋 Loss: GAN (vanilla) + L1 (λ={LAMBDA_L1})")
print(f"  📋 LR: {LR}, β₁={BETA1}")
print()

# ── Training loop ────────────────────────────────────────────────────────────
for epoch in range(start_epoch, TOTAL_EPOCHS):
    netG.train()
    netD.train()

    g_loss_sum = 0.0
    d_loss_sum = 0.0
    num_batches = len(train_loader)
    t0 = time.time()

    for i, (real_A, real_B) in enumerate(train_loader):
        batch_t0 = time.time()
        real_A = real_A.to(DEVICE, non_blocking=True)
        real_B = real_B.to(DEVICE, non_blocking=True)

        # ── Forward pass ─────────────────────────────────────────────────
        fake_B = netG(real_A)

        # ══════════════════════════════════════════════════════════════════
        # (1) Update D
        # ══════════════════════════════════════════════════════════════════
        for p in netD.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()

        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = netD(real_AB)
        loss_D_real = criterionGAN(pred_real, True)

        # Fake
        fake_AB = torch.cat((real_A, fake_B.detach()), 1)
        pred_fake = netD(fake_AB)
        loss_D_fake = criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # ══════════════════════════════════════════════════════════════════
        # (2) Update G
        # ══════════════════════════════════════════════════════════════════
        for p in netD.parameters():
            p.requires_grad = False

        optimizer_G.zero_grad()

        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = netD(fake_AB)
        loss_G_GAN = criterionGAN(pred_fake, True)
        loss_G_L1  = criterionL1(fake_B, real_B) * LAMBDA_L1

        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optimizer_G.step()

        g_loss_sum += loss_G.item()
        d_loss_sum += loss_D.item()

        # ── Per-batch logging (every 100th batch + last batch) ─────────
        if (i + 1) % 100 == 0 or (i + 1) == num_batches:
            batch_time = time.time() - batch_t0
            print(f"    [Epoch {epoch+1:03d}] Batch {i+1:04d}/{num_batches} | "
                  f"G: {loss_G.item():.4f} (GAN:{loss_G_GAN.item():.4f} L1:{loss_G_L1.item():.4f}) | "
                  f"D: {loss_D.item():.4f} (R:{loss_D_real.item():.4f} F:{loss_D_fake.item():.4f}) | "
                  f"{batch_time:.2f}s")

    # ── Update LR ────────────────────────────────────────────────────────
    scheduler_G.step()
    scheduler_D.step()

    avg_g = g_loss_sum / num_batches
    avg_d = d_loss_sum / num_batches
    elapsed = time.time() - t0
    lr_now = optimizer_G.param_groups[0]["lr"]

    print(f"  ✅ Epoch {epoch+1:03d}/{TOTAL_EPOCHS} DONE | "
          f"Avg G: {avg_g:.4f} | Avg D: {avg_d:.4f} | "
          f"LR: {lr_now:.6f} | Total: {elapsed:.1f}s")

    # ── Save latest (always, for auto-resume) ────────────────────────────
    state = {
        "epoch": epoch + 1,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "scheduler_G": scheduler_G.state_dict(),
        "scheduler_D": scheduler_D.state_dict(),
    }
    torch.save(state, latest_path)

    # ── Periodic checkpoint ──────────────────────────────────────────────
    if (epoch + 1) % SAVE_EPOCH_FREQ == 0:
        save_ckpt(state, os.path.join(CKPT_DIR, f"epoch_{epoch+1:03d}.pth"), max_keep=MAX_CKPTS)

    # ── Visualization ────────────────────────────────────────────────────
    if (epoch + 1) % VIS_EPOCH_FREQ == 0 or (epoch + 1) == TOTAL_EPOCHS:
        save_samples(epoch + 1, netG, test_loader, OUTPUT_DIR)

# ── Save final model ─────────────────────────────────────────────────────────
torch.save({"netG": netG.state_dict(), "netD": netD.state_dict()},
           os.path.join(CKPT_DIR, "final.pth"))
torch.save({"netG": netG.state_dict()},
           os.path.join(CKPT_DIR, "generator_only.pth"))

# ── Storage report ───────────────────────────────────────────────────────────
total_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(CKPT_DIR, "*.pth")))
print(f"\n✅ Training complete!")
print(f"📊 Checkpoint storage: {total_bytes / 1e9:.2f} GB / 19.5 GB limit")
print(f"📁 Models: {CKPT_DIR}")
print(f"🖼️  Samples: {OUTPUT_DIR}")
print("🎉 Done!")
