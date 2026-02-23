import argparse
import functools
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super().__init__()
        self.outermost = outermost

        if isinstance(norm_layer, functools.partial):
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
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None,
            norm_layer=norm_layer, innermost=True
        )

        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                norm_layer=norm_layer, use_dropout=use_dropout
            )

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block,
            outermost=True, norm_layer=norm_layer
        )

    def forward(self, x):
        return self.model(x)


class InferenceDataset(Dataset):
    def __init__(self, input_dir, load_size=256, crop_size=256):
        self.input_dir = Path(input_dir)
        self.load_size = load_size
        self.crop_size = crop_size
        self.paths = sorted([
            p for p in self.input_dir.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ])
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.input_dir}")

    def __len__(self):
        return len(self.paths)

    def _resize_with_min_side(self, img):
        ow, oh = img.size
        scale = self.load_size / float(ow)
        new_w = int(round(ow * scale))
        new_h = int(round(oh * scale))
        if new_h < self.crop_size:
            scale = self.crop_size / float(oh)
            new_w = int(round(ow * scale))
            new_h = int(round(oh * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        return img

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self._resize_with_min_side(img)
        img = TF.center_crop(img, (self.crop_size, self.crop_size))
        tensor = TF.to_tensor(img) * 2.0 - 1.0
        return tensor, str(path)


def load_generator(checkpoint_path, device, ngf=64, use_dropout=True):
    netG = UnetGenerator(3, 3, num_downs=8, ngf=ngf, use_dropout=use_dropout).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "netG" in checkpoint:
        state_dict = checkpoint["netG"]
    else:
        state_dict = checkpoint
    missing, unexpected = netG.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading generator: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading generator: {unexpected}")
    netG.eval()
    return netG


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if not Path(args.checkpoint).exists():
        fallback = Path("models/final.pth")
        if fallback.exists():
            args.checkpoint = str(fallback)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    dataset = InferenceDataset(args.input_dir, load_size=args.load_size, crop_size=args.crop_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    netG = load_generator(args.checkpoint, device, ngf=args.ngf, use_dropout=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch_idx, (images, paths) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            fake = netG(images)
            fake = (fake.clamp(-1, 1) + 1) * 0.5
            for img_tensor, src_path in zip(fake, paths):
                out_name = Path(src_path).stem + ".png"
                out_path = output_dir / out_name
                img = TF.to_pil_image(img_tensor.cpu())
                img.save(out_path)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"Processed {batch_idx + 1}/{len(loader)} batches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLVIP pix2pix inference")
    parser.add_argument("--input_dir", default="input", help="Folder with visible images")
    parser.add_argument("--output_dir", default="output", help="Folder to save IR predictions")
    parser.add_argument("--checkpoint", default="models/generator_only.pth", help="Path to generator checkpoint")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--load_size", type=int, default=256, help="Resize shorter side to this")
    parser.add_argument("--crop_size", type=int, default=256, help="Center crop size")
    parser.add_argument("--ngf", type=int, default=64, help="Generator base filters (must match training)")
    args = parser.parse_args()

    run_inference(args)
