from PIL import Image
import torch
import torchvision.transforms.functional as TF

LOAD_SIZE: int = 256
CROP_SIZE: int = 256

def _scale_to_fit(img: Image.Image, size: int) -> Image.Image:
    ow, oh = img.size

    new_w = size
    new_h = int(size * oh / ow)

    if new_h < size:
        new_h = size
        new_w = int(size * ow / oh)

    new_w = max(new_w, size)
    new_h = max(new_h, size)

    return img.resize((new_w, new_h), Image.BICUBIC)

def preprocess_to_pil(img: Image.Image, crop_size: int = CROP_SIZE) -> Image.Image:
    img = _scale_to_fit(img, crop_size)
    img = TF.center_crop(img, (crop_size, crop_size))
    return img

def preprocess_to_tensor(img: Image.Image, crop_size: int = CROP_SIZE) -> torch.Tensor:
    img = preprocess_to_pil(img, crop_size)
    tensor = TF.to_tensor(img)
    tensor = tensor * 2.0 - 1.0
    return tensor

def postprocess_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) * 0.5
    return TF.to_pil_image(tensor)
