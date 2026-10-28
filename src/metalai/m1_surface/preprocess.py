"""THE single preprocessing path for the CNN - used by training, evaluation, Grad-CAM and the app.
(decisions.md D-007: never duplicate preprocessing code.)"""
from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Accept gray / RGB / RGBA arrays (any dtype) and return HxW uint8."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3:
        img = img[..., 0]
    if img.dtype != np.uint8:
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return img


def build_transform(train: bool, img_size: int = 224):
    from torchvision import transforms as T

    aug = []
    if train:
        aug = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10, fill=0),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    return T.Compose([
        T.ToPILImage(),
        T.Grayscale(num_output_channels=3),
        *aug,
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def display_rgb(gray: np.ndarray, img_size: int = 224) -> np.ndarray:
    """Float RGB [0,1] image at model resolution, for Grad-CAM overlays."""
    g = cv2.resize(to_gray_uint8(gray), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return np.repeat(g[..., None], 3, axis=2).astype(np.float32) / 255.0
