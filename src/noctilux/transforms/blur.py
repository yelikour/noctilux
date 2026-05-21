from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from noctilux.registry import register_transform
from noctilux.transforms.base import BaseTransform


@register_transform("gaussian_blur")
class GaussianBlurTransform(BaseTransform):
    name = "gaussian_blur"

    def validate_params(self) -> None:
        radius = self.params.get("radius")
        if not isinstance(radius, (int, float)) or float(radius) < 0:
            raise ValueError("gaussian_blur radius must be a number >= 0.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        return image.copy().filter(ImageFilter.GaussianBlur(radius=float(self.params["radius"])))


@register_transform("median_blur")
class MedianBlurTransform(BaseTransform):
    name = "median_blur"

    def validate_params(self) -> None:
        size = self.params.get("size")
        if not isinstance(size, int) or size < 1 or size % 2 == 0:
            raise ValueError("median_blur size must be a positive odd integer.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        return image.copy().filter(ImageFilter.MedianFilter(size=self.params["size"]))


@register_transform("motion_blur")
class MotionBlurTransform(BaseTransform):
    name = "motion_blur"

    def validate_params(self) -> None:
        kernel_size = self.params.get("kernel_size")
        angle = self.params.get("angle")
        if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("motion_blur kernel_size must be a positive odd integer.")
        if not isinstance(angle, (int, float)):
            raise ValueError("motion_blur angle must be a number.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = np.asarray(image.copy().convert("RGB"), dtype=np.float32)
        kernel = _build_motion_kernel(self.params["kernel_size"], float(self.params["angle"]))
        output = _convolve_rgb(source, kernel)
        return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def _build_motion_kernel(kernel_size: int, angle: float) -> np.ndarray:
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    radians = math.radians(angle)
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    for step in range(kernel_size):
        offset = step - center
        x = int(round(center + offset * cos_v))
        y = int(round(center + offset * sin_v))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    kernel /= kernel.sum()
    return kernel


def _convolve_rgb(source: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    padded = np.pad(source, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    output = np.empty_like(source)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            region = padded[y : y + kernel_size, x : x + kernel_size, :]
            output[y, x, :] = np.tensordot(region, kernel, axes=((0, 1), (0, 1)))
    return output
