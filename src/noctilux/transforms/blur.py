from __future__ import annotations

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
