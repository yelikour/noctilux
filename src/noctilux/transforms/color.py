from __future__ import annotations

from PIL import Image, ImageEnhance

from noctilux.registry import register_transform
from noctilux.transforms.base import BaseTransform


@register_transform("brightness_contrast")
class BrightnessContrastTransform(BaseTransform):
    name = "brightness_contrast"

    def validate_params(self) -> None:
        brightness = self.params.get("brightness")
        contrast = self.params.get("contrast")
        if not isinstance(brightness, (int, float)) or float(brightness) < 0:
            raise ValueError("brightness_contrast brightness must be a number >= 0.")
        if not isinstance(contrast, (int, float)) or float(contrast) < 0:
            raise ValueError("brightness_contrast contrast must be a number >= 0.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy().convert("RGB")
        brightness = ImageEnhance.Brightness(source).enhance(float(self.params["brightness"]))
        return ImageEnhance.Contrast(brightness).enhance(float(self.params["contrast"]))
