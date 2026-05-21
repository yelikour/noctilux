from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

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


@register_transform("gamma_correction")
class GammaCorrectionTransform(BaseTransform):
    name = "gamma_correction"

    def validate_params(self) -> None:
        gamma = self.params.get("gamma")
        if not isinstance(gamma, (int, float)) or float(gamma) <= 0:
            raise ValueError("gamma_correction gamma must be a number > 0.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy().convert("RGB")
        gamma = float(self.params["gamma"])
        table = [min(255, max(0, int(round(((index / 255.0) ** (1.0 / gamma)) * 255.0)))) for index in range(256)]
        return source.point(table * 3)


@register_transform("saturation_hue")
class SaturationHueTransform(BaseTransform):
    name = "saturation_hue"

    def validate_params(self) -> None:
        self.params.setdefault("saturation", 1.0)
        self.params.setdefault("hue_shift", 0.0)
        saturation = self.params["saturation"]
        hue_shift = self.params["hue_shift"]
        if not isinstance(saturation, (int, float)) or float(saturation) < 0:
            raise ValueError("saturation_hue saturation must be a number >= 0.")
        if not isinstance(hue_shift, (int, float)):
            raise ValueError("saturation_hue hue_shift must be a number.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy().convert("RGB")
        saturated = ImageEnhance.Color(source).enhance(float(self.params["saturation"]))
        hsv = np.asarray(saturated.convert("HSV"), dtype=np.uint8).copy()
        shift = int(round((float(self.params["hue_shift"]) % 360) / 360.0 * 255)) % 256
        hsv[..., 0] = (hsv[..., 0].astype(np.uint16) + shift) % 256
        return Image.fromarray(hsv, mode="HSV").convert("RGB")


@register_transform("grayscale")
class GrayscaleTransform(BaseTransform):
    name = "grayscale"

    def validate_params(self) -> None:
        self.params.setdefault("keep_rgb", True)
        if not isinstance(self.params["keep_rgb"], bool):
            raise ValueError("grayscale keep_rgb must be a boolean.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy()
        gray = ImageOps.grayscale(source)
        if self.params["keep_rgb"]:
            return gray.convert("RGB")
        return gray


@register_transform("sharpen")
class SharpenTransform(BaseTransform):
    name = "sharpen"

    def validate_params(self) -> None:
        factor = self.params.get("factor")
        if not isinstance(factor, (int, float)) or float(factor) < 0:
            raise ValueError("sharpen factor must be a number >= 0.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy().convert("RGB")
        return ImageEnhance.Sharpness(source).enhance(float(self.params["factor"]))


@register_transform("posterize")
class PosterizeTransform(BaseTransform):
    name = "posterize"

    def validate_params(self) -> None:
        bits = self.params.get("bits")
        if not isinstance(bits, int) or not 1 <= bits <= 8:
            raise ValueError("posterize bits must be an integer between 1 and 8.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy().convert("RGB")
        return ImageOps.posterize(source, bits=self.params["bits"])
