from __future__ import annotations

import math
import random

from PIL import Image

from noctilux.registry import register_transform
from noctilux.transforms.base import BaseTransform
from noctilux.transforms.resize import get_resample


@register_transform("center_crop_ratio")
class CenterCropRatioTransform(BaseTransform):
    name = "center_crop_ratio"

    def validate_params(self) -> None:
        ratio = self.params.get("ratio")
        if not isinstance(ratio, (int, float)) or not 0 < float(ratio) <= 1:
            raise ValueError("center_crop_ratio ratio must be a number in the range (0, 1].")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy()
        ratio = float(self.params["ratio"])
        width, height = source.size
        new_width = max(1, round(width * ratio))
        new_height = max(1, round(height * ratio))
        left = max(0, (width - new_width) // 2)
        top = max(0, (height - new_height) // 2)
        return source.crop((left, top, left + new_width, top + new_height))


@register_transform("random_crop_ratio")
class RandomCropRatioTransform(BaseTransform):
    name = "random_crop_ratio"

    def validate_params(self) -> None:
        ratio = self.params.get("ratio")
        if not isinstance(ratio, (int, float)) or not 0 < float(ratio) <= 1:
            raise ValueError("random_crop_ratio ratio must be a number in the range (0, 1].")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy()
        ratio = float(self.params["ratio"])
        width, height = source.size
        crop_width = max(1, round(width * ratio))
        crop_height = max(1, round(height * ratio))
        rng = _get_rng(context)
        left = 0 if crop_width >= width else rng.randint(0, width - crop_width)
        top = 0 if crop_height >= height else rng.randint(0, height - crop_height)
        return source.crop((left, top, left + crop_width, top + crop_height))


@register_transform("random_resized_crop")
class RandomResizedCropTransform(BaseTransform):
    name = "random_resized_crop"

    def validate_params(self) -> None:
        self.params.setdefault("interpolation", "bicubic")
        scale_min = self.params.get("scale_min")
        scale_max = self.params.get("scale_max")
        ratio_min = self.params.get("ratio_min")
        ratio_max = self.params.get("ratio_max")
        output_width = self.params.get("output_width")
        output_height = self.params.get("output_height")
        if not isinstance(scale_min, (int, float)) or not isinstance(scale_max, (int, float)):
            raise ValueError("random_resized_crop scale_min and scale_max must be numbers.")
        if not 0 < float(scale_min) <= float(scale_max) <= 1:
            raise ValueError("random_resized_crop scale_min/scale_max must satisfy 0 < min <= max <= 1.")
        if not isinstance(ratio_min, (int, float)) or not isinstance(ratio_max, (int, float)):
            raise ValueError("random_resized_crop ratio_min and ratio_max must be numbers.")
        if not 0 < float(ratio_min) <= float(ratio_max):
            raise ValueError("random_resized_crop ratio_min and ratio_max must satisfy 0 < min <= max.")
        if not isinstance(output_width, int) or output_width < 1:
            raise ValueError("random_resized_crop output_width must be an integer >= 1.")
        if not isinstance(output_height, int) or output_height < 1:
            raise ValueError("random_resized_crop output_height must be an integer >= 1.")
        get_resample(self.params["interpolation"], "random_resized_crop")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy()
        width, height = source.size
        area = width * height
        rng = _get_rng(context)

        crop_box: tuple[int, int, int, int] | None = None
        for _ in range(10):
            target_area = area * rng.uniform(float(self.params["scale_min"]), float(self.params["scale_max"]))
            aspect_ratio = rng.uniform(float(self.params["ratio_min"]), float(self.params["ratio_max"]))
            crop_width = max(1, round(math.sqrt(target_area * aspect_ratio)))
            crop_height = max(1, round(math.sqrt(target_area / aspect_ratio)))
            if crop_width <= width and crop_height <= height:
                left = 0 if crop_width == width else rng.randint(0, width - crop_width)
                top = 0 if crop_height == height else rng.randint(0, height - crop_height)
                crop_box = (left, top, left + crop_width, top + crop_height)
                break

        if crop_box is None:
            side = min(width, height)
            left = max(0, (width - side) // 2)
            top = max(0, (height - side) // 2)
            crop_box = (left, top, left + side, top + side)

        cropped = source.crop(crop_box)
        resample = get_resample(self.params["interpolation"], "random_resized_crop")
        return cropped.resize((self.params["output_width"], self.params["output_height"]), resample=resample)


@register_transform("square_crop")
class SquareCropTransform(BaseTransform):
    name = "square_crop"

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy()
        width, height = source.size
        side = min(width, height)
        left = max(0, (width - side) // 2)
        top = max(0, (height - side) // 2)
        return source.crop((left, top, left + side, top + side))


def _get_rng(context: dict | None) -> random.Random:
    return (context or {}).get("rng") or random.Random()
