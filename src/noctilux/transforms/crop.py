from __future__ import annotations

from PIL import Image

from noctilux.registry import register_transform
from noctilux.transforms.base import BaseTransform


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
