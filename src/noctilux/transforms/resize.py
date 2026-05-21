from __future__ import annotations

from PIL import Image

from noctilux.registry import register_transform
from noctilux.transforms.base import BaseTransform

INTERPOLATION_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@register_transform("resize_long_edge")
class ResizeLongEdgeTransform(BaseTransform):
    name = "resize_long_edge"

    def validate_params(self) -> None:
        long_edge = self.params.get("long_edge")
        interpolation = self.params.get("interpolation", "bicubic")
        if not isinstance(long_edge, int) or long_edge < 1:
            raise ValueError("resize_long_edge long_edge must be an integer >= 1.")
        if interpolation not in INTERPOLATION_MAP:
            raise ValueError(
                f"resize_long_edge interpolation must be one of {sorted(INTERPOLATION_MAP)}."
            )

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy()
        width, height = source.size
        target_long_edge = self.params["long_edge"]
        current_long_edge = max(width, height)
        if current_long_edge == target_long_edge:
            return source

        scale = target_long_edge / float(current_long_edge)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resample = INTERPOLATION_MAP[self.params.get("interpolation", "bicubic")]
        return source.resize(new_size, resample=resample)
