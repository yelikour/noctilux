from __future__ import annotations

from io import BytesIO

from PIL import Image

from noctilux.registry import register_transform
from noctilux.transforms.base import BaseTransform

SUBSAMPLING_MAP = {
    "4:4:4": 0,
    "4:2:2": 1,
    "4:2:0": 2,
    "keep": "keep",
}


@register_transform("jpeg_compression")
class JPEGCompressionTransform(BaseTransform):
    name = "jpeg_compression"

    def validate_params(self) -> None:
        quality = self.params.get("quality")
        if not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError("jpeg_compression quality must be an integer between 1 and 100.")
        subsampling = self.params.get("subsampling", "4:2:0")
        if subsampling not in SUBSAMPLING_MAP and subsampling not in {0, 1, 2}:
            raise ValueError("jpeg_compression subsampling must be 0, 1, 2, or one of 4:4:4, 4:2:2, 4:2:0, keep.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        payload = BytesIO()
        subsampling = self.params.get("subsampling", "4:2:0")
        subsampling_value = SUBSAMPLING_MAP.get(subsampling, subsampling)
        source = image.copy().convert("RGB")
        source.save(
            payload,
            format="JPEG",
            quality=self.params["quality"],
            subsampling=subsampling_value,
        )
        payload.seek(0)
        with Image.open(payload) as decoded:
            return decoded.convert("RGB")
