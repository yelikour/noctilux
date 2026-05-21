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


@register_transform("webp_compression")
class WebPCompressionTransform(BaseTransform):
    name = "webp_compression"

    def validate_params(self) -> None:
        self.params.setdefault("quality", 80)
        quality = self.params["quality"]
        if not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError("webp_compression quality must be an integer between 1 and 100.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        payload = BytesIO()
        source = image.copy().convert("RGB")
        source.save(payload, format="WEBP", quality=self.params["quality"])
        payload.seek(0)
        with Image.open(payload) as decoded:
            return decoded.convert("RGB")


@register_transform("png_resave")
class PNGResaveTransform(BaseTransform):
    name = "png_resave"

    def validate_params(self) -> None:
        self.params.setdefault("compress_level", 6)
        compress_level = self.params["compress_level"]
        if not isinstance(compress_level, int) or not 0 <= compress_level <= 9:
            raise ValueError("png_resave compress_level must be an integer between 0 and 9.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        payload = BytesIO()
        source = image.copy()
        source.save(payload, format="PNG", compress_level=self.params["compress_level"])
        payload.seek(0)
        with Image.open(payload) as decoded:
            return decoded.convert("RGB")


@register_transform("double_jpeg_compression")
class DoubleJPEGCompressionTransform(BaseTransform):
    name = "double_jpeg_compression"

    def validate_params(self) -> None:
        self.params.setdefault("quality1", 80)
        self.params.setdefault("quality2", 80)
        for key in ("quality1", "quality2"):
            value = self.params[key]
            if not isinstance(value, int) or not 1 <= value <= 100:
                raise ValueError(f"double_jpeg_compression {key} must be an integer between 1 and 100.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        first = _jpeg_roundtrip(image.copy().convert("RGB"), quality=self.params["quality1"])
        return _jpeg_roundtrip(first, quality=self.params["quality2"])


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    payload = BytesIO()
    image.save(payload, format="JPEG", quality=quality, subsampling=SUBSAMPLING_MAP["4:2:0"])
    payload.seek(0)
    with Image.open(payload) as decoded:
        return decoded.convert("RGB")
