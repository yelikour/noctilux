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


def get_resample(interpolation: str, transform_name: str) -> Image.Resampling:
    if interpolation not in INTERPOLATION_MAP:
        raise ValueError(f"{transform_name} interpolation must be one of {sorted(INTERPOLATION_MAP)}.")
    return INTERPOLATION_MAP[interpolation]


@register_transform("resize_long_edge")
class ResizeLongEdgeTransform(BaseTransform):
    name = "resize_long_edge"

    def validate_params(self) -> None:
        long_edge = self.params.get("long_edge")
        interpolation = self.params.get("interpolation", "bicubic")
        if not isinstance(long_edge, int) or long_edge < 1:
            raise ValueError("resize_long_edge long_edge must be an integer >= 1.")
        get_resample(interpolation, "resize_long_edge")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy()
        width, height = source.size
        target_long_edge = self.params["long_edge"]
        current_long_edge = max(width, height)
        if current_long_edge == target_long_edge:
            return source

        scale = target_long_edge / float(current_long_edge)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resample = get_resample(self.params.get("interpolation", "bicubic"), "resize_long_edge")
        return source.resize(new_size, resample=resample)


@register_transform("resize_exact")
class ResizeExactTransform(BaseTransform):
    name = "resize_exact"

    def validate_params(self) -> None:
        width = self.params.get("width")
        height = self.params.get("height")
        interpolation = self.params.get("interpolation", "bicubic")
        if not isinstance(width, int) or width < 1:
            raise ValueError("resize_exact width must be an integer >= 1.")
        if not isinstance(height, int) or height < 1:
            raise ValueError("resize_exact height must be an integer >= 1.")
        get_resample(interpolation, "resize_exact")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy()
        resample = get_resample(self.params.get("interpolation", "bicubic"), "resize_exact")
        return source.resize((self.params["width"], self.params["height"]), resample=resample)


@register_transform("resize_short_edge")
class ResizeShortEdgeTransform(BaseTransform):
    name = "resize_short_edge"

    def validate_params(self) -> None:
        short_edge = self.params.get("short_edge")
        interpolation = self.params.get("interpolation", "bicubic")
        if not isinstance(short_edge, int) or short_edge < 1:
            raise ValueError("resize_short_edge short_edge must be an integer >= 1.")
        get_resample(interpolation, "resize_short_edge")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy()
        width, height = source.size
        current_short_edge = min(width, height)
        if current_short_edge == self.params["short_edge"]:
            return source
        scale = self.params["short_edge"] / float(current_short_edge)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resample = get_resample(self.params.get("interpolation", "bicubic"), "resize_short_edge")
        return source.resize(new_size, resample=resample)


@register_transform("downscale_upscale")
class DownscaleUpscaleTransform(BaseTransform):
    name = "downscale_upscale"

    def validate_params(self) -> None:
        self.params.setdefault("down_interpolation", "bilinear")
        self.params.setdefault("up_interpolation", "bicubic")
        scale = self.params.get("scale")
        if not isinstance(scale, (int, float)) or not 0 < float(scale) < 1:
            raise ValueError("downscale_upscale scale must be a number in the range (0, 1).")
        get_resample(self.params["down_interpolation"], "downscale_upscale")
        get_resample(self.params["up_interpolation"], "downscale_upscale")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = image.copy()
        width, height = source.size
        scale = float(self.params["scale"])
        down_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        down = source.resize(
            down_size,
            resample=get_resample(self.params["down_interpolation"], "downscale_upscale"),
        )
        return down.resize(
            (width, height),
            resample=get_resample(self.params["up_interpolation"], "downscale_upscale"),
        )
