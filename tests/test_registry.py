from __future__ import annotations

from noctilux.registry import build_transform, list_transforms


def test_list_transforms_contains_builtins() -> None:
    names = list_transforms()
    assert names == [
        "brightness_contrast",
        "center_crop_ratio",
        "double_jpeg_compression",
        "downscale_upscale",
        "gamma_correction",
        "gaussian_blur",
        "gaussian_noise",
        "grayscale",
        "horizontal_flip",
        "jpeg_compression",
        "median_blur",
        "motion_blur",
        "png_resave",
        "poisson_noise",
        "posterize",
        "random_crop_ratio",
        "random_resized_crop",
        "resize_exact",
        "resize_long_edge",
        "resize_short_edge",
        "rotate",
        "salt_pepper_noise",
        "saturation_hue",
        "sharpen",
        "square_crop",
        "vertical_flip",
        "webp_compression",
    ]


def test_build_transform_creates_instance() -> None:
    transform = build_transform("resize_long_edge", params={"long_edge": 32, "interpolation": "nearest"})
    assert transform.name == "resize_long_edge"
