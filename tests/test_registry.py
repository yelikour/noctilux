from __future__ import annotations

from noctilux.registry import build_transform, list_transforms


def test_list_transforms_contains_builtins() -> None:
    names = list_transforms()
    assert names == [
        "brightness_contrast",
        "center_crop_ratio",
        "gaussian_blur",
        "gaussian_noise",
        "jpeg_compression",
        "resize_long_edge",
    ]


def test_build_transform_creates_instance() -> None:
    transform = build_transform("resize_long_edge", params={"long_edge": 32, "interpolation": "nearest"})
    assert transform.name == "resize_long_edge"
