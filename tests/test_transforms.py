from __future__ import annotations

import pytest
from PIL import Image

from noctilux.registry import build_transform


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("jpeg_compression", {"quality": 80, "subsampling": "4:2:0"}),
        ("resize_long_edge", {"long_edge": 32, "interpolation": "nearest"}),
        ("center_crop_ratio", {"ratio": 0.5}),
        ("gaussian_blur", {"radius": 1.0}),
        ("gaussian_noise", {"std": 4.0}),
        ("brightness_contrast", {"brightness": 1.1, "contrast": 1.2}),
    ],
)
def test_transforms_return_pil_images(name: str, params: dict[str, object]) -> None:
    image = Image.new("RGB", (64, 48), color=(120, 80, 40))
    transform = build_transform(name, params=params)
    output = transform(image, context={"seed": 0})
    assert isinstance(output, Image.Image)
    assert output.mode == "RGB"
    assert output.width >= 1
    assert output.height >= 1


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("jpeg_compression", {"quality": 101, "subsampling": "4:2:0"}),
        ("resize_long_edge", {"long_edge": 0, "interpolation": "nearest"}),
        ("center_crop_ratio", {"ratio": 1.5}),
        ("gaussian_blur", {"radius": -1}),
        ("gaussian_noise", {"std": -0.1}),
        ("brightness_contrast", {"brightness": -1, "contrast": 1.0}),
    ],
)
def test_invalid_transform_params_raise(name: str, params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        build_transform(name, params=params)
