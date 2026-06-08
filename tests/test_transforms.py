from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from noctilux.registry import build_transform


def _context(seed: int = 0) -> dict[str, object]:
    return {
        "rng": random.Random(seed),
        "np_rng": np.random.default_rng(seed),
        "seed": seed,
    }


@pytest.mark.parametrize(
    ("name", "params", "valid_modes"),
    [
        ("jpeg_compression", {"quality": 80, "subsampling": "4:2:0"}, {"RGB"}),
        ("webp_compression", {"quality": 80}, {"RGB"}),
        ("png_resave", {"compress_level": 6}, {"RGB"}),
        ("double_jpeg_compression", {"quality1": 90, "quality2": 70}, {"RGB"}),
        ("resize_long_edge", {"long_edge": 32, "interpolation": "nearest"}, {"RGB"}),
        ("resize_exact", {"width": 32, "height": 24, "interpolation": "nearest"}, {"RGB"}),
        ("resize_short_edge", {"short_edge": 24, "interpolation": "nearest"}, {"RGB"}),
        ("downscale_upscale", {"scale": 0.5, "down_interpolation": "nearest", "up_interpolation": "nearest"}, {"RGB"}),
        ("center_crop_ratio", {"ratio": 0.5}, {"RGB"}),
        ("random_crop_ratio", {"ratio": 0.5}, {"RGB"}),
        (
            "random_resized_crop",
            {
                "scale_min": 0.5,
                "scale_max": 0.8,
                "ratio_min": 0.75,
                "ratio_max": 1.25,
                "output_width": 32,
                "output_height": 32,
            },
            {"RGB"},
        ),
        ("square_crop", {}, {"RGB"}),
        ("horizontal_flip", {}, {"RGB"}),
        ("vertical_flip", {}, {"RGB"}),
        ("rotate", {"angle": 15, "expand": False, "fill_color": [0, 0, 0]}, {"RGB"}),
        ("gaussian_blur", {"radius": 1.0}, {"RGB"}),
        ("median_blur", {"size": 3}, {"RGB"}),
        ("motion_blur", {"kernel_size": 5, "angle": 45}, {"RGB"}),
        ("gaussian_noise", {"std": 4.0}, {"RGB"}),
        ("poisson_noise", {"scale": 1.0}, {"RGB"}),
        ("salt_pepper_noise", {"amount": 0.05, "salt_vs_pepper": 0.5}, {"RGB"}),
        ("brightness_contrast", {"brightness": 1.1, "contrast": 1.2}, {"RGB"}),
        ("gamma_correction", {"gamma": 1.2}, {"RGB"}),
        ("saturation_hue", {"saturation": 1.1, "hue_shift": 10}, {"RGB"}),
        ("grayscale", {"keep_rgb": True}, {"RGB"}),
        ("sharpen", {"factor": 1.5}, {"RGB"}),
        ("posterize", {"bits": 6}, {"RGB"}),
    ],
)
def test_transforms_return_pil_images(name: str, params: dict[str, object], valid_modes: set[str]) -> None:
    image = Image.new("RGB", (64, 48), color=(120, 80, 40))
    transform = build_transform(name, params=params)
    output = transform(image, context=_context(0))
    assert isinstance(output, Image.Image)
    assert output.mode in valid_modes
    assert output.width >= 1
    assert output.height >= 1


def test_resize_exact_outputs_requested_size() -> None:
    image = Image.new("RGB", (80, 60), color="red")
    transform = build_transform("resize_exact", params={"width": 33, "height": 21, "interpolation": "nearest"})
    output = transform(image, context=_context(0))
    assert output.size == (33, 21)


def test_resize_short_edge_sets_short_edge() -> None:
    image = Image.new("RGB", (80, 60), color="red")
    transform = build_transform("resize_short_edge", params={"short_edge": 20, "interpolation": "nearest"})
    output = transform(image, context=_context(0))
    assert min(output.size) == 20


def test_downscale_upscale_restores_original_size() -> None:
    image = Image.new("RGB", (80, 60), color="red")
    transform = build_transform(
        "downscale_upscale",
        params={"scale": 0.5, "down_interpolation": "nearest", "up_interpolation": "nearest"},
    )
    output = transform(image, context=_context(0))
    assert output.size == image.size


def test_random_crop_ratio_is_reproducible_with_seed() -> None:
    image = Image.new("RGB", (100, 80), color=(10, 20, 30))
    transform = build_transform("random_crop_ratio", params={"ratio": 0.7})
    output_a = transform(image, context=_context(123))
    output_b = transform(image, context=_context(123))
    assert output_a.size == output_b.size
    assert np.array_equal(np.asarray(output_a), np.asarray(output_b))


def test_random_resized_crop_outputs_requested_size() -> None:
    image = Image.new("RGB", (100, 80), color="red")
    transform = build_transform(
        "random_resized_crop",
        params={
            "scale_min": 0.5,
            "scale_max": 0.8,
            "ratio_min": 0.75,
            "ratio_max": 1.25,
            "output_width": 40,
            "output_height": 30,
        },
    )
    output = transform(image, context=_context(1))
    assert output.size == (40, 30)


def test_square_crop_outputs_square() -> None:
    image = Image.new("RGB", (100, 80), color="red")
    transform = build_transform("square_crop", params={})
    output = transform(image, context=_context(0))
    assert output.size == (80, 80)


def test_rotate_does_not_crash() -> None:
    image = Image.new("RGB", (32, 32), color="red")
    transform = build_transform("rotate", params={"angle": 15, "expand": True, "fill_color": [0, 0, 0]})
    output = transform(image, context=_context(0))
    assert isinstance(output, Image.Image)


@pytest.mark.parametrize("angle", [0, 90])
def test_motion_blur_runs_for_cardinal_angles(angle: int) -> None:
    image = Image.new("RGB", (40, 30), color="red")
    transform = build_transform("motion_blur", params={"kernel_size": 5, "angle": angle})
    output = transform(image, context=_context(0))
    assert output.size == image.size
    assert output.mode == "RGB"


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("jpeg_compression", {"quality": 101, "subsampling": "4:2:0"}),
        ("webp_compression", {"quality": 0}),
        ("png_resave", {"compress_level": 10}),
        ("double_jpeg_compression", {"quality1": 90, "quality2": 120}),
        ("resize_long_edge", {"long_edge": 0, "interpolation": "nearest"}),
        ("resize_exact", {"width": 0, "height": 32, "interpolation": "nearest"}),
        ("resize_short_edge", {"short_edge": 0, "interpolation": "nearest"}),
        ("downscale_upscale", {"scale": 1.0, "down_interpolation": "nearest", "up_interpolation": "nearest"}),
        ("center_crop_ratio", {"ratio": 1.5}),
        ("random_crop_ratio", {"ratio": 0}),
        (
            "random_resized_crop",
            {
                "scale_min": 0.8,
                "scale_max": 0.5,
                "ratio_min": 0.75,
                "ratio_max": 1.25,
                "output_width": 32,
                "output_height": 32,
            },
        ),
        ("gaussian_blur", {"radius": -1}),
        ("median_blur", {"size": 4}),
        ("motion_blur", {"kernel_size": 4, "angle": 45}),
        ("gaussian_noise", {"std": -0.1}),
        ("poisson_noise", {"scale": 0}),
        ("salt_pepper_noise", {"amount": 1.5, "salt_vs_pepper": 0.5}),
        ("gamma_correction", {"gamma": 0}),
        ("posterize", {"bits": 9}),
        ("brightness_contrast", {"brightness": -1, "contrast": 1.0}),
    ],
)
def test_invalid_transform_params_raise(name: str, params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        build_transform(name, params=params)


# --- crop_window metadata tests ---


def test_center_crop_ratio_records_crop_window() -> None:
    image = Image.new("RGB", (100, 80))
    transform = build_transform("center_crop_ratio", params={"ratio": 0.5})
    ctx = _context(0)
    output = transform(image, context=ctx)

    cw = ctx["crop_window"]
    assert cw == {
        "x": 25,
        "y": 20,
        "width": 50,
        "height": 40,
        "source_width": 100,
        "source_height": 80,
    }
    assert output.size == (cw["width"], cw["height"])


def test_random_crop_ratio_records_crop_window() -> None:
    image = Image.new("RGB", (100, 80))
    transform = build_transform("random_crop_ratio", params={"ratio": 0.7})
    ctx = _context(42)
    output = transform(image, context=ctx)

    cw = ctx["crop_window"]
    assert isinstance(cw["x"], int)
    assert isinstance(cw["y"], int)
    assert cw["width"] == 70
    assert cw["height"] == 56
    assert cw["source_width"] == 100
    assert cw["source_height"] == 80
    assert output.size == (cw["width"], cw["height"])


def test_random_crop_ratio_crop_window_is_seed_deterministic() -> None:
    image = Image.new("RGB", (100, 80))
    transform = build_transform("random_crop_ratio", params={"ratio": 0.7})

    ctx_a = _context(42)
    transform(image, context=ctx_a)

    ctx_b = _context(42)
    transform(image, context=ctx_b)

    assert ctx_a["crop_window"] == ctx_b["crop_window"]


def test_square_crop_records_crop_window() -> None:
    image = Image.new("RGB", (100, 80))
    transform = build_transform("square_crop", params={})
    ctx = _context(0)
    output = transform(image, context=ctx)

    cw = ctx["crop_window"]
    assert cw == {
        "x": 10,
        "y": 0,
        "width": 80,
        "height": 80,
        "source_width": 100,
        "source_height": 80,
    }
    assert output.size == (80, 80)


def test_random_resized_crop_records_crop_window() -> None:
    image = Image.new("RGB", (100, 80))
    transform = build_transform(
        "random_resized_crop",
        params={
            "scale_min": 0.5,
            "scale_max": 0.8,
            "ratio_min": 0.75,
            "ratio_max": 1.25,
            "output_width": 40,
            "output_height": 30,
        },
    )
    ctx = _context(99)
    output = transform(image, context=ctx)

    cw = ctx["crop_window"]
    assert isinstance(cw["x"], int)
    assert isinstance(cw["y"], int)
    assert isinstance(cw["width"], int)
    assert isinstance(cw["height"], int)
    assert cw["source_width"] == 100
    assert cw["source_height"] == 80
    assert 0 <= cw["x"]
    assert cw["x"] + cw["width"] <= 100
    assert 0 <= cw["y"]
    assert cw["y"] + cw["height"] <= 80
    assert output.size == (40, 30)


def test_random_resized_crop_crop_window_is_seed_deterministic() -> None:
    image = Image.new("RGB", (100, 80))
    transform = build_transform(
        "random_resized_crop",
        params={
            "scale_min": 0.5,
            "scale_max": 0.8,
            "ratio_min": 0.75,
            "ratio_max": 1.25,
            "output_width": 40,
            "output_height": 30,
        },
    )

    ctx_a = _context(99)
    transform(image, context=ctx_a)

    ctx_b = _context(99)
    transform(image, context=ctx_b)

    assert ctx_a["crop_window"] == ctx_b["crop_window"]


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("resize_exact", {"width": 50, "height": 40, "interpolation": "nearest"}),
        ("resize_long_edge", {"long_edge": 50, "interpolation": "nearest"}),
        ("horizontal_flip", {}),
        ("vertical_flip", {}),
        ("brightness_contrast", {"brightness": 1.0, "contrast": 1.0}),
        ("gaussian_blur", {"radius": 1.0}),
        ("jpeg_compression", {"quality": 80}),
    ],
)
def test_non_crop_transforms_do_not_produce_crop_window(name: str, params: dict[str, object]) -> None:
    image = Image.new("RGB", (100, 80))
    transform = build_transform(name, params=params)
    ctx = _context(0)
    transform(image, context=ctx)
    assert "crop_window" not in ctx, f"{name} should not produce crop_window"


def test_pipeline_crop_transform_produces_crop_window_in_log() -> None:
    from noctilux.pipeline import AugmentPipeline

    image = Image.new("RGB", (100, 80))
    pipeline = AugmentPipeline(
        name="crop_test",
        transforms=[
            {"name": "center_crop_ratio", "params": {"ratio": 0.5}},
            {"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}},
        ],
        seed=42,
    )
    _, transform_logs, _ = pipeline.apply(
        image=image,
        sample={"sample_id": "test", "image_path": "test.jpg"},
    )

    assert len(transform_logs) == 2
    crop_log = transform_logs[0]
    assert "crop_window" in crop_log
    assert crop_log["crop_window"]["width"] == 50
    assert crop_log["crop_window"]["height"] == 40
    assert crop_log["crop_window"]["source_width"] == 100
    assert crop_log["crop_window"]["source_height"] == 80
    assert crop_log["input_size"] == [100, 80]
    assert crop_log["output_size"] == [50, 40]

    photo_log = transform_logs[1]
    assert "crop_window" not in photo_log
    assert photo_log["input_size"] == [50, 40]
