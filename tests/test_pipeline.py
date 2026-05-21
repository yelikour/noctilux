from __future__ import annotations

import numpy as np
from PIL import Image

from noctilux.pipeline import AugmentPipeline


def test_single_transform_pipeline_executes() -> None:
    image = Image.new("RGB", (100, 50), color="red")
    pipeline = AugmentPipeline(
        name="resize",
        transforms=[{"name": "resize_long_edge", "params": {"long_edge": 20, "interpolation": "nearest"}}],
        seed=123,
    )

    output, logs, _ = pipeline.apply(image, sample={"sample_id": "sample-1"}, repeat_index=0)

    assert output.size == (20, 10)
    assert logs[0]["applied"] is True


def test_multi_transform_pipeline_preserves_order() -> None:
    image = Image.new("RGB", (100, 50), color="red")
    pipeline = AugmentPipeline(
        name="crop_resize",
        transforms=[
            {"name": "center_crop_ratio", "params": {"ratio": 0.5}},
            {"name": "resize_long_edge", "params": {"long_edge": 20, "interpolation": "nearest"}},
        ],
        seed=123,
    )

    output, _, _ = pipeline.apply(image, sample={"sample_id": "sample-1"}, repeat_index=0)

    assert output.size == (20, 10)


def test_probability_zero_records_not_applied() -> None:
    image = Image.new("RGB", (50, 50), color="red")
    pipeline = AugmentPipeline(
        name="skip_blur",
        transforms=[{"name": "gaussian_blur", "p": 0.0, "params": {"radius": 2.0}}],
        seed=123,
    )

    output, logs, _ = pipeline.apply(image, sample={"sample_id": "sample-1"}, repeat_index=0)

    assert output.size == image.size
    assert logs[0]["applied"] is False


def test_probability_one_always_applies() -> None:
    image = Image.new("RGB", (50, 50), color="red")
    pipeline = AugmentPipeline(
        name="always_blur",
        transforms=[{"name": "gaussian_blur", "p": 1.0, "params": {"radius": 2.0}}],
        seed=123,
    )

    _, logs, _ = pipeline.apply(image, sample={"sample_id": "sample-1"}, repeat_index=0)

    assert logs[0]["applied"] is True


def test_random_params_are_reproducible_with_seed() -> None:
    image = Image.new("RGB", (32, 32), color=(120, 120, 120))
    transforms = [
        {
            "name": "gaussian_noise",
            "params": {
                "std": {
                    "type": "choice",
                    "values": [1.0, 3.0, 5.0],
                }
            },
        }
    ]
    pipeline_a = AugmentPipeline(name="noise", transforms=transforms, seed=77)
    pipeline_b = AugmentPipeline(name="noise", transforms=transforms, seed=77)

    output_a, logs_a, seed_a = pipeline_a.apply(image, sample={"sample_id": "s1"}, repeat_index=0)
    output_b, logs_b, seed_b = pipeline_b.apply(image, sample={"sample_id": "s1"}, repeat_index=0)

    assert seed_a == seed_b
    assert logs_a == logs_b
    assert np.array_equal(np.asarray(output_a), np.asarray(output_b))


def test_random_params_log_actual_sampled_value() -> None:
    image = Image.new("RGB", (32, 32), color=(120, 120, 120))
    pipeline = AugmentPipeline(
        name="noise",
        transforms=[
            {
                "name": "gaussian_noise",
                "params": {
                    "std": {"type": "randint", "min": 2, "max": 5},
                },
            }
        ],
        seed=91,
    )

    _, logs, _ = pipeline.apply(image, sample={"sample_id": "s1"}, repeat_index=0)

    assert isinstance(logs[0]["params"]["std"], int)
    assert 2 <= logs[0]["params"]["std"] <= 5
