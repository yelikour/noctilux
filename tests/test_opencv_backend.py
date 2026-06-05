from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")

from noctilux.backends.opencv_backend import cv2_to_pil, pil_to_cv2  # noqa: E402
from noctilux.registry import build_transform  # noqa: E402


def _make_image(w: int = 100, h: int = 80) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 64, 32))


def test_pil_to_cv2_round_trip() -> None:
    img = _make_image()
    arr = pil_to_cv2(img)
    assert arr.shape == (80, 100, 3)
    restored = cv2_to_pil(arr)
    assert restored.size == (100, 80)
    assert restored.mode == "RGB"


def test_resize_exact_opencv() -> None:
    t = build_transform("resize_exact", params={"width": 50, "height": 40}, backend="opencv")
    result = t(_make_image(100, 80))
    assert result.size == (50, 40)
    assert isinstance(result, Image.Image)


def test_resize_long_edge_opencv() -> None:
    t = build_transform("resize_long_edge", params={"long_edge": 50}, backend="opencv")
    result = t(_make_image(100, 60))
    long_edge = max(result.size)
    assert long_edge == 50
    assert isinstance(result, Image.Image)


def test_gaussian_blur_opencv() -> None:
    t = build_transform("gaussian_blur", params={"radius": 2.0}, backend="opencv")
    result = t(_make_image(100, 80))
    assert result.size == (100, 80)
    assert isinstance(result, Image.Image)


def test_rotate_opencv() -> None:
    t = build_transform("rotate", params={"angle": 30, "fill_color": 0}, backend="opencv")
    result = t(_make_image(100, 80))
    assert result.size == (100, 80)
    assert isinstance(result, Image.Image)


def test_rotate_opencv_expand_raises() -> None:
    t = build_transform("rotate", params={"angle": 30, "expand": True, "fill_color": 0}, backend="opencv")
    with pytest.raises(NotImplementedError):
        t(_make_image())


def test_opencv_config_dry_run() -> None:
    from noctilux.cli import main

    exit_code = main(["run", "--config", str(Path("configs/examples/opencv_backend.yaml")), "--dry-run"])
    assert exit_code == 0
