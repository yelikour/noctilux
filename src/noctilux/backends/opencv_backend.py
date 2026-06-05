from __future__ import annotations

import numpy as np
from PIL import Image

from noctilux.exceptions import NoctiluxError


class BackendNotAvailableError(NoctiluxError):
    pass


def is_opencv_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("cv2") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def require_opencv() -> None:
    if not is_opencv_available():
        raise BackendNotAvailableError(
            "OpenCV backend requires 'opencv-python'. "
            "Install it with: pip install 'noctilux[opencv]'"
        )


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))[:, :, ::-1].copy()


def cv2_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array[:, :, ::-1].copy(), mode="RGB")


CV2_INTERPOLATION_MAP: dict[str, int] = {}

if is_opencv_available():
    import cv2

    CV2_INTERPOLATION_MAP = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }


def get_cv2_interpolation(interpolation: str, transform_name: str) -> int:
    import cv2

    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if interpolation not in mapping:
        raise ValueError(f"{transform_name} interpolation must be one of {sorted(mapping)}.")
    return mapping[interpolation]
