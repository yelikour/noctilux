from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_image(
    path: str | Path,
    apply_exif_orientation: bool = True,
    convert_mode: str = "RGB",
) -> tuple[Image.Image, dict[str, Any]]:
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")
    try:
        with Image.open(image_path) as opened:
            image = opened.copy()
            source_format = opened.format
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise OSError(f"Failed to load image '{image_path}': {exc}") from exc

    if apply_exif_orientation:
        image = ImageOps.exif_transpose(image)
    if convert_mode:
        image = image.convert(convert_mode)

    info = describe_image(image, image_format=source_format)
    return image, info


def describe_image(image: Image.Image, image_format: str | None = None) -> dict[str, Any]:
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image_format,
    }
