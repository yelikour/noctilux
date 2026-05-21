from __future__ import annotations

from pathlib import Path

from PIL import Image

SUPPORTED_OUTPUT_FORMATS = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}


def normalize_extension(extension: str) -> str:
    normalized = extension.lower().lstrip(".")
    if normalized not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {extension}")
    return "jpg" if normalized == "jpeg" else normalized


def save_image(
    image: Image.Image,
    path: str | Path,
    output_format: str | None = None,
    overwrite: bool = False,
    jpg_quality: int = 95,
    png_compression: int = 3,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")

    extension = normalize_extension(output_format or target.suffix or "jpg")
    pil_format = SUPPORTED_OUTPUT_FORMATS[extension]
    save_kwargs: dict[str, object] = {}
    save_image_obj = image.copy()

    if extension == "jpg":
        save_image_obj = save_image_obj.convert("RGB")
        save_kwargs["quality"] = int(jpg_quality)
    elif extension == "png":
        save_kwargs["compress_level"] = int(png_compression)

    save_image_obj.save(target, format=pil_format, **save_kwargs)
    return target
