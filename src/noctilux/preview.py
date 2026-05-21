from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from noctilux.config import load_config, resolve_config, validate_config
from noctilux.image_io.loader import load_image
from noctilux.image_io.writer import save_image
from noctilux.pipeline import build_pipelines

CAPTION_HEIGHT = 28
GRID_BACKGROUND = (245, 245, 245)
TILE_BACKGROUND = (255, 255, 255)
TEXT_COLOR = (20, 20, 20)


def add_preview_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--image", required=True, help="Path to a single input image.")
    parser.add_argument("--output", required=True, help="Output preview grid image path.")
    parser.add_argument(
        "--max-pipelines",
        type=int,
        default=8,
        help="Maximum number of pipelines to preview.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed for preview generation.")


def create_preview_grid(
    config_path: Path,
    image_path: Path,
    output_path: Path,
    max_pipelines: int = 8,
    seed: int | None = None,
) -> Path:
    if max_pipelines < 1:
        raise ValueError("max_pipelines must be >= 1.")
    if not image_path.exists():
        raise FileNotFoundError(f"Preview input image does not exist: {image_path}")

    config = resolve_config(load_config(config_path))
    validate_config(config)
    pipelines = build_pipelines(config)
    if not pipelines:
        raise ValueError(f"Config does not contain any enabled pipelines: {config_path}")

    image, _ = load_image(image_path)
    sample = {
        "sample_id": image_path.stem,
        "image_path": image_path,
        "label": "",
        "split": "preview",
        "task": "generic",
        "metadata": {"relative_path": image_path.name, "relative_dir": ""},
    }

    previews = [("original", image.copy())]
    for pipeline in pipelines[:max_pipelines]:
        output_image, _, _ = pipeline.apply(
            image=image,
            sample=sample,
            repeat_index=0,
            seed=seed,
        )
        previews.append((pipeline.name, output_image))

    grid = build_preview_grid_image(previews)
    resolved_output = Path(output_path)
    save_image(
        grid,
        resolved_output,
        output_format=resolved_output.suffix.lstrip(".") or "jpg",
        overwrite=True,
    )
    return resolved_output


def build_preview_grid_image(previews: Iterable[tuple[str, Image.Image]]) -> Image.Image:
    preview_list = list(previews)
    if not preview_list:
        raise ValueError("No previews were generated.")

    tile_width = max(image.width for _, image in preview_list)
    tile_height = max(image.height for _, image in preview_list)
    cell_width = tile_width
    cell_height = tile_height + CAPTION_HEIGHT
    columns = max(1, math.ceil(math.sqrt(len(preview_list))))
    rows = math.ceil(len(preview_list) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), color=GRID_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (label, image) in enumerate(preview_list):
        row = index // columns
        col = index % columns
        x0 = col * cell_width
        y0 = row * cell_height
        tile = Image.new("RGB", (cell_width, tile_height), color=TILE_BACKGROUND)
        fitted = ImageOps.contain(image.copy().convert("RGB"), (cell_width, tile_height))
        offset = ((cell_width - fitted.width) // 2, (tile_height - fitted.height) // 2)
        tile.paste(fitted, offset)
        canvas.paste(tile, (x0, y0))
        draw.rectangle(
            (x0, y0 + tile_height, x0 + cell_width, y0 + cell_height),
            fill=GRID_BACKGROUND,
        )
        draw.text((x0 + 6, y0 + tile_height + 6), label, fill=TEXT_COLOR, font=font)
    return canvas
