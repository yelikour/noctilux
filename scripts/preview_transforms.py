from __future__ import annotations

import argparse
from pathlib import Path

from noctilux.image_io.loader import load_image
from noctilux.image_io.writer import save_image
from noctilux.registry import build_transform


PREVIEW_SPECS = {
    "jpeg_compression": {"quality": 80, "subsampling": "4:2:0"},
    "resize_long_edge": {"long_edge": 512, "interpolation": "bicubic"},
    "center_crop_ratio": {"ratio": 0.75},
    "gaussian_blur": {"radius": 1.2},
    "gaussian_noise": {"std": 8.0},
    "brightness_contrast": {"brightness": 1.1, "contrast": 1.1},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one preview image per built-in transform.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output-dir", required=True, help="Directory for preview images.")
    args = parser.parse_args()

    image, _ = load_image(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, params in PREVIEW_SPECS.items():
        transform = build_transform(name, params=params)
        preview = transform(image, context={"seed": 0})
        save_image(preview, output_dir / f"{name}.jpg", output_format="jpg", overwrite=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
