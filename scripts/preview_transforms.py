from __future__ import annotations

import argparse
from pathlib import Path

from noctilux.preview import add_preview_arguments, create_preview_grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a preview grid from a Noctilux config and a single image."
    )
    add_preview_arguments(parser)
    args = parser.parse_args(argv)
    create_preview_grid(
        config_path=Path(args.config),
        image_path=Path(args.image),
        output_path=Path(args.output),
        max_pipelines=args.max_pipelines,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
