from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_preview_script_generates_nonempty_grid(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    output_path = tmp_path / "preview_grid.jpg"
    Image.new("RGB", (96, 64), color=(120, 80, 40)).save(image_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/preview_transforms.py",
            "--config",
            "configs/presets/all_basic_v021.yaml",
            "--image",
            str(image_path),
            "--output",
            str(output_path),
            "--max-pipelines",
            "4",
            "--seed",
            "42",
        ],
        cwd="/home/yeli/data/Projects/Noctilux",
        check=True,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
