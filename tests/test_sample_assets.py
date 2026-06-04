from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from noctilux.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = Path("examples/images/sample.jpg")


def test_sample_image_exists_and_is_small() -> None:
    assert SAMPLE_IMAGE.exists()
    assert SAMPLE_IMAGE.stat().st_size < 1_000_000


def test_sample_image_is_readable() -> None:
    with Image.open(SAMPLE_IMAGE) as image:
        assert image.format == "JPEG"
        assert image.size == (800, 600)
        assert image.mode == "RGB"


def test_sample_image_is_not_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(SAMPLE_IMAGE)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_preview_command_with_sample_image_generates_output(tmp_path: Path) -> None:
    output_path = tmp_path / "sample_preview.jpg"

    exit_code = main(
        [
            "preview",
            "--config",
            "configs/examples/full_v020.yaml",
            "--image",
            str(SAMPLE_IMAGE),
            "--output",
            str(output_path),
            "--max-pipelines",
            "6",
            "--seed",
            "42",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0
