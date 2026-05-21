from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from noctilux.saver import OutputSaver


def _output_config(root: Path, overwrite: bool = False) -> dict[str, object]:
    return {
        "root": root,
        "image_dir": "images",
        "metadata_dir": "metadata",
        "log_dir": "logs",
        "preview_dir": "previews",
        "preserve_subdirs": True,
        "overwrite": overwrite,
        "save_format": "jpg",
        "jpg_quality": 95,
        "png_compression": 3,
    }


def test_overwrite_false_uses_safe_suffix_on_conflict(tmp_path: Path) -> None:
    saver = OutputSaver(_output_config(tmp_path, overwrite=False))
    saver.prepare_directories()
    sample = {
        "image_path": tmp_path / "input.jpg",
        "metadata": {"relative_path": "class_a/input.jpg"},
    }
    image = Image.new("RGB", (16, 16), color="red")

    first_target = saver.build_output_path(sample, "resize", 0)
    saver.save(image, first_target)
    second_target = saver.build_output_path(sample, "resize", 0)

    assert second_target != first_target
    assert second_target.name.endswith("__dup1.jpg")


def test_output_path_rejects_escaping_relative_subdirs(tmp_path: Path) -> None:
    saver = OutputSaver(_output_config(tmp_path, overwrite=False))
    saver.prepare_directories()
    sample = {
        "image_path": tmp_path / "input.jpg",
        "metadata": {"relative_path": "../../evil/input.jpg"},
    }

    with pytest.raises(ValueError, match="unsafe"):
        saver.build_output_path(sample, "resize", 0)
