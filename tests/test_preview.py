from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from noctilux.preview import create_preview_grid


def test_create_preview_grid_supports_max_pipelines_one(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    output_path = tmp_path / "preview_grid.jpg"
    Image.new("RGB", (96, 64), color=(120, 80, 40)).save(image_path)

    result = create_preview_grid(
        config_path=Path("configs/examples/full_v020.yaml"),
        image_path=image_path,
        output_path=output_path,
        max_pipelines=1,
        seed=42,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_create_preview_grid_does_not_write_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    output_path = tmp_path / "nested" / "preview_grid.jpg"
    Image.new("RGB", (96, 64), color=(120, 80, 40)).save(image_path)

    create_preview_grid(
        config_path=Path("configs/presets/all_basic_v021.yaml"),
        image_path=image_path,
        output_path=output_path,
        max_pipelines=2,
        seed=42,
    )

    assert output_path.exists()
    assert not (tmp_path / "metadata").exists()
    assert not any(tmp_path.rglob("manifest.csv"))
    assert not any(tmp_path.rglob("transform_log.jsonl"))
    assert not any(tmp_path.rglob("failed_images.csv"))


def test_create_preview_grid_missing_image_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Preview input image does not exist"):
        create_preview_grid(
            config_path=Path("configs/examples/full_v020.yaml"),
            image_path=tmp_path / "missing.jpg",
            output_path=tmp_path / "preview_grid.jpg",
        )
