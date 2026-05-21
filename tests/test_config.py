from __future__ import annotations

from pathlib import Path

import pytest

from noctilux.config import load_config, resolve_config, validate_config


def test_load_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: demo
input:
  mode: folder
  image_root: images
output:
  root: outputs
runtime:
  dry_run: true
pipelines:
  - name: resize
    transforms:
      - name: resize_long_edge
        params:
          long_edge: 64
""".strip(),
        encoding="utf-8",
    )

    config = resolve_config(load_config(config_path))
    validate_config(config)

    assert config["project"]["name"] == "demo"
    assert config["runtime"]["dry_run"] is True


def test_missing_required_fields_raise(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text(
        """
project:
  name: broken
input:
  mode: folder
  image_root: images
output:
  root: outputs
""".strip(),
        encoding="utf-8",
    )

    config = resolve_config(load_config(config_path))
    with pytest.raises(ValueError, match="pipelines must be a non-empty list"):
        validate_config(config)


def test_example_config_passes_validation() -> None:
    config = resolve_config(load_config("configs/examples/basic_resize.yaml"))
    validate_config(config)
    assert len(config["pipelines"]) == 1


def test_invalid_pipeline_name_has_clear_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
project:
  name: invalid
input:
  mode: folder
  image_root: images
output:
  root: outputs
runtime:
  dry_run: true
pipelines:
  - name: ../escape
    transforms:
      - name: resize_long_edge
        params:
          long_edge: 64
""".strip(),
        encoding="utf-8",
    )

    config = resolve_config(load_config(config_path))
    with pytest.raises(ValueError, match="single safe path component"):
        validate_config(config)
