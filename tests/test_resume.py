from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from noctilux.resume import (
    build_processing_key,
    check_output_exists,
    load_failed_keys,
    load_success_keys,
    parse_processing_key,
    validate_resume_args,
)

# --- Unit tests for resume utilities ---


def test_build_processing_key_stable() -> None:
    key1 = build_processing_key("s1", "resize", 0)
    key2 = build_processing_key("s1", "resize", 0)
    assert key1 == key2
    assert key1 == "s1::resize::0"


def test_build_processing_key_different_args() -> None:
    k1 = build_processing_key("s1", "resize", 0)
    k2 = build_processing_key("s1", "blur", 0)
    k3 = build_processing_key("s1", "resize", 1)
    assert len({k1, k2, k3}) == 3


def test_parse_processing_key_roundtrip() -> None:
    original = ("s-abc", "pipeline_name", 3)
    key = build_processing_key(*original)
    parsed = parse_processing_key(key)
    assert parsed == original


def test_parse_processing_key_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid processing key"):
        parse_processing_key("no-separators")


def test_load_success_keys_empty_dir(tmp_path: Path) -> None:
    keys = load_success_keys(tmp_path)
    assert keys == set()


def test_load_success_keys_from_manifest(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "pipeline_name": "resize",
                "repeat_index": 0,
                "success": True,
            },
            {
                "sample_id": "s1",
                "pipeline_name": "blur",
                "repeat_index": 0,
                "success": False,
            },
            {
                "sample_id": "s2",
                "pipeline_name": "resize",
                "repeat_index": 0,
                "success": True,
            },
        ]
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    keys = load_success_keys(tmp_path)
    assert keys == {
        build_processing_key("s1", "resize", 0),
        build_processing_key("s2", "resize", 0),
    }


def test_load_success_keys_no_success_column(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [{"sample_id": "s1", "pipeline_name": "resize", "repeat_index": 0}]
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    keys = load_success_keys(tmp_path)
    assert keys == set()


def test_load_failed_keys_empty(tmp_path: Path) -> None:
    keys = load_failed_keys(tmp_path)
    assert keys == set()


def test_load_failed_keys_from_csv(tmp_path: Path) -> None:
    failed = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "pipeline_name": "resize",
                "repeat_index": 0,
                "seed": 42,
                "stage": "transform",
                "error": "x",
            },
            {
                "sample_id": "s2",
                "pipeline_name": "blur",
                "repeat_index": 1,
                "seed": 7,
                "stage": "save_image",
                "error": "y",
            },
        ]
    )
    failed.to_csv(tmp_path / "failed_images.csv", index=False)
    keys = load_failed_keys(tmp_path)
    assert keys == {
        build_processing_key("s1", "resize", 0),
        build_processing_key("s2", "blur", 1),
    }


def test_validate_resume_args_conflict() -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        validate_resume_args(resume=True, retry_failed=True)


def test_validate_resume_args_ok() -> None:
    validate_resume_args(resume=False, retry_failed=False)
    validate_resume_args(resume=True, retry_failed=False)
    validate_resume_args(resume=False, retry_failed=True)


def test_check_output_exists_true(tmp_path: Path) -> None:
    from noctilux.saver import OutputSaver

    output_config = _make_output_config(tmp_path)
    (tmp_path / "images" / "pipe1").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "pipe1" / "img__pipe1__000.jpg").touch()

    saver = OutputSaver(output_config)
    sample = {
        "image_path": str(tmp_path / "img.jpg"),
        "metadata": {"relative_path": ""},
    }
    assert check_output_exists(sample, "pipe1", 0, saver)


def test_check_output_exists_false(tmp_path: Path) -> None:
    from noctilux.saver import OutputSaver

    output_config = _make_output_config(tmp_path)
    saver = OutputSaver(output_config)
    sample = {
        "image_path": str(tmp_path / "img.jpg"),
        "metadata": {"relative_path": ""},
    }
    assert not check_output_exists(sample, "pipe1", 0, saver)


# --- Integration tests via CLI ---


def test_cli_resume_skips_completed(tmp_path: Path) -> None:
    config_path = _create_simple_config(tmp_path, with_sample_image=True)
    result = subprocess.run(
        ["noctilux", "run", "--config", str(config_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "success_count: 1" in result.stdout

    result2 = subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--resume"],
        capture_output=True, text=True, timeout=30,
    )
    assert result2.returncode == 0
    assert "skipped_count: 1" in result2.stdout
    assert "success_count: 0" in result2.stdout


def test_cli_skip_existing(tmp_path: Path) -> None:
    config_path = _create_simple_config(tmp_path, with_sample_image=True)
    result = subprocess.run(
        ["noctilux", "run", "--config", str(config_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "success_count: 1" in result.stdout

    result2 = subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--skip-existing"],
        capture_output=True, text=True, timeout=30,
    )
    assert result2.returncode == 0
    assert "skipped_count: 1" in result2.stdout


def test_cli_retry_failed_no_failures(tmp_path: Path) -> None:
    config_path = _create_simple_config(tmp_path, with_sample_image=True)
    result = subprocess.run(
        ["noctilux", "run", "--config", str(config_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0

    result2 = subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--retry-failed"],
        capture_output=True, text=True, timeout=30,
    )
    assert result2.returncode == 0
    assert "retry_failed_enabled: True" in result2.stdout


def test_cli_resume_and_retry_conflict(tmp_path: Path) -> None:
    config_path = _create_simple_config(tmp_path, with_sample_image=False)
    result = subprocess.run(
        [
            "noctilux", "run", "--config", str(config_path),
            "--resume", "--retry-failed",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert "cannot be used together" in result.stderr


def test_cli_dry_run_resume_no_metadata(tmp_path: Path) -> None:
    config_path = _create_simple_config(tmp_path, with_sample_image=True)
    result = subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--dry-run", "--resume"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    metadata_dir = tmp_path / "output" / "metadata"
    assert not metadata_dir.exists()


def test_cli_run_summary_includes_skipped(tmp_path: Path) -> None:
    config_path = _create_simple_config(tmp_path, with_sample_image=True)
    result = subprocess.run(
        ["noctilux", "run", "--config", str(config_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "skipped_count:" in result.stdout
    assert "resume_enabled:" in result.stdout
    assert "skip_existing_enabled:" in result.stdout
    assert "retry_failed_enabled:" in result.stdout


def test_cli_report_reads_resume_metadata(tmp_path: Path) -> None:
    config_path = _create_simple_config(tmp_path, with_sample_image=True)

    subprocess.run(
        ["noctilux", "run", "--config", str(config_path)],
        capture_output=True, text=True, timeout=30,
    )
    subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--resume"],
        capture_output=True, text=True, timeout=30,
    )

    report_path = tmp_path / "report.md"
    result = subprocess.run(
        [
            "noctilux", "report",
            "--metadata", str(tmp_path / "output" / "metadata"),
            "--output", str(report_path),
            "--overwrite",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Noctilux Metadata Report" in content


# --- Helpers ---


def _make_output_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "root": str(tmp_path),
        "image_dir": "images",
        "metadata_dir": "metadata",
        "log_dir": "logs",
        "preview_dir": "previews",
        "save_format": "jpg",
        "jpg_quality": 95,
        "png_compression": 3,
        "overwrite": False,
        "preserve_subdirs": False,
    }


def _create_simple_config(tmp_path: Path, with_sample_image: bool) -> Path:
    from PIL import Image

    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)

    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)

    if with_sample_image:
        img = Image.new("RGB", (64, 64), color=(128, 64, 32))
        img.save(image_dir / "test.jpg")

    config = {
        "project": {"name": "resume_test", "seed": 42},
        "input": {"mode": "folder", "image_root": str(image_dir)},
        "output": {
            "root": str(tmp_path / "output"),
            "image_dir": "images",
            "metadata_dir": "metadata",
            "log_dir": "logs",
            "preview_dir": "previews",
            "save_format": "jpg",
            "jpg_quality": 95,
            "png_compression": 3,
            "overwrite": False,
            "preserve_subdirs": False,
        },
        "runtime": {
            "dry_run": False,
            "num_workers": 1,
            "skip_broken_images": True,
            "fail_fast": False,
            "show_progress": False,
        },
        "pipelines": [
            {
                "name": "resize_test",
                "transforms": [
                    {"name": "resize_long_edge", "params": {"long_edge": 32}},
                ],
            }
        ],
    }

    import yaml

    config_path = config_dir / "test.yaml"
    config_path.write_text(
        yaml.dump(config, default_flow_style=False), encoding="utf-8"
    )
    return config_path
