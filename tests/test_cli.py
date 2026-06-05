from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from noctilux.cli import main
from noctilux.registry import TRANSFORM_REGISTRY
from noctilux.transforms.base import BaseTransform


def _write_config(tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _base_config(tmp_path: Path, *, mode: str = "folder", dry_run: bool = False, repeat: int = 1) -> dict:
    image_root = tmp_path / "images"
    output_root = tmp_path / "output"
    base = {
        "project": {"name": "cli-test", "seed": 42},
        "input": {
            "mode": mode,
            "image_root": str(image_root),
            "infer_label_from_subdir": True,
            "recursive": True,
        },
        "output": {
            "root": str(output_root),
            "preserve_subdirs": True,
            "overwrite": False,
            "save_format": "jpg",
        },
        "runtime": {
            "dry_run": dry_run,
            "num_workers": 1,
            "skip_broken_images": True,
            "fail_fast": False,
            "show_progress": False,
        },
        "pipelines": [
            {
                "name": "resize_32",
                "repeat": repeat,
                "transforms": [
                    {
                        "name": "resize_long_edge",
                        "params": {"long_edge": 32, "interpolation": "nearest"},
                    }
                ],
            }
        ],
    }
    return base


def _make_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 80, 40)).save(path)


def test_dry_run_does_not_write_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _base_config(tmp_path, dry_run=True)
    _make_image(tmp_path / "images" / "class_a" / "sample.jpg")
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "total_samples: 1" in captured.out
    assert not (tmp_path / "output").exists()


def test_run_records_failed_transform_in_failed_images(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    class FailingTransform(BaseTransform):
        name = "always_fail"

        def __call__(self, image, context=None):
            raise RuntimeError("intentional failure")

    monkey_name = "test_always_fail"
    original = TRANSFORM_REGISTRY.get(monkey_name)
    TRANSFORM_REGISTRY[monkey_name] = FailingTransform
    try:
        config = _base_config(tmp_path)
        config["pipelines"][0]["transforms"] = [{"name": monkey_name, "params": {}}]
        _make_image(tmp_path / "images" / "class_a" / "sample.jpg")
        config_path = _write_config(tmp_path, config)

        exit_code = main(["run", "--config", str(config_path)])

        assert exit_code == 0
        failed = pd.read_csv(tmp_path / "output" / "metadata" / "failed_images.csv")
        manifest = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
        lines = (tmp_path / "output" / "metadata" / "transform_log.jsonl").read_text(encoding="utf-8").splitlines()
        payload = json.loads(lines[0])
        assert len(failed) == 1
        assert failed.loc[0, "pipeline_name"] == "resize_32"
        assert failed.loc[0, "repeat_index"] == 0
        assert failed.loc[0, "seed"] > 0
        assert failed.loc[0, "stage"] == "transform"
        assert "intentional failure" in failed.loc[0, "error"]
        assert bool(manifest.loc[0, "success"]) is False
        assert payload["success"] is False
        assert payload["transforms"][0]["error"] == "intentional failure"
    finally:
        if original is None:
            TRANSFORM_REGISTRY.pop(monkey_name, None)
        else:
            TRANSFORM_REGISTRY[monkey_name] = original
    capsys.readouterr()


def test_run_repeat_three_generates_three_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _base_config(tmp_path, repeat=3)
    _make_image(tmp_path / "images" / "class_a" / "sample.jpg")
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])

    assert exit_code == 0
    output_dir = tmp_path / "output" / "images" / "resize_32" / "class_a"
    outputs = sorted(output_dir.glob("*.jpg"))
    assert [path.name for path in outputs] == [
        "sample__resize_32__000.jpg",
        "sample__resize_32__001.jpg",
        "sample__resize_32__002.jpg",
    ]
    capsys.readouterr()


def test_inspect_config_prints_required_fields(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _base_config(tmp_path, dry_run=True)
    config_path = _write_config(tmp_path, config)

    exit_code = main(["inspect-config", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "project: cli-test" in captured.out
    assert "input_mode: folder" in captured.out
    assert "image_root:" in captured.out
    assert "output_root:" in captured.out
    assert "dry_run: True" in captured.out
    assert "overwrite: False" in captured.out
    assert "num_workers: 1" in captured.out


def test_make_manifest_recurses_and_filters_non_images(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "images"
    _make_image(root / "class_a" / "sample.jpg")
    _make_image(root / "class_a" / "nested" / "sample2.png")
    (root / "class_a" / "note.txt").write_text("ignore", encoding="utf-8")
    output_path = tmp_path / "manifest.csv"

    exit_code = main(
        [
            "make-manifest",
            "--image-root",
            str(root),
            "--output",
            str(output_path),
            "--infer-label-from-subdir",
        ]
    )

    manifest = pd.read_csv(output_path)
    assert exit_code == 0
    assert len(manifest) == 2
    assert set(manifest["label"]) == {"class_a"}
    capsys.readouterr()


def test_run_records_load_image_failure_stage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _base_config(tmp_path)
    broken_path = tmp_path / "images" / "class_a" / "broken.jpg"
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    broken_path.write_text("not-an-image", encoding="utf-8")
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])

    assert exit_code == 0
    failed = pd.read_csv(tmp_path / "output" / "metadata" / "failed_images.csv")
    assert len(failed) == 1
    assert failed.loc[0, "pipeline_name"] == "resize_32"
    assert failed.loc[0, "repeat_index"] == 0
    assert failed.loc[0, "seed"] > 0
    assert failed.loc[0, "stage"] == "load_image"
    capsys.readouterr()


def test_serial_load_failure_respects_skip_broken_false(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _base_config(tmp_path)
    config["runtime"]["skip_broken_images"] = False
    broken_path = tmp_path / "images" / "class_a" / "broken.jpg"
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    broken_path.write_text("not-an-image", encoding="utf-8")
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])

    assert exit_code == 1
    failed = pd.read_csv(tmp_path / "output" / "metadata" / "failed_images.csv")
    assert len(failed) == 1
    assert failed.loc[0, "stage"] == "load_image"
    capsys.readouterr()


def test_serial_load_failure_skip_broken_true_records_and_continues(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _base_config(tmp_path)
    broken_path = tmp_path / "images" / "class_a" / "broken.jpg"
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    broken_path.write_text("not-an-image", encoding="utf-8")
    _make_image(tmp_path / "images" / "class_a" / "sample.jpg")
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])

    assert exit_code == 0
    failed = pd.read_csv(tmp_path / "output" / "metadata" / "failed_images.csv")
    manifest = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    lines = (tmp_path / "output" / "metadata" / "transform_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(failed) == 1
    assert failed.loc[0, "stage"] == "load_image"
    assert manifest["success"].tolist().count(True) == 1
    for line in lines:
        json.loads(line)
    capsys.readouterr()


def test_serial_save_failure_respects_skip_broken_false(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _base_config(tmp_path)
    config["output"]["overwrite"] = True
    config["runtime"]["skip_broken_images"] = False
    _make_image(tmp_path / "images" / "class_a" / "sample.jpg")
    target_dir = tmp_path / "output" / "images" / "resize_32" / "class_a" / "sample__resize_32__000.jpg"
    target_dir.mkdir(parents=True)
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])

    assert exit_code == 1
    failed = pd.read_csv(tmp_path / "output" / "metadata" / "failed_images.csv")
    summary = pd.read_csv(tmp_path / "output" / "metadata" / "summary.csv")
    assert len(failed) == 1
    assert failed.loc[0, "stage"] == "save_image"
    assert summary.iloc[0]["failed"] == 1
    capsys.readouterr()


def test_num_workers_gt_one_uses_parallel_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _base_config(tmp_path)
    config["runtime"]["num_workers"] = 2
    _make_image(tmp_path / "images" / "class_a" / "sample.jpg")
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "success_count: 1" in captured.out
    output_dir = tmp_path / "output" / "images" / "resize_32" / "class_a"
    outputs = sorted(output_dir.glob("*.jpg"))
    assert len(outputs) == 1


def test_full_v020_example_supports_cli_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--config", "configs/examples/full_v020.yaml", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "total_outputs: 1" in captured.out


def test_all_basic_v021_preset_supports_cli_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--config", "configs/presets/all_basic_v021.yaml", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "total_outputs: 1" in captured.out


def test_quickstart_sample_config_supports_cli_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--config", "configs/examples/quickstart_sample.yaml", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "total_outputs:" in captured.out


def test_preview_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["preview", "--help"])

    captured = capsys.readouterr()
    assert "--config" in captured.out
    assert "--image" in captured.out
    assert "--output" in captured.out


def test_preview_command_generates_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image_path = tmp_path / "sample.jpg"
    output_path = tmp_path / "preview_grid.jpg"
    _make_image(image_path, size=(96, 64))

    exit_code = main(
        [
            "preview",
            "--config",
            "configs/examples/full_v020.yaml",
            "--image",
            str(image_path),
            "--output",
            str(output_path),
            "--max-pipelines",
            "4",
            "--seed",
            "42",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(output_path) in captured.out
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_preview_command_missing_image_returns_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    output_path = tmp_path / "preview_grid.jpg"

    exit_code = main(
        [
            "preview",
            "--config",
            "configs/examples/full_v020.yaml",
            "--image",
            str(tmp_path / "missing.jpg"),
            "--output",
            str(output_path),
        ]
    )

    capsys.readouterr()
    assert exit_code == 1
    assert "Preview input image does not exist" in caplog.text
