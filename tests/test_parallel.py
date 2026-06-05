from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from noctilux.worker import (
    ProcessingResult,
    ProcessingTask,
    pre_allocate_output_paths,
    process_task,
)


def _make_sample(tmp_path: Path, name: str = "test.jpg") -> dict[str, Any]:
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    img.save(tmp_path / name)
    return {
        "sample_id": Path(name).stem,
        "image_path": str(tmp_path / name),
        "metadata": {"relative_path": ""},
    }


def _make_output_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "root": str(tmp_path / "output"),
        "image_dir": "images",
        "metadata_dir": "metadata",
        "log_dir": "logs",
        "preview_dir": "previews",
        "save_format": "jpg",
        "jpg_quality": 95,
        "png_compression": 3,
        "overwrite": True,
        "preserve_subdirs": False,
    }


def _make_pipeline_config() -> dict[str, Any]:
    return {
        "name": "resize_test",
        "transforms": [
            {"name": "resize_long_edge", "params": {"long_edge": 32}},
        ],
    }


def _make_task(
    sample: dict[str, Any],
    output_config: dict[str, Any],
    pipeline_config: dict[str, Any] | None = None,
    repeat_index: int = 0,
) -> ProcessingTask:
    output_path = (
        Path(output_config["root"])
        / output_config["image_dir"]
        / "resize_test"
        / f"{sample['sample_id']}__resize_test__{repeat_index:03d}.jpg"
    )
    return ProcessingTask(
        sample_id=sample["sample_id"],
        image_path=Path(sample["image_path"]),
        pipeline_config=pipeline_config or _make_pipeline_config(),
        pipeline_name="resize_test",
        repeat_index=repeat_index,
        seed=42,
        output_path=output_path,
        output_config=output_config,
        sample=sample,
    )


# --- Unit tests ---


def test_process_task_success(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    output_config = _make_output_config(tmp_path)
    task = _make_task(sample, output_config)

    result = process_task(task)
    assert result.success
    assert result.stage == ""
    assert result.error is None
    assert result.output_path.exists()
    assert result.input_info.get("width") == 64
    assert result.output_info.get("width") == 32
    assert len(result.transform_log) == 1


def test_process_task_load_failure(tmp_path: Path) -> None:
    sample = {
        "sample_id": "missing",
        "image_path": str(tmp_path / "nonexistent.jpg"),
        "metadata": {},
    }
    output_config = _make_output_config(tmp_path)
    task = _make_task(sample, output_config)

    result = process_task(task)
    assert not result.success
    assert result.stage == "load_image"
    assert result.error is not None


def test_process_task_transform_failure(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    output_config = _make_output_config(tmp_path)
    pipeline_config = {
        "name": "bad",
        "transforms": [
            {"name": "resize_long_edge", "params": {"long_edge": -1}},
        ],
    }
    task = _make_task(sample, output_config, pipeline_config=pipeline_config)

    result = process_task(task)
    assert not result.success
    assert result.stage == "transform"


def test_pre_allocate_paths(tmp_path: Path) -> None:
    from noctilux.pipeline import AugmentPipeline

    sample = _make_sample(tmp_path)
    output_config = _make_output_config(tmp_path)
    pipeline = AugmentPipeline(name="p1", transforms=[], seed=1)

    paths = pre_allocate_output_paths([sample], [pipeline], output_config)
    assert len(paths) == 1
    key = (sample["sample_id"], "p1", 0)
    assert key in paths
    assert paths[key].name == "test__p1__000.jpg"


def test_pre_allocate_paths_multiple_repeats(tmp_path: Path) -> None:
    from noctilux.pipeline import AugmentPipeline

    sample = _make_sample(tmp_path)
    output_config = _make_output_config(tmp_path)
    pipeline = AugmentPipeline(name="p1", transforms=[], seed=1, repeat=3)

    paths = pre_allocate_output_paths([sample], [pipeline], output_config)
    assert len(paths) == 3
    for i in range(3):
        assert (sample["sample_id"], "p1", i) in paths


def test_seed_determinism_serial_vs_worker(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    output_config = _make_output_config(tmp_path)

    task1 = _make_task(sample, output_config)
    result1 = process_task(task1)

    output_config2 = _make_output_config(tmp_path)
    task2 = _make_task(sample, output_config2)
    result2 = process_task(task2)

    assert result1.seed == result2.seed
    assert result1.output_info == result2.output_info


def test_processing_result_dataclass() -> None:
    result = ProcessingResult(
        sample_id="s1",
        pipeline_name="p1",
        repeat_index=0,
        seed=42,
        output_path=Path("/tmp/out.jpg"),
        success=True,
        stage="",
        error=None,
        input_info={"width": 64},
        output_info={"width": 32},
        transform_log=[{"name": "resize"}],
    )
    assert result.sample_id == "s1"
    assert result.success


# --- Integration tests via CLI ---


def _create_parallel_config(tmp_path: Path, num_workers: int = 2) -> Path:
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)

    for i in range(3):
        img = Image.new("RGB", (64, 64), color=(i * 50, 64, 32))
        img.save(image_dir / f"img{i}.jpg")

    config = {
        "project": {"name": "parallel_test", "seed": 42},
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
            "overwrite": True,
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

    config_path = tmp_path / "config" / "test.yaml"
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
    return config_path


def test_cli_num_workers_1_equals_serial(tmp_path: Path) -> None:
    config_path = _create_parallel_config(tmp_path)
    result = subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--num-workers", "1"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "success_count: 3" in result.stdout


def test_cli_num_workers_2_produces_same_outputs(tmp_path: Path) -> None:
    config_path = _create_parallel_config(tmp_path)
    result = subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--num-workers", "2"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "success_count: 3" in result.stdout

    manifest_path = tmp_path / "output" / "metadata" / "manifest.csv"
    assert manifest_path.exists()
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 3
    assert all(manifest["success"])


def test_cli_parallel_metadata_correctness(tmp_path: Path) -> None:
    config_path = _create_parallel_config(tmp_path)
    subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--num-workers", "2"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    manifest = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    assert set(manifest.columns).issuperset({"sample_id", "pipeline_name", "repeat_index", "success", "seed"})

    transform_log = tmp_path / "output" / "metadata" / "transform_log.jsonl"
    assert transform_log.exists()
    lines = transform_log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    summary = pd.read_csv(tmp_path / "output" / "metadata" / "summary.csv")
    assert len(summary) == 1
    assert summary.iloc[0]["total"] == 3
    assert summary.iloc[0]["success"] == 3


def test_cli_parallel_resume_skips_completed(tmp_path: Path) -> None:
    config_path = _create_parallel_config(tmp_path)

    result1 = subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--num-workers", "2"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result1.returncode == 0
    assert "success_count: 3" in result1.stdout

    result2 = subprocess.run(
        [
            "noctilux", "run", "--config", str(config_path),
            "--num-workers", "2", "--resume",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result2.returncode == 0
    assert "skipped_count: 3" in result2.stdout


def test_cli_parallel_skip_existing(tmp_path: Path) -> None:
    config_path = _create_parallel_config(tmp_path)

    result1 = subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--num-workers", "2"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result1.returncode == 0

    result2 = subprocess.run(
        [
            "noctilux", "run", "--config", str(config_path),
            "--num-workers", "2", "--skip-existing",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result2.returncode == 0
    assert "skipped_count: 3" in result2.stdout


def test_cli_parallel_no_collisions(tmp_path: Path) -> None:
    config_path = _create_parallel_config(tmp_path)
    result = subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--num-workers", "2"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0

    image_dir = tmp_path / "output" / "images" / "resize_test"
    output_files = list(image_dir.glob("*.jpg"))
    assert len(output_files) == 3
    assert len({f.name for f in output_files}) == 3


def test_cli_parallel_with_repeat(tmp_path: Path) -> None:
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    img.save(image_dir / "test.jpg")

    config = {
        "project": {"name": "repeat_test", "seed": 42},
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
            "overwrite": True,
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
                "name": "repeat_pipe",
                "repeat": 3,
                "transforms": [
                    {"name": "resize_long_edge", "params": {"long_edge": 32}},
                ],
            }
        ],
    }

    config_path = tmp_path / "config" / "test.yaml"
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")

    result = subprocess.run(
        ["noctilux", "run", "--config", str(config_path), "--num-workers", "2"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "success_count: 3" in result.stdout
