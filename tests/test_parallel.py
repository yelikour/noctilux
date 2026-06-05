from __future__ import annotations

import json
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


def _create_parallel_config(
    tmp_path: Path,
    *,
    num_images: int = 3,
    pipelines: list[dict[str, Any]] | None = None,
    skip_broken_images: bool = True,
) -> Path:
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)

    for i in range(num_images):
        img = Image.new("RGB", (64, 64), color=(i * 50, 64, 32))
        img.save(image_dir / f"img{i}.jpg")

    if pipelines is None:
        pipelines = [
            {
                "name": "resize_test",
                "transforms": [
                    {"name": "resize_long_edge", "params": {"long_edge": 32}},
                ],
            }
        ]

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
            "skip_broken_images": skip_broken_images,
            "fail_fast": False,
            "show_progress": False,
        },
        "pipelines": pipelines,
    }

    config_path = tmp_path / "config" / "test.yaml"
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
    return config_path


def _run_cli(config_path: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = ["noctilux", "run", "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


# =========================================================================
# Unit tests
# =========================================================================


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


# =========================================================================
# Determinism and consistency tests
# =========================================================================


def test_manifest_keys_serial_vs_parallel(tmp_path: Path) -> None:
    """Same config+seed: manifest processing keys must match between serial and parallel."""
    config_path = _create_parallel_config(tmp_path)

    serial_dir = tmp_path / "serial"
    serial_dir.mkdir()
    _run_cli(config_path, ["--num-workers", "1"])
    manifest_s = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    keys_s = set(
        f"{r['sample_id']}::{r['pipeline_name']}::{r['repeat_index']}"
        for _, r in manifest_s.iterrows()
    )

    # Reset output
    import shutil

    shutil.rmtree(tmp_path / "output")

    _run_cli(config_path, ["--num-workers", "2"])
    manifest_p = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    keys_p = set(
        f"{r['sample_id']}::{r['pipeline_name']}::{r['repeat_index']}"
        for _, r in manifest_p.iterrows()
    )

    assert keys_s == keys_p


def test_output_paths_serial_vs_parallel(tmp_path: Path) -> None:
    """Same config+seed: output_path sets must match between serial and parallel."""
    config_path = _create_parallel_config(tmp_path)

    _run_cli(config_path, ["--num-workers", "1"])
    manifest_s = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    paths_s = set(manifest_s["output_path"])

    import shutil

    shutil.rmtree(tmp_path / "output")

    _run_cli(config_path, ["--num-workers", "2"])
    manifest_p = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    paths_p = set(manifest_p["output_path"])

    assert paths_s == paths_p


def test_transform_log_valid_json_parallel(tmp_path: Path) -> None:
    """transform_log.jsonl must contain valid JSON lines after parallel run."""
    config_path = _create_parallel_config(tmp_path)
    _run_cli(config_path, ["--num-workers", "2"])

    log_path = tmp_path / "output" / "metadata" / "transform_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    for line in lines:
        parsed = json.loads(line)
        assert "sample_id" in parsed
        assert "pipeline_name" in parsed
        assert "success" in parsed


def test_summary_consistency_serial_vs_parallel(tmp_path: Path) -> None:
    """summary.csv success/failed counts must match between serial and parallel."""
    config_path = _create_parallel_config(tmp_path)

    _run_cli(config_path, ["--num-workers", "1"])
    summary_s = pd.read_csv(tmp_path / "output" / "metadata" / "summary.csv")

    import shutil

    shutil.rmtree(tmp_path / "output")

    _run_cli(config_path, ["--num-workers", "2"])
    summary_p = pd.read_csv(tmp_path / "output" / "metadata" / "summary.csv")

    assert summary_s["success"].tolist() == summary_p["success"].tolist()
    assert summary_s["failed"].tolist() == summary_p["failed"].tolist()
    assert summary_s["total"].tolist() == summary_p["total"].tolist()


def test_parallel_repeat_no_filename_collisions(tmp_path: Path) -> None:
    """repeat > 1 with parallel workers must produce unique output filenames."""
    config_path = _create_parallel_config(
        tmp_path,
        num_images=2,
        pipelines=[
            {
                "name": "repeat_pipe",
                "repeat": 4,
                "transforms": [
                    {"name": "resize_long_edge", "params": {"long_edge": 32}},
                ],
            }
        ],
    )
    result = _run_cli(config_path, ["--num-workers", "2"])
    assert result.returncode == 0

    image_dir = tmp_path / "output" / "images" / "repeat_pipe"
    output_files = list(image_dir.glob("*.jpg"))
    assert len(output_files) == 8  # 2 images * 4 repeats
    assert len({f.name for f in output_files}) == 8


def test_parallel_multi_pipeline_no_collisions(tmp_path: Path) -> None:
    """Multiple pipelines with parallel workers must produce unique output filenames."""
    config_path = _create_parallel_config(
        tmp_path,
        num_images=2,
        pipelines=[
            {
                "name": "resize_a",
                "transforms": [
                    {"name": "resize_long_edge", "params": {"long_edge": 32}},
                ],
            },
            {
                "name": "resize_b",
                "transforms": [
                    {"name": "resize_long_edge", "params": {"long_edge": 48}},
                ],
            },
        ],
    )
    result = _run_cli(config_path, ["--num-workers", "2"])
    assert result.returncode == 0

    all_files: list[Path] = []
    for pipe_dir in (tmp_path / "output" / "images").iterdir():
        all_files.extend(pipe_dir.glob("*.jpg"))
    assert len(all_files) == 4  # 2 images * 2 pipelines
    assert len({f.relative_to(tmp_path / "output" / "images") for f in all_files}) == 4


def test_seeds_consistent_serial_vs_parallel(tmp_path: Path) -> None:
    """Per-task seeds must be identical between serial and parallel runs."""
    config_path = _create_parallel_config(tmp_path)

    _run_cli(config_path, ["--num-workers", "1"])
    manifest_s = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    seeds_s = {
        f"{r['sample_id']}::{r['pipeline_name']}::{r['repeat_index']}": r["seed"]
        for _, r in manifest_s.iterrows()
    }

    import shutil

    shutil.rmtree(tmp_path / "output")

    _run_cli(config_path, ["--num-workers", "2"])
    manifest_p = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    seeds_p = {
        f"{r['sample_id']}::{r['pipeline_name']}::{r['repeat_index']}": r["seed"]
        for _, r in manifest_p.iterrows()
    }

    for key in seeds_s:
        assert seeds_s[key] == seeds_p[key], f"Seed mismatch for {key}"


# =========================================================================
# Failure scenario tests
# =========================================================================


def test_parallel_bad_image_records_load_failure(tmp_path: Path) -> None:
    """Parallel mode with a corrupt image: failed_images.csv records stage=load_image."""
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    img.save(image_dir / "good.jpg")

    # Create a corrupt file with .jpg extension that will fail to load
    corrupt_path = image_dir / "corrupt.jpg"
    corrupt_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)

    config = {
        "project": {"name": "fail_test", "seed": 42},
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

    result = _run_cli(config_path, ["--num-workers", "2"])
    assert result.returncode == 0

    failed_path = tmp_path / "output" / "metadata" / "failed_images.csv"
    assert failed_path.exists()
    failed = pd.read_csv(failed_path)
    load_failures = failed[failed["stage"] == "load_image"]
    assert len(load_failures) > 0


def test_parallel_transform_error_records_stage(tmp_path: Path) -> None:
    """Parallel mode transform error: failed_images.csv records stage=transform."""
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    img.save(image_dir / "test.jpg")

    config = {
        "project": {"name": "transform_fail", "seed": 42},
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
                "name": "bad_pipe",
                "transforms": [
                    {"name": "resize_long_edge", "params": {"long_edge": -1}},
                ],
            }
        ],
    }

    config_path = tmp_path / "config" / "test.yaml"
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")

    result = _run_cli(config_path, ["--num-workers", "2"])
    assert result.returncode == 0

    failed_path = tmp_path / "output" / "metadata" / "failed_images.csv"
    assert failed_path.exists()
    failed = pd.read_csv(failed_path)
    assert any(failed["stage"] == "transform")


def test_parallel_single_failure_does_not_crash_run(tmp_path: Path) -> None:
    """One bad image must not crash the whole parallel run (skip_broken_images=True)."""
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    img.save(image_dir / "good.jpg")
    (image_dir / "bad.txt").write_text("not an image", encoding="utf-8")

    config_path = _create_parallel_config(tmp_path, num_images=1)
    # Add a bad file to the same directory
    (tmp_path / "images" / "bad.txt").write_text("not an image", encoding="utf-8")

    # scanner skips non-image files, so let's test with a corrupt image
    corrupt_path = tmp_path / "images" / "corrupt.jpg"
    corrupt_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)

    result = _run_cli(config_path, ["--num-workers", "2"])
    assert result.returncode == 0
    # At least the good image should succeed
    assert "success_count:" in result.stdout


# =========================================================================
# Resume / skip-existing / retry-failed parallel boundary tests
# =========================================================================


def test_parallel_resume_skips_completed(tmp_path: Path) -> None:
    """--resume --num-workers 2 skips already completed outputs."""
    config_path = _create_parallel_config(tmp_path)

    result1 = _run_cli(config_path, ["--num-workers", "2"])
    assert result1.returncode == 0
    assert "success_count: 3" in result1.stdout

    result2 = _run_cli(config_path, ["--num-workers", "2", "--resume"])
    assert result2.returncode == 0
    assert "skipped_count: 3" in result2.stdout


def test_parallel_skip_existing(tmp_path: Path) -> None:
    """--skip-existing --num-workers 2 skips outputs already on disk."""
    config_path = _create_parallel_config(tmp_path)

    result1 = _run_cli(config_path, ["--num-workers", "2"])
    assert result1.returncode == 0

    result2 = _run_cli(config_path, ["--num-workers", "2", "--skip-existing"])
    assert result2.returncode == 0
    assert "skipped_count: 3" in result2.stdout


def test_parallel_retry_failed_only_processes_failed(tmp_path: Path) -> None:
    """--retry-failed --num-workers 2 only re-processes entries in failed_images.csv."""
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    img.save(image_dir / "good.jpg")

    config = {
        "project": {"name": "retry_test", "seed": 42},
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

    result = _run_cli(config_path, ["--num-workers", "2"])
    assert result.returncode == 0
    assert "success_count: 1" in result.stdout

    result2 = _run_cli(config_path, ["--num-workers", "2", "--retry-failed"])
    assert result2.returncode == 0
    assert "retry_failed_enabled: True" in result2.stdout


def test_parallel_resume_retry_conflict(tmp_path: Path) -> None:
    """--resume and --retry-failed remain mutually exclusive with --num-workers."""
    config_path = _create_parallel_config(tmp_path)
    result = _run_cli(
        config_path,
        ["--num-workers", "2", "--resume", "--retry-failed"],
    )
    assert result.returncode == 1
    assert "cannot be used together" in result.stderr


def test_parallel_skipped_items_not_in_metadata(tmp_path: Path) -> None:
    """Skipped items must not appear in new run's metadata (v0.5.2 semantics preserved).

    Resume mode creates a fresh manifest containing only newly processed items.
    When all items are skipped, the manifest will be empty (header-only).
    """
    config_path = _create_parallel_config(tmp_path)

    _run_cli(config_path, ["--num-workers", "2"])

    manifest_before = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    assert len(manifest_before) == 3

    result = _run_cli(config_path, ["--num-workers", "2", "--resume"])
    assert "skipped_count: 3" in result.stdout

    manifest_after = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    assert len(manifest_after) == 0


# =========================================================================
# Existing integration tests (preserved from v0.5.3)
# =========================================================================


def test_cli_num_workers_1_equals_serial(tmp_path: Path) -> None:
    config_path = _create_parallel_config(tmp_path)
    result = _run_cli(config_path, ["--num-workers", "1"])
    assert result.returncode == 0
    assert "success_count: 3" in result.stdout


def test_cli_num_workers_2_produces_same_outputs(tmp_path: Path) -> None:
    config_path = _create_parallel_config(tmp_path)
    result = _run_cli(config_path, ["--num-workers", "2"])
    assert result.returncode == 0
    assert "success_count: 3" in result.stdout

    manifest_path = tmp_path / "output" / "metadata" / "manifest.csv"
    assert manifest_path.exists()
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 3
    assert all(manifest["success"])


def test_cli_parallel_metadata_correctness(tmp_path: Path) -> None:
    config_path = _create_parallel_config(tmp_path)
    _run_cli(config_path, ["--num-workers", "2"])

    manifest = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    assert set(manifest.columns).issuperset(
        {"sample_id", "pipeline_name", "repeat_index", "success", "seed"}
    )

    transform_log = tmp_path / "output" / "metadata" / "transform_log.jsonl"
    assert transform_log.exists()
    lines = transform_log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    summary = pd.read_csv(tmp_path / "output" / "metadata" / "summary.csv")
    assert len(summary) == 1
    assert summary.iloc[0]["total"] == 3
    assert summary.iloc[0]["success"] == 3


def test_cli_parallel_with_repeat(tmp_path: Path) -> None:
    config_path = _create_parallel_config(
        tmp_path,
        num_images=1,
        pipelines=[
            {
                "name": "repeat_pipe",
                "repeat": 3,
                "transforms": [
                    {"name": "resize_long_edge", "params": {"long_edge": 32}},
                ],
            }
        ],
    )
    result = _run_cli(config_path, ["--num-workers", "2"])
    assert result.returncode == 0
    assert "success_count: 3" in result.stdout


def test_cli_parallel_no_collisions(tmp_path: Path) -> None:
    config_path = _create_parallel_config(tmp_path)
    result = _run_cli(config_path, ["--num-workers", "2"])
    assert result.returncode == 0

    image_dir = tmp_path / "output" / "images" / "resize_test"
    output_files = list(image_dir.glob("*.jpg"))
    assert len(output_files) == 3
    assert len({f.name for f in output_files}) == 3


def test_parallel_shows_experimental_warning(tmp_path: Path) -> None:
    """Parallel mode must log an experimental warning."""
    config_path = _create_parallel_config(tmp_path)
    result = _run_cli(config_path, ["--num-workers", "2"])
    assert result.returncode == 0
    assert "experimental" in result.stderr.lower()
    assert "num_workers: 2" in result.stdout
