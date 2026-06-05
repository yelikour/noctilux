from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from noctilux.metadata import MetadataRecorder, MetadataWriter


def test_metadata_recorder_files_are_written(tmp_path: Path) -> None:
    recorder = MetadataRecorder(tmp_path)
    recorder.add_manifest_record(
        {
            "sample_id": "s1",
            "original_path": "a.jpg",
            "output_path": "out/a.jpg",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "input_width": 10,
            "input_height": 10,
            "output_width": 8,
            "output_height": 8,
            "input_format": "JPEG",
            "output_format": "JPG",
            "success": True,
            "error": "",
            "seed": 42,
            "label": "cat",
            "split": "train",
            "task": "generic",
        }
    )
    recorder.add_manifest_record(
        {
            "sample_id": "s2",
            "original_path": "b.jpg",
            "output_path": "",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "input_width": 10,
            "input_height": 10,
            "output_width": None,
            "output_height": None,
            "input_format": "JPEG",
            "output_format": None,
            "success": False,
            "error": "broken",
            "seed": 42,
            "label": "",
            "split": "unknown",
            "task": "generic",
        }
    )
    recorder.add_transform_log(
        {
            "sample_id": "s1",
            "original_path": "a.jpg",
            "output_path": "out/a.jpg",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "seed": 42,
            "label": "cat",
            "split": "train",
            "task": "generic",
            "transforms": [{"name": "resize_long_edge", "applied": True, "params": {"long_edge": 8}}],
            "input_info": {"width": 10, "height": 10},
            "output_info": {"width": 8, "height": 8},
            "success": True,
            "error": None,
        }
    )
    recorder.add_failed_image(
        sample_id="s2",
        image_path="b.jpg",
        pipeline_name="resize",
        repeat_index=0,
        seed=42,
        stage="transform",
        error="broken",
    )
    recorder.write_all()

    metadata_dir = tmp_path / "metadata"
    manifest = pd.read_csv(metadata_dir / "manifest.csv")
    failed = pd.read_csv(metadata_dir / "failed_images.csv")
    summary = pd.read_csv(metadata_dir / "summary.csv")

    assert len(manifest) == 2
    assert len(failed) == 1
    assert summary.loc[0, "total"] == 2
    assert summary.loc[0, "failed"] == 1
    assert failed.loc[0, "pipeline_name"] == "resize"
    assert failed.loc[0, "repeat_index"] == 0
    assert failed.loc[0, "seed"] == 42
    assert failed.loc[0, "stage"] == "transform"

    lines = (metadata_dir / "transform_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[0])
    assert payload["sample_id"] == "s1"
    assert payload["label"] == "cat"
    assert payload["transforms"][0]["name"] == "resize_long_edge"


def test_writer_writes_success(tmp_path: Path) -> None:
    writer = MetadataWriter(tmp_path)
    writer.write_success(
        manifest_row={
            "sample_id": "s1",
            "original_path": "a.jpg",
            "output_path": "out/a.jpg",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "input_width": 10,
            "input_height": 10,
            "output_width": 8,
            "output_height": 8,
            "input_format": "JPEG",
            "output_format": "JPG",
            "success": True,
            "error": "",
            "seed": 42,
            "label": "cat",
            "split": "train",
            "task": "generic",
        },
        transform_log_row={
            "sample_id": "s1",
            "original_path": "a.jpg",
            "output_path": "out/a.jpg",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "seed": 42,
            "label": "cat",
            "split": "train",
            "task": "generic",
            "transforms": [{"name": "resize_long_edge", "applied": True, "params": {"long_edge": 8}}],
            "input_info": {"width": 10, "height": 10},
            "output_info": {"width": 8, "height": 8},
            "success": True,
            "error": None,
        },
    )
    writer.close()

    assert writer.success_count == 1
    assert writer.failed_count == 0
    assert writer.total_count == 1

    manifest = pd.read_csv(tmp_path / "manifest.csv")
    assert len(manifest) == 1
    assert manifest.loc[0, "sample_id"] == "s1"
    assert manifest.loc[0, "success"] == True  # noqa: E712

    lines = (tmp_path / "transform_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["transforms"][0]["name"] == "resize_long_edge"


def test_writer_writes_failure(tmp_path: Path) -> None:
    writer = MetadataWriter(tmp_path)
    writer.write_failure(
        manifest_row={
            "sample_id": "s2",
            "original_path": "b.jpg",
            "output_path": "",
            "pipeline_name": "blur",
            "repeat_index": 0,
            "input_width": 10,
            "input_height": 10,
            "output_width": None,
            "output_height": None,
            "input_format": "JPEG",
            "output_format": None,
            "success": False,
            "error": "broken",
            "seed": 7,
            "label": "",
            "split": "unknown",
            "task": "generic",
        },
        transform_log_row={
            "sample_id": "s2",
            "original_path": "b.jpg",
            "output_path": "",
            "pipeline_name": "blur",
            "repeat_index": 0,
            "seed": 7,
            "label": "",
            "split": "unknown",
            "task": "generic",
            "transforms": [],
            "input_info": {},
            "output_info": {},
            "success": False,
            "error": "broken",
        },
        failed_row={
            "sample_id": "s2",
            "image_path": "b.jpg",
            "pipeline_name": "blur",
            "repeat_index": 0,
            "seed": 7,
            "stage": "transform",
            "error": "broken",
        },
    )
    writer.close()

    assert writer.success_count == 0
    assert writer.failed_count == 1

    failed = pd.read_csv(tmp_path / "failed_images.csv")
    assert len(failed) == 1
    assert failed.loc[0, "stage"] == "transform"
    assert failed.loc[0, "error"] == "broken"


def test_writer_writes_summary(tmp_path: Path) -> None:
    writer = MetadataWriter(tmp_path)
    for i in range(3):
        writer.write_success(
            manifest_row={
                "sample_id": f"s{i}",
                "original_path": f"{i}.jpg",
                "output_path": f"out/{i}.jpg",
                "pipeline_name": "resize",
                "repeat_index": 0,
                "input_width": 10,
                "input_height": 10,
                "output_width": 8,
                "output_height": 8,
                "input_format": "JPEG",
                "output_format": "JPG",
                "success": True,
                "error": "",
                "seed": i,
                "label": "",
                "split": "train",
                "task": "generic",
            },
            transform_log_row={
                "sample_id": f"s{i}",
                "original_path": f"{i}.jpg",
                "output_path": f"out/{i}.jpg",
                "pipeline_name": "resize",
                "repeat_index": 0,
                "seed": i,
                "label": "",
                "split": "train",
                "task": "generic",
                "transforms": [],
                "input_info": {},
                "output_info": {},
                "success": True,
                "error": None,
            },
        )
    writer.write_failure(
        manifest_row={
            "sample_id": "s_fail",
            "original_path": "bad.jpg",
            "output_path": "",
            "pipeline_name": "resize",
            "repeat_index": 1,
            "input_width": 10,
            "input_height": 10,
            "output_width": None,
            "output_height": None,
            "input_format": "JPEG",
            "output_format": None,
            "success": False,
            "error": "oops",
            "seed": 99,
            "label": "",
            "split": "unknown",
            "task": "generic",
        },
        transform_log_row={
            "sample_id": "s_fail",
            "original_path": "bad.jpg",
            "output_path": "",
            "pipeline_name": "resize",
            "repeat_index": 1,
            "seed": 99,
            "label": "",
            "split": "unknown",
            "task": "generic",
            "transforms": [],
            "input_info": {},
            "output_info": {},
            "success": False,
            "error": "oops",
        },
        failed_row={
            "sample_id": "s_fail",
            "image_path": "bad.jpg",
            "pipeline_name": "resize",
            "repeat_index": 1,
            "seed": 99,
            "stage": "transform",
            "error": "oops",
        },
    )
    writer.close()

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert len(summary) == 1
    assert summary.loc[0, "pipeline_name"] == "resize"
    assert summary.loc[0, "total"] == 4
    assert summary.loc[0, "success"] == 3
    assert summary.loc[0, "failed"] == 1


def test_writer_multi_pipeline_summary(tmp_path: Path) -> None:
    writer = MetadataWriter(tmp_path)
    for pipe in ("resize", "blur"):
        writer.write_success(
            manifest_row={
                "sample_id": "s1",
                "original_path": "a.jpg",
                "output_path": f"out/{pipe}.jpg",
                "pipeline_name": pipe,
                "repeat_index": 0,
                "input_width": 10,
                "input_height": 10,
                "output_width": 8,
                "output_height": 8,
                "input_format": "JPEG",
                "output_format": "JPG",
                "success": True,
                "error": "",
                "seed": 1,
                "label": "",
                "split": "train",
                "task": "generic",
            },
            transform_log_row={
                "sample_id": "s1",
                "original_path": "a.jpg",
                "output_path": f"out/{pipe}.jpg",
                "pipeline_name": pipe,
                "repeat_index": 0,
                "seed": 1,
                "label": "",
                "split": "train",
                "task": "generic",
                "transforms": [],
                "input_info": {},
                "output_info": {},
                "success": True,
                "error": None,
            },
        )
    writer.close()

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert len(summary) == 2
    assert set(summary["pipeline_name"]) == {"resize", "blur"}


def test_writer_manifest_columns_compat(tmp_path: Path) -> None:
    from noctilux.metadata import MANIFEST_COLUMNS

    writer = MetadataWriter(tmp_path)
    writer.write_success(
        manifest_row={
            "sample_id": "s1",
            "original_path": "a.jpg",
            "output_path": "out/a.jpg",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "input_width": 10,
            "input_height": 10,
            "output_width": 8,
            "output_height": 8,
            "input_format": "JPEG",
            "output_format": "JPG",
            "success": True,
            "error": "",
            "seed": 42,
            "label": "cat",
            "split": "train",
            "task": "generic",
        },
        transform_log_row={
            "sample_id": "s1",
            "original_path": "a.jpg",
            "output_path": "out/a.jpg",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "seed": 42,
            "label": "cat",
            "split": "train",
            "task": "generic",
            "transforms": [],
            "input_info": {},
            "output_info": {},
            "success": True,
            "error": None,
        },
    )
    writer.close()

    manifest = pd.read_csv(tmp_path / "manifest.csv")
    assert list(manifest.columns) == MANIFEST_COLUMNS


def test_writer_failed_columns_compat(tmp_path: Path) -> None:
    from noctilux.metadata import FAILED_COLUMNS

    writer = MetadataWriter(tmp_path)
    writer.write_failure(
        manifest_row={
            "sample_id": "s2",
            "original_path": "b.jpg",
            "output_path": "",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "input_width": 10,
            "input_height": 10,
            "output_width": None,
            "output_height": None,
            "input_format": "JPEG",
            "output_format": None,
            "success": False,
            "error": "err",
            "seed": 1,
            "label": "",
            "split": "unknown",
            "task": "generic",
        },
        transform_log_row={
            "sample_id": "s2",
            "original_path": "b.jpg",
            "output_path": "",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "seed": 1,
            "label": "",
            "split": "unknown",
            "task": "generic",
            "transforms": [],
            "input_info": {},
            "output_info": {},
            "success": False,
            "error": "err",
        },
        failed_row={
            "sample_id": "s2",
            "image_path": "b.jpg",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "seed": 1,
            "stage": "load_image",
            "error": "err",
        },
    )
    writer.close()

    failed = pd.read_csv(tmp_path / "failed_images.csv")
    assert list(failed.columns) == FAILED_COLUMNS
