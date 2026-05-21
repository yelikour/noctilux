from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from noctilux.metadata import MetadataRecorder


def test_metadata_files_are_written(tmp_path: Path) -> None:
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
