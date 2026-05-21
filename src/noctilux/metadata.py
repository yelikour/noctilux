from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

MANIFEST_COLUMNS = [
    "sample_id",
    "original_path",
    "output_path",
    "pipeline_name",
    "repeat_index",
    "input_width",
    "input_height",
    "output_width",
    "output_height",
    "input_format",
    "output_format",
    "success",
    "error",
    "seed",
    "label",
    "split",
    "task",
]

FAILED_COLUMNS = [
    "sample_id",
    "image_path",
    "pipeline_name",
    "repeat_index",
    "seed",
    "stage",
    "error",
]
TRANSFORM_LOG_COLUMNS = [
    "sample_id",
    "original_path",
    "output_path",
    "pipeline_name",
    "repeat_index",
    "seed",
    "label",
    "split",
    "task",
    "transforms",
    "input_info",
    "output_info",
    "success",
    "error",
]


class MetadataRecorder:
    def __init__(self, output_root: str | Path, metadata_dir: str = "metadata") -> None:
        self.output_root = Path(output_root)
        self.metadata_root = self.output_root / metadata_dir
        self.manifest_rows: list[dict[str, Any]] = []
        self.transform_logs: list[dict[str, Any]] = []
        self.failed_rows: list[dict[str, Any]] = []

    def add_manifest_record(self, row: dict[str, Any]) -> None:
        record = {column: row.get(column) for column in MANIFEST_COLUMNS}
        self.manifest_rows.append(record)

    def add_transform_log(self, row: dict[str, Any]) -> None:
        record = {column: row.get(column) for column in TRANSFORM_LOG_COLUMNS}
        self.transform_logs.append(record)

    def add_failed_image(
        self,
        sample_id: str,
        image_path: str,
        error: str,
        pipeline_name: str = "",
        repeat_index: int | None = None,
        seed: int | None = None,
        stage: str = "",
    ) -> None:
        self.failed_rows.append(
            {
                "sample_id": sample_id,
                "image_path": image_path,
                "pipeline_name": pipeline_name,
                "repeat_index": repeat_index,
                "seed": seed,
                "stage": stage,
                "error": error,
            }
        )

    def write_all(self) -> None:
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        self._write_manifest()
        self._write_transform_logs()
        self._write_failed_images()
        self._write_summary()

    def _write_manifest(self) -> None:
        frame = pd.DataFrame(self.manifest_rows, columns=MANIFEST_COLUMNS)
        frame.to_csv(self.metadata_root / "manifest.csv", index=False)

    def _write_transform_logs(self) -> None:
        output_path = self.metadata_root / "transform_log.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in self.transform_logs:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_failed_images(self) -> None:
        frame = pd.DataFrame(self.failed_rows, columns=FAILED_COLUMNS)
        frame.to_csv(self.metadata_root / "failed_images.csv", index=False)

    def _write_summary(self) -> None:
        if self.manifest_rows:
            frame = pd.DataFrame(self.manifest_rows)
            summary = (
                frame.groupby("pipeline_name", dropna=False)["success"]
                .agg(total="size", success="sum")
                .reset_index()
            )
            summary["failed"] = summary["total"] - summary["success"]
        else:
            summary = pd.DataFrame(columns=["pipeline_name", "total", "success", "failed"])
        summary.to_csv(self.metadata_root / "summary.csv", index=False)
