from __future__ import annotations

import csv
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


class MetadataWriter:
    """Streaming metadata writer for batch processing runs.

    Writes manifest.csv, transform_log.jsonl, and failed_images.csv
    incrementally as results arrive. Summary.csv is written on close().

    Designed for single-process use in serial execution (v0.5.x) and
    as the centralized writer for future parallel execution.
    """

    def __init__(self, metadata_root: Path, append: bool = False) -> None:
        self.metadata_root = Path(metadata_root)
        self.metadata_root.mkdir(parents=True, exist_ok=True)

        self._success_count = 0
        self._failed_count = 0
        self._pipeline_counts: dict[str, dict[str, int]] = {}
        self._closed = False

        self._manifest_path = self.metadata_root / "manifest.csv"
        self._log_path = self.metadata_root / "transform_log.jsonl"
        self._failed_path = self.metadata_root / "failed_images.csv"

        manifest_mode = "a" if append else "w"
        log_mode = "a" if append else "w"
        failed_mode = "a" if append else "w"

        self._manifest_file = self._manifest_path.open(manifest_mode, encoding="utf-8", newline="")
        self._manifest_writer = csv.DictWriter(self._manifest_file, fieldnames=MANIFEST_COLUMNS)
        if not append or self._manifest_path.stat().st_size == 0:
            self._manifest_writer.writeheader()

        self._log_file = self._log_path.open(log_mode, encoding="utf-8")

        self._failed_file = self._failed_path.open(failed_mode, encoding="utf-8", newline="")
        self._failed_writer = csv.DictWriter(self._failed_file, fieldnames=FAILED_COLUMNS)
        if not append or self._failed_path.stat().st_size == 0:
            self._failed_writer.writeheader()

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def failed_count(self) -> int:
        return self._failed_count

    @property
    def total_count(self) -> int:
        return self._success_count + self._failed_count

    def write_success(self, manifest_row: dict[str, Any], transform_log_row: dict[str, Any]) -> None:
        self._write_manifest(manifest_row)
        self._write_transform_log(transform_log_row)
        self._increment_pipeline(manifest_row.get("pipeline_name", ""), success=True)
        self._success_count += 1

    def write_failure(
        self,
        manifest_row: dict[str, Any],
        transform_log_row: dict[str, Any],
        failed_row: dict[str, Any],
    ) -> None:
        self._write_manifest(manifest_row)
        self._write_transform_log(transform_log_row)
        self._write_failed(failed_row)
        self._increment_pipeline(manifest_row.get("pipeline_name", ""), success=False)
        self._failed_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._write_summary()
        self._manifest_file.close()
        self._log_file.close()
        self._failed_file.close()
        self._closed = True

    def _write_manifest(self, row: dict[str, Any]) -> None:
        self._manifest_writer.writerow({col: row.get(col) for col in MANIFEST_COLUMNS})

    def _write_transform_log(self, row: dict[str, Any]) -> None:
        self._log_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_failed(self, row: dict[str, Any]) -> None:
        self._failed_writer.writerow({col: row.get(col) for col in FAILED_COLUMNS})

    def _increment_pipeline(self, pipeline_name: str, success: bool) -> None:
        counts = self._pipeline_counts.setdefault(pipeline_name, {"total": 0, "success": 0, "failed": 0})
        counts["total"] += 1
        if success:
            counts["success"] += 1
        else:
            counts["failed"] += 1

    def _write_summary(self) -> None:
        self._manifest_file.flush()
        try:
            frame = pd.read_csv(self._manifest_path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame(columns=MANIFEST_COLUMNS)

        if not frame.empty and {"pipeline_name", "success"}.issubset(frame.columns):
            success_values = frame["success"]
            if success_values.dtype != bool:
                success_values = success_values.astype(str).str.lower().isin({"true", "1", "yes"})
            summary = (
                frame.assign(_success=success_values)
                .groupby("pipeline_name", dropna=False)["_success"]
                .agg(total="size", success="sum")
                .reset_index()
            )
            summary["success"] = summary["success"].astype(int)
            summary["failed"] = summary["total"] - summary["success"]
            summary = summary[["pipeline_name", "total", "success", "failed"]]
        else:
            summary = pd.DataFrame(columns=["pipeline_name", "total", "success", "failed"])
        summary.to_csv(self.metadata_root / "summary.csv", index=False)


class MetadataRecorder:
    """Legacy metadata recorder that accumulates records in memory and writes at the end.

    Retained for backward compatibility. New code should use MetadataWriter.
    """

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
