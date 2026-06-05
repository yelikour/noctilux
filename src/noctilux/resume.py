from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def build_processing_key(sample_id: str, pipeline_name: str, repeat_index: int) -> str:
    return f"{sample_id}::{pipeline_name}::{repeat_index}"


def parse_processing_key(key: str) -> tuple[str, str, int]:
    parts = key.rsplit("::", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid processing key: {key!r}")
    return parts[0], parts[1], int(parts[2])


def load_success_keys(metadata_dir: Path) -> set[str]:
    manifest_path = metadata_dir / "manifest.csv"
    if not manifest_path.exists():
        return set()
    try:
        frame = pd.read_csv(manifest_path)
    except Exception as exc:
        raise ValueError(f"Cannot read manifest.csv in {metadata_dir}: {exc}") from exc
    if "success" not in frame.columns:
        return set()

    success_mask = frame["success"]
    if success_mask.dtype != bool:
        success_mask = success_mask.astype(str).str.lower().isin({"true", "1", "yes"})

    keys: set[str] = set()
    for _, row in frame[success_mask].iterrows():
        key = build_processing_key(
            sample_id=str(row.get("sample_id", "")),
            pipeline_name=str(row.get("pipeline_name", "")),
            repeat_index=int(row.get("repeat_index", 0)),
        )
        keys.add(key)
    return keys


def load_failed_keys(metadata_dir: Path) -> set[str]:
    failed_path = metadata_dir / "failed_images.csv"
    if not failed_path.exists():
        return set()
    try:
        frame = pd.read_csv(failed_path)
    except Exception as exc:
        raise ValueError(f"Cannot read failed_images.csv in {metadata_dir}: {exc}") from exc

    keys: set[str] = set()
    for _, row in frame.iterrows():
        key = build_processing_key(
            sample_id=str(row.get("sample_id", "")),
            pipeline_name=str(row.get("pipeline_name", "")),
            repeat_index=int(row.get("repeat_index", 0)),
        )
        keys.add(key)
    return keys


def check_output_exists(sample: dict[str, Any], pipeline_name: str, repeat_index: int, saver: Any) -> bool:
    try:
        from noctilux.image_io.writer import normalize_extension

        sample_path = Path(sample["image_path"])
        extension = normalize_extension(saver.output_config["save_format"])
        filename = f"{sample_path.stem}__{pipeline_name}__{repeat_index:03d}.{extension}"
        relative_dir = Path()
        if saver.output_config.get("preserve_subdirs", True):
            relative_dir = saver._get_relative_dir(sample)
        target = saver.images_root / pipeline_name / relative_dir / filename
        return target.exists()
    except Exception:
        return False


def validate_resume_args(resume: bool, retry_failed: bool) -> None:
    if resume and retry_failed:
        raise ValueError("--resume and --retry-failed cannot be used together. Choose one.")
