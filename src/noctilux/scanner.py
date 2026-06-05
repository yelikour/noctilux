from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def scan_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    input_cfg = config["input"]
    mode = input_cfg["mode"]
    if mode == "folder":
        return scan_folder(
            image_root=input_cfg["image_root"],
            recursive=bool(input_cfg.get("recursive", True)),
            infer_label_from_subdir=bool(input_cfg.get("infer_label_from_subdir", False)),
            extensions=input_cfg.get("extensions", list(SUPPORTED_EXTENSIONS)),
        )
    if mode == "manifest":
        return scan_manifest(
            manifest_path=input_cfg["manifest_path"],
            image_root=input_cfg.get("image_root"),
            path_column=input_cfg.get("path_column", "image_path"),
            label_column=input_cfg.get("label_column", "label"),
            split_column=input_cfg.get("split_column", "split"),
            task_column=input_cfg.get("task_column", "task"),
            sample_id_column=input_cfg.get("sample_id_column", "sample_id"),
            extensions=input_cfg.get("extensions", list(SUPPORTED_EXTENSIONS)),
        )
    raise ValueError(f"Unsupported input mode: {mode}")


def scan_folder(
    image_root: str | Path,
    recursive: bool = True,
    infer_label_from_subdir: bool = False,
    extensions: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Input image_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"input.image_root must be a directory: {root}")

    allowed_extensions = _normalize_extensions(extensions)
    pattern = "**/*" if recursive else "*"
    samples: list[dict[str, Any]] = []

    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in allowed_extensions:
            continue
        relative_path = path.relative_to(root)
        label = relative_path.parts[0] if infer_label_from_subdir and len(relative_path.parts) > 1 else ""
        samples.append(
            {
                "sample_id": _make_sample_id(relative_path),
                "image_path": path,
                "label": label,
                "split": "unknown",
                "task": "generic",
                "metadata": {
                    "relative_path": relative_path.as_posix(),
                    "relative_dir": relative_path.parent.as_posix() if relative_path.parent != Path(".") else "",
                },
            }
        )
    _validate_unique_sample_ids(samples)
    return samples


def scan_manifest(
    manifest_path: str | Path,
    image_root: str | Path | None = None,
    path_column: str = "image_path",
    label_column: str = "label",
    split_column: str = "split",
    task_column: str = "task",
    sample_id_column: str = "sample_id",
    extensions: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Input manifest does not exist: {manifest}")

    frame = pd.read_csv(manifest)
    if path_column not in frame.columns:
        raise ValueError(f"Manifest is missing required column: {path_column}")

    root = Path(image_root) if image_root is not None else None
    allowed_extensions = _normalize_extensions(extensions)
    samples: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        raw_path = row[path_column]
        if pd.isna(raw_path):
            continue
        image_path = Path(str(raw_path))
        if not image_path.is_absolute() and root is not None:
            image_path = root / image_path
        if image_path.suffix.lower() not in allowed_extensions:
            continue

        relative_path = _relative_path_for_sample(image_path, root)
        metadata = _extract_metadata(row.to_dict())
        metadata["relative_path"] = relative_path.as_posix()
        metadata["relative_dir"] = relative_path.parent.as_posix() if relative_path.parent != Path(".") else ""

        sample_id = row.get(sample_id_column)
        if pd.isna(sample_id) or not sample_id:
            sample_id = _make_sample_id(relative_path, index=index)

        samples.append(
            {
                "sample_id": str(sample_id),
                "image_path": image_path,
                "label": _optional_string(row.get(label_column)),
                "split": _optional_string(row.get(split_column), default="unknown"),
                "task": _optional_string(row.get(task_column), default="generic"),
                "metadata": metadata,
            }
        )
    _validate_unique_sample_ids(samples)
    return samples


def build_manifest_from_folder(
    image_root: str | Path,
    infer_label_from_subdir: bool = False,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
) -> pd.DataFrame:
    samples = scan_folder(
        image_root=image_root,
        recursive=recursive,
        infer_label_from_subdir=infer_label_from_subdir,
        extensions=extensions,
    )
    rows = []
    root = Path(image_root)
    for sample in samples:
        path = Path(sample["image_path"])
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "image_path": path.relative_to(root).as_posix(),
                "label": sample["label"],
                "split": sample["split"],
                "task": sample["task"],
            }
        )
    return pd.DataFrame(rows)


def _validate_unique_sample_ids(samples: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        if sample_id in seen:
            duplicates.add(sample_id)
        seen.add(sample_id)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate sample_id values found; sample_id must be unique: {duplicate_list}")


def _make_sample_id(relative_path: Path, index: int | None = None) -> str:
    digest = hashlib.md5(relative_path.as_posix().encode("utf-8")).hexdigest()[:8]
    if index is None:
        return f"{relative_path.stem}-{digest}"
    return f"{index:06d}-{relative_path.stem}-{digest}"


def _normalize_extensions(extensions: Iterable[str] | None) -> set[str]:
    values = set()
    for extension in extensions or SUPPORTED_EXTENSIONS:
        normalized = extension.lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        values.add(normalized)
    return values


def _relative_path_for_sample(image_path: Path, image_root: Path | None) -> Path:
    if image_root is not None:
        try:
            return image_path.relative_to(image_root)
        except ValueError:
            return Path(image_path.name)
    return Path(image_path.name)


def _optional_string(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def _extract_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    raw_metadata = row.get("metadata")
    if isinstance(raw_metadata, str) and raw_metadata.strip():
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                metadata.update(parsed)
        except json.JSONDecodeError:
            metadata["metadata_raw"] = raw_metadata

    for key, value in row.items():
        if key in {"sample_id", "image_path", "label", "split", "task", "metadata"}:
            continue
        if value is None or pd.isna(value):
            continue
        metadata[key] = value
    return metadata
