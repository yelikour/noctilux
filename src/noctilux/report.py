from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_METADATA_FILES = ("manifest.csv", "summary.csv")
OPTIONAL_METADATA_FILES = ("failed_images.csv", "transform_log.jsonl")


def generate_report(
    metadata_dir: Path,
    output_path: Path,
    csv_output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    metadata_root = Path(metadata_dir)
    markdown_path = Path(output_path)
    csv_path = Path(csv_output_path) if csv_output_path is not None else None

    if not metadata_root.exists() or not metadata_root.is_dir():
        raise FileNotFoundError(f"Metadata directory does not exist: {metadata_root}")
    _ensure_writable(markdown_path, overwrite=overwrite)
    if csv_path is not None:
        _ensure_writable(csv_path, overwrite=overwrite)

    missing_required = [name for name in REQUIRED_METADATA_FILES if not (metadata_root / name).exists()]
    if missing_required:
        missing = ", ".join(missing_required)
        raise FileNotFoundError(f"Required metadata file missing in {metadata_root}: {missing}")

    missing_optional = [name for name in OPTIONAL_METADATA_FILES if not (metadata_root / name).exists()]
    manifest = pd.read_csv(metadata_root / "manifest.csv")
    summary = pd.read_csv(metadata_root / "summary.csv")
    failed = _read_optional_csv(metadata_root / "failed_images.csv")
    transform_logs = _read_transform_logs(metadata_root / "transform_log.jsonl")

    success_series = _success_series(manifest)
    total_records = int(len(manifest))
    success_count = int(success_series.sum())
    failed_count = int(total_records - success_count)
    success_rate = success_count / total_records if total_records else 0.0
    pipeline_count = _pipeline_count(manifest, transform_logs)

    pipeline_rows = _pipeline_rows(summary)
    format_rows = _value_count_rows(manifest, "output_format")
    size_rows = _image_size_rows(manifest)
    failed_stage_rows = _value_count_rows(failed, "stage") if failed is not None else []
    error_rows = _top_error_rows(manifest, failed)
    transform_rows = _transform_rows(transform_logs)

    markdown = _build_markdown(
        metadata_root=metadata_root,
        total_records=total_records,
        success_count=success_count,
        failed_count=failed_count,
        success_rate=success_rate,
        pipeline_count=pipeline_count,
        missing_optional=missing_optional,
        pipeline_rows=pipeline_rows,
        format_rows=format_rows,
        size_rows=size_rows,
        failed_stage_rows=failed_stage_rows,
        error_rows=error_rows,
        transform_rows=transform_rows,
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv_summary(
            csv_path,
            total_records=total_records,
            success_count=success_count,
            failed_count=failed_count,
            success_rate=success_rate,
            pipeline_count=pipeline_count,
            pipeline_rows=pipeline_rows,
            format_rows=format_rows,
            failed_stage_rows=failed_stage_rows,
            transform_rows=transform_rows,
        )

    return markdown_path


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report file: {path}")


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_transform_logs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _success_series(frame: pd.DataFrame) -> pd.Series:
    if "success" not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index)
    series = frame["success"]
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _pipeline_count(manifest: pd.DataFrame, transform_logs: list[dict[str, Any]]) -> int:
    names = {str(row.get("pipeline_name")) for row in transform_logs if row.get("pipeline_name")}
    if names:
        return len(names)
    if "pipeline_name" not in manifest.columns:
        return 0
    return int(manifest["pipeline_name"].dropna().nunique())


def _pipeline_rows(summary: pd.DataFrame) -> list[list[str]]:
    if summary.empty:
        return []
    rows: list[list[str]] = []
    for _, row in summary.iterrows():
        rows.append(
            [
                _cell(row.get("pipeline_name", "")),
                _int_cell(row.get("total")),
                _int_cell(row.get("success")),
                _int_cell(row.get("failed")),
            ]
        )
    return rows


def _value_count_rows(frame: pd.DataFrame | None, column: str) -> list[list[str]]:
    if frame is None or frame.empty or column not in frame.columns:
        return []
    series = frame[column].dropna().astype(str)
    series = series[series.str.len() > 0]
    return [[str(name), str(count)] for name, count in series.value_counts().items()]


def _image_size_rows(manifest: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for column in ("input_width", "input_height", "output_width", "output_height"):
        if column not in manifest.columns:
            continue
        values = pd.to_numeric(manifest[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            [
                column,
                _number_cell(values.min()),
                _number_cell(values.median()),
                _number_cell(values.max()),
            ]
        )
    return rows


def _top_error_rows(manifest: pd.DataFrame, failed: pd.DataFrame | None, limit: int = 5) -> list[list[str]]:
    errors: list[str] = []
    for frame in (manifest, failed):
        if frame is None or "error" not in frame.columns:
            continue
        for value in frame["error"].dropna().astype(str):
            if value and value.lower() != "nan":
                errors.append(value)
    return [[message, str(count)] for message, count in Counter(errors).most_common(limit)]


def _transform_rows(transform_logs: list[dict[str, Any]]) -> list[list[str]]:
    counts: dict[str, dict[str, int]] = {}
    for row in transform_logs:
        transforms = row.get("transforms") or []
        if not isinstance(transforms, list):
            continue
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            name = str(transform.get("name") or "unknown")
            bucket = counts.setdefault(name, {"total": 0, "applied_true": 0, "applied_false": 0})
            bucket["total"] += 1
            if bool(transform.get("applied")):
                bucket["applied_true"] += 1
            else:
                bucket["applied_false"] += 1
    return [
        [name, str(values["total"]), str(values["applied_true"]), str(values["applied_false"])]
        for name, values in sorted(counts.items())
    ]


def _build_markdown(
    metadata_root: Path,
    total_records: int,
    success_count: int,
    failed_count: int,
    success_rate: float,
    pipeline_count: int,
    missing_optional: list[str],
    pipeline_rows: list[list[str]],
    format_rows: list[list[str]],
    size_rows: list[list[str]],
    failed_stage_rows: list[list[str]],
    error_rows: list[list[str]],
    transform_rows: list[list[str]],
) -> str:
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# Noctilux Metadata Report",
        "",
        f"- Generated time: {generated}",
        f"- Metadata directory: `{metadata_root}`",
        "",
        "## Summary",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Total records", str(total_records)],
                ["Success count", str(success_count)],
                ["Failed count", str(failed_count)],
                ["Success rate", f"{success_rate:.2%}"],
                ["Pipeline count", str(pipeline_count)],
            ],
        ),
        "",
        "## Pipelines summary",
        "",
        _markdown_table(["Pipeline", "Total", "Success", "Failed"], pipeline_rows),
        "",
        "## Output formats summary",
        "",
        _markdown_table(["Output format", "Count"], format_rows),
        "",
        "## Image size summary",
        "",
        _markdown_table(["Field", "Min", "Median", "Max"], size_rows),
        "",
        "## Failed stages summary",
        "",
        _markdown_table(["Stage", "Count"], failed_stage_rows),
        "",
        "## Top error messages",
        "",
        _markdown_table(["Error", "Count"], error_rows),
        "",
        "## Transform usage summary",
        "",
        _markdown_table(["Transform", "Total", "Applied true", "Applied false"], transform_rows),
        "",
        "## Notes / limitations",
        "",
        "- This report reads existing metadata only and does not reprocess images.",
        "- The report is a lightweight Markdown summary, not a visualization dashboard.",
    ]
    if missing_optional:
        lines.append(f"- Missing metadata files: {', '.join(missing_optional)}")
    else:
        lines.append("- All expected metadata files were present.")
    lines.append("")
    return "\n".join(lines)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data._"
    escaped_headers = [_escape_markdown_cell(header) for header in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in padded[: len(headers)]) + " |")
    return "\n".join(lines)


def _write_csv_summary(
    path: Path,
    total_records: int,
    success_count: int,
    failed_count: int,
    success_rate: float,
    pipeline_count: int,
    pipeline_rows: list[list[str]],
    format_rows: list[list[str]],
    failed_stage_rows: list[list[str]],
    transform_rows: list[list[str]],
) -> None:
    rows: list[dict[str, str]] = [
        {"section": "summary", "metric": "total_records", "value": str(total_records)},
        {"section": "summary", "metric": "success_count", "value": str(success_count)},
        {"section": "summary", "metric": "failed_count", "value": str(failed_count)},
        {"section": "summary", "metric": "success_rate", "value": f"{success_rate:.6f}"},
        {"section": "summary", "metric": "pipeline_count", "value": str(pipeline_count)},
    ]
    for pipeline, total, success, failed in pipeline_rows:
        rows.append(
            {
                "section": "pipeline",
                "metric": pipeline,
                "value": f"total={total};success={success};failed={failed}",
            }
        )
    for output_format, count in format_rows:
        rows.append({"section": "output_format", "metric": output_format, "value": count})
    for stage, count in failed_stage_rows:
        rows.append({"section": "failed_stage", "metric": stage, "value": count})
    for transform, total, applied_true, applied_false in transform_rows:
        rows.append(
            {
                "section": "transform",
                "metric": transform,
                "value": f"total={total};applied_true={applied_true};applied_false={applied_false}",
            }
        )
    pd.DataFrame(rows, columns=["section", "metric", "value"]).to_csv(path, index=False)


def _escape_markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _int_cell(value: Any) -> str:
    if pd.isna(value):
        return "0"
    return str(int(value))


def _number_cell(value: Any) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.2f}"
