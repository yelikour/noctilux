from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from noctilux.cli import main
from noctilux.metadata import MetadataRecorder
from noctilux.report import generate_report


def _write_metadata(tmp_path: Path) -> Path:
    recorder = MetadataRecorder(tmp_path)
    recorder.add_manifest_record(
        {
            "sample_id": "s1",
            "original_path": "a.jpg",
            "output_path": "out/a.jpg",
            "pipeline_name": "resize",
            "repeat_index": 0,
            "input_width": 100,
            "input_height": 80,
            "output_width": 64,
            "output_height": 51,
            "input_format": "JPEG",
            "output_format": "JPG",
            "success": True,
            "error": "",
            "seed": 42,
            "label": "",
            "split": "unknown",
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
            "input_width": 100,
            "input_height": 80,
            "output_width": None,
            "output_height": None,
            "input_format": "JPEG",
            "output_format": None,
            "success": False,
            "error": "broken image",
            "seed": 43,
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
            "label": "",
            "split": "unknown",
            "task": "generic",
            "transforms": [
                {"name": "resize_long_edge", "applied": True, "params": {"long_edge": 64}},
                {"name": "gaussian_blur", "applied": False, "params": {"radius": 1.0}},
            ],
            "input_info": {"width": 100, "height": 80},
            "output_info": {"width": 64, "height": 51},
            "success": True,
            "error": None,
        }
    )
    recorder.add_failed_image(
        sample_id="s2",
        image_path="b.jpg",
        pipeline_name="resize",
        repeat_index=0,
        seed=43,
        stage="load_image",
        error="broken image",
    )
    recorder.write_all()
    return tmp_path / "metadata"


def test_report_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["report", "--help"])

    captured = capsys.readouterr()
    assert "--metadata" in captured.out
    assert "--output" in captured.out
    assert "--csv-output" in captured.out


def test_generate_report_writes_markdown_and_csv(tmp_path: Path) -> None:
    metadata_dir = _write_metadata(tmp_path)
    output_path = tmp_path / "report.md"
    csv_path = tmp_path / "summary_report.csv"

    result = generate_report(metadata_dir, output_path, csv_output_path=csv_path)

    assert result == output_path
    text = output_path.read_text(encoding="utf-8")
    assert "Total records" in text
    assert "Success count" in text
    assert "Failed count" in text
    assert "Pipelines summary" in text
    assert "resize_long_edge" in text
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0
    frame = pd.read_csv(csv_path)
    assert "summary" in set(frame["section"])


def test_report_cli_missing_metadata_returns_clear_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exit_code = main(
        [
            "report",
            "--metadata",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "report.md"),
        ]
    )

    assert exit_code == 1
    assert "Metadata directory does not exist" in caplog.text


def test_report_refuses_to_overwrite_without_flag(tmp_path: Path) -> None:
    metadata_dir = _write_metadata(tmp_path)
    output_path = tmp_path / "report.md"
    output_path.write_text("keep", encoding="utf-8")

    exit_code = main(["report", "--metadata", str(metadata_dir), "--output", str(output_path)])

    assert exit_code == 1
    assert output_path.read_text(encoding="utf-8") == "keep"


def test_report_overwrite_allows_existing_output(tmp_path: Path) -> None:
    metadata_dir = _write_metadata(tmp_path)
    output_path = tmp_path / "report.md"
    output_path.write_text("old", encoding="utf-8")

    exit_code = main(
        [
            "report",
            "--metadata",
            str(metadata_dir),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert "Total records" in output_path.read_text(encoding="utf-8")


def test_report_notes_missing_optional_metadata_file(tmp_path: Path) -> None:
    metadata_dir = _write_metadata(tmp_path)
    (metadata_dir / "failed_images.csv").unlink()
    output_path = tmp_path / "report.md"

    generate_report(metadata_dir, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "Missing metadata files" in text
    assert "failed_images.csv" in text
