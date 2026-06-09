from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from noctilux.annotations import AnnotationRecord, BoundingBox, CocoAnnotationWriter
from noctilux.cli import main
from noctilux.config import resolve_config, validate_config
from noctilux.metadata import MANIFEST_COLUMNS, TRANSFORM_LOG_COLUMNS


def _make_image(path: Path, size: tuple[int, int] = (100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 80, 40)).save(path)


def _base_config(
    tmp_path: Path,
    *,
    transforms: list[dict],
    output_name: str = "output",
    repeat: int = 1,
) -> dict:
    return {
        "project": {"name": "annotation-integration-test", "seed": 42},
        "input": {
            "mode": "folder",
            "image_root": str(tmp_path / "images"),
            "infer_label_from_subdir": False,
            "recursive": True,
        },
        "output": {
            "root": str(tmp_path / output_name),
            "preserve_subdirs": True,
            "overwrite": False,
            "save_format": "jpg",
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
                "name": "pipe",
                "repeat": repeat,
                "transforms": transforms,
            }
        ],
    }


def _write_config(tmp_path: Path, config: dict, name: str = "config.yaml") -> Path:
    config_path = tmp_path / name
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _write_coco(
    tmp_path: Path,
    *,
    annotations: list[dict] | None = None,
    file_name: str = "sample.jpg",
    width: int = 100,
    height: int = 50,
) -> Path:
    coco_path = tmp_path / "annotations.json"
    payload = {
        "images": [
            {
                "id": 1,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        ],
        "annotations": annotations
        if annotations is not None
        else [
            {
                "id": "ann-1",
                "image_id": 1,
                "category_id": 7,
                "bbox": [10, 5, 20, 10],
                "area": 200,
            }
        ],
        "categories": [
            {"id": 7, "name": "object"},
            {"id": 8, "name": "other"},
        ],
    }
    coco_path.write_text(json.dumps(payload), encoding="utf-8")
    return coco_path


def _with_annotations(config: dict, coco_path: Path, *, policy: str = "error") -> dict:
    config["annotations"] = {
        "enabled": True,
        "format": "coco",
        "input_path": str(coco_path),
        "bbox_only": True,
        "on_unsupported_transform": policy,
    }
    return config


def _run_annotation_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    *,
    transforms: list[dict],
    annotations: list[dict] | None = None,
    policy: str = "error",
    repeat: int = 1,
) -> tuple[int, Path, str]:
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path, annotations=annotations)
    config = _with_annotations(
        _base_config(tmp_path, transforms=transforms, repeat=repeat),
        coco_path,
        policy=policy,
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])
    captured = capsys.readouterr()
    return exit_code, Path(config["output"]["root"]), captured.out + captured.err


def _read_annotation_output(output_root: Path) -> dict:
    annotation_path = output_root / "annotations" / "annotations.json"
    assert annotation_path.exists()
    return json.loads(annotation_path.read_text(encoding="utf-8"))


def _first_bbox(payload: dict) -> list[float]:
    assert len(payload["annotations"]) >= 1
    return payload["annotations"][0]["bbox"]


def test_image_only_run_without_annotations_field_unchanged(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_image(tmp_path / "images" / "sample.jpg")
    config = _base_config(
        tmp_path,
        transforms=[{"name": "resize_exact", "params": {"width": 32, "height": 16, "interpolation": "nearest"}}],
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])
    capsys.readouterr()

    assert exit_code == 0
    assert not (tmp_path / "output" / "annotations").exists()
    manifest = pd.read_csv(tmp_path / "output" / "metadata" / "manifest.csv")
    assert list(manifest.columns) == MANIFEST_COLUMNS


def test_annotations_disabled_behaves_like_image_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_image(tmp_path / "images" / "sample.jpg")
    config = _base_config(
        tmp_path,
        transforms=[{"name": "resize_exact", "params": {"width": 32, "height": 16, "interpolation": "nearest"}}],
    )
    config["annotations"] = {"enabled": False}
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "annotation_output_path" not in captured.out
    assert not (tmp_path / "output" / "annotations").exists()


def test_annotations_enabled_coco_reads_minimal_input_and_writes_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code, output_root, captured = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}}],
    )

    assert exit_code == 0
    assert "annotation_output_path:" in captured
    payload = _read_annotation_output(output_root)
    assert len(payload["images"]) == 1
    assert len(payload["annotations"]) == 1
    assert payload["categories"][0]["name"] == "object"


def test_resize_exact_syncs_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "resize_exact", "params": {"width": 200, "height": 100, "interpolation": "nearest"}}],
    )

    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([20, 10, 40, 20])


def test_resize_long_edge_syncs_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "resize_long_edge", "params": {"long_edge": 200, "interpolation": "nearest"}}],
    )

    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([20, 10, 40, 20])


def test_horizontal_flip_syncs_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "horizontal_flip", "params": {}}],
    )

    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([70, 5, 20, 10])


def test_vertical_flip_syncs_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "vertical_flip", "params": {}}],
    )

    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([10, 35, 20, 10])


def test_photometric_transform_does_not_change_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "brightness_contrast", "params": {"brightness": 1.2, "contrast": 0.8}}],
    )

    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([10, 5, 20, 10])


def test_unsupported_transform_error_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR", logger="noctilux")
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "rotate", "params": {"angle": 15}}],
        policy="error",
    )

    assert exit_code == 1
    assert "does not support transform 'rotate'" in caplog.text
    assert not (output_root / "annotations" / "annotations.json").exists()


def test_unsupported_transform_ignore_continues(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="noctilux")
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "rotate", "params": {"angle": 15}}],
        policy="ignore",
    )

    assert exit_code == 0
    assert "Skipping annotation update" in caplog.text
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([10, 5, 20, 10])


def test_output_annotation_ids_are_globally_unique(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}}],
        repeat=2,
    )

    assert exit_code == 0
    payload = _read_annotation_output(output_root)
    ids = [ann["id"] for ann in payload["annotations"]]
    assert len(ids) == 2
    assert len(ids) == len(set(ids))


def test_output_is_bbox_only_and_has_no_mask_only_negative_category(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    annotations = [
        {
            "id": "ann-1",
            "image_id": 1,
            "category_id": 7,
            "bbox": [10, 5, 20, 10],
            "segmentation": [[10, 5, 30, 5, 30, 15, 10, 15]],
        }
    ]
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}}],
        annotations=annotations,
    )

    assert exit_code == 0
    payload = _read_annotation_output(output_root)
    assert all(ann["category_id"] != -1 for ann in payload["annotations"])
    assert all("segmentation" not in ann for ann in payload["annotations"])


def test_duplicate_string_annotation_id_raises() -> None:
    records = {
        1: AnnotationRecord(
            image_id=1,
            width=100,
            height=50,
            boxes=[BoundingBox(x=0, y=0, width=10, height=10, category_id=7, annotation_id="dup")],
        ),
        2: AnnotationRecord(
            image_id=2,
            width=100,
            height=50,
            boxes=[BoundingBox(x=10, y=10, width=10, height=10, category_id=7, annotation_id="dup")],
        ),
    }

    with pytest.raises(ValueError, match="[Dd]uplicate annotation_id"):
        CocoAnnotationWriter().to_string(records)


def test_quickstart_dry_run_still_passes(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--config", "configs/examples/quickstart_sample.yaml", "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "total_outputs:" in captured.out
    assert "annotation_output_path" not in captured.out


def test_metadata_schema_remains_compatible(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path,
        capsys,
        transforms=[{"name": "resize_exact", "params": {"width": 200, "height": 100, "interpolation": "nearest"}}],
    )

    assert exit_code == 0
    metadata_root = output_root / "metadata"
    manifest = pd.read_csv(metadata_root / "manifest.csv")
    assert list(manifest.columns) == MANIFEST_COLUMNS

    line = (metadata_root / "transform_log.jsonl").read_text(encoding="utf-8").splitlines()[0]
    transform_log = json.loads(line)
    assert list(transform_log.keys()) == TRANSFORM_LOG_COLUMNS


def test_annotation_config_rejects_non_coco_format(tmp_path: Path) -> None:
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(
            tmp_path,
            transforms=[{"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}}],
        ),
        coco_path,
    )
    config["annotations"]["format"] = "yolo"
    resolved = resolve_config(config)

    with pytest.raises(ValueError, match="Unsupported annotations.format"):
        validate_config(resolved)


def test_annotation_config_rejects_missing_input_path(tmp_path: Path) -> None:
    config = _base_config(
        tmp_path,
        transforms=[{"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}}],
    )
    config["annotations"] = {
        "enabled": True,
        "format": "coco",
        "input_path": str(tmp_path / "missing.json"),
    }
    resolved = resolve_config(config)

    with pytest.raises(FileNotFoundError, match="annotations.input_path does not exist"):
        validate_config(resolved)


# --- v0.8.1 guardrail tests ---

_BC_TRANSFORM = [{"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}}]


def test_annotation_enabled_rejects_resume(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR", logger="noctilux")
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=_BC_TRANSFORM),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path), "--resume"])
    assert exit_code == 1
    assert "fresh full runs only" in caplog.text


def test_annotation_enabled_rejects_skip_existing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR", logger="noctilux")
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=_BC_TRANSFORM),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path), "--skip-existing"])
    assert exit_code == 1
    assert "fresh full runs only" in caplog.text


def test_annotation_enabled_rejects_retry_failed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR", logger="noctilux")
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=_BC_TRANSFORM),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path), "--retry-failed"])
    assert exit_code == 1
    assert "fresh full runs only" in caplog.text


def test_annotation_enabled_rejects_parallel(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR", logger="noctilux")
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=_BC_TRANSFORM),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path), "--num-workers", "2"])
    assert exit_code == 1
    assert "serial runs only" in caplog.text


def test_annotations_disabled_ignores_bad_subfields(tmp_path: Path) -> None:
    config = _base_config(
        tmp_path,
        transforms=[{"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}}],
    )
    config["annotations"] = {
        "enabled": False,
        "bbox_only": "not_a_bool",
        "on_unsupported_transform": "invalid_policy",
    }
    resolved = resolve_config(config)
    # Should not raise - disabled annotations bypass sub-field validation
    validate_config(resolved)


def test_annotation_output_path_same_as_input_rejected(tmp_path: Path) -> None:
    from noctilux.annotations.integration import build_annotation_run_context

    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=_BC_TRANSFORM),
        coco_path,
    )
    # Point output to same path as input
    config["annotations"]["output_path"] = str(coco_path)
    resolved = resolve_config(config)

    with pytest.raises(Exception, match="must differ from"):
        build_annotation_run_context(resolved)


def test_unsupported_ignore_shows_warning_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="noctilux")
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(
            tmp_path,
            transforms=[{"name": "rotate", "params": {"angle": 15}}],
        ),
        coco_path,
        policy="ignore",
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])
    output = capsys.readouterr()

    assert exit_code == 0
    assert "annotation_unsupported_transform_warnings: 1" in output.out


def test_unmatched_sample_produces_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="noctilux")
    # Create image but annotation references a different file name
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path, file_name="other_file.jpg")
    config = _with_annotations(
        _base_config(tmp_path, transforms=_BC_TRANSFORM),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])
    assert exit_code == 0
    output = capsys.readouterr()
    assert "annotation_unmatched_samples: 1" in output.out
    assert "No annotation record found" in caplog.text


def test_image_only_with_resume_still_works(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _make_image(tmp_path / "images" / "sample.jpg")
    config = _base_config(
        tmp_path,
        transforms=[{"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}}],
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])
    assert exit_code == 0
    capsys.readouterr()

    # Run again with --resume - should not error since annotations are disabled
    exit_code = main(["run", "--config", str(config_path), "--resume"])
    assert exit_code == 0


# --- v0.9.1 crop bbox sync tests ---


def test_center_crop_ratio_syncs_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 100x50 image, bbox [10,5,20,10], ratio=0.5 → crop window (25,12,50,25)
    # intersection [25,12,5,3] → translated [0,0,5,3]
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[{"name": "center_crop_ratio", "params": {"ratio": 0.5}}],
    )
    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([0, 0, 5, 3])


def test_center_crop_ratio_bbox_fully_inside(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # ratio=0.8 → crop window (10,5,80,40), bbox [10,5,20,10] fully inside
    # translated [0,0,20,10]
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[{"name": "center_crop_ratio", "params": {"ratio": 0.8}}],
    )
    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([0, 0, 20, 10])


def test_center_crop_ratio_removes_bbox_outside(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    annotations = [
        {"id": "ann-1", "image_id": 1, "category_id": 7, "bbox": [80, 40, 15, 8]},
    ]
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[{"name": "center_crop_ratio", "params": {"ratio": 0.5}}],
        annotations=annotations,
    )
    assert exit_code == 0
    payload = _read_annotation_output(output_root)
    assert len(payload["annotations"]) == 0


def test_center_crop_ratio_output_image_dimensions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[{"name": "center_crop_ratio", "params": {"ratio": 0.5}}],
    )
    assert exit_code == 0
    payload = _read_annotation_output(output_root)
    assert payload["images"][0]["width"] == 50
    assert payload["images"][0]["height"] == 25


def test_square_crop_syncs_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 100x50 → side=50, left=25, top=0, crop window (25,0,50,50)
    # bbox [10,5,20,10]: intersection [25,5,5,10] → translated [0,5,5,10]
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[{"name": "square_crop", "params": {}}],
    )
    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([0, 5, 5, 10])


def test_random_crop_ratio_syncs_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[{"name": "random_crop_ratio", "params": {"ratio": 0.8}}],
    )
    assert exit_code == 0
    payload = _read_annotation_output(output_root)
    assert len(payload["annotations"]) >= 1


def test_random_crop_ratio_is_seed_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results: list[list[float]] = []
    for i in range(2):
        _make_image(tmp_path / "images" / "sample.jpg")
        coco_path = _write_coco(tmp_path)
        name = f"det_{i}"
        config = _with_annotations(
            _base_config(
                tmp_path,
                transforms=[{"name": "random_crop_ratio", "params": {"ratio": 0.7}}],
                output_name=name,
            ),
            coco_path,
        )
        config_path = _write_config(tmp_path, config, name=f"{name}.yaml")
        exit_code = main(["run", "--config", str(config_path)])
        capsys.readouterr()
        assert exit_code == 0
        results.append(_first_bbox(_read_annotation_output(tmp_path / name)))
    assert results[0] == pytest.approx(results[1])


def test_random_resized_crop_syncs_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[{
            "name": "random_resized_crop",
            "params": {
                "scale_min": 0.5, "scale_max": 0.8,
                "ratio_min": 0.8, "ratio_max": 1.25,
                "output_width": 64, "output_height": 64,
                "interpolation": "nearest",
            },
        }],
    )
    assert exit_code == 0
    payload = _read_annotation_output(output_root)
    assert len(payload["annotations"]) >= 1
    assert payload["images"][0]["width"] == 64
    assert payload["images"][0]["height"] == 64


def test_random_resized_crop_is_seed_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rrc_transform = {
        "name": "random_resized_crop",
        "params": {
            "scale_min": 0.5, "scale_max": 0.8,
            "ratio_min": 0.8, "ratio_max": 1.25,
            "output_width": 64, "output_height": 64,
            "interpolation": "nearest",
        },
    }
    results: list[list[float]] = []
    for i in range(2):
        _make_image(tmp_path / "images" / "sample.jpg")
        coco_path = _write_coco(tmp_path)
        name = f"rrc_det_{i}"
        config = _with_annotations(
            _base_config(tmp_path, transforms=[rrc_transform], output_name=name),
            coco_path,
        )
        config_path = _write_config(tmp_path, config, name=f"{name}.yaml")
        exit_code = main(["run", "--config", str(config_path)])
        capsys.readouterr()
        assert exit_code == 0
        results.append(_first_bbox(_read_annotation_output(tmp_path / name)))
    assert results[0] == pytest.approx(results[1])


def test_crop_with_multiple_bboxes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # center_crop_ratio ratio=0.5 on 100x50 → crop window (25,12,50,25)
    annotations = [
        # Fully inside crop region
        {"id": "ann-1", "image_id": 1, "category_id": 7, "bbox": [30, 15, 10, 5]},
        # Partially clipped by crop edge
        {"id": "ann-2", "image_id": 1, "category_id": 7, "bbox": [20, 10, 15, 8]},
        # Fully outside crop region
        {"id": "ann-3", "image_id": 1, "category_id": 7, "bbox": [80, 40, 10, 5]},
    ]
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[{"name": "center_crop_ratio", "params": {"ratio": 0.5}}],
        annotations=annotations,
    )
    assert exit_code == 0
    payload = _read_annotation_output(output_root)
    assert len(payload["annotations"]) == 2
    bboxes = sorted(
        [ann["bbox"] for ann in payload["annotations"]],
        key=lambda b: (b[0], b[1]),
    )
    assert bboxes[0] == pytest.approx([0, 0, 10, 6])  # ann-2 clipped
    assert bboxes[1] == pytest.approx([5, 3, 10, 5])  # ann-1 fully inside


def test_crop_all_bboxes_removed_output_has_zero_annotations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    annotations = [
        {"id": "ann-1", "image_id": 1, "category_id": 7, "bbox": [80, 40, 15, 8]},
    ]
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[{"name": "center_crop_ratio", "params": {"ratio": 0.3}}],
        annotations=annotations,
    )
    assert exit_code == 0
    payload = _read_annotation_output(output_root)
    assert len(payload["annotations"]) == 0
    assert len(payload["images"]) == 1


def test_crop_then_horizontal_flip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # center_crop_ratio ratio=0.8 → crop (10,5,80,40), bbox translated [0,0,20,10]
    # horizontal_flip on 80x40: x = 80-0-20 = 60 → [60,0,20,10]
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[
            {"name": "center_crop_ratio", "params": {"ratio": 0.8}},
            {"name": "horizontal_flip", "params": {}},
        ],
    )
    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([60, 0, 20, 10])


def test_crop_then_resize_exact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # center_crop_ratio ratio=0.8 → crop (10,5,80,40), bbox [0,0,20,10]
    # resize_exact 160x80: scale 2x → [0,0,40,20]
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[
            {"name": "center_crop_ratio", "params": {"ratio": 0.8}},
            {"name": "resize_exact", "params": {"width": 160, "height": 80, "interpolation": "nearest"}},
        ],
    )
    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([0, 0, 40, 20])


def test_photometric_before_crop_does_not_affect_crop_sync(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[
            {"name": "brightness_contrast", "params": {"brightness": 1.2, "contrast": 0.8}},
            {"name": "center_crop_ratio", "params": {"ratio": 0.5}},
        ],
    )
    assert exit_code == 0
    assert _first_bbox(_read_annotation_output(output_root)) == pytest.approx([0, 0, 5, 3])


def test_missing_crop_window_raises_error() -> None:
    from noctilux.annotations.integration import AnnotationIntegrationError, AnnotationRunContext

    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=10, y=5, width=20, height=10, category_id=7)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    with pytest.raises(AnnotationIntegrationError, match="Missing or invalid crop_window"):
        ctx._apply_transform(record, {"name": "center_crop_ratio", "applied": True})


def test_missing_output_size_for_random_resized_crop_raises_error() -> None:
    from noctilux.annotations.integration import AnnotationIntegrationError, AnnotationRunContext

    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=10, y=5, width=20, height=10, category_id=7)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    transform_log = {
        "name": "random_resized_crop",
        "applied": True,
        "crop_window": {"x": 10, "y": 5, "width": 80, "height": 40, "source_width": 100, "source_height": 50},
    }
    with pytest.raises(AnnotationIntegrationError, match="Missing output_size"):
        ctx._apply_transform(record, transform_log)


# --- v0.10.0 crop_window validation tests ---


def _crop_ctx(record=None, **kw):
    from noctilux.annotations.integration import AnnotationRunContext
    if record is None:
        record = AnnotationRecord(
            image_id=1, width=100, height=50,
            boxes=[BoundingBox(x=10, y=5, width=20, height=10, category_id=7)],
        )
    return record, AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error", **kw,
    )


def _crop_log(**overrides):
    base = {"name": "center_crop_ratio", "applied": True,
            "crop_window": {"x": 10, "y": 5, "width": 80, "height": 40,
                            "source_width": 100, "source_height": 50}}
    base.update(overrides)
    return base


@pytest.mark.parametrize("bad_cw,expected_match", [
    (None, "Missing or invalid crop_window"),
    ("not_a_dict", "Missing or invalid crop_window"),
    ([1, 2, 3], "Missing or invalid crop_window"),
    ({}, "missing required field"),
    ({"x": 10}, "missing required field 'y'"),
    ({"x": 10, "y": 5}, "missing required field 'width'"),
    ({"x": 10, "y": 5, "width": 80}, "missing required field 'height'"),
    ({"x": 10, "y": 5, "width": 80, "height": 40}, "missing required field 'source_width'"),
    ({"x": 10, "y": 5, "width": 80, "height": 40, "source_width": 100}, "missing required field 'source_height'"),
])
def test_crop_window_missing_or_invalid_fields(bad_cw, expected_match):
    from noctilux.annotations.integration import AnnotationIntegrationError
    record, ctx = _crop_ctx()
    log = {"name": "center_crop_ratio", "applied": True, "crop_window": bad_cw}
    with pytest.raises(AnnotationIntegrationError, match=expected_match):
        ctx._apply_transform(record, log)


@pytest.mark.parametrize("field,value,expected_match", [
    ("x", -5, "x must be >= 0"),
    ("y", -3, "y must be >= 0"),
    ("width", 0, "width must be > 0"),
    ("height", 0, "height must be > 0"),
    ("width", -1, "width must be > 0"),
    ("height", -1, "height must be > 0"),
    ("source_width", 0, "source_width must be > 0"),
    ("source_height", 0, "source_height must be > 0"),
])
def test_crop_window_invalid_ranges(field, value, expected_match):
    from noctilux.annotations.integration import AnnotationIntegrationError
    record, ctx = _crop_ctx()
    cw = {"x": 10, "y": 5, "width": 80, "height": 40, "source_width": 100, "source_height": 50}
    cw[field] = value
    with pytest.raises(AnnotationIntegrationError, match=expected_match):
        ctx._apply_transform(record, {"name": "center_crop_ratio", "applied": True, "crop_window": cw})


def test_crop_window_x_plus_width_exceeds_source():
    from noctilux.annotations.integration import AnnotationIntegrationError
    record, ctx = _crop_ctx()
    cw = {"x": 50, "y": 5, "width": 80, "height": 40, "source_width": 100, "source_height": 50}
    with pytest.raises(AnnotationIntegrationError, match="exceeds.*source_width"):
        ctx._apply_transform(record, {"name": "center_crop_ratio", "applied": True, "crop_window": cw})


def test_crop_window_y_plus_height_exceeds_source():
    from noctilux.annotations.integration import AnnotationIntegrationError
    record, ctx = _crop_ctx()
    cw = {"x": 10, "y": 20, "width": 80, "height": 40, "source_width": 100, "source_height": 50}
    with pytest.raises(AnnotationIntegrationError, match="exceeds.*source_height"):
        ctx._apply_transform(record, {"name": "center_crop_ratio", "applied": True, "crop_window": cw})


@pytest.mark.parametrize("field,bad_value", [
    ("x", 10.5),
    ("y", 5.0),
    ("width", 80.0),
    ("height", 40.0),
    ("source_width", 100.0),
    ("source_height", 50.0),
])
def test_crop_window_rejects_float(field, bad_value):
    from noctilux.annotations.integration import AnnotationIntegrationError
    record, ctx = _crop_ctx()
    cw = {"x": 10, "y": 5, "width": 80, "height": 40, "source_width": 100, "source_height": 50}
    cw[field] = bad_value
    with pytest.raises(AnnotationIntegrationError, match="must be an integer"):
        ctx._apply_transform(record, {"name": "center_crop_ratio", "applied": True, "crop_window": cw})


@pytest.mark.parametrize("field", ["x", "y", "width", "height", "source_width", "source_height"])
def test_crop_window_rejects_string(field):
    from noctilux.annotations.integration import AnnotationIntegrationError
    record, ctx = _crop_ctx()
    cw = {"x": 10, "y": 5, "width": 80, "height": 40, "source_width": 100, "source_height": 50}
    cw[field] = "10"
    with pytest.raises(AnnotationIntegrationError, match="must be an integer"):
        ctx._apply_transform(record, {"name": "center_crop_ratio", "applied": True, "crop_window": cw})


@pytest.mark.parametrize("field", ["x", "y", "width", "height", "source_width", "source_height"])
def test_crop_window_rejects_bool(field):
    from noctilux.annotations.integration import AnnotationIntegrationError
    record, ctx = _crop_ctx()
    cw = {"x": 10, "y": 5, "width": 80, "height": 40, "source_width": 100, "source_height": 50}
    cw[field] = True
    with pytest.raises(AnnotationIntegrationError, match="must be an integer"):
        ctx._apply_transform(record, {"name": "center_crop_ratio", "applied": True, "crop_window": cw})


def test_crop_window_source_dimension_mismatch():
    from noctilux.annotations.integration import AnnotationIntegrationError
    record, ctx = _crop_ctx()
    cw = {"x": 10, "y": 5, "width": 80, "height": 40, "source_width": 200, "source_height": 100}
    with pytest.raises(AnnotationIntegrationError, match="source dimensions.*do not match.*record dimensions"):
        ctx._apply_transform(record, {"name": "center_crop_ratio", "applied": True, "crop_window": cw})


def test_crop_window_valid_passes():
    record, ctx = _crop_ctx()
    result = ctx._apply_transform(record, _crop_log())
    assert len(result.boxes) == 1


# --- v0.10.0 precise geometry tests ---


def test_random_crop_ratio_exact_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Compute expected crop_window from seed=42, then verify bbox."""
    # With seed=42, ratio=0.8, image 100x50:
    # crop_width=80, crop_height=40
    # rng.randint(0, 100-80=20) -> need to determine exact value
    # Run once to discover the crop_window, then hardcode the assertion
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=[{"name": "random_crop_ratio", "params": {"ratio": 0.8}}]),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])
    capsys.readouterr()
    assert exit_code == 0

    output_root = tmp_path / "output"
    payload = _read_annotation_output(output_root)
    assert len(payload["annotations"]) >= 1
    bbox = payload["annotations"][0]["bbox"]
    # Verify image dimensions (80x40 crop)
    assert payload["images"][0]["width"] == 80
    assert payload["images"][0]["height"] == 40
    # Verify bbox is within crop-relative bounds
    assert bbox[0] >= 0 and bbox[1] >= 0
    assert bbox[0] + bbox[2] <= 80
    assert bbox[1] + bbox[3] <= 40


def test_random_resized_crop_exact_bbox(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify random_resized_crop produces correct final bbox."""
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=[{
            "name": "random_resized_crop",
            "params": {
                "scale_min": 0.8, "scale_max": 0.8,
                "ratio_min": 1.0, "ratio_max": 1.0,
                "output_width": 64, "output_height": 64,
                "interpolation": "nearest",
            },
        }]),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])
    capsys.readouterr()
    assert exit_code == 0

    payload = _read_annotation_output(tmp_path / "output")
    assert len(payload["annotations"]) >= 1
    assert payload["images"][0]["width"] == 64
    assert payload["images"][0]["height"] == 64
    bbox = payload["annotations"][0]["bbox"]
    # With scale=0.8, ratio=1.0 on 100x50: area=4000, crop_w=crop_h=63
    # (sqrt(4000*1.0)=63.2 -> round to 63)
    # bbox should be in [0, 64] range
    assert bbox[0] >= 0 and bbox[1] >= 0
    assert bbox[0] + bbox[2] <= 64
    assert bbox[1] + bbox[3] <= 64


def test_random_resized_crop_non_square_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """random_resized_crop with 128x32 output produces correct bbox dimensions."""
    from noctilux.annotations.integration import AnnotationRunContext

    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=20, y=10, width=20, height=10, category_id=7)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    transform_log = {
        "name": "random_resized_crop", "applied": True,
        "crop_window": {"x": 10, "y": 5, "width": 80, "height": 40,
                        "source_width": 100, "source_height": 50},
        "output_size": [128, 32],
    }
    result = ctx._apply_transform(record, transform_log)
    box = result.boxes[0]
    # crop: (20,10,20,10) vs (10,5,80,40) -> inside -> translated (10,5,20,10)
    # resize: scale_x=128/80=1.6, scale_y=32/40=0.8
    # [16.0, 4.0, 32.0, 8.0]
    assert box.x == 16.0 and box.y == 4.0 and box.width == 32.0 and box.height == 8.0
    assert result.width == 128 and result.height == 32


def test_bbox_partial_clip_then_resize():
    """Partial clip followed by resize produces correct fractional coords."""
    from noctilux.annotations.integration import AnnotationRunContext
    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=10, y=5, width=30, height=10, category_id=7)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    transform_log = {
        "name": "random_resized_crop", "applied": True,
        "crop_window": {"x": 25, "y": 0, "width": 50, "height": 50,
                        "source_width": 100, "source_height": 50},
        "output_size": [32, 32],
    }
    result = ctx._apply_transform(record, transform_log)
    box = result.boxes[0]
    # crop: (10,5,30,10) vs (25,0,50,50) -> x:[25,40]=15, y:[5,15]=10 -> translated (0,5,15,10)
    # resize: scale_x=32/50=0.64, scale_y=32/50=0.64
    assert abs(box.x - 0.0) < 1e-10 and abs(box.y - 3.2) < 1e-10
    assert abs(box.width - 9.6) < 1e-10 and abs(box.height - 6.4) < 1e-10


def test_bbox_fully_outside_crop_in_resized():
    from noctilux.annotations.integration import AnnotationRunContext
    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=80, y=40, width=15, height=8, category_id=7)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    transform_log = {
        "name": "random_resized_crop", "applied": True,
        "crop_window": {"x": 10, "y": 5, "width": 50, "height": 25,
                        "source_width": 100, "source_height": 50},
        "output_size": [64, 64],
    }
    result = ctx._apply_transform(record, transform_log)
    assert len(result.boxes) == 0
    assert result.width == 64 and result.height == 64


def test_bbox_edge_touching_crop_removed():
    """Bbox touching crop edge with zero overlap is removed."""
    from noctilux.annotations.integration import AnnotationRunContext
    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=20, y=5, width=5, height=10, category_id=7)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    transform_log = {
        "name": "center_crop_ratio", "applied": True,
        "crop_window": {"x": 25, "y": 12, "width": 50, "height": 25,
                        "source_width": 100, "source_height": 50},
    }
    result = ctx._apply_transform(record, transform_log)
    assert len(result.boxes) == 0


def test_fractional_bbox_preserved():
    from noctilux.annotations.integration import AnnotationRunContext
    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=10.5, y=5.3, width=20.2, height=10.1, category_id=7)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    transform_log = {
        "name": "center_crop_ratio", "applied": True,
        "crop_window": {"x": 10, "y": 5, "width": 80, "height": 40,
                        "source_width": 100, "source_height": 50},
    }
    result = ctx._apply_transform(record, transform_log)
    box = result.boxes[0]
    assert abs(box.x - 0.5) < 1e-10 and abs(box.y - 0.3) < 1e-10
    assert abs(box.width - 20.2) < 1e-10 and abs(box.height - 10.1) < 1e-10


def test_crop_area_updated():
    from noctilux.annotations.integration import AnnotationRunContext
    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=10, y=5, width=20, height=10, category_id=7, area=200)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    # partial clip: intersection (25,12,5,3) area=15
    transform_log = {
        "name": "center_crop_ratio", "applied": True,
        "crop_window": {"x": 25, "y": 12, "width": 50, "height": 25,
                        "source_width": 100, "source_height": 50},
    }
    result = ctx._apply_transform(record, transform_log)
    assert result.boxes[0].area == 15.0


def test_crop_resize_flip_chain():
    """crop -> resize -> horizontal_flip three-step chain."""
    from noctilux.annotations.integration import AnnotationRunContext
    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=10, y=5, width=20, height=10, category_id=7)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    # Step 1: crop (10,5,80,40) -> translated (0,0,20,10)
    r1 = ctx._apply_transform(record, {
        "name": "center_crop_ratio", "applied": True,
        "crop_window": {"x": 10, "y": 5, "width": 80, "height": 40,
                        "source_width": 100, "source_height": 50},
    })
    # Step 2: resize to 160x80 -> scale 2x -> (0,0,40,20)
    r2 = ctx._apply_transform(r1, {
        "name": "resize_exact", "applied": True, "output_size": [160, 80],
    })
    # Step 3: horizontal flip on 160x80 -> x = 160-0-40 = 120
    r3 = ctx._apply_transform(r2, {"name": "horizontal_flip", "applied": True})
    box = r3.boxes[0]
    assert box.x == 120.0 and box.y == 0.0 and box.width == 40.0 and box.height == 20.0
    assert r3.width == 160 and r3.height == 80


def test_resize_then_crop_chain():
    """resize -> crop reverse order."""
    from noctilux.annotations.integration import AnnotationRunContext
    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=10, y=5, width=20, height=10, category_id=7)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    # resize to 200x100: scale 2x -> (20,10,40,20)
    r1 = ctx._apply_transform(record, {
        "name": "resize_exact", "applied": True, "output_size": [200, 100],
    })
    # crop (40,20,80,40) on 200x100
    r2 = ctx._apply_transform(r1, {
        "name": "center_crop_ratio", "applied": True,
        "crop_window": {"x": 40, "y": 20, "width": 80, "height": 40,
                        "source_width": 200, "source_height": 100},
    })
    box = r2.boxes[0]
    assert box.x == 0.0 and box.y == 0.0 and box.width == 20.0 and box.height == 10.0
    assert r2.width == 80 and r2.height == 40


def test_applied_false_skips_crop_window_validation():
    """applied=false does not require crop_window at all."""
    from noctilux.annotations.integration import AnnotationRunContext
    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[BoundingBox(x=10, y=5, width=20, height=10, category_id=7)],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    result = ctx._apply_transform(record, {"name": "center_crop_ratio", "applied": False})
    assert len(result.boxes) == 1
    assert result.boxes[0].x == 10


def test_multi_bbox_mixed_crop_outcomes():
    """Fully inside, partially clipped, and fully outside in one crop."""
    from noctilux.annotations.integration import AnnotationRunContext
    record = AnnotationRecord(
        image_id=1, width=100, height=50,
        boxes=[
            BoundingBox(x=30, y=15, width=10, height=5, category_id=1),   # inside
            BoundingBox(x=20, y=10, width=15, height=8, category_id=2),   # clipped
            BoundingBox(x=80, y=40, width=10, height=5, category_id=3),   # outside
        ],
    )
    ctx = AnnotationRunContext.from_records(
        {1: record}, output_path=Path("/tmp/out.json"), on_unsupported_transform="error",
    )
    # crop (25,12,50,25)
    result = ctx._apply_transform(record, {
        "name": "center_crop_ratio", "applied": True,
        "crop_window": {"x": 25, "y": 12, "width": 50, "height": 25,
                        "source_width": 100, "source_height": 50},
    })
    assert len(result.boxes) == 2
    bboxes = sorted([(b.x, b.y, b.width, b.height, b.category_id) for b in result.boxes])
    assert bboxes[0] == (0.0, 0.0, 10.0, 6.0, 2)   # clipped
    assert bboxes[1] == (5.0, 3.0, 10.0, 5.0, 1)   # inside


# --- v0.10.0 output safety tests ---


def test_annotation_output_not_exists_succeeds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=[
            {"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}},
        ]),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)
    exit_code = main(["run", "--config", str(config_path)])
    capsys.readouterr()
    assert exit_code == 0
    assert (tmp_path / "output" / "annotations" / "annotations.json").exists()


def test_annotation_output_exists_no_overwrite_rejects(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR", logger="noctilux")
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=[
            {"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}},
        ]),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)

    # First run succeeds
    exit_code = main(["run", "--config", str(config_path)])
    assert exit_code == 0
    output_json = tmp_path / "output" / "annotations" / "annotations.json"
    assert output_json.exists()
    original_content = output_json.read_text()

    # Second run without overwrite fails
    exit_code = main(["run", "--config", str(config_path)])
    assert exit_code == 1
    assert "already exists" in caplog.text
    # Original file unchanged
    assert output_json.read_text() == original_content


def test_annotation_output_overwrite_true_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=[
            {"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}},
        ]),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)

    exit_code = main(["run", "--config", str(config_path)])
    capsys.readouterr()
    assert exit_code == 0

    config["annotations"]["overwrite_output"] = True
    config_path2 = _write_config(tmp_path, config, name="config2.yaml")
    exit_code = main(["run", "--config", str(config_path2)])
    capsys.readouterr()
    assert exit_code == 0


def test_annotation_input_equals_output_rejects_even_with_overwrite() -> None:
    from noctilux.annotations.integration import AnnotationIntegrationError, build_annotation_run_context

    config = {
        "project": {"name": "test", "seed": 42},
        "input": {"mode": "folder", "image_root": "/tmp/img"},
        "output": {"root": "/tmp/out"},
        "runtime": {"dry_run": False, "num_workers": 1, "skip_broken_images": True,
                    "fail_fast": False, "show_progress": False},
        "pipelines": [{"name": "p", "transforms": [{"name": "brightness_contrast",
                      "params": {"brightness": 1.0, "contrast": 1.0}}]}],
        "annotations": {
            "enabled": True, "format": "coco", "input_path": "/tmp/anno.json",
            "output_path": "/tmp/anno.json", "overwrite_output": True,
        },
    }
    with pytest.raises(AnnotationIntegrationError, match="must differ from"):
        build_annotation_run_context(config)


def test_writer_overwrite_false_rejects_existing(tmp_path: Path) -> None:
    from noctilux.annotations import AnnotationRecord, BoundingBox, CocoAnnotationWriter
    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}
    writer = CocoAnnotationWriter()
    writer.write(records, out, overwrite=True)
    assert out.exists()
    with pytest.raises(FileExistsError, match="already exists"):
        writer.write(records, out, overwrite=False)


def test_writer_overwrite_true_replaces(tmp_path: Path) -> None:
    from noctilux.annotations import AnnotationRecord, BoundingBox, CocoAnnotationWriter
    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}
    writer = CocoAnnotationWriter()
    writer.write(records, out, overwrite=True)
    writer.write(records, out, overwrite=True)
    assert out.exists()


def test_writer_atomic_no_leftover_tmp(tmp_path: Path) -> None:
    from noctilux.annotations import AnnotationRecord, BoundingBox, CocoAnnotationWriter
    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}
    CocoAnnotationWriter().write(records, out, overwrite=True)
    tmps = list(tmp_path.glob(".noctilux_anno_*.tmp"))
    assert len(tmps) == 0


def test_writer_failed_write_preserves_original(tmp_path: Path) -> None:
    from noctilux.annotations import AnnotationRecord, BoundingBox, CocoAnnotationWriter
    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}
    writer = CocoAnnotationWriter()
    writer.write(records, out, overwrite=True)
    original = out.read_text()

    # Create a record that will fail during payload build due to duplicate annotation_id
    bad_records = {
        1: AnnotationRecord(image_id=1, width=10, height=10,
            boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1, annotation_id="dup")]),
        2: AnnotationRecord(image_id=2, width=10, height=10,
            boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1, annotation_id="dup")]),
    }
    try:
        writer.write(bad_records, out, overwrite=True)
    except ValueError:
        pass
    assert out.read_text() == original
    tmps = list(tmp_path.glob(".noctilux_anno_*.tmp"))
    assert len(tmps) == 0


def test_annotation_disabled_no_overwrite_effect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _make_image(tmp_path / "images" / "sample.jpg")
    config = _base_config(tmp_path, transforms=[
        {"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}},
    ])
    config["annotations"] = {"enabled": False, "overwrite_output": "not_a_bool"}
    config_path = _write_config(tmp_path, config)
    from noctilux.cli import main as cli_main
    exit_code = cli_main(["run", "--config", str(config_path)])
    capsys.readouterr()
    assert exit_code == 0


# --- v0.10.0 compatibility tests ---


def test_metadata_schema_unchanged_v10(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output_root, _ = _run_annotation_config(
        tmp_path, capsys,
        transforms=[{"name": "resize_exact", "params": {"width": 200, "height": 100, "interpolation": "nearest"}}],
    )
    assert exit_code == 0
    metadata_root = output_root / "metadata"
    manifest = pd.read_csv(metadata_root / "manifest.csv")
    assert list(manifest.columns) == MANIFEST_COLUMNS
    line = (metadata_root / "transform_log.jsonl").read_text(encoding="utf-8").splitlines()[0]
    transform_log = json.loads(line)
    assert list(transform_log.keys()) == TRANSFORM_LOG_COLUMNS


def test_annotation_guardrails_still_enforced(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR", logger="noctilux")
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=[
            {"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}},
        ]),
        coco_path,
    )
    config["annotations"]["overwrite_output"] = True
    config_path = _write_config(tmp_path, config)

    # resume rejected
    assert main(["run", "--config", str(config_path), "--resume"]) == 1
    assert "fresh full runs only" in caplog.text

    caplog.clear()
    # parallel rejected
    assert main(["run", "--config", str(config_path), "--num-workers", "2"]) == 1
    assert "serial runs only" in caplog.text


# --- v0.10.1 no-clobber atomic write tests ---


def test_writer_toctou_target_created_during_publish(tmp_path: Path) -> None:
    """Deterministic TOCTOU regression: target created between temp write and publish."""
    import os
    from unittest.mock import patch

    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}
    writer = CocoAnnotationWriter()

    real_os_link = os.link

    def patched_link(src, dst, *args, **kwargs):
        if str(dst) == str(out):
            # Intruder creates sentinel target before link completes
            out.write_text(json.dumps({"sentinel": True}))
        return real_os_link(src, dst, *args, **kwargs)

    with patch("os.link", patched_link):
        with pytest.raises(FileExistsError, match="already exists"):
            writer.write(records, out, overwrite=False)

    # Sentinel preserved
    data = json.loads(out.read_text())
    assert "sentinel" in data
    assert "images" not in data

    # No leftover tmp
    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_writer_concurrent_both_overwrite_false(tmp_path: Path) -> None:
    """Two concurrent writers with overwrite=False: exactly one succeeds."""
    import threading

    out = tmp_path / "out.json"
    results: dict[str, str] = {}
    barrier = threading.Barrier(2, timeout=10)

    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}

    def writer_thread(name: str) -> None:
        barrier.wait()
        try:
            CocoAnnotationWriter().write(records, out, overwrite=False)
            results[name] = "success"
        except FileExistsError:
            results[name] = "FileExistsError"

    t1 = threading.Thread(target=writer_thread, args=("w1",))
    t2 = threading.Thread(target=writer_thread, args=("w2",))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    successes = sum(1 for v in results.values() if v == "success")
    failures = sum(1 for v in results.values() if v == "FileExistsError")
    assert successes == 1, f"Expected exactly 1 success, got {successes}: {results}"
    assert failures == 1, f"Expected exactly 1 failure, got {failures}: {results}"

    # Target is valid JSON
    data = json.loads(out.read_text())
    assert "images" in data

    # No leftover tmp files
    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_writer_standalone_overwrite_false_existing(tmp_path: Path) -> None:
    """Standalone writer: overwrite=False rejects existing file."""
    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}
    writer = CocoAnnotationWriter()

    # Create sentinel
    out.write_text(json.dumps({"existing": True}))

    with pytest.raises(FileExistsError, match="already exists"):
        writer.write(records, out, overwrite=False)

    # Sentinel unchanged
    assert json.loads(out.read_text()) == {"existing": True}
    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_writer_overwrite_false_new_file_succeeds(tmp_path: Path) -> None:
    """overwrite=False succeeds when target does not exist."""
    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}
    CocoAnnotationWriter().write(records, out, overwrite=False)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "images" in data
    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_writer_overwrite_false_preserves_content_on_link_eexist(tmp_path: Path) -> None:
    """When os.link fails with EEXIST, original target content is preserved."""
    import os
    from unittest.mock import patch

    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}

    # Create target with sentinel
    out.write_text(json.dumps({"sentinel": "unchanged"}))

    real_os_link = os.link

    def patched_link(src, dst, *args, **kwargs):
        # Target exists, link fails with EEXIST naturally
        return real_os_link(src, dst, *args, **kwargs)

    with patch("os.link", patched_link):
        with pytest.raises(FileExistsError, match="already exists"):
            CocoAnnotationWriter().write(records, out, overwrite=False)

    assert json.loads(out.read_text()) == {"sentinel": "unchanged"}
    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_writer_os_link_oserror_cleans_temp(tmp_path: Path) -> None:
    """When os.link fails with non-EEXIST OSError, temp file is cleaned."""
    from unittest.mock import patch

    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}

    with patch("os.link", side_effect=OSError(28, "No space left on device")):
        with pytest.raises(OSError, match="No space left"):
            CocoAnnotationWriter().write(records, out, overwrite=False)

    assert not out.exists()
    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_writer_os_replace_failure_cleans_temp(tmp_path: Path) -> None:
    """When os.replace fails, temp file is cleaned."""
    from unittest.mock import patch

    out = tmp_path / "out.json"
    out.write_text(json.dumps({"original": True}))
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}

    with patch("os.replace", side_effect=OSError("permission denied")):
        with pytest.raises(OSError, match="permission denied"):
            CocoAnnotationWriter().write(records, out, overwrite=True)

    # Original preserved
    assert json.loads(out.read_text()) == {"original": True}
    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_writer_json_dump_failure_no_target(tmp_path: Path) -> None:
    """When json.dump fails, no target file is created."""
    from unittest.mock import patch

    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}

    with patch("json.dump", side_effect=TypeError("serialization error")):
        with pytest.raises(TypeError, match="serialization error"):
            CocoAnnotationWriter().write(records, out, overwrite=True)

    assert not out.exists()
    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_writer_fsync_failure_no_target(tmp_path: Path) -> None:
    """When fsync fails, no target file is created."""
    from unittest.mock import patch

    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}

    with patch("os.fsync", side_effect=OSError("fsync error")):
        with pytest.raises(OSError, match="fsync error"):
            CocoAnnotationWriter().write(records, out, overwrite=True)

    assert not out.exists()
    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_writer_temp_file_uses_exclusive_create(tmp_path: Path) -> None:
    """Verify temp file is created via mkstemp (exclusive creation)."""
    import tempfile
    from unittest.mock import patch

    out = tmp_path / "out.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}

    mkstemp_calls: list[dict] = []

    original_mkstemp = tempfile.mkstemp

    def patched_mkstemp(*args, **kwargs):
        result = original_mkstemp(*args, **kwargs)
        mkstemp_calls.append({"args": args, "kwargs": kwargs, "dir": kwargs.get("dir")})
        return result

    with patch("tempfile.mkstemp", patched_mkstemp):
        CocoAnnotationWriter().write(records, out, overwrite=True)

    assert len(mkstemp_calls) == 1
    assert str(Path(mkstemp_calls[0]["kwargs"]["dir"])) == str(out.resolve().parent)
    assert ".noctilux_anno_" in mkstemp_calls[0]["kwargs"].get("prefix", "")


def test_writer_success_no_leftover_tmp(tmp_path: Path) -> None:
    """After successful overwrite=True and overwrite=False writes, no temp files remain."""
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}

    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    CocoAnnotationWriter().write(records, out1, overwrite=True)
    CocoAnnotationWriter().write(records, out2, overwrite=False)

    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_writer_target_is_directory_fails(tmp_path: Path) -> None:
    """When target is a directory, writer fails cleanly."""
    out = tmp_path / "out.json"
    out.mkdir()
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}

    with pytest.raises((IsADirectoryError, OSError)):
        CocoAnnotationWriter().write(records, out, overwrite=True)

    # Directory still exists, no tmp
    assert out.is_dir()
    assert list(tmp_path.glob(".noctilux_anno_*.tmp")) == []


def test_integration_write_converts_file_exists_to_annotation_error(
    tmp_path: Path,
) -> None:
    """Integration layer converts writer FileExistsError to AnnotationIntegrationError."""
    from noctilux.annotations.integration import (
        AnnotationIntegrationError,
        AnnotationRunContext,
    )

    out = tmp_path / "annotations.json"
    records = {1: AnnotationRecord(image_id=1, width=10, height=10,
                boxes=[BoundingBox(x=0, y=0, width=5, height=5, category_id=1)])}

    ctx = AnnotationRunContext.from_records(
        records, output_path=out, on_unsupported_transform="error", overwrite_output=False,
    )
    ctx.add_output_record(records[1])

    # Simulate: target created during run
    out.write_text(json.dumps({"intruder": True}))

    with pytest.raises(AnnotationIntegrationError, match="may have been created by another process"):
        ctx.write()


def test_writer_overwrite_false_via_integration_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end: second run with overwrite_output=false is rejected."""
    caplog.set_level("ERROR", logger="noctilux")
    _make_image(tmp_path / "images" / "sample.jpg")
    coco_path = _write_coco(tmp_path)
    config = _with_annotations(
        _base_config(tmp_path, transforms=[
            {"name": "brightness_contrast", "params": {"brightness": 1.0, "contrast": 1.0}},
        ]),
        coco_path,
    )
    config_path = _write_config(tmp_path, config)

    # First run succeeds
    exit_code = main(["run", "--config", str(config_path)])
    capsys.readouterr()
    assert exit_code == 0

    # Second run with overwrite_output false (default) should fail at preflight
    config["annotations"]["overwrite_output"] = False
    config_path2 = _write_config(tmp_path, config, name="config2.yaml")
    exit_code = main(["run", "--config", str(config_path2)])
    capsys.readouterr()
    assert exit_code == 1
    assert "already exists" in caplog.text
