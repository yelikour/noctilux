from __future__ import annotations

import json
import pathlib

import pytest

from noctilux.annotations import (
    AnnotationRecord,
    BoundingBox,
    CocoAnnotationParser,
    CocoAnnotationWriter,
    MaskRef,
    YoloAnnotationWriter,
    resize_record,
)
from noctilux.cli import main


def _make_records() -> dict[int, AnnotationRecord]:
    return {
        1: AnnotationRecord(
            image_id=1,
            image_path="img001.jpg",
            width=640,
            height=480,
            boxes=[
                BoundingBox(x=10, y=20, width=100, height=80, category_id=1, annotation_id=101, area=8000),
                BoundingBox(x=200, y=150, width=50, height=60, category_id=2, annotation_id=102, area=3000),
            ],
        ),
        2: AnnotationRecord(
            image_id=2,
            image_path="img002.jpg",
            width=800,
            height=600,
            boxes=[
                BoundingBox(x=50, y=50, width=200, height=150, category_id=1, area=30000),
            ],
        ),
    }


# --- COCO writer core tests ---


def test_coco_writer_produces_images_annotations_categories() -> None:
    writer = CocoAnnotationWriter()
    payload = json.loads(writer.to_string(_make_records()))

    assert "images" in payload
    assert "annotations" in payload
    assert "categories" in payload
    assert len(payload["images"]) == 2
    assert len(payload["annotations"]) == 3
    assert len(payload["categories"]) == 2


def test_coco_writer_bbox_format() -> None:
    writer = CocoAnnotationWriter()
    payload = json.loads(writer.to_string(_make_records()))

    bboxes = [ann["bbox"] for ann in payload["annotations"]]
    assert bboxes[0] == [10, 20, 100, 80]
    assert bboxes[1] == [200, 150, 50, 60]


def test_coco_writer_uses_existing_annotation_id() -> None:
    writer = CocoAnnotationWriter()
    payload = json.loads(writer.to_string(_make_records()))

    ann_ids = [ann["id"] for ann in payload["annotations"]]
    assert 101 in ann_ids
    assert 102 in ann_ids


def test_coco_writer_generates_stable_id_when_missing() -> None:
    records = {
        1: AnnotationRecord(
            image_id=1,
            width=100,
            height=100,
            boxes=[BoundingBox(x=0, y=0, width=10, height=10, category_id=1)],
        ),
    }
    writer = CocoAnnotationWriter()
    payload = json.loads(writer.to_string(records))

    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["id"] == 1


def test_coco_writer_does_not_modify_records() -> None:
    records = _make_records()
    original = {k: v.to_dict() for k, v in records.items()}

    writer = CocoAnnotationWriter()
    writer.to_string(records)

    for k, v in records.items():
        assert v.to_dict() == original[k]


def test_coco_writer_writes_json_file(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out.json"
    writer = CocoAnnotationWriter()
    writer.write(_make_records(), out)

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["images"]) == 2


# --- COCO writer annotation_id uniqueness (v0.7.5) ---


def test_coco_writer_auto_id_does_not_collide_with_explicit() -> None:
    records = {
        1: AnnotationRecord(
            image_id=1,
            width=100,
            height=100,
            boxes=[
                BoundingBox(x=0, y=0, width=10, height=10, category_id=1),
                BoundingBox(x=20, y=20, width=10, height=10, category_id=2),
            ],
        ),
        2: AnnotationRecord(
            image_id=2,
            width=100,
            height=100,
            boxes=[
                BoundingBox(x=0, y=0, width=10, height=10, category_id=1, annotation_id=2),
            ],
        ),
    }
    writer = CocoAnnotationWriter()
    payload = json.loads(writer.to_string(records))

    ids = [ann["id"] for ann in payload["annotations"]]
    assert len(ids) == len(set(ids)), f"Duplicate ids found: {ids}"


def test_coco_writer_duplicate_explicit_annotation_id_raises() -> None:
    records = {
        1: AnnotationRecord(
            image_id=1,
            width=100,
            height=100,
            boxes=[
                BoundingBox(x=0, y=0, width=10, height=10, category_id=1, annotation_id=5),
            ],
        ),
        2: AnnotationRecord(
            image_id=2,
            width=100,
            height=100,
            boxes=[
                BoundingBox(x=0, y=0, width=10, height=10, category_id=1, annotation_id=5),
            ],
        ),
    }
    writer = CocoAnnotationWriter()

    with pytest.raises(ValueError, match="[Dd]uplicate annotation_id"):
        writer.to_string(records)


def test_coco_writer_multi_image_ids_globally_unique() -> None:
    records = {
        1: AnnotationRecord(
            image_id=1,
            width=100,
            height=100,
            boxes=[
                BoundingBox(x=0, y=0, width=10, height=10, category_id=1, annotation_id=10),
                BoundingBox(x=20, y=20, width=10, height=10, category_id=2),
            ],
        ),
        2: AnnotationRecord(
            image_id=2,
            width=100,
            height=100,
            boxes=[
                BoundingBox(x=0, y=0, width=10, height=10, category_id=1),
                BoundingBox(x=30, y=30, width=10, height=10, category_id=3, annotation_id=20),
            ],
        ),
        3: AnnotationRecord(
            image_id=3,
            width=100,
            height=100,
            boxes=[
                BoundingBox(x=0, y=0, width=10, height=10, category_id=2),
            ],
        ),
    }
    writer = CocoAnnotationWriter()
    payload = json.loads(writer.to_string(records))

    ids = [ann["id"] for ann in payload["annotations"]]
    assert len(ids) == len(set(ids)), f"Duplicate ids: {ids}"
    assert 10 in ids
    assert 20 in ids


# --- COCO writer mask handling (v0.7.5) ---


def test_coco_writer_mask_only_record_no_negative_category_id() -> None:
    records = {
        1: AnnotationRecord(
            image_id=1,
            width=100,
            height=100,
            boxes=[],
            masks=[MaskRef(segmentation=[[1.0, 2.0, 3.0, 4.0]], size=[100, 100])],
        ),
    }
    writer = CocoAnnotationWriter()
    payload = json.loads(writer.to_string(records))

    cat_ids = [ann["category_id"] for ann in payload["annotations"]]
    assert -1 not in cat_ids


def test_coco_writer_bbox_with_segmentation_preserves_both() -> None:
    seg = [[1.0, 2.0, 3.0, 4.0]]
    records = {
        1: AnnotationRecord(
            image_id=1,
            width=100,
            height=100,
            boxes=[BoundingBox(x=0, y=0, width=10, height=10, category_id=1)],
            masks=[MaskRef(segmentation=seg, size=[100, 100])],
        ),
    }
    writer = CocoAnnotationWriter()
    payload = json.loads(writer.to_string(records))

    assert len(payload["annotations"]) == 1
    ann = payload["annotations"][0]
    assert ann["bbox"] == [0, 0, 10, 10]
    assert ann["segmentation"] == seg
    assert ann["category_id"] == 1


def test_coco_writer_empty_boxes_empty_masks_no_annotations() -> None:
    records = {
        1: AnnotationRecord(
            image_id=1,
            width=100,
            height=100,
        ),
    }
    writer = CocoAnnotationWriter()
    payload = json.loads(writer.to_string(records))

    assert len(payload["annotations"]) == 0
    assert len(payload["images"]) == 1


# --- COCO writer round-trip ---


def test_coco_writer_output_readable_by_parser(tmp_path: pathlib.Path) -> None:
    records = {
        1: AnnotationRecord(
            image_id=1,
            image_path="img.jpg",
            width=200,
            height=100,
            boxes=[
                BoundingBox(x=10, y=20, width=30, height=40, category_id=1, annotation_id=1, area=1200),
            ],
        ),
    }
    out = tmp_path / "coco.json"
    writer = CocoAnnotationWriter()
    writer.write(records, out)

    parser = CocoAnnotationParser()
    parsed = parser.parse(out)

    assert 1 in parsed
    assert parsed[1].image_path == "img.jpg"
    assert parsed[1].width == 200
    assert parsed[1].height == 100
    assert len(parsed[1].boxes) == 1
    assert parsed[1].boxes[0].category_id == 1
    assert parsed[1].boxes[0].x == pytest.approx(10)
    assert parsed[1].boxes[0].y == pytest.approx(20)
    assert parsed[1].boxes[0].width == pytest.approx(30)
    assert parsed[1].boxes[0].height == pytest.approx(40)


# --- YOLO writer core tests ---


def test_yolo_writer_outputs_normalized_cx_cy_w_h() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=100,
        height=200,
        boxes=[BoundingBox(x=10, y=40, width=20, height=60, category_id=3)],
    )
    writer = YoloAnnotationWriter()
    text = writer.to_string(record)

    parts = text.strip().split()
    assert len(parts) == 5
    assert parts[0] == "3"
    cx = float(parts[1])
    cy = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    assert cx == pytest.approx(0.2)
    assert cy == pytest.approx(0.35)
    assert w == pytest.approx(0.2)
    assert h == pytest.approx(0.3)


def test_yolo_writer_missing_dimensions_raises() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=None,
        height=None,
        boxes=[BoundingBox(x=0, y=0, width=10, height=10, category_id=1)],
    )
    writer = YoloAnnotationWriter()

    with pytest.raises(ValueError, match="record.width and record.height"):
        writer.to_string(record)


def test_yolo_writer_multiple_boxes_produce_multiple_lines() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=200,
        height=200,
        boxes=[
            BoundingBox(x=10, y=10, width=20, height=20, category_id=1),
            BoundingBox(x=100, y=100, width=50, height=50, category_id=2),
        ],
    )
    writer = YoloAnnotationWriter()
    text = writer.to_string(record)

    lines = [line for line in text.strip().split("\n") if line.strip()]
    assert len(lines) == 2
    assert lines[0].split()[0] == "1"
    assert lines[1].split()[0] == "2"


def test_yolo_writer_does_not_modify_record() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=100,
        height=100,
        boxes=[BoundingBox(x=0, y=0, width=10, height=10, category_id=1)],
    )
    original = record.to_dict()

    writer = YoloAnnotationWriter()
    writer.to_string(record)

    assert record.to_dict() == original


def test_yolo_writer_writes_txt_file(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "label.txt"
    record = AnnotationRecord(
        image_id=1,
        width=100,
        height=100,
        boxes=[BoundingBox(x=0, y=0, width=10, height=10, category_id=1)],
    )
    writer = YoloAnnotationWriter()
    writer.write(record, out)

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    parts = content.strip().split()
    assert len(parts) == 5


def test_yolo_writer_zero_dimensions_raises() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=0,
        height=100,
        boxes=[BoundingBox(x=0, y=0, width=10, height=10, category_id=1)],
    )
    writer = YoloAnnotationWriter()

    with pytest.raises(ValueError, match="positive"):
        writer.to_string(record)


# --- YOLO writer v0.7.5 additions ---


def test_yolo_writer_empty_boxes_outputs_empty() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=100,
        height=100,
        boxes=[],
    )
    writer = YoloAnnotationWriter()
    text = writer.to_string(record)

    assert text.strip() == ""


def test_yolo_writer_edge_bbox_normalized_correctly() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=200,
        height=200,
        boxes=[BoundingBox(x=190, y=190, width=10, height=10, category_id=1)],
    )
    writer = YoloAnnotationWriter()
    text = writer.to_string(record)

    parts = text.strip().split()
    cx = float(parts[1])
    cy = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    assert cx == pytest.approx(0.975)
    assert cy == pytest.approx(0.975)
    assert w == pytest.approx(0.05)
    assert h == pytest.approx(0.05)


def test_yolo_writer_validate_bounds_rejects_out_of_range() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=100,
        height=100,
        boxes=[BoundingBox(x=150, y=0, width=30, height=10, category_id=1)],
    )
    writer = YoloAnnotationWriter(validate_bounds=True)

    with pytest.raises(ValueError, match="out of bounds"):
        writer.to_string(record)


def test_yolo_writer_validate_bounds_passes_for_valid_boxes() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=100,
        height=100,
        boxes=[BoundingBox(x=10, y=10, width=20, height=20, category_id=1)],
    )
    writer = YoloAnnotationWriter(validate_bounds=True)
    text = writer.to_string(record)

    parts = text.strip().split()
    assert len(parts) == 5


def test_yolo_writer_default_does_not_validate_bounds() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=100,
        height=100,
        boxes=[BoundingBox(x=-10, y=0, width=30, height=10, category_id=1)],
    )
    writer = YoloAnnotationWriter()
    text = writer.to_string(record)

    assert text.strip() != ""


# --- Geometry + writer end-to-end ---


def test_resize_then_yolo_writer_normalized_correctly() -> None:
    record = AnnotationRecord(
        image_id=1,
        width=200,
        height=200,
        boxes=[BoundingBox(x=50, y=50, width=100, height=100, category_id=1)],
    )

    resized = resize_record(record, new_width=400, new_height=400)

    writer = YoloAnnotationWriter()
    text = writer.to_string(resized)
    parts = text.strip().split()
    cx = float(parts[1])
    cy = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])

    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert w == pytest.approx(0.5)
    assert h == pytest.approx(0.5)


# --- Image-only smoke ---


def test_image_only_run_dry_run_still_passes(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--config", "configs/examples/quickstart_sample.yaml", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "total_outputs:" in captured.out
