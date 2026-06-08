from __future__ import annotations

import json

import pytest

from noctilux.annotations import (
    AnnotationRecord,
    BoundingBox,
    CocoAnnotationWriter,
    MaskRef,
    YoloAnnotationWriter,
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


def test_coco_writer_preserves_segmentation() -> None:
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

    mask_anns = [a for a in payload["annotations"] if "segmentation" in a]
    assert len(mask_anns) == 1
    assert mask_anns[0]["segmentation"] == seg


def test_coco_writer_does_not_modify_records() -> None:
    records = _make_records()
    original = {k: v.to_dict() for k, v in records.items()}

    writer = CocoAnnotationWriter()
    writer.to_string(records)

    for k, v in records.items():
        assert v.to_dict() == original[k]


def test_coco_writer_writes_json_file(tmp_path: object) -> None:
    import pathlib

    out = pathlib.Path(str(tmp_path)) / "out.json"
    writer = CocoAnnotationWriter()
    writer.write(_make_records(), out)

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["images"]) == 2


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


def test_yolo_writer_writes_txt_file(tmp_path: object) -> None:
    import pathlib

    out = pathlib.Path(str(tmp_path)) / "label.txt"
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


def test_image_only_run_dry_run_still_passes(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--config", "configs/examples/quickstart_sample.yaml", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "total_outputs:" in captured.out
