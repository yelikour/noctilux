from __future__ import annotations

import pytest

from noctilux.annotations import (
    AnnotationRecord,
    BoundingBox,
    horizontal_flip_box,
    horizontal_flip_record,
    resize_box,
    resize_record,
    vertical_flip_box,
    vertical_flip_record,
)
from noctilux.cli import main


def test_resize_box_scales_bbox_coordinates() -> None:
    box = BoundingBox(x=10, y=20, width=30, height=40, category_id=3)

    resized = resize_box(box, scale_x=2, scale_y=0.5)

    assert resized.x == 20
    assert resized.y == 10
    assert resized.width == 60
    assert resized.height == 20


def test_resize_record_computes_scale_from_image_size() -> None:
    record = _make_record()

    resized = resize_record(record, new_width=400, new_height=50)

    assert resized.width == 400
    assert resized.height == 50
    assert [(box.x, box.y, box.width, box.height) for box in resized.boxes] == [
        (20, 5, 60, 10),
        (200, 10, 40, 15),
    ]


def test_horizontal_flip_box_uses_coco_xywh_formula() -> None:
    box = BoundingBox(x=10, y=20, width=30, height=40, category_id=3)

    flipped = horizontal_flip_box(box, image_width=200)

    assert flipped.x == 160
    assert flipped.y == 20
    assert flipped.width == 30
    assert flipped.height == 40


def test_vertical_flip_box_uses_coco_xywh_formula() -> None:
    box = BoundingBox(x=10, y=20, width=30, height=40, category_id=3)

    flipped = vertical_flip_box(box, image_height=100)

    assert flipped.x == 10
    assert flipped.y == 40
    assert flipped.width == 30
    assert flipped.height == 40


def test_horizontal_flip_record_updates_all_boxes() -> None:
    record = _make_record()

    flipped = horizontal_flip_record(record)

    assert [(box.x, box.y, box.width, box.height) for box in flipped.boxes] == [
        (160, 10, 30, 20),
        (80, 20, 20, 30),
    ]


def test_vertical_flip_record_updates_all_boxes() -> None:
    record = _make_record()

    flipped = vertical_flip_record(record)

    assert [(box.x, box.y, box.width, box.height) for box in flipped.boxes] == [
        (10, 70, 30, 20),
        (100, 50, 20, 30),
    ]


def test_geometry_helpers_do_not_mutate_original_objects() -> None:
    record = _make_record()
    original_payload = record.to_dict()
    original_box_payload = record.boxes[0].to_dict()

    resized_box = resize_box(record.boxes[0], scale_x=2, scale_y=2)
    flipped_record = horizontal_flip_record(record)

    assert resized_box is not record.boxes[0]
    assert flipped_record is not record
    assert flipped_record.boxes[0] is not record.boxes[0]
    assert record.to_dict() == original_payload
    assert record.boxes[0].to_dict() == original_box_payload


def test_resize_record_requires_existing_width_and_height() -> None:
    record = _make_record(width=None, height=100)

    with pytest.raises(ValueError, match="record.width is required"):
        resize_record(record, new_width=200, new_height=100)


def test_flip_record_requires_existing_dimensions() -> None:
    missing_width = _make_record(width=None, height=100)
    missing_height = _make_record(width=200, height=None)

    with pytest.raises(ValueError, match="record.width is required"):
        horizontal_flip_record(missing_width)
    with pytest.raises(ValueError, match="record.height is required"):
        vertical_flip_record(missing_height)


def test_resize_box_rejects_invalid_scale() -> None:
    box = BoundingBox(x=10, y=20, width=30, height=40, category_id=3)

    with pytest.raises(ValueError, match="scale_x"):
        resize_box(box, scale_x=-1, scale_y=1)
    with pytest.raises(ValueError, match="scale_y"):
        resize_box(box, scale_x=1, scale_y=0)


def test_geometry_helpers_preserve_metadata_fields() -> None:
    record = _make_record()

    resized = resize_record(record, new_width=400, new_height=200)
    box = resized.boxes[0]

    assert box.category_id == 3
    assert box.annotation_id == "ann-1"
    assert box.iscrowd == 0
    assert box.area == 600
    assert resized.raw == record.raw
    assert resized.raw is not record.raw


def test_image_only_run_smoke_still_passes(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--config", "configs/examples/quickstart_sample.yaml", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "total_outputs:" in captured.out


def _make_record(width: int | None = 200, height: int | None = 100) -> AnnotationRecord:
    return AnnotationRecord(
        image_id="image-1",
        image_path="sample.jpg",
        width=width,
        height=height,
        boxes=[
            BoundingBox(
                x=10,
                y=10,
                width=30,
                height=20,
                category_id=3,
                annotation_id="ann-1",
                iscrowd=0,
                area=600,
            ),
            BoundingBox(
                x=100,
                y=20,
                width=20,
                height=30,
                category_id=4,
                annotation_id="ann-2",
                iscrowd=1,
                area=600,
            ),
        ],
        raw={"source": {"format": "unit"}},
    )
