from __future__ import annotations

import pytest

from noctilux.annotations import (
    AnnotationRecord,
    BoundingBox,
    crop_box,
    crop_record,
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


def test_crop_box_inside_window_shifts_to_crop_origin() -> None:
    box = BoundingBox(x=20, y=30, width=40, height=50, category_id=3, annotation_id="ann", iscrowd=0)

    cropped = crop_box(box, crop_x=10, crop_y=20, crop_width=100, crop_height=100)

    assert cropped == BoundingBox(
        x=10,
        y=10,
        width=40,
        height=50,
        category_id=3,
        annotation_id="ann",
        iscrowd=0,
        area=2000,
    )


def test_crop_box_clips_left_and_top_edges() -> None:
    box = BoundingBox(x=5, y=10, width=30, height=40, category_id=3)

    cropped = crop_box(box, crop_x=20, crop_y=25, crop_width=100, crop_height=100)

    assert cropped is not None
    assert (cropped.x, cropped.y, cropped.width, cropped.height, cropped.area) == (0, 0, 15, 25, 375)


def test_crop_box_clips_right_and_bottom_edges() -> None:
    box = BoundingBox(x=80, y=90, width=40, height=50, category_id=3)

    cropped = crop_box(box, crop_x=20, crop_y=25, crop_width=90, crop_height=100)

    assert cropped is not None
    assert (cropped.x, cropped.y, cropped.width, cropped.height, cropped.area) == (60, 65, 30, 35, 1050)


def test_crop_box_outside_window_returns_none() -> None:
    box = BoundingBox(x=120, y=130, width=10, height=10, category_id=3)

    assert crop_box(box, crop_x=0, crop_y=0, crop_width=100, crop_height=100) is None


def test_crop_box_smaller_than_min_area_returns_none() -> None:
    box = BoundingBox(x=95, y=95, width=10, height=10, category_id=3)

    assert crop_box(box, crop_x=0, crop_y=0, crop_width=100, crop_height=100, min_area=26) is None


def test_crop_record_filters_boxes_and_updates_size() -> None:
    record = _make_record()

    cropped = crop_record(record, crop_x=20, crop_y=0, crop_width=90, crop_height=60)

    assert cropped.width == 90
    assert cropped.height == 60
    assert [(box.x, box.y, box.width, box.height) for box in cropped.boxes] == [
        (0, 10, 20, 20),
        (80, 20, 10, 30),
    ]


def test_crop_record_drops_non_intersecting_boxes() -> None:
    record = AnnotationRecord(
        image_id="image-1",
        width=200,
        height=100,
        boxes=[
            BoundingBox(x=10, y=10, width=20, height=20, category_id=1),
            BoundingBox(x=150, y=80, width=20, height=10, category_id=2),
        ],
    )

    cropped = crop_record(record, crop_x=0, crop_y=0, crop_width=50, crop_height=50)

    assert len(cropped.boxes) == 1
    assert cropped.boxes[0].category_id == 1


def test_crop_record_does_not_mutate_original_record() -> None:
    record = _make_record()
    original_payload = record.to_dict()

    cropped = crop_record(record, crop_x=20, crop_y=0, crop_width=90, crop_height=60)

    assert cropped is not record
    assert cropped.boxes[0] is not record.boxes[0]
    assert record.to_dict() == original_payload


def test_crop_updates_bbox_area_to_cropped_area() -> None:
    box = BoundingBox(x=5, y=5, width=20, height=20, category_id=3, area=400)

    cropped = crop_box(box, crop_x=10, crop_y=10, crop_width=100, crop_height=100)

    assert cropped is not None
    assert cropped.area == 225


def test_crop_helpers_preserve_metadata_fields() -> None:
    record = _make_record()

    cropped = crop_record(record, crop_x=20, crop_y=0, crop_width=90, crop_height=60)
    box = cropped.boxes[0]

    assert box.category_id == 3
    assert box.annotation_id == "ann-1"
    assert box.iscrowd == 0
    assert cropped.raw == record.raw
    assert cropped.raw is not record.raw


def test_crop_rejects_invalid_window_and_min_area() -> None:
    box = BoundingBox(x=10, y=10, width=20, height=20, category_id=3)

    with pytest.raises(ValueError, match="crop_width"):
        crop_box(box, crop_x=0, crop_y=0, crop_width=0, crop_height=20)
    with pytest.raises(ValueError, match="crop_height"):
        crop_box(box, crop_x=0, crop_y=0, crop_width=20, crop_height=0)
    with pytest.raises(ValueError, match="min_area"):
        crop_box(box, crop_x=0, crop_y=0, crop_width=20, crop_height=20, min_area=-1)
    with pytest.raises(ValueError, match="crop_x"):
        crop_record(_make_record(), crop_x=-1, crop_y=0, crop_width=20, crop_height=20)
    with pytest.raises(ValueError, match="crop_y"):
        crop_record(_make_record(), crop_x=0, crop_y=-1, crop_width=20, crop_height=20)


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


# --- v0.7.4 crop int-validation tests ---


def test_crop_record_output_dimensions_are_int() -> None:
    record = _make_record()

    cropped = crop_record(record, crop_x=10, crop_y=0, crop_width=90, crop_height=60)

    assert isinstance(cropped.width, int)
    assert isinstance(cropped.height, int)
    assert cropped.width == 90
    assert cropped.height == 60


def test_crop_record_rejects_float_crop_dimensions() -> None:
    record = _make_record()

    with pytest.raises(ValueError, match="crop_width"):
        crop_record(record, crop_x=0, crop_y=0, crop_width=100.5, crop_height=100)
    with pytest.raises(ValueError, match="crop_height"):
        crop_record(record, crop_x=0, crop_y=0, crop_width=100, crop_height=100.5)


def test_crop_record_edge_touching_bbox_returns_none() -> None:
    box = BoundingBox(x=100, y=100, width=10, height=10, category_id=3)

    result = crop_box(box, crop_x=0, crop_y=0, crop_width=100, crop_height=100)

    assert result is None


def test_crop_box_exact_min_area_retained() -> None:
    box = BoundingBox(x=0, y=0, width=5, height=5, category_id=3)

    result = crop_box(box, crop_x=0, crop_y=0, crop_width=3, crop_height=3, min_area=9.0)

    assert result is not None
    assert result.area == 9.0


def test_crop_record_works_without_record_dimensions() -> None:
    record = AnnotationRecord(
        image_id="img",
        width=None,
        height=None,
        boxes=[BoundingBox(x=5, y=5, width=20, height=20, category_id=1)],
    )

    cropped = crop_record(record, crop_x=0, crop_y=0, crop_width=30, crop_height=30)

    assert len(cropped.boxes) == 1
    assert isinstance(cropped.width, int)
    assert cropped.width == 30


def test_crop_record_copies_masks_and_keypoints_without_sync() -> None:
    from noctilux.annotations.schema import Keypoint, MaskRef

    record = AnnotationRecord(
        image_id="img",
        width=200,
        height=200,
        boxes=[BoundingBox(x=5, y=5, width=20, height=20, category_id=1)],
        masks=[MaskRef(segmentation=[[1, 2, 3]], size=[200, 200])],
        keypoints=[Keypoint(x=15, y=15, visible=2)],
    )
    original_masks = record.masks[0].to_dict()
    original_kps = record.keypoints[0].to_dict()

    cropped = crop_record(record, crop_x=0, crop_y=0, crop_width=100, crop_height=100)

    assert len(cropped.masks) == 1
    assert cropped.masks[0].to_dict() == original_masks
    assert len(cropped.keypoints) == 1
    assert cropped.keypoints[0].to_dict() == original_kps
