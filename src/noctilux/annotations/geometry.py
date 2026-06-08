from __future__ import annotations

import copy
import math

from noctilux.annotations.schema import AnnotationRecord, BoundingBox, Keypoint, MaskRef


def resize_box(box: BoundingBox, scale_x: float, scale_y: float) -> BoundingBox:
    _validate_scale(scale_x, "scale_x")
    _validate_scale(scale_y, "scale_y")
    return BoundingBox(
        x=box.x * scale_x,
        y=box.y * scale_y,
        width=box.width * scale_x,
        height=box.height * scale_y,
        category_id=box.category_id,
        annotation_id=box.annotation_id,
        iscrowd=box.iscrowd,
        area=box.area,
    )


def resize_record(record: AnnotationRecord, new_width: int, new_height: int) -> AnnotationRecord:
    old_width = _require_dimension(record.width, "record.width")
    old_height = _require_dimension(record.height, "record.height")
    target_width = _require_dimension(new_width, "new_width")
    target_height = _require_dimension(new_height, "new_height")
    scale_x = target_width / old_width
    scale_y = target_height / old_height
    return _copy_record(
        record,
        width=target_width,
        height=target_height,
        boxes=[resize_box(box, scale_x, scale_y) for box in record.boxes],
    )


def horizontal_flip_box(box: BoundingBox, image_width: int) -> BoundingBox:
    width = _require_dimension(image_width, "image_width")
    return BoundingBox(
        x=width - box.x - box.width,
        y=box.y,
        width=box.width,
        height=box.height,
        category_id=box.category_id,
        annotation_id=box.annotation_id,
        iscrowd=box.iscrowd,
        area=box.area,
    )


def vertical_flip_box(box: BoundingBox, image_height: int) -> BoundingBox:
    height = _require_dimension(image_height, "image_height")
    return BoundingBox(
        x=box.x,
        y=height - box.y - box.height,
        width=box.width,
        height=box.height,
        category_id=box.category_id,
        annotation_id=box.annotation_id,
        iscrowd=box.iscrowd,
        area=box.area,
    )


def horizontal_flip_record(record: AnnotationRecord) -> AnnotationRecord:
    width = _require_dimension(record.width, "record.width")
    return _copy_record(
        record,
        width=width,
        height=record.height,
        boxes=[horizontal_flip_box(box, width) for box in record.boxes],
    )


def vertical_flip_record(record: AnnotationRecord) -> AnnotationRecord:
    height = _require_dimension(record.height, "record.height")
    return _copy_record(
        record,
        width=record.width,
        height=height,
        boxes=[vertical_flip_box(box, height) for box in record.boxes],
    )


def _copy_record(
    record: AnnotationRecord,
    *,
    width: int | None,
    height: int | None,
    boxes: list[BoundingBox],
) -> AnnotationRecord:
    return AnnotationRecord(
        image_id=record.image_id,
        image_path=record.image_path,
        width=width,
        height=height,
        boxes=boxes,
        masks=[MaskRef.from_dict(mask.to_dict()) for mask in record.masks],
        keypoints=[Keypoint.from_dict(keypoint.to_dict()) for keypoint in record.keypoints],
        raw=copy.deepcopy(record.raw),
    )


def _validate_scale(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be a positive finite scale.")


def _require_dimension(value: int | None, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value
