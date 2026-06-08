from __future__ import annotations

import copy
import math

from noctilux.annotations.schema import AnnotationRecord, BoundingBox, Keypoint, MaskRef


def crop_box(
    box: BoundingBox,
    crop_x: float,
    crop_y: float,
    crop_width: float,
    crop_height: float,
    min_area: float = 1.0,
) -> BoundingBox | None:
    crop_x = _validate_nonnegative_number(crop_x, "crop_x")
    crop_y = _validate_nonnegative_number(crop_y, "crop_y")
    crop_width = _validate_positive_number(crop_width, "crop_width")
    crop_height = _validate_positive_number(crop_height, "crop_height")
    min_area = _validate_nonnegative_number(min_area, "min_area")

    crop_x2 = crop_x + crop_width
    crop_y2 = crop_y + crop_height
    box_x2 = box.x + box.width
    box_y2 = box.y + box.height

    intersection_x1 = max(box.x, crop_x)
    intersection_y1 = max(box.y, crop_y)
    intersection_x2 = min(box_x2, crop_x2)
    intersection_y2 = min(box_y2, crop_y2)

    if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
        return None

    new_width = intersection_x2 - intersection_x1
    new_height = intersection_y2 - intersection_y1
    new_area = new_width * new_height
    if new_area < min_area:
        return None

    return BoundingBox(
        x=intersection_x1 - crop_x,
        y=intersection_y1 - crop_y,
        width=new_width,
        height=new_height,
        category_id=box.category_id,
        annotation_id=box.annotation_id,
        iscrowd=box.iscrowd,
        area=new_area,
    )


def crop_record(
    record: AnnotationRecord,
    crop_x: float,
    crop_y: float,
    crop_width: int,
    crop_height: int,
    min_area: float = 1.0,
) -> AnnotationRecord:
    crop_x = _validate_nonnegative_number(crop_x, "crop_x")
    crop_y = _validate_nonnegative_number(crop_y, "crop_y")
    crop_width = _validate_positive_int(crop_width, "crop_width")
    crop_height = _validate_positive_int(crop_height, "crop_height")
    min_area = _validate_nonnegative_number(min_area, "min_area")
    cropped_boxes = [
        cropped_box
        for box in record.boxes
        if (
            cropped_box := crop_box(
                box,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_width=crop_width,
                crop_height=crop_height,
                min_area=min_area,
            )
        )
        is not None
    ]
    return _copy_record(
        record,
        width=crop_width,
        height=crop_height,
        boxes=cropped_boxes,
    )


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
    width: int | float | None,
    height: int | float | None,
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


def _validate_positive_number(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be a positive finite number.")
    return float(value)


def _validate_nonnegative_number(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number.")
    return float(value)


def _validate_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer pixel dimension.")
    return value


def _require_dimension(value: int | None, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value
