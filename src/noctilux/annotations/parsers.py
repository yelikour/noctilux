from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from noctilux.annotations.schema import AnnotationRecord, BoundingBox, Keypoint, MaskRef


class BaseAnnotationParser:
    def parse(self, source: str | Path, **kwargs: Any) -> Any:
        raise NotImplementedError


class CocoAnnotationParser(BaseAnnotationParser):
    """Minimal COCO-like JSON reader."""

    def parse(self, source: str | Path, **kwargs: Any) -> dict[int | str, AnnotationRecord]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("COCO annotation file must contain a JSON object.")

        images = _expect_list(payload.get("images", []), "images")
        annotations = _expect_list(payload.get("annotations", []), "annotations")
        categories = _expect_list(payload.get("categories", []), "categories")

        records: dict[int | str, AnnotationRecord] = {}
        for image in images:
            if not isinstance(image, dict):
                raise ValueError("COCO images entries must be objects.")
            image_id = image.get("id")
            if image_id is None:
                raise ValueError("COCO image entry is missing id.")
            records[image_id] = AnnotationRecord(
                image_id=image_id,
                image_path=image.get("file_name"),
                width=image.get("width"),
                height=image.get("height"),
                raw={"image": image, "categories": categories},
            )

        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError("COCO annotations entries must be objects.")
            image_id = annotation.get("image_id")
            if image_id is None:
                raise ValueError("COCO annotation entry is missing image_id.")
            record = records.setdefault(
                image_id,
                AnnotationRecord(image_id=image_id, raw={"categories": categories}),
            )

            bbox = _parse_coco_bbox(annotation.get("bbox"))
            category_id = annotation.get("category_id")
            if category_id is None:
                raise ValueError("COCO annotation entry is missing category_id.")
            record.boxes.append(
                BoundingBox(
                    x=bbox[0],
                    y=bbox[1],
                    width=bbox[2],
                    height=bbox[3],
                    category_id=category_id,
                    annotation_id=annotation.get("id"),
                    iscrowd=annotation.get("iscrowd"),
                    area=_optional_float(annotation.get("area"), "area"),
                )
            )

            segmentation = annotation.get("segmentation")
            if segmentation is not None:
                size = [record.height, record.width] if record.height is not None and record.width is not None else None
                record.masks.append(MaskRef(segmentation=segmentation, size=size))

            keypoints = annotation.get("keypoints")
            if keypoints is not None:
                record.keypoints.extend(_parse_coco_keypoints(keypoints))

        return records


class YoloAnnotationParser(BaseAnnotationParser):
    """Minimal single-file YOLO label reader."""

    def parse(
        self,
        source: str | Path,
        *,
        image_width: int | None = None,
        image_height: int | None = None,
        image_id: int | str | None = None,
        image_path: str | None = None,
    ) -> AnnotationRecord:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        if image_width is None or image_height is None:
            raise ValueError("YOLO parser requires image_width and image_height.")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("YOLO image_width and image_height must be positive.")

        record = AnnotationRecord(
            image_id=image_id if image_id is not None else path.stem,
            image_path=image_path,
            width=image_width,
            height=image_height,
            raw={"format": "yolo", "source": str(path)},
        )
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line {line_number}: expected 5 fields.")

            class_id = _parse_int(parts[0], f"class_id on YOLO line {line_number}")
            center_x = _parse_float(parts[1], f"center_x on YOLO line {line_number}")
            center_y = _parse_float(parts[2], f"center_y on YOLO line {line_number}")
            width = _parse_float(parts[3], f"width on YOLO line {line_number}")
            height = _parse_float(parts[4], f"height on YOLO line {line_number}")
            _validate_normalized_bbox(center_x, center_y, width, height, line_number)

            pixel_width = width * image_width
            pixel_height = height * image_height
            record.boxes.append(
                BoundingBox(
                    x=(center_x - width / 2) * image_width,
                    y=(center_y - height / 2) * image_height,
                    width=pixel_width,
                    height=pixel_height,
                    category_id=class_id,
                    annotation_id=line_number,
                    area=pixel_width * pixel_height,
                )
            )

        return record


def _expect_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"COCO {field_name} field must be a list.")
    return value


def _parse_coco_bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("COCO bbox must be [x, y, width, height].")
    x = _parse_float(value[0], "bbox x")
    y = _parse_float(value[1], "bbox y")
    width = _parse_float(value[2], "bbox width")
    height = _parse_float(value[3], "bbox height")
    if width <= 0 or height <= 0:
        raise ValueError("COCO bbox width and height must be positive.")
    return x, y, width, height


def _parse_coco_keypoints(value: Any) -> list[Keypoint]:
    if not isinstance(value, list) or len(value) % 3 != 0:
        raise ValueError("COCO keypoints must be a flat [x, y, visible, ...] list.")
    keypoints = []
    for index in range(0, len(value), 3):
        keypoints.append(
            Keypoint(
                x=_parse_float(value[index], "keypoint x"),
                y=_parse_float(value[index + 1], "keypoint y"),
                visible=_parse_int(value[index + 2], "keypoint visible"),
            )
        )
    return keypoints


def _validate_normalized_bbox(center_x: float, center_y: float, width: float, height: float, line_number: int) -> None:
    values = {
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
    }
    for name, value in values.items():
        if value < 0 or value > 1:
            raise ValueError(f"YOLO {name} on line {line_number} must be normalized between 0 and 1.")
    if width <= 0 or height <= 0:
        raise ValueError(f"YOLO bbox width and height on line {line_number} must be positive.")


def _parse_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field_name}: {value!r}") from exc


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _parse_float(value, field_name)


def _parse_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for {field_name}: {value!r}") from exc
