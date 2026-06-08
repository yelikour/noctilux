from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from noctilux.annotations.schema import AnnotationRecord


class BaseAnnotationWriter:
    """Base class for annotation writers."""

    def write(self, records: Any, output: str | Path) -> None:
        raise NotImplementedError

    def to_string(self, records: Any) -> str:
        raise NotImplementedError


class CocoAnnotationWriter(BaseAnnotationWriter):
    """Prototype COCO-like JSON annotation writer.

    Accepts a mapping of image_id -> AnnotationRecord and produces a COCO-style
    dict with ``images``, ``annotations``, and ``categories`` fields.

    This writer assumes bboxes are already valid and within image bounds.
    """

    def write(self, records: Any, output: str | Path) -> None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._build_coco_payload(records)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def to_string(self, records: Any) -> str:
        payload = self._build_coco_payload(records)
        return json.dumps(payload, indent=2)

    def _build_coco_payload(self, records: dict[int | str, AnnotationRecord]) -> dict[str, Any]:
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        categories: dict[int, dict[str, Any]] = {}

        annotation_counter = 0

        for image_id, record in records.items():
            image_entry: dict[str, Any] = {"id": image_id}
            if record.image_path is not None:
                image_entry["file_name"] = record.image_path
            if record.width is not None:
                image_entry["width"] = record.width
            if record.height is not None:
                image_entry["height"] = record.height
            images.append(image_entry)

            for box in record.boxes:
                annotation_counter += 1
                ann_id = box.annotation_id if box.annotation_id is not None else annotation_counter
                annotation_entry: dict[str, Any] = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": box.category_id,
                    "bbox": [box.x, box.y, box.width, box.height],
                    "area": box.area if box.area is not None else box.width * box.height,
                }
                if box.iscrowd is not None:
                    annotation_entry["iscrowd"] = box.iscrowd

                _collect_category(categories, box.category_id)
                annotations.append(annotation_entry)

            for mask in record.masks:
                if mask.segmentation is not None:
                    annotation_counter += 1
                    mask_entry: dict[str, Any] = {
                        "id": annotation_counter,
                        "image_id": image_id,
                        "category_id": -1,
                        "segmentation": mask.segmentation,
                        "iscrowd": 0,
                    }
                    if mask.size is not None:
                        mask_entry["size"] = mask.size
                    annotations.append(mask_entry)

        return {
            "images": images,
            "annotations": annotations,
            "categories": list(categories.values()),
        }


class YoloAnnotationWriter(BaseAnnotationWriter):
    """Prototype YOLO TXT annotation writer.

    Accepts a single AnnotationRecord and produces normalized
    ``class_id center_x center_y width height`` lines.

    Requires ``record.width`` and ``record.height`` to be set.
    This writer assumes boxes are already valid and within image bounds.
    """

    def write(self, records: Any, output: str | Path) -> None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._format_record(records), encoding="utf-8")

    def to_string(self, records: Any) -> str:
        return self._format_record(records)

    def _format_record(self, record: AnnotationRecord) -> str:
        if record.width is None or record.height is None:
            raise ValueError("YOLO writer requires record.width and record.height.")
        if record.width <= 0 or record.height <= 0:
            raise ValueError("record.width and record.height must be positive.")

        img_w = float(record.width)
        img_h = float(record.height)
        lines: list[str] = []

        for box in record.boxes:
            cx = (box.x + box.width / 2) / img_w
            cy = (box.y + box.height / 2) / img_h
            w = box.width / img_w
            h = box.height / img_h
            lines.append(f"{box.category_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        return "\n".join(lines) + ("\n" if lines else "")


def _collect_category(categories: dict[int, dict[str, Any]], category_id: int) -> None:
    if category_id not in categories:
        categories[category_id] = {"id": category_id, "name": str(category_id)}
