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
    Annotation IDs are globally unique: explicit IDs are preserved, and
    auto-generated IDs start above the maximum explicit ID to avoid collisions.
    Duplicate explicit annotation IDs across records raise ValueError.

    Standalone mask annotations without a linked category_id are not emitted.
    Segmentation payloads from MaskRef are only attached when they can be
    associated with an existing bbox annotation.
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

        explicit_ids = _collect_explicit_ids(records)
        next_auto_id = max(explicit_ids) + 1 if explicit_ids else 1

        # Build index: image_id -> list of mask segmentations
        mask_index = _build_mask_index(records)

        for image_id, record in records.items():
            image_entry: dict[str, Any] = {"id": image_id}
            if record.image_path is not None:
                image_entry["file_name"] = record.image_path
            if record.width is not None:
                image_entry["width"] = record.width
            if record.height is not None:
                image_entry["height"] = record.height
            images.append(image_entry)

            for box_idx, box in enumerate(record.boxes):
                ann_id = box.annotation_id if box.annotation_id is not None else next_auto_id
                if box.annotation_id is None:
                    next_auto_id += 1

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

                # Attach segmentation from corresponding mask if available
                segs = mask_index.get(image_id, [])
                if box_idx < len(segs) and segs[box_idx] is not None:
                    annotation_entry["segmentation"] = segs[box_idx]
                    if record.width is not None and record.height is not None:
                        annotation_entry["size"] = [record.height, record.width]

                annotations.append(annotation_entry)

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
    It does not clip or validate out-of-bounds coordinates in v0.7.5.

    Args:
        validate_bounds: If True, raise ValueError when any normalized
            coordinate falls outside [0, 1]. Defaults to False for
            prototype simplicity.
    """

    def __init__(self, *, validate_bounds: bool = False) -> None:
        self._validate_bounds = validate_bounds

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

            if self._validate_bounds:
                for name, value in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                    if value < 0 or value > 1:
                        raise ValueError(
                            f"YOLO normalized {name}={value:.6f} out of bounds [0, 1]. "
                            "Pass validate_bounds=False to skip this check."
                        )

            lines.append(f"{box.category_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        return "\n".join(lines) + ("\n" if lines else "")


def _collect_explicit_ids(records: dict[int | str, AnnotationRecord]) -> set[int]:
    seen: set[int] = set()
    for record in records.values():
        for box in record.boxes:
            if box.annotation_id is not None:
                aid = box.annotation_id
                if isinstance(aid, int) and aid in seen:
                    raise ValueError(f"Duplicate annotation_id {aid} found across records.")
                if isinstance(aid, int):
                    seen.add(aid)
    return seen


def _build_mask_index(records: dict[int | str, AnnotationRecord]) -> dict[int | str, list[Any]]:
    """Build per-image mask segmentation list, indexed by position.

    Only returns segmentation payloads; mask-only records without a
    corresponding bbox position are not included.
    """
    index: dict[int | str, list[Any]] = {}
    for image_id, record in records.items():
        segs: list[Any] = []
        for mask in record.masks:
            segs.append(mask.segmentation)
        if segs:
            index[image_id] = segs
    return index


def _collect_category(categories: dict[int, dict[str, Any]], category_id: int) -> None:
    if category_id not in categories:
        categories[category_id] = {"id": category_id, "name": str(category_id)}
