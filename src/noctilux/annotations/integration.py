from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noctilux.annotations.geometry import horizontal_flip_record, resize_record, vertical_flip_record
from noctilux.annotations.parsers import CocoAnnotationParser
from noctilux.annotations.schema import AnnotationRecord, BoundingBox
from noctilux.annotations.writers import CocoAnnotationWriter

LOGGER = logging.getLogger("noctilux")

SUPPORTED_BBOX_TRANSFORMS = {"resize_exact", "resize_long_edge", "horizontal_flip", "vertical_flip"}
PHOTOMETRIC_TRANSFORMS = {
    "jpeg_compression",
    "webp_compression",
    "png_resave",
    "double_jpeg_compression",
    "gaussian_blur",
    "median_blur",
    "motion_blur",
    "gaussian_noise",
    "poisson_noise",
    "salt_pepper_noise",
    "brightness_contrast",
    "gamma_correction",
    "saturation_hue",
    "grayscale",
    "sharpen",
    "posterize",
}


class AnnotationIntegrationError(RuntimeError):
    """Raised when opt-in annotation sync cannot be completed safely."""


@dataclass
class AnnotationRunContext:
    records_by_image_id: dict[str, AnnotationRecord]
    records_by_path: dict[str, AnnotationRecord]
    ambiguous_path_keys: set[str]
    output_path: Path
    on_unsupported_transform: str = "error"
    output_records: dict[str, AnnotationRecord] = field(default_factory=dict)
    unsupported_transform_warnings: list[str] = field(default_factory=list)
    unmatched_sample_count: int = 0

    @classmethod
    def from_records(
        cls,
        records: dict[int | str, AnnotationRecord],
        *,
        output_path: Path,
        on_unsupported_transform: str,
    ) -> AnnotationRunContext:
        records_by_image_id: dict[str, AnnotationRecord] = {}
        records_by_path: dict[str, AnnotationRecord] = {}
        ambiguous_path_keys: set[str] = set()

        for image_id, record in records.items():
            records_by_image_id[str(image_id)] = record
            records_by_image_id[str(record.image_id)] = record
            if record.image_path is None:
                continue
            for key in _annotation_path_keys(record.image_path):
                if key in records_by_path and records_by_path[key] is not record:
                    ambiguous_path_keys.add(key)
                    records_by_path.pop(key, None)
                    continue
                if key not in ambiguous_path_keys:
                    records_by_path[key] = record

        return cls(
            records_by_image_id=records_by_image_id,
            records_by_path=records_by_path,
            ambiguous_path_keys=ambiguous_path_keys,
            output_path=output_path,
            on_unsupported_transform=on_unsupported_transform,
        )

    def build_output_record(
        self,
        *,
        sample: dict[str, Any],
        pipeline_name: str,
        repeat_index: int,
        output_path: Path,
        transforms: list[dict[str, Any]],
        input_info: dict[str, Any],
    ) -> AnnotationRecord | None:
        source_record = self.find_record(sample)
        if source_record is None:
            self.unmatched_sample_count += 1
            LOGGER.warning(
                "No annotation record found for sample %s (image: %s) in pipeline %s. "
                "Annotation output will not be generated for this sample.",
                sample.get("sample_id", "?"),
                sample.get("image_path", "?"),
                pipeline_name,
            )
            return None

        record = _bbox_only_record(source_record)
        record = _ensure_dimensions(record, input_info=input_info, transforms=transforms)
        for transform_log in transforms:
            record = self._apply_transform(record, transform_log)

        output_image_id = f"{sample['sample_id']}::{pipeline_name}::{repeat_index}"
        return _output_record(record, image_id=output_image_id, image_path=output_path.as_posix())

    def find_record(self, sample: dict[str, Any]) -> AnnotationRecord | None:
        for key in _sample_image_id_keys(sample):
            record = self.records_by_image_id.get(key)
            if record is not None:
                return record
        for key in _sample_path_keys(sample):
            if key in self.ambiguous_path_keys:
                continue
            record = self.records_by_path.get(key)
            if record is not None:
                return record
        return None

    def add_output_record(self, record: AnnotationRecord | None) -> None:
        if record is None:
            return
        key = str(record.image_id)
        if key in self.output_records:
            raise AnnotationIntegrationError(f"Duplicate annotation output image_id: {key}")
        self.output_records[key] = record

    def write(self) -> Path:
        CocoAnnotationWriter().write(self.output_records, self.output_path)
        return self.output_path

    def _apply_transform(self, record: AnnotationRecord, transform_log: dict[str, Any]) -> AnnotationRecord:
        if not transform_log.get("applied", False):
            return record

        name = str(transform_log.get("name") or "")
        if name == "resize_exact":
            width, height = _target_size_from_log(transform_log)
            return resize_record(record, new_width=width, new_height=height)
        if name == "resize_long_edge":
            width, height = _target_size_from_log(transform_log)
            return resize_record(record, new_width=width, new_height=height)
        if name == "horizontal_flip":
            return horizontal_flip_record(record)
        if name == "vertical_flip":
            return vertical_flip_record(record)
        if name in PHOTOMETRIC_TRANSFORMS:
            return record

        return self._handle_unsupported(record, name)

    def _handle_unsupported(self, record: AnnotationRecord, name: str) -> AnnotationRecord:
        message = (
            f"Annotation bbox sync does not support transform {name!r}. "
            "Supported bbox transforms are resize_exact, resize_long_edge, "
            "horizontal_flip, and vertical_flip. "
            "ignore may produce stale or incorrect bboxes and should only be used knowingly."
        )
        if self.on_unsupported_transform == "ignore":
            LOGGER.warning("%s Skipping annotation update for this transform.", message)
            self.unsupported_transform_warnings.append(
                f"transform={name} image_id={record.image_id}"
            )
            return record
        raise AnnotationIntegrationError(message)


def build_annotation_run_context(config: dict[str, Any]) -> AnnotationRunContext | None:
    annotations_cfg = config.get("annotations", {})
    if not annotations_cfg.get("enabled", False):
        return None

    if annotations_cfg.get("format", "coco") != "coco":
        raise AnnotationIntegrationError("v0.8.0 annotation integration only supports annotations.format='coco'.")
    if not annotations_cfg.get("bbox_only", True):
        raise AnnotationIntegrationError("v0.8.0 annotation integration only supports bbox_only=true.")

    input_path = Path(annotations_cfg["input_path"])
    output_path = annotations_cfg.get("output_path")
    if output_path is None:
        output_path = Path(config["output"]["root"]) / "annotations" / "annotations.json"
    else:
        output_path = Path(output_path)

    records = CocoAnnotationParser().parse(input_path)

    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_input == resolved_output:
        raise AnnotationIntegrationError(
            f"annotations.output_path ({output_path}) must differ from "
            f"annotations.input_path ({input_path}). "
            "Overwriting the original annotation file is not allowed."
        )

    return AnnotationRunContext.from_records(
        records,
        output_path=output_path,
        on_unsupported_transform=annotations_cfg.get("on_unsupported_transform", "error"),
    )


def _bbox_only_record(record: AnnotationRecord) -> AnnotationRecord:
    copied = AnnotationRecord.from_dict(record.to_dict())
    copied.masks = []
    copied.keypoints = []
    return copied


def _output_record(record: AnnotationRecord, *, image_id: str, image_path: str) -> AnnotationRecord:
    boxes = [
        BoundingBox(
            x=box.x,
            y=box.y,
            width=box.width,
            height=box.height,
            category_id=box.category_id,
            annotation_id=None,
            iscrowd=box.iscrowd,
            area=None,
        )
        for box in record.boxes
    ]
    return AnnotationRecord(
        image_id=image_id,
        image_path=image_path,
        width=record.width,
        height=record.height,
        boxes=boxes,
        masks=[],
        keypoints=[],
        raw=copy.deepcopy(record.raw),
    )


def _ensure_dimensions(
    record: AnnotationRecord,
    *,
    input_info: dict[str, Any],
    transforms: list[dict[str, Any]],
) -> AnnotationRecord:
    if record.width is not None and record.height is not None:
        return record

    width = input_info.get("width")
    height = input_info.get("height")
    for transform in transforms:
        size = transform.get("input_size")
        if isinstance(size, list | tuple) and len(size) == 2:
            width, height = size
            break

    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        copied = _bbox_only_record(record)
        copied.width = width
        copied.height = height
        return copied
    return record


def _target_size_from_log(transform_log: dict[str, Any]) -> tuple[int, int]:
    output_size = transform_log.get("output_size")
    if isinstance(output_size, list | tuple) and len(output_size) == 2:
        width, height = output_size
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height

    name = transform_log.get("name")
    params = transform_log.get("params") or {}
    if name == "resize_exact":
        width = params.get("width")
        height = params.get("height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height

    raise AnnotationIntegrationError(f"Missing output_size for annotation bbox sync transform {name!r}.")


def _sample_image_id_keys(sample: dict[str, Any]) -> list[str]:
    keys = [str(sample.get("sample_id", ""))]
    metadata = sample.get("metadata") or {}
    image_id = metadata.get("image_id")
    if image_id is not None:
        keys.append(str(image_id))
    return _dedupe(keys)


def _sample_path_keys(sample: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    metadata = sample.get("metadata") or {}
    relative_path = metadata.get("relative_path")
    if relative_path:
        keys.extend(_annotation_path_keys(str(relative_path)))
    image_path = sample.get("image_path")
    if image_path is not None:
        keys.extend(_annotation_path_keys(str(image_path)))
    return _dedupe(keys)


def _annotation_path_keys(value: str) -> list[str]:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    keys = [normalized]
    if path.name:
        keys.append(path.name)
    return _dedupe(keys)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
