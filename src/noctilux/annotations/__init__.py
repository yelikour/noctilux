"""Annotation schema and parser prototypes."""

from noctilux.annotations.geometry import (
    crop_box,
    crop_record,
    horizontal_flip_box,
    horizontal_flip_record,
    resize_box,
    resize_record,
    vertical_flip_box,
    vertical_flip_record,
)
from noctilux.annotations.parsers import BaseAnnotationParser, CocoAnnotationParser, YoloAnnotationParser
from noctilux.annotations.schema import AnnotationRecord, BoundingBox, Keypoint, MaskRef
from noctilux.annotations.writers import BaseAnnotationWriter, CocoAnnotationWriter, YoloAnnotationWriter

__all__ = [
    "AnnotationRecord",
    "BaseAnnotationParser",
    "BaseAnnotationWriter",
    "BoundingBox",
    "CocoAnnotationParser",
    "CocoAnnotationWriter",
    "crop_box",
    "crop_record",
    "horizontal_flip_box",
    "horizontal_flip_record",
    "Keypoint",
    "MaskRef",
    "resize_box",
    "resize_record",
    "vertical_flip_box",
    "vertical_flip_record",
    "YoloAnnotationParser",
    "YoloAnnotationWriter",
]
