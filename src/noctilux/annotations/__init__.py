"""Annotation schema and parser prototypes."""

from noctilux.annotations.geometry import (
    horizontal_flip_box,
    horizontal_flip_record,
    resize_box,
    resize_record,
    vertical_flip_box,
    vertical_flip_record,
)
from noctilux.annotations.parsers import BaseAnnotationParser, CocoAnnotationParser, YoloAnnotationParser
from noctilux.annotations.schema import AnnotationRecord, BoundingBox, Keypoint, MaskRef

__all__ = [
    "AnnotationRecord",
    "BaseAnnotationParser",
    "BoundingBox",
    "CocoAnnotationParser",
    "horizontal_flip_box",
    "horizontal_flip_record",
    "Keypoint",
    "MaskRef",
    "resize_box",
    "resize_record",
    "vertical_flip_box",
    "vertical_flip_record",
    "YoloAnnotationParser",
]
