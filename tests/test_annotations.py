from __future__ import annotations

import json
from pathlib import Path

import pytest

from noctilux.annotations import (
    AnnotationRecord,
    BoundingBox,
    CocoAnnotationParser,
    Keypoint,
    MaskRef,
    YoloAnnotationParser,
)
from noctilux.cli import main


def test_bounding_box_to_dict_from_dict() -> None:
    box = BoundingBox(
        x=10,
        y=20,
        width=30,
        height=40,
        category_id=3,
        annotation_id=99,
        iscrowd=0,
        area=1200,
    )

    payload = box.to_dict()

    assert payload["x"] == 10
    assert payload["category_id"] == 3
    assert BoundingBox.from_dict(payload) == box


def test_annotation_record_to_dict_from_dict() -> None:
    record = AnnotationRecord(
        image_id=1,
        image_path="images/sample.jpg",
        width=640,
        height=480,
        boxes=[BoundingBox(x=1, y=2, width=3, height=4, category_id=5)],
        masks=[MaskRef(path="masks/sample.png", segmentation=[[1, 2, 3, 4]], size=[480, 640])],
        keypoints=[Keypoint(x=10, y=12, visible=2)],
        raw={"source": "unit-test"},
    )

    restored = AnnotationRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.masks[0].segmentation == [[1, 2, 3, 4]]


def test_coco_parser_reads_minimal_json(tmp_path: Path) -> None:
    coco_path = _write_coco(
        tmp_path,
        annotations=[
            {
                "id": 10,
                "image_id": 1,
                "category_id": 7,
                "bbox": [10, 20, 30, 40],
            }
        ],
    )

    records = CocoAnnotationParser().parse(coco_path)

    assert set(records) == {1}
    record = records[1]
    assert record.image_path == "sample.jpg"
    assert record.width == 640
    assert record.height == 480
    assert record.boxes == [
        BoundingBox(
            x=10,
            y=20,
            width=30,
            height=40,
            category_id=7,
            annotation_id=10,
        )
    ]


def test_coco_parser_supports_multiple_bboxes_for_one_image(tmp_path: Path) -> None:
    coco_path = _write_coco(
        tmp_path,
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [5, 6, 7, 8]},
        ],
    )

    record = CocoAnnotationParser().parse(coco_path)[1]

    assert len(record.boxes) == 2
    assert [box.category_id for box in record.boxes] == [1, 2]


def test_coco_parser_preserves_segmentation(tmp_path: Path) -> None:
    segmentation = [[10, 20, 30, 40, 50, 60]]
    coco_path = _write_coco(
        tmp_path,
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 2, 3, 4],
                "segmentation": segmentation,
            }
        ],
    )

    record = CocoAnnotationParser().parse(coco_path)[1]

    assert len(record.masks) == 1
    assert record.masks[0].segmentation == segmentation


def test_yolo_parser_reads_single_label_and_denormalizes_bbox(tmp_path: Path) -> None:
    label_path = tmp_path / "sample.txt"
    label_path.write_text("1 0.5 0.5 0.25 0.5\n", encoding="utf-8")

    record = YoloAnnotationParser().parse(label_path, image_width=200, image_height=100, image_path="sample.jpg")

    assert record.image_id == "sample"
    assert record.image_path == "sample.jpg"
    assert record.width == 200
    assert record.height == 100
    assert len(record.boxes) == 1
    box = record.boxes[0]
    assert box.category_id == 1
    assert box.x == pytest.approx(75)
    assert box.y == pytest.approx(25)
    assert box.width == pytest.approx(50)
    assert box.height == pytest.approx(50)


def test_yolo_parser_missing_image_size_raises(tmp_path: Path) -> None:
    label_path = tmp_path / "sample.txt"
    label_path.write_text("1 0.5 0.5 0.25 0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image_width and image_height"):
        YoloAnnotationParser().parse(label_path)


def test_coco_parser_invalid_bbox_raises(tmp_path: Path) -> None:
    coco_path = _write_coco(
        tmp_path,
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 2, 0, 4],
            }
        ],
    )

    with pytest.raises(ValueError, match="bbox width and height"):
        CocoAnnotationParser().parse(coco_path)


def test_annotation_module_does_not_affect_image_only_run_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--config", "configs/examples/quickstart_sample.yaml", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "total_outputs:" in captured.out


def _write_coco(tmp_path: Path, annotations: list[dict]) -> Path:
    coco_path = tmp_path / "annotations.json"
    payload = {
        "images": [
            {
                "id": 1,
                "file_name": "sample.jpg",
                "width": 640,
                "height": 480,
            }
        ],
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "object",
            },
            {
                "id": 7,
                "name": "other",
            },
        ],
    }
    coco_path.write_text(json.dumps(payload), encoding="utf-8")
    return coco_path
