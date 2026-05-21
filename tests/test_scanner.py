from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from noctilux.scanner import scan_folder, scan_manifest


def test_folder_mode_scans_images(tmp_path: Path) -> None:
    class_dir = tmp_path / "cat"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (32, 16), color="red").save(class_dir / "a.jpg")
    Image.new("RGB", (24, 24), color="blue").save(class_dir / "b.png")
    (class_dir / "note.txt").write_text("ignore", encoding="utf-8")

    samples = scan_folder(tmp_path, recursive=True, infer_label_from_subdir=True)

    assert len(samples) == 2
    assert all(sample["label"] == "cat" for sample in samples)


def test_manifest_mode_reads_image_list(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (32, 16), color="green").save(image_root / "a.jpg")
    Image.new("RGB", (16, 16), color="yellow").save(image_root / "b.png")
    frame = pd.DataFrame(
        [
            {"image_path": "a.jpg", "label": "x", "split": "train", "task": "generic"},
            {"image_path": "b.png", "label": "y", "split": "val", "task": "generic"},
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    frame.to_csv(manifest_path, index=False)

    samples = scan_manifest(manifest_path, image_root=image_root)

    assert len(samples) == 2
    assert samples[0]["split"] == "train"
    assert samples[1]["label"] == "y"


def test_scanner_skips_non_image_files(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    frame = pd.DataFrame(
        [
            {"image_path": "a.jpg"},
            {"image_path": "document.pdf"},
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    frame.to_csv(manifest_path, index=False)
    Image.new("RGB", (10, 10), color="white").save(image_root / "a.jpg")

    samples = scan_manifest(manifest_path, image_root=image_root)

    assert len(samples) == 1
    assert samples[0]["image_path"].name == "a.jpg"


def test_manifest_mode_supports_absolute_paths(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "nested" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (10, 10), color="white").save(image_path)
    frame = pd.DataFrame(
        [
            {"image_path": str(image_path), "label": "abs", "split": "test", "task": "generic"},
        ]
    )
    manifest_path = tmp_path / "manifest_abs.csv"
    frame.to_csv(manifest_path, index=False)

    samples = scan_manifest(manifest_path, image_root=image_root)

    assert len(samples) == 1
    assert samples[0]["image_path"] == image_path
    assert samples[0]["metadata"]["relative_path"] == "nested/a.jpg"
