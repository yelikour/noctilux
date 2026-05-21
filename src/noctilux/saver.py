from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from noctilux.image_io.writer import normalize_extension, save_image


class OutputSaver:
    def __init__(self, output_config: dict[str, Any]) -> None:
        self.output_config = output_config
        self.root = Path(output_config["root"])
        self.images_root = self.root / output_config["image_dir"]
        self.metadata_root = self.root / output_config["metadata_dir"]
        self.logs_root = self.root / output_config["log_dir"]
        self.previews_root = self.root / output_config["preview_dir"]

    def prepare_directories(self) -> None:
        for path in (self.images_root, self.metadata_root, self.logs_root, self.previews_root):
            path.mkdir(parents=True, exist_ok=True)

    def build_output_path(
        self,
        sample: dict[str, Any],
        pipeline_name: str,
        repeat_index: int,
    ) -> Path:
        extension = normalize_extension(self.output_config["save_format"])
        sample_path = Path(sample["image_path"])
        relative_dir = Path()
        if self.output_config.get("preserve_subdirs", True):
            relative_dir = self._get_relative_dir(sample)

        filename = f"{sample_path.stem}__{pipeline_name}__{repeat_index:03d}.{extension}"
        target = self.images_root / pipeline_name / relative_dir / filename
        return self._ensure_safe_path(self._resolve_conflict(target))

    def save(self, image: Image.Image, target: Path) -> Path:
        save_image(
            image=image,
            path=target,
            output_format=self.output_config["save_format"],
            overwrite=bool(self.output_config.get("overwrite", False)),
            jpg_quality=self.output_config["jpg_quality"],
            png_compression=self.output_config["png_compression"],
        )
        return target

    def _get_relative_dir(self, sample: dict[str, Any]) -> Path:
        metadata = sample.get("metadata", {})
        relative_path = metadata.get("relative_path")
        if not relative_path:
            return Path()
        relative_dir = Path(relative_path).parent
        self._validate_relative_dir(relative_dir)
        return Path() if relative_dir == Path(".") else relative_dir

    def _resolve_conflict(self, target: Path) -> Path:
        overwrite = bool(self.output_config.get("overwrite", False))
        if overwrite or not target.exists():
            return target

        counter = 1
        while True:
            candidate = target.with_name(f"{target.stem}__dup{counter}{target.suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _ensure_safe_path(self, target: Path) -> Path:
        resolved_root = self.root.resolve()
        resolved_target = target.resolve()
        try:
            resolved_target.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"Output path escapes output root: {resolved_target}") from exc
        return target

    def _validate_relative_dir(self, relative_dir: Path) -> None:
        if relative_dir.is_absolute() or ".." in relative_dir.parts:
            raise ValueError(f"Sample relative path is unsafe for output preservation: {relative_dir}")
