from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "noctilux",
        "description": "",
        "seed": None,
    },
    "input": {
        "recursive": True,
        "infer_label_from_subdir": False,
        "extensions": DEFAULT_EXTENSIONS,
        "path_column": "image_path",
        "label_column": "label",
        "split_column": "split",
        "task_column": "task",
        "sample_id_column": "sample_id",
    },
    "output": {
        "root": Path("outputs/example_run"),
        "image_dir": "images",
        "metadata_dir": "metadata",
        "log_dir": "logs",
        "preview_dir": "previews",
        "preserve_subdirs": True,
        "keep_original": False,
        "overwrite": False,
        "save_format": "jpg",
        "jpg_quality": 95,
        "png_compression": 3,
    },
    "runtime": {
        "num_workers": 1,
        "skip_broken_images": True,
        "fail_fast": False,
        "show_progress": True,
        "dry_run": False,
    },
    "annotations": {
        "enabled": False,
        "format": "coco",
        "input_path": None,
        "output_path": None,
        "bbox_only": True,
        "on_unsupported_transform": "error",
    },
}

VALID_INPUT_MODES = {"folder", "manifest"}
VALID_SAVE_FORMATS = {"jpg", "jpeg", "png", "webp"}
VALID_ANNOTATION_FORMATS = {"coco"}
VALID_ANNOTATION_UNSUPPORTED_POLICIES = {"error", "ignore"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    config["__config_path__"] = config_path
    return config


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary.")

    for section in ("project", "input", "output", "runtime", "pipelines"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    if not isinstance(config["project"], dict):
        raise ValueError("Config section 'project' must be a mapping.")
    if not isinstance(config["input"], dict):
        raise ValueError("Config section 'input' must be a mapping.")
    if not isinstance(config["output"], dict):
        raise ValueError("Config section 'output' must be a mapping.")
    if not isinstance(config["runtime"], dict):
        raise ValueError("Config section 'runtime' must be a mapping.")

    seed = config.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("Config field 'seed' must be an integer or null.")

    input_cfg = config["input"]
    mode = input_cfg.get("mode")
    if mode not in VALID_INPUT_MODES:
        raise ValueError(f"Unsupported input mode: {mode!r}. Expected one of {sorted(VALID_INPUT_MODES)}.")

    if mode == "folder" and not input_cfg.get("image_root"):
        raise ValueError("input.image_root is required for folder mode.")
    if mode == "manifest":
        if not input_cfg.get("manifest_path"):
            raise ValueError("input.manifest_path is required for manifest mode.")
        if not input_cfg.get("path_column"):
            raise ValueError("input.path_column is required for manifest mode.")

    extensions = input_cfg.get("extensions", [])
    if not isinstance(extensions, list) or not all(isinstance(item, str) for item in extensions):
        raise ValueError("input.extensions must be a list of strings.")

    output_cfg = config["output"]
    if not output_cfg.get("root"):
        raise ValueError("output.root is required.")
    for field in ("image_dir", "metadata_dir", "log_dir", "preview_dir"):
        _validate_relative_component(output_cfg.get(field), f"output.{field}")
    if output_cfg.get("save_format") not in VALID_SAVE_FORMATS:
        raise ValueError(
            f"Unsupported output.save_format: {output_cfg.get('save_format')!r}. "
            f"Expected one of {sorted(VALID_SAVE_FORMATS)}."
        )

    jpg_quality = output_cfg.get("jpg_quality")
    if not isinstance(jpg_quality, int) or not 1 <= jpg_quality <= 100:
        raise ValueError("output.jpg_quality must be an integer between 1 and 100.")

    png_compression = output_cfg.get("png_compression")
    if not isinstance(png_compression, int) or not 0 <= png_compression <= 9:
        raise ValueError("output.png_compression must be an integer between 0 and 9.")

    runtime_cfg = config["runtime"]
    if not isinstance(runtime_cfg.get("num_workers"), int) or runtime_cfg["num_workers"] < 1:
        raise ValueError("runtime.num_workers must be an integer >= 1.")
    for field in ("skip_broken_images", "fail_fast", "show_progress", "dry_run"):
        if not isinstance(runtime_cfg.get(field), bool):
            raise ValueError(f"runtime.{field} must be a boolean.")

    _validate_annotations_config(config.get("annotations", {}))

    pipelines = config["pipelines"]
    if not isinstance(pipelines, list) or not pipelines:
        raise ValueError("pipelines must be a non-empty list.")

    seen_names: set[str] = set()
    for index, pipeline in enumerate(pipelines):
        if not isinstance(pipeline, dict):
            raise ValueError(f"Pipeline #{index} must be a mapping.")
        name = pipeline.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"Pipeline #{index} is missing a valid 'name'.")
        _validate_relative_component(name, f"pipelines[{index}].name")
        if name in seen_names:
            raise ValueError(f"Duplicate pipeline name: {name}")
        seen_names.add(name)

        repeat = pipeline.get("repeat", 1)
        if not isinstance(repeat, int) or repeat < 1:
            raise ValueError(f"Pipeline '{name}' repeat must be an integer >= 1.")

        transforms = pipeline.get("transforms")
        if not isinstance(transforms, list) or not transforms:
            raise ValueError(f"Pipeline '{name}' must contain at least one transform.")

        for t_index, transform in enumerate(transforms):
            if not isinstance(transform, dict):
                raise ValueError(f"Transform #{t_index} in pipeline '{name}' must be a mapping.")
            transform_name = transform.get("name")
            if not transform_name or not isinstance(transform_name, str):
                raise ValueError(f"Transform #{t_index} in pipeline '{name}' is missing a valid 'name'.")
            p_value = transform.get("p", 1.0)
            if not isinstance(p_value, (int, float)) or not 0.0 <= float(p_value) <= 1.0:
                raise ValueError(
                    f"Transform '{transform_name}' in pipeline '{name}' has invalid probability p={p_value!r}."
                )
            params = transform.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(
                    f"Transform '{transform_name}' in pipeline '{name}' must define params as a mapping."
                )


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary.")

    raw = deepcopy(config)
    resolved = deepcopy(DEFAULT_CONFIG)

    for section in ("project", "input", "output", "runtime", "annotations"):
        section_data = raw.pop(section, {})
        if section_data is None:
            section_data = {}
        if not isinstance(section_data, dict):
            raise ValueError(f"Config section '{section}' must be a mapping.")
        resolved[section].update(section_data)

    resolved["pipelines"] = raw.pop("pipelines", [])
    if "seed" in raw and resolved["project"].get("seed") is None:
        resolved["project"]["seed"] = raw["seed"]

    resolved["seed"] = resolved["project"].get("seed")
    resolved["runtime"]["overwrite"] = bool(resolved["output"]["overwrite"])

    input_cfg = resolved["input"]
    output_cfg = resolved["output"]
    config_path = config.get("__config_path__")
    resolved["__config_path__"] = config_path

    if input_cfg.get("image_root") is not None:
        input_cfg["image_root"] = Path(input_cfg["image_root"]).expanduser()
    if input_cfg.get("manifest_path") is not None:
        input_cfg["manifest_path"] = Path(input_cfg["manifest_path"]).expanduser()

    output_cfg["root"] = Path(output_cfg["root"]).expanduser()

    annotations_cfg = resolved["annotations"]
    if annotations_cfg.get("input_path") is not None:
        annotations_cfg["input_path"] = Path(annotations_cfg["input_path"]).expanduser()
    if annotations_cfg.get("output_path") is not None:
        annotations_cfg["output_path"] = Path(annotations_cfg["output_path"]).expanduser()

    return resolved


def _validate_annotations_config(annotations_cfg: Any) -> None:
    if not isinstance(annotations_cfg, dict):
        raise ValueError("Config section 'annotations' must be a mapping.")

    enabled = annotations_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("annotations.enabled must be a boolean.")

    bbox_only = annotations_cfg.get("bbox_only", True)
    if not isinstance(bbox_only, bool):
        raise ValueError("annotations.bbox_only must be a boolean.")

    on_unsupported = annotations_cfg.get("on_unsupported_transform", "error")
    if on_unsupported not in VALID_ANNOTATION_UNSUPPORTED_POLICIES:
        raise ValueError(
            "annotations.on_unsupported_transform must be one of "
            f"{sorted(VALID_ANNOTATION_UNSUPPORTED_POLICIES)}."
        )

    if not enabled:
        return

    annotation_format = annotations_cfg.get("format", "coco")
    if annotation_format not in VALID_ANNOTATION_FORMATS:
        raise ValueError(
            f"Unsupported annotations.format: {annotation_format!r}. "
            f"Expected one of {sorted(VALID_ANNOTATION_FORMATS)}."
        )
    if not bbox_only:
        raise ValueError("annotations.bbox_only must be true for v0.8.0 bbox-only integration.")

    input_path = annotations_cfg.get("input_path")
    if input_path is None:
        raise ValueError("annotations.input_path is required when annotations.enabled is true.")
    if not Path(input_path).exists():
        raise FileNotFoundError(f"annotations.input_path does not exist: {input_path}")


def _validate_relative_component(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or len(path.parts) != 1:
        raise ValueError(f"{field_name} must be a single safe path component, got: {value!r}.")
