from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from noctilux.image_io.loader import describe_image, load_image
from noctilux.image_io.writer import normalize_extension
from noctilux.pipeline import PipelineExecutionError


@dataclass
class ProcessingTask:
    sample_id: str
    image_path: Path
    pipeline_config: dict[str, Any]
    pipeline_name: str
    repeat_index: int
    seed: int | None
    output_path: Path
    output_config: dict[str, Any]
    sample: dict[str, Any]


@dataclass
class ProcessingResult:
    sample_id: str
    pipeline_name: str
    repeat_index: int
    seed: int | None
    output_path: Path
    success: bool
    stage: str
    error: str | None
    input_info: dict[str, Any]
    output_info: dict[str, Any]
    transform_log: list[dict[str, Any]]


def process_task(task: ProcessingTask) -> ProcessingResult:
    """Execute a single processing task in a worker process."""
    from noctilux.pipeline import AugmentPipeline

    pipeline = AugmentPipeline(
        name=task.pipeline_name,
        transforms=task.pipeline_config["transforms"],
        repeat=1,
        seed=task.seed,
    )

    try:
        image, input_info = load_image(task.image_path)
    except Exception as exc:
        return ProcessingResult(
            sample_id=task.sample_id,
            pipeline_name=task.pipeline_name,
            repeat_index=task.repeat_index,
            seed=task.seed,
            output_path=task.output_path,
            success=False,
            stage="load_image",
            error=str(exc),
            input_info={},
            output_info={},
            transform_log=[],
        )

    run_seed: int | None = None
    transform_log: list[dict[str, Any]] = []

    try:
        output_image, transform_log, run_seed = pipeline.apply(
            image=image,
            sample=task.sample,
            repeat_index=task.repeat_index,
        )
    except Exception as exc:
        if isinstance(exc, PipelineExecutionError):
            transform_log = exc.transform_logs
            run_seed = exc.run_seed
        return ProcessingResult(
            sample_id=task.sample_id,
            pipeline_name=task.pipeline_name,
            repeat_index=task.repeat_index,
            seed=run_seed if run_seed is not None else task.seed,
            output_path=task.output_path,
            success=False,
            stage="transform",
            error=str(exc),
            input_info=input_info,
            output_info={},
            transform_log=transform_log,
        )

    try:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        from noctilux.image_io.writer import save_image

        save_image(
            image=output_image,
            path=task.output_path,
            output_format=task.output_config["save_format"],
            overwrite=bool(task.output_config.get("overwrite", False)),
            jpg_quality=task.output_config["jpg_quality"],
            png_compression=task.output_config["png_compression"],
        )
        output_info = describe_image(
            output_image,
            image_format=task.output_config["save_format"].upper(),
        )
    except Exception as exc:
        return ProcessingResult(
            sample_id=task.sample_id,
            pipeline_name=task.pipeline_name,
            repeat_index=task.repeat_index,
            seed=run_seed,
            output_path=task.output_path,
            success=False,
            stage="save_image",
            error=traceback.format_exc() if not str(exc) else str(exc),
            input_info=input_info,
            output_info={},
            transform_log=transform_log,
        )

    return ProcessingResult(
        sample_id=task.sample_id,
        pipeline_name=task.pipeline_name,
        repeat_index=task.repeat_index,
        seed=run_seed,
        output_path=task.output_path,
        success=True,
        stage="",
        error=None,
        input_info=input_info,
        output_info=output_info,
        transform_log=transform_log,
    )


def pre_allocate_output_paths(
    samples: list[dict[str, Any]],
    pipelines: list[Any],
    output_config: dict[str, Any],
    include_keys: set[tuple[str, str, int]] | None = None,
    avoid_existing: bool = True,
) -> dict[tuple[str, str, int], Path]:
    """Pre-compute globally unique output paths for processing tasks.

    Existing files are avoided when output.overwrite is false and
    avoid_existing is true. Paths are also reserved in memory so multiple tasks
    in the same run cannot target the same file. When output.overwrite is true,
    existing files may be overwritten, but task-to-task collisions within the
    current run still receive deterministic __dupN suffixes.
    """
    from noctilux.saver import OutputSaver

    saver = OutputSaver(output_config)
    paths: dict[tuple[str, str, int], Path] = {}
    reserved_paths: set[Path] = set()

    for sample in samples:
        for pipeline in pipelines:
            for repeat_index in range(pipeline.repeat):
                key = (sample["sample_id"], pipeline.name, repeat_index)
                if include_keys is not None and key not in include_keys:
                    continue
                target = _build_base_output_path(sample, pipeline.name, repeat_index, saver)
                paths[key] = _reserve_output_path(
                    target,
                    saver,
                    reserved_paths,
                    avoid_existing=avoid_existing,
                )

    return paths


def _build_base_output_path(sample: dict[str, Any], pipeline_name: str, repeat_index: int, saver: Any) -> Path:
    extension = normalize_extension(saver.output_config["save_format"])
    sample_path = Path(sample["image_path"])
    relative_dir = Path()
    if saver.output_config.get("preserve_subdirs", True):
        relative_dir = saver._get_relative_dir(sample)
    filename = f"{sample_path.stem}__{pipeline_name}__{repeat_index:03d}.{extension}"
    return saver.images_root / pipeline_name / relative_dir / filename


def _reserve_output_path(
    target: Path,
    saver: Any,
    reserved_paths: set[Path],
    *,
    avoid_existing: bool = True,
) -> Path:
    overwrite = bool(saver.output_config.get("overwrite", False))
    counter = 0

    while True:
        candidate = target if counter == 0 else target.with_name(f"{target.stem}__dup{counter}{target.suffix}")
        candidate = saver._ensure_safe_path(candidate)
        disk_conflict = avoid_existing and candidate.exists() and not overwrite
        run_conflict = candidate in reserved_paths
        if not disk_conflict and not run_conflict:
            reserved_paths.add(candidate)
            return candidate
        counter += 1
