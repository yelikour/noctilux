from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from noctilux.config import load_config, resolve_config, validate_config
from noctilux.image_io.loader import describe_image, load_image
from noctilux.metadata import MetadataWriter
from noctilux.pipeline import PipelineExecutionError, build_pipelines
from noctilux.preview import add_preview_arguments, create_preview_grid
from noctilux.registry import list_transforms
from noctilux.report import generate_report
from noctilux.resume import (
    build_processing_key,
    check_output_exists,
    load_failed_keys,
    load_success_keys,
    validate_resume_args,
)
from noctilux.saver import OutputSaver
from noctilux.scanner import build_manifest_from_folder, scan_inputs

LOGGER = logging.getLogger("noctilux")


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="noctilux", description="Offline image batch processing toolkit.")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect-config", help="Validate and summarize a config file.")
    inspect_parser.add_argument("--config", required=True, help="Path to YAML config.")
    inspect_parser.set_defaults(func=inspect_config_command)

    list_parser = subparsers.add_parser("list-transforms", help="List registered transforms.")
    list_parser.set_defaults(func=list_transforms_command)

    run_parser = subparsers.add_parser("run", help="Run an offline processing job.")
    run_parser.add_argument("--config", required=True, help="Path to YAML config.")
    run_parser.add_argument("--dry-run", action="store_true", help="Override config and run without writing outputs.")
    run_parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-completed outputs from existing metadata.",
    )
    run_parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip outputs whose target file already exists on disk.",
    )
    run_parser.add_argument(
        "--retry-failed", action="store_true",
        help="Re-process only previously failed outputs.",
    )
    run_parser.set_defaults(func=run_command)

    preview_parser = subparsers.add_parser("preview", help="Generate a preview grid for a single image.")
    add_preview_arguments(preview_parser)
    preview_parser.set_defaults(func=preview_command)

    report_parser = subparsers.add_parser("report", help="Generate a Markdown report from metadata.")
    report_parser.add_argument("--metadata", required=True, help="Metadata directory from a noctilux run.")
    report_parser.add_argument("--output", required=True, help="Markdown report output path.")
    report_parser.add_argument("--csv-output", default=None, help="Optional CSV summary output path.")
    report_parser.add_argument("--overwrite", action="store_true", help="Allow overwriting report output files.")
    report_parser.set_defaults(func=report_command)

    manifest_parser = subparsers.add_parser("make-manifest", help="Generate a CSV manifest from a folder.")
    manifest_parser.add_argument("--image-root", required=True, help="Input image root.")
    manifest_parser.add_argument("--output", required=True, help="Output CSV path.")
    manifest_parser.add_argument(
        "--infer-label-from-subdir",
        action="store_true",
        help="Use the first subdirectory name as label.",
    )
    manifest_parser.add_argument("--no-recursive", action="store_true", help="Disable recursive scanning.")
    manifest_parser.set_defaults(func=make_manifest_command)

    return parser


def inspect_config_command(args: argparse.Namespace) -> int:
    config = _load_and_validate(args.config)
    pipelines = config["pipelines"]
    transform_count = sum(len(pipeline["transforms"]) for pipeline in pipelines if pipeline.get("enabled", True))
    print(f"project: {config['project']['name']}")
    print(f"input_mode: {config['input']['mode']}")
    if config["input"]["mode"] == "folder":
        print(f"image_root: {config['input']['image_root']}")
    else:
        print(f"manifest_path: {config['input']['manifest_path']}")
        if config["input"].get("image_root") is not None:
            print(f"image_root: {config['input']['image_root']}")
    print(f"output_root: {config['output']['root']}")
    print(f"pipelines: {len(pipelines)}")
    print(f"transforms: {transform_count}")
    print(f"dry_run: {config['runtime']['dry_run']}")
    print(f"overwrite: {config['output']['overwrite']}")
    print(f"num_workers: {config['runtime']['num_workers']} (serial execution in v0.3.x)")
    return 0


def list_transforms_command(args: argparse.Namespace) -> int:
    for name in list_transforms():
        print(name)
    return 0


def preview_command(args: argparse.Namespace) -> int:
    output_path = create_preview_grid(
        config_path=Path(args.config),
        image_path=Path(args.image),
        output_path=Path(args.output),
        max_pipelines=args.max_pipelines,
        seed=args.seed,
    )
    print(output_path)
    return 0


def report_command(args: argparse.Namespace) -> int:
    output_path = generate_report(
        metadata_dir=Path(args.metadata),
        output_path=Path(args.output),
        csv_output_path=Path(args.csv_output) if args.csv_output else None,
        overwrite=args.overwrite,
    )
    print(output_path)
    return 0


def run_command(args: argparse.Namespace) -> int:
    do_resume = getattr(args, "resume", False)
    do_skip_existing = getattr(args, "skip_existing", False)
    do_retry_failed = getattr(args, "retry_failed", False)
    validate_resume_args(do_resume, do_retry_failed)

    config = _load_and_validate(args.config)
    if getattr(args, "dry_run", False):
        config["runtime"]["dry_run"] = True
    samples: list[dict[str, Any]]
    try:
        samples = scan_inputs(config)
    except (FileNotFoundError, NotADirectoryError) as exc:
        if config["runtime"]["dry_run"]:
            LOGGER.warning("Dry run could not scan inputs: %s", exc)
            samples = []
        else:
            raise
    pipelines = build_pipelines(config)

    if config["runtime"]["num_workers"] > 1:
        LOGGER.warning("num_workers is currently reserved and execution is still serial in v0.3.x.")

    if config["runtime"]["dry_run"]:
        planned_outputs = sum(pipeline.repeat for pipeline in pipelines) * len(samples)
        print(f"total_samples: {len(samples)}")
        print(f"total_outputs: {planned_outputs}")
        print("success_count: 0")
        print("failed_count: 0")
        print(f"metadata_path: {Path(config['output']['root']) / config['output']['metadata_dir']}")
        LOGGER.info(
            "Dry run complete. samples=%d pipelines=%d planned_outputs=%d",
            len(samples),
            len(pipelines),
            planned_outputs,
        )
        return 0

    saver = OutputSaver(config["output"])
    saver.prepare_directories()

    skip_keys: set[str] = set()
    if do_resume:
        skip_keys = load_success_keys(saver.metadata_root)
        LOGGER.info("Resume mode: %d completed outputs will be skipped.", len(skip_keys))
    elif do_retry_failed:
        skip_keys = load_failed_keys(saver.metadata_root)
        LOGGER.info("Retry-failed mode: %d failed outputs will be retried.", len(skip_keys))

    writer = MetadataWriter(saver.metadata_root)
    skipped_count = 0

    iterable = samples
    if config["runtime"].get("show_progress", True):
        iterable = tqdm(samples, desc="Processing", unit="image")

    for sample in iterable:
        sample_path = Path(sample["image_path"])
        try:
            image, input_info = load_image(sample_path)
        except Exception as exc:
            _handle_load_failure(writer, sample, pipelines, str(exc), skip_keys, do_retry_failed)
            if config["runtime"]["fail_fast"]:
                raise
            continue

        for pipeline in pipelines:
            for repeat_index in range(pipeline.repeat):
                key = build_processing_key(sample["sample_id"], pipeline.name, repeat_index)

                if do_resume and key in skip_keys:
                    skipped_count += 1
                    continue

                if do_retry_failed and key not in skip_keys:
                    skipped_count += 1
                    continue

                if do_skip_existing and not do_retry_failed:
                    if check_output_exists(sample, pipeline.name, repeat_index, saver):
                        skipped_count += 1
                        continue

                run_seed = pipeline._resolve_run_seed(sample=sample, repeat_index=repeat_index)
                try:
                    output_image, transform_log, run_seed = pipeline.apply(
                        image=image,
                        sample=sample,
                        repeat_index=repeat_index,
                    )
                except Exception as exc:
                    transform_logs = []
                    if isinstance(exc, PipelineExecutionError):
                        transform_logs = exc.transform_logs
                        run_seed = exc.run_seed
                    _record_failure(
                        writer=writer,
                        sample=sample,
                        pipeline_name=pipeline.name,
                        repeat_index=repeat_index,
                        seed=run_seed,
                        input_info=input_info,
                        transforms=transform_logs,
                        stage="transform",
                        error=str(exc),
                    )
                    if config["runtime"]["fail_fast"]:
                        raise
                    if not config["runtime"]["skip_broken_images"]:
                        raise
                    continue

                try:
                    output_path = saver.build_output_path(sample, pipeline.name, repeat_index)
                    saver.save(output_image, output_path)
                    output_info = describe_image(
                        output_image,
                        image_format=config["output"]["save_format"].upper(),
                    )
                except Exception as exc:
                    _record_failure(
                        writer=writer,
                        sample=sample,
                        pipeline_name=pipeline.name,
                        repeat_index=repeat_index,
                        seed=run_seed,
                        input_info=input_info,
                        transforms=transform_log,
                        stage="save_image",
                        error=str(exc),
                    )
                    if config["runtime"]["fail_fast"]:
                        raise
                    if not config["runtime"]["skip_broken_images"]:
                        raise
                    continue

                writer.write_success(
                    manifest_row=_build_manifest_record(
                        sample=sample,
                        pipeline_name=pipeline.name,
                        repeat_index=repeat_index,
                        input_info=input_info,
                        output_info=output_info,
                        output_path=output_path,
                        seed=run_seed,
                        success=True,
                        error="",
                    ),
                    transform_log_row=_build_transform_record(
                        sample=sample,
                        pipeline_name=pipeline.name,
                        repeat_index=repeat_index,
                        seed=run_seed,
                        transforms=transform_log,
                        input_info=input_info,
                        output_info=output_info,
                        output_path=output_path,
                        success=True,
                        error=None,
                    ),
                )

    writer.close()
    print(f"total_samples: {len(samples)}")
    print(f"total_outputs: {writer.total_count}")
    print(f"success_count: {writer.success_count}")
    print(f"failed_count: {writer.failed_count}")
    print(f"skipped_count: {skipped_count}")
    print(f"resume_enabled: {do_resume}")
    print(f"skip_existing_enabled: {do_skip_existing}")
    print(f"retry_failed_enabled: {do_retry_failed}")
    print(f"metadata_path: {saver.metadata_root}")
    LOGGER.info("Completed run. metadata=%s", saver.metadata_root)
    return 0


def make_manifest_command(args: argparse.Namespace) -> int:
    frame = build_manifest_from_folder(
        image_root=args.image_root,
        infer_label_from_subdir=args.infer_label_from_subdir,
        recursive=not args.no_recursive,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Wrote {len(frame)} rows to {output_path}")
    return 0


def _load_and_validate(path: str | Path) -> dict[str, Any]:
    config = resolve_config(load_config(path))
    validate_config(config)
    return config


def _build_manifest_record(
    sample: dict[str, Any],
    pipeline_name: str,
    repeat_index: int,
    input_info: dict[str, Any],
    output_info: dict[str, Any],
    output_path: Path | None,
    seed: int | None,
    success: bool,
    error: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "original_path": str(sample["image_path"]),
        "output_path": str(output_path) if output_path is not None else "",
        "pipeline_name": pipeline_name,
        "repeat_index": repeat_index,
        "input_width": input_info.get("width"),
        "input_height": input_info.get("height"),
        "output_width": output_info.get("width"),
        "output_height": output_info.get("height"),
        "input_format": input_info.get("format"),
        "output_format": output_info.get("format"),
        "success": bool(success),
        "error": error,
        "seed": seed,
        "label": sample.get("label", ""),
        "split": sample.get("split", "unknown"),
        "task": sample.get("task", "generic"),
    }


def _build_transform_record(
    sample: dict[str, Any],
    pipeline_name: str,
    repeat_index: int,
    seed: int | None,
    transforms: list[dict[str, Any]],
    input_info: dict[str, Any],
    output_info: dict[str, Any],
    output_path: Path | None,
    success: bool,
    error: str | None,
) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "original_path": str(sample["image_path"]),
        "output_path": str(output_path) if output_path is not None else "",
        "pipeline_name": pipeline_name,
        "repeat_index": repeat_index,
        "seed": seed,
        "label": sample.get("label", ""),
        "split": sample.get("split", "unknown"),
        "task": sample.get("task", "generic"),
        "transforms": transforms,
        "input_info": input_info,
        "output_info": output_info,
        "success": bool(success),
        "error": error,
    }


def _handle_load_failure(
    writer: MetadataWriter,
    sample: dict[str, Any],
    pipelines: list[Any],
    error: str,
    skip_keys: set[str],
    retry_only: bool,
) -> None:
    for pipeline in pipelines:
        for repeat_index in range(pipeline.repeat):
            if retry_only:
                from noctilux.resume import build_processing_key

                key = build_processing_key(sample["sample_id"], pipeline.name, repeat_index)
                if key not in skip_keys:
                    continue
            run_seed = pipeline._resolve_run_seed(sample=sample, repeat_index=repeat_index)
            _record_failure(
                writer=writer,
                sample=sample,
                pipeline_name=pipeline.name,
                repeat_index=repeat_index,
                seed=run_seed,
                input_info={},
                transforms=[],
                stage="load_image",
                error=error,
            )


def _record_failure(
    writer: MetadataWriter,
    sample: dict[str, Any],
    pipeline_name: str,
    repeat_index: int,
    seed: int | None,
    input_info: dict[str, Any],
    transforms: list[dict[str, Any]],
    stage: str,
    error: str,
) -> None:
    writer.write_failure(
        manifest_row=_build_manifest_record(
            sample=sample,
            pipeline_name=pipeline_name,
            repeat_index=repeat_index,
            input_info=input_info,
            output_info={},
            output_path=None,
            seed=seed,
            success=False,
            error=error,
        ),
        transform_log_row=_build_transform_record(
            sample=sample,
            pipeline_name=pipeline_name,
            repeat_index=repeat_index,
            seed=seed,
            transforms=transforms,
            input_info=input_info,
            output_info={},
            output_path=None,
            success=False,
            error=error,
        ),
        failed_row={
            "sample_id": sample["sample_id"],
            "image_path": str(sample["image_path"]),
            "pipeline_name": pipeline_name,
            "repeat_index": repeat_index,
            "seed": seed,
            "stage": stage,
            "error": error,
        },
    )


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


if __name__ == "__main__":
    sys.exit(main())
