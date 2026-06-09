# Changelog

## 0.10.0

- Added annotations.overwrite_output config (default false) to control annotation output overwrite behavior.
- Annotation output file now uses atomic write (write to temp then os.replace) to prevent truncated output on failure.
- Annotation output_path preflight check: if file exists and overwrite_output is false, fails before any image processing.
- Full crop_window defensive validation: all six fields must be strict integers (bool rejected), with range and bounds checks.
- Source dimension validation: crop_window source dimensions must match current record dimensions before applying crop sync.
- All validation errors consistently raise AnnotationIntegrationError with descriptive messages (no raw KeyError/ValueError leak).
- Added 62 new tests: crop_window validation (18), precise geometry (12), output safety (11), compatibility (2), plus existing 19 tests updated.
- Kept image-only behavior and metadata schema unchanged.
- Annotation + resume/skip-existing/retry-failed/parallel still prohibited.

## 0.9.1

- Added annotation crop bbox sync for all four crop transforms: center_crop_ratio, random_crop_ratio, square_crop, and random_resized_crop.
- Crop transforms clip bboxes to the crop window, translate to crop-relative coordinates, and filter by min_area.
- random_resized_crop applies crop then resize, producing correct bbox coordinates in the final output size.
- Missing crop_window in transform log raises AnnotationIntegrationError instead of silently outputting stale bboxes.
- Updated unsupported transform error message to list all supported bbox transforms.
- Added 16 new crop bbox sync integration tests.
- Kept image-only behavior and metadata schema unchanged.

## 0.9.0

- Added crop_window metadata to transform log for all crop transforms (center_crop_ratio, random_crop_ratio, square_crop, random_resized_crop).
- crop_window records exact pixel coordinates (x, y, width, height) and source image dimensions in the original coordinate system.
- Random crop transforms produce seed-deterministic crop_window values.
- Non-crop transforms do not produce crop_window.
- Fixed unsupported transform error message wording (no longer mentions ignore under error policy).
- Fixed annotation input/output path equality check to run before COCO parse.
- Annotation crop bbox sync remains deferred; this release only exposes metadata.
- Kept image-only behavior and metadata schema unchanged.

## 0.8.1

- Added guardrails: annotation IO rejects resume, skip-existing, retry-failed, and parallel execution modes.
- Added guardrail: annotations.output_path must differ from annotations.input_path.
- Added unsupported transform warning count and unmatched sample count to run summary output.
- Added warning log for samples with no matching annotation record.
- Fixed annotation config validation: annotations.enabled=false now bypasses sub-field validation entirely.
- Added 10 new integration tests for annotation IO guardrails.
- Kept image-only behavior and metadata schema unchanged.

## 0.8.0

- Added experimental opt-in COCO bbox-only annotation IO integration.
- Added bbox sync for selected supported transforms.
- Added annotation output writing.
- Kept image-only run behavior unchanged.
- Kept annotation integration experimental.

## 0.7.5

- Fixed COCO writer annotation_id uniqueness across records.
- Avoided invalid standalone mask annotations without category linkage.
- Added optional YOLO writer bounds validation via validate_bounds parameter.
- Clarified YOLO writer in-bounds assumption in documentation.
- Added annotation writer cleanup tests.
- Kept annotation writers separate from image-only run workflow.

## 0.7.4

- Added prototype COCO and YOLO annotation writers.
- Clarified crop min_area behavior vs planned min_bbox_visibility.
- Tightened crop_record output dimensions to require positive integer pixel values.
- Added annotation writer tests and crop int-validation tests.
- Kept annotation IO separate from image-only run workflow.

## 0.7.3

- Added bbox crop geometry primitives.
- Added crop bbox clipping and filtering behavior.
- Added crop geometry tests.
- Kept annotation sync separate from image-only run workflow.

## 0.7.2

- Added bbox geometry sync primitives for resize and flip.
- Added annotation geometry tests.
- Kept annotation sync separate from image-only run workflow.

## 0.7.1

- Added annotation schema dataclasses.
- Added prototype COCO and YOLO annotation parsers.
- Kept annotation parsing separate from image-only run workflow.
- Added annotation parser tests.

## 0.7.0

- Added annotation synchronization design document (`docs/annotation_sync_design.md`).
- Documented future detection, segmentation, and keypoint annotation roadmap.
- Defined phased implementation plan: v0.7.x design/prototype, v0.8.x COCO/YOLO support.
- No code changes. Image-only behavior unchanged.

## 0.6.0

- Hardened experimental parallel execution without changing the serial default.
- Added clearer future exception handling for worker, pickle, and broken-pool failures.
- Added bounded in-flight task submission to reduce large-run future memory pressure.
- Added spawn-mode smoke coverage for parallel execution.
- Aligned save-image failure handling with `skip_broken_images` in serial and parallel modes.
- Kept metadata schema fields unchanged.

## 0.5.6

- Rejected duplicate manifest `sample_id` values before processing to avoid resume/retry key ambiguity.
- Fixed `--skip-existing` same-stem collision handling so skips are based on final reserved output paths.
- Aligned serial load-image failures with `skip_broken_images=False` so bad inputs fail the run after metadata is recorded.
- Kept metadata schema unchanged and default execution serial.

## 0.5.5

- Fixed parallel overwrite behavior so `output.overwrite: false` avoids existing files and `overwrite: true` is explicit.
- Fixed parallel output path uniqueness with deterministic in-run conflict suffixes.
- Preserved metadata during `--resume`; old manifest and transform logs remain, new results append, and summary reflects full metadata.
- Aligned parallel `skip_broken_images=False` behavior with serial-style failure handling for load and transform failures.
- Ensured `MetadataWriter` closes on parallel exceptions and writes summary for completed results.
- Added CLI validation for `--num-workers` so values must be >= 1.

## 0.5.4

- Added parallel execution stabilization tests (28 total in `tests/test_parallel.py`).
- Added determinism tests: manifest keys, output paths, seeds, summary stats consistent between serial and parallel.
- Added JSONL validity test for `transform_log.jsonl` in parallel mode.
- Added failure scenario tests: corrupt image (load_image stage), transform errors, single-failure isolation.
- Added resume / skip-existing / retry-failed boundary tests for parallel mode.
- Added experimental warning when `--num-workers > 1`: logs that parallel execution is experimental in v0.5.x.
- Added `num_workers` status line to run summary output.
- Updated `inspect-config` output: removed stale `v0.3.x` serial-only note.
- Updated README with parallel execution experimental status and `--num-workers` example.
- Updated `docs/parallel_resume_design.md` with v0.5.4 stabilization checklist.
- Serial execution (num_workers=1) unchanged. Metadata schema unchanged. Default remains serial.

## 0.5.3

- Added experimental `ProcessPoolExecutor`-based parallel execution via `--num-workers N`.
- Added `src/noctilux/worker.py` with `ProcessingTask`/`ProcessingResult` dataclasses and `process_task` worker function.
- Main process pre-allocates output paths, dispatches tasks to workers, collects results, and writes metadata.
- Workers handle image loading, pipeline transforms, and output saving; main process is the sole metadata writer.
- Added `--num-workers` CLI argument (overrides `runtime.num_workers` config).
- Seed determinism preserved: `combine_seed` produces identical seeds regardless of execution order.
- Resume (`--resume`), skip-existing (`--skip-existing`), and retry-failed (`--retry-failed`) work in parallel mode.
- Default remains `num_workers=1` (serial execution) for safety.
- Added `tests/test_parallel.py` with 14 tests covering unit and integration scenarios.
- Serial execution unchanged when `num_workers=1`.

## 0.5.2

- Added serial resume support (`--resume`): skip already-completed outputs from existing metadata.
- Added skip-existing support (`--skip-existing`): skip outputs whose target file already exists on disk.
- Added retry-failed mode (`--retry-failed`): re-process only previously failed outputs.
- Added `src/noctilux/resume.py` module with resume utility functions.
- Added resume tests and documentation.
- `--resume` and `--retry-failed` are mutually exclusive.
- Run summary now includes `skipped_count`, `resume_enabled`, `skip_existing_enabled`, `retry_failed_enabled`.
- Metadata schema unchanged. Execution remains serial.

## 0.5.1

- Refactored metadata writing into `MetadataWriter` with streaming writes.
- Preserved existing metadata file formats (manifest.csv, transform_log.jsonl, failed_images.csv, summary.csv).
- Improved internal structure for future resume and parallel execution.
- Kept execution serial. No CLI behavior changes.

## 0.5.0

- Added `docs/parallel_resume_design.md` with parallel execution and resume architecture design.
- Documented metadata-safe writer, task/result data structures, seed determinism, and resume semantics.
- Updated `PROJECT_GUIDE.md` roadmap with phased v0.5.x plan (design, metadata writer, serial resume, parallel prototype).
- No code changes. Serial execution unchanged.

## 0.4.2

- Updated GitHub Actions workflow dependencies to `actions/checkout@v6` and `actions/setup-python@v6`.
- Reduced CI Node.js 20 deprecation warnings.
- Kept test matrix and OpenCV backend job unchanged.

## 0.4.1

- Added CI coverage for optional OpenCV backend (Python 3.12, `.[dev,opencv]`).
- Unified OpenCV installation instructions and error messages across docs and source.
- Improved OpenCV backend project-quality tests.
- Kept Pillow + NumPy as the default backend.

## 0.4.0

- Added optional OpenCV backend foundation (`noctilux[opencv]`).
- Added OpenCV backend support for `resize_exact`, `resize_long_edge`, `gaussian_blur`, and `rotate`.
- Added `src/noctilux/backends/` module with PIL-OpenCV conversion utilities.
- Added `src/noctilux/exceptions.py` for structured error types.
- Added `configs/examples/opencv_backend.yaml` example config.
- Added tests for backend availability, config validation, and optional OpenCV execution.
- Pillow + NumPy remains the default backend. OpenCV is opt-in via `backend: opencv` in YAML.

## 0.3.7

- Added `docs/backend_design.md` with OpenCV backend architecture plan.
- Documented planned optional backend design: registry, configuration format, error handling, test strategy.
- Updated roadmap and agent handoff notes for v0.4.0 backend phases.
- Updated README roadmap with OpenCV backend plan.

## 0.3.6

- Added GitHub issue templates (bug report, feature request).
- Added pull request template.
- Added `SECURITY.md` with vulnerability reporting policy.
- Added `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1).
- Updated `README.md` with contributing, support status, and roadmap sections.
- Updated `docs/agent_handoff.md` and `docs/public_readiness.md`.

## 0.3.5

- Added `CONTRIBUTING.md` with setup, guidelines, and contributing workflow.
- Updated `README.md` badges and removed private-visibility note (repository is now public).
- Applied GitHub repository description and topics.
- Updated `docs/public_readiness.md` to reflect public visibility.
- Updated `docs/github_public_setup.md` to mark visibility transition as completed.
- Updated `docs/agent_handoff.md` with v0.3.5 record and public status.

## 0.3.4

- Added `docs/github_public_setup.md` with suggested repository metadata and public transition steps.
- Polished `README.md` for public-readiness with English description and structured Features section.
- Updated `docs/public_readiness.md` with sensitive information checks, GitHub metadata items, and sample image verification.
- Updated `docs/agent_handoff.md` with v0.3.4 release record and public setup reference.
- Added public-readiness consistency tests to `tests/test_project_quality.py`.
- Verified no sensitive information in committed files.

## 0.3.3

- Aligned Python version classifiers with CI coverage (removed 3.13, kept 3.10–3.12).
- Updated PROJECT_GUIDE.md roadmap to match actual release history.
- Added `docs/public_readiness.md` checklist for future repository visibility change.
- Updated `docs/agent_handoff.md` with current version, Python support scope, and public readiness reference.
- Updated `README.md` with tested Python versions and public readiness link.
- Added documentation consistency tests to `tests/test_project_quality.py`.

## 0.3.2

- Added agent handoff documentation for future maintainers and automation agents.
- Documented current project scope, release/tag state, validation commands, smoke tests, and development rules.
- Updated README contributor guidance to point agents and contributors to the handoff document.

## 0.3.1

- Fixed Python 3.10 compatibility in report timestamp generation.
- Kept the already-published `v0.3.0` tag unchanged.
- Published `v0.3.1` as the corrected report release.

## 0.3.0

- Added `noctilux report` CLI command.
- Added metadata report generation from Noctilux run outputs.
- Added Markdown and optional CSV report outputs.
- Added report tests and CI smoke test.
- Updated quickstart documentation with report generation.

## 0.2.4

- Added a tracked synthetic sample image for runnable quickstart and preview workflows.
- Updated README and docs so preview and dry-run examples can be executed directly in the repository.
- Added CI preview smoke testing with `examples/images/sample.jpg`.
- Added tests for sample image availability, readability, size, and ignore-rule consistency.
- Added a dedicated `configs/examples/quickstart_sample.yaml` for lightweight local and CI checks.

## 0.2.3

- Added `noctilux preview` as a first-class CLI command for single-image preview grids.
- Refactored preview generation into a reusable `noctilux.preview` module.
- Kept `scripts/preview_transforms.py` as a compatibility wrapper over the shared preview logic.
- Added preview CLI and module tests, including non-metadata behavior checks.
- Improved preview workflow documentation and CLI-oriented examples.

## 0.2.2

- Added GitHub Actions CI for pytest, ruff, and core CLI checks across Python 3.10, 3.11, and 3.12.
- Added ruff lint configuration and aligned developer dependencies for local quality checks.
- Improved `pyproject.toml` project metadata, optional dev dependencies, and pytest cache handling.
- Added documentation and configuration consistency tests for example configs, presets, and CI workflow coverage.
- Updated developer workflow documentation with editable install, build, lint, and CI-oriented guidance.

## 0.2.1

- Added reusable preset configs for classification, compression robustness, resize/crop suites, light degradations, and smoke testing.
- Reworked `scripts/preview_transforms.py` into a config-driven preview grid generator for single images.
- Optimized `motion_blur` with vectorized NumPy convolution and improved edge padding behavior.
- Expanded smoke tests, preset validation, and preview generation coverage.

## 0.2.0

- Added 21 common image transforms across compression, resize, crop, geometric, blur, noise, and color categories.
- Added `run --dry-run` CLI override and dry-run support for example configs without local image assets.
- Added new example configs for advanced transform groups and a balanced `full_v020` pipeline.
- Expanded tests to cover new transforms, config examples, and dry-run behavior.

## 0.1.2

- Expanded `failed_images.csv` with pipeline, repeat, seed, and stage fields.
- Added explicit serial-execution warnings and inspect output for `num_workers`.
- Improved installation guidance for offline or restricted environments.

## 0.1.1

- Hardened output path safety and non-overwrite behavior.
- Improved CLI inspect and run summaries.
- Added stronger tests for dry-run, output conflicts, repeat outputs, manifest path handling, and failure recording.
- Aligned metadata fields across manifest and transform logs.

## 0.1.0

- Initial MVP release of Noctilux.
- Added YAML-driven offline image batch processing CLI.
- Added folder and manifest input scanning.
- Added transform registry, pipeline execution, metadata export, and six core transforms.
