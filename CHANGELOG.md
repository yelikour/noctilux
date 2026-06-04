# Changelog

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
