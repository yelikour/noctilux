# Changelog

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
