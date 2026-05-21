# Changelog

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
