# Annotation Synchronization Design

This document describes the planned annotation synchronization architecture for Noctilux.

## Current State

Noctilux v0.7.4 still processes image-level augmentation only. During `noctilux run`, transforms like resize, crop, or flip do not update object detection bounding boxes, segmentation masks, or keypoint coordinates. This is the correct behavior for image classification, where annotations are simple labels unaffected by geometric transforms.

v0.7.4 adds prototype COCO and YOLO annotation writers on top of the v0.7.1 schema/parser, v0.7.2 resize/flip helpers, and v0.7.3 crop helpers. Writers are not wired into `noctilux run`, do not change pipeline or metadata behavior, and annotation sync is not yet supported.

Image-only behavior remains unchanged and will remain the default. Annotation synchronization is a future opt-in feature for task-aware pipelines.

## v0.7.1 Prototype Status

- Completed: lightweight annotation schema dataclasses and simple dictionary round trips.
- Completed: minimal read-only COCO and YOLO parser prototypes.
- Not implemented: annotation writers, transform synchronization, config schema, metadata changes, or run integration.
- The COCO/YOLO readers are prototypes, not complete dataset-format or synchronization support.

## v0.7.2 Geometry Primitive Status

- Completed: immutable bbox resize helpers for boxes and records.
- Completed: immutable horizontal and vertical bbox flip helpers for boxes and records.
- Not implemented: crop, rotate, mask/polygon, or keypoint synchronization.
- Not integrated: `noctilux run`, transforms, CLI, config, writers, and metadata remain unchanged.

## v0.7.3 Crop Primitive Status

- Completed: immutable bbox crop helpers for boxes and records.
- Crop intersections are clipped to the crop window and translated to the crop-relative coordinate system.
- Fully removed boxes and intersections smaller than `min_area` are filtered out.
- Cropped bbox `area` is updated to the new intersection area.
- Not implemented: rotate, mask/polygon, or keypoint synchronization.
- Not integrated: `noctilux run`, transforms, CLI, config, writers, and metadata remain unchanged.

## Why Annotation Sync Matters

Offline augmentation is commonly used for object detection, instance segmentation, and keypoint estimation tasks. Without annotation synchronization:

- Bounding boxes become misaligned after crop or resize.
- Segmentation masks no longer match the transformed image.
- Keypoint coordinates point to wrong locations after geometric transforms.
- The augmented dataset is unusable for training detection or segmentation models without manual re-annotation.

Supporting annotation sync makes Noctilux useful for a broader range of computer vision tasks beyond classification.

## Supported Task Scope

| Task | Annotation Sync Required | Priority |
|------|--------------------------|----------|
| Classification | No (labels are invariant) | N/A |
| Object Detection | Yes (bbox coordinates) | High |
| Instance Segmentation | Yes (mask / polygon coordinates) | Medium |
| Semantic Segmentation | Yes (mask pixel transform) | Medium |
| Keypoint Detection | Yes (keypoint coordinates) | Low |
| Multimodal / OCR | Separate design needed | Not planned |

## Annotation Input Format Candidates

### COCO JSON

The most widely used format for detection and segmentation. A single JSON file contains all images, categories, and annotations. Bounding boxes are `[x, y, width, height]`. Segmentation is polygon arrays or RLE.

Advantages:
- Standard format, broad tooling support.
- Single-file layout simplifies path management.
- Supports bbox, polygon, and RLE masks natively.

Challenges:
- Loading and writing the full COCO JSON can be memory-intensive for large datasets.
- Image-level filtering requires building an image-to-annotation index.

### YOLO TXT

One `.txt` file per image, each line is `class_id center_x center_y width height` (normalized coordinates).

Advantages:
- Simple, one-file-per-image mapping.
- Normalized coordinates are naturally resolution-independent.

Challenges:
- Only supports bounding boxes, not masks or keypoints.
- Requires a separate `classes.txt` file.
- Normalized coordinates need denormalization before crop operations.

### Pascal VOC XML

One `.xml` file per image with `<object>` elements containing `<bndbox>` coordinates in absolute pixels.

Advantages:
- Explicit, well-documented schema.
- Supports additional fields like `difficult`, `truncated`, `pose`.

Challenges:
- Verbose XML parsing.
- Absolute coordinates require recalculation on resize/crop.

### Custom Manifest Columns

Noctilux manifest CSV with additional columns for inline annotations (e.g., `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`).

Advantages:
- No external annotation files needed.
- Integrates directly with existing manifest workflows.

Challenges:
- Does not scale to multi-object images.
- Cannot represent masks or polygons inline.

### Recommendation

Start with COCO JSON as the primary format, add YOLO TXT and VOC XML in later phases. Custom manifest columns can be supported for single-object cases.

## Core Design Principles

1. **Opt-in by default**: Annotation sync is only active when explicitly configured. Image-only pipelines behave exactly as before.
2. **Image-only pipeline invariance**: Existing configs without annotation settings must produce identical output to v0.6.0.
3. **Geometry-only transforms sync**: Only geometric transforms (resize, crop, flip, rotate) update annotation coordinates. Photometric transforms (color, blur, noise, compression) leave annotations unchanged.
4. **Traceability**: Output annotations must reference their source, and metadata must record annotation input/output paths.
5. **Metadata compatibility**: No existing metadata fields may be renamed or removed. New fields are added as optional columns.
6. **Fail clearly**: If an annotation-incompatible transform is used in an annotation-aware pipeline, the system must warn or error, not silently corrupt annotations.

## Transform Classification

### Geometry-safe (require annotation sync)

These transforms change pixel positions and must update annotations:

- `resize_long_edge`, `resize_short_edge`, `resize_exact`
- `center_crop_ratio`, `random_crop_ratio`, `random_resized_crop`, `square_crop`
- `horizontal_flip`, `vertical_flip`
- `rotate`

### Photometric-only (no annotation change needed)

These transforms modify pixel values but not positions:

- `jpeg_compression`, `webp_compression`, `png_resave`, `double_jpeg_compression`
- `gaussian_blur`, `median_blur`, `motion_blur`
- `gaussian_noise`, `poisson_noise`, `salt_pepper_noise`
- `brightness_contrast`, `gamma_correction`, `saturation_hue`, `grayscale`, `sharpen`, `posterize`

### Annotation-incompatible or requires explicit support

- `downscale_upscale`: may introduce aliasing artifacts in mask edges; needs careful handling.
- `random_resized_crop`: bbox clipping semantics need clear specification.
- `rotate` with expand: changes canvas size; bbox must account for padding.
- Any future transform that changes image geometry in non-trivial ways.

## Data Structure Prototype (v0.7.1)

The internal `noctilux.annotations` module provides:

- `BoundingBox`: COCO-style pixel coordinates (`x`, `y`, `width`, `height`) plus category and optional annotation fields.
- `Keypoint`: pixel coordinates plus visibility.
- `MaskRef`: an optional mask path, raw segmentation payload, and optional size.
- `AnnotationRecord`: one image's dimensions, paths, boxes, masks, keypoints, and optional raw source data.

Each schema dataclass supports a simple `to_dict()` / `from_dict()` round trip. v0.7.3 uses these structures in
standalone bbox resize/flip/crop helpers. `AnnotationTransformResult`, transform binding, and run integration remain future work.

## Pipeline Design

### Image Transform and Annotation Transform Binding

Each registered transform declares its annotation capability:

```python
class BaseTransform:
    name: str = "base_transform"

    def __call__(self, image, context=None):
        raise NotImplementedError

    def supports_annotations(self) -> bool:
        """Whether this transform can synchronize annotations."""
        return False

    def transform_annotations(
        self,
        annotations: AnnotationRecord,
        image_before: Image.Image,
        image_after: Image.Image,
        params: dict,
    ) -> AnnotationTransformResult:
        """Transform annotations to match the transformed image."""
        raise NotImplementedError
```

Geometry transforms override both `supports_annotations` (return `True`) and `transform_annotations`. Photometric transforms use the default (no annotation change).

### Transform Annotation Declaration

Transforms that support annotations implement `transform_annotations`. The base implementation returns the annotations unchanged with `applied=False`.

For geometry transforms, the implementation applies the same geometric operation to annotation coordinates:

```python
@register_transform("resize_long_edge")
class ResizeLongEdge(BaseTransform):
    def supports_annotations(self) -> bool:
        return True

    def transform_annotations(self, annotations, image_before, image_after, params):
        scale_x = image_after.width / image_before.width
        scale_y = image_after.height / image_before.height
        new_bboxes = [
            BoundingBox(
                x_min=bbox.x_min * scale_x,
                y_min=bbox.y_min * scale_y,
                x_max=bbox.x_max * scale_x,
                y_max=bbox.y_max * scale_y,
                label=bbox.label,
            )
            for bbox in annotations.bboxes
        ]
        return AnnotationTransformResult(
            transform_name=self.name,
            applied=True,
            bboxes=new_bboxes,
            masks=annotations.masks,
            keypoints=annotations.keypoints,
            removed_bboxes=[],
        )
```

### Unsupported Transform Handling

When an annotation-aware pipeline encounters a transform that does not declare annotation support:

1. If the transform is photometric-only (no geometry change), annotations pass through unchanged. This is the default behavior.
2. If the transform changes geometry but has not implemented `transform_annotations`, the system must either:
   - Raise a warning and skip annotation sync for that transform (lenient mode).
   - Raise an error and halt the pipeline (strict mode, default for annotation-aware configs).

This prevents silent annotation corruption from incomplete implementations.

### Config Schema (Draft)

```yaml
annotations:
  enabled: true
  format: coco          # coco | yolo | voc | manifest
  source: annotations/instances_train2017.json
  strict: true          # error on unsupported geometry transforms

  # For YOLO format:
  # classes_file: annotations/classes.txt

  # For VOC format:
  # annotations_dir: annotations/voc

pipelines:
  - name: detection_augment
    transforms:
      - name: horizontal_flip
        p: 0.5
      - name: resize_long_edge
        params:
          long_edge: 640
      - name: brightness_contrast
        p: 0.3
        params:
          brightness:
            type: uniform
            min: -0.1
            max: 0.1
```

When `annotations` is absent or `annotations.enabled` is `false`, the pipeline runs in image-only mode with zero overhead.

## Output Design

```
output_root/
├── images/
│   └── pipeline_name/
│       ├── img001__pipeline__000.jpg
│       └── img002__pipeline__000.jpg
├── annotations/
│   └── pipeline_name/
│       ├── img001__pipeline__000.json   # per-image annotation
│       └── img002__pipeline__000.json
├── metadata/
│   ├── manifest.csv
│   ├── transform_log.jsonl
│   ├── failed_images.csv
│   └── summary.csv
```

Key rules:
- Output annotations are stored per-image alongside output images.
- The annotation output format matches the input format unless overridden.
- `manifest.csv` gains optional columns: `annotation_input_path`, `annotation_output_path`.
- `transform_log.jsonl` gains an `annotations` field recording the annotation transform log.

## Error Handling

### Invalid Bounding Box

- Bbox with `x_min >= x_max` or `y_min >= y_max` is logged as a warning and skipped.
- Zero-area bboxes after transform are removed and recorded in `removed_bboxes`.

### Bbox Eliminated by Crop

- Bboxes with zero overlap area after crop are removed.
- Bboxes with partial overlap are clipped to the new image bounds.
- The current crop primitive (`crop_box` / `crop_record` in v0.7.3) uses `min_area` as an absolute area threshold: intersections with area below `min_area` are removed. This threshold defaults to 1.0 pixel².
- `min_bbox_visibility` (a ratio-based threshold, e.g. 0.1 = 10% visible) is a planned future option for annotation-aware configs. It is not implemented yet. When added, it would complement `min_area`:

```yaml
annotations:
  min_bbox_visibility: 0.1  # future option: remove bboxes with < 10% visible area after crop
```

### Mask File Missing

- If a referenced mask file does not exist, the record is logged as a warning and the mask is dropped from the output annotations.
- The image is still processed; only the mask is skipped.

### Unsupported Transform

- In strict mode (default for annotation-aware configs), an unsupported geometry transform raises an error.
- In lenient mode, a warning is logged and annotations pass through unchanged.
- Photometric-only transforms never trigger this check.

## Test Strategy

### Unit Tests

- Each geometry transform with annotation support: resize bbox, flip bbox, crop bbox with clipping, rotate bbox.
- Photometric transforms do not change annotation coordinates.
- Annotation passthrough when no `annotations` config is present.
- Invalid bbox handling (zero area, negative dimensions).
- Crop bbox elimination and partial clipping.
- Keypoint coordinate transformation.

### Integration Tests

- Full pipeline with annotation sync: input COCO JSON, output per-image annotation files.
- Round-trip: original annotations and synced annotations are consistent with the transformed image dimensions.
- Manifest includes `annotation_input_path` and `annotation_output_path` columns.
- Transform log includes annotation transform details.

### Compatibility Tests

- Image-only pipeline output is identical with and without the annotation sync module present.
- Existing metadata schema fields are unchanged.
- New annotation columns are optional and absent in image-only runs.

## Implementation Phases

### v0.7.0 — Design (this document)

- Document annotation synchronization architecture.
- Define data structures, transform classification, config schema, and error handling.
- No code changes. Image-only behavior unchanged.

### v0.7.1 — Annotation Schema and Parser Prototype (completed)

- Implemented `AnnotationRecord`, `BoundingBox`, `MaskRef`, and `Keypoint` dataclasses.
- Added `BaseAnnotationParser` plus minimal read-only COCO-like JSON and YOLO TXT parser prototypes.
- Added parser/schema unit tests and image-only run smoke coverage.
- Did not add annotation writers, transform synchronization, config changes, metadata changes, or CLI integration.
- Parser prototypes remain separate from `noctilux run`; image-only behavior is unchanged.

### v0.7.2 — Bbox Sync Primitives for Resize and Flip (completed)

- Added standalone bbox and record helpers for resize, horizontal flip, and vertical flip.
- Helpers return new objects and preserve annotation metadata fields.
- Added unit tests for geometry formulas, validation, field preservation, and image-only smoke behavior.
- Did not implement crop, rotate, mask/polygon, or keypoint synchronization.
- Did not integrate annotation sync into transforms, pipeline execution, CLI, config, or metadata.

### v0.7.3 — Crop Bbox Handling Primitives (completed)

- Added standalone `crop_box` and `crop_record` helpers.
- Added crop-relative coordinate translation, clipping, and `min_area` filtering.
- Cropped bbox `area` is updated to the new intersection area.
- Added tests for full containment, partial clipping, filtering, immutability, validation, and field preservation.
- Did not implement rotate, mask/polygon, or keypoint synchronization.
- Did not integrate annotation sync into transforms, pipeline execution, CLI, config, or metadata.

### v0.7.4 — Annotation Writer Prototype (completed)

- Added prototype COCO-like JSON writer (`CocoAnnotationWriter`) and YOLO TXT writer (`YoloAnnotationWriter`).
- COCO writer produces `images`, `annotations`, and `categories` with `[x, y, width, height]` bbox format.
- YOLO writer produces normalized `class_id cx cy w h` lines; requires `record.width` and `record.height`.
- Writers are standalone prototypes, not wired into `noctilux run` or CLI.
- Tightened `crop_record` to require positive integer pixel dimensions for `crop_width`/`crop_height`.
- Clarified `min_area` (current) vs `min_bbox_visibility` (planned future option) documentation.
- Image-only behavior unchanged.

### v0.8.0 — COCO and YOLO Minimal Sync Support

- Harden COCO/YOLO readers and add annotation writers.
- Add VOC XML parser (read-only for initial support).
- Integration with `noctilux run` CLI for annotation-aware configs.
- End-to-end tests with real annotation formats.
- Documentation and examples.

## Out of Scope

- Training framework or model evaluation.
- Automatic dataset format conversion (e.g., VOC to COCO).
- Full COCO feature parity (e.g., segmentation RLE encoding, caption annotations).
- OCR or text-region annotation synchronization.
- Video annotation synchronization.
- 3D bounding boxes or point cloud annotation.
- Annotation visualization tools.
- Automatic label correction or cleaning.
