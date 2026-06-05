# Backend Design Document

This document describes the planned optional backend architecture for Noctilux, focusing on the OpenCV backend as the first non-default option.

## Current State

Noctilux v0.3.x uses Pillow and NumPy exclusively:

- All transforms accept `PIL.Image.Image` as input and return `PIL.Image.Image` as output.
- PIL handles image I/O (read, save, EXIF orientation, format conversion).
- NumPy is used for pixel-level operations (noise, motion blur, some color transforms).
- No external image processing libraries are required.

This is the default backend and will remain the default for the foreseeable future.

## Why Optional Backends

Different backends offer different strengths:

- **OpenCV**: faster resize/blur/rotate for large images, broader interpolation options.
- **Albumentations**: rich augmentation API with research-oriented transforms.
- **imagecorruptions**: standardized corruption benchmarks.
- **AugLy**: platform-specific compression and social media simulation.

None of these should be required. Users who only need Pillow should not need to install OpenCV.

## Design Principles

1. **Pillow + NumPy remains the default backend.** No configuration change is needed for existing users.
2. **All transform input/output remains `PIL.Image.Image`.** If a backend internally uses NumPy arrays or cv2 matrices, the conversion must happen inside the transform. The pipeline should never see a non-PIL image.
3. **Backends are opt-in.** Adding `backend: opencv` to a transform in YAML selects the OpenCV implementation. Without the `backend` field, the default (Pillow/NumPy) implementation is used.
4. **Clear error on missing backend.** If a user specifies `backend: opencv` but `opencv-python` is not installed, Noctilux must raise a clear error message with installation instructions, not an opaque ImportError.
5. **Existing YAML configs are not broken.** The `backend` field is optional and defaults to `pillow`.
6. **Backend implementations share the same interface.** Each backend-specific transform class inherits from `BaseTransform` and is registered under the same transform name but with a backend tag.

## Configuration Format

### Default backend (no change needed)

```yaml
transforms:
  - name: resize_long_edge
    params:
      long_edge: 512
```

### Explicit Pillow backend

```yaml
transforms:
  - name: resize_long_edge
    backend: pillow
    params:
      long_edge: 512
```

### OpenCV backend

```yaml
transforms:
  - name: resize_long_edge
    backend: opencv
    params:
      long_edge: 512
```

### Mixed backends in one pipeline

```yaml
pipelines:
  - name: mixed_pipeline
    transforms:
      - name: resize_long_edge
        backend: opencv
        params:
          long_edge: 512
      - name: jpeg_compression
        params:
          quality: 85
```

## Backend Registry Design

The current `TRANSFORM_REGISTRY` maps `(name) -> transform_class`.

With backend support, the registry should map `(name, backend) -> transform_class`:

```python
TRANSFORM_REGISTRY: dict[tuple[str, str], type[BaseTransform]] = {}

# Default Pillow registration
@register_transform("resize_long_edge", backend="pillow")
class ResizeLongEdgePillow(BaseTransform):
    ...

# OpenCV registration
@register_transform("resize_long_edge", backend="opencv")
class ResizeLongEdgeOpenCV(BaseTransform):
    ...
```

Lookup logic:

1. If `backend` is specified in config, look up `(name, backend)`.
2. If `backend` is not specified, look up `(name, "pillow")` as default.
3. If the requested backend is not registered for that transform, fall back to Pillow with a warning.
4. If the requested backend package is not installed, raise a clear error.

## Optional Dependency Design

In `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["build", "pytest", "ruff"]
opencv = ["opencv-python-headless"]
# Future:
# albumentations = ["albumentations"]
# full = ["opencv-python-headless", "albumentations"]
```

Users install with:

```bash
pip install -e ".[opencv]"
```

`opencv-python-headless` is preferred over `opencv-python` because Noctilux is a CLI tool and does not need GUI features.

## Error Handling

### Missing backend package

```python
class BackendNotAvailableError(NoctiluxError):
    """Raised when a backend is requested but its package is not installed."""
    pass

def require_backend(backend: str) -> None:
    if backend == "opencv":
        try:
            import cv2
        except ImportError:
            raise BackendNotAvailableError(
                "OpenCV backend requires 'opencv-python-headless'. "
                "Install it with: pip install 'noctilux[opencv]'"
            )
```

### Backend not registered for transform

If a transform has no OpenCV implementation, log a warning and fall back to Pillow:

```python
logger.warning(
    "Transform '%s' has no '%s' backend implementation. Falling back to Pillow.",
    name, backend
)
```

## Test Strategy

### Without OpenCV installed

- All existing tests continue to pass.
- Tests for `backend: opencv` configuration should be skipped if `cv2` is not available.
- Error messages for missing backend should be tested.

### With OpenCV installed

- Transform output comparison: same YAML config with `backend: pillow` and `backend: opencv` should produce visually equivalent results (within a small pixel tolerance).
- Performance benchmarks: not required for CI, but useful for documentation.
- Round-trip test: PIL -> cv2 -> PIL should not corrupt image data or lose metadata.

### CI matrix

- Default CI continues without OpenCV.
- Optionally add a separate CI job with `pip install -e ".[opencv]"` for backend tests.

## Priority OpenCV Transforms

Not all transforms benefit equally from an OpenCV backend. The following are recommended for the initial v0.4.0 implementation:

| Transform | OpenCV advantage |
|-----------|-----------------|
| `resize_exact` | More interpolation methods, faster on large images |
| `resize_long_edge` | Same as above |
| `gaussian_blur` | Optimized kernel computation |
| `rotate` | Affine transform with better border handling |

These four transforms have clear Pillow equivalents, making comparison testing straightforward.

## Out of Scope for v0.4.0

- Annotation synchronization (detection/segmentation) — planned for v0.6.0.
- GPU acceleration (CUDA) — no current plan.
- OpenCV-only pipelines — all transforms must have a Pillow fallback.
- Automatic backend selection — users must explicitly opt in via config.
- Albumentations / imagecorruptions / AugLy backends — future consideration.
- Changing the default backend — Pillow remains default.

## Implementation Phases

### v0.4.0a — Design (completed in v0.3.7)

- Document backend architecture.
- Define configuration format and registry changes.
- No code changes.

### v0.4.0 — Minimal OpenCV backend (completed)

- Added `src/noctilux/backends/` module with PIL-OpenCV conversion utilities.
- Added `opencv` optional dependency group (`noctilux[opencv]`).
- Implemented OpenCV backend for `resize_exact`, `resize_long_edge`, `gaussian_blur`, `rotate`.
- Added `configs/examples/opencv_backend.yaml`.
- Added backend-aware tests (with `pytest.importorskip("cv2")` for optional tests).
- Kept Pillow as default backend. OpenCV is opt-in via `backend: opencv` in YAML.

### v0.4.1 — OpenCV CI and stabilization (completed)

- Added dedicated `opencv-backend` CI job (Python 3.12, `.[dev,opencv]`).
- Unified OpenCV installation instructions and error messages.
- Improved project-quality tests for backend consistency.

### v0.4.x — Broader coverage (planned)

- Add OpenCV implementations for more transforms where beneficial.
- Consider Albumentations backend.
- Performance benchmarks.
