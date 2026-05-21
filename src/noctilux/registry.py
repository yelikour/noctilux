from __future__ import annotations

from collections.abc import Callable
from typing import Any

TRANSFORM_REGISTRY: dict[str, type] = {}


def register_transform(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        if name in TRANSFORM_REGISTRY:
            raise ValueError(f"Transform already registered: {name}")
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return decorator


def build_transform(name: str, params: dict[str, Any] | None = None, backend: str | None = None) -> Any:
    if name not in TRANSFORM_REGISTRY:
        raise KeyError(f"Unknown transform: {name}. Available: {', '.join(list_transforms())}")
    transform_cls = TRANSFORM_REGISTRY[name]
    init_params = dict(params or {})
    if backend is not None:
        init_params["backend"] = backend
    return transform_cls(**init_params)


def list_transforms() -> list[str]:
    return sorted(TRANSFORM_REGISTRY)


import noctilux.transforms  # noqa: E402,F401
