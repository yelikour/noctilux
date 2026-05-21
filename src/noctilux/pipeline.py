from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
from PIL import Image

from noctilux.registry import build_transform


class PipelineExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        transform_logs: list[dict[str, Any]],
        run_seed: int,
    ) -> None:
        super().__init__(message)
        self.transform_logs = transform_logs
        self.run_seed = run_seed


class AugmentPipeline:
    def __init__(
        self,
        name: str,
        transforms: list[dict[str, Any]],
        repeat: int = 1,
        seed: int | None = None,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.transforms = transforms
        self.repeat = repeat
        self.seed = seed
        self.enabled = enabled

    def apply(
        self,
        image: Image.Image,
        sample: dict[str, Any],
        repeat_index: int = 0,
        seed: int | None = None,
    ) -> tuple[Image.Image, list[dict[str, Any]], int]:
        if not isinstance(image, Image.Image):
            raise TypeError("Pipeline input must be a PIL.Image.Image instance.")

        run_seed = self._resolve_run_seed(sample=sample, repeat_index=repeat_index, seed=seed)
        run_rng = random.Random(run_seed)
        current_image = image.copy()
        transform_logs: list[dict[str, Any]] = []

        for transform_index, spec in enumerate(self.transforms):
            actual_params = resolve_random_params(spec.get("params", {}), run_rng)
            probability = float(spec.get("p", 1.0))
            should_apply = probability >= 1.0 or (probability > 0.0 and run_rng.random() < probability)
            log_entry = {
                "name": spec["name"],
                "backend": spec.get("backend", "pillow"),
                "p": probability,
                "applied": should_apply,
                "params": actual_params,
            }
            if not should_apply:
                transform_logs.append(log_entry)
                continue

            transform_seed = combine_seed(run_seed, self.name, transform_index)
            transform = build_transform(
                spec["name"],
                params=actual_params,
                backend=spec.get("backend"),
            )
            context = {
                "sample": sample,
                "pipeline_name": self.name,
                "repeat_index": repeat_index,
                "seed": transform_seed,
                "rng": random.Random(transform_seed),
                "np_rng": np.random.default_rng(transform_seed),
            }
            try:
                current_image = transform(current_image, context=context)
            except Exception as exc:
                log_entry["error"] = str(exc)
                transform_logs.append(log_entry)
                raise PipelineExecutionError(
                    f"Pipeline '{self.name}' failed in transform '{spec['name']}': {exc}",
                    transform_logs=transform_logs,
                    run_seed=run_seed,
                ) from exc
            if not isinstance(current_image, Image.Image):
                raise TypeError(
                    f"Transform '{spec['name']}' returned {type(current_image)!r}, expected PIL.Image.Image."
                )
            transform_logs.append(log_entry)

        return current_image, transform_logs, run_seed

    def _resolve_run_seed(
        self,
        sample: dict[str, Any],
        repeat_index: int,
        seed: int | None = None,
    ) -> int:
        base_seed = seed if seed is not None else self.seed
        if base_seed is None:
            return random.SystemRandom().randrange(0, 2**32)
        return combine_seed(base_seed, self.name, sample.get("sample_id", ""), repeat_index)


def build_pipelines(config: dict[str, Any]) -> list[AugmentPipeline]:
    seed = config.get("seed")
    pipelines = []
    for spec in config["pipelines"]:
        if not spec.get("enabled", True):
            continue
        pipelines.append(
            AugmentPipeline(
                name=spec["name"],
                transforms=spec["transforms"],
                repeat=spec.get("repeat", 1),
                seed=seed,
                enabled=spec.get("enabled", True),
            )
        )
    return pipelines


def resolve_random_params(params: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    return {key: resolve_random_value(value, rng) for key, value in params.items()}


def resolve_random_value(value: Any, rng: random.Random) -> Any:
    if isinstance(value, list):
        if not value:
            raise ValueError("Random choice list cannot be empty.")
        return rng.choice(value)
    if isinstance(value, dict) and value.get("type") in {"choice", "randint", "uniform"}:
        value_type = value["type"]
        if value_type == "choice":
            values = value.get("values", [])
            if not isinstance(values, list) or not values:
                raise ValueError("choice random parameter requires a non-empty 'values' list.")
            return rng.choice(values)
        if value_type == "randint":
            minimum = value.get("min")
            maximum = value.get("max")
            if not isinstance(minimum, int) or not isinstance(maximum, int):
                raise ValueError("randint random parameter requires integer min and max.")
            if minimum > maximum:
                raise ValueError("randint random parameter min cannot exceed max.")
            return rng.randint(minimum, maximum)
        minimum = value.get("min")
        maximum = value.get("max")
        if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
            raise ValueError("uniform random parameter requires numeric min and max.")
        if minimum > maximum:
            raise ValueError("uniform random parameter min cannot exceed max.")
        return rng.uniform(float(minimum), float(maximum))
    return value


def combine_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % (2**32)
