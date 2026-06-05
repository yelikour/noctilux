# Parallel Execution and Resume Design

This document describes the planned parallel execution and resume architecture for Noctilux.

## Current State

Noctilux v0.5.x uses serial execution only:

- `cli.py` iterates over all `(sample, pipeline, repeat_index)` tuples in a single process.
- `num_workers > 1` logs a warning and still runs serially.
- All metadata is collected in memory via `MetadataRecorder` and written to disk at the end of the run (`write_all()`).
- Progress is tracked with a simple `tqdm` wrapper over the sample list.
- Output paths are generated on the fly by `OutputSaver.build_output_path()` with conflict detection.

### Why Serial is Safe

The current serial design avoids all concurrency problems by construction:

- Only one write to any output path at a time.
- Metadata is accumulated in a single in-memory object.
- No shared mutable state between iterations.
- Deterministic seed derivation via `combine_seed()` is order-independent.

## Why Not Just Enable `num_workers`

Setting `num_workers > 1` without architectural changes would cause:

1. **Metadata write conflicts**: `MetadataRecorder` is not thread-safe. Multiple workers appending to the same lists would corrupt data.
2. **`transform_log.jsonl` interleaving**: Concurrent JSON line writes would produce malformed JSONL.
3. **`failed_images.csv` ordering**: Rows from parallel workers would arrive non-deterministically, making the file unreproducible.
4. **Output file name collisions**: `OutputSaver._resolve_conflict()` checks disk state; two workers could race and both write the same path.
5. **Seed determinism**: Seeds are derived correctly in serial mode. In parallel, task scheduling order is non-deterministic, so per-task seeds must be pre-computed, not derived from iteration order.
6. **`tqdm` output corruption**: Multiple workers updating the same progress bar produces garbled terminal output.
7. **Cross-platform `multiprocessing` differences**: `multiprocessing` on macOS uses `spawn`, Linux defaults to `fork`, Windows requires `spawn`. Lambda captures and PIL/Image objects may not serialize the same way across start methods.

## Recommended Architecture

### Core Principle: Workers Return Results, Main Process Writes

```
┌──────────────────────────────────────────────────┐
│ Main Process                                      │
│                                                   │
│  1. Pre-compute all tasks                         │
│  2. Dispatch tasks to worker pool                 │
│  3. Collect result objects                        │
│  4. Write metadata (single writer)                │
│  5. Update progress bar                           │
│                                                   │
└──────────────────────────────────────────────────┘
         │                              ▲
         │ task                         │ result
         ▼                              │
┌──────────────────────────────────────────────────┐
│ Worker Process                                    │
│                                                   │
│  1. Load image                                    │
│  2. Apply pipeline transforms                     │
│  3. Save output image                             │
│  4. Return result dict                            │
│                                                   │
└──────────────────────────────────────────────────┘
```

### Task Definition

Each task is a self-contained unit of work:

```python
@dataclass
class Task:
    sample_id: str
    image_path: Path
    pipeline_name: str
    pipeline_config: dict
    repeat_index: int
    seed: int
    output_path: Path  # pre-allocated
```

### Result Object

Workers return a result, never write metadata directly:

```python
@dataclass
class TaskResult:
    sample_id: str
    pipeline_name: str
    repeat_index: int
    seed: int
    output_path: Path
    success: bool
    error: str | None
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    input_format: str
    output_format: str
    transform_log: list[dict]
```

### Output Path Pre-allocation

The main process generates all output paths before dispatching tasks. This eliminates race conditions in `OutputSaver._resolve_conflict()`.

```python
def pre_allocate_paths(samples, pipelines, repeats, output_root):
    paths = {}
    for sample in samples:
        for pipeline in pipelines:
            for r in range(repeats):
                task_key = (sample["sample_id"], pipeline.name, r)
                paths[task_key] = output_saver.build_output_path(...)
    return paths
```

### Metadata-Safe Writer

A refactored metadata writer that:

1. Receives `TaskResult` objects one at a time from the main process.
2. Writes each result immediately (streaming) instead of accumulating all in memory.
3. Uses a single write thread/process — never concurrent writers.
4. Handles partial writes gracefully (crash safety).

```python
class MetadataWriter:
    def write_result(self, result: TaskResult) -> None:
        """Append a single result to all metadata files."""
        self._append_manifest_row(result)
        self._append_transform_log(result)
        if not result.success:
            self._append_failed_image(result)

    def finalize(self) -> None:
        """Write summary.csv after all results are collected."""
        self._write_summary()
```

This replaces the current `MetadataRecorder` which collects everything in memory and writes at the end.

### Progress Tracking

Only the main process updates `tqdm`. Workers do not print or update progress bars.

```python
with tqdm(total=len(tasks), desc="Processing", unit="task") as pbar:
    for result in pool.imap_unordered(process_task, tasks):
        metadata_writer.write_result(result)
        pbar.update(1)
```

## Seed Strategy

### Current Seed Derivation

Seeds are already deterministic and order-independent in serial mode:

```python
def combine_seed(*parts) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % (2**32)
```

Each task seed is derived from: `global_seed + pipeline_name + sample_id + repeat_index`.

### Parallel Seed Consistency

Since `combine_seed` uses cryptographic hashing of fixed inputs (not iteration order), the derived seed for any `(sample_id, pipeline_name, repeat_index)` tuple is the same regardless of execution order. This means:

- Serial and parallel runs produce identical per-task seeds.
- `imap_unordered` does not affect seed determinism.
- Reproducibility is preserved across execution modes.

No changes to `combine_seed` are needed for parallel execution.

## Resume Design

### Resume from Existing Metadata

When `--resume` is active, the system:

1. Reads `manifest.csv` from the target metadata directory.
2. Builds a set of completed `(sample_id, pipeline_name, repeat_index)` tuples.
3. Filters the task list to only uncompleted tasks.
4. Appends new results to existing metadata files instead of overwriting.

### Skip-Existing

`--skip-existing` is a lighter alternative to `--resume`:

1. Checks if output image files already exist on disk.
2. Skips tasks whose output path is already occupied.
3. Does not require reading metadata files.

This is useful when the user knows outputs are correct but wants to re-run the config on new images only.

### Retry-Failed

`--retry-failed` reads `failed_images.csv` and re-processes only previously failed samples:

1. Reads `failed_images.csv` from the target metadata directory.
2. Extracts `(pipeline_name, repeat_index, sample_id)` from failed rows.
3. Reconstructs tasks for only those tuples.
4. Replaces failed rows in metadata (not append duplicates).

### Resume Manifest

The resumed run writes a complete manifest that includes both pre-existing and new results. This means:

- `manifest.csv` contains all outputs (old + new).
- `transform_log.jsonl` contains all logs (old + new).
- `summary.csv` is regenerated from the full manifest.
- `failed_images.csv` reflects the latest state (retried failures removed if they now succeed).

### Failed Sample Retry Strategy

| Flag | Behavior |
|------|----------|
| `--resume` | Skip completed, re-attempt failed |
| `--skip-existing` | Skip any output that exists on disk |
| `--retry-failed` | Only re-attempt previously failed samples |
| `--resume --retry-failed` | Skip completed, re-attempt failed (same as `--resume` alone) |
| No flags | Start fresh (overwrite if allowed, skip if not) |

## CLI Parameters

```bash
noctilux run \
  --config config.yaml \
  --num-workers 4 \
  --resume \
  --skip-existing \
  --retry-failed
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num-workers` | int | 1 | Number of worker processes. 1 = serial. |
| `--resume` | flag | off | Skip already-completed tasks from existing metadata. |
| `--skip-existing` | flag | off | Skip tasks whose output file already exists on disk. |
| `--retry-failed` | flag | off | Re-process only previously failed samples. |

These are CLI-only overrides. YAML `runtime.num_workers` remains supported as the default.

## Test Plan

### Phase Tests (v0.5.1 — Metadata Writer Refactor)

- Streaming write: `write_result` appends one row at a time.
- Partial write recovery: if the process crashes mid-run, existing rows are valid.
- Round-trip: write N results, read back, verify all rows present.
- Append mode: write K results, close, re-open, write M more, verify K+M rows.
- Summary generation from partial results.

### Phase Tests (v0.5.2 — Serial Resume)

- Fresh run produces full metadata.
- Second run with `--resume` skips all completed tasks.
- Partial run (interrupted) + resume completes the remainder.
- `--skip-existing` skips based on file presence, not metadata.
- `--retry-failed` only re-processes entries from `failed_images.csv`.
- Resume manifest merges old and new results correctly.

### Phase Tests (v0.5.3 — Process Pool Prototype)

- `num_workers=1` produces identical output to current serial execution.
- `num_workers=2` produces the same images and seeds as `num_workers=1`.
- No output file collisions.
- No metadata corruption.
- Progress bar updates correctly.
- Works on Linux (`fork`), macOS (`spawn`), Windows (`spawn`).

## Implementation Phases

### v0.5.0 — Design (this document)

- Document parallel and resume architecture.
- Define task/result data structures.
- Define seed strategy and resume semantics.
- No code changes.

### v0.5.1 — Metadata-Safe Writer Refactor (completed)

- Replaced `MetadataRecorder` with streaming `MetadataWriter` in CLI run.
- `MetadataWriter.write_success` and `write_failure` write manifest and transform_log rows immediately.
- `MetadataWriter.close` writes `summary.csv`.
- Internal counters track success/failed counts for summary.
- `MetadataRecorder` retained for backward compatibility.
- Serial execution unchanged. All existing tests pass.
- Parallel and resume not yet implemented.

### v0.5.2 — Serial Resume (completed)

- Implemented `--resume`, `--skip-existing`, `--retry-failed` in serial mode.
- `--resume` reads `manifest.csv` success records and skips completed outputs.
- `--skip-existing` checks if output file already exists on disk.
- `--retry-failed` reads `failed_images.csv` and only re-processes those keys.
- `--resume` and `--retry-failed` are mutually exclusive.
- Added `src/noctilux/resume.py` with `load_success_keys`, `load_failed_keys`, `build_processing_key`.
- Run summary includes `skipped_count` and flag status.
- Skipped items are not written to metadata.
- Metadata schema unchanged. Execution still serial. Parallel not yet implemented.

### v0.5.3 — Process Pool Prototype (completed)

- Implemented `ProcessPoolExecutor`-based parallel execution.
- Added `src/noctilux/worker.py` with `ProcessingTask`/`ProcessingResult` dataclasses.
- Worker function `process_task` handles load → transform → save → return result.
- Main process pre-allocates output paths via `pre_allocate_output_paths`.
- Main process dispatches tasks, collects results, writes metadata (single writer).
- Added `--num-workers N` CLI argument (overrides `runtime.num_workers` config).
- Resume/skip-existing/retry-failed filtering happens before task dispatch.
- Seed determinism preserved: `combine_seed` is order-independent.
- `num_workers=1` remains the safe default (serial execution path).
- Added `tests/test_parallel.py` with 14 tests.
- Serial execution unchanged when `num_workers=1`.

### v0.5.4 — Parallel Stabilization (completed)

- Added 28 tests in `tests/test_parallel.py` (up from 14 in v0.5.3).
- Determinism tests: manifest keys, output paths, seeds, summary stats consistent between serial and parallel.
- JSONL validity test for `transform_log.jsonl` in parallel mode.
- Failure scenario tests: corrupt image (load_image), transform error (transform stage), single-failure isolation.
- Resume/skip-existing/retry-failed boundary tests for parallel mode.
- Experimental warning logged when `--num-workers > 1`.
- `num_workers` status line in run summary output.
- Removed stale `v0.3.x` serial-only note from `inspect-config`.
- Serial execution (num_workers=1) unchanged. Metadata schema unchanged. Default remains serial.
- This is still not "stable parallel execution" — that goal remains at v0.6.0.

#### Stabilization Checklist

- [x] Manifest keys match between serial and parallel
- [x] Output paths match between serial and parallel
- [x] Seeds are deterministic across execution modes
- [x] summary.csv counts are consistent
- [x] transform_log.jsonl lines are valid JSON
- [x] Repeat > 1 produces no filename collisions
- [x] Multiple pipelines produce no filename collisions
- [x] Corrupt images record stage=load_image in failed_images.csv
- [x] Transform errors record stage=transform in failed_images.csv
- [x] Single task failure does not crash the run (skip_broken_images=True)
- [x] --resume works with --num-workers 2
- [x] --skip-existing works with --num-workers 2
- [x] --retry-failed works with --num-workers 2
- [x] --resume and --retry-failed remain mutually exclusive with --num-workers
- [x] Skipped items not written to metadata
- [x] Experimental warning displayed when parallel mode is active
- [ ] Worker crash recovery (e.g., SIGKILL, OOM) — deferred to v0.6.0
- [ ] Timeout handling for hung workers — deferred to v0.6.0
- [ ] Cross-platform (macOS spawn, Windows spawn) validation — deferred to v0.6.0
- [ ] Performance benchmarks — deferred to v0.6.0

### v0.6.0 — Stable Parallel Execution

- Harden error handling (worker crashes, timeouts).
- Performance benchmarks.
- Documentation and examples.
- Consider `concurrent.futures.ProcessPoolExecutor` vs `multiprocessing.Pool`.

## Out of Scope

- GPU acceleration (CUDA).
- Distributed execution across multiple machines.
- Async or event-driven processing.
- Changing the default `num_workers` from 1.
- Thread-based parallelism (PIL is not thread-safe for all operations).
- Annotation synchronization in parallel mode (planned for v0.7.0+).
