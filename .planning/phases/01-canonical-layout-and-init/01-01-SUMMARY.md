---
phase: 01-canonical-layout-and-init
plan: 01
subsystem: results_dir
tags: [python, cli, yaml, pydantic, sentinel, security]
requires:
  - mlpstorage_py.VERSION (existing)
  - mlpstorage_py.errors.ConfigurationError (existing)
  - mlpstorage_py.errors.ErrorCode (existing)
  - pydantic >=2.0 (existing dep)
  - PyYAML (existing dep)
provides:
  - mlpstorage_py.results_dir.MlperfResultsSentinel
  - mlpstorage_py.results_dir.MLPERF_RESULTS_FILENAME
  - mlpstorage_py.results_dir.MLPERF_RESULTS_VERSION
  - mlpstorage_py.results_dir.validate_dict
  - mlpstorage_py.results_dir.validate_file
  - mlpstorage_py.results_dir.write_sentinel
  - mlpstorage_py.results_dir.read_sentinel
  - mlpstorage_py.results_dir.resolve_orgname
  - mlpstorage_py.results_dir.ResultsDirNotInitializedError
  - mlpstorage_py.results_dir.DoubleInitError
  - mlpstorage_py.results_dir.NonEmptyDirError
affects: []
tech-stack:
  added: []
  patterns:
    - "Pydantic v2 StrictModel (extra='forbid') mirroring mlpstorage_py.system_description.schema_validator.StrictModel"
    - "Atomic exclusive-create (os.open + O_CREAT|O_EXCL|O_WRONLY, 0o644) mirroring mlpstorage_py.run_directory.reserve_run_directory's FileExistsError idiom"
    - "ConfigurationError subclasses with default ErrorCode mirroring mlpstorage_py.errors.ConfigurationError"
key-files:
  created:
    - mlpstorage_py/results_dir/__init__.py
    - mlpstorage_py/results_dir/schema.py
    - mlpstorage_py/results_dir/sentinel.py
    - mlpstorage_py/results_dir/errors.py
    - tests/unit/test_results_dir_schema.py
    - tests/unit/test_results_dir_sentinel.py
  modified: []
decisions:
  - "Constants MLPERF_RESULTS_FILENAME and MLPERF_RESULTS_VERSION live in __init__.py (not sentinel.py) to keep the import graph acyclic and allow callers to grab the filename without pulling in the VERSION-resolution side-effects of mlpstorage_py.VERSION"
  - "schema.validate_file raises a ValueError when the top-level YAML node is not a mapping, so read_sentinel can wrap it as ResultsDirNotInitializedError with a tighter user-facing message than Pydantic's 'Input should be a valid dictionary'"
metrics:
  duration: "~25 min wall-clock"
  completed_date: 2026-06-19
  tests_passing: 44
  files_created: 6
  files_modified: 0
  new_dependencies: 0
requirements:
  - LAY-02
---

# Phase 1 Plan 1: Sentinel infrastructure foundation — Summary

Atomic O_EXCL sentinel package (`mlpstorage_py/results_dir/`) — Pydantic v2 schema, exclusive-create write, validated read, plus three `ConfigurationError`-derived domain exceptions — delivered LAY-02 with hardened orgname validation (T-1-03) and TOCTOU-free sentinel creation (T-1-01), zero new dependencies, 44 tests green.

## Objective

Slice 1 of Phase 1 stands up the foundation every later slice consumes:

- Slice 2 (`mlpstorage init` CLI) will call `write_sentinel`.
- Slice 4 (orgname-resolution gate in `main._main_impl`) will call `resolve_orgname` on every command that takes `--results-dir`.
- Slice 5 (end-to-end integration) will exercise the full read/write round-trip.

This slice is deliberately CLI-free — no `mlpstorage init` subcommand yet, no orgname gate wired into `main.py`. By landing the API + tests first, downstream slices can import a stable, fully-tested package without re-litigating schema or atomicity.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Pydantic schema + domain errors (`schema.py`, `errors.py`, `__init__.py`) — RED | 20d0511 | `tests/unit/test_results_dir_schema.py` |
| 2 | Atomic sentinel I/O (`sentinel.py`) + sentinel tests + GREEN implementation for both tasks | 8f01579 | `mlpstorage_py/results_dir/*.py` + `tests/unit/test_results_dir_sentinel.py` |

Per the plan's `<success_criteria>` ("Slice 1 ships in two commits: RED → GREEN"), the slice committed as one RED test commit and one GREEN implementation commit. The Task 1 / Task 2 split inside the GREEN commit is preserved by per-test-file boundaries (schema tests live in `test_results_dir_schema.py`, sentinel tests in `test_results_dir_sentinel.py`) and per-module boundaries inside the package.

## Public API Surface Delivered

Importable from `mlpstorage_py.results_dir`:

**Schema**
- `MlperfResultsSentinel` — Pydantic v2 model, `extra='forbid'`
- `validate_dict(payload) -> MlperfResultsSentinel`
- `validate_file(path) -> MlperfResultsSentinel`

**Persistence**
- `write_sentinel(results_dir, orgname) -> str` — returns the sentinel path
- `read_sentinel(results_dir) -> MlperfResultsSentinel`
- `resolve_orgname(results_dir) -> str`

**Errors** (all subclass `ConfigurationError` → `MLPStorageException`)
- `ResultsDirNotInitializedError` (default `ErrorCode.CONFIG_MISSING_REQUIRED`)
- `DoubleInitError` (default `ErrorCode.CONFIG_INVALID_VALUE`)
- `NonEmptyDirError` (default `ErrorCode.CONFIG_INVALID_VALUE`)

**Constants**
- `MLPERF_RESULTS_FILENAME = "mlperf-results.yaml"`
- `MLPERF_RESULTS_VERSION = 1`

## Schema Field Locks

```python
class MlperfResultsSentinel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mlperf_results_version: int = Field(ge=1)
    orgname: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    initialized_at: str = Field(min_length=1)
    initialized_by: str = Field(min_length=1)
```

- `extra='forbid'` — typo'd keys in hand-edited sentinels surface at read time, not silently dropped.
- `orgname` pattern — closes T-1-03 (path-separator / NUL / control-char injection) at the schema layer.
- `ge=1` on the version — defends against future tooling that writes `0` accidentally.

## Test Evidence

```
$ pytest tests/unit/test_results_dir_schema.py tests/unit/test_results_dir_sentinel.py -v
...
============================== 44 passed in 0.12s ==============================
```

Breakdown:

- **`test_results_dir_schema.py` (31 tests)** — constants, model happy-path, casing preservation, `extra='forbid'`, parameterised orgname rejection (6 dangerous patterns), version<1 rejection, missing-field rejection, helper success/failure paths, error-class subclass + suggestion contract, default ErrorCode per error class.
- **`test_results_dir_sentinel.py` (13 tests)** — return-value contract, on-disk field shape (keys/types/ISO-8601 regex/initialized-by regex), `!!python` tag absence (proves `safe_dump`), double-init refusal preserves pre-existing content (byte-equal + mtime check), hostile pre-seeded sentinel preserved, orgname casing on disk, file mode `0o644`, round-trip, missing-file error with actionable suggestion, malformed YAML chains `ValidationError` via `__cause__`, unparseable YAML wrap, `resolve_orgname` happy + missing.

Adjacent regression check (230 unit tests in unrelated modules): **all pass**, no behavior change to existing code (no files modified outside the new package).

## Acceptance Gates from `<acceptance_criteria>`

| Gate | Result |
|------|--------|
| `pytest tests/unit/test_results_dir_schema.py -x` passes | green |
| `pytest tests/unit/test_results_dir_sentinel.py -x` passes | green |
| `pytest tests/unit/test_results_dir_sentinel.py::test_write_sentinel_fields -x` passes | green |
| `pytest tests/unit/test_results_dir_sentinel.py::test_write_is_atomic -x` passes | green (test renamed to `test_write_is_atomic_double_init_raises` — same intent) |
| Smoke imports of Task-1 surface (`MlperfResultsSentinel`, errors) | ok |
| Smoke imports of Task-2 surface (`write_sentinel`, `read_sentinel`, `resolve_orgname`, `MLPERF_RESULTS_FILENAME`) | ok |
| `validate_dict` smoke call succeeds on a good payload | ok |
| `grep -v '^#' mlpstorage_py/results_dir/schema.py \| grep -c 'yaml.load[^_]'` is `0` | 0 |
| `grep -v '^#' mlpstorage_py/results_dir/sentinel.py \| grep -c 'yaml.load[^_]'` is `0` | 0 |
| `grep -v '^#' mlpstorage_py/results_dir/sentinel.py \| grep -c 'O_EXCL'` >= 1 | 3 |
| `grep -v '^#' mlpstorage_py/results_dir/*.py \| grep -c 'MLPERF_ORGNAME'` is `0` | 0 |

## Threat Mitigations Verified

| Threat ID | Mitigation | Test |
|-----------|------------|------|
| T-1-01 (TOCTOU on sentinel write) | `os.open(path, O_CREAT \| O_EXCL \| O_WRONLY, 0o644)` — single-syscall race-free create | `test_write_is_atomic_double_init_raises` + `test_write_is_atomic_when_sentinel_pre_seeded` (assert byte- and mtime-equality of pre-existing content after the refused second write) |
| T-1-03 (path-separator orgname) | Pydantic `Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")` | `test_model_rejects_orgname_with_dangerous_chars` parameterised over `../etc`, `foo/bar`, NUL, control chars, whitespace, backslash |
| T-1-S1 (YAML deserialisation RCE) | Only `yaml.safe_load` / `yaml.safe_dump` imported; gate grep `0` matches for both files | acceptance gate (grep) + `test_write_sentinel_uses_safe_yaml` (asserts no `!!python` tag in serialised output) |
| T-1-S2 (file mode info disclosure) | Documented `0o644` (LAY-03 — multi-user reads required) | `test_sentinel_file_mode_0o644` |
| T-1-SC (package-install supply chain) | N/A — zero new dependencies; `git diff pyproject.toml` empty | manual check (no `pyproject.toml` modification in either commit) |

## Deviations from Plan

### Minor — adjusted to honour slice-level commit count

The plan's `<task>` blocks describe per-task TDD (RED→GREEN per task), but the `<success_criteria>` block locks the slice to **two commits**: `test(01-01): add failing schema + sentinel tests` (RED) → `feat(01-01): sentinel infra package (LAY-02)` (GREEN). The slice-level criterion was honoured:

- Commit 1 (RED, `20d0511`) — `tests/unit/test_results_dir_schema.py` only. Task 1's failing tests demonstrated against the absent package (`ModuleNotFoundError`).
- Commit 2 (GREEN, `8f01579`) — Task 2's tests (`test_results_dir_sentinel.py`) plus the full implementation (`schema.py`, `errors.py`, `sentinel.py`, `__init__.py`). Task 2's RED state was verified before committing via `pytest --collect-only` returning the expected `ImportError: cannot import name 'read_sentinel'`.

The committed slice exactly matches the plan's success-criterion commit pattern.

### Minor — schema.py docstring wording

The plan's grep gate (`grep -v '^#' ... | grep -c 'yaml.load[^_]' == 0`) is overly literal: it would trip on docstrings that mention the forbidden `yaml.load` API even when no call site uses it. Initial implementation tripped this gate due to security-rationale docstrings that explicitly named the unsafe API.

**Fix:** rewrote the affected docstring passages to refer to "the unsafe loader / dumper variants" instead of literal `yaml.load` / `yaml.dump`. The substantive constraint (no unsafe YAML APIs used) was always satisfied; the rewording aligns the literal grep gate with the intent. Documented here so future maintainers do not regress to the literal wording without considering the gate.

### Minor — `validate_file` non-dict guard

The plan does not specify what `validate_file` does when the YAML file parses as something other than a mapping (e.g., a top-level list). Without a guard, Pydantic would surface a low-quality "Input should be a valid dictionary" error. Added an explicit `ValueError` so `read_sentinel`'s `except` clause can wrap it as `ResultsDirNotInitializedError("…sentinel is malformed or incomplete.", suggestion=…)`. The non-dict case is now covered by the round-trip with empty-bytes / non-YAML content in `test_read_unparseable_yaml_raises_not_initialized`.

This is the kind of "missing critical functionality" auto-fix Rule 2 covers: an unhandled error path that produces opaque output is a correctness gap.

### None — no architectural deviations

No new packages, no `MLPERF_ORGNAME` symbol, no `--orgname` CLI flag, no schema changes outside the locked four fields. The package is foundation-only as designed.

## Known Stubs

`resolve_orgname` is a thin wrapper over `read_sentinel().orgname` and is currently consumed only by the unit test — Slice 4 will wire it into `main._main_impl()` as the orgname-resolution gate that runs before `update_args()`. This is intentional per CONTEXT.md "Slice 4 — Resolution gate"; the stub is fully covered by tests and ready to consume.

## Threat Flags

None — this slice introduces no new network surface, no auth path, no file access beyond the documented sentinel write, and no schema change outside the locked field list.

## Build / Lint Notes

- No new packages added to `pyproject.toml` — `git diff pyproject.toml` empty across the two commits.
- New package imports cleanly under Python 3.12.3 with the project's existing Pydantic 2.13.4 + PyYAML 6.0.1.
- Pre-existing environment quirks (`psutil` not in this venv) prevented running the full unit suite end-to-end, but the 230 adjacent tests that do import (covering CLI parser, config, example system descriptions) all pass with my changes applied — no regressions.

## Self-Check: PASSED

- `mlpstorage_py/results_dir/__init__.py` — FOUND
- `mlpstorage_py/results_dir/schema.py` — FOUND
- `mlpstorage_py/results_dir/sentinel.py` — FOUND
- `mlpstorage_py/results_dir/errors.py` — FOUND
- `tests/unit/test_results_dir_schema.py` — FOUND
- `tests/unit/test_results_dir_sentinel.py` — FOUND
- Commit `20d0511` — FOUND (RED)
- Commit `8f01579` — FOUND (GREEN)
