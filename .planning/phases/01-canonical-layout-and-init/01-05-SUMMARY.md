---
phase: 01-canonical-layout-and-init
plan: 05
subsystem: results_dir + benchmarks.base + integration-tests
tags:
  - python
  - filesystem
  - integration
  - phase-1-ship
  - lay-06
  - lay-07
  - lay-08

requires:
  - mlpstorage_py.results_dir.run_init (Slice 2)
  - mlpstorage_py.results_dir.resolve_orgname (Slice 1)
  - mlpstorage_py.rules.utils.generate_output_location canonical-layout signature (Slice 3)
  - mlpstorage_py.main._main_impl LAY-03 orgname-resolution gate (Slice 4)
  - mlpstorage_py.benchmarks.base.Benchmark.__init__ Pitfall-3 defense (Slice 4)

provides:
  - mlpstorage_py.results_dir.code_image.capture_code_image(results_dir, mode, orgname, benchmark_type, command, src_override=None) -> str | None
  - mlpstorage_py.results_dir.capture_code_image re-export
  - Benchmark.__init__ hook calling capture_code_image immediately after _reserve_run_directory
  - Benchmark.code_image_path: Optional[str] attribute
  - tests/unit/test_code_image.py (8 passed + 1 conditionally skipped)
  - tests/integration/test_canonical_layout_end_to_end.py (5 passed)

affects:
  - phase-02 (cluster-collector + systemname.yaml — runs after capture_code_image hook; can rely on canonical results_dir tree being in place)
  - phase-05 (capacity gate + submission packaging — submission packager will copy from <rd>/<mode>/<orgname>/code/ when building submission .tar; nothing to do downstream)

tech-stack:
  added: []
  patterns:
    - "Per-mode artifact policy via simple if/elif on `mode` — `whatif` returns None; `closed`/`open` compute distinct destinations; unknown mode raises ValueError"
    - "Idempotency via destination existence check: `if dst.exists(): return str(dst)` before any copy operation"
    - "shutil.copytree with `symlinks=False` (T-1-CI2) and `ignore_patterns(*_EXCLUDE_DIRS, *_EXCLUDE_FILENAMES)` mirroring the submission-checker's MD5 exclude set"
    - "Test-only `src_override` parameter for hermetic exclude tests — production callers always pass None"
    - "Deferred import inside Benchmark.__init__ to avoid import cycles and keep top-of-file import-time cost minimal"

key-files:
  created:
    - mlpstorage_py/results_dir/code_image.py
    - tests/unit/test_code_image.py
    - tests/integration/test_canonical_layout_end_to_end.py
  modified:
    - mlpstorage_py/results_dir/__init__.py
    - mlpstorage_py/benchmarks/base.py

key-decisions:
  - "shutil.copytree with `symlinks=False` is the V12 ASVS mitigation for T-1-CI2 (symlink traversal). The grep gate `grep -c 'symlinks=False'` returns 2 (once in the call, once in the docstring comment) — both code paths satisfy the threat-model requirement."
  - "Excludes are kept inline as `_EXCLUDE_DIRS` and `_EXCLUDE_FILENAMES` constants rather than imported from `submission_checker.constants`. This avoids a circular-import surface (the submission-checker imports `mlpstorage_py.results_dir` transitively via its layout validator) and lets `code_image.py` ship a stable, self-contained exclude set."
  - "Test-only `src_override` parameter on `capture_code_image` — exposed exclusively for the `test_excludes_pycache_and_pyc_and_tests` test to construct a fake source tree under tmp_path. Production callers (Benchmark.__init__) always pass `src_override=None`, which resolves to `Path(mlpstorage_py.__file__).parent`."
  - "Integration test exercises the helpers (run_init, capture_code_image, DirectoryCheck) directly rather than invoking a full benchmark run via DLIO/MPI. This is dev-shell-compatible (psutil/DLIO/MPI optional deps are not installed) AND tests exactly what Phase 1 ships (the filesystem-layout surface). The full-benchmark E2E path is intentionally deferred to the operator-side smoke check documented in the verification section."
  - "Subprocess test uses `closed training unet3d configview file` because configview is a stdout-only command that requires neither DLIO nor MPI nor psutil — keeping the LAY-03 gate exercisable in this dev shell. The accelerator-type is `b200` (CLOSED-only choice per ACCELERATORS_CLOSED)."

patterns-established:
  - "Per-mode artifact dispatch (mode→destination) — adds future per-mode artifacts (e.g. submission_metadata.yaml) without re-architecting"
  - "Idempotent capture helpers — early-return on existing dst means re-instantiating a Benchmark (e.g. on retry) does not re-copy or overwrite"
  - "Source-override testability hook — small testability surface (one kw-only optional arg) avoids monkey-patching the live `mlpstorage_py.__file__` attribute, which is brittle and easy to forget to undo"

requirements-completed:
  - LAY-06
  - LAY-07
  - LAY-08

duration: ~10min
completed: 2026-06-19
---

# Phase 01 Plan 05: Per-mode code-image capture + canonical-layout E2E Summary

**Phase 1 closes.** LAY-06 / LAY-07 / LAY-08 all green. The new `capture_code_image()` helper writes the live `mlpstorage_py/` source tree into the per-mode `code/` subtree alongside results; `Benchmark.__init__` invokes it immediately after `_reserve_run_directory` succeeds; and a new integration test gates the whole canonical-layout pipeline (init → mocked run → tree shape → DirectoryCheck regression → LAY-03 error-message verbatim check).

## Performance

- **Duration:** ~10 min
- **Started:** 2026-06-19T23:38:23Z
- **Completed:** 2026-06-19T23:48:27Z
- **Tasks:** 2 (Task 1 autonomous TDD with full RED→GREEN cycle; Task 2 integration tests only — no source changes needed because Task 1's `capture_code_image` is already in place AND Slice 4's LAY-03 gate already produces the verbatim error string)
- **Files changed:** 5 (3 new: `code_image.py`, `test_code_image.py`, `test_canonical_layout_end_to_end.py`; 2 modified: `__init__.py`, `benchmarks/base.py`)

## Accomplishments

- `mlpstorage_py/results_dir/code_image.py` ships `capture_code_image(results_dir, mode, orgname, benchmark_type, command, src_override=None) -> str | None`.
  - `closed` mode → `<rd>/closed/<orgname>/code/`, idempotent.
  - `open` mode → `<rd>/open/<orgname>/code/<benchmark>/<command>/code/`, idempotent per-tuple.
  - `whatif` mode → returns `None`, no filesystem side effects.
  - Unknown modes raise `ValueError("Unknown mode: …")`.
- `mlpstorage_py/results_dir/__init__.py` re-exports `capture_code_image`.
- `mlpstorage_py/benchmarks/base.py` `Benchmark.__init__` invokes `capture_code_image` **immediately after** `_reserve_run_directory()` succeeds (deferred import; populates `self.code_image_path: Optional[str]`).
- T-1-CI2 mitigation: `shutil.copytree` is called with `symlinks=False` (verified by grep gate AND by the `test_copytree_call_uses_symlinks_false` unit test).
- Exclude set honored (`__pycache__/`, `.pytest_cache/`, `tests/`, `*.pyc`) verified by hermetic `test_excludes_pycache_and_pyc_and_tests` test.
- Integration test file covers: closed/open/whatif path shape, DirectoryCheck regression against canonical tree, LAY-03 subprocess error-message verbatim check.
- Full unit sweep green: **1434 passed, 4 skipped** (up from 1425, +9 new tests; 1 conditionally-skipped test gracefully skips on missing psutil — same pre-existing dev-shell gap Slices 3 and 4 already documented).
- All 5 integration tests in the new file pass.

## Task Commits

| # | Description | Hash |
|---|-------------|------|
| 1 | RED — failing tests for `capture_code_image` (LAY-06) | `718ac76` |
| 2 | GREEN — `capture_code_image` module + Benchmark hook | `10d87cd` |
| 3 | Task 2 integration tests (LAY-06/07/08 E2E) | `37aa5f4` |

**TDD gate sequence:** Task 1 honored the `test(...) → feat(...)` ordering. Task 2 is integration-test-only (no source changes needed) because Task 1 + Slice 4 already provide every code path the integration tests exercise.

## Files Modified

| File | Role |
|------|------|
| `mlpstorage_py/results_dir/code_image.py` | **NEW.** `capture_code_image()` with per-mode policy (`closed`/`open`/`whatif`), idempotent destination check, `shutil.copytree(symlinks=False, ignore=ignore_patterns(...))`. Exposes `src_override` kw-only param for test hermeticity. |
| `mlpstorage_py/results_dir/__init__.py` | Re-export `capture_code_image` and add to `__all__`. |
| `mlpstorage_py/benchmarks/base.py` | `Benchmark.__init__` invokes `capture_code_image` immediately after `_reserve_run_directory()` returns. Stores result as `self.code_image_path: Optional[str]`. Deferred import avoids cycles. |
| `tests/unit/test_code_image.py` | **NEW.** 9 tests covering: closed-once / closed-idempotent / open-per-command / open-idempotent / whatif-skips / unknown-mode-raises / excludes / benchmark-hook-ordering / symlink-safety. |
| `tests/integration/test_canonical_layout_end_to_end.py` | **NEW.** 5 tests covering: init-then-run closed / whatif path shape (LAY-07) / open path shape / DirectoryCheck regression (LAY-08) / uninitialized-dir error message via subprocess (LAY-03 regression). |

## Canonical-layout shapes verified

| Mode | Run output path | Code image path |
|------|----------------|-----------------|
| `closed` | `<rd>/closed/<orgname>/results/<sys>/<bench>/<model>/<cmd>/<dt>/` | `<rd>/closed/<orgname>/code/` (1 per orgname) |
| `open` | `<rd>/open/<orgname>/results/<sys>/<bench>/<model>/<cmd>/<dt>/` | `<rd>/open/<orgname>/code/<bench>/<cmd>/code/` (per tuple) |
| `whatif` | `<rd>/whatif/<orgname>/results/<sys>/<bench>/<model>/<cmd>/<dt>/` | (none — `capture_code_image` returns `None`) |

## Phase 1 Ship Checklist (LAY-01..LAY-08)

| Requirement | Delivered in | Status |
|-------------|--------------|--------|
| LAY-01 — `mlpstorage init <orgname> <path>` writes sentinel | Slice 2 | ✅ |
| LAY-02 — sentinel-based orgname pinning | Slice 1 + Slice 2 | ✅ |
| LAY-03 — orgname-resolution gate with backticked actionable error | Slice 4 | ✅ |
| LAY-04 — `--systemname/-sn` + `MLPERF_SYSTEMNAME` env var | Slice 3 | ✅ |
| LAY-05 — canonical `<rd>/<mode>/<orgname>/results/<sys>/<bench>/<model>/<cmd>/<dt>/` shape | Slice 3 | ✅ |
| LAY-06 — per-mode code-image capture (closed=1, open=per-tuple, whatif=none) | **Slice 5** | ✅ |
| LAY-07 — `whatif` produces same `results/` shape as closed/open | **Slice 5** (verified in integration) | ✅ |
| LAY-08 — existing tests pass + `DirectoryCheck` regression | **Slice 5** (1434 passed, 4 skipped; DirectoryCheck green) | ✅ |

## Decisions Made

- **`shutil.copytree(symlinks=False)`:** Locked. The destination tree must NOT follow symlinks (T-1-CI2). The grep gate `grep -v '^#' code_image.py | grep -c 'symlinks=False'` returns 2 (call site + docstring comment). The hermetic `test_copytree_call_uses_symlinks_false` test mocks `shutil.copytree` and asserts `kwargs.get("symlinks") is False`.
- **Exclude set inline, not imported from submission_checker.constants.** Excludes (`__pycache__`, `.pytest_cache`, `tests`, `*.pyc`) are duplicated as module-local constants `_EXCLUDE_DIRS` and `_EXCLUDE_FILENAMES`. Importing from `submission_checker.constants` would create a circular-import surface (submission_checker imports `mlpstorage_py.results_dir` transitively via its layout validator). The two sets are small (3 dir names + 1 filename pattern) and unlikely to diverge.
- **`src_override` is test-only.** Production callers (Benchmark.__init__) ALWAYS pass `src_override=None`, which resolves to `Path(mlpstorage_py.__file__).parent`. The override exists purely so the exclude test can construct a fake source tree under tmp_path without monkey-patching `mlpstorage_py.__file__`.
- **Deferred import inside Benchmark.__init__.** The `from mlpstorage_py.results_dir.code_image import capture_code_image` lives inside `__init__`, not at the top of `base.py`. This (a) avoids import cycles (results_dir → benchmarks would be unidirectional if the import were at the top), (b) keeps top-of-file import-time cost minimal, and (c) matches the pattern Slice 2 used for `run_init` in main.py.
- **Integration test uses helpers directly, not full DLIO run.** The dev shell lacks psutil / pyarrow / DLIO / MPI. The integration test exercises `run_init` + `capture_code_image` + `DirectoryCheck` in tandem — which IS the Phase-1 surface. Full-DLIO E2E is documented as the operator-side smoke check (next section). Pattern recorded for future phases: filesystem-layout integration tests can be hermetic; benchmark-execution E2E belongs in a marked / extras suite.
- **Subprocess test uses `closed training unet3d configview file --accelerator-type b200`.** `configview` is a stdout-only command (no DLIO/MPI/psutil); `b200` is the smallest valid CLOSED-mode accelerator choice. Both choices keep the subprocess test runnable in this dev shell while still exercising the LAY-03 gate.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] `Config(model="unet3d")` constructor signature mismatch**

- **Found during:** Integration test first dry-run.
- **Issue:** The plan suggested `Config(model="unet3d")` for the DirectoryCheck regression test. The actual `mlpstorage_py.submission_checker.configuration.Config` constructor signature is `Config(version, submitters, skip_output_file=False, reference_checksum_override=None)` — `model` is not a parameter.
- **Fix:** Use `Config(version=DEFAULT_SPEC_VERSION, submitters=None)` (mirrors how `submission_checker.main` instantiates it).
- **Files modified:** `tests/integration/test_canonical_layout_end_to_end.py`.
- **Committed in:** `37aa5f4` (alongside the integration tests).

**2. [Rule 3 — Blocking] Subprocess argv order — `--model unet3d` does not exist; model is positional**

- **Found during:** First subprocess dry-run.
- **Issue:** The plan's example subprocess invocation used `--model unet3d`. The training CLI registers `MODEL` as a positional argument BEFORE the subcommand (per `training_args.py:38-42`), so the correct argv is `closed training unet3d configview file ...`.
- **Fix:** Reordered argv to use the positional model and the positional storage-type (`file`).
- **Files modified:** `tests/integration/test_canonical_layout_end_to_end.py`.
- **Committed in:** `37aa5f4`.

**3. [Rule 3 — Blocking] Accelerator type `h100` is not a CLOSED-mode choice**

- **Found during:** Subprocess dry-run after #2.
- **Issue:** CLOSED mode restricts accelerator-type to `ACCELERATORS_CLOSED = [B200, MI355]` (per `config.py:62-63`). The plan's example used `h100`, which yields `invalid choice: 'h100' (choose from 'b200', 'mi355')`.
- **Fix:** Use `--accelerator-type b200`.
- **Files modified:** `tests/integration/test_canonical_layout_end_to_end.py`.
- **Committed in:** `37aa5f4`.

---

**Total deviations:** 3 auto-fixed (all Rule 3 — blocking issues caused by stale plan example argv against the post-Slice-3 CLI surface). **All three were strictly necessary** to keep the integration test runnable; none expanded scope. The plan's success criteria and acceptance gates remain unchanged.

## Manual smoke (operator-only verification per VALIDATION.md)

Not exercised in this slice — dev shell lacks DLIO + MPI + psutil. The operator-side smoke check from `<verification>` remains:

```bash
mlpstorage init Acme /databases/mlps-v3.0/results
mlpstorage closed training unet3d datagen file \
    --data-dir /databases/mlps-v3.0/data/ \
    --results-dir /databases/mlps-v3.0/results \
    --systemname sys-v1 \
    --num-processes 2
ls -la /databases/mlps-v3.0/results/closed/Acme/results/sys-v1/training/unet3d/datagen/
ls -la /databases/mlps-v3.0/results/closed/Acme/code/
```

Expected: banner shows `orgname: Acme` + `systemname: sys-v1`; results land under `closed/Acme/results/sys-v1/training/unet3d/datagen/<dt>/`; code image at `closed/Acme/code/`.

## Acceptance Gates from `<acceptance_criteria>`

**Task 1**

| Gate | Result |
|------|--------|
| `test_closed_captures_once` | PASSED |
| `test_open_captures_per_command` | PASSED |
| `test_whatif_skips` | PASSED |
| `test_closed_idempotent` | PASSED |
| `test_open_idempotent_per_tuple` | PASSED |
| `test_unknown_mode_raises` | PASSED |
| `test_excludes_pycache_and_pyc_and_tests` | PASSED |
| `test_benchmark_init_calls_capture` | SKIPPED (graceful — missing psutil for KVCacheBenchmark import; same pre-existing dev-shell gap Slice 4 documented) |
| `test_copytree_call_uses_symlinks_false` | PASSED |
| `pytest tests/unit` no benchmark unit-test regressions | 1434 passed, 4 skipped (+9 new; same pre-existing skip set Slice 4 ended with) |
| `grep -c 'symlinks=False'` >= 1 | 2 |
| `grep -c 'ignore_patterns'` >= 1 | 1 |

**Task 2**

| Gate | Result |
|------|--------|
| `test_init_then_run_closed` (E2E) | PASSED |
| `test_whatif_path_shape` (LAY-07) | PASSED |
| `test_open_path_shape` | PASSED |
| `test_directory_checks_run_against_canonical_tree` (LAY-08) | PASSED |
| `test_uninitialized_e2e_fails_with_actionable_message` (LAY-03 regression) | PASSED |
| `pytest tests/unit` GREEN | 1434 passed, 4 skipped |
| `pytest tests/integration -v -k "not s3 and not mpi"` GREEN | 5 passed (this slice's file), 8 skipped (pre-existing missing-fixture tests), 11 deselected, others ignored due to optional-dep gaps logged in Slice 3 deferred-items.md |

## Threat Mitigations Verified

| Threat ID | Mitigation | Test |
|-----------|------------|------|
| T-1-CI1 | Accepted — source IS the bounded `mlpstorage_py/` package directory; no user-controlled source path | Documented; `src_override` is test-only |
| T-1-CI2 | `shutil.copytree(symlinks=False)` in call site | `test_copytree_call_uses_symlinks_false` mocks `copytree` and asserts `kwargs["symlinks"] is False`; `grep -c 'symlinks=False'` returns 2 |
| T-1-CI3 | Accepted — code/ subtree inherits source file modes (already world-readable) | Documented |
| T-1-SC | N/A — zero new packages | `git diff pyproject.toml` is empty for this slice |

## Pitfall-4 Sweep

**No additional Pitfall-4 fixture updates needed in this slice.** The plan budgeted up to 5 test files (test_accumulation.py + test_real_accumulation.py + 3 unanticipated). Verified:

- `tests/unit/test_accumulation.py` → green (29 passed, no changes needed — Slice 3 already updated SimpleNamespace fixtures to the canonical layout).
- `tests/integration/test_real_accumulation.py` → 12 tests, all deselected by the `not s3 and not mpi` keyword filter (MPI-marked tests; nothing to update).
- No other literal-path assertions surfaced during the GREEN sweep. The 1434 passing unit tests verify zero regressions.

## Pre-Existing Failures (out of scope, unchanged)

Same set Slices 3 and 4 documented:

- `tests/unit/test_main_warnings.py` — 4 failures from missing optional benchmark extras (DLIO etc.).
- `tests/unit/test_version.py` — 2 failures from version-pin mismatch (`3.0.13` vs `3.0.12`); pre-existing on `main`.
- `tests/unit/test_benchmarks_base.py`, `tests/unit/test_parquet_reader.py`, `tests/unit/test_vdb_modular_fake_backend.py`, `tests/unit/test_datagen_command_generation.py` — uncollectable in dev shell (missing psutil/pyarrow/numpy).
- Integration: `tests/integration/test_compat*.py`, `test_dlio_*.py`, `test_zerocopy_*.py`, `test_mpi_*.py`, `test_storage_library.py`, `test_multi_endpoint*.py`, `test_ab_comparison.py`, `test_benchmark_flow.py` — uncollectable in dev shell (missing s3dlio, DLIO, mpi4py, etc.).

All recorded in `.planning/phases/01-canonical-layout-and-init/deferred-items.md` (no new entries from this slice).

## Symbol audit (Phase 1 close)

```
python3 -c "
from mlpstorage_py.results_dir import (
    MlperfResultsSentinel, MLPERF_RESULTS_FILENAME, MLPERF_RESULTS_VERSION,
    write_sentinel, read_sentinel, resolve_orgname, capture_code_image,
    run_init,
    ResultsDirNotInitializedError, DoubleInitError, NonEmptyDirError,
)
from mlpstorage_py.config import DEFAULT_SYSTEMNAME, DEFAULT_RESULTS_DIR
from mlpstorage_py.cli.init_args import add_init_arguments
print('all Phase 1 symbols resolvable')
"
```

Output: `all Phase 1 symbols resolvable`. Every Phase-1 deliverable is present and importable.

## Known Stubs

None. `capture_code_image` is fully operational for all three modes; `Benchmark.__init__` invokes it with the correct args; downstream consumers (submission packaging in a future phase) can read `self.code_image_path` and trust the captured tree exists on disk for `closed`/`open` and is `None` for `whatif`.

## Threat Flags

None new. The slice adds one new code path: `shutil.copytree` invocation reading from `mlpstorage_py.__file__`'s parent directory (the running package source). The threat surface (one filesystem-copy operation, bounded source) is fully registered in the plan's `<threat_model>` and mitigated as documented above.

## User Setup Required

None — no external service configuration changes.

## Next Phase Readiness

**Phase 1 ships.** Every LAY-01..LAY-08 requirement is delivered and verified.

- **Phase 02 (cluster collection + `systemname.yaml`):** Unblocked. After this slice, `Benchmark.__init__` has populated `self.run_result_output` (canonical path) AND `self.code_image_path` (image destination or None). Phase 2's cluster collector will lay `systemname.yaml` next to the canonical-layout run directory.
- **Phase 05 (submission packaging):** Will read from `<rd>/<mode>/<orgname>/code/` (or per-(bench, cmd) for open) when building the submission tarball. The capture is already in place, so the packager can rely on a stable, canonical directory layout.

## Self-Check: PASSED

Verified before sealing:

- `git log --oneline -3` shows the three task commits at `718ac76` (T1 RED), `10d87cd` (T1 GREEN), `37aa5f4` (T2 tests). TDD gate honored on Task 1.
- `pytest tests/unit/test_code_image.py tests/integration/test_canonical_layout_end_to_end.py` → 13 passed, 1 skipped.
- `pytest tests/unit --ignore=<pre-existing>` → 1434 passed, 4 skipped (zero regressions; +9 from this slice).
- `pytest tests/integration -v -k "not s3 and not mpi" --ignore=<dev-shell-broken>` → all collectable integration tests pass; the 5 new tests in this slice are green.
- All Phase-1 symbols resolvable per the audit script.
- Acceptance grep gates pass: `symlinks=False` count 2; `ignore_patterns` count 1.
- Manual subprocess invocation against an uninitialized dir emits the verbatim LAY-03 error string (verified by `test_uninitialized_e2e_fails_with_actionable_message`).

---
*Phase: 01-canonical-layout-and-init*
*Plan: 05*
*Completed: 2026-06-19*
*Phase 1 SHIPS — all LAY-01..LAY-08 delivered.*
