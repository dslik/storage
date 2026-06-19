---
phase: 01-canonical-layout-and-init
plan: 04
subsystem: main + cli + run_summary
tags:
  - python
  - cli
  - dispatch
  - error-handling
  - sentinel
  - lay-03

requires:
  - mlpstorage_py.results_dir.resolve_orgname (Slice 1)
  - mlpstorage_py.results_dir.errors.ResultsDirNotInitializedError (Slice 1)
  - mlpstorage_py.errors.ConfigurationError (existing)
  - mlpstorage_py.errors.ErrorCode.CONFIG_MISSING_REQUIRED (existing)
  - DEFAULT_SYSTEMNAME / --systemname plumbing (Slice 3)
  - generate_output_location canonical-layout signature (Slice 3)

provides:
  - mlpstorage_py.main.NON_BENCHMARK_NO_ORGNAME_MODES (D-12 bypass list)
  - mlpstorage_py.main._main_impl orgname-resolution gate (LAY-03)
  - args.orgname populated for every gated mode before update_args / run_benchmark
  - Banner Tier 1 orgname + systemname rows (LAY-03 / D-10 user-visible surfacing)
  - Banner Environment MLPERF_SYSTEMNAME row (LAY-04 env-var visibility)
  - Pitfall-3 defense-in-depth: Benchmark.__init__ refuses empty args.orgname

affects:
  - 01-05 (test-fixture migration sweep — most paths now canonical; some integration tests still need updates)
  - phase-02 (cluster-collector + systemname.yaml — runs only after gate; args.orgname guaranteed populated)
  - phase-05 (capacity gate + code-image dispatch — both consume args.orgname from the gate)

tech-stack:
  added: []
  patterns:
    - "Dispatch reordering — gate runs after `init`/`version`/`lockfile`/`rules-coverage` bypass branches, before the `history`/`reports`/`validate` branches"
    - "Locked-verbatim error string — backticks around the path are part of the LAY-03 spec; no `!r`, no backslash escapes"
    - "Defense-in-depth guard at Benchmark.__init__ — production never trips it; the main gate populates args.orgname upstream"
    - "Bypass list as frozenset module-constant — adding a bypass is a deliberate code change, not a flag"

key-files:
  created: []
  modified:
    - mlpstorage_py/main.py
    - mlpstorage_py/benchmarks/base.py
    - mlpstorage_py/run_summary.py
    - tests/unit/test_main_orgname_gate.py
    - tests/unit/test_run_summary.py
    - tests/conftest.py
    - tests/unit/test_benchmarks_kvcache.py
    - tests/unit/test_benchmarks_vectordb.py

key-decisions:
  - "Bypass list locked to exactly four modes: {init, version, lockfile, rules-coverage}. history, reports, validate are GATED per D-12; their dispatch branches were physically moved AFTER the gate in _main_impl. validate has no --results-dir attribute in its CLI builder (positional `input` instead), so the gate's `if results_dir_value:` guard quietly skips it — documented as a known divergence below."
  - "Error message uses literal backticks (`<path>`) NOT single quotes — LAY-03 / ROADMAP success criterion #2 is locked verbatim. f-string interpolation with `{var}` (not `{var!r}`) is the correct shape; no backslash escapes (backtick is not a Python escape sequence)."
  - "Pitfall-3 defense added in Benchmark.__init__ immediately after `self.args = args` storage. Internal-only; production callers always have args.orgname populated by the main gate. Caught here a class of test bugs (constructing benchmark instances directly without orgname) — the resolution is to update those fixtures, not to soften the guard."
  - "kvcache test fixture _make_run_benchmark uses mode='open' (NOT 'closed') so the strict CLOSED-mode override checks (seed/trials/inter-option-delay) don't fire on tests that deliberately override those args. TestClosedEnforcement still sets bm.args.mode='closed' explicitly to exercise the enforcement path."

patterns-established:
  - "Sentinel-gate insertion shape: bypass-modes-before, gated-modes-after, gate raises ConfigurationError that flows to main()'s existing handler. Future gates (e.g., capacity gate in Phase 5) should follow this exact shape."
  - "Conftest fixture canonical-layout migration: temp_result_dir, temp_checkpointing_result_dir, sample_benchmark_run_data.result_dir, mock_benchmark_instance.run_result_output all updated. The `<orgname>` segment is `Acme` and `<systemname>` is `sys-v1` in fixtures — same values Slice 3 standardised on."

requirements-completed:
  - LAY-03

duration: ~34min
completed: 2026-06-19
---

# Phase 01 Plan 04: Orgname-resolution gate + banner + canonical fixtures Summary

**LAY-03 enforcement complete. Every command that takes `--results-dir` reads orgname from `<results-dir>/mlperf-results.yaml` at startup, fails before any work begins with the EXACT actionable message from CONTEXT.md when the sentinel is missing, and surfaces orgname + systemname in the run banner.**

## Performance

- **Duration:** ~34 min
- **Started:** 2026-06-19T22:46:14Z
- **Completed:** 2026-06-19T23:20:19Z
- **Tasks:** 2 (both autonomous TDD)
- **Files modified:** 8 (3 source files, 5 test files)

## Accomplishments

- `mlpstorage_py.main._main_impl` gains an LAY-03 orgname-resolution gate between bypass dispatch and gated-command dispatch.
- `NON_BENCHMARK_NO_ORGNAME_MODES = frozenset({"init", "version", "lockfile", "rules-coverage"})` is the locked bypass list (D-12 commitment, exactly four modes).
- Dispatch reordered so `history`, `reports`, `validate` branches now run AFTER the gate — each inherits a resolved `args.orgname` when `--results-dir` is supplied.
- Error message locked verbatim per CONTEXT.md / ROADMAP success criterion #2 — backticks around the path, no single quotes, no backslash escapes.
- `Benchmark.__init__` gains Pitfall-3 defense-in-depth: raises `ConfigurationError` when `args.orgname` is empty/missing.
- `print_run_summary` banner surfaces `orgname` + `systemname` rows in Tier 1 (right under `mode`) and `MLPERF_SYSTEMNAME` in the Environment section.
- `tests/conftest.py` fixtures (`temp_result_dir`, `temp_checkpointing_result_dir`, `sample_benchmark_run_data`, `mock_benchmark_instance`) produce canonical-layout paths and Namespace shapes.
- `tests/unit/test_benchmarks_kvcache.py` and `tests/unit/test_benchmarks_vectordb.py` Namespace fixtures gain `mode/orgname/systemname` (Pitfall 4 — direct benchmark construction needs the guard's prerequisites).
- Full unit-test sweep green: **1425 passed, 4 skipped** (excluding the 4 pre-existing `test_main_warnings.py` failures and 2 pre-existing `test_version.py` failures — both documented in Slice 3 SUMMARY as out of scope, both unchanged).

## Task Commits

Each TDD-pair was committed atomically (TDD gate sequence: `test(01-04)` → `feat(01-04)` per task):

1. **Task 1 RED — failing tests for orgname-resolution gate** — `5fb7627` (`test(01-04)`)
2. **Task 1 GREEN — gate in `_main_impl` + Pitfall-3 defense in `Benchmark.__init__`** — `e5b479c` (`feat(01-04)`)
3. **Task 2 RED — failing tests for orgname + systemname banner rows** — `5ecf739` (`test(01-04)`)
4. **Task 2 GREEN — banner rows + canonical-layout conftest fixtures + benchmark test fixture migration** — `d93b314` (`feat(01-04)`)

## Files Modified

| File | Role |
|------|------|
| `mlpstorage_py/main.py` | Added `NON_BENCHMARK_NO_ORGNAME_MODES` constant; added orgname-resolution gate between bypass dispatch and gated dispatch; moved `history`, `reports`, `validate` branches AFTER the gate so each inherits a resolved `args.orgname` when `--results-dir` is supplied; imports `resolve_orgname` and `ResultsDirNotInitializedError` from `results_dir` package. |
| `mlpstorage_py/benchmarks/base.py` | Pitfall-3 defense-in-depth: raises `ConfigurationError` immediately after `self.args = args` storage when `args.orgname` is empty/missing. Added `ConfigurationError`, `ErrorCode` imports. |
| `mlpstorage_py/run_summary.py` | Added two Tier 1 banner rows (`orgname`, `systemname`) immediately under `mode`; added `MLPERF_SYSTEMNAME` row in the Environment section. |
| `tests/unit/test_main_orgname_gate.py` | 15 new tests covering every LAY-03 gate path: uninitialized-fails, exact-message, gated-commands, env-var-ignored, bypass-list, happy-path, no-flag-leak, Pitfall-3 defense. Adjusted the parametrised `reports` patch to gracefully fall back when `mlpstorage_py.report_generator` cannot import (psutil-free venvs). |
| `tests/unit/test_run_summary.py` | New `TestOrgnameSystemnameBanner` class with three tests: orgname row, systemname row, MLPERF_SYSTEMNAME env row. |
| `tests/conftest.py` | `temp_result_dir` and `temp_checkpointing_result_dir` build canonical layout paths; `sample_benchmark_run_data.result_dir` uses canonical shape; `mock_benchmark_instance` populates `mock.args.mode/orgname/systemname` and a canonical `run_result_output`. |
| `tests/unit/test_benchmarks_kvcache.py` | All four `Namespace(...)` fixture constructions gain `mode/orgname/systemname`. `_make_run_benchmark` uses `mode='open'` so CLOSED-mode strict checks don't fire on tests that override seed/trials/inter-option-delay. |
| `tests/unit/test_benchmarks_vectordb.py` | All five `Namespace(...)` fixture constructions gain `mode/orgname/systemname`. |

## Locked Error Strings (for downstream Slice 5 verification)

```text
Error message:  results-dir `<path>` has not been initialized.
Suggestion:     Run `mlpstorage init <orgname> <path>` first.
```

- Backticks (U+0060) around the path are LOCKED VERBATIM per CONTEXT.md LAY-03 and ROADMAP success criterion #2.
- f-string interpolation uses `{var}` not `{var!r}` (the latter would render `'...'`).
- No backslash escapes — `\`` is not a Python escape sequence; it would render as `\` + backtick.

Smoke verification (live executable run):

```text
$ python3 -m mlpstorage_py.main closed training unet3d datagen file \
    --data-dir /tmp --results-dir /tmp/tmp.X/uninit \
    --systemname sys-v1 --num-processes 1
ERROR: [E101] results-dir `/tmp/tmp.X/uninit` has not been initialized.
  Suggestion: Run `mlpstorage init <orgname> /tmp/tmp.X/uninit` first.
INFO: Suggestion: Run `mlpstorage init <orgname> /tmp/tmp.X/uninit` first.
```

## Bypass List (CONTEXT.md D-12, exactly four modes)

| Mode | Why bypassed |
|------|--------------|
| `init` | Bootstraps the sentinel — can't read what it's about to create. |
| `version` | Reads `mlpstorage_py/VERSION` and exits; no filesystem interaction. |
| `lockfile` | Generates/verifies `requirements.txt`-style lockfile; uses its own paths, not `--results-dir`. |
| `rules-coverage` | Reports submission-checker rules coverage; does not consume the canonical layout. |

Every other mode (`closed`, `open`, `whatif`, `reports`, `history`, `validate`) is gated. See "Known Divergence" below for the `validate`-specific note.

## Insertion-Point Line Numbers (post-edit)

In `mlpstorage_py/main.py` (after Slice 4 edits):

- Bypass dispatches: `init` (line ~285), `version` (~289), `lockfile` (~302), `rules-coverage` (~305) — all BEFORE the gate.
- LAY-03 gate: lines ~317-336 (the `if args.mode not in NON_BENCHMARK_NO_ORGNAME_MODES:` block).
- Gated dispatches (moved AFTER the gate): `history` (~340-361), `reports` (~363-369), `validate` (~371-373).
- `run_datetime = datetime_str` (~375), `update_args(args)` (~378).

## Decisions Made

- **Bypass list locked.** Exactly `{init, version, lockfile, rules-coverage}` — four modes. Encoded as a `frozenset` module-constant. Adding a mode to bypass is a deliberate code change, not a CLI flag. Test `test_bypass_commands_skip_gate` is parametrised over exactly these four modes.
- **Dispatch reordered, not merely guarded.** The plan instructed me to physically move `history`, `reports`, `validate` branches BELOW the gate so the code structure makes it impossible to fall through them without resolving `args.orgname` first. This is preferred over scattered `if args.mode in {...}: skip_gate_for_me` guards — the linear flow now is "bypass branches → gate → gated branches → benchmark runner".
- **Locked backtick error string.** Used a plain Python f-string with `{var}` (not `{var!r}`) and plain backticks. No backslash escapes (backtick is not a Python escape sequence). The verbatim string assertion in `test_failure_message_text` is the canonical guard.
- **Pitfall-3 defense raises with `CONFIG_MISSING_REQUIRED`.** Matches the gate's own raise so the user-facing severity is consistent if the guard ever fires (it shouldn't in production).
- **kvcache fixture `mode='open'`, not `'closed'`.** Selected because `TestExecuteRun` and friends deliberately override seed/trials/inter-option-delay. Keeping `mode='closed'` would trigger CLOSED-mode strictness and break those tests. `TestClosedEnforcement` sets `bm.args.mode='closed'` explicitly after fixture setup, so the closed-mode path is still exercised.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] `test_main_orgname_gate.py::test_gated_commands_fail_uninitialized[reports]` AttributeError on `mlpstorage_py.report_generator`**

- **Found during:** Task 1 GREEN run.
- **Issue:** The test patches `"mlpstorage_py.report_generator.ReportGenerator"`. The `report_generator` submodule is lazy-imported inside the `args.mode == "reports"` branch in `main.py`, so before that branch runs, `mlpstorage_py.report_generator` does not exist as an attribute on the `mlpstorage_py` package. `patch()` then raises `AttributeError: module 'mlpstorage_py' has no attribute 'report_generator'`. Worse — `report_generator` transitively imports `psutil`, which is not installed in this venv.
- **Fix:** Wrapped the patch context in a `try: import mlpstorage_py.report_generator / except ModuleNotFoundError:` and provide a psutil-free fallback path that omits the report_generator patch but still asserts the gate's behavior. The `reports` mode is still parametrised in this venv; in a `[full]`-extras install both paths reach the same gate.
- **Files modified:** `tests/unit/test_main_orgname_gate.py`.
- **Committed in:** `e5b479c` (alongside Task 1 GREEN).

**2. [Rule 3 — Blocking] `test_benchmarks_kvcache.py` + `test_benchmarks_vectordb.py` ConfigurationError on missing orgname (Pitfall 4 — fixture migration)**

- **Found during:** Task 1 GREEN regression sweep — 24 errors + 16 failures in `test_benchmarks_kvcache.py`, 9 more in `test_benchmarks_vectordb.py`.
- **Issue:** Both test files construct `Namespace(...)` fixtures and pass them directly into `KVCacheBenchmark` / `VectorDBBenchmark` (bypassing `_main_impl`'s gate). Once the Pitfall-3 defense landed in `Benchmark.__init__`, every fixture missing `orgname` started failing in setup.
- **Fix:** Added `mode='closed' / 'open'`, `orgname='Acme'`, `systemname='sys-v1'` to every `Namespace(...)` block in both files. The kvcache `_make_run_benchmark` helper uses `mode='open'` (NOT `'closed'`) so the strict CLOSED-mode override checks in `KVCacheBenchmark._execute_run` (seed/trials/inter-option-delay) don't fire on tests that deliberately override those args. `TestClosedEnforcement` still sets `bm.args.mode='closed'` explicitly after fixture setup.
- **Files modified:** `tests/unit/test_benchmarks_kvcache.py`, `tests/unit/test_benchmarks_vectordb.py`.
- **Verification:** `pytest tests/unit/test_benchmarks_kvcache.py tests/unit/test_benchmarks_vectordb.py` → 71 passed.
- **Committed in:** `d93b314` (alongside Task 2 GREEN).

---

**Total deviations:** 2 auto-fixed (both Rule 3 — blocking issues caused by this slice's own changes). The plan's Task 2 explicitly authorised both (Pitfall 4 instructs me to fix downstream literal-asserting / fixture-shape tests in the same slice).

## Known Divergence — `validate` mode

The plan's `must_haves.truths` says the gate "applies to every command in D-12 scope: closed, open, whatif, reportgen, validate, history — ALL six must fail-fast on missing sentinel". The implemented gate handles 5 of these 6 fully; `validate` is a partial case:

- **What we implemented:** `validate` is NOT in `NON_BENCHMARK_NO_ORGNAME_MODES`, so the gate's `if args.mode not in NON_BENCHMARK_NO_ORGNAME_MODES:` check passes. Inside the gate, `getattr(args, 'results_dir', None)` returns `None` for validate (its CLI builder uses a positional `input` argument, not `--results-dir`). The gate's `if results_dir_value:` guard then quietly skips the sentinel check.
- **Why this is fine for Phase 1:** `validate` doesn't WRITE to `--results-dir`; it reads from `input` (a submission directory the user is validating). It does not consume `args.orgname` and does not produce canonical-layout output.
- **What the test does:** `test_gated_commands_fail_uninitialized` deliberately omits `validate` from its `@pytest.mark.parametrize` list (line 167 of `test_main_orgname_gate.py`) with an inline comment explaining the rationale: "validate is NOT in this parameter list because the existing add_validate_arguments builder does not register --results-dir".
- **Follow-up:** If a future plan adds `--results-dir` to `add_validate_arguments`, the gate will automatically pick it up — no code change needed. Until then this is a documented inert-gate divergence.

## Pre-Existing Failures (out of scope, unchanged)

Same set Slice 3 SUMMARY documented:

- `tests/unit/test_main_warnings.py` — 4 failures (`AttributeError: module 'mlpstorage_py' has no attribute 'benchmarks'`) caused by missing optional benchmark extras (DLIO etc.) in this dev shell. Verified unchanged on parent of this slice; reproducible with `git stash`.
- `tests/unit/test_version.py` — 2 failures (version-pin mismatch: assertion expects `3.0.13`, code reports `3.0.12`). Pre-existing on `main`.
- Three test files cannot be collected in this dev shell due to missing optional deps (`psutil`, `pyarrow`, `numpy`): `test_benchmarks_base.py`, `test_parquet_reader.py`, `test_vdb_modular_fake_backend.py`, plus `test_datagen_command_generation.py` indirectly. Logged in `deferred-items.md`.

## Conftest Fixture Enumeration

The plan flagged RESEARCH.md lines 329, 346, 408, 473. Verifying by name:

| Plan line ref | Fixture name (actual) | Updated? |
|---------------|----------------------|----------|
| ~329 | `sample_benchmark_run_data` | yes — `result_dir` now canonical |
| ~346 | `temp_result_dir` | yes — path now canonical |
| ~408 | `temp_checkpointing_result_dir` | yes — path now canonical (omits `<command>` per Slice 3's checkpointing convention) |
| ~473 | `mock_benchmark_instance` | yes — sets `mock.args.mode/orgname/systemname` AND canonical `run_result_output` |

All four flagged fixtures are now canonical. No "fourth fixture surprise" — the line-number drift was within ±5 lines; the four-fixture set is exactly the plan's set.

## Acceptance Gates from `<acceptance_criteria>`

**Task 1**

| Gate | Result |
|------|--------|
| `test_uninitialized_results_dir_fails` passes | green |
| `test_failure_message_text` passes (backticks verbatim) | green |
| `test_gated_commands_fail_uninitialized` passes (5 of 6 gated modes — `validate` documented divergence) | green (5/5 parameter cases pass; `validate` not parametrised per known-divergence rationale) |
| `test_env_orgname_ignored` passes | green |
| `test_bypass_commands_skip_gate` passes (4 bypass modes) | green |
| `test_initialized_dir_resolves_orgname_to_args` passes | green |
| `test_no_orgname_flag_on_non_init_commands` passes | green |
| `test_benchmark_init_raises_when_orgname_missing` skipped (TrainingBenchmark requires optional deps; Pitfall-3 path is exercised via the indirect kvcache/vectordb fixture failures Task 2 had to fix) | skipped |
| `grep -v '^#' main.py \| grep -c MLPERF_ORGNAME` is `0` | 0 |
| `grep -rn "\-\-orgname" mlpstorage_py/cli/ ... main.py \| grep -v '^[^:]*:[[:space:]]*#'` is empty | empty |
| Manual smoke (`mlpstorage closed training ... --results-dir <uninit> ...`) emits "has not been initialized" | ok (verbatim above) |

**Task 2**

| Gate | Result |
|------|--------|
| `pytest test_run_summary.py -x` overall | 13/13 green |
| `pytest tests/unit -k "not init and not results_dir and not orgname_gate"` shows zero new regressions | 1425 passed (+1 added for test_imports) |
| `grep -v '^#' run_summary.py \| grep -c orgname` >= 1 | 3 |
| `grep -v '^#' run_summary.py \| grep -c systemname` >= 1 | 3 |
| `grep -v '^#' run_summary.py \| grep -c MLPERF_SYSTEMNAME` >= 1 | 1 |
| `grep -nE "^def (temp_result_dir\|mock_benchmark_instance\|sample_benchmark_run_data)" conftest.py \| wc -l` is `3` | 3 |

## Threat Mitigations Verified

| Threat ID | Mitigation | Test |
|-----------|------------|------|
| T-1-03 | Inherited from Slice 1: Pydantic orgname regex `r"^[A-Za-z0-9._-]+$"` rejects path-traversal payloads BEFORE they reach `args.orgname` | Slice 1 schema tests + `test_initialized_dir_resolves_orgname_to_args` confirms happy-path |
| T-1-G1 | `MLPERF_ORGNAME` env var never read by main.py — grep gate `0`, test `test_env_orgname_ignored` sets the env var and confirms the gate still fails uninitialized | green |
| T-1-G2 | Bypass list as frozenset module-constant `NON_BENCHMARK_NO_ORGNAME_MODES` — parametrised test covers all four modes | green |
| T-1-G3 | Banner discloses orgname — accepted per plan, banner-only is local terminal output | documented |
| T-1-SC | N/A — zero new packages | `git diff pyproject.toml` empty |

## Known Stubs

None. The orgname-resolution gate is fully wired into `_main_impl`; downstream consumers (`generate_output_location`, `Benchmark.__init__`, `run_summary`) read `args.orgname` directly with no remaining gaps.

## Threat Flags

None new. The gate adds a new code path that reads the sentinel file at startup; the threat surface (read from `<results-dir>/mlperf-results.yaml`) was already registered in Slice 1's threat model (T-1-03) and is still mitigated by the Pydantic schema regex from Slice 1.

## User Setup Required

None — no external service configuration changes. The `MLPERF_SYSTEMNAME` env var (now visible in the banner Environment section) was already introduced in Slice 3 and is optional.

## Next Phase Readiness

- **Plan 01-05 (test-fixture migration sweep + final integration):** Ready. Most `tests/unit/` literal-path assertions and benchmark fixtures are now canonical. Slice 5 will primarily handle:
  1. Submission-checker fixture tree (`tests/fixtures/sample_results/` etc.) for end-to-end validation.
  2. Any remaining integration tests that build literal paths from the old layout.
  3. The `validate` command's `--results-dir` decision (gated or genuinely bypass) — currently inert-gated, see Known Divergence above.
- **Phase 02 (cluster collection + `systemname.yaml`):** Unblocked. `args.orgname` is guaranteed populated before `Benchmark.__init__` runs, so the collector can safely write `<results-dir>/<mode>/<orgname>/results/<systemname>/systemname.yaml`.

## Self-Check: PASSED

Verified before sealing:

- `git log --oneline -4` shows the four task commits: `5fb7627` (T1 RED), `e5b479c` (T1 GREEN), `5ecf739` (T2 RED), `d93b314` (T2 GREEN). TDD gate sequence honored.
- `pytest tests/unit/test_main_orgname_gate.py tests/unit/test_run_summary.py` → 27 passed, 1 skipped.
- `pytest tests/unit --ignore=<env-broken> --ignore=<pre-existing>` → 1425 passed, 4 skipped.
- Manual smoke command emits the exact LAY-03 error string verbatim (backticks around path).
- All grep acceptance gates pass: `MLPERF_ORGNAME` not referenced in main.py; no `--orgname` leak; banner shows orgname/systemname/MLPERF_SYSTEMNAME.

---
*Phase: 01-canonical-layout-and-init*
*Plan: 04*
*Completed: 2026-06-19*
