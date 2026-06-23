---
phase: 03-chassis-model-networking-coverage
plan: 02
subsystem: collector
tags: [python, sysfs, dmi, mpi, collector, frozenset, path-indirection]

# Dependency graph
requires:
  - phase: 03-chassis-model-networking-coverage
    plan: 01
    provides: "NetworkPort.state schema lock; Phase 3 schema gate closed — chassis collector is independent of schema but ships in the same wave"
provides:
  - "_DMI_PRODUCT_NAME_PATH constant — testable path-indirection seam for sysfs reader"
  - "_DMI_PLACEHOLDERS: Final[frozenset[str]] — D-21 verbatim placeholder set, lower-cased"
  - "_normalize_dmi(s) — case-insensitive, post-strip placeholder collapse"
  - "collect_chassis_model(dmi_path=...) — sysfs reader with universal D-2 failure and T-3-05 8KB read cap"
  - "result['chassis_model'] key wired into collect_local_system_info()"
  - "Inline duplicates of the four symbols above in MPI_COLLECTOR_SCRIPT (Pattern B)"
  - "Parallel chassis_model wiring inside MPI_COLLECTOR_SCRIPT's collect_local_info()"
affects:
  - 03-03-networking-collector
  - 03-04-transform-fingerprint-extension
  - 03-05-integration-host-info-flow

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern A — per-field sysfs read with universal D-2 collection-failure rule (try/except wraps open + read + normalize, any exception → empty string for that single field)"
    - "Pattern B — MPI script vs. module duplication discipline for SSH-fan-out self-containment; parity test (TestMPIScriptParity) locks behavioral equivalence on representative inputs"
    - "Pattern — tmp_path + path-indirection function parameter (default to production sysfs constant) for sysfs unit tests, avoiding builtins.open mock_open which corrupts PyYAML"

key-files:
  created: []
  modified:
    - "mlpstorage_py/cluster_collector.py"
    - "tests/unit/test_cluster_collector.py"

key-decisions:
  - "D-21 placeholder set landed verbatim as a Final[frozenset[str]] module constant with all entries lower-cased; the comparison happens after .strip().lower() on input. Empty string included as a member so an empty file collapses through the same set-membership branch."
  - "T-3-05 DoS mitigation: explicit f.read(8192) cap in collect_chassis_model. Sysfs is kernel-buffered to PAGE_SIZE (~4KB on x86) per kernel ABI docs; 8KB is defense-in-depth against any future exotic kernel."
  - "Pattern B duplication into MPI_COLLECTOR_SCRIPT uses untyped form (no Final[], no frozenset[str] subscript). Rationale: SSH fan-out hits heterogeneous Python environments; Python 3.8 fleets cannot import the subscripted-generic form. Matches the script's existing untyped style (parse_proc_meminfo has no return-type hint at line 856)."
  - "Wrapper try/except in collect_local_system_info is defense-in-depth — collect_chassis_model already swallows all exceptions per D-2. The wrapper is structurally consistent with the surrounding per-field blocks (meminfo, os_release) and the universal-rule contract is 'the key is always present, the value is always a string'."

patterns-established:
  - "Pattern: when adding a new per-host data field to the collector, ship in lockstep on both sides (collect_local_system_info AND MPI_COLLECTOR_SCRIPT's inline collect_local_info), and add a parity unit test that exec()'s the script in a controlled namespace and compares pure-function behavior."

requirements-completed:
  - COLL-03

# Metrics
duration: ~7min
completed: 2026-06-23
---

# Phase 3 Plan 02: Chassis Model Collector Summary

**`collect_chassis_model` reads `/sys/class/dmi/id/product_name`, applies D-21 placeholder normalization (case-insensitive frozenset lookup post-strip), and wires `chassis_model` into both the module-level `collect_local_system_info()` and the inline `collect_local_info()` inside `MPI_COLLECTOR_SCRIPT` (Pattern B) — the chassis side of COLL-03 lands as a self-contained, MPI-parity-locked sysfs slice that Plan 03-05 will consume via `HostInfo.from_collected_data`.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-06-23T02:08:58Z
- **Completed:** 2026-06-23T02:15:32Z
- **Tasks:** 2 (both autonomous, both TDD)
- **Files modified:** 2 (one production, one test)

## Accomplishments

- **D-21 placeholder normalization shipped verbatim:** ten entries, all lower-cased in the frozenset, comparison via `s.strip().lower() in _DMI_PLACEHOLDERS`. 30 parametrized cases (10 strings × 3 case variants — original / upper / mixed) all GREEN.
- **D-2 universal-failure rule preserved:** `collect_chassis_model` swallows any exception via a broad `except Exception: return ""`; the wrapper try/except in `collect_local_system_info` is defense-in-depth so the contract "the `chassis_model` key is always present and is always a `str`" holds even in pathological futures.
- **T-3-05 mitigation in place:** `f.read(8192)` cap on the DMI file. Sysfs is kernel-buffered to PAGE_SIZE (~4KB on x86); 8KB is defense-in-depth against any future exotic kernel exposing an unbounded blob.
- **Pattern B (MPI script ↔ module duplication) honored:** `_DMI_PRODUCT_NAME_PATH`, `_DMI_PLACEHOLDERS`, `_normalize_dmi`, `collect_chassis_model` all duplicated inline in `MPI_COLLECTOR_SCRIPT`. The MPI worker script remains self-contained for SSH fan-out robustness.
- **Pattern B drift is now test-locked:** `TestMPIScriptParity::test_chassis_functions_match_module` exec's the script in a controlled namespace (broad `BaseException` swallow for the top-level MPI-only `SystemExit`/`ImportError`) and asserts `ns['_normalize_dmi']` matches the module on 4 representative inputs including `"  default STRING  "` (the strip + lower + placeholder collapse boundary).
- **143 tests green in `test_cluster_collector.py`** (was 105 before Plan 03-02; 38 added — 30 parametrized placeholders + 8 logical cases).
- **No regression in the wider unit suite:** 1626 passed, 4 skipped (with pre-existing dev-env collection-error files ignored per STATE.md Deferred Items: `test_benchmarks_base.py` / `test_parquet_reader.py` / `test_vdb_modular_fake_backend.py`). The 7 still-failing tests in `test_datagen_command_generation` and `test_rules_calculations` are the same pre-existing `TypeError: expected string or bytes-like object, got 'MagicMock'` fixture issue carried from 02-02; their import chains do not touch `cluster_collector.py`.

## Task Commits

1. **Task 1: Test-impact scan + RED tests for collect_chassis_model + _normalize_dmi + _DMI_PLACEHOLDERS** — `e4bcf23` (test)
2. **Task 2: Implement collect_chassis_model + DMI placeholder normalization + MPI script duplication** — `630e454` (feat)

## Files Created/Modified

- `mlpstorage_py/cluster_collector.py` — **+140 / -1 lines**.
  - Import line: `Final` added to the existing `from typing import ...` line.
  - New module-scope section "Chassis Model Collection (DMI / SMBIOS) — Phase 3 Plan 02 (D-21, COLL-03)" immediately above the existing "Local System Information Collection" section header, holding `_DMI_PRODUCT_NAME_PATH`, `_DMI_PLACEHOLDERS`, `_normalize_dmi`, `collect_chassis_model`.
  - Inside `collect_local_system_info`: new try/except block after the `os_release` block, before the `vmstat` block, that sets `result['chassis_model']` and (on exception) `result['errors']['chassis_model']`.
  - Inside `MPI_COLLECTOR_SCRIPT` string literal: inline duplicates of the four symbols (untyped form) between the existing `parse_os_release` def and `collect_local_info` def; parallel chassis try/except inside `collect_local_info` after the script's `os_release` block, before the `if not result['errors']:` cleanup branch.

- `tests/unit/test_cluster_collector.py` — **+177 / -0 lines**.
  - `os` added to top-level imports (the `@pytest.mark.skipif(os.geteuid() == 0, ...)` decorator runs at collection time and needs `os` in module scope).
  - `MPI_COLLECTOR_SCRIPT` added to the existing import block (the symbol already existed in the production module from Phase 2; only chassis symbols are new).
  - Module-level helper `_mixed_case(s)` and constant `_DMI_PLACEHOLDER_CASE_CASES` built once at import time for the parametrized placeholder cases (30 entries: 10 originals × 3 case variants).
  - `TestDMIPlaceholders` — 30 parametrized cases + `test_real_product_name_passes_through` + `test_strip_then_compare`.
  - `TestCollectChassisModel` — `test_reads_dmi_file`, `test_placeholder_file_returns_empty`, `test_missing_file_returns_empty`, `test_unreadable_file_returns_empty` (skip-if-root via `os.geteuid() == 0`), `test_local_system_info_includes_chassis_model_key`.
  - `TestMPIScriptParity` — `test_chassis_functions_match_module` (exec MPI_COLLECTOR_SCRIPT in fresh ns under broad `BaseException` swallow; assert ns has `_normalize_dmi`, `collect_chassis_model`, `_DMI_PLACEHOLDERS`; 4 sample-input parity assertions).

## Dev-host chassis_model literal value

`""` (empty string).

This dev shell is WSL2 (Linux 6.18.33.1-microsoft-standard-WSL2). Per RESEARCH.md line 371, the Microsoft WSL kernel does not expose `/sys/class/dmi/id/` at all — the directory itself is absent. `collect_chassis_model()` correctly catches the `FileNotFoundError` and returns `""`. End-to-end D-2 universal-failure semantics confirmed live, not just via mocked tests. On a real Dell/Supermicro/etc. host the live value will be the actual DMI product name.

## Decisions Made

- **Empty string as a `_DMI_PLACEHOLDERS` member.** PLAN spec'd it; the rationale is that an empty product_name file (or one containing only whitespace) collapses through the same set-membership branch as the literal vendor placeholders, rather than via a separate guard. Smaller code, same observable contract.
- **8KB `f.read(8192)` cap (T-3-05 mitigation).** Sysfs is kernel-buffered to PAGE_SIZE per `Documentation/filesystems/sysfs.rst`; the cap is defense-in-depth against any future kernel surface exposing an unbounded blob. No legitimate `product_name` exceeds ~80 bytes.
- **Untyped form for inline MPI script duplicates.** The script is exec'd over SSH on potentially heterogeneous fleets; `Final[frozenset[str]]` (the subscripted-generic form) is Python 3.9+. Matches the script's existing untyped style (`parse_proc_meminfo` at line 856 has no return-type hint).
- **`BaseException` swallow in the parity test's exec wrapper.** Pre-existing PLAN guidance was `(ImportError, NameError, SystemExit, AttributeError)`. The script's bare `sys.exit(1)` path raises `SystemExit` which IS a `BaseException` but NOT a base-Exception subclass; on heterogeneous dev hosts other base-exceptions are theoretically reachable. `BaseException` is the safest catch-all since we only care that the function DEFs have already executed before the top-level raise.
- **Defense-in-depth wrapper try/except in `collect_local_system_info`.** `collect_chassis_model` already swallows all exceptions per D-2, so the wrapper is structurally redundant for the current implementation. Kept anyway because (a) it matches the surrounding per-field blocks line-for-line, (b) it provides a clear `result['errors']['chassis_model']` slot if a future bug somewhere lets an exception escape, (c) the per-field universal-rule contract is "key always present, value always str" — being structurally identical to siblings is cheap insurance.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Top-level `os` import missing from test file**
- **Found during:** Task 1 RED collection — `NameError: name 'os' is not defined` from the `@pytest.mark.skipif(os.geteuid() == 0, ...)` decorator on `test_unreadable_file_returns_empty`.
- **Issue:** The decorator's expression is evaluated at class-body / collection time, NOT at test runtime; so the `import os` inside the test method body (which the PLAN's Step 2 implied) does not satisfy the decorator. The decorator needs `os` in the module's top-level namespace.
- **Fix:** Added `import os` to the top of `tests/unit/test_cluster_collector.py` alongside the existing `import json`, `import subprocess`, `import time` block. Single-line additive change.
- **Files modified:** `tests/unit/test_cluster_collector.py`
- **Verification:** Re-run of the three test classes flipped from `NameError` at collection to the expected 38 RED failures on missing `_normalize_dmi` / `collect_chassis_model` symbols.
- **Committed in:** `e4bcf23` (Task 1 commit — bundled with the test additions because the import is structurally part of bringing the new test classes online).

### No Process Deviations

No `git stash` was used during this plan. The 03-01 SUMMARY's process note (prefer `git diff <ref> -- <path>` and `git show <ref>:<path>` over `git stash`) was honored throughout. The only "compare against baseline" check needed was `git show HEAD:tests/unit/test_datagen_command_generation.py | head` to confirm the pre-existing failures pre-date Phase 3 — read-only, no working-tree mutation.

---

**Total deviations:** 1 auto-fixed (Rule 3 - Blocking, single-line import addition) + 0 process deviations.
**Impact on plan:** The auto-fix was necessary to make the RED tests collect at all; the fix is structural fixture-setup, not test-of-new-behavior. Zero scope creep. PLAN's two-commit success criterion ("Plan 03-02 ships in 2 commits") honored exactly.

## Issues Encountered

- **Pre-existing dev-env gaps (carried from 02-02, documented in STATE.md Deferred Items):** `tests/unit/test_benchmarks_base.py`, `tests/unit/test_parquet_reader.py`, `tests/unit/test_vdb_modular_fake_backend.py` all fail at COLLECTION time with `ModuleNotFoundError` (psutil / pyarrow.parquet / numpy not installed in this dev shell). Out-of-scope per Rule 3 scope boundary. Resolution: `pip install -e ".[test,full]"` once. The full-suite verification command therefore uses `--ignore=` for these three files; the resulting 1626-passed result is the apples-to-apples comparison to 02-06's "1588 passed" baseline (the additional 38 are the new chassis tests, plus other tests added in 03-01).
- **Pre-existing fixture issues in `test_datagen_command_generation` and `test_rules_calculations`:** 7 failing tests with `TypeError: expected string or bytes-like object, got 'MagicMock'` in `mlpstorage_py/rules/utils.py::_check_safe_path_component`. These are the SAME 7 failures documented in 03-01's SUMMARY (same root cause: a MagicMock substitution somewhere in the test setup feeds a non-string into the safe-path regex). Their import chains do not touch `cluster_collector.py` — confirmed via `grep -l cluster_collector tests/unit/test_datagen_command_generation.py tests/unit/test_rules_calculations.py` returning nothing. Out-of-scope per Rule 3 scope boundary.

## Surprises About the MPI Script's Python-Version Portability

PLAN.md's Step 3 noted: "`Final[frozenset[str]]` may not work on older Python interpreters that the MPI script encounters on heterogeneous hosts (Python 3.8 has Final but subscripted generic frozenset is 3.9+)." This was correct guidance and I followed it. The actual surprise: in modern Python 3.9+ the bare `frozenset({...})` literal still works without any typing import at all — the script's inline form is `_DMI_PLACEHOLDERS = frozenset({...})` with NO `from typing import Final` added to the script, NO type annotation on the variable. The script's existing parse_proc_meminfo / parse_proc_cpuinfo / parse_proc_diskstats are also untyped and have always been; chassis additions follow the same convention. Result: zero new imports inside the script string, zero version-compatibility risk.

The module-side, by contrast, DID need `Final` added to the existing `from typing import ...` line; that's a single token in an existing import line and is gated by `mlpstorage_py`'s Python 3.9+ minimum (per `pyproject.toml`).

## Threat Flags

None. Phase 3 Plan 02 reads one kernel-managed sysfs file with an explicit 8KB read cap. No new network exposure, no shell execution, no third-party packages. The threat register in PLAN.md (T-3-04 tampering, T-3-05 DoS, T-3-06 info disclosure, T-3-SC supply chain) covered all surfaces with mitigation / accept dispositions and concluded no blocking threats. The 8KB read cap (T-3-05) is implemented exactly as PLAN spec'd it.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **COLL-03 collector-side complete.** Per-host `chassis_model` (str) flows up through `comm.gather` to rank 0 in the rank-0 collected dict, and through `collect_local_system_info()` on the D-8 fallback path. Plan 03-05 will consume it via two-line `data.get('chassis_model', '')` additions in `HostInfo.from_collected_data` and a corresponding flat field on the `HostInfo` dataclass.
- **Plan 03-03 (networking collector) remains independent** and can ship in either order relative to 03-02 — the two Wave-2 slices touch disjoint sysfs surfaces (DMI vs `/sys/class/net/` + `/sys/class/infiniband/`). 03-03's MPI script duplication will follow the exact pattern established here (inline untyped defs between `parse_os_release` and `collect_local_info`).
- **No deferred-items.md entries created** by this plan.

## Self-Check: PASSED

- `mlpstorage_py/cluster_collector.py`: FOUND (modified — +140 / -1 lines).
- `tests/unit/test_cluster_collector.py`: FOUND (modified — +177 / -0 lines).
- Commit `e4bcf23`: FOUND — `test(03-02): add failing tests for collect_chassis_model + _DMI_PLACEHOLDERS + MPI parity (D-21, COLL-03)`.
- Commit `630e454`: FOUND — `feat(03-02): collect_chassis_model + DMI placeholder normalization + MPI script duplication (COLL-03, D-21)`.
- AI-attribution check: `git log -1 --format='%B' | grep -ci "co-authored\|claude\|anthropic"` returns `0` for both commits.
- Module-side surface contract: `python -c "from mlpstorage_py.cluster_collector import collect_chassis_model, _normalize_dmi, _DMI_PLACEHOLDERS; assert _normalize_dmi('Default string') == ''; assert _normalize_dmi('PowerEdge R760') == 'PowerEdge R760'; assert 'default string' in _DMI_PLACEHOLDERS"` prints `chassis surface ok`.
- MPI parity contract: exec'd MPI_COLLECTOR_SCRIPT in a fresh namespace; `ns['_normalize_dmi']` matches module on 4 sample inputs including `"  default STRING  "`.
- Test counts: `grep -c "TestDMIPlaceholders\|TestCollectChassisModel\|TestMPIScriptParity" tests/unit/test_cluster_collector.py` returns 3 (the three class declarations).
- Production grep counts: `grep -n "def collect_chassis_model" mlpstorage_py/cluster_collector.py` returns 2 (module + MPI script); `grep -c "_DMI_PLACEHOLDERS" mlpstorage_py/cluster_collector.py` returns 5 (≥ 4 required); `grep -c "result\['chassis_model'\]" mlpstorage_py/cluster_collector.py` returns 4 (≥ 2 required — module write + module errors-fallback + MPI write + MPI errors-fallback).

---
*Phase: 03-chassis-model-networking-coverage*
*Completed: 2026-06-23*
