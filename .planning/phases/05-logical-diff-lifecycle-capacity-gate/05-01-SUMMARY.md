---
phase: 05-logical-diff-lifecycle-capacity-gate
plan: 01
subsystem: system_description
tags:
  - phase-5
  - mvp
  - tdd
  - life-02
  - life-03
  - diff-core
  - jsonpath
  - unified-diff

# Dependency graph
requires:
  - phase: 04-sysctl-environment-drives-coverage
    provides: "_FINGERPRINT_KEYS 11-tuple + _resolve_fingerprint_key dispatch (chassis.cpu_model, chassis.cpu_qty, chassis.cpu_cores, chassis.memory_capacity, chassis.model_name, operating_system.name/version, networking_sig, sysctl_sig, environment_sig, drives_sig) — Phase 5 reuses these as the D-38 round-trip-recompute identity rule"
  - phase: 02-first-run-write
    provides: "Phase-2 7-key node-dict emit shape (friendly_description, chassis, networking, sysctl, environment, drives, operating_system, quantity) — the diff layer consumes this shape directly"
provides:
  - "DiffEntry frozen dataclass (path, old, new)"
  - "DiffResult dataclass with .empty property and entries: list[DiffEntry]"
  - "_flatten_to_paths generator over nested dict/list → (jsonpath, leaf) pairs"
  - "_compute_fingerprint helper reusing Phase-4 _FINGERPRINT_KEYS for D-38 identity"
  - "_render_fingerprint verbatim-value renderer (NOT repr) so D-41 long sysctl tuples round-trip"
  - "diff_node_dict_lists(on_disk, in_memory) → DiffResult with SER-02 Pitfall 3(a) blank preservation"
  - "format_unified_diff(result, on_disk_path) → str with --- / +++ / @@ <JSONPath> @@ / -/+ / Remediation block"
affects:
  - "Plan 05-02 (Slice 2 SystemDriftError + parse_on_disk_systemname_yaml + replace FileExistsError no-op)"
  - "Plan 05-05 (Slice 5 end-to-end integration tests TestPhase5Lifecycle)"

# Tech tracking
tech-stack:
  added: []  # Slice 1 ships ZERO new packages — only stdlib dataclasses/typing + existing auto_generator symbols
  patterns:
    - "Pure-function transformation module (no I/O, no exception raises — consistent with auto_generator.py)"
    - "_SENTINEL_ABSENT object() to disambiguate 'absent' from 'empty string' at the leaf level"
    - "_render_fingerprint verbatim-str renderer over recursive tuple walk (D-41 lock)"
    - "key=repr defense applied at union-of-paths sort (D-22 mixed-type fingerprint defense)"

key-files:
  created:
    - "mlpstorage_py/system_description/diff.py (337 lines)"
    - "tests/unit/test_diff.py (434 lines, 29 tests across 8 classes)"
  modified: []

key-decisions:
  - "_render_fingerprint helper added during GREEN (Rule 1 fix): PLAN's verbatim f'clients[fingerprint={fp!r}]' uses repr() which escapes literal tabs in multi-value sysctl leaves like '4096\\t87380\\t16777216' as '\\\\t', breaking the D-41 round-trip-verbatim contract. _render_fingerprint walks the tuple structure recursively and emits each leaf via plain str() so the original bytes survive end-to-end. This is the load-bearing fix the PLAN's <action> block explicitly anticipated: 'If test_long_sysctl_value_round_trips_verbatim_no_truncation surfaces a real truncation bug, fix the value rendering in format_unified_diff (most likely: a misplaced repr() call).'"
  - "Pitfall 3(a) SER-02 blank preservation direction-lock: the skip ONLY fires when mem_v == '' AND disk_v is present AND disk_v != ''. Direction (b) — disk empty + memory filled — is the normal first-run write path and MUST surface as drift on re-run if it ever happens (e.g. submitter manually emptied a field then re-ran). Locked by test_in_memory_NONEMPTY_disk_filled_with_different_value_IS_diff which would have caught a naive 'always skip when disk filled' implementation."
  - "Fingerprint orphan paths render as `clients[fingerprint=<rendered-tuple>]` rather than positional indices because the two sides may have different cardinalities; positional indices would silently misalign and produce confusing report shapes."
  - "_FINGERPRINT_KEYS reused via import (NOT copied) so the diff layer and the quantity-grouper share a single source of truth. Any future fingerprint extension automatically propagates to drift detection without touching diff.py."
  - "DiffEntry is @dataclass(frozen=True); DiffResult is mutable to let Slice 2 extend entries if needed. T-5-01-02 threat register accepts this trade-off since results are per-call and not stored in shared state."
  - "key=repr applied at the union-of-paths sort inside the fingerprint-matched branch as well as at the fingerprint union sort, defending against TypeError on mixed-type fingerprint tuples (the same Plan 03-04 / 04-04 surprise re-applied)."

patterns-established:
  - "Diff layer = pure transformation: zero I/O, zero exception raises. Slice 2 owns SystemDriftError at the call site, not in the diff module."
  - "Verbatim value rendering in unified-diff output: str(), never repr(), so long opaque strings (sysctl tuples, redacted env values, multi-line drive fields) survive byte-for-byte through the report."
  - "Sentinel object disambiguation pattern: when an algorithm needs to distinguish 'absent' from 'present but empty', use a module-level object() rather than None or '' (both of which are legal data values in this domain)."

requirements-completed:
  - LIFE-02
  - LIFE-03

# Metrics
duration: 5min
completed: 2026-06-24
---

# Phase 5 Plan 01: Pure-Function Diff Core Summary

**Pure-function diff core (DiffEntry, DiffResult, diff_node_dict_lists, format_unified_diff) reusing Phase-4 _FINGERPRINT_KEYS for D-38 identity, with SER-02 Pitfall 3(a) blank preservation and D-41 verbatim-value rendering via a custom _render_fingerprint helper.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-06-24T03:30:52Z
- **Completed:** 2026-06-24T03:35:51Z
- **Tasks:** 2 (Task 1 RED test file + Task 2 GREEN implementation)
- **Files modified:** 0 (two new files created)
- **Files created:** 2

## Accomplishments

- Pure-function diff layer landed end-to-end: byte-equal inputs produce DiffResult.empty == True; any leaf-level or fingerprint-level drift surfaces as structured DiffEntry rows.
- Unified-diff output shape locked per D-40 (`--- on-disk:` / `+++ in-memory:` headers, `@@ <JSONPath> @@` hunks, `-`/`+` lines, trailing Remediation block with both rename and rm hints).
- D-41 verbatim round-trip contract enforced via _render_fingerprint helper: long sysctl tuples like `4096\t87380\t16777216` appear unmodified in the rendered report (test_long_sysctl_value_round_trips_verbatim_no_truncation locks the contract).
- SER-02 Pitfall 3(a) blank-preservation direction-lock shipped: submitter-filled values survive when collector returns blank; a regression test (test_in_memory_NONEMPTY_disk_filled_with_different_value_IS_diff) locks the direction so a naive 'always skip when disk filled' refactor would fail immediately.
- D-38 round-trip-recompute identity contract reused via `from mlpstorage_py.system_description.auto_generator import _FINGERPRINT_KEYS, _resolve_fingerprint_key` — single source of truth across the quantity-grouper and the diff layer.

## Task Commits

Each task was committed atomically:

1. **Task 1: RED — Failing tests for diff core** — `810bc15` (test)
2. **Task 2: GREEN — Implement diff.py to make all RED tests pass** — `fd011ff` (feat)

Two-commit RED/GREEN cadence verified via `git log --oneline -2 | grep -E 'RED|GREEN' | wc -l` → 2.

**Plan metadata commit:** SKIPPED — `.planning/config.json` sets `commit_docs: false` so the SUMMARY.md and STATE.md updates land via the host integration step, not as a separate metadata commit. (Same convention as Phase 4 plans.)

## Files Created/Modified

- `mlpstorage_py/system_description/diff.py` (337 lines) — Pure-function diff core: DiffEntry frozen dataclass, DiffResult dataclass, _SENTINEL_ABSENT sentinel, _flatten_to_paths generator, _compute_fingerprint helper, _render_fingerprint helper, diff_node_dict_lists public function, format_unified_diff public function, explicit `__all__` export list.
- `tests/unit/test_diff.py` (434 lines, 29 tests) — RED→GREEN coverage of all 8 must-have truths across 8 test classes: TestDataclasses, TestFlattenToPaths, TestRoundTripEqualIsEmpty, TestFieldChangeIsDrift, TestQuantityChangeIsDrift, TestSymmetricDriftDetection, TestSer02BlankPreservation (including the load-bearing direction-lock regression), TestUnifiedDiffFormat (including the D-41 verbatim long-value test).

## Decisions Made

See `key-decisions` in frontmatter — three load-bearing decisions:

1. `_render_fingerprint` helper added during GREEN (Rule 1 fix per PLAN's explicit anticipation).
2. Pitfall 3(a) direction-lock: `mem_v == ''` AND `disk_v` present-and-non-empty.
3. Orphan paths rendered with rendered-tuple fingerprint, not positional index.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] D-41 truncation regression in fingerprint orphan path rendering**
- **Found during:** Task 2 (GREEN verification — `test_long_sysctl_value_round_trips_verbatim_no_truncation` failed on first run)
- **Issue:** PLAN's verbatim `f"clients[fingerprint={fp!r}]"` uses Python's `repr()` which escapes literal tab characters in string leaves as `\\t`. A long sysctl tuple value `4096\t87380\t16777216` (three values separated by literal tabs) appears in the rendered report as `4096\t87380\t16777216` (backslash-t escape sequences in the output bytes), failing the D-41 round-trip-verbatim contract.
- **Fix:** Added `_render_fingerprint` helper that walks the fingerprint tuple recursively and emits each leaf via plain `str()` (NOT `repr()`). Cosmetic-only change: fingerprints are still keyed and sorted by the tuple itself, only the rendering changed. The PLAN's `<action>` block explicitly anticipated this fix path: "If test_long_sysctl_value_round_trips_verbatim_no_truncation surfaces a real truncation bug, fix the value rendering in format_unified_diff (most likely: a misplaced repr() call)." The misplaced repr() was in the fingerprint orphan path construction inside `diff_node_dict_lists`, not in `format_unified_diff` directly — but the value flows through `format_unified_diff` so the contract is owned at the boundary.
- **Files modified:** `mlpstorage_py/system_description/diff.py` (single helper function added + two `f"...={_render_fingerprint(fp)}..."` substitutions inside `diff_node_dict_lists`).
- **Verification:** All 29 tests in `test_diff.py` pass, including the D-41 lock; 402 tests pass in the Phase 4+5 target slice with no regressions.
- **Committed in:** `fd011ff` (folded into the GREEN commit since the helper is structurally part of the GREEN implementation, not a separate hygiene pass — same convention as Plan 04-03's RM-bool Rule 1 fix and Plan 04-02's contract-test updates).

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug).
**Impact on plan:** Single load-bearing fix the PLAN explicitly anticipated in the `<action>` block; no scope creep. The fix is integral to making the D-41 round-trip-verbatim contract hold end-to-end on real submission hosts where multi-value sysctl leaves like `net.ipv4.tcp_rmem` are common.

## Issues Encountered

None beyond the Rule-1 fix documented above. RED→GREEN cadence was clean otherwise: 28 of 29 tests passed on first GREEN run; only the D-41 truncation test surfaced the misplaced-repr bug.

## Process Notes

- **NO `git stash` used.** Read-only inspection only via the Read tool against committed files. Project STATE.md notes the system-prompt prohibition on `git stash` due to the shared stash list across worktrees; this sequential session honored that throughout.
- **Test count delta:** PLAN.md baseline was 29 minimum; shipped 29 exactly. No surprise expansion needed.
- **Two-commit cadence preserved.** RED commit (810bc15) contains ONLY the test file; GREEN commit (fd011ff) contains ONLY the production file. `git diff 810bc15 HEAD -- tests/unit/test_diff.py` returns empty — Task 2 made zero modifications to the test file (the acceptance criterion explicitly checks this).
- **Threat surface:** No new STRIDE flags. T-5-01-01 (credential-leak in diff report) accepted per upstream Phase-4 D-23/D-24 redaction; T-5-01-02 (mutable DiffResult.entries list) mitigated by `frozen=True` DiffEntry; T-5-01-SC (supply-chain) accepted because Slice 1 installs zero new packages.

## User Setup Required

None — Slice 1 is a pure-function library module with no environment variables, no service configuration, and no external dependencies.

## Forward Notes (for Slice 2)

- **Public API is LOCKED.** Slice 2 (Plan 05-02) will `from mlpstorage_py.system_description.diff import diff_node_dict_lists, format_unified_diff` and pass `(on_disk_stanzas, in_memory_stanzas)` plus the on-disk path. The signatures are stable.
- **No SystemDriftError construction here.** Slice 2 owns the exception class definition AND the raise site (`write_systemname_yaml` will replace its FileExistsError no-op with a load-diff-raise branch). The diff module returns structured data; the raise lives at the call site by design (T-5-01-02 / pure-function discipline).
- **_FINGERPRINT_KEYS reuse is verifiable via grep.** If Plan 05-02 (or any future phase) introduces a parallel re-implementation of `_compute_fingerprint`, a single grep across `mlpstorage_py/` will flag the divergence. The current single-source-of-truth shape is the load-bearing D-38 contract.
- **D-41 verbatim-value-rendering contract is established.** Any future formatter that ships values to a human-readable report MUST use `str()` (NOT `repr()`) at the leaf rendering layer. The `_render_fingerprint` helper is private but the pattern (recursive `str()`-emitting walk over tuple structures) is the canonical example.
- **Pitfall 3(a) skip is leaf-level only.** If Slice 2 extends the diff to surface fingerprint-level drift with leaf-level resolution (e.g. "fingerprint diverged because chassis.cpu_model changed from X to Y"), the SER-02 blank skip must be re-applied at the new resolution layer. Currently the orphan path emits a coarse `<present>` / `<absent>` marker.

## Self-Check: PASSED

- `mlpstorage_py/system_description/diff.py` exists: FOUND
- `tests/unit/test_diff.py` exists: FOUND
- Commit `810bc15` (RED): FOUND
- Commit `fd011ff` (GREEN): FOUND
- `pytest tests/unit/test_diff.py -x -q` exit code 0 with 29 passed: VERIFIED
- `python3 -c "from mlpstorage_py.system_description.diff import DiffEntry, DiffResult, diff_node_dict_lists, format_unified_diff; print('imports ok')"` prints "imports ok": VERIFIED
- Two-commit RED/GREEN cadence (`git log --oneline -2 | grep -E 'RED|GREEN' | wc -l` returns 2): VERIFIED
- `git diff 810bc15 HEAD -- tests/unit/test_diff.py` returns empty (no test modifications after RED): VERIFIED
- No `Co-Authored-By: Claude` lines in either commit: VERIFIED

## Next Phase Readiness

- Slice 1 API locked and ready for Slice 2 (Plan 05-02 SystemDriftError + parse_on_disk_systemname_yaml + write_systemname_yaml load-diff-raise branch).
- Phase 4 + Phase 5 Slice 1 target slice green: 402 tests passed in `tests/unit/test_diff.py tests/unit/test_auto_generator.py tests/unit/test_cluster_collector.py` with no regressions.
- No blockers, no concerns.

---
*Phase: 05-logical-diff-lifecycle-capacity-gate*
*Completed: 2026-06-24*
