---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: phase-complete
stopped_at: Completed 01-05-PLAN.md (per-mode code-image capture + canonical-layout E2E — LAY-06/07/08); Phase 1 SHIPS
last_updated: "2026-06-19T23:48:27Z"
last_activity: 2026-06-19 -- Phase 01 plan 05 executed; Phase 1 complete (LAY-01..LAY-08)
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 5
  completed_plans: 5
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-18)

**Core value:** A storage submitter can hand a benchmark result directory to the MLCommons submission checker and have it pass — without hand-tuning the submission package against a moving target.
**Current focus:** Phase 02 — first-run write of partial systemname.yaml (next)

## Current Position

Phase: 01 (canonical-layout-and-init) — COMPLETE
Plan: 5 of 5 complete; Phase 1 ships
Status: Phase 01 complete — ready for Phase 02 kickoff
Last activity: 2026-06-19 -- Plan 01-05 complete (LAY-06 per-mode code-image; LAY-07 whatif layout uniformity; LAY-08 DirectoryCheck regression)

Progress (Phase 1): [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 5
- Average duration: ~32 min
- Total execution time: ~160 min

**By Phase:**

| Phase | Plans | Total       | Avg/Plan |
| ----- | ----- | ----------- | -------- |
| 01    | 5     | ~160 min    | ~32 min  |

**Recent Trend:**

- Last 5 plans: 01-01 (~34min), 01-02, 01-03 (61min), 01-04 (~34min), 01-05 (~10min)
- Trend: 01-05 was integration-test heavy with minimal source changes — short execution

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Init: Vertical MVP project mode — each phase ships an end-to-end usable improvement, not a horizontal module.
- Init: Skip codebase mapping and project-level research; enable per-phase research for `/sys`/`lsblk`/container edge cases.
- Init: Leave non-derivable schema fields blank so the schema validator surfaces "submitter work to do" naturally.
- Discuss-phase 1: Insert canonical-layout-and-init as new Phase 1 before yaml-write work; original Phase 1 becomes Phase 2.
- Discuss-phase 1: Pin orgname to the results-dir via `mlperf-results.yaml` sentinel written by `mlpstorage init`; no `--orgname` flag, no `MLPERF_ORGNAME` env var consulted by non-init commands.
- Discuss-phase 1: `--systemname` stays per-run via CLI flag + `MLPERF_SYSTEMNAME` env-var default (E5 pattern, matches existing `MLPERF_RESULTS_DIR`).
- Discuss-phase 1: Cluster collection and systemname.yaml lifecycle fire on `run` only — never `datagen` (datagen client fleet may differ from run fleet).
- Discuss-phase 1: Universal collection rule — any failure (missing file, missing field, missing tool, parse error) yields empty string for that single data point; collector never fails the benchmark over a collection failure.
- Discuss-phase 1: Fingerprint for quantity-grouping is extensible (D2) — grows automatically as collector fills more fields.
- Discuss-phase 1: Single-host fallback (C1) — when `--hosts` is empty, collect from the local host only via existing `collect_local_system_info()`.
- Discuss-phase 1: Per-mode systemname.yaml — closed/open/whatif each own their own file at their own path, generated and diffed independently.
- Discuss-phase 1: PDF generation out of scope; remains submitter responsibility.
- Execute 01-02: orgname idempotency comparison is case-sensitive (RESEARCH.md Pitfall 7) — `Acme` ≠ `acme` is a `DoubleInitError`, not a match.
- Execute 01-02: `test_init_help_renders` weakened to description+usage assertions because `MLPStorageHelpFormatter` (`common_args.py:49-51`) project-wide suppresses positional docs; changing the formatter is out of scope for Slice 2.
- Execute 01-02: anti-pattern guard tests in this milestone use AST scans, not textual grep, so educational docstring mentions of forbidden symbols are tolerated (matches Slice 1's resolution of the `yaml.load` grep gate).
- Execute 01-03: argparse `required=True` is enforced unconditionally on emitting commands; `MLPERF_SYSTEMNAME` populates the default but does not satisfy required-flag semantics — users must still pass `--systemname` on the CLI even when the env var is set.
- Execute 01-03: checkpointing intentionally omits the `<command>` segment in the canonical path (preserves the pre-refactor shape downstream submission-checkers already accept).
- Execute 01-03: `generate_output_location()` stays pure (zero calls to `resolve_orgname`/`read_sentinel`); the upstream `main._main_impl()` gate (Slice 4 / Plan 01-04) is the sole orgname resolver, with defense-in-depth empty-string raises as a backstop.
- Execute 01-03: test-argv migration is owned by this plan (Rule 3 auto-fix); every existing unit test exercising an emitting subcommand now includes `-sn sys-v1`. Argv-only; no semantic change.
- Execute 01-04: bypass list locked to exactly four modes — `{init, version, lockfile, rules-coverage}`. `history`, `reports`, `validate` are gated per D-12; their dispatch branches were physically moved AFTER the gate in `_main_impl` (not merely guarded inline).
- Execute 01-04: LAY-03 error message uses literal backticks (not single quotes) via plain f-string interpolation `{var}` (not `{var!r}`) — locked verbatim per CONTEXT.md / ROADMAP success criterion #2.
- Execute 01-04: `validate` mode currently bypasses the sentinel check at runtime because `add_validate_arguments` registers a positional `input` rather than `--results-dir`. The gate's `if results_dir_value:` guard quietly skips it. Documented as an inert-gate divergence; if a future plan adds `--results-dir` to validate the gate will catch it automatically.
- Execute 01-04: kvcache `_make_run_benchmark` fixture uses `mode='open'` (not `'closed'`) so the CLOSED-mode override checks (seed/trials/inter-option-delay) don't fire on tests that deliberately override those args. TestClosedEnforcement sets `bm.args.mode='closed'` explicitly to exercise the enforcement path.
- Execute 01-05: `shutil.copytree(symlinks=False)` is locked as the V12 ASVS mitigation for T-1-CI2 (symlink traversal in code-image source). The grep gate (`grep -c 'symlinks=False'`) returns 2 (call site + docstring); the `test_copytree_call_uses_symlinks_false` test mocks `copytree` and asserts `kwargs["symlinks"] is False`.
- Execute 01-05: Exclude set kept inline in `code_image.py` (`_EXCLUDE_DIRS`, `_EXCLUDE_FILENAMES`) rather than imported from `submission_checker.constants` — avoids a circular-import surface and lets the two consumers (the on-disk image and the reader-side MD5 checksum) evolve independently.
- Execute 01-05: `capture_code_image` exposes `src_override` as a test-only kw-only parameter so the hermetic exclude test can construct a fake source tree under tmp_path. Production callers (Benchmark.__init__) ALWAYS pass None, which resolves to `Path(mlpstorage_py.__file__).parent`.
- Execute 01-05: Integration test exercises helpers directly (run_init → capture_code_image → DirectoryCheck) rather than full DLIO/MPI runs — dev-shell-compatible AND tests exactly what Phase 1 ships (the filesystem-layout surface). Full-DLIO E2E remains the operator-side manual smoke check.

### Pending Todos

None — Phase 1 ships.

### Blockers/Concerns

None.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-19 (resumed)
Stopped at: Completed 01-05-PLAN.md (per-mode code-image + canonical-layout E2E — LAY-06/07/08); Phase 1 SHIPS
Resume file: .planning/phases/02-first-run-systemname-yaml/02-CONTEXT.md (or `/gsd-transition` → `/gsd-discuss-phase 2`)
