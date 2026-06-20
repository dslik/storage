---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Plan 02-03 complete (auto_generator schema-aware blanks scaffolding shipped)
last_updated: "2026-06-19T00:00:00.000Z"
last_activity: 2026-06-19 -- Plan 02-03 complete (stub literals + _splice_stub_lists + _build_outer_dict)
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 10
  completed_plans: 8
  percent: 60
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-18)

**Core value:** A storage submitter can hand a benchmark result directory to the MLCommons submission checker and have it pass — without hand-tuning the submission package against a moving target.
**Current focus:** Phase 02 — first-run-write-of-partial-systemname-yaml

## Current Position

Phase: 02 (first-run-write-of-partial-systemname-yaml) — EXECUTING
Plan: 4 of 5
Status: Executing Phase 02 — Wave 3 complete (Plan 02-03 shipped); Wave 4 (02-04) is next
Last activity: 2026-06-19 -- Plan 02-03 complete (stub literals + _splice_stub_lists + _build_outer_dict)

Progress (Phase 1): [██████████] 100%
Progress (Phase 2): [██████░░░░] 60% (3/5 plans)

## Performance Metrics

**Velocity:**

- Total plans completed: 13
- Average duration: ~28 min
- Total execution time: ~211 min

**By Phase:**

| Phase | Plans | Total       | Avg/Plan |
| ----- | ----- | ----------- | -------- |
| 01 | 5 | - | - |
| 02 | 3 | ~51 min | ~17 min |

**Recent Trend:**

- Last 5 plans: 01-04 (~34min), 01-05 (~10min), 02-01 (~8min), 02-02 (~25min), 02-03 (~18min)
- Trend: 02-03 is a tightly-scoped append to 02-02's module — two `Final[dict]` constants + two pure functions + 14 new tests. RED was a clean ImportError; GREEN landed all 39 file tests on the first run. Comfortably below 02-02's size because the schema-aware blanks logic is much simpler than per-host HostInfo adaptation.

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
- Execute 02-01: D-16 lands as a 3-line additive diff to `mlpstorage_py/rules/models.py` (field + from_dict kwarg + from_collected_data kwarg). `summarize_cpuinfo` already produces `num_sockets` at `cluster_collector.py:826` (len(physical_ids) else 1); this plan simply stops `HostInfo.from_collected_data` from dropping it. Assumption A4 confirmed: every existing HostCPUInfo caller uses keyword construction, so the additive default is a pure no-op for all current sites. Test placement = `tests/unit/test_cluster_collector.py::TestHostCPUInfoNumSockets` (canonical home — TimeSeriesSample/TimeSeriesData tests already live here, so rules/models dataclass tests stay co-located).
- Execute 02-02: `auto_generator` shipped as a pure-transformation module — zero I/O, zero Pydantic construction (Pitfall 2 locked), four symbols (`_FINGERPRINT_KEYS`, `_get_dotted`, `group_by_fingerprint`, `node_dict_from_host`). Universal collection-failure rule applied uniformly: `cpu=None`, `os_release={}`, `memory.total=0`, `system=None` all map to `""` for the affected field without raising. PLAN.md two-task structure compressed to two commits per `<success_criteria>` mandate ("Slice 2 ships in two commits") — RED gate verifiably observed before any production code existed.
- Execute 02-02: Surprise — the 271_652_882_432-byte parametrized memory test case is actually ~252.9965 GiB, not 252.5 GiB as the PLAN author cited; result is still 253 but comes from straight float arithmetic, not from banker's rounding at a true half-GiB boundary. Test docstring corrected.
- Execute 02-02: Quoting-convention divergence — PLAN's `'NAME'/'VERSION_ID'` grep gate expects single-quoted Python strings; the project's convention (and surrounding modules) is double-quoted. Semantic intent (Pitfall 4 explicit key selection) is honored; the double-quoted equivalent grep returns the required count of 2.
- Execute 02-03: D-3 stub literals (`_NETWORKING_STUB`, `_DRIVE_STUB`) land as `Final[dict]` module constants with TEST-time field-name parity reflection against `NetworkPort.model_fields` and `DriveInstance.model_fields` (minus optional `performance` per D-2 row 4). Any future schema field addition fires `test_stub_keys_match_pydantic_fields` — single-source-of-truth (D-1) locked at the constant level.
- Execute 02-03: D-14 outer-dict scaffold locked: `_build_outer_dict` returns `{system_under_test: {clients: stanzas}}` with `solution`, `deployment`, `product_nodes`, `product_switches`, `total_rack_units`, `rack_power_supplies` ALL absent. Pitfall 1 honored — zero top-level Pydantic construction attempted; `schema_validator.validate_file()` naturally surfaces the missing required blocks as the intended "submitter has work to do" SER-02 UX.
- Execute 02-03: `_splice_stub_lists` is idempotent (replace-not-append) and defensive on missing `system_under_test`/`clients` keys. Returns the input dict (mutates in place) — this is part of the contract so 02-04 callers can write `dump = _splice_stub_lists(_build_outer_dict(...))` even after re-grouping. Shallow `dict(_STUB)` copy is sufficient in Phase 02 (all stub values are immutable scalars except an empty `traffic` list, which no production code mutates in place); switch to `copy.deepcopy` if Phase 4 ever mutates the spliced lists in place.
- Execute 02-03: PLAN grep-gate-vs-docstring divergence (same flavor as 02-02's quoting note). The D-14 forbidden-token grep uses `grep -v '^#'` which only strips column-0 hashes; multi-line docstrings and indented `# comment` lines pass through. AST-aware verification confirms 0 true code references — semantic D-14 intent fully honored. The docstring enumeration of omitted blocks is intentional for human readability and is NOT stripped to satisfy a flawed grep.
- Execute 02-03: PLAN.md two-task structure compressed to two commits per `<success_criteria>` mandate (same decision as 02-02). RED gate verifiably observed (`ImportError: cannot import name '_DRIVE_STUB'`) before production code.

### Pending Todos

- Phase 2 Wave 4: Plan 02-04 (write_systemname_yaml atomic orchestrator + D-7 sort) — now unblocked by 02-03. Owns the YAML I/O, atomic write, FileExistsError no-op, and the only call sites of `yaml.safe_dump` in the auto-generator subsystem.

### Blockers/Concerns

None.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Test env | `psutil` not installed in dev shell; `pytest tests/unit/test_benchmarks_*.py` and any test transitively importing `mlpstorage_py.utils` fails at collection time with `ModuleNotFoundError: No module named 'psutil'`. Pre-existing, not introduced by 02-02 or 02-03. Resolution: `pip install -e ".[test]"` once. | Deferred | 02-02 |

## Session Continuity

Last session: 2026-06-19T00:00:00.000Z
Stopped at: Plan 02-03 complete — Wave 4 (02-04) is next
Resume file: .planning/phases/02-first-run-write-of-partial-systemname-yaml/02-04-PLAN.md
