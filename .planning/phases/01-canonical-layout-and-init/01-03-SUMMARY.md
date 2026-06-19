---
phase: 01-canonical-layout-and-init
plan: 03
subsystem: cli
tags:
  - python
  - cli
  - argparse
  - paths
  - canonical-layout

requires:
  - phase: 01-canonical-layout-and-init
    provides: DEFAULT_RESULTS_DIR env-var-with-default pattern, add_universal_arguments(parser, req_results) signature, BENCHMARK_TYPES enum, ConfigurationError / ErrorCode framework
provides:
  - DEFAULT_SYSTEMNAME module-level constant honoring MLPERF_SYSTEMNAME env var (LAY-04)
  - Extended add_universal_arguments(parser, req_results, req_systemname=False) signature with --systemname/-sn flag wired through every emitting subcommand (D-10 scope)
  - Canonical-layout generate_output_location() emitting <rd>/<mode>/<orgname>/results/<sys>/<bench>/<model>/<cmd>/<dt>/ for all four BENCHMARK_TYPES (LAY-05)
  - T-1-02 mitigation: empty args.systemname raises actionable ConfigurationError before path assembly
  - Pitfall-1 defense: empty args.orgname raises ConfigurationError (the upstream Slice-4 main gate is the primary resolver)
affects:
  - 01-04 (orgname-resolution main gate; consumes the new generate_output_location signature and reads args.orgname / args.systemname)
  - 01-05 (test-fixture migration sweep; downstream callers / integration tests still use the legacy literal-path shape outside tests/unit)
  - phase-02 (cluster-collector and systemname.yaml lifecycle land in the canonical results subdir produced here)
  - phase-05 (capacity gate + code-image dispatch read from the new layout)

tech-stack:
  added: []
  patterns:
    - "env-var-with-default constants (MLPERF_SYSTEMNAME mirrors MLPERF_RESULTS_DIR)"
    - "Universal CLI arg with optional-required toggle (req_systemname=False default + per-emitting-command opt-in)"
    - "Pure path generator: read args.* purely; resolve sentinel state upstream (Pitfall 1)"
    - "Defense-in-depth empty-string guards on filesystem-path segments (T-1-02)"

key-files:
  created: []
  modified:
    - mlpstorage_py/config.py
    - mlpstorage_py/cli/common_args.py
    - mlpstorage_py/cli/training_args.py
    - mlpstorage_py/cli/checkpointing_args.py
    - mlpstorage_py/cli/vectordb_args.py
    - mlpstorage_py/cli/kvcache_args.py
    - mlpstorage_py/cli/utility_args.py
    - mlpstorage_py/rules/utils.py
    - tests/unit/test_config.py
    - tests/unit/test_cli_parser.py
    - tests/unit/test_rules_calculations.py
    - tests/unit/test_accumulation.py
    - tests/unit/test_parser_modes.py
    - tests/unit/test_cli.py
    - tests/unit/test_cli_kvcache.py
    - tests/unit/test_cli_vectordb.py
    - tests/unit/test_help_behavior.py
    - tests/unit/test_rules_parser_gating_consistency.py

key-decisions:
  - "Argparse required=True is enforced unconditionally on emitting commands; MLPERF_SYSTEMNAME populates the default but does NOT satisfy required-flag semantics. Users must always pass --systemname on emitting commands; the env var is a convenience default the user can still override on CLI."
  - "Checkpointing intentionally omits the <command> segment in the canonical path — preserves the pre-refactor shape that downstream submission-checker fixtures already accept."
  - "generate_output_location() stays pure: zero calls to resolve_orgname/read_sentinel; the upstream main._main_impl() gate (Slice 4) is the sole resolver. Defense-in-depth empty-string guards catch bypass."
  - "Test-argv migration is owned by this plan (Rule 3 auto-fix): every existing test that exercises an emitting subcommand now includes --systemname sys-v1. No semantic change to the tests; argv-only."

patterns-established:
  - "Optional-required toggle via add_universal_arguments(req_systemname=...): backward-compatible default + per-command opt-in. Future universal-flag additions should follow this same shape."
  - "Pure path generator + upstream sentinel resolver split (mirrors Rules.md §2.1.5-8 separation of orgname-from-sentinel and systemname-from-CLI)."

requirements-completed:
  - LAY-04
  - LAY-05

duration: 61min
completed: 2026-06-19
---

# Phase 01 Plan 03: --systemname plumbing + canonical generate_output_location Summary

**`--systemname/-sn` + `MLPERF_SYSTEMNAME` env var threaded through every emitting subcommand (D-10), and `generate_output_location()` rewritten to emit the Rules.md §2.1-shaped `<rd>/<mode>/<orgname>/results/<sys>/<bench>/<model>/<cmd>/<dt>/` layout for all four BENCHMARK_TYPES with empty-string raises (T-1-02 + Pitfall-1 defenses).**

## Performance

- **Duration:** 61 min
- **Started:** 2026-06-19T21:14:12Z
- **Completed:** 2026-06-19T22:16:06Z
- **Tasks:** 2 (both autonomous TDD)
- **Files modified:** 18 (8 source files, 10 test files)

## Accomplishments

- `DEFAULT_SYSTEMNAME` constant honors `MLPERF_SYSTEMNAME` env var, falling back to empty string.
- `add_universal_arguments(parser, req_results, req_systemname=False)` registers `--systemname/-sn` universally; required-mode opt-in per emitting subcommand.
- Six args-builder call sites updated (training/checkpointing/vectordb/kvcache/utility-reportgen/utility-history) with explicit per-subcommand `req_systemname=` expressions.
- `generate_output_location()` emits canonical layout for training, vector_database, kv_cache, checkpointing (checkpointing omits `<command>`).
- T-1-02 mitigation in place: empty `args.systemname` raises `ConfigurationError` with actionable suggestion.
- Pitfall-1 defense: empty `args.orgname` raises `ConfigurationError` directing the user to either run `mlpstorage init` or trace why the upstream gate did not populate `args.orgname`.
- Full unit-test sweep green: **1411 passed, 4 skipped**. Zero regressions outside the scope of the plan.

## Task Commits

Each TDD-pair was committed atomically:

1. **Task 1 RED — failing systemname tests** — `05e4c2c` (`test(01-03)`)
2. **Task 1 GREEN — --systemname plumbing on every emitting subcommand** — `aa28a1f` (`feat(01-03)`)
3. **Task 2 RED — canonical-layout assertions** — `a02325d` (`test(01-03)`)
4. **Task 2 GREEN — generate_output_location rewrite** — `c2ce3b6` (`feat(01-03)`)

_TDD gate sequence: a `test(01-03)` commit precedes each `feat(01-03)` commit; no refactor pass was needed (post-GREEN code already minimal)._

## Files Created/Modified

| File | Role |
|------|------|
| `mlpstorage_py/config.py` | Added `DEFAULT_SYSTEMNAME = os.environ.get("MLPERF_SYSTEMNAME", "")` mirroring the existing `DEFAULT_RESULTS_DIR` pattern. |
| `mlpstorage_py/cli/common_args.py` | Extended `add_universal_arguments` signature with `req_systemname=False`; registers `--systemname/-sn` with `required=req_systemname` and `default=DEFAULT_SYSTEMNAME`. Added `HELP_MESSAGES['systemname']`. |
| `mlpstorage_py/cli/training_args.py` | `req_systemname=(command in ("datagen","run","configview","datasize"))` per D-10. |
| `mlpstorage_py/cli/checkpointing_args.py` | `req_systemname=(command in ("run","configview"))` — datasize is non-emitting; datagen/validate do not exist on this builder. |
| `mlpstorage_py/cli/vectordb_args.py` | `req_systemname=(command in ("datagen","run"))` — datasize is non-emitting. |
| `mlpstorage_py/cli/kvcache_args.py` | `req_systemname=(_parser is run_benchmark)` — only `run` emits results. |
| `mlpstorage_py/cli/utility_args.py` | `req_systemname=True` on reportgen and on the history loop. |
| `mlpstorage_py/rules/utils.py` | Rewrote `generate_output_location()` to canonical layout; added `ConfigurationError` import + empty-string guards. |
| `tests/unit/test_config.py` | Added `TestDefaultSystemname::test_default_systemname_env_var`. |
| `tests/unit/test_cli_parser.py` | New `TestSystemname` class (parameterized over emitting commands; `test_empty_systemname_errors`, `test_lockfile_does_not_require_systemname`, `test_init_does_not_require_systemname`, env-var default). Updated existing tests' argv. |
| `tests/unit/test_rules_calculations.py` | Updated literal-path assertions for canonical layout; added vectordb/kvcache path tests; added `test_canonical_prefix_*` parametrized tests; added empty-systemname/empty-orgname raise tests. |
| `tests/unit/test_accumulation.py` | Updated SimpleNamespace fixtures for canonical layout (Rule 3 — Pitfall 4 surfaced here). |
| `tests/unit/test_parser_modes.py`, `tests/unit/test_cli.py`, `tests/unit/test_cli_kvcache.py`, `tests/unit/test_cli_vectordb.py`, `tests/unit/test_help_behavior.py`, `tests/unit/test_rules_parser_gating_consistency.py` | argv-only updates: added `-sn sys-v1` to every test exercising an emitting subcommand so they continue to parse under the new required-flag regime. |

## Canonical layout shapes produced

| BENCHMARK_TYPE | Path shape |
|----------------|-----------|
| `training` | `<rd>/<mode>/<orgname>/results/<sys>/training/<model>/<command>/<datetime>/` |
| `vector_database` | `<rd>/<mode>/<orgname>/results/<sys>/vector_database/<vdb_engine>/<command>/<datetime>/` |
| `kv_cache` | `<rd>/<mode>/<orgname>/results/<sys>/kv_cache/<model>/<command>/<datetime>/` |
| `checkpointing` | `<rd>/<mode>/<orgname>/results/<sys>/checkpointing/<model>/<datetime>/` (no `<command>` segment by design) |

## Subcommands now requiring `--systemname` (D-10)

| Builder | Subcommands requiring `--systemname` |
|---------|---------------------------------------|
| `training_args.py` | `datagen`, `run`, `configview`, `datasize` |
| `checkpointing_args.py` | `run`, `configview` |
| `vectordb_args.py` | `datagen`, `run` |
| `kvcache_args.py` | `run` |
| `utility_args.py` (reportgen) | `reportgen` |
| `utility_args.py` (history) | `show`, `rerun` |

Pure utility surfaces — `lockfile {generate, verify}`, `init`, `version`, `rules-coverage`, `submission-checker validate` — keep `req_systemname=False` (the default). They register `--systemname` optionally so a future caller can still pass it, but argparse does not demand it.

## Decisions Made

- **argparse `required=True` semantics:** When the plan said `required=True`, I left it strict — that is, even if `MLPERF_SYSTEMNAME` is set, the user still must pass `--systemname` on the CLI for emitting subcommands. The env var supplies a default value for the parser to record, but does not satisfy the required-flag check. This matches RESEARCH.md Assumption A5 ("Safer choice"). The CLI test originally written to assert "env var alone satisfies the flag" was reframed to assert the config-module constant only, since the parser-level test would have demanded a different — and weaker — required policy.
- **Checkpointing `<command>` omission:** Preserved per the plan (must_haves.truths). The submission checker already validates this shape; reintroducing `<command>` would have required updating submission-checker fixtures in scope-creeping ways.
- **`getattr(args, ..., '')` defensiveness:** Per RESEARCH.md Pitfall 2, the function uses `getattr(args, 'orgname', '')` and `getattr(args, 'systemname', '')` rather than direct attribute access. This guards against `_apply_yaml_config_overrides()` dropping the attribute when the YAML key is missing.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Test-argv migration in additional unit-test files**

- **Found during:** Task 1 GREEN regression sweep.
- **Issue:** Setting `req_systemname=True` on every emitting subcommand caused **196 test failures** across `test_parser_modes.py`, `test_cli.py`, `test_cli_kvcache.py`, `test_cli_vectordb.py`, `test_help_behavior.py`, and `test_rules_parser_gating_consistency.py`. The plan's `read_first` list and `files_modified` list only covered `test_config.py` and `test_cli_parser.py`, but RESEARCH.md Pitfall 4 explicitly warns about this exact failure shape (literal-arg tests in other files breaking).
- **Fix:** Updated `sys.argv` lists in every emitting-subcommand test to include `-sn sys-v1` (or `--systemname sys-v1`). Used inline `Edit` for clarity, with a single targeted `sed` pass for repetitive `--results-dir` patterns in `test_cli_vectordb.py` (33 hits). No semantic changes; argv-only.
- **Files modified:** the six test files listed above.
- **Verification:** `pytest tests/unit -v --ignore=<env-broken-files>` shows 1411 passed, 4 skipped (zero failures attributable to this plan).
- **Committed in:** `aa28a1f` (alongside Task 1 GREEN).

**2. [Rule 3 — Blocking] `test_accumulation.py` SimpleNamespace fixtures missing canonical-layout attributes**

- **Found during:** Task 2 GREEN regression sweep.
- **Issue:** Four tests in `TestPreviewBenchmarkAccumulation` (`test_vectordb_path_includes_engine`, `test_vectordb_path_requires_engine`, `test_kvcache_path_includes_model`, `test_kvcache_path_requires_model`) constructed `SimpleNamespace(results_dir=..., command=..., model_or_engine=...)` without `mode/orgname/systemname`. After the rewrite, `generate_output_location()` raises `ConfigurationError` on empty `orgname` before the path is assembled.
- **Fix:** Added `mode="closed", orgname="Acme", systemname="sys-v1"` to each `SimpleNamespace` and updated the literal-path assertions to the canonical form.
- **Files modified:** `tests/unit/test_accumulation.py`.
- **Verification:** `pytest tests/unit/test_accumulation.py -v` passes (29 passed).
- **Committed in:** `c2ce3b6` (alongside Task 2 GREEN).

---

**Total deviations:** 2 auto-fixed (both Rule 3 — blocking issues directly caused by the plan's call-site changes).
**Impact on plan:** Both auto-fixes were strictly necessary to keep CI green; neither expanded scope. The plan correctly identified the issue class (RESEARCH.md Pitfall 4) but did not enumerate the exact files affected. Slices 4 and 5 will inherit a green baseline.

## Issues Encountered

- **Three test files cannot be collected in this dev shell** because `psutil`, `pyarrow`, and `numpy` are not installed: `tests/unit/test_benchmarks_base.py`, `tests/unit/test_parquet_reader.py`, `tests/unit/test_vdb_modular_fake_backend.py`, and (indirectly via `mlpstorage_py.benchmarks` import) `tests/unit/test_datagen_command_generation.py`. Pre-existing environment gap unrelated to this plan; logged in `.planning/phases/01-canonical-layout-and-init/deferred-items.md`.
- **`tests/unit/test_version.py` has 2 pre-existing failures** (version-pin mismatch: assertion expects `3.0.13`, code reports `3.0.12`). Already failing on `main` before this plan; out of scope per execute-plan scope boundary. Also logged in `deferred-items.md`.

## Threat Flags

No new threat surface introduced beyond the registered T-1-02 / T-1-03 / T-1-S3 mitigations. The plan's threat register correctly identified all paths; no new endpoints, auth surfaces, or trust-boundary changes.

## Known Stubs

None. Both Task 1 and Task 2 ship fully wired functionality:

- `--systemname` flows from CLI/env → argparse → `args.systemname` → `generate_output_location` → on-disk path.
- `generate_output_location` is fully operational for all four BENCHMARK_TYPES; downstream `Benchmark.__init__` and `_reserve_run_directory()` already consume its return value (no Slice-3 changes required there).

The remaining piece — populating `args.orgname` from `mlperf-results.yaml` — is the explicit deliverable of Slice 4 (Plan 01-04). Until then, the empty-orgname `ConfigurationError` defense fires for any non-test caller, which is the documented expected behavior.

## User Setup Required

None — no external service configuration changes. The new env var `MLPERF_SYSTEMNAME` is optional and never required.

## Next Phase Readiness

- **Plan 01-04 (orgname-resolution main gate):** Ready. Consumes `args.systemname` (already populated) and adds `args.orgname` upstream of `Benchmark.__init__`. No surprises expected.
- **Plan 01-05 (test-fixture migration sweep):** Reduced scope. Most `tests/unit/` literal-path assertions are already updated. Slice 5 will primarily handle integration tests and the submission-checker fixture tree.
- **Phase 02 (cluster collection):** Unblocked. The `<systemname>` subdir is now part of the canonical path so `systemname.yaml` can land there in Phase 2.

## Self-Check: PASSED

Verified before sealing:

- `git log --oneline FileSystemGuy-client-system-collector -8` shows the four task commits at `05e4c2c`, `aa28a1f`, `a02325d`, `c2ce3b6` (RED, GREEN, RED, GREEN — TDD gate sequence honored).
- `pytest tests/unit --ignore=<env-broken>` → `1411 passed, 4 skipped`.
- Manual smoke: `generate_output_location(training, closed/Acme/sys-v1)` → `/r/closed/Acme/results/sys-v1/training/unet3d/run/<dt>`.
- All grep acceptance criteria pass: `DEFAULT_SYSTEMNAME` present in `config.py`; `req_systemname` referenced 3× in `common_args.py` (signature + 2 branches); 6 emitting call sites opt in; `lockfile_args.py`/`init_args.py` never reference it; `rules/utils.py` has 0 calls to `resolve_orgname/read_sentinel`.

---
*Phase: 01-canonical-layout-and-init*
*Completed: 2026-06-19*
