---
phase: 04-sysctl-environment-and-drives-coverage
plan: 01
subsystem: cluster_collector
tags: [collector, sysctl, allowlist, pattern-b, COLL-05]
requires:
  - HostInfo dataclass shape (carried from Phase 3 unchanged)
  - MPI_COLLECTOR_SCRIPT Pattern B body (carried from Phase 3)
provides:
  - mlpstorage_py.cluster_collector.collect_sysctl
  - mlpstorage_py.cluster_collector._load_sysctl_allowlist
  - mlpstorage_py.cluster_collector._PROC_SYS_ROOT
  - mlpstorage_py.cluster_collector._SYSCTL_ALLOWLIST_PATH
  - mlpstorage_py/system_description/sysctl_allowlist.txt (shipped data file)
  - MPI_COLLECTOR_SCRIPT inline twins of collect_sysctl + _load_sysctl_allowlist
  - collect_local_system_info result['sysctl'] key (always-present list contract)
affects:
  - mlpstorage_py/cluster_collector.py (module + MPI script body)
  - mlpstorage_py/system_description/ (new data file)
tech-stack:
  added:
    - fnmatch (stdlib; was unused in cluster_collector, now imported in both module and MPI script)
    - re.Pattern type alias from typing (was already in re; surfaced via typing for annotations)
  patterns:
    - D-2 universal collection-failure rule, applied at two scopes (outer envelope + per-leaf isolation)
    - D-27/28/29 allowlist file + walk + verbatim multi-value emit
    - D-36 Pattern B MPI script twin discipline
    - Path indirection (proc_sys_root parameter default) for testability without monkeypatch-of-builtins
key-files:
  created:
    - mlpstorage_py/system_description/sysctl_allowlist.txt
    - .planning/phases/04-sysctl-environment-and-drives-coverage/04-01-SUMMARY.md
  modified:
    - mlpstorage_py/cluster_collector.py
    - tests/unit/test_cluster_collector.py
    - .planning/ROADMAP.md
decisions:
  - "Pattern B twin bakes allowlist into the script body as a tuple literal (_SYSCTL_ALLOWLIST_LINES) — the script is exec'd over SSH on heterogeneous fleets and cannot read package data. Drift between the script tuple and the shipped file is a manual-discipline concern (parity test asserts behavioral equivalence on shared fixtures, not allowlist-content equivalence)."
  - "Per-leaf try/except over open() — not a pre-flight stat() — because Linux exposes write-only sysctl leaves (vm.drop_caches, kernel.sysrq, route/flush) as mode 0200; the EACCES surfaces on read, not on stat."
  - "8 KiB read cap per leaf (matches the chassis_model defense-in-depth pattern) even though /proc/sys is PAGE_SIZE-buffered to ~4 KiB — guards against any future kernel exposing an unbounded blob."
  - "Strip only the trailing newline (rstrip('\\n')), NOT .strip(), so multi-value tab-separated leaves (net.ipv4.tcp_rmem → '4096\\t87380\\t16777216') round-trip verbatim per D-29."
metrics:
  duration_minutes: ~25
  completed_date: 2026-06-23
  tasks_completed: 2
  files_created: 2
  files_modified: 3
  commits: 2
---

# Phase 04 Plan 01: Sysctl Collector + Allowlist File Summary

Data-driven sysctl collector with on-disk glob allowlist (COLL-05). Shipped 4-glob allowlist file + module collector + MPI script twin in two-commit RED/GREEN cadence; 18 new tests green, no regressions.

## What Shipped

**1. Shipped allowlist file** — `mlpstorage_py/system_description/sysctl_allowlist.txt` (8 lines: 3 comment, 1 blank, 4 globs). This is the load-bearing artifact for COLL-05's "no code change required to add a sysctl key" success criterion: editing the file in an editable install adds keys to the next run's output.

**2. Module collector** — `_load_sysctl_allowlist()` + `collect_sysctl()` in `mlpstorage_py/cluster_collector.py`:
- `_PROC_SYS_ROOT` / `_SYSCTL_ALLOWLIST_PATH` module constants
- Loader returns a tuple of `re.compile(fnmatch.translate(glob))` objects; skips blank lines and `#`-comments; returns `tuple()` on missing/unreadable file (D-2)
- Collector walks `/proc/sys` (parameter-defaulted for testability), per-leaf 8 KiB cap (D-28), trailing-newline strip only (D-29 verbatim), per-leaf try/except (D-2 / RESEARCH Q2 write-only-leaf isolation), outer try/except → `[]` on catastrophic failure (D-2)
- Imports added at the module top: `fnmatch`, `pathlib.Path`, `typing.Pattern`

**3. `collect_local_system_info` wiring** — three-line try/except block after the Phase 3 networking block, mirroring the chassis_model / networking shape exactly:
```python
try:
    result['sysctl'] = collect_sysctl()
except Exception as e:
    result['errors']['sysctl'] = str(e)
    result['sysctl'] = []
```

**4. Pattern B twin in `MPI_COLLECTOR_SCRIPT`** (D-36):
- Inline imports: added `import fnmatch` to the script body
- Inline `_PROC_SYS_ROOT` + `_SYSCTL_ALLOWLIST_LINES` (tuple literal — NO file I/O in the script; see decision note)
- Inline `_load_sysctl_allowlist()` + `collect_sysctl()` untyped twins
- Parallel try/except in script's `collect_local_info` (mirrors module wiring)

**5. Tests** — 18 new tests across 5 new classes in `tests/unit/test_cluster_collector.py`:
- `TestSysctlAllowlistFile` (3): shipped file exists, 4 patterns, canonical keys match
- `TestLoadSysctlAllowlist` (4): comments/blanks skipped, whitespace stripped, missing→empty, .match() method present
- `TestSysctlCollector` (7): allowlist filter, exclusion, multi-value verbatim, per-leaf PermissionError isolation, missing root→[], 8KiB cap, dotted-form conversion
- `TestSysctlMPIScriptParity` (1): exec script, behavioral equivalence vs module on shared `tmp_path` fixture
- `TestSysctlWiring` (3): result['sysctl'] always present + monkeypatch happy-path + monkeypatch failure-isolation

## Two-Commit RED/GREEN Cadence

| Commit | Type | Files | Purpose |
|---|---|---|---|
| 0198011 | test(04-01) | tests/unit/test_cluster_collector.py | RED — 18 failing tests across 5 new classes (ImportError + AssertionError + AttributeError mix) |
| 263d11c | feat(04-01) | mlpstorage_py/system_description/sysctl_allowlist.txt (NEW) + mlpstorage_py/cluster_collector.py | GREEN — allowlist file + module collector + MPI script twin + wiring |

## Pattern B Allowlist-Baked-Into-Script Note

The MPI script body cannot read `mlpstorage_py/system_description/sysctl_allowlist.txt` at exec time (it's shipped as a string, exec'd over SSH on a heterogeneous fleet where the package may not be installed). Pattern B therefore bakes the four globs into the script body as `_SYSCTL_ALLOWLIST_LINES = ('vm.dirty_*', 'net.core.*', 'net.ipv4.tcp_*', 'kernel.numa_balancing')`. A comment in the script flags this as a manual-sync discipline: a future editor adding a glob to the shipped file must also add it to the tuple here. The parity test (`TestSysctlMPIScriptParity`) catches behavioral drift on shared fixtures (the same allowlist passed to both copies must produce identical output), but does NOT catch allowlist-content drift — the test passes its own allowlist into both copies, not the shipped allowlist. This is a deliberate test-design tradeoff: testing the shipped allowlist against the script's baked-in tuple would require Yet Another grep gate, and the planner-recommended discipline is sync-by-comment.

## `_PROC_SYS_ROOT` Path-Indirection Note

Locking `proc_sys_root` as a parameter default rather than reading a module-level walk root via global lookup made every `TestSysctlCollector` test trivial: each test just builds a `tmp_path/proc/sys/...` tree and passes `proc_sys_root=str(root)` directly. No `monkeypatch.setattr` on globals, no fragile builtins.open patching for the happy path (only one test needs builtins.open patching, and only for the specific PermissionError-isolation scenario where the production code's per-leaf exception path is the contract under test).

## Pytest Output Anomalies

The same 7 pre-existing `_check_safe_path_component` MagicMock fixture failures persist out-of-scope per Rule 3:
- `tests/unit/test_datagen_command_generation.py` (5 failures)
- `tests/unit/test_rules_calculations.py::TestGenerateOutputLocation` (2 failures)

These are pre-existing dev-environment fixture failures noted in STATE.md and unrelated to this plan. The same 3 pre-existing dev-env collection-error files (`test_benchmarks_base.py`, `test_parquet_reader.py`, `test_vdb_modular_fake_backend.py`) still require `--ignore=` flags due to missing `psutil` in the dev shell (STATE.md Deferred Items).

Full unit suite: 1728 passed, 7 failed (all pre-existing), 4 skipped, ~28s.
Targeted sysctl class run: 18 passed, ~0.06s.

## Verification

```bash
# All sysctl-related test classes pass.
python3 -m pytest tests/unit/test_cluster_collector.py::TestSysctlAllowlistFile \
    tests/unit/test_cluster_collector.py::TestLoadSysctlAllowlist \
    tests/unit/test_cluster_collector.py::TestSysctlCollector \
    tests/unit/test_cluster_collector.py::TestSysctlMPIScriptParity \
    tests/unit/test_cluster_collector.py::TestSysctlWiring -q
# → 18 passed

# Shipped allowlist file lines.
grep -c '^vm\.dirty_\*$' mlpstorage_py/system_description/sysctl_allowlist.txt
# → 1

# Two defs of collect_sysctl (module + script twin), excluding comment lines.
grep -v '^#' mlpstorage_py/cluster_collector.py | grep -c 'def collect_sysctl'
# → 2

# End-to-end allowlist load via the installed package.
python3 -c "from mlpstorage_py.cluster_collector import _load_sysctl_allowlist; print(len(_load_sysctl_allowlist()))"
# → 4
```

All three plan-level success criteria from `<verification>` satisfied.

## Deviations from Plan

None — plan executed exactly as written. No Rule 1/2/3 auto-fixes triggered; the RED→GREEN cycle was clean on first run.

The plan acceptance criteria explicitly allowed `os.geteuid()==0` decorator concerns (carried from Phase 03-02), but no sysctl test required a geteuid-skip decorator — the only PermissionError test patches `builtins.open` inside the test body, which works for both root and non-root users.

## Authentication Gates

None.

## Forward Notes for Plan 04-02

Plan 04-02 ships the environment collector + D-25 redactor unification in `storage_config.py`. Pattern A/B discipline mirrors what 04-01 just established:
- New module-scope constants block alongside `_PROC_SYS_ROOT` / `_SYSCTL_ALLOWLIST_PATH`
- New helpers `_env_allowlist_match`, plus refactored `_mask_credential_id` / `_redact_secret` in `storage_config.py`
- Pattern B twins of the redactors inline in the MPI script next to `collect_environment` (storage_config can't be imported at exec time)
- Per-leaf D-2 (each env var is independently redactable; one redaction failure skips that var)
- ROADMAP SC #2 reconciliation: drops the sha256 fingerprint wording per D-24 (length-only redaction). Hygiene commit in 04-02 or 04-05; planner's call.

## Self-Check: PASSED

- mlpstorage_py/system_description/sysctl_allowlist.txt: FOUND
- mlpstorage_py/cluster_collector.py: collect_sysctl defined (module + script twin)
- mlpstorage_py/cluster_collector.py: _load_sysctl_allowlist defined (module + script twin)
- mlpstorage_py/cluster_collector.py: collect_local_system_info wired with try/except
- tests/unit/test_cluster_collector.py: 5 new test classes, 18 new tests, all GREEN
- Commit 0198011 (RED): FOUND
- Commit 263d11c (GREEN): FOUND
