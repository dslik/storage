---
phase: 02-first-run-write-of-partial-systemname-yaml
plan: 04
subsystem: system_description/auto_generator
tags:
  - python
  - filesystem
  - atomic-io
  - yaml
  - security
  - tdd
dependency_graph:
  requires:
    - mlpstorage_py/system_description/auto_generator.py (from 02-02/02-03 — node_dict_from_host, group_by_fingerprint, _FINGERPRINT_KEYS, _splice_stub_lists, _build_outer_dict)
    - mlpstorage_py/cluster_collector.py:collect_local_system_info (D-8 fallback)
    - mlpstorage_py/rules/models.py:HostInfo.from_collected_data (D-8 bridge)
    - mlpstorage_py/results_dir/sentinel.py:113-134 (atomic-write recipe analog)
  provides:
    - mlpstorage_py.system_description.auto_generator._SYSTEMNAME_YAML_MODE (Final[int] = 0o644)
    - mlpstorage_py.system_description.auto_generator._resolve_host_info_list(cluster_info) (D-8 fallback)
    - mlpstorage_py.system_description.auto_generator.write_systemname_yaml(args, cluster_info, logger) (LIFE-01 orchestrator)
  affects:
    - 02-05 (Benchmark.run() hook will call write_systemname_yaml as the sole call site; integration tests will assert the on-disk YAML; kvcache/vectordb regressions will confirm no behavior leakage)
    - Phase 5 LIFE-02 (the FileExistsError branch — currently `return None` + debug log — will be replaced with diff-and-fail)
tech_stack:
  added: []
  patterns:
    - "Atomic exclusive create — os.open(..., O_CREAT|O_EXCL|O_WRONLY, 0o644) verbatim from results_dir/sentinel.py:113-134"
    - "FileExistsError as the no-op-if-exists branch (Phase 5 will replace with diff-and-fail)"
    - "D-8 single-host fallback via cluster_collector.collect_local_system_info + HostInfo.from_collected_data"
    - "D-7 sort tuple (-quantity, chassis.cpu_model) — Python's stable sort gives alphabetical tiebreak"
    - "yaml.safe_dump with default_style='\"' + explicit_start=True + default_flow_style=False + sort_keys=False (D-10)"
    - "mkdir(parents=True, exist_ok=True) on the parent — deviates from sentinel.py which relies on init having pre-created the parent"
key_files:
  created:
    - tests/unit/test_auto_generator_write.py
  modified:
    - mlpstorage_py/system_description/auto_generator.py
decisions:
  - "Pitfall 6 CORRECTED: modern PyYAML with default_style='\"' emits integers as `!!int \"N\"` (tagged double-quoted), NOT as bare unquoted ints as the PLAN/RESEARCH claimed. The PLAN's `test_yaml_formatting_integers_unquoted` test (regex `quantity:\\s+\\d+\\s*$`) cannot pass — the on-disk byte pattern is `\"quantity\": !!int \"3\"`. What matters for the schema validator and submission checker is that `yaml.safe_load` recovers Python `int`, which the `!!int` tag guarantees. Replaced the byte-pattern test with two tests: (a) `test_yaml_formatting_integers_round_trip_as_int` locks the round-trip type behavior; (b) `test_yaml_formatting_integers_tagged_not_string` locks the `!!int` tag emission so a future PyYAML version-bump that drops the tag (silently turning ints into strings) is caught at test time. Both tests honor the semantic intent of Pitfall 6 (ints stay ints across emit→load) without depending on the incorrect on-disk byte-pattern assumption."
  - "PyYAML default_style='\"' quotes KEYS too. The PLAN test `test_yaml_formatting_strings_double_quoted` used regex `cpu_model:\\s*\"...\"` expecting unquoted keys with double-quoted values. Actual emit: `\"cpu_model\": \"Intel...\"`. Updated the regex to `\"cpu_model\":\\s*\"...\"` and added a yaml.safe_load round-trip assertion to lock the semantic intent (strings round-trip as Python str, no plain-scalar misinterpretation). The PLAN's `test_yaml_block_style` also needed an updated traffic-list pattern: `\"traffic\": []` not `traffic: []`."
  - "Test framework choice: SimpleNamespace for `args` (strict attribute access — missing attributes raise AttributeError, catching drift in the function's expected args.* surface), MagicMock for `cluster_info` (lets us set `host_info_list` as an attribute) and `logger` (lets us assert .debug / .info call patterns). Mirrors test_results_dir_sentinel.py's style discipline."
  - "Patch target choice for D-8 fallback test: `mlpstorage_py.system_description.auto_generator.collect_local_system_info` (the writer's IMPORT site), NOT `mlpstorage_py.cluster_collector.collect_local_system_info` (the source module). Patching the source module would silently miss because the writer has already bound a local reference at import time. RESEARCH.md explicitly flagged this; locked via the patch context manager."
  - "Test ordering compressed to two commits per success_criteria mandate. Same structural decision as 02-02 / 02-03. RED gate verifiably observed before any production code change (ImportError: cannot import name '_SYSTEMNAME_YAML_MODE')."
  - "Grep-vs-docstring divergence (same flavor as 02-02 / 02-03). The PLAN's forbidden-token grep gates for `generate_output_location` (expected 0, actual 1) and `validate_file` (expected 0, actual 3) all match docstring mentions explaining the deferred-to-Phase-5 / D-15 boundaries — NOT production code calls. The semantic D-15 / Pitfall 10 intent (no runtime invocation) is fully honored. The docstring text exists specifically to make the deliberate-omission decisions discoverable to future maintainers; stripping it to satisfy a flawed grep would degrade the code's documentation."
  - "Upstream args.systemname contract verified (Pitfall 10): the PLAN required confirming `generate_output_location` raises before Benchmark.run() is reached. Verified at `mlpstorage_py/rules/utils.py:187-195` and `mlpstorage_py/benchmarks/base.py:_reserve_run_directory`: an empty/missing `args.systemname` raises ConfigurationError during Benchmark.__init__, BEFORE Benchmark.run() executes and BEFORE write_systemname_yaml is ever invoked. Phase 2's writer can therefore safely consume `args.systemname` without an additional guard."
metrics:
  duration_min: ~30
  tasks_total: 1
  tasks_completed: 1
  files_changed: 2
  commits: 2
  completed_date: 2026-06-19
---

# Phase 02 Plan 04: write_systemname_yaml Atomic Orchestrator Summary

The on-disk side of the auto-generator vertical ships in
`mlpstorage_py.system_description.auto_generator`: three new symbols
(`_SYSTEMNAME_YAML_MODE`, `_resolve_host_info_list`, `write_systemname_yaml`)
compose 02-02's adapter + grouping with 02-03's outer dict + stub splice,
add the D-7 sort + D-11 path + D-12 command gate + D-9 atomic
`O_CREAT|O_EXCL|O_WRONLY` write + D-10 YAML formatting, and produce a real
file on disk at `<results_dir>/<mode>/<orgname>/systems/<systemname>.yaml`.

## What Was Built

Slice 4 of Phase 02 — the filesystem write orchestrator. Two commits
(RED `0978405` + GREEN `509bcf7`), 28 new test cases, one consolidated
TDD slice.

The deliverable closes the standalone-testable side of LIFE-01: a unit
test now can construct a `MagicMock(args)` with `command='run'`,
`results_dir=tmp_path`, `mode='closed'`, `orgname='Acme'`,
`systemname='sys-v1'`, hand it a fake cluster_info with N homogeneous
`HostInfo` objects, call `write_systemname_yaml(args, cluster_info, logger)`,
open the returned path, `yaml.safe_load` it, and assert
`data['system_under_test']['clients'][0]['quantity'] == N`.

What 02-05 now consumes:

```python
from mlpstorage_py.system_description.auto_generator import (
    write_systemname_yaml,
)

# Inside Benchmark.run() (the 02-05 hook):
path = write_systemname_yaml(self.args, self.cluster_info, self.logger)
# path is the canonical D-11 path on first run, None on re-run or
# datagen / configview / etc.
```

## Public Symbols Added

| Symbol | Signature | Purpose |
|---|---|---|
| `_SYSTEMNAME_YAML_MODE` | `Final[int] = 0o644` | File mode for emitted YAML. LAY-03 parity with `results_dir/sentinel.py:_SENTINEL_MODE`. World-readable on purpose (every command must read on shared boxes). |
| `_resolve_host_info_list(cluster_info)` | `(cluster_info) -> list[HostInfo]` | D-8 fallback: returns `cluster_info.host_info_list` if populated; otherwise `collect_local_system_info` → `HostInfo.from_collected_data` → `[single_host]`. |
| `write_systemname_yaml(args, cluster_info, logger)` | `(args, cluster_info, logger) -> Optional[str]` | LIFE-01 orchestrator. Composes 02-02 adapter + grouping, D-7 sort, 02-03 outer dict + splice, D-11 path with mkdir-on-demand, D-12 command gate, D-9 atomic O_EXCL write + FileExistsError no-op, D-10 yaml.safe_dump kwargs. Returns path on first-run write, None on D-12 skip or D-9 FileExistsError. Propagates non-FileExistsError filesystem errors per D-9. |

Symbols arriving in 02-05 (NOT in this module): the Benchmark.run() hook
call site itself lives in `mlpstorage_py/benchmarks/base.py`.

## Race-Test Threading Pattern

`test_concurrent_writers_one_wins` mirrors RESEARCH.md Code Example
lines 676-700 verbatim:

```python
barrier = threading.Barrier(2)
results: list = []

def worker():
    barrier.wait()  # Synchronize both threads' entry.
    results.append(write_systemname_yaml(args, cluster_info, MagicMock()))

t1 = threading.Thread(target=worker)
t2 = threading.Thread(target=worker)
t1.start(); t2.start(); t1.join(); t2.join()

paths = [r for r in results if r is not None]
nones = [r for r in results if r is None]
assert len(paths) == 1 and len(nones) == 1
```

The `threading.Barrier(2)` blocks both threads at `barrier.wait()` until
both have called it, then releases them simultaneously. Both threads then
race into `os.open(... O_CREAT|O_EXCL|O_WRONLY)`. The kernel guarantees
exactly one syscall succeeds. The test ran cleanly on every invocation
during development — no flakiness, no race hazards beyond the intended one.

## Tasks

| Task | Name | Commit |
| --- | --- | --- |
| 1 (RED) | failing write-path tests (LIFE-01 / D-9 / D-12 + T-2-01 / T-2-08) | `0978405` |
| 1 (GREEN) | write_systemname_yaml orchestrator (LIFE-01) | `509bcf7` |

The PLAN.md describes Task 1 as a single TDD cycle ("Step 1 RED, Step 2
GREEN, Step 3 iterate, Step 4 regression"). This plan honored that
single-task structure directly — one RED commit, one GREEN commit. RED
gate verifiably observed (`ImportError: cannot import name
'_SYSTEMNAME_YAML_MODE'`) before any production code existed.

## Test Count Delta

| Metric | Before | After | Δ |
| --- | --- | --- | --- |
| `tests/unit/test_auto_generator_write.py` cases | 0 (file did not exist) | 28 | +28 |
| `tests/unit/test_auto_generator.py` cases (regression) | 39 | 39 | 0 |
| Adjacent regression suite (auto_generator + auto_generator_write + cluster_collector + rules_dataclasses + example_system_descriptions + results_dir_sentinel + cli) | 294 | 322 | +28 |

- `pytest tests/unit/test_auto_generator_write.py -v` → 28 passed in 0.16s
  (including the parametrized `test_non_run_commands_skip_write` × 6 + the
  threading race test).
- `pytest tests/unit/test_auto_generator.py tests/unit/test_auto_generator_write.py -v`
  → 67 passed in 0.21s (no regression in 02-02 / 02-03 tests).
- Adjacent regression suite → 322 passed in 3.38s, zero new failures.

## Verification (acceptance criteria from PLAN)

- `pytest tests/unit/test_auto_generator_write.py -x` → exit 0.
- `pytest tests/unit/test_auto_generator.py -x` → exit 0 (no regression).
- `pytest tests/unit/test_auto_generator_write.py::test_concurrent_writers_one_wins -x` → exit 0 (T-2-01 locked).
- `pytest tests/unit/test_auto_generator_write.py::test_symlink_attack_at_target_path_returns_none -x` → exit 0 (T-2-08 locked).
- `pytest tests/unit/test_auto_generator_write.py::test_yaml_formatting_integers_round_trip_as_int -x` → exit 0 (Pitfall 6 semantic intent locked — see Decisions for the byte-pattern correction).
- `pytest tests/unit/test_auto_generator_write.py::test_writer_does_not_call_schema_validator_validate_file -x` → exit 0 (D-15 locked).
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c 'O_CREAT.*O_EXCL'` → 2 (call site + module docstring mention).
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c 'yaml\.dump[^_(]\|yaml\.load[^_]'` → 0 (T-2-04 grep gate).
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c 'yaml\.safe_dump'` → 2.
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c "default_style"` → 2 (call site + docstring).
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c "explicit_start=True"` → 2.
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c "generate_output_location"` → 1 (docstring only; see Decisions).
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c "validate_file"` → 3 (docstrings only; see Decisions).
- PLAN inline smoke `python -c "...write_systemname_yaml..."` → `ok`.
- PLAN end-to-end smoke (homogeneous 2-host, asserts `quantity == 2`,
  `cpu_qty == 2`, `operating_system.name == 'Rocky Linux'`,
  `networking == [{...}]`, `'solution' not in system_under_test`) → confirmed via
  test_writes_at_canonical_path + test_systemname_yaml_omits_solution_deployment_in_emitted_file +
  test_yaml_formatting_strings_double_quoted (round-trip assertion).

## Surprise Discoveries

### 1. PyYAML `default_style='"'` quotes KEYS, not just values

The PLAN's `test_yaml_formatting_strings_double_quoted` used regex
`cpu_model:\s*"..."` — bare key followed by colon followed by
double-quoted value. Actual emit: `"cpu_model": "Intel..."` — keys are
ALSO double-quoted. This is documented PyYAML behavior for
`default_style='"'`: it applies the style to ALL scalars including
mapping keys.

Fix: update the regex to `"cpu_model":\s*"..."` and supplement with a
`yaml.safe_load` round-trip assertion confirming `cpu_model` recovers as
a Python string. The semantic D-10 intent (strings stay strings, no
plain-scalar misinterpretation) is locked via the round-trip; the
on-disk byte pattern is locked via the updated regex.

### 2. PyYAML emits integers as `!!int "N"` (tagged), not bare unquoted

The PLAN explicitly cited "Pitfall 6: PyYAML emits int natively even
with `default_style='"'`" and asked for a test
`test_yaml_formatting_integers_unquoted` matching `quantity:\s+\d+\s*$`.
Actual emit: `"quantity": !!int "3"`. PyYAML uses the `!!int` YAML tag
to disambiguate — a tagged double-quoted scalar still round-trips to
Python `int` via `yaml.safe_load`, which IS what the schema validator
and submission checker need.

The Pitfall 6 SEMANTIC intent (the round-trip type is preserved across
emit→load, so an `int 3` stored in the source dict comes back as `int 3`)
is correct and important. The on-disk BYTE PATTERN the PLAN cited is
incorrect for this PyYAML version. Replaced with two tests:

- `test_yaml_formatting_integers_round_trip_as_int` — locks the
  round-trip type (the actually-important contract for downstream
  consumers).
- `test_yaml_formatting_integers_tagged_not_string` — locks the `!!int`
  tag emission. A future PyYAML that drops the tag and silently turns
  ints into untagged double-quoted strings (which would round-trip as
  str) is caught immediately.

Both tests honor Pitfall 6 without depending on the incorrect on-disk
byte-pattern assumption.

### 3. `Path.mkdir(parents=True, exist_ok=True)` is benign-race-safe

Per RESEARCH.md Pitfall 7, the mkdir race is benign — POSIX `mkdir(2)`
with `EEXIST` is exactly what `exist_ok=True` handles. The test
`test_path_parent_mkdir_creates_systems_dir` confirms creation when
absent; the concurrent-write race test (T-2-01) implicitly confirms
mkdir doesn't blow up when called from both threads near-simultaneously.

### 4. The `validate` command's writer-side gate fires correctly even though Phase 1 Slice 4 left the runtime gate inert

The PLAN's `test_non_run_commands_skip_write` parametrizes over
`['datagen', 'configview', 'datasize', 'validate', 'history', 'reportgen']`
and asserts the writer skips for all six. `validate` is interesting
because Phase 1 / 01-04 documented an "inert-gate" divergence where the
sentinel-check gate quietly skips it (validate uses a positional `input`
not `--results-dir`). That divergence does NOT affect THIS plan — the
writer's gate checks `args.command != 'run'` and fires for `validate`
just as it does for the other five non-run commands. The two gates are
independent.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] PyYAML byte-pattern assumptions in two of the formatting tests were incorrect.**
- **Found during:** Task 1 GREEN iteration, after the implementation passed but the formatting tests failed.
- **Issue:** The PLAN cited byte patterns (`cpu_model:\s*"..."` for strings, `quantity:\s+\d+\s*$` for ints) that don't match the actual PyYAML 5.x/6.x behavior with `default_style='"'`. Keys are quoted alongside values; integers carry the `!!int` tag.
- **Fix:** Updated `test_yaml_formatting_strings_double_quoted` regex to match `"cpu_model"` (quoted key); replaced `test_yaml_formatting_integers_unquoted` with `test_yaml_formatting_integers_round_trip_as_int` (locks semantic intent) + `test_yaml_formatting_integers_tagged_not_string` (locks tag emission so a version drift is caught). Both new tests honor Pitfall 6's semantic intent (ints round-trip as ints) without depending on the incorrect byte-pattern assumption.
- **Files modified:** `tests/unit/test_auto_generator_write.py`.
- **Commit:** GREEN commit `509bcf7` (the corrected tests landed alongside the production code, so the slice still ships in two commits per success_criteria mandate).
- **Why this is safe:** the round-trip-via-yaml.safe_load assertion is a STRONGER lock than the byte-pattern regex — it asserts the contract that actually matters for the schema validator (Python int recovered as Python int). A future PyYAML change that breaks the round-trip will be caught immediately.

### Documentation deviations

**2. [Rule 3 — Grep gate vs docstring text]** Same flavor as 02-02 and 02-03. The PLAN's forbidden-token grep gates for `generate_output_location` (expected 0, actual 1) and `validate_file` (expected 0, actual 3) all match docstring mentions explaining the deferred-to-Phase-5 / D-15 boundaries — NOT production code calls.
- **Found during:** Acceptance-criteria verification.
- **Resolution:** Semantic D-15 / Pitfall 10 intent (no runtime invocation) is fully honored via test_writer_does_not_call_schema_validator_validate_file (D-15 lock). The docstring text exists specifically to make the deliberate-omission decisions discoverable to future maintainers.
- **Files modified:** None — the docstring mentions stay for human readability.

### Structural deviations

None this slice. The PLAN described Task 1 as a single TDD cycle and the
success_criteria mandated exactly two commits (RED + GREEN); the
single-task structure was followed directly.

## Known Stubs

**None introduced.** This plan composes 02-02's adapter and 02-03's
stub literals into an on-disk artifact; it does not introduce new
blanks. The stubs in the emitted YAML (the `networking[]` /
`drives[]` SER-02 to-do reminders, the `friendly_description` /
`chassis.model_name` blanks waiting for human input) are all from
02-02 / 02-03 and were catalogued in those summaries. Plans 3 and 4 of
the milestone will replace some of those blanks with real auto-collected
data.

## Threat Flags

**None new.** The plan introduces the FILESYSTEM I/O surface but every
threat dispositioned in the PLAN's `<threat_model>` block was already
mitigated by design:

- **T-2-01 (Tampering, race):** `os.open(..., O_CREAT|O_EXCL|O_WRONLY,
  0o644)` — kernel-level race-free. Verified by
  `test_concurrent_writers_one_wins` using `threading.Barrier(2)`.
- **T-2-02 (Tampering, path traversal):** mitigated upstream by Phase
  1's `generate_output_location` (raises ConfigurationError on
  empty/malformed systemname) and sentinel orgname regex. Phase 2's
  writer relies on the upstream contract per Pitfall 10 — no redundant
  guard added.
- **T-2-04 (Tampering, YAML emit):** `yaml.safe_dump` ONLY. Grep gate
  enforced at acceptance criteria. (Note: a `yaml.safe_load` call in
  `test_yaml_formatting_strings_double_quoted` exists in TEST code only,
  which is the documented exception.)
- **T-2-08 (Tampering, symlink-at-target):** POSIX `O_EXCL` fails with
  EEXIST if the path resolves to anything existing, including a symlink.
  Verified by `test_symlink_attack_at_target_path_returns_none` — the
  innocent file at the symlink target is read after the failed write and
  confirmed unchanged.
- **T-2-05 / T-2-06 / T-2-07 / T-2-03 / T-2-SC:** dispositioned `accept`
  or `N/A` per the PLAN; nothing in this plan changes their
  disposition.

Block-on: high. T-2-01 (race) and T-2-08 (symlink) are blocking; both
tests pass green.

## TDD Gate Compliance

- **RED gate:** `0978405 test(02-04): add failing write-path tests
  (LIFE-01/D-9/D-12 + T-2-01/T-2-08)`. Confirmed `ImportError: cannot
  import name '_SYSTEMNAME_YAML_MODE' from
  'mlpstorage_py.system_description.auto_generator'` on
  `pytest tests/unit/test_auto_generator_write.py -x` before any
  production-code change.
- **GREEN gate:** `509bcf7 feat(02-04): write_systemname_yaml
  orchestrator (LIFE-01)`. `pytest tests/unit/test_auto_generator_write.py
  tests/unit/test_auto_generator.py -v` → 67 passed in 0.21s.
- **REFACTOR gate:** not needed. The module landed at its intended
  final shape; no internal reshuffling required.

Both commits omit any `Co-Authored-By:` AI attribution per
`feedback_no_attribution.md` / MEMORY.md.

## What Plan 02-05 Can Now Do

**Plan 02-05 (Benchmark.run() hook + integration tests)** can now hook
the writer into `mlpstorage_py/benchmarks/base.py:Benchmark.run()`:

```python
from mlpstorage_py.system_description.auto_generator import (
    write_systemname_yaml,
)

# Inside Benchmark.run(), just after MPI collection completes:
write_systemname_yaml(self.args, self.cluster_info, self.logger)
```

The writer self-gates on `args.command != 'run'`, so a single call site
suffices — the writer correctly no-ops for datagen / configview /
datasize / validate / history / reportgen. The `cluster_info=None`
fallback handles dev-iteration paths where the MPI collector has not
populated a cluster_info yet. The FileExistsError branch handles re-runs
(LIFE-01: "exists → don't touch", Phase 5 will replace with diff-and-fail).

Plan 02-05's integration tests can assert the on-disk YAML directly:

```python
# After running a full Benchmark.run() against a tmp results-dir:
target = tmp_path / "closed" / "Acme" / "systems" / "sys-v1.yaml"
data = yaml.safe_load(target.read_text())
assert data["system_under_test"]["clients"][0]["quantity"] == 3
```

## Deferred Items

| Category | Item | Status | Notes |
|---|---|---|---|
| Test env | `psutil` and `numpy` not installed in dev shell; collection-time `ModuleNotFoundError` on `tests/unit/test_benchmarks_*.py`, `test_utils.py`, `test_datagen_command_generation.py`, `test_dlio_object_storage.py`, `test_parquet_reader.py`, `test_reporting.py`, `test_vdb_modular_fake_backend.py`. | Deferred — pre-existing, not introduced by this plan. | Same as 02-02 / 02-03. Resolution: `pip install -e ".[test]"` once. Out of scope per the scope-boundary rule. The 322-test targeted regression suite this plan ran covers every test file that imports from auto_generator + the adjacent rules/cluster/sentinel/cli subsystems. |

## Self-Check: PASSED

- `mlpstorage_py/system_description/auto_generator.py` exists with three
  new symbols (`_SYSTEMNAME_YAML_MODE`, `_resolve_host_info_list`,
  `write_systemname_yaml`). FOUND.
- `tests/unit/test_auto_generator_write.py` exists with 28 test cases.
  FOUND.
- Commit `0978405` (RED) present in `git log --oneline`. FOUND.
- Commit `509bcf7` (GREEN) present in `git log --oneline`. FOUND.
- Full `pytest tests/unit/test_auto_generator_write.py -v` → 28 passed.
  PASSED.
- Full `pytest tests/unit/test_auto_generator.py
  tests/unit/test_auto_generator_write.py -v` → 67 passed. PASSED.
- Adjacent 322-test regression suite → 322 passed in 3.38s, zero new
  failures. PASSED.
- Module surface importable:
  `from mlpstorage_py.system_description.auto_generator import
  _SYSTEMNAME_YAML_MODE, _resolve_host_info_list, write_systemname_yaml`
  → no error. PASSED.
- T-2-01 race-test green (5+ consecutive runs, no flakiness). PASSED.
- T-2-08 symlink-test green (POSIX O_EXCL semantics confirmed). PASSED.
- D-15 no-validate_file-call lock green
  (`test_writer_does_not_call_schema_validator_validate_file`). PASSED.
