---
phase: 02-first-run-write-of-partial-systemname-yaml
plan: 02
subsystem: system_description/auto_generator
tags:
  - python
  - pure-functions
  - data-transformation
  - tdd
dependency_graph:
  requires:
    - mlpstorage_py/rules/models.py:HostInfo (existing dataclass)
    - mlpstorage_py/rules/models.py:HostCPUInfo.num_sockets (added in Plan 02-01)
    - mlpstorage_py/rules/models.py:HostMemoryInfo (existing; .total is bytes per from_proc_meminfo_dict)
    - mlpstorage_py/cluster_collector.py:HostSystemInfo (existing; os_release is Optional[Dict[str, str]])
    - mlpstorage_py/system_description/schema_validator.py:Chassis (TEST-only model_fields reflection)
    - mlpstorage_py/system_description/schema_validator.py:OperatingSystem (TEST-only model_fields reflection)
  provides:
    - mlpstorage_py.system_description.auto_generator._FINGERPRINT_KEYS (locked six-key tuple per D-4)
    - mlpstorage_py.system_description.auto_generator._get_dotted(d, dotted_key) (nested-dict walker; missing key → "")
    - mlpstorage_py.system_description.auto_generator.group_by_fingerprint(items, fingerprint_keys, count_field) (generic quantity-grouping helper)
    - mlpstorage_py.system_description.auto_generator.node_dict_from_host(host) (HostInfo → NodeDescription-shaped dict adapter; covers COLL-01, COLL-02, D-6)
  affects:
    - 02-03 (stub-splice + outer-dict builder — imports node_dict_from_host + group_by_fingerprint from here)
    - 02-04 (write_systemname_yaml atomic write — imports the same plus _FINGERPRINT_KEYS; owns D-7 sort)
    - 02-05 (Benchmark.run() hook — calls write_systemname_yaml; transitively depends on this module)
tech_stack:
  added: []
  patterns:
    - "Pure-function transformation core — no I/O, no global state, no side effects"
    - "Universal collection-failure rule — every field defaults to empty string on any missing/zero source"
    - "Pydantic-as-test-oracle — leaf models exercised only via .model_fields.keys() reflection, never constructed inside the adapter (Pitfall 2)"
    - "copy.deepcopy(item) per accepted entry — preserves the no-mutation invariant for repeated grouping calls"
    - "Dotted-key fingerprint resolution — keeps the grouping helper schema-agnostic"
key_files:
  created:
    - mlpstorage_py/system_description/auto_generator.py
    - tests/unit/test_auto_generator.py
  modified: []
decisions:
  - "D-4 / _FINGERPRINT_KEYS shape locked: ('chassis.cpu_model', 'chassis.cpu_qty', 'chassis.cpu_cores', 'chassis.memory_capacity', 'operating_system.name', 'operating_system.version'). Order is human-readable convention only; group_by_fingerprint hashes via tuple equality."
  - "D-5 implementation: _get_dotted returns '' on any missing nested key OR non-dict intermediate value; group_by_fingerprint treats those empty strings as full fingerprint participants. Failed-collection hosts group together as their own stanza rather than being absorbed into real-CPU stanzas."
  - "D-6 implementation: memory_capacity = round(host.memory.total / (1024**3)) (binary GiB). Verified the 271_652_882_432-byte parametrized test case ROUNDS UP to 253 — that quantity is actually ~252.9965 GiB, not the 252.5 GiB the plan author tentatively cited. Python round-half-to-even is therefore not exercised at this boundary; the result follows from straight float arithmetic. Note added to memory test docstring."
  - "D-16 consumption: node_dict_from_host reads chassis.cpu_qty from host.cpu.num_sockets (the field 02-01 added). The truthy guard `host.cpu.num_sockets if (host.cpu and host.cpu.num_sockets) else ''` preserves the 0-as-collection-failure semantics — when summarize_cpuinfo could not determine socket count and emitted the dataclass default 0, the adapter blanks the field instead of pretending we saw zero sockets."
  - "Pitfall 2 enforced: zero leaf Pydantic construction in auto_generator.py (grep gate `NodeDescription(\\|Chassis(\\|OperatingSystem(` returns 0). Chassis.model_fields and OperatingSystem.model_fields are imported in TESTS only, exclusively via .keys() reflection for schema-drift detection."
  - "Pitfall 4 enforced: only NAME → name and VERSION_ID → version (grep gate for `\"NAME\"`/`\"VERSION_ID\"` returns 2 in production code). PRETTY_NAME / ID / VERSION / VERSION_CODENAME are explicitly NOT consulted. Test test_os_field_mapping injects all six keys with distinguishable values to lock against future drift."
  - "Pitfall 9 enforced: OS extraction uses the .get(k, '') or '' idiom to collapse both missing keys AND explicit None values to empty string."
  - "Quoting convention: project style uses double-quoted strings (PEP-8 silent, but consistent across mlpstorage_py/). The plan's acceptance-criteria grep `'NAME'\\|'VERSION_ID'` was authored expecting single quotes; the semantic intent (Pitfall 4 explicit key selection) is honored — the equivalent double-quoted grep returns 2."
  - "Two-commit slice ordering: PLAN.md describes Task 1 and Task 2 as separate TDD cycles, but the success_criteria explicitly require exactly TWO total commits (RED for tests + GREEN for production). Followed success_criteria — one consolidated RED test commit covering both helpers + adapter, one consolidated GREEN feat commit shipping both. RED gate was verifiably observed (ModuleNotFoundError on auto_generator before any production code)."
metrics:
  duration_min: ~25
  tasks_total: 2
  tasks_completed: 2
  files_changed: 2
  commits: 2
  completed_date: 2026-06-19
---

# Phase 02 Plan 02: node_dict_from_host adapter + group_by_fingerprint helper Summary

`mlpstorage_py.system_description.auto_generator` — a new pure-transformation module with four symbols (`_FINGERPRINT_KEYS`, `_get_dotted`, `group_by_fingerprint`, `node_dict_from_host`) — closes the in-memory side of the Phase 2 vertical: a list of `HostInfo` objects now flows through `node_dict_from_host` and `group_by_fingerprint` into a quantity-grouped `list[dict]` ready for Plan 02-03 to splice with networking/drives stubs and for Plan 02-04 to atomically write to `systemname.yaml`.

## What Was Built

Slice 2 of Phase 02 — the pure-transformation core of the auto-collector. Zero I/O. Zero Pydantic construction. Two source files, 25 test cases, two commits.

The deliverable is the boundary between "MPI-collected `HostInfo` objects sitting in memory" and "node_description-shaped dicts ready for YAML emission". Nothing in this module touches the filesystem, the network, or any external process; everything is dict-in, dict-out, deterministic on input order. That keeps the I/O ownership cleanly scoped to Plan 02-04, the stub-splice + outer-dict scaffolding cleanly scoped to Plan 02-03, and the `Benchmark.run()` integration cleanly scoped to Plan 02-05.

The adapter satisfies COLL-01 (cpu_model, cpu_qty, cpu_cores, memory_capacity) and COLL-02 (operating_system.name, operating_system.version) at the per-host level. The grouping helper satisfies SER-01 (homogeneous fleets → one stanza with `quantity: N`; heterogeneous fleets → multiple stanzas whose quantities sum to N).

## Public Symbols Added

| Symbol | Signature | Purpose |
|---|---|---|
| `_FINGERPRINT_KEYS` | `tuple[str, ...]` | Locked six-key tuple per D-4: `("chassis.cpu_model", "chassis.cpu_qty", "chassis.cpu_cores", "chassis.memory_capacity", "operating_system.name", "operating_system.version")` |
| `_get_dotted(d, dotted_key)` | `(dict, str) -> Any` | Walk nested dicts via dotted path; missing key at any depth → `""` (D-5) |
| `group_by_fingerprint(items, fingerprint_keys, count_field)` | `(list[dict], tuple[str, ...], str) -> list[dict]` | Collapse items sharing all fingerprint keys into one entry per group annotated with `count_field: N`; first-occurrence order; deep-copies inputs; empty strings participate (D-5) |
| `node_dict_from_host(host)` | `(HostInfo) -> dict` | Map `HostInfo` → NodeDescription-shaped dict with COLL-01/COLL-02 fields filled defensively; emits `""` for any missing source per the universal collection-failure rule |

Symbols arriving in later slices (NOT in this module yet, by design): `_NETWORKING_STUB`, `_DRIVE_STUB`, `_splice_stub_lists`, `_build_outer_dict` (Plan 02-03); `write_systemname_yaml`, atomic write, FileExistsError no-op (Plan 02-04).

## Tasks

| Task | Name | Commit |
| --- | --- | --- |
| 1+2 (RED) | failing adapter + grouping tests (COLL-01/COLL-02/SER-01) | `71342e9` |
| 1+2 (GREEN) | node_dict_from_host + group_by_fingerprint (D-4, D-6, COLL-01/02 prep) | `5edcf5f` |

The PLAN.md split Task 1 (grouping) from Task 2 (adapter) for narrative clarity, but the success_criteria explicitly mandate exactly two commits for Slice 2 ("Slice 2 ships in two commits: `test(02-02): ...` (RED) → `feat(02-02): ...` (GREEN)"). Honored the success_criteria with one RED test commit covering both helpers and the adapter, and one GREEN feat commit shipping both production functions in the same module. RED gate verifiably observed (`ModuleNotFoundError: No module named 'mlpstorage_py.system_description.auto_generator'`) before any production code was written.

## Test Count Delta

| Metric | Before | After | Δ |
| --- | --- | --- | --- |
| `tests/unit/test_auto_generator.py` cases | 0 (file did not exist) | 25 | +25 |
| Adjacent regression suite (auto_generator + cluster_collector + rules_dataclasses + example_system_descriptions) | 150 | 175 | +25 |

`pytest tests/unit/test_auto_generator.py -v` → 25 passed in 0.12s.
`pytest tests/unit/test_auto_generator.py tests/unit/test_cluster_collector.py tests/unit/test_rules_dataclasses.py tests/unit/test_example_system_descriptions.py` → 175 passed in 0.65s.

## Verification (acceptance criteria from PLAN)

**Task 1 (grouping):**
- `pytest tests/unit/test_auto_generator.py -x -v -k "get_dotted or group_by_fingerprint or empty_strings_participate"` → exit 0.
- `grep -c '_FINGERPRINT_KEYS'` → 3 (definition + 2 docstring/usage mentions).
- `grep -c 'copy.deepcopy'` → 1.
- `grep -c 'def group_by_fingerprint'` → 1.
- `grep -c 'yaml\.load[^_]'` → 0 (T-2-04 grep gate).
- `grep -c 'yaml\.dump[^_(]'` → 0.
- Inline Python smoke check (homogeneous fleet → `quantity == 3`) → `ok`.

**Task 2 (adapter):**
- `pytest tests/unit/test_auto_generator.py -x -v` → exit 0 (all 25 tests green).
- `grep -c 'def node_dict_from_host'` → 1.
- `grep -c '1024 \*\* 3\|1024\*\*3\|1024 \* 1024 \* 1024'` → 3.
- `grep -c 'round('` → 2.
- `grep -c 'NodeDescription(\|Chassis(\|OperatingSystem('` → 0 (Pitfall 2 lock honored).
- `grep -c "\"NAME\"\|\"VERSION_ID\""` → 2 (Pitfall 4 explicit key selection; double-quoted to match project style — see Decisions).
- Inline Python smoke check (real HostInfo → `cpu_qty == 2 and memory_capacity == 1 and os name == 'Rocky'`) → `ok`.

**Module surface:**
```
python3 -c "from mlpstorage_py.system_description.auto_generator import (
    node_dict_from_host, group_by_fingerprint, _get_dotted, _FINGERPRINT_KEYS,
)"
```
All four required symbols importable.

## Surprise Discoveries

### 1. The 271_652_882_432-byte parametrized test case

PLAN.md described this case as "252.5 GiB → rounds to 253 with banker's rounding". Independent verification with Python shows the actual value is `~252.9965 GiB`, not 252.5 — `271_652_882_432 / (1024**3) == 252.99646186828613`. The result is still 253, but it comes from straight float arithmetic, not from banker's rounding at a true half-GiB boundary. The Pitfall would have been worse if it had been 252.5 exactly: `round(252.5)` returns 252 in Python (round-half-to-even), not 253. The test value the PLAN author picked accidentally avoids the half-to-even hazard. Test docstring updated to reflect the actual semantics.

### 2. `HostMemoryInfo` accepts no-args construction (`HostMemoryInfo()` → `total=0`)

The dataclass default is `total: int = 0`. The `test_node_dict_empty_memory` test exercises this default and confirms the adapter blanks `memory_capacity` rather than emitting `0`. This is the dataclass-default flavor of the universal collection-failure rule: even when the entire `HostMemoryInfo` is constructed without a `total` keyword, the adapter still produces a deterministic `""`.

### 3. `HostSystemInfo.os_release` defaults to `{}`, not `None`

PLAN.md's `<read_first>` says "HostSystemInfo: os_release is `Optional[Dict[str, str]]` defaulting to `{}`". Confirmed: `field(default_factory=dict)` at `cluster_collector.py:143`. The adapter's `if host.system and host.system.os_release:` guard therefore catches BOTH the `system=None` case AND the `os_release={}` case (`{}` is falsy in Python). One guard, two collection-failure paths covered. The Pitfall 9 `.get(k, "") or ""` idiom is the second line of defense for the case where the dict is populated but a specific key is missing or `None`.

### 4. The plan's `'NAME'/'VERSION_ID'` grep gate uses single quotes; module uses double quotes

Acceptance criterion: `grep -c "'NAME'\|'VERSION_ID'"` → expected `>= 2`. Actual: `0` against the production module, because the project's quoting convention is double-quoted strings end-to-end. The double-quoted equivalent (`grep -c "\"NAME\"\|\"VERSION_ID\""`) returns the expected `2` from production code. Decision logged. The PLAN gate's semantic intent (Pitfall 4 explicit key selection) is fully honored.

## Deviations from Plan

### Auto-fixed Issues

**None.** No bugs, no missing critical functionality, no blocking issues encountered during execution.

### Documentation deviations

**1. [Rule 3 - Quoting style]** PLAN acceptance grep `'NAME'\|'VERSION_ID'` does not match the project's double-quote convention.
- **Found during:** Task 2 acceptance-criteria verification.
- **Issue:** PLAN was authored assuming single-quoted Python string literals; the project's actual convention (and the convention I followed for consistency with surrounding code) is double-quoted.
- **Resolution:** Honored the semantic intent of the gate (Pitfall 4 explicit key selection). The double-quoted equivalent grep returns the required count of 2.
- **Files modified:** None — production code stays double-quoted to match project style.

### Structural deviations

**2. [Plan structure]** The PLAN.md describes Task 1 (grouping helpers) and Task 2 (adapter) as two separate TDD cycles ("Step 1 RED, Step 2 GREEN" for each), but the `<success_criteria>` block explicitly mandates exactly TWO total commits for Slice 2 (one RED test commit, one GREEN feat commit).
- **Resolution:** Followed the success_criteria. One RED test commit (`71342e9`) covers all tests for both helpers and the adapter; one GREEN feat commit (`5edcf5f`) ships both production functions. RED gate was nonetheless verifiably observed (`ModuleNotFoundError` before any production code).
- **Why this is safe:** The grouping tests and the adapter tests live in the same file, so RED-then-GREEN-then-RED-then-GREEN would have required an intermediate state where some tests in `tests/unit/test_auto_generator.py` pass and some fail — a worse signal than a single clean RED → GREEN transition.

## Known Stubs

**None introduced.** The only "empty string" values this module emits are deliberate per the universal collection-failure rule (D-2 / Pitfall 9) and per SER-02 (`friendly_description` and `chassis.model_name` blanks waiting for human input). Both are explicitly called out in `<artifacts_this_phase_produces>` and in the module docstring. Plan 02-03 will splice the networking and drive stub LISTS; this plan emits no list stubs.

## Threat Flags

**None new.** The module introduces zero new I/O surface, zero new parsing surface, and zero new trust boundaries. All four symbols are pure functions operating on Python dicts produced from the already-trusted `HostInfo` representation. Carried-forward threat IDs:

- **T-2-04 (Tampering, YAML dump):** owned by Plan 02-04. This module's acceptance gate `grep -c 'yaml\.dump[^_(]\|yaml\.load[^_]'` returns 0; the constraint propagates forward.
- **T-2-05 (DoS, adapter slow path):** dispositioned `accept`. `node_dict_from_host` is O(1) per host; `group_by_fingerprint` is O(N) over hosts; realistic fleets are O(100).
- **T-2-SC (Package install legitimacy):** dispositioned `accept`. Zero new packages — only standard-library `copy` and `typing.Any` imports.

## TDD Gate Compliance

- **RED gate:** `71342e9 test(02-02): add failing adapter + grouping tests (COLL-01/COLL-02/SER-01)`. Confirmed `ModuleNotFoundError: No module named 'mlpstorage_py.system_description.auto_generator'` on `pytest tests/unit/test_auto_generator.py -x` before any production code existed.
- **GREEN gate:** `5edcf5f feat(02-02): node_dict_from_host + group_by_fingerprint (D-4, D-6, COLL-01/02 prep)`. `pytest tests/unit/test_auto_generator.py -v` → 25 passed in 0.12s.
- **REFACTOR gate:** not needed. The module landed at its intended final shape; no internal helper reshuffling required.

Both commits omit any `Co-Authored-By:` AI attribution per `feedback_no_attribution.md` / MEMORY.md.

## What Plans 02-03 / 02-04 Can Now Do

**Plan 02-03 (stub splice + outer dict)** can import the adapter and the grouping helper directly:

```python
from mlpstorage_py.system_description.auto_generator import (
    node_dict_from_host,
    group_by_fingerprint,
    _FINGERPRINT_KEYS,
)

per_host_dicts = [node_dict_from_host(h) for h in hosts]
grouped = group_by_fingerprint(per_host_dicts, _FINGERPRINT_KEYS, "quantity")
# 02-03 then splices networking/drives stubs into each `grouped` entry
# and wraps it in {system_under_test: {clients: ...}}
```

**Plan 02-04 (atomic write)** owns the D-7 sort and the `yaml.safe_dump` call — neither lands here.

## Deferred Items

| Category | Item | Status | Notes |
|---|---|---|---|
| Test environment | `psutil` is not installed in this dev environment; `tests/unit/test_benchmarks_*.py` (and any collector test that transitively imports `mlpstorage_py.utils`) fail at collection time with `ModuleNotFoundError: No module named 'psutil'`. | Deferred — pre-existing, not introduced by this plan. | Confirmed pre-existing by inspecting `pip install -e ".[test]"` requirements vs. the bare interpreter state. Out of scope per the scope-boundary rule ("Only auto-fix issues DIRECTLY caused by the current task's changes"). Resolution path: run `pip install -e ".[test]"` once in the dev environment; not actioned here. |

## Self-Check: PASSED

- `mlpstorage_py/system_description/auto_generator.py` exists with all four required symbols. FOUND.
- `tests/unit/test_auto_generator.py` exists with 25 test cases (including the 5-case parametrized memory-rounding test). FOUND.
- Commit `71342e9` present in `git log --oneline`. FOUND.
- Commit `5edcf5f` present in `git log --oneline`. FOUND.
- Full `pytest tests/unit/test_auto_generator.py -v` → 25 passed. PASSED.
- Adjacent regression suite → 175 passed in 0.65s, zero new failures. PASSED.
