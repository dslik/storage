---
phase: 02-first-run-write-of-partial-systemname-yaml
plan: 01
subsystem: rules/models
tags:
  - python
  - data-model
  - tdd
dependency_graph:
  requires:
    - mlpstorage_py/rules/models.py:HostCPUInfo (existing dataclass)
    - mlpstorage_py/cluster_collector.py:summarize_cpuinfo (already returns 'num_sockets' key at line 826)
  provides:
    - mlpstorage_py.rules.models.HostCPUInfo.num_sockets (int field, default 0)
    - HostInfo.from_collected_data wiring: cpu_summary['num_sockets'] → HostCPUInfo.num_sockets
    - HostCPUInfo.from_dict wiring: data['num_sockets'] → HostCPUInfo.num_sockets
  affects:
    - 02-02 (node_dict_from_host adapter — will read host.cpu.num_sockets for chassis.cpu_qty)
    - 02-03 (stub-splice / outer-dict builder — same)
tech_stack:
  added: []
  patterns:
    - "Additive dataclass field with default — preserves all existing keyword-construction call sites"
    - "Factory keyword-pass-through via .get(key, default) — never raises on missing key"
key_files:
  created: []
  modified:
    - mlpstorage_py/rules/models.py
    - tests/unit/test_cluster_collector.py
decisions:
  - "D-16 implementation: num_sockets sourced from summarize_cpuinfo['num_sockets'] (which equals len(unique physical ids) else 1). Default on the dataclass is 0; default on the from_dict / from_collected_data .get() calls is 0. The single-socket fallback (no 'physical id' lines in cpuinfo) lands as num_sockets == 1 via summarize_cpuinfo's existing `else 1` branch, NOT via the .get default."
  - "Test placement: new TestHostCPUInfoNumSockets class appended to tests/unit/test_cluster_collector.py (the file's TestTimeSeriesSampleDataclass / TestTimeSeriesDataDataclass classes already test rules/models dataclasses from here, so this is the canonical location — no new file needed)."
  - "RED/GREEN gate observed: separate commits 4be6fbb (test, RED — AttributeError) and 2ede1d3 (feat, GREEN — 3-line additive diff)."
metrics:
  duration_min: ~8
  tasks_total: 1
  tasks_completed: 1
  files_changed: 2
  commits: 2
  completed_date: 2026-06-19
---

# Phase 02 Plan 01: HostCPUInfo.num_sockets Data-Model Extension Summary

`HostCPUInfo` gains an additive `num_sockets: int = 0` field and `HostInfo.from_collected_data` plus `HostCPUInfo.from_dict` now propagate it from the existing `summarize_cpuinfo()['num_sockets']` value — unblocking plans 02-02 / 02-03's `chassis.cpu_qty` wiring per D-16.

## What Was Built

Slice 1 of Phase 02. A surgical three-line additive change to `mlpstorage_py/rules/models.py` plus a six-case regression test in `tests/unit/test_cluster_collector.py`. The data-model surface now exposes `host.cpu.num_sockets` consistently from both code paths that build a `HostCPUInfo` (MPI-collected raw cpuinfo via `from_collected_data`, and JSON-serialized data via `from_dict`).

The existing `summarize_cpuinfo()` already produced `num_sockets` at `cluster_collector.py:826` — Pitfall 5 from the research doc was simply that `from_collected_data` had been dropping it on the floor. This plan plugs that single leak.

## Edited Locations (final line numbers)

| File | Line | Change |
| --- | --- | --- |
| `mlpstorage_py/rules/models.py` | 156 | `num_sockets: int = 0` appended to `HostCPUInfo` dataclass fields (after `architecture`) |
| `mlpstorage_py/rules/models.py` | 166 | `num_sockets=data.get('num_sockets', 0),` appended to `HostCPUInfo.from_dict` return |
| `mlpstorage_py/rules/models.py` | 220 | `num_sockets=cpu_summary.get('num_sockets', 0),` appended to `HostCPUInfo(...)` keyword construction inside `HostInfo.from_collected_data` |
| `tests/unit/test_cluster_collector.py` | 1359 | New `TestHostCPUInfoNumSockets::test_host_cpu_info_carries_num_sockets` (six in-test cases) |

## Tasks

| Task | Name | Commit |
| --- | --- | --- |
| 1 (RED) | failing num_sockets assertion (D-16) | `4be6fbb` |
| 1 (GREEN) | HostCPUInfo.num_sockets field + wiring (COLL-01 prep) | `2ede1d3` |

## Test Count Delta

| Metric | Before | After | Δ |
| --- | --- | --- | --- |
| `tests/unit/test_cluster_collector.py` tests | 104 | 105 | +1 |
| Related-suite total (test_cluster_collector + test_rules_dataclasses + mlpstorage_py/tests/test_rules) | 174 | 175 | +1 |

Full `pytest tests/unit/test_cluster_collector.py -x -v` → 105 passed in 2.95s. Broader related suite → 175 passed in 3.05s.

## Verification (acceptance criteria from PLAN)

- `pytest tests/unit/test_cluster_collector.py::TestHostCPUInfoNumSockets::test_host_cpu_info_carries_num_sockets -x` → exit 0.
- `pytest tests/unit/test_cluster_collector.py -x` → exit 0 (zero regressions).
- `grep -c 'num_sockets' mlpstorage_py/rules/models.py` → 3 (field + from_dict + from_collected_data).
- `python -c "from mlpstorage_py.rules.models import HostCPUInfo; c = HostCPUInfo(); assert c.num_sockets == 0; c2 = HostCPUInfo(num_sockets=2); assert c2.num_sockets == 2; print('ok')"` → `ok`.
- Positional-call audit (`grep -rn 'HostCPUInfo(' mlpstorage_py tests | grep -E 'HostCPUInfo\([^a-zA-Z_]'` filtered) → 0 lines. Assumption A4 confirmed: every existing caller already uses keyword construction.
- `git diff mlpstorage_py/rules/models.py | grep -c '^+'` → 4 (3 added lines + 1 diff header). Bounded additive footprint as planned.

## Assumption A4 — HostCPUInfo Callsite Audit

Per the PLAN's pre-commit drift check, all production and test callers of `HostCPUInfo(...)` were enumerated:

```
mlpstorage_py/tests/test_rules.py:           8 sites — all kw construction (cpu=HostCPUInfo(num_cores=8))
tests/unit/test_rules_dataclasses.py:        3 sites — all kw construction (or no-arg)
mlpstorage_py/rules_legacy.py:               2 sites — kw construction
mlpstorage_py/rules/models.py:               2 sites — kw construction (from_dict + from_collected_data, both edited here)
tests/unit/test_cluster_collector.py:        2 sites — kw construction (this plan's new test)
```

Zero positional callers. The new `num_sockets: int = 0` default means every existing site continues to compile and run unchanged. No surprise discoveries.

## Deviations from Plan

None — plan executed exactly as written. Three lines added to `models.py` at exactly the three locations PATTERNS.md called out; one test class appended at the file's existing data-class-test idiom; both RED and GREEN commits emitted without any Co-Authored-By trailer per project rule.

## Known Stubs

None introduced. This plan is a pure data-model widening — no UI surfaces, no schema bindings, no placeholder values.

## Threat Flags

None. This plan touches only a dataclass field and two factory keyword pass-throughs; no new I/O surface, no new parsing, no new trust boundary. T-2-05 (DoS on huge cpuinfo) and T-2-SC (package install) remain dispositioned `accept` per PLAN's threat register.

## TDD Gate Compliance

- RED gate: `4be6fbb test(02-01): add failing num_sockets assertion (D-16)` — confirmed AttributeError before any production code.
- GREEN gate: `2ede1d3 feat(02-01): HostCPUInfo.num_sockets field + wiring (COLL-01 prep)` — three-line additive diff, all 105 tests pass.
- REFACTOR gate: not needed (additive change is already minimal).

Both commits omit `Co-Authored-By:` AI attribution per `feedback_no_attribution.md` / MEMORY.md.

## What Plans 02-02 / 02-03 Can Now Do

Both downstream plans can reference `host.cpu.num_sockets` directly without further models changes:

```python
node_dict["chassis"]["cpu_qty"] = host.cpu.num_sockets  # 02-02 adapter
```

The data-model prerequisite for COLL-01's `chassis.cpu_qty` autopopulation is now satisfied.

## Self-Check: PASSED

- `mlpstorage_py/rules/models.py` exists with `num_sockets` field + 2 wiring sites (grep count == 3). FOUND.
- `tests/unit/test_cluster_collector.py` exists with new `TestHostCPUInfoNumSockets` class. FOUND.
- Commit `4be6fbb` present in `git log --oneline`. FOUND.
- Commit `2ede1d3` present in `git log --oneline`. FOUND.
- Full `pytest tests/unit/test_cluster_collector.py -x` → 105 passed. PASSED.
