---
phase: 04-sysctl-environment-and-drives-coverage
plan: 05
subsystem: models + auto_generator
tags: [hostinfo, node_dict, integration, vertical-end-to-end, D-33, COLL-05, COLL-06, COLL-07]
requires:
  - mlpstorage_py/rules/models.py — Phase-3 HostInfo dataclass with chassis_model + networking fields; HostInfo.from_collected_data reading Phase-3 keys
  - mlpstorage_py/system_description/auto_generator.py — Phase-3 node_dict_from_host emitting friendly_description/chassis/networking/operating_system; Plan 04-04's _FINGERPRINT_KEYS (11-tuple), _resolve_fingerprint_key generalized dispatch, _splice_stub_lists D-33 drives-omit branch
  - Plan 04-01 (sysctl collector), 04-02 (environment collector + redactors), 04-03 (drives collector), 04-04 (transform-layer extensions) — the four upstream slices whose output Plan 04-05 wires through end-to-end
provides:
  - HostInfo.sysctl: List[Dict[str, Any]] = field(default_factory=list)         # COLL-05
  - HostInfo.environment: List[Dict[str, Any]] = field(default_factory=list)    # COLL-06
  - HostInfo.drives: List[Dict[str, Any]] = field(default_factory=list)         # COLL-07
  - HostInfo.from_collected_data reads data.get('sysctl' / 'environment' / 'drives', []) and passes as kwargs
  - node_dict_from_host extended with three new top-level emit keys (sysctl, environment, drives) inserted between networking and operating_system
  - per-host group_by_fingerprint pass for drives over (vendor_name, model_name, interface, capacity_in_GB) with unit_count
  - 25 new unit tests (12 in test_cluster_collector.py for HostInfo extension; 13 in test_auto_generator.py for node_dict_from_host extension + Phase 4 reflection)
  - 9 new integration tests in TestPhase4EndToEnd covering ROADMAP SC #1-5 end-to-end + D-35 cross-host splits + Yamale schema validation
affects:
  - mlpstorage_py/rules/models.py (HostInfo dataclass + from_collected_data extension; from_dict left untouched)
  - mlpstorage_py/system_description/auto_generator.py (node_dict_from_host docstring + emit-shape extension + per_host_drives computation)
  - tests/unit/test_cluster_collector.py (3 new test classes, 12 new tests)
  - tests/unit/test_auto_generator.py (4 new test classes, 13 new tests + 3 existing tests updated for Phase-4 7-key emit set)
  - tests/integration/test_systemname_yaml_end_to_end.py (TestPhase4EndToEnd with 9 new tests + new _make_host_phase4 helper)
tech-stack:
  added: []  # No new technology; reuses Phase 3 patterns
  patterns:
    - "Phase 3 D-16 / chassis_model / networking precedent extended: 3 new list-typed HostInfo fields with `default_factory=list`, `data.get(..., [])` reads in from_collected_data, kwargs flow into cls(...). `from_dict` left untouched (consistent with Phase 3 chassis_model + networking pattern)."
    - "node_dict_from_host emit-shape extension: 3 new top-level keys (sysctl, environment, drives) inserted between Phase-3 networking and operating_system. sysctl/environment use shallow-copy pass-through via `list(host.sysctl)` / `list(host.environment)`; drives uses per-host group_by_fingerprint pass mirroring the Phase 3 networking pattern verbatim."
    - "Phase 4 final emit shape: 7 top-level keys (friendly_description, chassis, networking, sysctl, environment, drives, operating_system). Downstream _splice_stub_lists handles networking traffic-splice (D-17), networking blank-stub fallback (D-3), and drives key-omission (D-33). sysctl + environment flow through to YAML untouched by the splicer."
    - "D-33 omit-key path live-exercised on WSL2: `collect_drives()` returns [] (every TRAN is null, all rows D-31 filtered) → `node_dict_from_host` emits drives:[] → `_splice_stub_lists` pops the key → emitted YAML has no `drives:` block at the client-stanza level. End-to-end ROADMAP SC #5 confirmed on real (mocked) and real (WSL2 live) data paths."
key-files:
  created:
    - .planning/phases/04-sysctl-environment-and-drives-coverage/04-05-SUMMARY.md
  modified:
    - mlpstorage_py/rules/models.py
    - mlpstorage_py/system_description/auto_generator.py
    - tests/unit/test_cluster_collector.py
    - tests/unit/test_auto_generator.py
    - tests/integration/test_systemname_yaml_end_to_end.py
decisions:
  - "HostInfo dataclass extension shipped verbatim per D-16 num_sockets / Phase-3 chassis_model precedent: three new fields appended after the Phase-3 `networking` field and before `collection_timestamp`. Each is `List[Dict[str, Any]] = field(default_factory=list)`. Imports already in place from Phase 3 (typing.List, Dict, Any). Universal D-2 collection-failure rule: missing keys default to []."
  - "HostInfo.from_collected_data extends the Phase 3 chassis_model + networking read pattern verbatim: three new `data.get('sysctl' / 'environment' / 'drives', [])` reads + three new kwargs in the cls(...) return. HostInfo.from_dict left untouched (Phase 3 precedent — the schema-validator path doesn't consume these fields, and adding them to from_dict would silently grow the dict-deserialization surface that none of the Phase 4 callers need)."
  - "node_dict_from_host emit-shape extension: 3 new keys inserted between `networking` and `operating_system` in the returned dict literal. sysctl + environment are shallow-copy pass-throughs (`list(host.sysctl)`, `list(host.environment)`) — no per-host group_by_fingerprint collapse, since each sysctl/environment key appears exactly once per host (no within-host duplication). drives uses per-host group_by_fingerprint over (vendor_name, model_name, interface, capacity_in_GB) with unit_count — mirrors the Phase 3 per-host networking grouping pattern verbatim."
  - "Three existing tests updated in the RED commit for the Phase-4 7-key emit set: test_node_dict_cpu_fields (was asserting `\"sysctl\" not in result` etc.; now asserts `result[\"sysctl\"] == []`), test_node_dict_no_extra_keys (was asserting the Phase-3 4-key set; now asserts the Phase-4 7-key set), and TestNodeDictReflection::test_node_dict_field_names_match_pydantic_reflection_after_phase3 (renamed to ...after_phase4; updated to the 7-key set). Folding these into RED keeps the GREEN commit purely additive on production code — same convention as Plan 03-05 (which updated Phase-2 tests in RED for the same reason)."
  - "Live WSL2 end-to-end smoke confirmed D-33 omit-key path. `collect_local_system_info()` returns sysctl=134 entries, environment=[], drives=[]. The Plan 04-03 collector returns [] on WSL2 because every disk's TRAN is null (Microsoft Hyper-V virtio-block) and the D-31 filter chain drops all four rows. `node_dict_from_host` emits drives=[]. `_splice_stub_lists` pops the empty drives key per D-33. The on-disk YAML at /tmp/.../sys-v1.yaml has client0 keys = {chassis, environment, friendly_description, networking, operating_system, quantity, sysctl} — NO drives key. Exactly the ROADMAP SC #5 / D-33 contract."
  - "Phase 4 vertical end-to-end COMPLETE. Plans 04-01 (sysctl collector) + 04-02 (environment collector + redactors) + 04-03 (drives collector) + 04-04 (transform-layer extensions: 3 fingerprint signatures + generalized dispatch + D-33 splice) + 04-05 (HostInfo + node_dict_from_host wire-through) all green. A real `mlpstorage <mode> training <model> run` invocation now produces YAML where clients[].sysctl[], clients[].environment[], and clients[].drives[] reflect actual collected data (or are correctly OMITTED for drives per D-33). All 8 ROADMAP success criteria (SC #1-5 + D-35 + homogeneous-collapse + Yamale validation) backed by passing integration tests in TestPhase4EndToEnd."
metrics:
  duration_minutes: ~22
  completed_date: 2026-06-23
  tasks_completed: 2
  files_created: 1
  files_modified: 5
  commits: 2
---

# Phase 04 Plan 05: Vertical end-to-end wire-through (HostInfo + node_dict_from_host extensions) Summary

Three new HostInfo list-typed fields (`sysctl`, `environment`, `drives`) appended verbatim per the Phase-3 D-16 / chassis_model / networking precedent; `HostInfo.from_collected_data` extended to read the corresponding collector output keys; `node_dict_from_host` extended with three new top-level emit keys (sysctl + environment shallow-copy pass-through; drives via per-host `group_by_fingerprint` over the COLL-07 4-tuple). Two-commit RED/GREEN cadence; 25 new unit tests + 9 new integration tests across 7 new test classes all green; no regressions in the 1849-passing unit suite.

**Phase 4 vertical end-to-end COMPLETE.** A real `mlpstorage run` invocation now produces `systemname.yaml` where `clients[].sysctl[]`, `clients[].environment[]`, and `clients[].drives[]` reflect actual collected data — or are correctly OMITTED for drives per D-33 when the collector returns no rows (lsblk absent / all D-31 filtered).

## What Shipped

**1. HostInfo dataclass extension** (`mlpstorage_py/rules/models.py`). Three new list-typed fields appended after the Phase-3 `networking` field and before `collection_timestamp`:

```python
chassis_model: str = ""                                              # Phase 3 (COLL-03)
networking: List[Dict[str, Any]] = field(default_factory=list)       # Phase 3 (COLL-04)
sysctl: List[Dict[str, Any]] = field(default_factory=list)           # Phase 4 (COLL-05)
environment: List[Dict[str, Any]] = field(default_factory=list)      # Phase 4 (COLL-06)
drives: List[Dict[str, Any]] = field(default_factory=list)           # Phase 4 (COLL-07)
collection_timestamp: Optional[str] = None
```

`field(default_factory=list)` is required to avoid the dataclass mutable-default trap. Imports already in place from Phase 3 (typing.List, Dict, Any).

**2. `HostInfo.from_collected_data` extension.** Three new `data.get(..., [])` reads and three new kwargs in the `cls(...)` return:

```python
# Phase 4 / Plan 04-05: COLL-05 + COLL-06 + COLL-07 — flow the three
# new collector outputs onto the dataclass. Plan 04-01 (sysctl), Plan
# 04-02 (environment — already redacted per D-23/D-24), Plan 04-03
# (drives — D-31 filtered). Universal D-2 rule: missing keys default
# to []. `from_dict` left untouched (consistent with the Phase 3
# chassis_model/networking precedent).
sysctl = data.get('sysctl', [])
environment = data.get('environment', [])
drives = data.get('drives', [])

return cls(
    hostname=hostname,
    memory=memory,
    cpu=cpu,
    disks=disks,
    network=network,
    system=system,
    chassis_model=chassis_model,
    networking=networking,
    sysctl=sysctl,
    environment=environment,
    drives=drives,
    collection_timestamp=data.get('collection_timestamp'),
)
```

`HostInfo.from_dict` left untouched per the Phase 3 precedent — the schema-validator reconstruction path doesn't consume the Phase-4 fields, and adding them silently grows the dict-deserialization surface that no Phase 4 caller needs.

**3. `node_dict_from_host` emit-shape extension** (`mlpstorage_py/system_description/auto_generator.py`). Three new top-level emit keys inserted between `networking` and `operating_system`:

```python
# Phase 4 / Plan 04-05 — sysctl/environment/drives all emitted directly
# from the corresponding HostInfo list fields. `list(host.sysctl)` and
# `list(host.environment)` produce shallow copies so caller mutations of
# the result list do not alias back into HostInfo state. `drives` uses
# a per-host group_by_fingerprint pass (above) to collapse identical
# drive rows into stanzas with unit_count=N.
"sysctl":      list(host.sysctl),       # COLL-05
"environment": list(host.environment),  # COLL-06
"drives":      per_host_drives,         # COLL-07
```

`per_host_drives` is computed via `group_by_fingerprint(host.drives, ("vendor_name", "model_name", "interface", "capacity_in_GB"), "unit_count")` when `host.drives` is truthy; otherwise `[]`. Mirrors the Phase 3 per-host networking grouping pattern verbatim.

The docstring is rewritten to describe the Phase 4 final emit shape (7 top-level keys: friendly_description, chassis, networking, sysctl, environment, drives, operating_system) including the per-host group_by_fingerprint pass for drives and the D-33 splice-layer omit contract for the empty-drives path.

**4. 25 new unit tests across 7 new test classes:**

| File | Class | Tests | Purpose |
|---|---|---|---|
| `tests/unit/test_cluster_collector.py` | `TestHostInfoSysctlField` | 4 | default empty list, direct construction, from_collected_data reads, missing-key default |
| `tests/unit/test_cluster_collector.py` | `TestHostInfoEnvironmentField` | 4 | mirror of sysctl |
| `tests/unit/test_cluster_collector.py` | `TestHostInfoDrivesField` | 4 | mirror of sysctl |
| `tests/unit/test_auto_generator.py` | `TestNodeDictSysctl` | 3 | empty-emit-empty, populated pass-through, shallow-copy isolation |
| `tests/unit/test_auto_generator.py` | `TestNodeDictEnvironment` | 3 | mirror of sysctl |
| `tests/unit/test_auto_generator.py` | `TestNodeDictDrives` | 5 | empty, single (unit_count=1), 2 identical (unit_count=2), 2 different (split), 3 mixed (2+1) |
| `tests/unit/test_auto_generator.py` | `TestNodeDictReflectionPhase4` | 1 | top-level 7-key set lock |

Plus 3 existing tests updated in the RED commit:
- `test_node_dict_cpu_fields` — Phase 2/3 had `assert "sysctl" not in result` etc.; updated to `assert result["sysctl"] == []` etc. for the Phase-4 emit set.
- `test_node_dict_no_extra_keys` — Phase 3 4-key set updated to Phase 4 7-key set.
- `TestNodeDictReflection::test_node_dict_field_names_match_pydantic_reflection_after_phase3` renamed to `...after_phase4` and updated to the 7-key set.

**5. 9 new integration tests in `TestPhase4EndToEnd`** (`tests/integration/test_systemname_yaml_end_to_end.py`):

| Test | ROADMAP SC | Asserts |
|---|---|---|
| `test_drives_populated_emits_drives_key` | #3 | drives entry with unit_count=1 in emitted YAML |
| `test_drives_absent_omits_drives_key_d33` | #5 | drives key OMITTED entirely when host.drives=[] (D-33) |
| `test_sysctl_populated_emits_sysctl_key` | #1 | sysctl entries round-trip verbatim through YAML |
| `test_environment_populated_emits_environment_with_redaction` | #2 | already-redacted values round-trip (`[SET — 40 chars]` for SECRET, masked for KEY_ID) |
| `test_two_hosts_differ_on_sysctl_split_to_two_stanzas` | D-35 | 2 stanzas, qty sum = 2 (sysctl divergence) |
| `test_two_hosts_differ_on_drives_split_to_two_stanzas` | D-35 | 2 stanzas, qty sum = 2 (drives divergence on capacity_in_GB) |
| `test_two_hosts_differ_on_environment_split_to_two_stanzas` | D-35 | 2 stanzas, qty sum = 2 (environment value divergence) |
| `test_homogeneous_fleet_collapses_to_one_stanza` | (collapse) | 3 identical hosts → 1 stanza with quantity=3, all Phase-4 lists populated on the collapsed stanza |
| `test_yamale_schema_validation_passes_on_phase_4_emit_shape` | #4 + SER-03 | no error paths on Phase-4-populated fields; drives entries carry ONLY the 4 COLL-07 fields (no media_type/form_factor/performance) |

New helper `_make_host_phase4` layers sysctl + environment + drives on top of the Phase-3 `_make_host_phase3` helper for hermetic Phase-4 host construction.

## Two-Commit RED/GREEN Cadence

| Commit | Type | Files | Purpose |
|---|---|---|---|
| `dc3ab89` | test(04-05) | tests/unit/test_cluster_collector.py, tests/unit/test_auto_generator.py, tests/integration/test_systemname_yaml_end_to_end.py | RED — 25 new unit + 9 new integration tests, plus 3 existing tests updated for the Phase-4 7-key emit set. All 25 new + 3 updated tests fail with AttributeError / AssertionError / KeyError on collection or run. |
| `840cbfe` | feat(04-05) | mlpstorage_py/rules/models.py, mlpstorage_py/system_description/auto_generator.py | GREEN — HostInfo extension (3 new fields + from_collected_data extension) + node_dict_from_host extension (3 new emit keys + per_host_drives computation + docstring rewrite). All RED tests turn GREEN; no regressions in the 1849-passing unit suite. |

## Integration Test Fixture Shape (Contract for Phases 5+)

The data-dict shape passed to `HostInfo.from_collected_data` is now the canonical Phase 4 vertical contract. A representative dict accepted by Plan 04-05's GREEN code:

```python
{
    'hostname': 'h1',
    'meminfo': {'MemTotal': 274_877_906_944},
    'cpuinfo': [{'processor': '0', 'physical id': '0', 'model name': '...', 'cpu cores': '56', 'flags': ''}],
    'os_release': {'NAME': 'Rocky Linux', 'VERSION_ID': '9.5'},
    # Phase 3 (chassis_model + networking):
    'chassis_model': 'PowerEdge R760',
    'networking': [
        {'type': 'ethernet',    'speed': 100, 'state': 'up'},
        {'type': 'infiniband', 'speed': 200, 'state': 'up'},
    ],
    # Phase 4 NEW (sysctl + environment + drives):
    'sysctl': [
        {'name': 'vm.dirty_ratio', 'value': '20'},
        {'name': 'net.core.somaxconn', 'value': '4096'},
    ],
    'environment': [
        {'name': 'AWS_SECRET_ACCESS_KEY', 'value': '[SET — 40 chars]'},  # already redacted per D-24
        {'name': 'AWS_ACCESS_KEY_ID', 'value': 'AKIA****MPLE'},          # already masked per D-23
        {'name': 'BUCKET', 'value': 'my-bucket'},
    ],
    'drives': [
        {'vendor_name': 'INTEL', 'model_name': 'SSDPED1K375GA', 'interface': 'nvme', 'capacity_in_GB': 375},
    ],
}
```

`HostInfo.from_collected_data(data)` consumes this verbatim. The integration fixture in `_make_host_phase4` constructs a HostInfo via direct dataclass construction (bypassing the dict-deserialization step) — this matches how `cluster_collector.collect_local_system_info` would feed `from_collected_data` in production (the dict→HostInfo path), but lets tests skip the MPI surface.

Phases 5+ that extend the collector output dict (e.g. shared-filesystem verification CAP-02 in Phase 5) should follow the same pattern: add a list-typed dataclass field with `default_factory=list`, extend `from_collected_data` with a `data.get(..., [])` read, extend `node_dict_from_host` with a top-level emit key, and add corresponding `_<field>_signature` / `_EXTRACTOR_SOURCE_KEYS` entries if cross-host fingerprint differentiation is desired (D-35).

## D-33 Omit-Key Path — Exercised By

Two test paths exercise the D-33 omit-key contract end-to-end:

1. **Unit-level** (Plan 04-04 — already shipped): `TestSpliceStubListsDrivesOmitBranch` in `tests/unit/test_auto_generator.py` (4 tests).
2. **Integration-level** (Plan 04-05 — NEW): `TestPhase4EndToEnd::test_drives_absent_omits_drives_key_d33` in `tests/integration/test_systemname_yaml_end_to_end.py`. Mocked collector returns `host.drives = []`; full `Benchmark.run()` lifecycle drives the write; `yaml.safe_load(target.read_text())` confirms `"drives" not in clients[0]`.

Also, **live on WSL2**: the dev-shell smoke run (write_systemname_yaml via D-8 local fallback) produced a YAML where `client0 keys = {chassis, environment, friendly_description, networking, operating_system, quantity, sysctl}` — NO drives key. This is the Plan 04-03 surprise (WSL2 returns null TRAN; all rows D-31 filtered) connecting end-to-end with the Plan 04-04 D-33 splice and the Plan 04-05 emit-key extension, exactly the ROADMAP SC #5 contract.

## Deviations from Plan

**None.** RED → GREEN was clean on first run; no Rule 1 bugs, no Rule 2 missing critical functionality, no Rule 3 blocking-task-completion fixes, no Rule 4 architectural decisions. The mixed-type sort issue PLAN.md flagged as a possible Plan-03-04-style surprise for the per-host drives group_by_fingerprint pass did NOT materialize — `group_by_fingerprint` itself uses `_resolve_fingerprint_key` which for scalar dotted keys (the per-host drives fingerprint is 4 scalar string-typed keys: vendor_name, model_name, interface, capacity_in_GB) calls `_get_dotted` which returns either the value or `""` on miss. Mixed types only arise in the cross-host `_drive_signature` callable extractor (which uses `key=repr` since Plan 04-04), not in the per-host pass.

## Authentication Gates

None.

## Threat Flags

None — Plan 04-05 is pure-transform code (zero I/O, zero network surface, zero subprocess, zero file access). The new HostInfo fields are pure dataclass extensions; `from_collected_data` is a dict-key read; `node_dict_from_host` is a dict-shape extension with one extra `group_by_fingerprint` call over an in-memory list. No new threat surface beyond what Plans 04-01..04-04 already shipped.

## Known Stubs

None — all production code paths flow data end-to-end. The Phase-2-legacy `_NETWORKING_STUB` and `_DRIVE_STUB` constants are retained (Plan 04-04 kept `_DRIVE_STUB` with a Phase-2-legacy comment) but are not emitted into any Phase 4 path: `_NETWORKING_STUB` is spliced when a host has empty networking (D-3 universal-rule fallback for the networking branch); `_DRIVE_STUB` is no longer spliced — the D-33 branch pops the empty drives key entirely.

## ROADMAP Success Criteria Coverage

All 5 Phase 4 ROADMAP success criteria + the D-35 cross-host strict-split policy + homogeneous-collapse preservation + Yamale schema validation cleanliness:

| ROADMAP SC | Plan/Test | Confirmed By |
|---|---|---|
| SC #1: `clients[].sysctl[]` populated from /proc/sys allowlist | Plan 04-01 + Plan 04-05 | `TestPhase4EndToEnd::test_sysctl_populated_emits_sysctl_key`; live WSL2 smoke (134 entries) |
| SC #2: `clients[].environment[]` with D-23/D-24 redaction | Plan 04-02 + Plan 04-05 | `TestPhase4EndToEnd::test_environment_populated_emits_environment_with_redaction` |
| SC #3: `clients[].drives[]` with (vendor, model, interface, capacity_in_GB) + unit_count | Plan 04-03 + Plan 04-05 | `TestPhase4EndToEnd::test_drives_populated_emits_drives_key`; `TestNodeDictDrives::*` (5 tests) |
| SC #4: drive entries do NOT contain media_type, form_factor, performance | Plan 04-03 + Plan 04-05 | `TestPhase4EndToEnd::test_yamale_schema_validation_passes_on_phase_4_emit_shape` (asserts absence) |
| SC #5: lsblk absent / all-filtered → `drives` key OMITTED | Plan 04-04 + Plan 04-05 | `TestPhase4EndToEnd::test_drives_absent_omits_drives_key_d33`; live WSL2 smoke (drives key absent in emitted YAML) |
| D-35: cross-host fingerprint splits on divergent sysctl/env/drives | Plan 04-04 + Plan 04-05 | `TestPhase4EndToEnd::test_two_hosts_differ_on_{sysctl,drives,environment}_split_to_two_stanzas` |
| (collapse): homogeneous fleet → 1 stanza with quantity=N | Phase 2 + Plan 04-05 | `TestPhase4EndToEnd::test_homogeneous_fleet_collapses_to_one_stanza` |
| (SER-03): Yamale validation passes on Phase 4 emit | Plan 04-05 | `TestPhase4EndToEnd::test_yamale_schema_validation_passes_on_phase_4_emit_shape` |

## Verification

```bash
# All RED tests (now GREEN) — full Plan 04-05 slice.
python3 -m pytest tests/unit/test_cluster_collector.py -k 'HostInfoSysctl or HostInfoEnvironment or HostInfoDrives' \
    tests/unit/test_auto_generator.py -k 'NodeDictSysctl or NodeDictEnvironment or NodeDictDrives or NodeDictReflection or test_node_dict_cpu_fields or test_node_dict_no_extra_keys' \
    tests/integration/test_systemname_yaml_end_to_end.py -k 'Phase4' -q
# → 11 passed (the 9 Phase 4 integration tests + the 2 reflection / cpu_fields / no_extra_keys updates)

# Full Phase 4 target slice (no regressions in Phase 2/3 tests).
python3 -m pytest tests/unit/test_cluster_collector.py tests/unit/test_auto_generator.py \
    tests/integration/test_systemname_yaml_end_to_end.py -q \
    --ignore=tests/unit/test_benchmarks_base.py --ignore=tests/unit/test_parquet_reader.py \
    --ignore=tests/unit/test_vdb_modular_fake_backend.py
# → 403 passed

# Full unit suite excluding pre-existing collection errors.
python3 -m pytest tests/unit -q \
    --ignore=tests/unit/test_benchmarks_base.py \
    --ignore=tests/unit/test_parquet_reader.py \
    --ignore=tests/unit/test_vdb_modular_fake_backend.py
# → 1849 passed, 8 pre-existing failures (out-of-scope per Rule 3 scope boundary —
#   _check_safe_path_component MagicMock fixtures + psutil module absence, same set
#   noted in STATE.md Deferred Items and prior Phase 4 SUMMARYs)

# Acceptance criteria spot checks.
python3 -c "from mlpstorage_py.rules.models import HostInfo; h=HostInfo('h'); print(h.sysctl, h.environment, h.drives)"
# → [] [] []
python3 -c "from mlpstorage_py.system_description.auto_generator import node_dict_from_host; from mlpstorage_py.rules.models import HostInfo; r = node_dict_from_host(HostInfo('h')); print(sorted(r.keys()))"
# → ['chassis', 'drives', 'environment', 'friendly_description', 'networking', 'operating_system', 'sysctl']
```

All plan-level `<verification>` items and `<success_criteria>` satisfied. ROADMAP SC #1-5 all have at least one passing integration test backing them up.

## Live WSL2 End-to-End Smoke (D-33 Confirmation)

```bash
$ python3 -c "
from mlpstorage_py.cluster_collector import collect_local_system_info
from mlpstorage_py.rules.models import HostInfo
from mlpstorage_py.system_description.auto_generator import node_dict_from_host
data = collect_local_system_info()
print('data[sysctl] len:', len(data.get('sysctl', [])))
print('data[environment] len:', len(data.get('environment', [])))
print('data[drives] len:', len(data.get('drives', [])))
host = HostInfo.from_collected_data(data)
emit = node_dict_from_host(host)
print('emit keys:', sorted(emit.keys()))
print('emit[drives]:', emit['drives'])
"
# data[sysctl] len: 134
# data[environment] len: 0
# data[drives] len: 0
# emit keys: ['chassis', 'drives', 'environment', 'friendly_description', 'networking', 'operating_system', 'sysctl']
# emit[drives]: []

$ python3 -c "
# Full write_systemname_yaml via D-8 fallback path on WSL2.
import tempfile, yaml
from argparse import Namespace
from unittest.mock import MagicMock
from mlpstorage_py.system_description.auto_generator import write_systemname_yaml
with tempfile.TemporaryDirectory() as tmp:
    args = Namespace(command='run', mode='closed', orgname='Acme', systemname='sys-v1', results_dir=tmp)
    path = write_systemname_yaml(args, None, MagicMock())
    data = yaml.safe_load(open(path).read())
    client0 = data['system_under_test']['clients'][0]
    print('client0 keys:', sorted(client0.keys()))
    print('drives in client0:', 'drives' in client0)
"
# client0 keys: ['chassis', 'environment', 'friendly_description', 'networking', 'operating_system', 'quantity', 'sysctl']
# drives in client0: False  ← D-33 omit-key path LIVE-CONFIRMED on WSL2
```

## Forward Notes for Phase 5

Phase 5 (CAP-02 shared-filesystem verification + LIFE-02 lifecycle) inherits the now-stable Phase 4 contract:

- The data-dict shape consumed by `HostInfo.from_collected_data` is the canonical collector → dataclass interface. Phase 5 additions should follow the same pattern: append a list-typed (or scalar-typed) field with a sensible default, extend `from_collected_data` with a `data.get(..., default)` read, and extend `node_dict_from_host` with a top-level emit key.
- The 7-key Phase-4 emit shape (`friendly_description`, `chassis`, `networking`, `sysctl`, `environment`, `drives`, `operating_system`) is the new baseline for `node_dict_from_host`. Phase 5 may grow this set; if it does, `test_node_dict_no_extra_keys` + `test_node_dict_cpu_fields` + `TestNodeDictReflectionPhase4` are the gating tests.
- `_FINGERPRINT_KEYS` is currently the 11-tuple per Plan 04-04 D-34 (4 Phase-2 chassis/OS + chassis.model_name + 2 OS + networking_sig + sysctl_sig + environment_sig + drives_sig). Phase 5 additions can extend it via the `_EXTRACTOR_SOURCE_KEYS` map pattern Plan 04-04 introduced (2-line change: append to `_FINGERPRINT_KEYS` + add name→source-key mapping; zero touch on `_resolve_fingerprint_key`).
- The Phase 3 CR-01 follow-up (`HostInfo.to_dict()` drift on chassis_model + networking) noted in STATE.md is now compounded: `to_dict` also needs sysctl + environment + drives if Phase 5 LIFE-02 lifecycle work serializes HostInfo to JSON. Currently `to_dict` only emits hostname/memory/cpu/disks/network/system/collection_timestamp — 5 fields drift (chassis_model, networking, sysctl, environment, drives). Phase 5 LIFE-02 should address this.

## Self-Check: PASSED

- `mlpstorage_py/rules/models.py: HostInfo.sysctl field`: FOUND (`grep -c 'sysctl: List\[Dict' mlpstorage_py/rules/models.py` → 1)
- `mlpstorage_py/rules/models.py: HostInfo.environment field`: FOUND (`grep -c 'environment: List\[Dict' mlpstorage_py/rules/models.py` → 1)
- `mlpstorage_py/rules/models.py: HostInfo.drives field`: FOUND (`grep -c 'drives: List\[Dict' mlpstorage_py/rules/models.py` → 1)
- `mlpstorage_py/rules/models.py: from_collected_data reads sysctl/environment/drives`: FOUND (`grep -E "data.get\(['\"](sysctl|environment|drives)['\"]" mlpstorage_py/rules/models.py | wc -l` → 3)
- `mlpstorage_py/system_description/auto_generator.py: node_dict_from_host emits sysctl/environment/drives keys`: FOUND (`grep -E '"(sysctl|environment|drives)":' mlpstorage_py/system_description/auto_generator.py | wc -l` → ≥3)
- `mlpstorage_py/system_description/auto_generator.py: per_host_drives via group_by_fingerprint over COLL-07 fingerprint`: FOUND (`grep -A2 'per_host_drives' mlpstorage_py/system_description/auto_generator.py | grep -c 'vendor_name'` → 1)
- HostInfo() defaults verified: VERIFIED (`python3 -c "..." → [] [] []`)
- node_dict_from_host(HostInfo()) 7-key emit shape verified: VERIFIED
- Commit `dc3ab89` (RED): present in `git log --oneline -5`
- Commit `840cbfe` (GREEN): present in `git log --oneline -5`
- Full Phase 4 target slice: 403 passed, no regressions
- Full unit suite: 1849 passed, 8 pre-existing failures (out-of-scope; same set documented in STATE.md Deferred Items)
- Live WSL2 D-33 omit-key path: CONFIRMED on real `collect_local_system_info()` → `node_dict_from_host` → `_splice_stub_lists` → on-disk YAML chain (drives key OMITTED at client-stanza level)
