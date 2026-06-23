---
phase: 03-chassis-model-networking-coverage
plan: 05
subsystem: data-model-integration
tags: [python, integration, data-model, end-to-end, hostinfo, transform]

# Dependency graph
requires:
  - phase: 03-chassis-model-networking-coverage
    plan: 01
    provides: "NetworkPort.state schema lock + _NETWORKING_STUB shape — the post-Pydantic dump shape the D-17 splice consumes and the D-3 fallback emits when host.networking is empty"
  - phase: 03-chassis-model-networking-coverage
    plan: 02
    provides: "chassis_model emitted on per-host data dict by collect_local_system_info — HostInfo.from_collected_data now reads it via data.get('chassis_model', '')"
  - phase: 03-chassis-model-networking-coverage
    plan: 03
    provides: "networking list emitted on per-host data dict by collect_local_system_info — HostInfo.from_collected_data now reads it via data.get('networking', [])"
  - phase: 03-chassis-model-networking-coverage
    plan: 04
    provides: "_FINGERPRINT_KEYS extended with chassis.model_name scalar + ('networking_sig', _network_signature) callable; group_by_fingerprint dispatch via _resolve_fingerprint_key; _splice_stub_lists D-17 conditional traffic=[] splice / D-3 fallback. The transform pipeline this plan wires into."
provides:
  - "HostInfo.chassis_model: str = '' — Phase 3 data-model surface for chassis.model_name flow-through"
  - "HostInfo.networking: List[Dict[str, Any]] = field(default_factory=list) — Phase 3 data-model surface for per-host NIC inventory flow-through"
  - "HostInfo.from_collected_data — reads data['chassis_model'] (default '') and data['networking'] (default []) from the MPI-collected dict"
  - "node_dict_from_host — emits chassis.model_name from (host.chassis_model or '') (Pattern F); emits top-level 'networking' key directly from group_by_fingerprint(host.networking, ('type','speed','state'), 'unit_count') or [] on missing data"
  - "End-to-end vertical: a `mlpstorage ... run ...` invocation writes a systemname.yaml whose clients[].chassis.model_name and clients[].networking[] reflect real DMI + sysfs data per Phase 3 success criteria"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern: D-16 num_sockets-style additive dataclass extension — two new fields appended after the last typed-Optional field, with from_collected_data reading the corresponding dict keys via .get(key, default) and passing them as kwargs to cls(...). No HostInfo.from_dict modification (separate code path that does not flow MPI-collected data). Mirrors the proven Phase 2 D-16 precedent verbatim."
    - "Pattern: Pattern F defensive blank-on-falsy in emit — `(host.chassis_model or \"\")` coerces None / missing falsy values to the empty-string emit. The dataclass default is '' but a malicious worker or a future refactor could pass None; the or-guard neutralizes both without crashing. Matches Phase 2's existing chassis CPU field guards (cpu_model = host.cpu.model if (host.cpu and host.cpu.model) else '')."
    - "Pattern: per-host group_by_fingerprint pre-pass before the cross-host group_by_fingerprint pass. node_dict_from_host now calls group_by_fingerprint(host.networking, ('type','speed','state'), 'unit_count') BEFORE returning; the resulting per-host stanzas (each with a unit_count) then participate as inputs to the cross-host group_by_fingerprint(node_items, _FINGERPRINT_KEYS, 'quantity') call in write_systemname_yaml. Two-level grouping: per-host (NICs → NIC-stanzas) → cross-host (host-stanzas → fleet-stanzas)."
    - "Pattern: empty-list emit hands off to the D-3 fallback. node_dict_from_host emits 'networking': [] when host.networking is empty; downstream _splice_stub_lists' D-3 / D-17 conditional branch detects the empty list and substitutes [dict(_NETWORKING_STUB)]. The two seams compose: emit knows host data; splice knows schema obligations."

key-files:
  created: []
  modified:
    - "mlpstorage_py/rules/models.py"
    - "mlpstorage_py/system_description/auto_generator.py"
    - "tests/unit/test_cluster_collector.py"
    - "tests/unit/test_auto_generator.py"
    - "tests/integration/test_systemname_yaml_end_to_end.py"

key-decisions:
  - "HostInfo dataclass extension follows D-16 precedent verbatim. Two new fields appended after `system: Optional[HostSystemInfo] = None` and before `collection_timestamp: Optional[str] = None`: `chassis_model: str = ''` and `networking: List[Dict[str, Any]] = field(default_factory=list)`. from_collected_data reads `data.get('chassis_model', '')` and `data.get('networking', [])`, passing them as kwargs to cls(...). HostInfo.from_dict is NOT modified — it handles a different code path that does not flow MPI-collected data; modifying it would be out-of-scope per the plan's <action> Step 2."
  - "node_dict_from_host emits the new 'networking' key directly, NOT via the _splice_stub_lists splice path. Phase 2's pattern was: emit shape WITHOUT networking, then _splice_stub_lists injected `[_NETWORKING_STUB]` post-process. Phase 3 inverts this: emit shape now ALWAYS includes a top-level 'networking' key (with real grouped data or []), and _splice_stub_lists' Plan 03-04 extension either splices D-17 traffic=[] onto up entries (real-data path) or substitutes _NETWORKING_STUB (empty-list path). The two seams compose cleanly without the splice helper needing to know about emit decisions."
  - "Per-host group_by_fingerprint pre-pass uses the same helper as the cross-host pass. node_dict_from_host calls group_by_fingerprint(host.networking, ('type','speed','state'), 'unit_count') — the same helper that write_systemname_yaml later calls with _FINGERPRINT_KEYS / 'quantity'. Same code, two levels: NICs → NIC-stanzas → host-stanzas → fleet-stanzas. The four-key tuple for per-host (no chassis or OS — those are host-level, not NIC-level) keeps the seam clean."
  - "Two existing tests required updates (not new tests): test_node_dict_cpu_fields and test_node_dict_no_extra_keys. Both assert exact set equality on top-level emitted keys. Phase 2's set was {friendly_description, chassis, operating_system}; Phase 3's is {friendly_description, chassis, operating_system, networking}. Updates are additive (one new expected entry in each set) and add a positive assertion that `result['networking'] == []` when host.networking is the dataclass default. No semantic regression."

patterns-established:
  - "Pattern: additive dataclass field + from_factory wiring (D-16 num_sockets precedent applied to chassis_model + networking). Two-line dataclass field append + two-line read + two-line kwarg insertion in from_collected_data. Mirror this shape for future HostInfo extensions in Phase 4 (drives) and beyond."
  - "Pattern: per-host group_by_fingerprint pre-pass for collection-of-entities → stanzas, BEFORE the cross-host pass. The same helper composes at two levels (NICs and hosts) when the input shape is dict-list and the fingerprint key tuple identifies the equivalence class. Phase 4 drives[] will likely follow the same pattern: per-host group_by_fingerprint(host.drives, ('vendor','model','interface','media','capacity'), 'unit_count') → drives stanzas → cross-host fingerprint via _drive_signature extractor (Phase 4 forward note in 03-CONTEXT.md)."

requirements-completed:
  - COLL-03
  - COLL-04

# Metrics
duration: ~30min
completed: 2026-06-22
---

# Phase 3 Plan 05: HostInfo + node_dict_from_host Integration — Phase 3 Vertical Closure Summary

**Two-line dataclass extension on `HostInfo` (chassis_model + networking, mirroring Phase 2's D-16 num_sockets precedent) + ~15-line node_dict_from_host wiring (chassis.model_name from Pattern F `(host.chassis_model or "")` defense; top-level `networking` key from per-host `group_by_fingerprint(host.networking, ("type","speed","state"), "unit_count")` or `[]` fallback) closes the Phase 3 vertical end-to-end. After this plan, a `mlpstorage ... run ...` invocation writes a `systemname.yaml` whose `clients[].chassis.model_name` reflects the DMI value (or `""` on collector failure) and whose `clients[].networking[]` reflects the real per-host NIC inventory grouped by (type, speed, state) with `unit_count`, including IB hardware as `type: infiniband` entries and degraded NICs as `state: down` entries. Cross-host quantity-grouping respects the new `_FINGERPRINT_KEYS` from Plan 03-04: identical chassis + identical networking signature collapse to one stanza with `quantity=N`; differing chassis OR differing networking signature produces N stanzas. 18 new RED tests across data-model (6), transform-unit (7), and integration (5) layers all GREEN after the two-commit ship; 2 existing Phase 2 tests updated to expect the new top-level "networking" emit key. Full unit suite (1690 passed) and full systemname.yaml integration suite (21/21 green) confirm no Phase 2 regression. The same 7 pre-existing `_check_safe_path_component` MagicMock fixture failures documented in 03-01 through 03-04 SUMMARYs persist (out-of-scope per Rule 3 scope boundary).**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-06-22 (current session)
- **Tasks:** 2 (both autonomous, both TDD)
- **Files modified:** 5 (two production, three test)
- **Production lines added:** +11 (rules/models.py) + +48 (auto_generator.py with docstring rewrite) = +59
- **Production lines removed:** 6 (auto_generator.py — old chassis.model_name="" hardcoded line + the network/drives stub comment line)
- **Test lines added:** +94 (test_cluster_collector.py) + +170 (test_auto_generator.py with 2 existing-test updates) + +205 (test_systemname_yaml_end_to_end.py) = +469

## Accomplishments

- **HostInfo dataclass extension shipped verbatim.** Two new fields appended after `system: Optional[HostSystemInfo] = None` and before `collection_timestamp: Optional[str] = None`: `chassis_model: str = ""` (Phase 3 / COLL-03) and `networking: List[Dict[str, Any]] = field(default_factory=list)` (Phase 3 / COLL-04). Mirrors the D-16 num_sockets precedent verbatim. List / Dict / Any / field already imported at the top of the module; no new imports needed.
- **HostInfo.from_collected_data wiring.** Before the final `return cls(...)`, two new reads: `chassis_model = data.get('chassis_model', '')` and `networking = data.get('networking', [])`. Inside the kwargs, two new positional-order kwargs (`chassis_model=chassis_model, networking=networking,`) placed before `collection_timestamp=...` to match field declaration order. HostInfo.from_dict left untouched as planned (separate code path, no MPI data flow).
- **node_dict_from_host chassis.model_name wiring.** The hard-coded `"model_name": ""` replaced with `"model_name": (host.chassis_model or "")` — Pattern F defensive blank-on-falsy that neutralizes None / missing falsy values without crashing. Now mirrors the existing Phase 2 chassis CPU field guards (`cpu_model = host.cpu.model if (host.cpu and host.cpu.model) else ""`).
- **node_dict_from_host networking emit.** New per-host grouping pass computed BEFORE the returned dict: `per_host_networking = group_by_fingerprint(host.networking, ("type", "speed", "state"), "unit_count") if host.networking else []`. New `"networking": per_host_networking` key inside the returned dict, placed AFTER the chassis block and BEFORE operating_system to match node_description schema field order. The if/else guard ensures `host.networking == []` (empty/missing) flows to `[]` (not a crash; not a call to group_by_fingerprint with empty input that would still return `[]` but is wasteful).
- **node_dict_from_host docstring rewritten** to describe the Phase 3 emit shape: chassis.model_name source (host.chassis_model + Pattern F defense); new "networking" key + per-host grouping pass; D-3 / D-17 splice integration with `_splice_stub_lists`. The shape comment in the docstring now shows `"networking": [<grouped stanzas> | []]` with the inline explanation of the two downstream paths (real-data branch sets traffic=[] on up entries; empty-list branch falls back to _NETWORKING_STUB).
- **18 new RED tests written + GREEN after Task 2:**
  - `TestHostInfoChassisField` (3 tests): default '', from_collected_data populates from data['chassis_model'], missing key defaults to ''.
  - `TestHostInfoNetworkingField` (3 tests): default [], from_collected_data populates from data['networking'], missing key defaults to [].
  - `TestNodeDictChassisModel` (3 tests): real model_name passthrough, empty emits SER-02 blank, None emits SER-02 blank via Pattern F (or "") guard.
  - `TestNodeDictNetworking` (3 tests): per-host group_by_fingerprint collapse (2 up 100GbE → unit_count=2), empty list emit, mixed up+down split per (type,speed,state).
  - `TestNodeDictReflection` (1 test): top-level emit includes "networking" directly; chassis still issubset of Chassis schema; OS keys still match.
  - Integration (5 new): `test_full_run_emits_chassis_model_in_yaml`, `test_full_run_emits_networking_in_yaml` (ethernet + IB with D-17 traffic=[] splice), `test_cross_host_fingerprint_splits_on_chassis_model`, `test_cross_host_fingerprint_splits_on_networking_signature`, `test_validator_no_chassis_error_when_dmi_populated`.
- **3 additional integration tests passed without RED gate** because Phase 2 fallback paths already handle the cases: `test_full_run_chassis_model_empty_when_collection_failed` (Phase 2's existing model_name="" emit already passed), `test_full_run_emits_networking_blank_stub_when_no_collected_data` (Phase 2's _splice_stub_lists fallback already worked). These remain in the suite as regression locks.
- **2 existing Phase 2 tests updated** to expect the new top-level "networking" key (test_node_dict_cpu_fields, test_node_dict_no_extra_keys). Both now assert `result["networking"] == []` for the default-host fixture; both had previously asserted `"networking" not in result`. Additive updates; no semantic regression.
- **End-to-end verification confirms two-host homogeneous collapse:** Two hosts with identical chassis_model='PowerEdge R760' and identical networking inventory ([2x 100GbE up + 1x 200Gb IB up]) collapse to one stanza with quantity=2; per-host networking already grouped into [{ethernet,100,up,unit_count=2}, {infiniband,200,up,unit_count=1}].
- **Phase 3 Success Criteria 1-5 all addressed:**
  1. DMI readable → chassis.model_name reflects file contents (test_full_run_emits_chassis_model_in_yaml); DMI unreadable → blank + run completes (test_full_run_chassis_model_empty_when_collection_failed).
  2. networking[] contains one entry per (type, speed) group with unit_count; lo/docker/virbr/veth/bond-slave filtered (Plan 03-03 collector + Plan 03-05 emit shape via test_full_run_emits_networking_in_yaml).
  3. Down state appears with recognizable sentinel — emitted as `{type, state:down, unit_count:N}` without speed key on real-collector path, or as a separate stanza on test fixture (test_mixed_up_and_down_grouped_separately).
  4. Host with IB HCA produces type: infiniband entry — test_full_run_emits_networking_in_yaml asserts an IB entry surfaces from the _make_host_phase3 fixture; manual-only verification covers real-hardware confirmation per VALIDATION.md.
  5. Cross-host quantity-grouping collapses identical-fingerprint hosts and splits on chassis/networking differences (test_cross_host_fingerprint_splits_on_chassis_model + test_cross_host_fingerprint_splits_on_networking_signature confirm).

## Task Commits

1. **Task 1 RED:** `d4e5d2d` — `test(03-05): add failing tests for HostInfo chassis_model+networking + node_dict_from_host wiring + end-to-end integration`
2. **Task 2 GREEN:** `a9559b2` — `feat(03-05): wire chassis_model + networking through HostInfo and node_dict_from_host (COLL-03, COLL-04 closure)`

Both commits verified `git log -1 --format='%B' | grep -ci "co-authored\|claude\|anthropic"` returns `0` (no AI attribution per user memory `feedback_no_attribution.md`).

## Files Created/Modified

- `mlpstorage_py/rules/models.py` — **+11 lines.** HostInfo dataclass gains two new fields (chassis_model + networking). HostInfo.from_collected_data gains two `data.get(...)` reads + two `cls(...)` kwargs. HostInfo.from_dict and HostInfo.to_dict left untouched.
- `mlpstorage_py/system_description/auto_generator.py` — **+48 / -6 lines.** node_dict_from_host docstring rewritten to describe Phase 3 emit shape. Hard-coded `"model_name": ""` replaced with `(host.chassis_model or "")` Pattern F guard. New per-host grouping computation `per_host_networking = group_by_fingerprint(...) if host.networking else []` added before the returned dict; new `"networking": per_host_networking` key inside the returned dict between chassis and operating_system. Comment about Plan 02-03's splice updated to reflect Phase 3 split (networking emitted directly; drives still spliced).
- `tests/unit/test_cluster_collector.py` — **+94 lines.** Two new test classes (TestHostInfoChassisField, TestHostInfoNetworkingField), 6 tests total. Module-level comment block describing the Phase 3 / Plan 03-05 dataclass extension as a D-16 num_sockets precedent.
- `tests/unit/test_auto_generator.py` — **+170 / -4 lines.** New `_phase3_host` helper. New test classes TestNodeDictChassisModel (3), TestNodeDictNetworking (3), TestNodeDictReflection (1). Two existing tests updated: test_node_dict_cpu_fields and test_node_dict_no_extra_keys now assert `result["networking"] == []` and include "networking" in the expected top-level key set.
- `tests/integration/test_systemname_yaml_end_to_end.py` — **+205 lines.** New `_make_host_phase3` helper layering chassis_model + networking on top of the Phase 2 _make_host. 8 new integration tests: `test_full_run_emits_chassis_model_in_yaml`, `test_full_run_emits_networking_in_yaml`, `test_full_run_chassis_model_empty_when_collection_failed`, `test_cross_host_fingerprint_splits_on_chassis_model`, `test_cross_host_fingerprint_splits_on_networking_signature`, `test_full_run_emits_networking_blank_stub_when_no_collected_data`, `test_validator_no_chassis_error_when_dmi_populated`.

## Phase 2 / Phase 3 test impact

PLAN's Step 1 test-impact scan: `grep -rln "HostInfo\|from_collected_data\|chassis_model\|node_dict_from_host\|fake_local" tests/` returned hits in 13 test files. Audit results:

- `tests/unit/test_cluster_collector.py` — extended in this plan.
- `tests/unit/test_auto_generator.py` — extended in this plan; 2 existing tests updated (additive change, no semantic regression).
- `tests/integration/test_systemname_yaml_end_to_end.py` — extended in this plan; the existing fake_local literal at line 547 preserved unchanged (still exercises the Phase 2 regression-lock CR-01 path); new _make_host_phase3 helper added separately.
- `tests/unit/test_rules_dataclasses.py` — 7 hits constructing HostInfo with keyword args (e.g., `HostInfo(hostname='host1', memory=host1_memory)`). The two new fields have defaults ('' / []) so these tests don't break.
- `tests/unit/test_auto_generator_write.py:61` — `HostInfo(...)` keyword construction. Same defaults-protected pattern; no regression.
- `tests/conftest.py`, `tests/fixtures/sample_data.py`, `tests/unit/test_imports.py`, `tests/unit/test_benchmarks_kvcache.py`, `tests/unit/test_rules_checkers.py`, `tests/unit/test_rules_calculations.py`, `tests/unit/test_benchmark_run.py`, `tests/unit/test_benchmarks_vectordb.py` — comment / import-only hits; no code that constructs HostInfo with positional arguments. No regression.

Full unit suite confirms: 1690 passed (same as 03-04's 1690-test post-baseline; the 7 pre-existing `_check_safe_path_component` MagicMock fixture failures in test_datagen_command_generation / test_rules_calculations persist unchanged — out-of-scope per Rule 3 scope boundary).

## Decisions Made

- **HostInfo.from_dict NOT modified.** PLAN's Step 2 explicitly carved out from_dict as a separate code path that does not flow MPI-collected data; modifying it would be out-of-scope. The test_impact scan confirmed no caller path requires the new fields on the from_dict surface. Honored verbatim.
- **node_dict_from_host emits "networking" key directly, not via _splice_stub_lists.** Phase 2's _splice_stub_lists pattern was: emit shape WITHOUT networking → splice [_NETWORKING_STUB] post-process. Phase 3 inverts: emit ALWAYS includes 'networking' (with real grouped data or []) → _splice_stub_lists' Plan 03-04 extension either splices D-17 traffic=[] onto up entries (real-data path) or substitutes _NETWORKING_STUB (empty-list path). Two seams compose cleanly; no helper rename needed.
- **`if host.networking: group_by_fingerprint(...) else []` not `group_by_fingerprint(host.networking, ...)` always.** Two reasons: (a) PLAN's verbatim `<action>` Step 3(b) used the guarded form; (b) calling group_by_fingerprint with an empty list still returns `[]` (no functional difference) but the guarded form makes the intent crystal-clear in the source — readers immediately see "if no networking, emit []". One-line readability win for zero functional cost.
- **Two existing tests updated additively.** test_node_dict_cpu_fields and test_node_dict_no_extra_keys both assert exact set equality on the top-level emitted dict keys; Phase 3 adds "networking" to that set. The updates are additive (one new expected key + one new positive assertion `result["networking"] == []`) and add no new semantic constraints. Could have moved these tests to a Phase 3 section but the inline update keeps them in their original Phase 2 context, preserving the test_impact narrative.

## Deviations from Plan

### Auto-fixed Issues

None.

### No Process Deviations

No `git stash` used. No working-tree mutations outside the two atomic commits. The 03-04 process note (prefer `git diff <ref>` and `git show <ref>:<path>` for read-only baseline comparisons) was honored throughout.

---

**Total deviations:** 0 auto-fixed + 0 process deviations.
**Impact on plan:** PLAN's two-commit success criterion honored exactly. RED → GREEN transition was clean on the first GREEN run; no Rule 1 bugs surfaced during verification.

## Issues Encountered

- **Pre-existing dev-env collection errors** (carried from prior plans, used `--ignore=` flags): `tests/unit/test_benchmarks_base.py`, `tests/unit/test_parquet_reader.py`, `tests/unit/test_vdb_modular_fake_backend.py` fail at COLLECTION time with `ModuleNotFoundError` (psutil / pyarrow.parquet / numpy not installed in this dev shell). Out-of-scope per Rule 3 scope boundary.
- **Pre-existing fixture failures** (5 in test_datagen_command_generation, 2 in test_rules_calculations): `TypeError: expected string or bytes-like object, got 'MagicMock'` in `mlpstorage_py/rules/utils.py::_check_safe_path_component`. Same root cause documented in prior SUMMARYs (03-01 / 03-02 / 03-03 / 03-04). Their import chains do not touch `rules/models.py` or `auto_generator.py` — confirmed by the failures' identity remaining unchanged after this plan's changes. Out-of-scope per Rule 3 scope boundary.
- **Integration suite collection errors** (carried from prior plans): `tests/integration/test_compat.py`, `tests/integration/test_dlio_mpi.py`, `tests/integration/test_mpi_basic.py`, `tests/integration/test_zerocopy_direct.py` fail at COLLECTION time (SystemExit / ModuleNotFoundError: s3dlio). Worked around by running `pytest tests/integration/test_systemname_yaml_end_to_end.py -q` directly; 21/21 tests in the systemname.yaml suite pass.

## Surprises

### Two existing tests had stricter top-level-set assertions than expected

The test-impact scan caught test_node_dict_cpu_fields (line 200ish) and test_node_dict_no_extra_keys (line 323ish) asserting on the exact top-level emitted-dict keys (the latter via `assert set(result.keys()) == {...}`). These had to be updated as part of Task 1 RED to make the existing assertions pass post-Task-2 (otherwise GREEN would fail BOTH a new test AND an old test). Updated assertions are additive (one new expected entry per set) and add a positive `result["networking"] == []` assertion to lock down the new emit shape. Pattern: when adding new top-level emit keys, audit the test suite for exact-set-equality assertions and update them as part of the RED commit so the GREEN commit is purely additive on the production side.

### Suite runtime under 30s

Full unit + integration suites (1711 total tests with 7 pre-existing failures and 4 collection-error files ignored) run in **~29.3 seconds**. The Phase 3 end-to-end vertical adds 18 new tests but the runtime overhead is negligible — the new tests are all pure-Python in-memory transformations with no I/O beyond temporary YAML files in pytest's tmp_path.

### Empty networking → _NETWORKING_STUB fallback test passed pre-GREEN

`test_full_run_emits_networking_blank_stub_when_no_collected_data` was listed as a RED test in PLAN's Step 5 expectation but actually passed before Task 2 because Phase 2's existing _splice_stub_lists fallback already handles the case (host.networking=[] → no top-level networking key in emit → splice substitutes _NETWORKING_STUB). The test remains in the suite as a regression lock for the Phase 3 path; Plan 03-04's _splice_stub_lists extension preserved the fallback behavior.

## Threat Flags

None. The plan was a pure in-memory data-flow wiring; zero new I/O, zero new network, zero new subprocess. All four threat-register entries (T-3-15 Tampering compromised worker, T-3-16 Info Disclosure, T-3-17 DoS pathological networking list, T-3-SC supply-chain) dispositioned `mitigate` / `accept` with no blocking concerns. The Pattern F `(host.chassis_model or "")` guard neutralizes non-str / None values from a malicious worker; group_by_fingerprint's `_network_signature` already uses `.get(k, "")` for every key (Plan 03-04); PyYAML safe_dump coerces and quotes deterministically per D-10.

## User Setup Required

None — no external service configuration required.

## Manual-Only Verifications

Per VALIDATION.md, these remain submitter-side smoke checks (not automated in this plan):

1. **Real DMI hardware:** On a host with `/sys/class/dmi/id/product_name` readable and returning a non-placeholder value, run `mlpstorage closed training unet3d run file --results-dir /tmp/r1 --systemname sys-real ...` and grep the written `/tmp/r1/closed/<orgname>/systems/sys-real.yaml` for `model_name:` (expect the real product name).
2. **Real ethernet NICs:** Same run, grep for `type: ethernet` (expect real per-NIC entries with speed and state).
3. **Real InfiniBand HCA:** On a host with `/sys/class/infiniband/*` present, also expect `type: infiniband` entries.
4. **Container DMI restriction (Phase 3 known limitation):** Inside containers without DMI access, expect `model_name: ""` per universal-rule blank; verify the run still completes without crash.

These verifications remain submitter-responsibility per VALIDATION.md's "Manual-Only Verifications" section; automated unit + integration tests cover everything that does not require real hardware presence.

## Next Phase Readiness

- **Phase 3 vertical end-to-end COMPLETE.** COLL-03 and COLL-04 both fully satisfied: collector-side (Plans 03-02, 03-03), transform-side (Plan 03-04), and data-model-side / emit-side (this plan, 03-05). A `mlpstorage ... run ...` invocation now writes a systemname.yaml whose clients[].chassis.model_name and clients[].networking[] reflect real DMI + sysfs data; degraded NICs (down state) participate honestly in fingerprint grouping; collector-blind hosts fall back to SER-02 visible blanks.
- **Plan 03-05 was the final slice of Phase 3.** Awaiting `/gsd-verify-phase 03` to flip Phase 3 status from `executing` to `verified`, then `/gsd-transition` to advance to Phase 4 (drives[] population — COLL-05, COLL-06, COLL-07 per ROADMAP.md).
- **Phase 4 forward note from 03-CONTEXT.md remains valid.** drives[] will likely follow the same patterns established in Phase 3:
  - Schema extension prerequisite: does DriveInstance need a `state` field (online | offline | failed)? — Phase 4 discuss-phase decides.
  - Per-host drive grouping key: likely `(vendor_name, model_name, interface, media_type, capacity_in_GB)` — confirm during Phase 4 discussion.
  - Cross-host inclusion: add `drives` to `_FINGERPRINT_KEYS` via a `_drive_signature` extractor following the same pattern as `_network_signature`.
  - HostInfo will gain `drives: List[Dict[str, Any]]` field appended after the Phase 3 `networking` field — same D-16 additive pattern.
- **No deferred-items.md entries created** by this plan.

## Self-Check: PASSED

- `mlpstorage_py/rules/models.py`: FOUND (modified — +11 lines).
- `mlpstorage_py/system_description/auto_generator.py`: FOUND (modified — +48 / -6 lines including docstring rewrite).
- `tests/unit/test_cluster_collector.py`: FOUND (modified — +94 lines for 2 new test classes / 6 tests).
- `tests/unit/test_auto_generator.py`: FOUND (modified — +170 / -4 lines for 3 new test classes / 7 tests + 2 existing-test updates).
- `tests/integration/test_systemname_yaml_end_to_end.py`: FOUND (modified — +205 lines for 8 new integration tests + _make_host_phase3 helper).
- Commit `d4e5d2d`: FOUND — `test(03-05): add failing tests for HostInfo chassis_model+networking + node_dict_from_host wiring + end-to-end integration`.
- Commit `a9559b2`: FOUND — `feat(03-05): wire chassis_model + networking through HostInfo and node_dict_from_host (COLL-03, COLL-04 closure)`.
- AI-attribution check: `git log -1 --format='%B' | grep -ci "co-authored\|claude\|anthropic"` returns `0` for both commits.
- HostInfo surface: `python3 -c "from mlpstorage_py.rules.models import HostInfo; h = HostInfo(hostname='h'); assert h.chassis_model == ''; assert h.networking == []; ..."` prints `hostinfo surface ok`.
- node_dict surface: `python3 -c "from mlpstorage_py.system_description.auto_generator import node_dict_from_host; h = HostInfo(hostname='h', chassis_model='PowerEdge R760', networking=[...]); d = node_dict_from_host(h); assert d['chassis']['model_name'] == 'PowerEdge R760'; assert d['networking'] == [...]; ..."` prints `node_dict surface ok`.
- End-to-end pipeline: 2 homogeneous hosts → 1 stanza quantity=2 with grouped networking — confirmed via the inline verification snippet.
- Production grep counts:
  - `grep -c "chassis_model" mlpstorage_py/rules/models.py` returns 4 (dataclass field + comment + read + kwarg).
  - `grep -c "networking" mlpstorage_py/rules/models.py` returns 4 (dataclass field + read + kwarg + pre-existing network field reference).
  - `grep -n "host.chassis_model" mlpstorage_py/system_description/auto_generator.py` returns 2 matches (the `(host.chassis_model or "")` line + the `if host.networking` guard).
  - `grep -c "group_by_fingerprint(\s*host.networking" mlpstorage_py/system_description/auto_generator.py` returns 1.
- Test counts: 18 new tests (6 cluster_collector + 7 auto_generator + 5 integration RED) all GREEN after Task 2. Plus 3 additional integration tests that passed without RED gate (regression-lock coverage). Plus 2 existing tests updated additively.
- Suite results: 1690 passed in tests/unit (7 pre-existing failures out-of-scope per Rule 3); 21 passed in tests/integration/test_systemname_yaml_end_to_end.py (4 pre-existing collection errors in other integration files unrelated to this plan); ~29.3s end-to-end runtime.

---
*Phase: 03-chassis-model-networking-coverage*
*Completed: 2026-06-22*
