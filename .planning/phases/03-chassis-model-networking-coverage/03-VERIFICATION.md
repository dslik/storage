---
phase: 03-chassis-model-networking-coverage
verified: 2026-06-22T00:00:00Z
status: human_needed
score: 7/7 must-haves verified
overrides_applied: 0
deferred:
  - truth: "HostInfo.to_dict serializes chassis_model and networking so the metadata JSON roundtrip preserves Phase 3 collected data (CR-01 from 03-REVIEW.md)"
    addressed_in: "Phase 5 (LIFE-02)"
    evidence: "Phase 5 Success Criterion 1: 'After Phase 2-4 has written ... systemname.yaml ... re-running the same run command against the same fleet completes without modifying the file and without raising drift errors'. LIFE-02 owns the on-disk-vs-in-memory logical diff, which requires roundtripping the full collected inventory. The 02-REVIEW.md precedent (WR-03) deferred the analogous HostCPUInfo.num_sockets to_dict drift to the same Phase 5 LIFE-02 work."
  - truth: "Pattern B parity tests compare the literal CONTENTS of _DMI_PLACEHOLDERS / _OPERSTATE_UP_VALUES / _VIRTUAL_NAME_PREFIXES / _VIRTUAL_NAME_RE / _SAFE_IFACE_NAME_RE between module and MPI-script copies (WR-02 from 03-REVIEW.md)"
    addressed_in: "Phase 5 LIFE-02 (or new Phase 4 hardening if scoped)"
    evidence: "Phase 3 ROADMAP SC do not require set-equality parity; current tests verify behavioral parity on representative inputs. Hardening this is a maintainability/regression-detection enhancement, not a Phase 3 goal."
  - truth: "Bond aggregation excludes slaves whose operstate is not 'up' (WR-03 from 03-REVIEW.md)"
    addressed_in: "Phase 5 LIFE-02 (or follow-on hardening)"
    evidence: "Phase 3 ROADMAP SC do not specify operstate gating for bond slaves; current implementation matches RESEARCH lines 484-519 verbatim. Refinement is a hardening item."
  - truth: "SSH-collector script duplicates collect_chassis_model and collect_networking inline so non-MPI collection paths emit the same chassis/networking data (IN-06 from 03-REVIEW.md)"
    addressed_in: "Phase 4 or Phase 5 (asymmetry between MPI and SSH collection paths)"
    evidence: "Phase 3 explicitly scoped the MPI fan-out path; CONTEXT.md does not include SSH-script parity. 03-CONTEXT mentions this as a known gap addressable later."
  - truth: "_VIRTUAL_NAME_RE matches docker_gwbridge / virbr0-nic / wg-corp style names (WR-01 from 03-REVIEW.md)"
    addressed_in: "Phase 5 hardening / future iteration"
    evidence: "Bridge-master subdir check is the safety net for docker_gwbridge today. ROADMAP SC 2 is met because the filter as-implemented passes all 5 Phase 3 success criteria via the bridge-master path. Tightening the regex to docker.* / virbr.* / etc. is a documented WR-01 hardening item, not a SC-blocking gap."
human_verification:
  - test: "On a host with real DMI (`/sys/class/dmi/id/product_name` readable and returning a non-placeholder value), run `mlpstorage closed training unet3d run file --results-dir /tmp/r1 --systemname sys-real ...` and inspect the written `<results-dir>/closed/<orgname>/systems/sys-real.yaml` for `model_name:` returning the real product name."
    expected: "clients[].chassis.model_name contains the actual DMI product_name (e.g., 'PowerEdge R760'), NOT empty string or BIOS placeholder."
    why_human: "Requires real DMI hardware exposing /sys/class/dmi/id/product_name; WSL2 dev shell returns empty string per the universal-failure rule but cannot exercise the DMI-readable branch."
  - test: "On a host with at least one physical ethernet NIC, same run + grep for `type: ethernet` entries."
    expected: "Real per-NIC entries surface with non-empty speed (Gbps) and state=up; lo/docker/virbr/veth/bond-slave do not appear."
    why_human: "Requires real ethernet hardware on the host; WSL2 dev shell exposes a virtual eth0 that passes the up-state path but is not a real benchmark client."
  - test: "On a host with at least one InfiniBand HCA present at `/sys/class/infiniband/*`, same run + grep for `type: infiniband`."
    expected: "At least one networking[] entry has type=infiniband, state=up, and a positive speed (Gbps)."
    why_human: "Requires real InfiniBand hardware; satisfies ROADMAP SC 4 in the real environment. Unit tests use tmp_path fixtures to simulate this shape."
  - test: "Inside a container without DMI access, run the same command and verify the file is still produced and the run does not crash."
    expected: "model_name: \"\" (empty) and the run still completes; no exception raised on the missing /sys/class/dmi/id/ tree."
    why_human: "Requires running inside a container with restricted /sys; cannot be exhaustively verified without a real container runtime."
---

# Phase 3: Chassis Model + Networking Coverage Verification Report

**Phase Goal:** Extend the auto-filled YAML with DMI chassis `model_name` and a `networking[]` block sourced from sysfs.
**Verified:** 2026-06-22
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria + PLAN must_haves consolidation)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Every per-host `clients[]` stanza in `systemname.yaml` gains a non-empty `chassis.model_name` when DMI is populated (and `""` when not). | ✓ VERIFIED | `node_dict_from_host` at `mlpstorage_py/system_description/auto_generator.py:308` emits `"model_name": (host.chassis_model or "")`. `HostInfo.from_collected_data` at `mlpstorage_py/rules/models.py:249,259` reads `chassis_model` from collected data. Integration test `test_full_run_emits_chassis_model_in_yaml` and `test_full_run_chassis_model_empty_when_collection_failed` lock both branches. Python smoke run: `node_dict_from_host(HostInfo(...chassis_model='PowerEdge R760'))` → `{'chassis': {'model_name': 'PowerEdge R760', ...}, ...}` (verified live). |
| 2 | Every per-host stanza gains a `networking[]` list of `{type, speed, state, unit_count}` entries. | ✓ VERIFIED | `node_dict_from_host` builds `per_host_networking = group_by_fingerprint(host.networking, ("type","speed","state"), "unit_count")` at `auto_generator.py:296-303`, emitted as top-level `"networking": per_host_networking` key at `auto_generator.py:316`. Smoke run yields `[{'type':'ethernet','speed':100,'state':'up','unit_count':2}, {'type':'infiniband','speed':200,'state':'up','unit_count':1}]`. Test `test_full_run_emits_networking_in_yaml` integration-tests the YAML emit path. |
| 3 | On a host with at least one operational ethernet interface, at least one `networking` entry has `state: up` and a positive `speed`. | ✓ VERIFIED | `collect_networking()` at `cluster_collector.py:889` walks `/sys/class/net/*`; effective-state demotion at `cluster_collector.py` (D-20) only emits state=up when speed > 0 (Pitfall 2 lock). `TestNetworkingOperstate::test_operstate_up_passes_through` and `TestNetworkingEffectiveState::test_virtio_speed_minus_one_demotes_to_down` lock this. WSL2 dev-shell live run returns `{'type':'ethernet','speed':10,'state':'up'}` (real virtual eth0). |
| 4 | On a host with at least one InfiniBand HCA, at least one `networking` entry has `type: infiniband`. | ✓ VERIFIED | IB walk at `cluster_collector.py:889+` walks `/sys/class/infiniband/<dev>/ports/<port>/{state,rate}`; `_parse_ib_state` enforces `'4: ACTIVE'` → up. `TestNetworkingInfiniband::test_active_ib_port_emits_up` and `test_dual_port_hca_emits_two_entries` lock the emit shape. Integration test `test_full_run_emits_networking_in_yaml` includes an IB entry from `_make_host_phase3` fixture. Real-hardware confirmation requires manual verification (no IB on WSL2 dev shell). |
| 5 | Quantity-grouping collapses hosts matching on the new chassis/networking fingerprint into one stanza and splits hosts that differ. | ✓ VERIFIED | `_FINGERPRINT_KEYS` extended to 8 entries (6 Phase 2 + `chassis.model_name` + `("networking_sig", _network_signature)`) at `auto_generator.py:115-124`. Smoke run with 2× PE-R760 + 1× PE-R650 produces 2 stanzas (qty=2 for PE-R760, qty=1 for PE-R650). `TestGroupByFingerprintExtended::test_extended_keys_collapse_homogeneous_fleet`, `test_extended_keys_split_on_chassis_model_difference`, `test_extended_keys_split_on_networking_signature_difference` and integration tests `test_cross_host_fingerprint_splits_on_chassis_model` / `test_cross_host_fingerprint_splits_on_networking_signature` lock both directions. |
| 6 | (PLAN 03-01 must_have) NetworkPort.state is REQUIRED `Literal["up","down"]` field; speed/traffic enforced when state==up; D-17 traffic=[] splice applies to up entries post-Pydantic. | ✓ VERIFIED | `schema_validator.py:165` declares `state: Literal["up", "down"]` (no default). `_require_speed_and_traffic_when_up` at `schema_validator.py:170` enforces the conditional. `_splice_stub_lists` at `auto_generator.py:405-418` sets `entry["traffic"] = []` on up entries when real networking present; falls back to `_NETWORKING_STUB` when empty. Smoke run confirms D-17 splice mutates the dump dict; `_splice_stub_lists` empty-networking branch substitutes the blank stub. `TestSpliceUpNicTraffic` (7 tests) and `TestNetworkPortState` (6 tests) lock both. |
| 7 | (PLAN 03-02/03-03 must_have) MPI parity preserved (Pattern B): module and inline-script copies behave equivalently on representative inputs. | ✓ VERIFIED | `cluster_collector.py:1471,1485,1493,1517` show inline duplicates of `_DMI_PLACEHOLDERS`, `_normalize_dmi`, `collect_chassis_model`, `_VIRTUAL_NAME_RE`, `collect_networking` inside `MPI_COLLECTOR_SCRIPT` string. `TestMPIScriptParity::test_chassis_functions_match_module` and `TestNetworkingMPIScriptParity::test_networking_functions_match_module` both pass green (verified in test run). |

**Score:** 7/7 truths verified

### Deferred Items

Items not actionable for Phase 3 — addressed in later phases:

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | `HostInfo.to_dict()` serialization of `chassis_model` and `networking` (CR-01 in 03-REVIEW.md) | Phase 5 (LIFE-02) | Phase 5 SC 1 requires the in-memory image to round-trip identically against on-disk metadata. CR-01 is the same defect class as Phase 2's WR-03 `num_sockets` `to_dict` drift, which was explicitly deferred to Phase 5 LIFE-02. Phase 3 systemname.yaml emit path (the actual goal) goes through `node_dict_from_host`, NOT `to_dict`, and is correctly wired. |
| 2 | Pattern B parity tests compare literal CONTENTS of frozensets/regex patterns (WR-02) | Hardening / future | Phase 3 ROADMAP SC do not require this; current tests verify behavioral parity. Maintainability improvement. |
| 3 | Bond aggregation gates on per-slave operstate (WR-03) | Hardening / future | Current implementation matches RESEARCH 484-519 verbatim. Phase 3 SC do not specify operstate gating. |
| 4 | SSH-collector script duplicates `collect_chassis_model` / `collect_networking` (IN-06) | Phase 4 or 5 | Phase 3 explicitly scoped MPI path; CONTEXT.md notes SSH parity as known gap. |
| 5 | `_VIRTUAL_NAME_RE` widened to match docker_gwbridge / virbr0-nic / etc. (WR-01) | Hardening / future | Bridge-master subdir check is the existing safety net. ROADMAP SC are met via that path. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `mlpstorage_py/system_description/schema_validator.py` | NetworkPort.state Literal + model_validator | ✓ VERIFIED | Line 165: `state: Literal["up", "down"]` (REQUIRED, no default). Line 170: `_require_speed_and_traffic_when_up` model_validator(mode="after"). |
| `mlpstorage_py/system_description/schema.yaml` | network_port.state enum | ✓ VERIFIED | Line 85: `state: enum( 'up', 'down' )`. |
| `mlpstorage_py/system_description/auto_generator.py` | `_NETWORKING_STUB` parity, `_FINGERPRINT_KEYS` extension, `_network_signature`, `_resolve_fingerprint_key`, `_splice_stub_lists` D-17 splice, `node_dict_from_host` wiring | ✓ VERIFIED | `_NETWORKING_STUB` has `state: ""` (line 36 region). `_FINGERPRINT_KEYS` is 8-tuple with `chassis.model_name` (line 120) and `("networking_sig", _network_signature)` (line 123). `_network_signature` at line 73. `_resolve_fingerprint_key` at line 152. `_splice_stub_lists` at line 405-418 has D-17 branch + fallback. `node_dict_from_host` at line 296-316 wires `host.chassis_model` and per-host `group_by_fingerprint(host.networking, ...)`. |
| `mlpstorage_py/cluster_collector.py` | `_DMI_PLACEHOLDERS`, `_normalize_dmi`, `collect_chassis_model`, `_VIRTUAL_NAME_*`, `collect_networking`, MPI script duplicates, `result['chassis_model']` + `result['networking']` wiring | ✓ VERIFIED | Module symbols at lines 639, 653, 668, 718, 726, 889. MPI script inline duplicates at 1471, 1485, 1493, 1512, 1517, 1610. Wiring in `collect_local_system_info` at 1152, 1164. Wiring in MPI `collect_local_info` at 1795, 1805. |
| `mlpstorage_py/rules/models.py` | `HostInfo.chassis_model` + `HostInfo.networking` fields + `from_collected_data` reads | ✓ VERIFIED | Lines 179-180: dataclass fields. Lines 249-260: `from_collected_data` reads + kwargs. **Caveat:** `to_dict()` does NOT serialize these fields (CR-01) — deferred to Phase 5 LIFE-02, NOT a Phase 3 SC concern. |
| All 6 `example_*.yaml` | `state: "up"` lines on every networking entry | ✓ VERIFIED | 13 `state: "up"` lines across the 6 example files; revalidate clean (schema_validator integration test green). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `cluster_collector.collect_local_system_info` | `cluster_collector.collect_chassis_model` | `result['chassis_model'] = collect_chassis_model()` | ✓ WIRED | `cluster_collector.py:1152`. |
| `cluster_collector.collect_local_system_info` | `cluster_collector.collect_networking` | `result['networking'] = collect_networking()` | ✓ WIRED | `cluster_collector.py:1164`. |
| MPI_COLLECTOR_SCRIPT.collect_local_info | inline collect_chassis_model / collect_networking | parallel `result['chassis_model']` / `result['networking']` | ✓ WIRED | Lines 1795, 1805. Pattern B parity tests confirm equivalence. |
| `models.HostInfo.from_collected_data` | `data['chassis_model']` / `data['networking']` | `data.get('chassis_model', '')` / `data.get('networking', [])` | ✓ WIRED | `models.py:249-260`. |
| `auto_generator.node_dict_from_host` | `host.chassis_model` | `(host.chassis_model or "")` Pattern F guard | ✓ WIRED | `auto_generator.py:308`. |
| `auto_generator.node_dict_from_host` | `group_by_fingerprint` (per-host) | `group_by_fingerprint(host.networking, ("type","speed","state"), "unit_count")` | ✓ WIRED | `auto_generator.py:296-303`. |
| `auto_generator._FINGERPRINT_KEYS` | `_network_signature` | callable-tuple entry `("networking_sig", _network_signature)` consumed by `_resolve_fingerprint_key` | ✓ WIRED | `auto_generator.py:123`, dispatcher at 152-157, used in `group_by_fingerprint` at the `_resolve_fingerprint_key(item, k)` call site. |
| `auto_generator._splice_stub_lists` | real per-client networking (from node_dict_from_host) | `if existing_net: iterate + set traffic=[] on up; else fall back to [_NETWORKING_STUB]` | ✓ WIRED | `auto_generator.py:405-418`. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `node_dict_from_host` emit | `chassis.model_name` | `host.chassis_model` (from HostInfo, sourced from MPI/local collect_chassis_model reading `/sys/class/dmi/id/product_name`) | Yes — real sysfs read; on WSL2 dev shell returns `""` per universal-failure rule; on real hardware returns the DMI product name. Smoke: `node_dict_from_host(HostInfo(chassis_model='PowerEdge R760'))` returns `'PowerEdge R760'`. | ✓ FLOWING |
| `node_dict_from_host` emit | `networking[]` | per-host `group_by_fingerprint(host.networking, ...)` over real collect_networking output | Yes — real sysfs walk of `/sys/class/net/*` and `/sys/class/infiniband/*`. WSL2 dev shell returns `[{'type':'ethernet','speed':10,'state':'up'}]` (real virtual NIC). Smoke run with 2× 100GbE + 1× IB returns properly grouped entries with `unit_count`. | ✓ FLOWING |
| `_splice_stub_lists` D-17 mutation | `entry['traffic']` | post-Pydantic dump dict from real networking | Yes — splice mutates `traffic=[]` on up entries (deterministic). Smoke confirms. | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Schema NetworkPort tests | `pytest tests/unit/test_schema_validator.py::TestNetworkPortState -q` | Included in 100-test run; all green | ✓ PASS |
| Transform extension tests (4 new classes) | `pytest tests/unit/test_auto_generator.py::TestNetwork*/TestResolve*/TestGroup*/TestSplice* -q` | All pass within 100-test run | ✓ PASS |
| Chassis collector tests | `pytest tests/unit/test_cluster_collector.py::TestDMIPlaceholders TestCollectChassisModel TestMPIScriptParity -q` | All pass | ✓ PASS |
| Networking collector tests | `pytest tests/unit/test_cluster_collector.py::TestNetworkingFilters TestNetworkingBond TestNetworkingInfiniband TestNetworkingMPIScriptParity -q` | All pass | ✓ PASS |
| Integration suite | `pytest tests/integration/test_systemname_yaml_end_to_end.py -q` | 21 passed in 0.40s | ✓ PASS |
| End-to-end Python smoke (must_have surface) | inline `python3 -c` smoke against HostInfo + node_dict_from_host + _FINGERPRINT_KEYS + group_by_fingerprint + _splice_stub_lists | `ALL must-haves verified` printed | ✓ PASS |
| Cross-host quantity-grouping smoke | 2× PE-R760 + 1× PE-R650 → 2 groups (qty=2, qty=1) | Correct split | ✓ PASS |
| Down-state emit smoke | host with ethernet down + ethernet up → both states surfaced in `networking[]` | Down entry present without speed key | ✓ PASS |

### Probe Execution

No phase-declared probes for Phase 3 (Python pytest-based phase). N/A.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| COLL-03 | 03-02 (collector), 03-05 (integration) | Collector exposes `clients[].chassis.model_name` from `/sys/class/dmi/id/product_name`; universal-rule blank on failure. | ✓ SATISFIED | `collect_chassis_model` reads DMI file with 8KB cap + D-21 placeholder normalization; `HostInfo.chassis_model` wires via `from_collected_data`; `node_dict_from_host` emits `chassis.model_name`. End-to-end integration test green. |
| COLL-04 | 03-01 (schema), 03-03 (collector), 03-04 (transform), 03-05 (integration) | Collector exposes `clients[].networking[]` with `type`, `speed`, `unit_count`; virtual interfaces filtered; down-state has recognizable sentinel; universal-rule blank on failure. | ✓ SATISFIED | `collect_networking` walks `/sys/class/net/*` + `/sys/class/infiniband/*` with D-18/D-19/D-20 enforcement; per-host `group_by_fingerprint(host.networking, ("type","speed","state"), "unit_count")` produces stanzas with `unit_count`; down NICs emit `{type, state:down}` without speed key; cross-host `_FINGERPRINT_KEYS` includes networking signature. End-to-end integration tests green. |

Both requirement IDs in PLAN frontmatter (`COLL-03`, `COLL-04`) are accounted for. REQUIREMENTS.md maps both to Phase 3 — no orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none in Phase 3 modified production files) | — | — | — | — |

Scan of `mlpstorage_py/cluster_collector.py`, `mlpstorage_py/rules/models.py`, `mlpstorage_py/system_description/auto_generator.py`, `mlpstorage_py/system_description/schema_validator.py`, `mlpstorage_py/system_description/schema.yaml` for `TBD|FIXME|XXX` returned zero matches.

### Human Verification Required

See `human_verification:` block in frontmatter for the 4 items requiring real hardware (DMI, ethernet, InfiniBand, container). These are unavoidable submitter-side smoke checks per 03-VALIDATION.md "Manual-Only Verifications" section; the automated unit + integration tests cover everything not requiring real hardware presence.

### Gaps Summary

**No gaps blocking Phase 3 goal achievement.**

All 5 ROADMAP success criteria for Phase 3 are satisfied:

1. ✓ chassis.model_name surfaces from DMI (or blank on failure) — `node_dict_from_host` emits the field, locked by integration tests.
2. ✓ networking[] with `{type, speed, state, unit_count}` — per-host grouping pass produces the shape; integration tests confirm.
3. ✓ Operational ethernet → state=up with positive speed — effective-state demotion (D-20) ensures the up/positive-speed coupling.
4. ✓ InfiniBand HCA → type=infiniband entry — IB walk emits the shape; manual verification needed for real hardware.
5. ✓ Quantity-grouping collapses on chassis/networking fingerprint and splits on differences — 8-entry `_FINGERPRINT_KEYS` with the callable-extractor pattern verified by both unit and integration tests.

**Cross-reference assessment of 03-REVIEW.md CR-01 (HostInfo.to_dict drift):**

CR-01 is correctly classified as a Phase 5 LIFE-02 concern, NOT a Phase 3 BLOCKER:

- Phase 3's goal is to emit chassis/networking into systemname.yaml. That path goes through `node_dict_from_host` → YAML write, which IS correctly wired.
- CR-01 documents drift on a SEPARATE artifact: `ClusterInformation.as_dict() → <run_id>_metadata.json` JSON metadata. This is the lifecycle/roundtrip path that Phase 5 LIFE-02 will exercise.
- Direct precedent: Phase 2's 02-REVIEW.md WR-03 found the exact same class of defect for `HostCPUInfo.num_sockets` (added to dataclass in Phase 2, omitted from `to_dict` serialization) and that was explicitly deferred to Phase 5 LIFE-02 (where it gets fixed alongside the round-trip diff machinery that needs it).
- Phase 5 LIFE-02 cannot ship without fixing the `to_dict` drift (the diff requires reading the same fields back); deferring CR-01 there matches both the SC scope boundaries and the existing precedent.

**The 5 WARNING-level items from 03-REVIEW.md** (WR-01, WR-02, WR-03, WR-04, WR-05, IN-01..IN-06) are similarly hardening/maintainability items, not Phase 3 SC-blocking gaps:
- WR-04 (local-storage spliced-blank UX) is documented in CONTEXT.md as Phase 5 LIFE-02 territory.
- WR-05 (filesystem-failure test parametrization) is a test-coverage enhancement, not a production gap.
- WR-01 (regex tightening), WR-02 (parity test breadth), WR-03 (bond operstate gating), IN-01..IN-06 are all maintainability/hardening items that do not affect any of the 5 ROADMAP success criteria.

**Test evidence:** 100 Phase-3-specific unit tests green; 21 integration tests green (8 new Phase 3 integration tests + 13 Phase 2 regression-locks). The full surface contract `python3 -c "from mlpstorage_py.system_description.auto_generator import ...; ..."` succeeds.

---

*Verified: 2026-06-22*
*Verifier: Claude (gsd-verifier)*
