---
phase: 04-sysctl-environment-and-drives-coverage
verified: 2026-06-23T23:18:23Z
resolved: 2026-06-23T23:58:00Z
status: passed
score: 8/8 must-haves verified
overrides_applied: 0
gaps: []
deferred: []
resolution:
  - gap: "Full unit test suite is regression-free"
    fix: "Extended test_auto_generator_write.py::test_yaml_block_style regex to allow `sysctl: []` and `environment: []` alongside the existing `traffic: []` exception. Resolution path (a) from the verifier — keeps empty allowlist-driven blocks as explicit self-documenting markers: `sysctl: []` clearly says 'we looked and there is nothing' vs. silent omission which would require the reader to know the block was even possible. D-33 drives-omit behavior preserved — client nodes commonly have no drives so omission is the right signal there. After fix: 1850 passed, 7 failed (the documented pre-existing MagicMock set: 5 test_datagen + 2 test_rules_calculations)."
---

# Phase 4: Sysctl, Environment, and Drives Coverage Verification Report

**Phase Goal:** The auto-generated `systemname.yaml` also reports a curated sysctl snapshot, the relevant filtered environment, and an `lsblk`-sourced drive inventory — so a submitter who looks at the generated YAML sees a near-complete client description, with only the truly non-derivable fields left to fill in.
**Verified:** 2026-06-23T23:18:23Z (initial); **Resolved:** 2026-06-23T23:58:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                                                                                                                                                                                              | Status     | Evidence                                                                                                                                                                                                                                                                          |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | SC #1 (COLL-05): `clients[].sysctl[]` populated from `/proc/sys` allowlist; adding a pattern to the allowlist file picks it up without code changes                                                                                                                | ✓ VERIFIED | `_load_sysctl_allowlist()` reads `mlpstorage_py/system_description/sysctl_allowlist.txt` (4 globs shipped); `collect_sysctl()` walks /proc/sys, applies fnmatch-translated regex, per-leaf D-2 isolation. Live: `len(collect_sysctl()) == 134` entries on WSL2 dev shell.        |
| 2   | SC #2 (COLL-06): `clients[].environment[]` filtered to allowlist with D-23 first-4/last-4 mask on AWS_ACCESS_KEY_ID and D-24 length-only sentinel on AWS_SECRET_ACCESS_KEY via the unified `storage_config.py` policy                                              | ✓ VERIFIED | `_redact_secret` + `_mask_credential_id` in `storage_config.py`; D-26 prefix+literal allowlist with `_ENV_LITERALS = {'BUCKET'}` and `_ENV_PREFIXES = ('AWS_','STORAGE_','OMPI_','UCX_','NCCL_')`. Live: `AKIAIOSFODNN7EXAMPLE` → `AKIA****MPLE`; 40-char secret → `[SET — 40 chars]`. |
| 3   | SC #3 (COLL-07): On a host with lsblk + ≥1 device, `clients[].drives[]` contains one entry per `(vendor_name, model_name, interface, capacity_in_GB)` group with `unit_count`, `capacity_in_GB` base-10, and `interface` in `{nvme, sata, sas}` (NOT `'other'`)    | ✓ VERIFIED | `collect_drives()` shells out to `lsblk -J -b -d -o NAME,MODEL,VENDOR,SIZE,ROTA,TRAN,RM`; D-31 4-rule filter chain (RM skip via `rm in (1, True, '1')` Rule 1 fix; virtual NAME/TRAN reject; unknown-TRAN drop with empty-TRAN-nvme rescue); `// 10**9` decimal GB; `node_dict_from_host` runs per-host group_by_fingerprint with `unit_count` aggregation.       |
| 4   | SC #4: Drive entries do NOT contain `media_type`, `form_factor`, or `performance` (submitter-filled per SER-02)                                                                                                                                                     | ✓ VERIFIED | `collect_drives()` line 1360-1365: emit dict carries exactly 4 keys (vendor_name, model_name, interface, capacity_in_GB); `TestPhase4EndToEnd::test_yamale_schema_validation_passes_on_phase_4_emit_shape` locks the contract.                                                       |
| 5   | SC #5 (D-33): On a host where lsblk is absent or returns no devices/all-filtered, `clients[].drives` is OMITTED entirely from the YAML and `run` still completes                                                                                                  | ✓ VERIFIED | `_splice_stub_lists` line 566-577: `if existing_drives: pass; else: client.pop("drives", None)`. Live verified: empty-drives client dict → `drives` key absent in result. Live WSL2 smoke (all 4 disks dropped via D-31 rule 3 null TRAN) → emitted YAML has no `drives:` block.    |
| 6   | D-34/D-35 cross-host fingerprint splits stanzas on sysctl/environment/drives divergence                                                                                                                                                                            | ✓ VERIFIED | `_FINGERPRINT_KEYS` is the 11-tuple (verified via `len(_FINGERPRINT_KEYS) == 11`); `_EXTRACTOR_SOURCE_KEYS` has all 4 keys (drives_sig, environment_sig, networking_sig, sysctl_sig); `_resolve_fingerprint_key` generalized via the map. Live: 3 identical hosts → 1 stanza; sysctl-divergent → 2 stanzas; env-divergent → 2 stanzas; drives-divergent → 2 stanzas. |
| 7   | End-to-end YAML emission via `node_dict_from_host` shows all 7 top-level keys (friendly_description, chassis, networking, sysctl, environment, drives, operating_system); HostInfo carries the 3 new list-typed fields with `default_factory=list`                  | ✓ VERIFIED | `HostInfo.sysctl/environment/drives` all `List[Dict[str,Any]] = field(default_factory=list)` (rules/models.py:181-183); `HostInfo.from_collected_data` reads `data.get(...)` for all three (lines 261-263); `node_dict_from_host` emits all 3 keys (lines 441-443). Live: `sorted(result.keys()) == ['chassis','drives','environment','friendly_description','networking','operating_system','sysctl']`. |
| 8   | Full unit test suite is regression-free (Phase 4 introduces no new failing tests)                                                                                                                                                                                    | ✓ VERIFIED | Resolved inline post-verification: extended `test_yaml_block_style` regex to allow `sysctl: []` and `environment: []` as explicit self-documenting empty-block markers (companion to the existing `traffic: []` exception; D-33 drives-omit retained for the "client commonly has no drives" case). `.venv/bin/python -m pytest tests/unit/ -q` now: **1850 passed, 7 failed** — matches the documented pre-existing MagicMock set (5 `test_datagen_command_generation` + 2 `test_rules_calculations::TestGenerateOutputLocation`). No Phase 4 regression remains. |

**Score:** 8/8 truths verified (after inline gap-closure)

### Required Artifacts

| Artifact                                                            | Expected                                                                                                | Status     | Details                                                                                                                                                                                            |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mlpstorage_py/system_description/sysctl_allowlist.txt`             | Shipped 4-glob allowlist file (vm.dirty_*, net.core.*, net.ipv4.tcp_*, kernel.numa_balancing)            | ✓ VERIFIED | 9 lines including 3 comment lines + 1 blank + 4 globs; load-bearing file for SC #1's "no code change required" claim. Verified.                                                                       |
| `mlpstorage_py/cluster_collector.py::collect_sysctl`                | Module + MPI script twin (Pattern B / D-36)                                                              | ✓ VERIFIED | Module line 1104; MPI script twin line 2116. Both walk /proc/sys, apply allowlist, 8 KiB cap, per-leaf D-2 isolation.                                                                                |
| `mlpstorage_py/cluster_collector.py::_load_sysctl_allowlist`        | Module + MPI script twin (file in module; tuple literal in script)                                       | ✓ VERIFIED | Module line 1066 reads on-disk file; script line 2110 returns regex tuple from baked-in `_SYSCTL_ALLOWLIST_LINES` literal (script can't read package data over SSH).                                  |
| `mlpstorage_py/cluster_collector.py::collect_environment`            | Module + MPI script twin; sorts os.environ; D-26 prefix-or-literal filter; D-23/D-24 redaction dispatch | ✓ VERIFIED | Module line 1210; MPI script twin line 2186. Both filter via `_env_allowlist_match`; dispatch AWS_ACCESS_KEY_ID through `_mask_credential_id` and AWS_SECRET_ACCESS_KEY through `_redact_secret`.    |
| `mlpstorage_py/storage_config.py::_redact_secret` + `_mask_credential_id` | Unified D-25 redactors; legacy `_redact` deleted (Option B)                                          | ✓ VERIFIED | Lines 28-69. `_redact(val)` deleted (grep returns only comment refs); `resolve_object_storage_config()` updated to use new helpers at lines 146-147.                                                |
| `mlpstorage_py/cluster_collector.py::collect_drives`                | Module + MPI script twin; lsblk JSON parse; D-31 4-rule filter chain                                     | ✓ VERIFIED | Module line 1289; MPI script twin line 2224. RM coercion via `rm in (1, True, '1')` (Rule 1 fix for bool variant); virtual NAME/TRAN; unknown-TRAN drop; empty-TRAN-nvme rescue; decimal GB.        |
| `mlpstorage_py/system_description/auto_generator.py::_sysctl_signature`, `_environment_signature`, `_drive_signature` | 3 callable extractors (D-22 + D-34) with `key=repr` defense       | ✓ VERIFIED | Lines 103-155. All use `tuple(sorted(..., key=repr))` shape; drive_signature's `key=repr` is load-bearing (int 500 vs str '' collision). |
| `_FINGERPRINT_KEYS` extended to 11-tuple                             | Phase-3 8-tuple + 3 new (name, extractor) tails                                                          | ✓ VERIFIED | Lines 172-184; verified `len(_FINGERPRINT_KEYS) == 11`.                                                                                                                                              |
| `_EXTRACTOR_SOURCE_KEYS` dispatch map                                | dict mapping extractor name → host-data source key                                                       | ✓ VERIFIED | Lines 193-199; verified `sorted(keys) == ['drives_sig','environment_sig','networking_sig','sysctl_sig']`.                                                                                            |
| `_resolve_fingerprint_key` generalized                               | Body uses `_EXTRACTOR_SOURCE_KEYS[name]` (not hardcoded networking)                                       | ✓ VERIFIED | Line 240: `return extractor(item.get(_EXTRACTOR_SOURCE_KEYS[name], []))`.                                                                                                                            |
| `_splice_stub_lists` D-33 drives-omit branch                         | Conditional: drives present → pass-through; empty/missing → `client.pop("drives", None)`                  | ✓ VERIFIED | Lines 566-577. `_DRIVE_STUB` retained at module scope with Phase-2-legacy comment (line 474-480) but no longer emitted.                                                                              |
| `HostInfo.sysctl`, `HostInfo.environment`, `HostInfo.drives`         | 3 new `List[Dict[str, Any]] = field(default_factory=list)` fields                                        | ✓ VERIFIED | rules/models.py lines 181-183.                                                                                                                                                                      |
| `HostInfo.from_collected_data` reads 3 new keys                       | `data.get('sysctl' / 'environment' / 'drives', [])` + kwarg flow into `cls(...)`                         | ✓ VERIFIED | Lines 261-263, 274-276.                                                                                                                                                                              |
| `node_dict_from_host` 7-key emit shape                                | sysctl + environment shallow-copy pass-through; drives via per-host `group_by_fingerprint`               | ✓ VERIFIED | Lines 441-443. drives uses `group_by_fingerprint(host.drives, ("vendor_name","model_name","interface","capacity_in_GB"), "unit_count")` on lines 414-419.                                              |

### Key Link Verification

| From                                          | To                                              | Via                                           | Status     | Details                                                                                                                                  |
| --------------------------------------------- | ----------------------------------------------- | --------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `_load_sysctl_allowlist`                      | `sysctl_allowlist.txt` (data file)              | `open(_SYSCTL_ALLOWLIST_PATH, 'r')`           | ✓ WIRED    | Module reads on-disk file; loader returns regex tuple. Adding a pattern to the file picks it up without code change (SC #1).             |
| `collect_sysctl` / `collect_environment` / `collect_drives` | `collect_local_system_info`              | per-field try/except block                    | ✓ WIRED    | Lines 1516-1542 — `result['sysctl' / 'environment' / 'drives'] = collect_*()` with D-2 outer envelope.                                    |
| MPI script `collect_sysctl/env/drives` twins   | script's `collect_local_info`                   | per-field try/except                           | ✓ WIRED    | Lines 2384-2408 — Pattern B parallel wiring mirror of module-side.                                                                        |
| `collect_environment`                          | `_mask_credential_id` / `_redact_secret`        | cross-module `from storage_config import ...` | ✓ WIRED    | Line 26: `from mlpstorage_py.storage_config import _mask_credential_id, _redact_secret`.                                                  |
| `collect_local_system_info` result dict        | `HostInfo.from_collected_data`                  | `data.get('sysctl' / 'environment' / 'drives', [])` | ✓ WIRED    | rules/models.py lines 261-263 read all 3 keys.                                                                                            |
| `HostInfo.sysctl/environment/drives`            | `node_dict_from_host` emit                       | `list(host.sysctl)`, `list(host.environment)`, `per_host_drives` | ✓ WIRED    | auto_generator.py lines 441-443.                                                                                                          |
| `node_dict_from_host` `drives` key (empty case) | `_splice_stub_lists` D-33 branch                | `client.pop("drives", None)`                  | ✓ WIRED    | Lines 566-577 + Phase 4-05 live WSL2 smoke confirmation.                                                                                  |
| `_FINGERPRINT_KEYS` + `_EXTRACTOR_SOURCE_KEYS` | `group_by_fingerprint` cross-host stanza splits | `_resolve_fingerprint_key`                    | ✓ WIRED    | Tested live: identical hosts collapse; sysctl/env/drives divergence splits.                                                               |

### Data-Flow Trace (Level 4)

| Artifact                          | Data Variable          | Source                              | Produces Real Data | Status      |
| --------------------------------- | ---------------------- | ----------------------------------- | ------------------ | ----------- |
| `collect_sysctl`                  | `out: List[Dict]`     | `/proc/sys` directory walk           | YES                | ✓ FLOWING   |
| `collect_environment`             | `out: List[Dict]`     | `os.environ` filtered + redacted     | YES                | ✓ FLOWING   |
| `collect_drives`                  | `out: List[Dict]`     | `subprocess.run(['lsblk', '-J', ...])` | YES (returns [] on WSL2 by design — D-31 rule 3 drops all rows with null TRAN; D-33 then omits the key) | ✓ FLOWING (with D-33 omit path) |
| `node_dict_from_host['sysctl']`   | `host.sysctl`         | `HostInfo.from_collected_data` ← collector dict | YES | ✓ FLOWING |
| `node_dict_from_host['environment']` | `host.environment` | same | YES | ✓ FLOWING |
| `node_dict_from_host['drives']`   | `per_host_drives`     | `group_by_fingerprint(host.drives, ...)` | YES | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Allowlist load returns 4 patterns | `python3 -c "from mlpstorage_py.cluster_collector import _load_sysctl_allowlist; print(len(_load_sysctl_allowlist()))"` | `4` | ✓ PASS |
| `collect_sysctl` returns real data on WSL2 | `python3 -c "from mlpstorage_py.cluster_collector import collect_sysctl; print(len(collect_sysctl()))"` | `134` | ✓ PASS |
| `_mask_credential_id` produces D-23 shape | `python3 -c "from mlpstorage_py.storage_config import _mask_credential_id; print(_mask_credential_id('AKIAIOSFODNN7EXAMPLE'))"` | `AKIA****MPLE` | ✓ PASS |
| `_redact_secret` produces D-24 shape | `python3 -c "from mlpstorage_py.storage_config import _redact_secret; print(_redact_secret('wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'))"` | `[SET — 40 chars]` | ✓ PASS |
| `collect_environment` sorts + dispatches AWS to redactors | live test with 5 env vars | All 5 emitted sorted with KEY_ID masked and SECRET length-only | ✓ PASS |
| `collect_drives` returns [] on WSL2 (D-31 rule 3 + D-33) | `python3 -c "from mlpstorage_py.cluster_collector import collect_drives; print(collect_drives())"` | `[]` (all 4 disks have null TRAN, all dropped) | ✓ PASS |
| `node_dict_from_host(HostInfo)` emits 7-key dict | live test | `['chassis','drives','environment','friendly_description','networking','operating_system','sysctl']` | ✓ PASS |
| `_splice_stub_lists` pops empty drives key (D-33) | live test with empty + populated drives | empty → key absent; populated → key present | ✓ PASS |
| `_FINGERPRINT_KEYS` is 11-tuple | `python3 -c "from mlpstorage_py.system_description.auto_generator import _FINGERPRINT_KEYS; print(len(_FINGERPRINT_KEYS))"` | `11` | ✓ PASS |
| Cross-host splits on sysctl/env/drives divergence (D-35) | live `group_by_fingerprint` test | identical → 1 stanza; each of 3 divergences → 2 stanzas | ✓ PASS |
| Phase 4 integration tests green | `pytest tests/integration/test_systemname_yaml_end_to_end.py -q` | `30 passed` (incl 9 TestPhase4EndToEnd) | ✓ PASS |
| Phase 4 scope unit tests green | `pytest tests/unit/test_cluster_collector.py tests/unit/test_auto_generator.py tests/unit/test_storage_config.py tests/unit/test_run_summary.py -q` | `461 passed` | ✓ PASS |
| Full unit suite | `pytest tests/unit -q --ignore=...` | `1849 passed, 8 failed` — **1 new (test_yaml_block_style) + 7 pre-existing** | ✗ FAIL |

### Probe Execution

No probes declared by Phase 4 plans (no `scripts/*/tests/probe-*.sh` convention used in this repo). Skipped.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| COLL-05 | 04-01, 04-04, 04-05 | sysctl[] from /proc/sys curated allowlist | ✓ SATISFIED | sysctl_allowlist.txt + _load_sysctl_allowlist + collect_sysctl (module + MPI twin) + HostInfo.sysctl + node_dict_from_host emit + _sysctl_signature fingerprint. Live: 134 entries on WSL2 dev shell. |
| COLL-06 | 04-02, 04-04, 04-05 | environment[] filtered + AWS redacted | ✓ SATISFIED | D-26 prefix+literal allowlist (BUCKET + AWS_* / STORAGE_* / OMPI_* / UCX_* / NCCL_*); D-23/24 redactors unified in storage_config.py and dispatched in collect_environment; run_summary.py KEY_ID display change confirmed (Plan 04-02 D-25 side effect). |
| COLL-07 | 04-03, 04-04, 04-05 | drives[] from lsblk -J + D-31 filter | ✓ SATISFIED | collect_drives + lsblk -J -b -d invocation + D-31 4-rule chain + D-30 emit shape + D-33 omit-when-empty (auto_generator splice layer) + per-host group_by_fingerprint with unit_count + Yamale-validated absence of media_type/form_factor/performance. Live WSL2 smoke: drives key absent from emitted YAML (D-33 path live-confirmed). |

No orphaned requirements detected. REQUIREMENTS.md lists exactly COLL-05/06/07 for Phase 4, all claimed by plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| (none) | — | — | — | No TBD/FIXME/XXX debt markers in Phase-4-modified files. No empty `return null`/`return []` stubs in production paths (the `[]` returns in `collect_*` are the D-2 universal-failure path, intentional and documented). No hardcoded empty data flowing to rendering. Console.log-only handlers absent (Python codebase). |

### Human Verification Required

No items requiring human verification — all collector behaviors and emit shapes verified programmatically via live spot-checks and unit/integration tests. The fleet-divergence behavior (D-35) is verified at the unit-test level; live multi-host MPI fan-out testing is Phase 5+ territory (covered by LIFE-02 drift testing).

### Gaps Summary

**One blocker:** Phase 4-05 introduces a regression in `tests/unit/test_auto_generator_write.py::test_yaml_block_style`. This test was passing on commit `80e3d17` (the merge-from-main commit immediately before Phase 4 work began) and fails on every commit from `840cbfe` (Plan 04-05 GREEN) onward. Direct git-bisect confirms causation: the new `node_dict_from_host` emit-shape extension produces flow-style `sysctl: []` and `environment: []` for hosts with no allowlist-matching sysctl entries (rare) or no allowlist-matching env vars (common — e.g., the WSL2 dev shell has zero AWS_*/BUCKET/STORAGE_*/OMPI_*/UCX_*/NCCL_* vars set). PyYAML serializes the empty list as flow-style `[]`, which the D-10 contract test rejects.

The Phase 4-04 SUMMARY claimed "1826 passed, 7 failed (all 7 pre-existing)" and Phase 4-05 SUMMARY claimed "1849 passed, 8 pre-existing failures". Both undercount: the pre-existing failure set is 7 (5 datagen MagicMock + 2 rules_calculations MagicMock). The 8th — `test_yaml_block_style` — is NEW from Phase 4. The SUMMARYs misclassified this as pre-existing, which let the regression ship.

**Two resolution paths** (planner picks):

1. **Test contract update (low-risk):** Extend the `test_yaml_block_style` regex to allow `sysctl: []` and `environment: []` alongside `traffic: []`. Analogous to Plan 04-04's Rule 3 contract updates for `test_outer_dict_with_spliced_stubs_yaml_roundtrip` and `test_validator_errors_only_on_blanks` (D-33 contract change). Justification: empty lists ARE valid block-style YAML content; the D-10 spirit is "no flow-style aggregates" and `[]` as a marker token in an otherwise block-styled file is the least-bad form when the alternative is omitting the key entirely.

2. **Production code change (D-33-style omit-when-empty):** Apply a parallel splice branch in `_splice_stub_lists` for `sysctl` and `environment` (pop the key when empty, mirror of the drives branch). This carries a stronger SER-02 signal — "the collector found nothing matching the allowlist on this host; submitter may want to investigate" — and avoids visual noise in the YAML.  However, it diverges from the Plan 04-05 explicit decision "sysctl/environment use shallow-copy pass-through" — the planner intentionally chose the always-present-list shape there. Reverting that would change the Plan 04-05 contract.

Recommendation: **Path 1** (test contract update) preserves the Plan 04-05 production contract while closing the regression. The intent of D-10 was to forbid PyYAML's flow-style mappings/lists for non-empty content; empty `[]` is the standard YAML idiom for "empty sequence" and is not a structural flow-style violation.

The remaining 7 truths PASS — every COLL-05/06/07 requirement is met by code that flows live data end-to-end. Cross-host fingerprint splits, D-33 omit-when-empty for drives, redactor unification, and the 7-key emit shape are all live-verified.

---

_Verified: 2026-06-23T23:18:23Z_
_Verifier: Claude (gsd-verifier)_
