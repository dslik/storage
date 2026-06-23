---
phase: 03-chassis-model-networking-coverage
plan: 03
subsystem: collector
tags: [python, sysfs, infiniband, bond, mpi, collector, regex, path-indirection]

# Dependency graph
requires:
  - phase: 03-chassis-model-networking-coverage
    plan: 01
    provides: "NetworkPort.state schema lock — the (type, speed, state) shape collect_networking emits is what NetworkPort consumes downstream"
  - phase: 03-chassis-model-networking-coverage
    plan: 02
    provides: "Pattern B MPI duplication discipline + chassis insertion-point template (symbols land between parse_os_release and collect_local_info); collect_networking inserts after collect_chassis_model on the same seam"
provides:
  - "_SYSFS_NET_ROOT, _SYSFS_INFINIBAND_ROOT path constants (path-indirection seam for tmp_path tests)"
  - "_VIRTUAL_NAME_PREFIXES tuple + _VIRTUAL_NAME_RE compiled regex (D-18 + D-19 source-of-truth prefix list)"
  - "_OPERSTATE_UP_VALUES frozenset (D-20 permissive mapping)"
  - "_SAFE_IFACE_NAME_RE compiled regex (T-3-07 belt-and-suspenders name whitelist)"
  - "_read_sysfs_text / _read_sysfs_int — single-line sysfs readers with 8KB cap and blank-on-failure"
  - "_is_virtual_by_name / _is_bridge_master / _is_vlan_subif / _is_bond_slave / _is_bond_master — D-18 filter predicates"
  - "_bond_aggregate_speed_mbps — sum-of-active-slave aggregation (bond's own speed ignored per Pitfall 4)"
  - "_map_operstate (D-20) / _parse_ib_state (D-19, '4:' prefix) / _parse_ib_rate (D-19, int(rate.split()[0]); None on parse failure)"
  - "collect_networking(net_root, ib_root) -> list[dict] — public entry point; ethernet walk + IB walk; per-iface D-2"
  - "result['networking'] key wired into collect_local_system_info"
  - "Inline duplicates of all the above in MPI_COLLECTOR_SCRIPT (Pattern B); parallel result['networking'] in MPI's collect_local_info"
affects:
  - 03-04-transform-fingerprint-extension
  - 03-05-integration-host-info-flow

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern A — per-field sysfs read with universal D-2 collection-failure rule, applied at per-iface scope (a single bad iface skips that one entry; the function returns whatever did succeed)"
    - "Pattern B — MPI script ↔ module duplication discipline; parity test (TestNetworkingMPIScriptParity) locks behavioral equivalence on a tmp_path ethernet+IB fixture"
    - "Pattern D — tmp_path + path-indirection function parameters (default to production sysfs constants) for sysfs unit tests, avoiding builtins.open mock_open which corrupts PyYAML"
    - "Compiled-regex D-18 filter — single re.compile() at module load for both the virtual-iface name match and the T-3-07 iface-name whitelist; the literal prefix tuple is preserved alongside so grep finds the D-18 source-of-truth list"

key-files:
  created: []
  modified:
    - "mlpstorage_py/cluster_collector.py"
    - "tests/unit/test_cluster_collector.py"

key-decisions:
  - "D-18 _is_virtual_by_name shipped as a single compiled regex (^(lo|docker[0-9]*|virbr[0-9]*|veth.*|tun[0-9]*|tap[0-9]*|gre[0-9]*|wg[0-9]*|ib[0-9]*|iboeth[0-9]*|ib_eth[0-9]*)$) rather than the per-prefix string-method chain the PLAN's Step 2 offered as an alternative. Rationale: one compiled regex is the canonical Python idiom for this kind of closed-list whole-name match; the literal _VIRTUAL_NAME_PREFIXES tuple is preserved alongside as a docstring/grep anchor so the D-18 source-of-truth list is still text-searchable. The chain form has a subtle correctness pitfall (`eth0` accidentally matching `eth` startswith were `eth` ever added to the list) that the anchored regex sidesteps for free."
  - "Down-NIC emission shape is `{type, state:'down'}` with the speed key OMITTED (not set to None or empty string). Rationale: Pydantic's `model_dump(exclude_none=True)` will drop a None speed during the downstream emit; emitting the dict with no speed key from the collector means the splice path in Plan 03-04 handles up vs down identically. Same shape used for IB down ports."
  - "Unparseable IB rate (rate file empty/garbled but state=ACTIVE) emits as `{type:'infiniband', state:'down'}` — Pitfall 8/T-3-08 mitigation. Cannot truthfully claim 'up' state without a speed since downstream NetworkPort.state='up' requires speed via the Plan 03-01 _require_speed_and_traffic_when_up validator."
  - "T-3-07 belt-and-suspenders: every iface/dev/port/slave name is validated against `^[A-Za-z0-9._-]+$` before being joined into a sysfs path. Kernel-side names are basename-only and don't contain path separators in any production case, but the defense costs nothing and catches the threat-model entry verbatim."
  - "VLAN sub-iface detector returns False when either iflink or ifindex is unreadable (rather than False-positive-skipping the iface). Rationale: stale -1 sentinels on both reads would otherwise collapse to 'equal' and we'd silently drop a real NIC. Erring toward inclusion matches the universal D-2 rule (failure on a metadata read does not vanish the iface; the next read failure will demote it via the standard path)."
  - "MPI script duplicates use untyped form (no Optional[int], no frozenset[str] subscript) so the script survives on Python 3.8 hosts in heterogeneous SSH-fan-out fleets — same convention as Plan 03-02's chassis additions."

patterns-established:
  - "Pattern: D-18 belt-and-suspenders defense layering — name-prefix shortcut (cheap O(1) regex) first, then per-iface sysfs property checks (bridge → vlan → bond-slave → bond-master) before the universal-rule per-field reads. Each defense is independent; any one catching the offender short-circuits the rest."

requirements-completed:
  - COLL-04

# Metrics
duration: ~20min
completed: 2026-06-23
---

# Phase 3 Plan 03: Networking Collector Summary

**`collect_networking` walks `/sys/class/net/*` and `/sys/class/infiniband/*` to emit a flat per-host list of `{type, speed, state}` dicts, applying D-18 interface filtering (regex prefix + bridge/VLAN/bond-slave property checks), bond-master aggregation (sum of active-slave Mbps, ignoring the bond's own unreliable speed file per Pitfall 4), D-20 operstate mapping (`up`|`unknown`→`up`) with effective-state demotion for Pitfall 2 virtio NICs (`operstate=up` AND `speed in {-1,0}` → `state=down`), and D-19 IB-first port-per-entry reporting. Pattern B inline duplication into `MPI_COLLECTOR_SCRIPT` ships in lockstep, parity-locked by `TestNetworkingMPIScriptParity` — the networking side of COLL-04 lands as a self-contained sysfs slice that Plan 03-04 will consume via `_network_signature` and Plan 03-05 will wire into `HostInfo.networking`.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-06-23T02:23:07Z
- **Completed:** 2026-06-23T02:43:35Z
- **Tasks:** 2 (both autonomous, both TDD)
- **Files modified:** 2 (one production, one test)
- **Lines added:** +569 production / +449 test = +1018 total

## Accomplishments

- **D-18 filter scope shipped verbatim:** lo, docker*, virbr*, veth*, tun*, tap*, gre*, wg*, ib*, iboeth*, ib_eth* via the single compiled `_VIRTUAL_NAME_RE` regex (whole-name anchored); bridge masters via `/sys/class/net/<iface>/bridge/` subdir presence; VLAN sub-interfaces via `iflink != ifindex` mismatch; bond slaves via `<iface>/master` symlink → bond* basename. The literal `_VIRTUAL_NAME_PREFIXES` tuple is preserved as a grep anchor so D-18's source-of-truth prefix list stays text-searchable.
- **D-18 bond master aggregation:** `_bond_aggregate_speed_mbps` reads `<bond>/bonding/slaves`, sums only positive slave speeds (slaves with `speed=-1` contribute zero), and emits ONE entry per LAG with `speed = total_mbps // 1000`. The bond's own `/sys/class/net/<bond>/speed` is ignored per Pitfall 4 (unreliable on many drivers). All-slaves-down → aggregate=0 → bond emits as `{type:ethernet, state:down}` (visible degraded LAG to the submitter).
- **D-19 IB-first reporting:** port-per-entry walk of `/sys/class/infiniband/<dev>/ports/<port>/{state,rate}`. `state` parsed via Pitfall 8 robust `'4:'` prefix match (only `'4: ACTIVE'` is up). `rate` parsed via `int(rate.split()[0])`; on parse failure `_parse_ib_rate` returns None and the port demotes to `{type:infiniband, state:down}` — cannot truthfully claim 'up' without a speed. Missing `/sys/class/infiniband/` directory is a clean no-op via the `os.path.isdir(ib_root)` guard.
- **D-20 operstate mapping + effective-state demotion:** `_map_operstate` returns 'up' iff input is in `{'up','unknown'}` (permissive Linux convention — virtio drivers don't update operstate properly); everything else (down, dormant, notpresent, lowerlayerdown, testing) → 'down'. Effective-state demotion: when mapped state is 'up' AND `speed in {-1, 0}`, force state='down'. The Pitfall 2 virtio NIC lock — no invalid `{speed:-1, state:up}` pair ever leaves the collector.
- **Per-iface D-2 defense (universal-failure rule at iface scope):** every iface and every IB port read is wrapped in a try/except that silently skips that one entry on any exception. A hot-unplugged NIC that vanishes between `os.listdir` and `open(<iface>/type)` does not abort the walk; the function returns whatever entries did succeed. Locked by `TestNetworkingHotUnplug::test_iface_disappears_between_listdir_and_read_skipped`.
- **Pattern B (MPI script ↔ module duplication) honored:** every new module-scope symbol (10 helpers + 5 constants + the public `collect_networking`) is duplicated inline in `MPI_COLLECTOR_SCRIPT` in untyped form. `import re` added to the script's import block. The MPI worker script remains self-contained for SSH fan-out on heterogeneous Python fleets.
- **Pattern B drift is test-locked:** `TestNetworkingMPIScriptParity::test_networking_functions_match_module` builds a tmp_path ethernet+IB fixture, exec's the script in a fresh namespace under broad `BaseException` swallow, and asserts `ns['collect_networking'](str(net), str(ib)) == collect_networking(str(net), str(ib))`. Any future drift between the two copies fails the parity test loudly.
- **T-3-07 belt-and-suspenders:** every iface / dev / port / slave name is validated against `_SAFE_IFACE_NAME_RE` (`^[A-Za-z0-9._-]+$`) before being joined into a sysfs path. POSIX device names never violate this; the defense costs nothing and catches the threat-model entry verbatim.
- **173 tests green in `test_cluster_collector.py`** (was 143 before Plan 03-03; 30 added — 10 parametrized prefix filters + 20 logical cases across the 8 new networking classes).
- **No regression in the wider unit suite:** 1656 passed, 4 skipped (with pre-existing dev-env collection-error files ignored per STATE.md Deferred Items: `test_benchmarks_base.py` / `test_parquet_reader.py` / `test_vdb_modular_fake_backend.py`). The same 7 pre-existing `MagicMock` vs `_check_safe_path_component` fixture failures in `test_datagen_command_generation` and `test_rules_calculations` are carried from 02-02 / 03-01 / 03-02; their import chains do not touch `cluster_collector.py`.

## Task Commits

1. **Task 1: Test-impact scan + RED tests for collect_networking (D-18, D-19, D-20)** — `f744584` (test)
2. **Task 2: Implement collect_networking sysfs + IB walk + wire into collect_local_system_info AND MPI_COLLECTOR_SCRIPT (D-18, D-19, D-20, COLL-04)** — `d52090c` (feat)

## Files Created/Modified

- `mlpstorage_py/cluster_collector.py` — **+569 / -0 lines**.
  - `import re` added to the top-level import block (line 11).
  - `import re` added to the MPI script's inline import block (line 1291).
  - New module-scope section "Networking Collection (sysfs + InfiniBand) — Phase 3 Plan 03" between `collect_chassis_model` and the existing "Local System Information Collection" section header, holding the 5 constants + 10 helpers + public `collect_networking`.
  - Inside `collect_local_system_info`: new try/except block after the chassis_model block, before the vmstat block, that sets `result['networking']` and (on exception) `result['errors']['networking']`.
  - Inside `MPI_COLLECTOR_SCRIPT` string literal: inline duplicates of the 5 constants + 10 helpers + `collect_networking` (untyped form) between the existing `collect_chassis_model` def and `collect_local_info` def; parallel networking try/except inside `collect_local_info` after its chassis block.

- `tests/unit/test_cluster_collector.py` — **+449 / -0 lines**.
  - Module-level helpers `_make_iface(net_dir, name, *, type_val, operstate, speed, bonding_slaves, master_target, bridge)` and `_make_ib_port(ib_dir, dev, port, *, state, rate)` follow RESEARCH 763-851 verbatim shape (Pattern D — tmp_path, NOT mock_open).
  - **TestNetworkingFilters** — 14 tests (10 parametrized prefix-name cases + bridge + VLAN + bond-slave + ib-prefix exclusion).
  - **TestNetworkingBond** — 3 tests (aggregate speed, all-slaves-down, single-active-slave).
  - **TestNetworkingOperstate** — 3 tests (up, unknown→up, down).
  - **TestNetworkingEffectiveState** — 2 tests (virtio speed=-1 demotion, speed=0 demotion).
  - **TestNetworkingInfiniband** — 5 tests (active port, down port, dual-port HCA, no-ib-root, unparseable rate).
  - **TestNetworkingHotUnplug** — 1 test (FileNotFoundError on read after listdir is silently skipped).
  - **TestNetworkingIntegration** — 1 test (`collect_local_system_info` result includes `networking` key as `list`).
  - **TestNetworkingMPIScriptParity** — 1 test (exec MPI script, build tmp_path eth+IB fixture, assert behavioral equivalence with module-level collect_networking).

## Dev-host networking literal value

Single entry: `{'type': 'ethernet', 'speed': 10, 'state': 'up'}`.

This dev shell is WSL2 (Linux 6.18.33.1-microsoft-standard-WSL2). The virtual eth0 surfaced by WSL2's vNIC reports `operstate=up` and `speed=10000` Mbps. None of the WSL2 interfaces match the D-18 filter (no docker on this shell, no bridges, no bonds, no IB hardware), so the walk emits a single ethernet entry. `os.path.isdir('/sys/class/infiniband')` is False on WSL2 — the defensive guard short-circuits the IB walk cleanly, exactly as intended for the typical no-IB host.

The MPI parity scratch test produces the same `[{type:ethernet, speed:10, state:up}]` output on the same tmp_path eth-only fixture — behavioral equivalence between module and MPI-script implementations confirmed live.

## Decisions Made

- **`_is_virtual_by_name` shipped as a single compiled regex** (anchored, whole-name match) rather than a per-prefix string-method chain. The chain form had a subtle correctness pitfall (`eth0` accidentally matching `eth` startswith were `eth` ever added to the list) that the anchored regex sidesteps for free. The literal `_VIRTUAL_NAME_PREFIXES` tuple is preserved as a docstring/grep anchor so the D-18 source-of-truth list is still text-searchable. PLAN's Step 2 explicitly recommended the regex form with this rationale; honored verbatim.
- **Down-NIC emission shape is `{type, state:'down'}` with the `speed` key OMITTED** (not set to None or empty string). Reason: Pydantic `model_dump(exclude_none=True)` drops None speed during emit anyway; emitting no speed key from the collector means the splice path in Plan 03-04 handles up vs down identically. Same shape used for IB down ports.
- **Unparseable IB rate emits as down with no speed key.** A port whose state file says 'ACTIVE' but whose rate file is empty/garbled cannot truthfully claim state='up' downstream because NetworkPort.state='up' requires speed via Plan 03-01's `_require_speed_and_traffic_when_up` validator. Demoting to down is consistent with the D-20 effective-state principle ("state=up means we have positive evidence this NIC works").
- **`_is_vlan_subif` returns False when iflink or ifindex is unreadable** (rather than False-positive-skipping the iface). Stale -1 sentinels on both reads would otherwise collapse to 'equal' and we'd silently drop a real NIC. Erring toward inclusion matches the universal D-2 rule — a failure on a metadata read does not vanish the iface; the next per-iface read failure will demote it through the standard path.
- **T-3-07 belt-and-suspenders: defensive name whitelist `^[A-Za-z0-9._-]+$`** applied to every iface, dev, port, and slave name before any sysfs path construction. Kernel-side names never violate this in practice; the defense is cheap and catches the threat-model entry verbatim. Same pattern would apply if an attacker got kernel-mode write access and crafted a malicious interface name.
- **MPI script duplicates use untyped form** (no `Optional[int]`, no `frozenset[str]` subscript). Same rationale as Plan 03-02's chassis duplicates: heterogeneous SSH-fan-out fleets may run Python 3.8 workers, and subscripted generic frozenset is 3.9+. Compiled `re.compile()` patterns are stdlib-safe inside the script string with no version-compat risk.
- **Defense-in-depth wrapper try/except in `collect_local_system_info`.** `collect_networking` already applies D-2 internally (per-iface scope). The outer wrapper is structurally consistent with the surrounding per-field blocks and provides a clear `result['errors']['networking']` slot if a future bug somewhere lets an exception escape — the universal-rule contract is "key always present, value always list[dict]".

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] _make_iface fixture helper declared `bridge=False` parameter but never implemented the body**
- **Found during:** Task 2 GREEN verification — `test_bridge_master_filtered` failed with `[('ethernet', 10), ('ethernet', 0)]` instead of `[('ethernet', 10)]`.
- **Issue:** The test fixture's signature accepted `bridge=True` but the function body had no corresponding `(d / "bridge").mkdir(...)` line. The br0 fixture in `test_bridge_master_filtered` therefore lacked the `/bridge` subdir the production code looks for, and br0 was being walked as a plain ethernet rather than filtered as a bridge master.
- **Fix:** Added the missing `if bridge: (d / "bridge").mkdir(exist_ok=True)` block to `_make_iface`. Single-block additive change.
- **Files modified:** `tests/unit/test_cluster_collector.py` (fixture helper).
- **Verification:** Re-run flipped the test from RED to GREEN; final test run shows 30/30 GREEN.
- **Committed in:** `d52090c` (Task 2 GREEN commit) — folded in because it's structurally a fixture-setup correction, not a test-of-new-behavior change. The production code is correct (and is what the new test catches the fixture missing); the bug is purely in the test harness.

### No Process Deviations

No `git stash` used during this plan. The system prompt prohibition was honored throughout; no read-only baseline comparisons were needed since this plan adds entirely new symbols rather than modifying existing ones. The only "review" of prior code was via the Read tool against the already-committed module file.

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug, single-block fixture-helper fix) + 0 process deviations.
**Impact on plan:** The auto-fix was necessary to make the existing test exercise the production code path correctly; the production filter is unchanged and is what the corrected fixture now actually tests. PLAN's two-commit success criterion ("Plan 03-03 ships in 2 commits") honored exactly.

## Issues Encountered

- **Pre-existing dev-env gaps (carried from 02-02 / 03-01 / 03-02, documented in STATE.md Deferred Items):** `tests/unit/test_benchmarks_base.py`, `tests/unit/test_parquet_reader.py`, `tests/unit/test_vdb_modular_fake_backend.py` fail at COLLECTION time with `ModuleNotFoundError` (psutil / pyarrow.parquet / numpy not installed in this dev shell). Resolved by `--ignore=` flags in the verification command. Out-of-scope per Rule 3 scope boundary.
- **Pre-existing fixture issues in `test_datagen_command_generation` and `test_rules_calculations`:** 7 failing tests with `TypeError: expected string or bytes-like object, got 'MagicMock'` in `mlpstorage_py/rules/utils.py::_check_safe_path_component`. Same root cause documented in 03-01 / 03-02 SUMMARYs. Their import chains do not touch `cluster_collector.py` — confirmed indirectly via the test result (failures unchanged in count and identity between this plan's verification and 03-02's). Out-of-scope per Rule 3 scope boundary.

## Surprises

### WSL2 dev shell DOES surface a real ethernet NIC

PLAN.md's `<output>` note anticipated "the dev-host networking output (will likely be empty or sparse — note)". The actual output on this WSL2 shell is one ethernet entry: `{'type': 'ethernet', 'speed': 10, 'state': 'up'}`. WSL2 exposes a virtual eth0 with `operstate=up` and `speed=10000` Mbps that passes every D-18 filter and emits cleanly through the up-state path. None of the WSL2 interfaces matched the virtual-prefix filter (no `eth*` is in the D-18 list — `eth` was deliberately omitted because it's the canonical real-NIC name). End-to-end semantic verification (D-2 per-iface universal-rule, D-18 filter, D-20 mapping, IB-walk skip on missing `/sys/class/infiniband`) confirmed live on this shell, not just via mocked tests.

### Compiled regex inside the MPI script string is fine

Plan 03-02's chassis surprises noted that the inline `frozenset({...})` literal worked without any typing import in the script string. Plan 03-03 extends the same observation to `re.compile()`: a compiled-at-script-exec-time regex is stdlib-safe inside the script string with no version-compat risk and no namespace-pollution concerns. The MPI script now contains two compiled regexes (`_VIRTUAL_NAME_RE`, `_SAFE_IFACE_NAME_RE`) and they exec cleanly in the fresh namespace under the parity test's `exec(MPI_COLLECTOR_SCRIPT, ns)`.

### IB-port "unparseable rate" emission shape disambiguated to `state=down` (no speed key)

PLAN's `<behavior>` for Task 1 hedged on whether unparseable IB rate should emit `{type:infiniband, state:down}` or full blank-splice (omit entirely). The decision documented in this SUMMARY's Decisions Made section: emit as down with no speed key. Rationale fits the universal pattern — every other "we found a port but can't fully describe it" case emits a down entry rather than silently dropping the port, and the downstream Plan 03-04 splice path then handles all down entries identically. Locked by `TestNetworkingInfiniband::test_unparseable_rate_demotes_to_down`.

## Threat Flags

None. Phase 3 Plan 03 reads kernel-managed sysfs files with explicit 8KB read caps (`_read_sysfs_text`, `_read_sysfs_int` both apply `f.read(8192)`), defensive iface-name whitelisting (`_SAFE_IFACE_NAME_RE`), and per-iface try/except that prevents any single bad file from crashing the walk. No new network exposure, no shell execution, no third-party packages. The threat register in PLAN.md (T-3-07 tampering, T-3-08 tampering, T-3-09 tampering, T-3-10 DoS, T-3-11 info disclosure, T-3-SC supply chain) is covered with mitigation / accept dispositions and concluded no blocking threats. T-3-07 (crafted iface names) is mitigated by `_SAFE_IFACE_NAME_RE` validation before any path construction; T-3-08 (garbled speed file) is mitigated by `_read_sysfs_int`'s default=-1 + effective-state demotion; T-3-09 (malicious bond slaves file) is mitigated by `_SAFE_IFACE_NAME_RE` per-name validation in `_bond_aggregate_speed_mbps` + 8KB read cap.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **COLL-04 collector-side complete.** Per-host `networking` (list[dict]) flows up through `comm.gather` to rank 0 in the rank-0 collected dict, and through `collect_local_system_info()` on the D-8 fallback path. Plan 03-05 will consume it via two-line `data.get('networking', [])` additions in `HostInfo.from_collected_data` and a corresponding flat field on the `HostInfo` dataclass.
- **Plan 03-04 (transform layer extensions) is now unblocked.** The collector-side shape `{type, speed, state}` is locked; Plan 03-04 owns the per-host `group_by_fingerprint(networking, ("type","speed","state"), "unit_count")` collapse, the `_network_signature` extractor for cross-host fingerprinting, and the D-17 `_splice_blank_networking_fields` post-Pydantic splice for `traffic: []` on up entries. Plan 03-05 then wires `HostInfo.networking` and the integration tests.
- **No deferred-items.md entries created** by this plan.

## Self-Check: PASSED

- `mlpstorage_py/cluster_collector.py`: FOUND (modified — +569 / -0 lines).
- `tests/unit/test_cluster_collector.py`: FOUND (modified — +449 / -0 lines).
- Commit `f744584`: FOUND — `test(03-03): add failing tests for collect_networking sysfs walk + IB walk (D-18, D-19, D-20)`.
- Commit `d52090c`: FOUND — `feat(03-03): collect_networking sysfs + IB walk + MPI script duplication (COLL-04, D-18/D-19/D-20)`.
- AI-attribution check: `git log -1 --format='%B' | grep -ci "co-authored\|claude\|anthropic"` returns `0` for both commits.
- Module-side surface contract: `python3 -c "from mlpstorage_py.cluster_collector import collect_networking, _is_virtual_by_name, _map_operstate, _parse_ib_state, _parse_ib_rate; ..."` prints `networking surface ok`.
- Wire-shape contract: `collect_local_system_info()` returns a `networking` key as a list; on this WSL2 shell the list has one ethernet entry.
- MPI parity contract: exec'd `MPI_COLLECTOR_SCRIPT` in a fresh namespace; `ns['collect_networking']` matches module on a tmp_path eth+IB fixture (single `{type:ethernet, speed:10, state:up}` entry).
- Test counts: `grep -c "TestNetworking" tests/unit/test_cluster_collector.py` returns 9 (8 new test class declarations + 1 reference in fixture helper docstring).
- Production grep counts: `grep -c "^def collect_networking" mlpstorage_py/cluster_collector.py` returns 2 (module + MPI script); `grep -cE "_SYSFS_NET_ROOT|_SYSFS_INFINIBAND_ROOT|_VIRTUAL_NAME_PREFIXES" mlpstorage_py/cluster_collector.py` returns 10 (≥ 4 required); `grep -c "result\['networking'\]" mlpstorage_py/cluster_collector.py` returns 4 (≥ 2 required — module write + module errors-fallback + MPI write + MPI errors-fallback).

---
*Phase: 03-chassis-model-networking-coverage*
*Completed: 2026-06-23*
