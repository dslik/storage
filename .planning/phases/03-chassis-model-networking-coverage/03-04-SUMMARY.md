---
phase: 03-chassis-model-networking-coverage
plan: 04
subsystem: transform
tags: [python, transform, fingerprint, dispatch, splice]

# Dependency graph
requires:
  - phase: 03-chassis-model-networking-coverage
    plan: 01
    provides: "NetworkPort.state schema lock and _NETWORKING_STUB shape (state: '') — the post-Pydantic dump shape D-17 splice consumes"
  - phase: 03-chassis-model-networking-coverage
    plan: 03
    provides: "collect_networking emit shape (flat list[dict] of {type, speed, state} per real interface) — the input contract _network_signature consumes; down entries with OMITTED speed key drive the .get('speed', '') defense"
provides:
  - "_network_signature(networking) -> tuple — D-22 order-independent multiset of (type, speed, state, unit_count) tuples; key=repr defense for mixed-type comparability"
  - "_resolve_fingerprint_key(item, key) -> Any — D-22 dispatch helper (scalar dotted vs (name, extractor) tuple)"
  - "_FINGERPRINT_KEYS extended from 6-tuple of strs to 8-tuple of (str | (str, callable)): adds chassis.model_name + ('networking_sig', _network_signature)"
  - "group_by_fingerprint: single-line dispatch swap _get_dotted → _resolve_fingerprint_key (existing dotted-only callers preserved)"
  - "_splice_stub_lists: D-17 conditional splice — real networking present → set traffic=[] on each up entry; no networking → Phase 2 stub fallback"
affects:
  - 03-05-integration-host-info-flow

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern: callable-extractor in fingerprint key tuple — generalized fingerprint composition for shapes that aren't scalar dotted paths. The keys tuple holds two element types (str | (name, callable)); a small dispatch helper (_resolve_fingerprint_key) keeps the call site (group_by_fingerprint) one-line clean and locks the dispatch contract in one place."
    - "Pattern: key=repr defense for sorted() over mixed-type tuples — the .get(k, '') universal-failure defense produces tuples whose positional types vary across entries (e.g. up entry's speed=100 vs down entry's speed=''). key=repr converts to deterministic string ordering without losing the native-type values in the resulting tuple (equal multisets still hash to equal sigs)."
    - "Pattern: post-Pydantic D-3 splice seam carried forward — D-17 traffic=[] mutation happens AFTER model_dump(), bypassing the NetworkPort _require_speed_and_traffic_when_up validator which would crash on empty traffic. The dumped dict is the splice site; the YAML serialization carries the seam through to schema_validator at submission time."

key-files:
  created: []
  modified:
    - "mlpstorage_py/system_description/auto_generator.py"
    - "tests/unit/test_auto_generator.py"

key-decisions:
  - "_network_signature sort uses key=repr to tolerate mixed-type fields. The original D-22 verbatim form `sorted(...)` over the raw tuples crashes with `TypeError: '<' not supported between instances of 'str' and 'int'` when an up entry (`speed=100`, int) and a down entry (`speed=''`, the .get default for missing key) appear in the same networking list. key=repr converts each tuple to its repr() string for sort ordering only; the returned tuple itself keeps native int/str types so equal multisets still hash to equal signatures. Deterministic, in-process, no observable side effects."
  - "_FINGERPRINT_KEYS placement: chassis.model_name placed after the existing chassis CPU/memory entries and before operating_system entries (keeps chassis-related keys visually clustered). The ('networking_sig', _network_signature) callable tuple placed at the tail of the 8-tuple."
  - "Helper rename _splice_stub_lists → _splice_blank_networking_fields explicitly NOT done. PLAN's artifacts section locked the decision: diff minimality wins; the function still semantically 'splices stub-shaped blanks into client lists' (the new D-17 splice is a stub-shaped blank — empty traffic list). Docstring updated to mention the D-17 contract; the name stays."
  - "Type annotation of _FINGERPRINT_KEYS widened from `tuple[str, ...]` to bare `tuple`. The widening matches the heterogeneous element type (str | (str, callable)) without introducing typing imports the module doesn't already have. Could be tightened with `tuple[str | tuple[str, Callable], ...]` if the project ever adopts stricter type checking; left as bare `tuple` for now."

patterns-established:
  - "Pattern: bare `tuple` annotation for heterogeneous tuples — when the element type is `str | (name, callable)` and stricter typing would force a `Callable` import + a Union construct, prefer the bare-`tuple` runtime annotation with a docstring comment describing the element-type contract. Pragmatic in a module that does not adopt comprehensive type annotations."

requirements-completed:
  - COLL-04

# Metrics
duration: ~25min
completed: 2026-06-23
---

# Phase 3 Plan 04: Transform-Layer Extensions Summary

**Pure-Python in-memory transform extensions for D-22 cross-host fingerprint and D-17 traffic-blank splice — `_network_signature` (order-independent multiset of `(type, speed, state, unit_count)` tuples with `key=repr` mixed-type defense), `_resolve_fingerprint_key` (scalar dotted vs callable-tuple dispatch), `_FINGERPRINT_KEYS` widened from 6 dotted strings to 8 entries (added `chassis.model_name` + `('networking_sig', _network_signature)`), and `_splice_stub_lists` extended with the D-17 post-Pydantic `traffic=[]` splice on up NIC entries. Zero I/O, zero Pydantic construction; 99-line additive production diff + 320-line additive test diff across 4 new test classes (21 new tests). All 60 auto_generator unit tests, 14 integration tests, and the full 1677-test unit suite (excluding 3 pre-existing dev-env collection errors and 7 pre-existing fixture failures carried from prior plans) green. Plan 03-05 can now wire `HostInfo.networking` + `HostInfo.chassis_model_name` into `node_dict_from_host` and the transform surface this plan ships will consume them via the existing fingerprint pipeline.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-06-23T (current session)
- **Tasks:** 2 (both autonomous, both TDD)
- **Files modified:** 2 (one production, one test)
- **Lines added:** +99 production / +320 test = +419 total
- **Lines removed:** 12 production (group_by_fingerprint dispatch one-liner + _splice_stub_lists loop body, replaced with the conditional fallback)

## Accomplishments

- **D-22 cross-host fingerprint composition shipped verbatim.** `_FINGERPRINT_KEYS` widened from `tuple[str, ...]` (6 entries) to bare `tuple` (8 entries): 4 existing chassis keys, the new `"chassis.model_name"` scalar, 2 existing `operating_system.*` keys, and the new `("networking_sig", _network_signature)` callable-extractor tuple at the tail. Order matches the planner's guidance (chassis cluster first, then OS, then networking).
- **`_network_signature` extractor.** Order-independent multiset of `(type, speed, state, unit_count)` tuples. Two hosts with identical NICs in different listdir order produce equal signatures because the per-entry tuples are sorted before being wrapped into the outer tuple. Uses `.get(k, "")` for every field so pre-grouped (no `unit_count`) and post-grouped (with `unit_count`) entries both work, and down NICs (no `speed` key per Plan 03-03 emit shape) participate without crashing.
- **`_resolve_fingerprint_key` dispatch helper.** Scalar dotted-string keys go through `_get_dotted` (preserving D-5 empty-string-on-miss); `(name, extractor)` callable tuples invoke `extractor(item.get("networking", []))`. Two-line body, one-line docstring. `group_by_fingerprint` updated with a single-line swap from `_get_dotted` to `_resolve_fingerprint_key` at the dispatch site; existing Phase 2 callers passing pure dotted-string tuples behave identically.
- **D-17 `_splice_stub_lists` extension.** When a client already has real networking data (provided by Plan 03-05's eventual `node_dict_from_host` wiring), iterate each entry and set `entry["traffic"] = []` on every entry whose `state == "up"`. Down entries are left untouched (Plan 03-03 emits them without `speed` or `traffic` keys; `model_dump(exclude_none=True)` will produce clean serialization). When the client has no networking (or an empty list), the helper falls back to the Phase 2 behavior: splice in `[dict(_NETWORKING_STUB)]`. Drives branch unchanged from Phase 2.
- **21 new tests across 4 classes:**
  - `TestNetworkSignature` — 5 tests (empty, single up entry, order-independent, down-without-speed, multiple-entries-sorted).
  - `TestResolveFingerprintKey` — 4 tests (scalar dotted, scalar missing, callable, callable with missing networking key).
  - `TestGroupByFingerprintExtended` — 5 tests (Phase 2 regression lock, homogeneous collapse, split on chassis.model_name, split on networking signature, order-independent networking).
  - `TestSpliceUpNicTraffic` — 7 tests (up gets traffic=[], down unchanged, mixed, no-networking fallback, empty-networking fallback, drives stub regression, splice idempotent).
- **Phase 2 regression-lock tests pass by construction.** `test_existing_dotted_only_behavior_unchanged` (group_by_fingerprint) and `test_no_networking_falls_back_to_stub` + `test_existing_phase2_drives_stub_unchanged` (_splice_stub_lists) verify the additive extension preserves every Phase 2 contract.
- **Full Phase 2 auto_generator suite + integration suite green.** 60 unit tests in `test_auto_generator.py` (39 pre-existing + 21 new) all pass. 14 integration tests in `test_systemname_yaml_end_to_end.py` (including the heterogeneous/homogeneous fleet tests that exercise `_FINGERPRINT_KEYS` via the full write path) all pass — the integration fixtures use HostInfo objects whose `chassis_model` and `networking` are at their dataclass defaults (`""` and `[]` respectively, since Plan 03-05 has not yet wired the new fields), so the new `_FINGERPRINT_KEYS` entries resolve to `""` and `()` for every host in the fixture and grouping behaves identically to Phase 2.
- **Final _FINGERPRINT_KEYS tuple length: 8** (6 Phase 2 entries + chassis.model_name scalar + networking_sig callable). Matches the PLAN's `<output>` requirement.

## Task Commits

1. **Task 1: Test-impact scan + RED tests for _network_signature + _resolve_fingerprint_key + extended _splice_stub_lists (D-22, D-17)** — `d0adeac` (test)
2. **Task 2: Implement _network_signature + _resolve_fingerprint_key + extend _FINGERPRINT_KEYS + extend _splice_stub_lists (D-22, D-17)** — `163ee2e` (feat)

## Files Created/Modified

- `mlpstorage_py/system_description/auto_generator.py` — **+99 / -12 lines.**
  - Added `_network_signature` as a module-scope function BEFORE `_FINGERPRINT_KEYS` (because the keys tuple references it as the second tuple-element extractor).
  - Extended `_FINGERPRINT_KEYS` from 6-tuple to 8-tuple; widened annotation from `tuple[str, ...]` to bare `tuple`; updated docstring to mention D-22.
  - Added `_resolve_fingerprint_key` helper between `_get_dotted` and `group_by_fingerprint`.
  - One-line swap in `group_by_fingerprint`: `_get_dotted(item, k)` → `_resolve_fingerprint_key(item, k)`. Docstring's "Dotted keys ... resolved by `_get_dotted`" line replaced with the D-22 callable-extractor description.
  - Replaced `_splice_stub_lists` per-client body with the D-17 conditional fallback: real networking present → iterate and set `traffic=[]` on up entries; no networking → Phase 2 stub branch. Docstring updated with D-17 contract.

- `tests/unit/test_auto_generator.py` — **+320 / -0 lines.**
  - New import block for `_network_signature` and `_resolve_fingerprint_key` (collection-time ImportError pre-Task-2; passes post-Task-2).
  - Four new test classes (`TestNetworkSignature`, `TestResolveFingerprintKey`, `TestGroupByFingerprintExtended`, `TestSpliceUpNicTraffic`) plus a `_extended_host_dict` helper for the cross-host grouping tests.

## Phase 2 test impact

PLAN's Step 1 test-impact scan: `grep -rn "_FINGERPRINT_KEYS|group_by_fingerprint|_splice_stub_lists|_NETWORKING_STUB|node_dict_from_host" tests/` returned hits in three test files:

- `tests/unit/test_auto_generator.py` — primary surface; extended in this plan.
- `tests/integration/test_systemname_yaml_end_to_end.py` — exercises the full write path through `_FINGERPRINT_KEYS` indirectly. No fixture changes needed: the test fixtures use HostInfo objects whose `chassis_model` and `networking` are at their dataclass defaults, so the new `_FINGERPRINT_KEYS` entries resolve to `""` and `()` for every host and grouping behaves identically to Phase 2 (14 tests green without modification).
- `tests/unit/test_cluster_collector.py:1929` — a comment-only reference inside Plan 03-03's docstring (`# Plan 03-04's group_by_fingerprint then ...`). Not a code reference; no impact.

## Decisions Made

- **`_network_signature` sort uses `key=repr` to tolerate mixed-type fields.** This was discovered as a Rule 1 bug during GREEN verification — `test_extended_keys_split_on_networking_signature_difference` failed with `TypeError: '<' not supported between instances of 'str' and 'int'` because the up entry's `speed=100` (int) and the down entry's `speed=""` (the `.get` default for the missing-key case from Plan 03-03's emit shape) ended up in the same `sorted(...)` call. The `key=repr` fix converts each tuple to its `repr()` string for sort ordering only; the returned tuple itself keeps native int/str types so equal multisets still hash to equal signatures. Fully deterministic and in-process. Folded into the GREEN commit as a single one-keyword addition to the existing `sorted(...)` call.
- **`_FINGERPRINT_KEYS` annotation widened from `tuple[str, ...]` to bare `tuple`.** The 8-tuple now holds two element types (`str` | `(str, callable)`); a `Callable` typing import is not needed at runtime, and the project does not adopt comprehensive type annotations elsewhere. The PLAN explicitly allowed either form ("keep `tuple` for runtime, narrow with a comment").
- **Helper not renamed.** Per PLAN's artifacts section: "Helper rename `_splice_stub_lists` → `_splice_blank_networking_fields` is SUGGESTED in CONTEXT.md and RESEARCH but explicitly OPTIONAL. Decision (planner's call per CONTEXT.md): keep the current name — diff minimality wins; the function semantically still 'splices stub-shaped blanks into client lists' and the new D-17 splice is a stub-shaped blank (empty traffic list)." Honored verbatim — the function name `_splice_stub_lists` retained; docstring updated to mention the D-17 contract.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `_network_signature` sort raised `TypeError` on mixed-type tuples**

- **Found during:** Task 2 GREEN verification — `test_extended_keys_split_on_networking_signature_difference` failed with `TypeError: '<' not supported between instances of 'str' and 'int'`.
- **Issue:** PLAN's verbatim D-22 form `tuple(sorted((e.get(k,'') for k in (...)) for e in networking))` crashes when up entries (with `speed=100`, int) and down entries (with `speed=""`, the `.get` default for the missing-key case per Plan 03-03's emit shape) appear in the same networking list. Python 3 forbids `<` comparison between `int` and `str` across tuple positions.
- **Root cause:** PLAN's D-22 verbatim code assumed all entries had all keys with consistent types (which is true POST per-host grouping in Plan 03-05's planned `group_by_fingerprint(host.networking, ("type","speed","state"), "unit_count")` pass — every entry would have an `int` unit_count). But the `.get(..., "")` defense PLAN explicitly required for pre-grouped/down-NIC robustness creates mixed types at this layer.
- **Fix:** Added `key=repr` to the `sorted(...)` call. Sort key is the repr() string of each tuple (deterministic); returned tuple elements keep native types so signature equality still holds.
- **Files modified:** `mlpstorage_py/system_description/auto_generator.py` (single-line addition in `_network_signature`).
- **Verification:** 21/21 new tests green after the fix; including the previously-failing `test_extended_keys_split_on_networking_signature_difference`.
- **Committed in:** `163ee2e` (Task 2 GREEN commit) — folded in because it's part of making the same implementation work; the fix is the production code Plan 03-04 promised to deliver.

### No Process Deviations

No `git stash` used. No working-tree mutations outside the two atomic commits. The 03-01 process note (prefer `git diff <ref>` and `git show <ref>:<path>` for read-only baseline comparisons) was honored throughout.

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug, one-keyword fix to `sorted()` call) + 0 process deviations.
**Impact on plan:** PLAN's two-commit success criterion honored exactly. The Rule 1 fix did not introduce a new commit — it was an integral part of making the GREEN code actually do what the test asked.

## Issues Encountered

- **Pre-existing dev-env collection errors** carried from prior plans: `tests/unit/test_benchmarks_base.py`, `tests/unit/test_parquet_reader.py`, `tests/unit/test_vdb_modular_fake_backend.py` fail at COLLECTION time with `ModuleNotFoundError` (psutil / pyarrow.parquet / numpy not installed in this dev shell). Resolved via `--ignore=` flags. Out-of-scope per Rule 3 scope boundary.
- **Pre-existing fixture failures** in `test_datagen_command_generation` (5 tests) and `test_rules_calculations` (2 tests): `TypeError: expected string or bytes-like object, got 'MagicMock'` in `mlpstorage_py/rules/utils.py::_check_safe_path_component`. Same root cause documented in prior SUMMARYs (03-01 / 03-02 / 03-03). Their import chains do not touch `auto_generator.py` or `cluster_collector.py` — confirmed by the failures' identity remaining unchanged after this plan's changes. Out-of-scope per Rule 3 scope boundary.

## Surprises

### `key=repr` sort defense found during GREEN verification, not test design

The mixed-type issue surfaced only when `test_extended_keys_split_on_networking_signature_difference` exercised a real-world degraded-fleet scenario (one clean host, one host with a down NIC). The PLAN's `<behavior>` listed this test as the canonical "down NICs distinguish hosts at cross-host level" case but didn't anticipate that the resulting sort would mix `int` and `str` types positionally. The fix (`key=repr`) is straightforward and the test now serves as the regression lock for the issue. A `int`-vs-`str` sort safety net is now part of `_network_signature`'s contract — future field additions need to honor the same defense if they can produce mixed types via `.get(..., "")`.

### Helper rename "explicitly OPTIONAL" honored

The PLAN's artifacts section explicitly weighed in on the helper-rename question and landed on "no rename, diff minimality wins." This was honored. The function name `_splice_stub_lists` remains; the docstring describes both the Phase 2 stub branch and the D-17 splice branch. A future reader confused by the name can read the two-paragraph docstring; the cost is one read, vs. a multi-test rename that would touch the integration tests too.

## Threat Flags

None. Phase 3 Plan 04 is pure-Python in-memory transformation; zero I/O, zero network, zero subprocess. The threat register's three threats (T-3-12 Tampering crafted entries, T-3-13 Info Disclosure, T-3-14 DoS pathological list) all dispositioned `mitigate` / `accept` with no blocking concerns: `_network_signature` only reads four specific keys (extra keys ignored at signature time); no logging or error messages emit from the pure-function transform; the collector caps the networking list at kernel-iface count (typically <100, worst-case <10K).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **COLL-04 transform-side complete.** The cross-host fingerprint extension (`_FINGERPRINT_KEYS` widened + `_resolve_fingerprint_key` dispatch + `_network_signature` extractor) and the D-17 splice (`_splice_stub_lists` extension) are both shipped. The collector-side (Plan 03-03) and the transform-side (Plan 03-04) of COLL-04 are now both green.
- **Plan 03-05 (HostInfo + node_dict_from_host wiring) is now unblocked.** The transform surface this plan delivers consumes a per-client dict whose `networking` field is a flat `list[dict]` of `{type, speed, state, unit_count}` entries; Plan 03-05 owns the `HostInfo.from_collected_data` + `HostInfo.chassis_model_name` + `HostInfo.networking` additions and the corresponding `node_dict_from_host` wiring that populates those fields on the per-client dicts.
- **No deferred-items.md entries created** by this plan.

## Self-Check: PASSED

- `mlpstorage_py/system_description/auto_generator.py`: FOUND (modified — +99 / -12 lines).
- `tests/unit/test_auto_generator.py`: FOUND (modified — +320 / -0 lines).
- Commit `d0adeac`: FOUND — `test(03-04): add failing tests for _network_signature, _resolve_fingerprint_key, _FINGERPRINT_KEYS extension, _splice_stub_lists D-17 splice (D-22, D-17)`.
- Commit `163ee2e`: FOUND — `feat(03-04): _network_signature + _resolve_fingerprint_key + _FINGERPRINT_KEYS extension + _splice_stub_lists D-17 splice (COLL-04, D-22, D-17)`.
- AI-attribution check: `git log -1 --format='%B' | grep -ci "co-authored\|claude\|anthropic"` returns `0` for both commits.
- Module-side surface contract: `python3 -c "from mlpstorage_py.system_description.auto_generator import _network_signature, _resolve_fingerprint_key, _FINGERPRINT_KEYS, _splice_stub_lists, _NETWORKING_STUB; ..."` prints `transform surface ok`.
- `len(_FINGERPRINT_KEYS) == 8`: confirmed.
- `any(isinstance(k, tuple) for k in _FINGERPRINT_KEYS)`: True (the `('networking_sig', _network_signature)` tuple).
- `_network_signature([]) == ()`: True.
- `_network_signature(a) == _network_signature(reversed(a))`: True (order-independent).
- Test counts: 21 new tests across 4 new classes (`TestNetworkSignature` 5, `TestResolveFingerprintKey` 4, `TestGroupByFingerprintExtended` 5, `TestSpliceUpNicTraffic` 7). Plus 1 modified import block.
- Production grep counts: `grep -c "def _network_signature\|def _resolve_fingerprint_key" mlpstorage_py/system_description/auto_generator.py` returns 2; `grep -c "networking_sig" mlpstorage_py/system_description/auto_generator.py` returns 3; `grep -c "chassis.model_name" mlpstorage_py/system_description/auto_generator.py` returns 2; `grep -c "_resolve_fingerprint_key" mlpstorage_py/system_description/auto_generator.py` returns 3 (≥ 2 required); `grep -c 'entry\.get("state")' mlpstorage_py/system_description/auto_generator.py` returns 1 (the D-17 splice inside `_splice_stub_lists`).

---
*Phase: 03-chassis-model-networking-coverage*
*Completed: 2026-06-23*
