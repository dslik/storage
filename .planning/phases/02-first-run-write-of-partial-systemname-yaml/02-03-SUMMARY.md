---
phase: 02-first-run-write-of-partial-systemname-yaml
plan: 03
subsystem: system_description/auto_generator
tags:
  - python
  - pydantic
  - schema-reflection
  - tdd
dependency_graph:
  requires:
    - mlpstorage_py/system_description/auto_generator.py (from 02-02 — provides typing.Any, copy, HostInfo imports and the pure-transformation half)
    - mlpstorage_py/system_description/schema_validator.py:NetworkPort (TEST-only model_fields reflection)
    - mlpstorage_py/system_description/schema_validator.py:DriveInstance (TEST-only model_fields reflection)
  provides:
    - mlpstorage_py.system_description.auto_generator._NETWORKING_STUB (Final[dict] with NetworkPort field-name parity)
    - mlpstorage_py.system_description.auto_generator._DRIVE_STUB (Final[dict] with DriveInstance field-name parity minus optional `performance`)
    - mlpstorage_py.system_description.auto_generator._splice_stub_lists(dump) (in-place idempotent mutator)
    - mlpstorage_py.system_description.auto_generator._build_outer_dict(stanzas) (D-14 outer scaffold)
  affects:
    - 02-04 (write_systemname_yaml atomic orchestrator — composes _splice_stub_lists(_build_outer_dict(sorted_stanzas)) then yaml.safe_dump's)
    - 02-05 (Benchmark.run() hook — calls write_systemname_yaml; transitively depends on this slice's contract)
tech_stack:
  added: []
  patterns:
    - "Final[dict] module constants (sentinel.py convention) — Pydantic-bypass stub literals with TEST-time field-name parity"
    - "dict(_CONSTANT) copy-on-emit — splice helper hands callers fresh dicts so mutation never aliases the module constant"
    - "Defensive `dict.get('k', {}).get('clients', [])` chain — missing keys yield empty iteration, never KeyError"
    - "Schema-aware omission (D-14) — top-level blocks the auto-collector can't construct legally are simply absent from the dict, so schema_validator surfaces them naturally as 'submitter has work to do'"
key_files:
  created: []
  modified:
    - mlpstorage_py/system_description/auto_generator.py
    - tests/unit/test_auto_generator.py
decisions:
  - "D-3 implementation locked: _NETWORKING_STUB carries the exact NetworkPort field-name set (unit_count, type, speed, traffic); _DRIVE_STUB carries the DriveInstance field-name set minus optional `performance` (D-2 row 4). Parity is enforced at test time by reflection (`set(_NETWORKING_STUB.keys()) == set(NetworkPort.model_fields.keys())`) so any future schema field addition fires test_stub_keys_match_pydantic_fields."
  - "D-14 implementation locked: _build_outer_dict returns {'system_under_test': {'clients': stanzas}} — every other top-level block (solution / deployment / product_nodes / product_switches / total_rack_units / rack_power_supplies) is absent. Verified by test_build_outer_dict_omits_solution_deployment. Pitfall 1 honored: zero top-level Pydantic construction is attempted; the schema_validator failure on missing required blocks IS the intended SER-02 UX."
  - "Idempotence as a contractual property: _splice_stub_lists replaces (does not append) on re-call. Two consecutive calls produce len(networking)==1 and len(drives)==1 per client. Test test_splice_stub_lists_idempotent locks this so Plan 02-04 can safely chain `dump = _splice_stub_lists(_build_outer_dict(...))` even after re-grouping or repeated writes (LIFE-01 hook will exercise this)."
  - "Copy-on-emit semantics: each spliced entry is `dict(_NETWORKING_STUB)` / `dict(_DRIVE_STUB)` — a SHALLOW copy. That is intentionally sufficient because the stub values are all immutable (empty strings) except `traffic: []`. A shallow copy aliases the empty-list `traffic` across clients, but since `traffic` is empty and there's no production-code path that mutates it (callers either fill it via human edit or the schema_validator rejects it), the aliasing is harmless and the test still asserts `is not _NETWORKING_STUB`. If Phase 4 ever adds production mutation of `traffic`, switch to `copy.deepcopy(_NETWORKING_STUB)`."
  - "PLAN.md grep gate divergence (documented deviation): the PLAN's acceptance criterion `grep -v '^#' ... | grep -c 'solution|deployment|product_nodes|...'` expects 0, but returns 10. All 10 hits are inside docstrings or indented `# comment` lines — `grep -v '^#'` only filters column-0 hashes and doesn't strip multi-line docstrings. An AST-aware verification (strip docstring spans + inline comments) returns 0 true code hits. The semantic D-14 intent (no production code references to forbidden blocks) is fully honored. Same flavor of divergence as 02-02's quoting-style note."
  - "RED gate observed cleanly: `pytest tests/unit/test_auto_generator.py -x` failed with `ImportError: cannot import name '_DRIVE_STUB'` before any production-code change. GREEN gate: all 39 tests pass after the additive feat commit (118 lines added, 3 modified — module docstring updated to advertise the new symbols and the `from typing import Final` import)."
metrics:
  duration_min: ~18
  tasks_total: 2
  tasks_completed: 2
  files_changed: 2
  commits: 2
  completed_date: 2026-06-19
---

# Phase 02 Plan 03: Stub literals + splice + outer-dict scaffold Summary

`mlpstorage_py.system_description.auto_generator` gains its schema-aware blanks
scaffolding — four new symbols (`_NETWORKING_STUB`, `_DRIVE_STUB`,
`_splice_stub_lists`, `_build_outer_dict`) — so Plan 02-04 can compose
`_splice_stub_lists(_build_outer_dict(sorted_stanzas))` to produce the exact
dict that gets `yaml.safe_dump`'d to disk.

## What Was Built

Slice 3 of Phase 02 — the schema-aware blanks scaffolding of the auto-collector.
Two `Final[dict]` constants, two pure functions, 14 new test cases, two commits.

The deliverable closes the in-memory side of the Phase 02 vertical: real-value
stanzas from 02-02's `node_dict_from_host` + `group_by_fingerprint` are now
composable with stub `networking[]` / `drives[]` lists (D-3 seam, SER-02 visible
to-do reminders) and wrapped in the D-14 outer scaffold that deliberately
OMITS `solution`, `deployment`, and the four optional top-level blocks the
auto-collector cannot legally construct.

What 02-04 now consumes:

```python
from mlpstorage_py.system_description.auto_generator import (
    node_dict_from_host,            # 02-02
    group_by_fingerprint,           # 02-02
    _FINGERPRINT_KEYS,              # 02-02
    _splice_stub_lists,             # 02-03 (this plan)
    _build_outer_dict,              # 02-03 (this plan)
)

per_host_dicts = [node_dict_from_host(h) for h in hosts]
grouped = group_by_fingerprint(per_host_dicts, _FINGERPRINT_KEYS, "quantity")
sorted_stanzas = sorted(grouped, key=lambda s: -s["quantity"])  # D-7 sort owned by 02-04
dump = _splice_stub_lists(_build_outer_dict(sorted_stanzas))
# dump is YAML-ready
```

## Public Symbols Added

| Symbol | Signature | Purpose |
|---|---|---|
| `_NETWORKING_STUB` | `Final[dict]` | `{unit_count: "", type: "", speed: "", traffic: []}` — empty-string stub bypassing NetworkPort's enum / min_length=1 validation |
| `_DRIVE_STUB` | `Final[dict]` | `{unit_count: "", vendor_name: "", model_name: "", interface: "", media_type: "", capacity_in_GB: ""}` — `performance` deliberately OMITTED per D-2 row 4 |
| `_splice_stub_lists(dump)` | `(dict) -> dict` | Inject one stub networking[] + one stub drives[] into every `clients[i]`. Idempotent (replace, not append). Defensive on missing keys. Returns input dict (mutated in place). |
| `_build_outer_dict(stanzas)` | `(list[dict]) -> dict` | `{"system_under_test": {"clients": stanzas}}` per D-14 — solution / deployment / product_nodes / product_switches / total_rack_units / rack_power_supplies all OMITTED |

Symbols arriving in later slices (NOT in this module yet): `_SYSTEMNAME_YAML_MODE`,
`_resolve_host_info_list`, `write_systemname_yaml`, atomic write, FileExistsError
no-op → Plan 02-04. `Benchmark.run()` hook → Plan 02-05.

## File-Conflict Resolution with 02-02 (per PLAN <output>)

02-02 and 02-03 share `mlpstorage_py/system_description/auto_generator.py` but the
symbol sets are fully disjoint:

| Symbol | Slice | Concern |
|---|---|---|
| `_FINGERPRINT_KEYS` | 02-02 | grouping fingerprint (D-4) |
| `_get_dotted` | 02-02 | nested dict walker (D-5) |
| `group_by_fingerprint` | 02-02 | quantity grouping (SER-01) |
| `node_dict_from_host` | 02-02 | HostInfo → dict adapter (COLL-01/02, D-6, D-16) |
| `_NETWORKING_STUB` | **02-03** | NetworkPort stub literal (D-3) |
| `_DRIVE_STUB` | **02-03** | DriveInstance stub literal (D-3, D-2 row 4) |
| `_splice_stub_lists` | **02-03** | post-dump mutator (D-3) |
| `_build_outer_dict` | **02-03** | outer scaffold (D-14) |

Zero edits to 02-02's symbols — the GREEN commit's diff against `auto_generator.py`
is purely additive (118 insertions, 3 modifications — module docstring update to
advertise the new symbols and `from typing import Any, Final` to add `Final`).

## Pydantic Field-Name Reflection — Locked Output

The PLAN's `<output>` block requests the exact `model_fields.keys()` set returned
by `NetworkPort` and `DriveInstance` for future-phase reference:

```
NetworkPort.model_fields.keys() == {'unit_count', 'type', 'speed', 'traffic'}
DriveInstance.model_fields.keys() == {'unit_count', 'vendor_name', 'model_name',
                                       'interface', 'media_type', 'capacity_in_GB',
                                       'performance'}
```

The drift-detection test `test_stub_keys_match_pydantic_fields` asserts:

- `set(_NETWORKING_STUB.keys()) == set(NetworkPort.model_fields.keys())` — exact equality
- `set(_DRIVE_STUB.keys()) == set(DriveInstance.model_fields.keys()) - {"performance"}` — minus the deliberately-omitted optional field

No surprises about Pydantic field reflection. `NetworkPort` and `DriveInstance` both
expose `.model_fields` as a plain dict with the YAML-key field names; the order
returned by `.keys()` matches the source-code declaration order which is also the
stub-literal declaration order (informal, since `set()` doesn't care).

## Tasks

| Task | Name | Commit |
| --- | --- | --- |
| 1+2 (RED) | failing stub + outer-dict tests (D-3, D-14) | `81c627c` |
| 1+2 (GREEN) | stub literals + splice + outer dict scaffold (SER-02 prep) | `e4f4ce2` |

The PLAN.md describes Task 1 (stubs + parity) and Task 2 (splice + outer dict) as
two separate TDD cycles, but the `<success_criteria>` explicitly mandates exactly
TWO commits for Slice 3 ("Slice 3 ships in two commits: `test(02-03): ...` (RED)
→ `feat(02-03): ...` (GREEN)"). Followed the success_criteria — same structural
decision as 02-02 — and the RED gate was nonetheless verifiably observed
(`ImportError: cannot import name '_DRIVE_STUB'` before any production code change).

## Test Count Delta

| Metric | Before | After | Δ |
| --- | --- | --- | --- |
| `tests/unit/test_auto_generator.py` cases | 25 | 39 | +14 |
| Adjacent regression suite (auto_generator + cluster_collector + rules_dataclasses + example_system_descriptions) | 175 | 189 | +14 |

`pytest tests/unit/test_auto_generator.py -v` → 39 passed in 0.10s.
Adjacent regression suite → 189 passed in 3.08s, zero new failures.

## Verification (acceptance criteria from PLAN)

**Task 1 (stubs + parity):**
- `pytest tests/unit/test_auto_generator.py::test_stub_keys_match_pydantic_fields -x` → exit 0.
- `pytest tests/unit/test_auto_generator.py::test_networking_stub_shape -x` → exit 0.
- `pytest tests/unit/test_auto_generator.py::test_drive_stub_shape -x` → exit 0.
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c '_NETWORKING_STUB'` → 4 (definition + 1 splice usage + 2 docstring mentions including a non-`^#` docstring line).
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c '_DRIVE_STUB'` → 3 (definition + 1 splice usage + 1 docstring mention).
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c 'performance'` → 1 (a comment inside `_DRIVE_STUB` indented `# performance: OMITTED...`; `grep -v '^#'` only filters column-0 hashes, so the indented inline comment passes through). The intent (no production code references `performance`) is fully honored — AST-aware verification confirms 0 true code hits.
- Inline Python parity smoke → `D-3 parity ok`.

**Task 2 (splice + outer dict):**
- `pytest tests/unit/test_auto_generator.py -x -v` → exit 0 (all 39 tests green).
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c 'def _splice_stub_lists'` → 1.
- `grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c 'def _build_outer_dict'` → 1.
- D-14 forbidden-token grep (PLAN spec) → 10 hits, but ALL inside docstrings or indented comments; AST-aware code-only grep → 0. See `Deviations` below.
- Inline Python compose+splice smoke → `compose+splice ok`.
- T-2-04 grep gate `grep -v '^#' ... | grep -c 'yaml\.dump[^_(]\|yaml\.load[^_]'` → 0. (No YAML I/O in this module — that's 02-04 territory.)

**Module surface:**
```
python3 -c "from mlpstorage_py.system_description.auto_generator import _NETWORKING_STUB, _DRIVE_STUB, _splice_stub_lists, _build_outer_dict; print('module surface ok')"
```
All four required symbols importable.

## Surprise Discoveries

### 1. The PLAN's D-14 grep gate is over-permissive on docstrings

PLAN acceptance criterion #4 expects
`grep -v '^#' mlpstorage_py/system_description/auto_generator.py | grep -c '"solution"|...'`
to return 0. Actual return is 10 — but every single hit is inside a Python
docstring (lines 42-43 in the module docstring, lines 287-290 in
`_build_outer_dict`'s docstring) or inside an indented `# comment` (lines
304-307 listing the omitted blocks). `grep -v '^#'` only filters lines whose
first character is `#`; indented comments (`    # comment`) and multi-line
docstrings sail through.

The PLAN gate's semantic intent is "no production code references the forbidden
blocks". An AST-aware verification that strips docstring spans and inline-comment
content reports 0 true code hits. Same flavor of grep-vs-AST divergence as
02-02's `'NAME'/'VERSION_ID'` quoting note.

### 2. Shallow `dict(_STUB)` copy is sufficient — for now

`_splice_stub_lists` writes `[dict(_NETWORKING_STUB)]` and `[dict(_DRIVE_STUB)]`
— shallow copies. `_NETWORKING_STUB["traffic"]` is `[]` (an empty list), which
is technically mutable; a shallow copy aliases that empty list across all
spliced clients. In Phase 02 this is harmless because:

- No production code path mutates the empty `traffic` list in place; the human
  fills it via direct edit of the YAML, after which the YAML is re-parsed into
  a fresh dict;
- The schema_validator will reject `traffic == []` (List with `min_length=1`),
  which is the intended SER-02 UX;
- The `is not _NETWORKING_STUB` test assertion still passes — the outer dict
  is a fresh object even though it aliases the empty list.

The decision documented in the frontmatter: if Phase 4 (which will start filling
some of these list stubs from sysfs/lsblk) ever mutates the spliced `traffic`
list in place rather than replacing it, the splice helper must switch to
`copy.deepcopy(_NETWORKING_STUB)`. Filed in `Decisions` for future-phase
awareness.

### 3. `performance` is `Optional[DrivePerformance]` not `Optional[str]`

Cross-checked the schema_validator: `DriveInstance.performance` is
`Optional[DrivePerformance] = None`, where `DrivePerformance` is itself a
nested Pydantic model with four required `int` fields plus two optionals.
The D-2 row 4 decision to OMIT this from `_DRIVE_STUB` is therefore correctly
scoped — even if we wanted to stub it, we'd need a *nested* stub literal, not
a top-level scalar. Leaving it absent matches both D-2 row 4 (optional +
non-derivable) and the principle that schema_validator naturally surfaces
absent optionals as "no work to do here unless you have spec-sheet data."

## Deviations from Plan

### Auto-fixed Issues

**None.** No bugs, no missing critical functionality, no blocking issues
encountered during execution. RED → GREEN flow proceeded exactly as planned.

### Documentation deviations

**1. [Rule 3 - Grep gate vs docstring text]** PLAN acceptance criterion #4
(D-14 forbidden-token grep) returns 10 hits instead of 0.

- **Found during:** Task 2 acceptance-criteria verification.
- **Issue:** The PLAN gate uses `grep -v '^#'` to "filter comments" before
  searching for forbidden tokens. That filter only strips lines beginning with
  `#` at column 0 — it leaves multi-line docstrings AND indented inline
  comments (`    # comment`) intact. The module's `_build_outer_dict` docstring
  enumerates the omitted blocks by name (for human readability), and its body
  has six indented `# Foo: OMITTED` comments restating the same list.
- **Resolution:** Confirmed via AST-aware verification (strip docstring spans
  + inline `#` content per source line) that zero true code references to the
  forbidden tokens exist. The semantic D-14 intent is fully honored.
- **Files modified:** None — the docstring and inline comments are exactly
  what makes the omission intent legible to future maintainers; stripping them
  to satisfy a flawed grep gate would degrade the code's documentation quality.

### Structural deviations

**2. [Plan structure]** The PLAN describes Task 1 (stubs + parity) and Task 2
(splice + outer dict) as two separate TDD cycles, but the `<success_criteria>`
explicitly mandates exactly TWO total commits for Slice 3 (one RED test commit,
one GREEN feat commit).

- **Resolution:** Followed the success_criteria. One RED test commit
  (`81c627c`) covers all 14 new tests across both tasks; one GREEN feat commit
  (`e4f4ce2`) ships both stub literals and both functions in the same module.
  RED gate was nonetheless verifiably observed (`ImportError: cannot import
  name '_DRIVE_STUB'`) before any production code existed.
- **Why this is safe:** The Task 1 and Task 2 tests live in the same file, so
  RED-then-GREEN-then-RED-then-GREEN would have required an intermediate state
  where some tests pass and some fail — a worse signal than a single clean
  RED → GREEN transition. Same structural decision as 02-02.

## Known Stubs

This plan deliberately INTRODUCES `_NETWORKING_STUB` and `_DRIVE_STUB` as
schema-aware blanks, which by the project's definition are stubs. They are
NOT a "Known Stubs" problem because:

- They are the load-bearing artifact of SER-02 ("visible to-do reminders for
  fields the collector can't auto-derive");
- They are explicitly catalogued in `_NETWORKING_STUB` and `_DRIVE_STUB`
  module docstrings, in the PLAN's `<artifacts_this_phase_produces>`, and in
  this Summary's "Public Symbols Added" section;
- They are field-name-parity-locked against the Pydantic schemas, so any
  future schema field addition fires the drift test immediately;
- The future-phase resolution path is clear and explicit:
  - Phase 3 (Plan 03-XX) — auto-fill `networking[]` from sysfs/InfiniBand
  - Phase 4 (Plan 04-XX) — auto-fill `drives[]` from `lsblk` (minus the
    non-derivable spec-sheet fields explicitly listed as out-of-scope in
    PROJECT.md "Out of Scope" → `media_type`, `form_factor`, `performance`)

## Threat Flags

**None new.** The plan introduces zero new I/O surface, zero new parsing
surface, zero new trust boundaries. The two new helper functions are pure
dict mutators operating on already-trusted in-memory dicts produced from
the equally-trusted `HostInfo` representation.

Carried-forward threat IDs from the PLAN's threat_model block:

- **T-2-04 (Tampering, YAML emit):** still owned by Plan 02-04. The
  acceptance gate `grep -v '^#' ... | grep -c 'yaml\.dump[^_(]\|yaml\.load[^_]'`
  on `auto_generator.py` returns 0; the constraint propagates forward. Note
  that the test file does call `yaml.safe_dump` and `yaml.safe_load` in
  `test_outer_dict_with_spliced_stubs_yaml_roundtrip` — both `safe_*`, both
  in test code only, which is the documented exception.
- **T-2-05 (DoS, splice iteration):** dispositioned `accept`. `_splice_stub_lists`
  is O(N) over clients; realistic fleets are O(100). Zero DoS surface.
- **T-2-SC (Package install legitimacy):** dispositioned `accept`. Zero new
  packages — only the existing `copy` and `typing.Any`/`Final` imports.
- **T-2-02 / T-2-01 / T-2-08:** owned by Plan 02-04 (atomic write). Flagged
  for traceability; nothing in this plan affects them.

Block-on: high. No blocking threats in this plan.

## TDD Gate Compliance

- **RED gate:** `81c627c test(02-03): add failing stub + outer-dict tests (D-3, D-14)`.
  Confirmed `ImportError: cannot import name '_DRIVE_STUB' from
  'mlpstorage_py.system_description.auto_generator'` on
  `pytest tests/unit/test_auto_generator.py -x` before any production code
  was added.
- **GREEN gate:** `e4f4ce2 feat(02-03): stub literals + splice + outer dict scaffold (SER-02 prep)`.
  `pytest tests/unit/test_auto_generator.py -v` → 39 passed in 0.10s.
- **REFACTOR gate:** not needed. The module landed at its intended final shape;
  no internal reshuffling required.

Both commits omit any `Co-Authored-By:` AI attribution per
`feedback_no_attribution.md` / MEMORY.md.

## What Plans 02-04 / 02-05 Can Now Do

**Plan 02-04 (atomic write_systemname_yaml)** can now compose:

```python
from mlpstorage_py.system_description.auto_generator import (
    node_dict_from_host,
    group_by_fingerprint,
    _FINGERPRINT_KEYS,
    _splice_stub_lists,
    _build_outer_dict,
)

per_host_dicts = [node_dict_from_host(h) for h in hosts]
grouped = group_by_fingerprint(per_host_dicts, _FINGERPRINT_KEYS, "quantity")
# D-7 sort (largest quantity first) lives here, not in 02-03 — 02-04 owns it.
sorted_stanzas = sorted(grouped, key=lambda s: -s["quantity"])
dump = _splice_stub_lists(_build_outer_dict(sorted_stanzas))
# yaml.safe_dump(dump, ...) under atomic write — 02-04's actual job.
```

**Plan 02-05 (Benchmark.run() hook + integration tests)** can invoke the
end-to-end pipeline at run startup, with the FileExistsError no-op landing
in 02-04 protecting against the LIFE-01 re-run case.

## Self-Check: PASSED

- `mlpstorage_py/system_description/auto_generator.py` exists with all four
  new symbols (_NETWORKING_STUB, _DRIVE_STUB, _splice_stub_lists,
  _build_outer_dict). FOUND.
- `tests/unit/test_auto_generator.py` exists with the 14 new test cases.
  FOUND.
- Commit `81c627c` (RED) present in `git log --oneline`. FOUND.
- Commit `e4f4ce2` (GREEN) present in `git log --oneline`. FOUND.
- Full `pytest tests/unit/test_auto_generator.py -v` → 39 passed. PASSED.
- Adjacent regression suite → 189 passed in 3.08s, zero new failures. PASSED.
- Module surface importable: `_NETWORKING_STUB, _DRIVE_STUB,
  _splice_stub_lists, _build_outer_dict` all import cleanly. PASSED.
- D-3 field-name parity smoke `D-3 parity ok`. PASSED.
- D-14 compose+splice smoke `compose+splice ok`. PASSED.
