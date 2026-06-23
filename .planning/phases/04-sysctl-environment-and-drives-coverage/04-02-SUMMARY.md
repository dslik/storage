---
phase: 04-sysctl-environment-and-drives-coverage
plan: 02
subsystem: cluster_collector + storage_config
tags: [collector, environment, redaction, pattern-b, COLL-06, D-23, D-24, D-25, D-26, D-36]
requires:
  - mlpstorage_py/storage_config.py (legacy `_redact` available for refactor)
  - mlpstorage_py/cluster_collector.py (sysctl block from Plan 04-01 as the insertion-point precedent)
  - MPI_COLLECTOR_SCRIPT Pattern B body (carried from Plans 03-02, 03-03, 04-01)
provides:
  - mlpstorage_py.storage_config._redact_secret
  - mlpstorage_py.storage_config._mask_credential_id
  - mlpstorage_py.cluster_collector._ENV_LITERALS
  - mlpstorage_py.cluster_collector._ENV_PREFIXES
  - mlpstorage_py.cluster_collector._env_allowlist_match
  - mlpstorage_py.cluster_collector.collect_environment
  - MPI_COLLECTOR_SCRIPT inline twins of all four new cluster_collector symbols + both storage_config redactors
  - collect_local_system_info result['environment'] key (always-present list contract)
affects:
  - mlpstorage_py/storage_config.py (legacy `_redact` deleted, two new helpers, callsite updated)
  - mlpstorage_py/cluster_collector.py (module + MPI script body)
  - tests/unit/test_storage_config.py (TestRedactSecret + TestMaskCredentialId + TestRedactBackwardCompat + TestCredentialRedaction.test_access_key_redacted_when_set additive update)
  - tests/unit/test_cluster_collector.py (TestEnvAllowlistMatch + TestEnvironmentCollector + TestEnvironmentMPIScriptParity + TestEnvironmentWiring)
  - tests/unit/test_run_summary.py (TestCredentialDisplay.test_credentials_never_plain_text Rule 3 contract update)
  - .planning/ROADMAP.md (Phase 4 SC #2 reconciled with D-23 / D-24 wording; plan checkbox flipped)
tech-stack:
  added:
    - Cross-module helper import: cluster_collector now imports `_mask_credential_id`, `_redact_secret` from storage_config
  patterns:
    - D-23 first-4/last-4 mask with short-value collapse (<8 chars → "****")
    - D-24 length-only sentinel with set-but-empty branch ("" → "[SET — empty]", deliberate UX improvement)
    - D-25 unified redaction policy across collector + run_summary (single source of truth)
    - D-26 prefix-or-literal allowlist (frozenset literals + tuple prefixes for startswith)
    - D-36 Pattern B MPI script twin discipline (untyped form, manual sync)
key-files:
  created:
    - .planning/phases/04-sysctl-environment-and-drives-coverage/04-02-SUMMARY.md
  modified:
    - mlpstorage_py/storage_config.py
    - mlpstorage_py/cluster_collector.py
    - tests/unit/test_storage_config.py
    - tests/unit/test_cluster_collector.py
    - tests/unit/test_run_summary.py
    - .planning/ROADMAP.md
decisions:
  - "Option B chosen over Option A (the planner offered both). The grep `_redact\\b mlpstorage_py/` returned zero non-self consumers, so deleting `_redact` and updating `resolve_object_storage_config()` to call `_mask_credential_id` and `_redact_secret` directly was the cleanest path. No alias was kept; `TestRedactBackwardCompat` exercises the Option B branch via dynamic import (the Option A pytest.skip branch is dead code but documents the intent, locking the test against a future re-introduction of the alias)."
  - "The KEY_ID dict key in `resolve_object_storage_config()` carries the masked form (`AKIA****MPLE`) instead of the legacy length-only sentinel. This is the deliberate D-25 UX change flagged in 04-CONTEXT.md. `run_summary.py` output for `AWS_ACCESS_KEY_ID:` rows is affected (previously printed `[SET — 20 chars]`; now prints `AKIA****MPLE`). SECRET keeps the length-only sentinel shape per D-24."
  - "MPI script Pattern B duplicates both `_redact_secret` and `_mask_credential_id` inline, since the script body cannot import storage_config (it ships as a string exec'd over SSH on hosts that may not have mlpstorage_py installed). The parity test exec's the script in a namespace and asserts behavioral equivalence on a monkeypatched os.environ snapshot; drift in either redactor shows up as a failed parity assertion. Manual sync between module and script bodies is the load-bearing discipline (same as the sysctl allowlist tuple in Plan 04-01)."
  - "ROADMAP.md SC #2 reconciled in the same docs metadata commit. The old wording referenced a 'length+sha256 fingerprint' for SECRET — never implemented and explicitly dropped in D-24. New wording cites D-23 and D-24 by name so the next verification pass reads the same contract as the implementation."
  - "The output is sorted by `name` rather than emitting `os.environ` insertion order. This is a D-34 fingerprint-stability requirement: two hosts running the same image with the same env config must produce byte-identical environment[] blocks so the fingerprint-grouping step does not split them spuriously."
metrics:
  duration_minutes: ~25
  completed_date: 2026-06-23
  tasks_completed: 2
  files_created: 1
  files_modified: 6
  commits: 2
---

# Phase 04 Plan 02: Environment Collector + Unified Redactors Summary

Environment variable collection (COLL-06) with D-26 prefix-or-literal allowlist + D-23/D-24 credential dispatch through the unified `_mask_credential_id` / `_redact_secret` helpers in `storage_config.py`. Shipped in two-commit RED/GREEN cadence; 31 new tests + 1 updated contract test green, no new regressions, ROADMAP SC #2 reconciled to drop sha256 wording.

## What Shipped

**1. Storage config refactor (Option B)** — `mlpstorage_py/storage_config.py`:

- Legacy `_redact(val)` **deleted** (the grep at refactor time returned zero non-self consumers; Option B was the cleaner path).
- `_redact_secret(val)` — length-only sentinel per D-24:
  - `None` → `"[not set]"`
  - `""` → `"[SET — empty]"` *(deliberate UX improvement vs. legacy "[not set]")*
  - non-empty → `f"[SET — {len(val)} chars]"`
- `_mask_credential_id(val)` — first-4/last-4 mask per D-23:
  - `None` → `"[not set]"`
  - `""` → `"[SET — empty]"`
  - 1..7 chars → `"****"` (collapse — too short to mask)
  - >= 8 chars → `f"{val[:4]}****{val[-4:]}"`
- `resolve_object_storage_config()` updated: `aws_access_key_id_redacted` now flows through `_mask_credential_id` (deliberate D-25 UX change); `aws_secret_access_key_redacted` flows through `_redact_secret`.

**2. Cluster collector additions** — `mlpstorage_py/cluster_collector.py`:

- Import: `from mlpstorage_py.storage_config import _mask_credential_id, _redact_secret`
- `_ENV_LITERALS: Final[frozenset] = frozenset({"BUCKET"})`
- `_ENV_PREFIXES: Final[Tuple[str, ...]] = ("AWS_", "STORAGE_", "OMPI_", "UCX_", "NCCL_")`
- `_env_allowlist_match(name)` — `name in _ENV_LITERALS or name.startswith(_ENV_PREFIXES)`
- `collect_environment() -> List[Dict[str, str]]` — sorts `os.environ.items()`, filters by allowlist, dispatches AWS_ACCESS_KEY_ID through `_mask_credential_id` / AWS_SECRET_ACCESS_KEY through `_redact_secret`, emits everything else verbatim. D-2 envelope (any exception → `[]`).

**3. `collect_local_system_info` wiring** — three-line try/except block immediately after the Plan 04-01 sysctl block, mirroring chassis_model / networking / sysctl exactly:
```python
try:
    result['environment'] = collect_environment()
except Exception as e:
    result['errors']['environment'] = str(e)
    result['environment'] = []
```

**4. Pattern B twins in `MPI_COLLECTOR_SCRIPT`** (D-36):
- Inline `_ENV_LITERALS` (tuple form), `_ENV_PREFIXES`
- Inline `_redact_secret(val)`, `_mask_credential_id(val)` (untyped twins of the storage_config helpers — script can't import storage_config)
- Inline `_env_allowlist_match(name)`, `collect_environment()` (untyped twins)
- Parallel try/except wiring in script's `collect_local_info` after the sysctl block

**5. ROADMAP SC #2 reconciliation** — text diff:
- **From:** `... with AWS_SECRET_ACCESS_KEY redacted as a length+sha256 fingerprint and AWS_ACCESS_KEY_ID rendered as a first-4/last-4 mask matching the policy in storage_config.py.`
- **To:** `... with AWS_SECRET_ACCESS_KEY redacted as a length-only sentinel (per D-24) and AWS_ACCESS_KEY_ID rendered as a first-4/last-4 mask (per D-23) using the unified policy in storage_config.py.`

**6. Tests** — 31 new tests across 5 new classes + 1 updated contract test:

| Class                                                          | Tests | Purpose                                                                                       |
| -------------------------------------------------------------- | ----- | --------------------------------------------------------------------------------------------- |
| `tests/unit/test_storage_config.py::TestRedactSecret`          | 4     | Parametrized branch coverage for D-24                                                         |
| `tests/unit/test_storage_config.py::TestMaskCredentialId`      | 6     | Parametrized branch coverage for D-23 (incl. 7→`****` collapse and 8-char boundary)           |
| `tests/unit/test_storage_config.py::TestRedactBackwardCompat`  | 3     | Locks Option B (alias removed, callsite uses new names); KEY_ID masked / SECRET length-only   |
| `tests/unit/test_cluster_collector.py::TestEnvAllowlistMatch`  | 16    | D-26 prefix-or-literal allowlist (`BUCKET_NAME`, `bucket`, `PATH`, etc.)                      |
| `tests/unit/test_cluster_collector.py::TestEnvironmentCollector` | 10   | Allowlist filter, redactor dispatch, sorted output, D-2 envelope                              |
| `tests/unit/test_cluster_collector.py::TestEnvironmentMPIScriptParity` | 1 | Pattern B (D-36) parity on a monkeypatched os.environ snapshot                                |
| `tests/unit/test_cluster_collector.py::TestEnvironmentWiring`  | 3     | result['environment'] always-present list contract + happy-path + failure-isolation           |
| **Updated** `tests/unit/test_storage_config.py::TestCredentialRedaction::test_access_key_redacted_when_set` | 1 | Asserts new `AKIA****MPLE` masked form instead of legacy `[SET —` marker        |
| **Updated** `tests/unit/test_run_summary.py::TestCredentialDisplay::test_credentials_never_plain_text` | 1 | Sets BOTH credentials; asserts (a) no raw values, (b) masked KEY_ID, (c) length-only SECRET    |

## Two-Commit RED/GREEN Cadence

| Commit  | Type        | Files                                                                                             | Purpose                                                                                                       |
| ------- | ----------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 6c7a71c | test(04-02) | tests/unit/test_storage_config.py, tests/unit/test_cluster_collector.py                          | RED — 27 failing tests across new classes + 16 allowlist match cases. ImportError / AttributeError / AssertionError mix. |
| 4f310cd | feat(04-02) | mlpstorage_py/storage_config.py, mlpstorage_py/cluster_collector.py, tests/unit/test_storage_config.py (existing test update), tests/unit/test_run_summary.py (Rule 3 contract update) | GREEN — D-25 unified redactors + collect_environment module + MPI script twin + wiring + Rule 3 contract updates |

## Why Option B (Not Option A)

The plan offered the executor a choice based on the grep result: Option A keeps `_redact` as an alias for backward compat, only updating the KEY_ID callsite to `_mask_credential_id`; Option B deletes `_redact` and updates the entire callsite to the new names.

The grep `_redact\b mlpstorage_py/ tests/` returned three matches at refactor time:

```
mlpstorage_py/storage_config.py:20:def _redact(val: Optional[str]) -> str:
mlpstorage_py/storage_config.py:104:'aws_access_key_id_redacted': _redact(...)
mlpstorage_py/storage_config.py:105:'aws_secret_access_key_redacted': _redact(...)
```

Three self-references in one file, zero external consumers. Option B (clean delete) was unambiguously the right call — no surface area to preserve, no risk of a downstream consumer breaking. The `TestRedactBackwardCompat` test was designed to tolerate both options at planning time; the Option B branch in `test_redact_alias_kept_or_removed_cleanly` does the actual work (the Option A branch is `pytest.fail`-guarded against `_redact_secret` missing and `pytest.skip`-equivalent on the alias-removed path).

## D-25 Side Effect: `run_summary.py` Output Changes

The deliberate UX change called out in 04-CONTEXT.md D-25 lands here. Pre-Plan-04-02, a `run_summary.py` `AWS_ACCESS_KEY_ID:` row read:

```
AWS_ACCESS_KEY_ID:              [SET — 20 chars]
```

Post-Plan-04-02, the same row reads:

```
AWS_ACCESS_KEY_ID:              AKIA****MPLE
```

The masked form gives an operator enough prefix to recognize *which* credential is configured (useful when juggling multiple AWS accounts) without leaking the secret. SECRET output is unchanged:

```
AWS_SECRET_ACCESS_KEY:          [SET — 40 chars]
```

`tests/unit/test_run_summary.py::TestCredentialDisplay::test_credentials_never_plain_text` was updated in the same commit to validate both shapes (Rule 3 inline fix — the test's contract was written when KEY_ID was length-only; D-25 changes the contract).

## Pattern B Drift Risk

The MPI script body now duplicates SIX symbols from this plan: `_redact_secret`, `_mask_credential_id`, `_env_allowlist_match`, `collect_environment`, plus the `_ENV_LITERALS` and `_ENV_PREFIXES` constants. The parity test (`TestEnvironmentMPIScriptParity`) catches behavioral drift on the same monkeypatched os.environ, but does **not** assert byte-identical function bodies. If a future editor changes one redactor in the module copy but not in the script copy, the parity test will surface the difference; the load-bearing discipline is "when you touch the module copy, also touch the script copy."

The script-side untyped form is significant: the module copy uses `Optional[str]` and `Final[frozenset]`, neither of which are valid Python 3.8 type expressions inside the SSH-shipped script body. The script body must remain valid on heterogeneous Python fleets, so:

- `_redact_secret(val):` (no `Optional[str]` annotation)
- `_mask_credential_id(val):`
- `_ENV_LITERALS = ("BUCKET",)` (tuple in script, `frozenset` in module — both work with the `in` operator)
- `_ENV_PREFIXES = (...)` (tuple in both; `startswith` accepts a tuple)

The string concatenation `"[SET — " + str(len(val)) + " chars]"` is used in the script's `_redact_secret` to avoid f-string parsing surprises inside the triple-quoted script body (no escape gymnastics required).

## Pytest Output Anomalies

Same 7 pre-existing `_check_safe_path_component` MagicMock fixture failures persist out-of-scope per Rule 3:
- `tests/unit/test_datagen_command_generation.py` (5 failures)
- `tests/unit/test_rules_calculations.py::TestGenerateOutputLocation` (2 failures)

Same 3 pre-existing dev-env collection-error files still require `--ignore=` flags (missing `psutil` in dev shell — STATE.md Deferred Items): `test_benchmarks_base.py`, `test_parquet_reader.py`, `test_vdb_modular_fake_backend.py`.

Full unit suite: 1771 passed, 7 failed (all pre-existing), 4 skipped, ~27s.
Targeted new-class run: 31 passed, ~0.1s.

## Verification

```bash
# All redactor + environment-collector + MPI-parity tests pass.
python3 -m pytest tests/unit/test_storage_config.py tests/unit/test_cluster_collector.py \
    -k 'Redact or Mask or Environment or env_allowlist or Credential' -q
# → 31 passed

# Module-side helper smoke.
python3 -c "from mlpstorage_py.storage_config import _redact_secret, _mask_credential_id; \
    print(_mask_credential_id('AKIAIOSFODNN7EXAMPLE'))"
# → AKIA****MPLE

# Collector-side end-to-end smoke.
python3 -c "
from mlpstorage_py.cluster_collector import collect_environment
import os
os.environ['AWS_SECRET_ACCESS_KEY']='secret'
os.environ['AWS_ACCESS_KEY_ID']='AKIAIOSFODNN7EXAMPLE'
os.environ['BUCKET']='my-bucket'
print(collect_environment())
"
# → [{'name': 'AWS_ACCESS_KEY_ID', 'value': 'AKIA****MPLE'},
#    {'name': 'AWS_SECRET_ACCESS_KEY', 'value': '[SET — 6 chars]'},
#    {'name': 'BUCKET', 'value': 'my-bucket'}]

# Two defs of collect_environment (module + script twin).
grep -c 'def collect_environment' mlpstorage_py/cluster_collector.py
# → 2

# One non-comment def each for the two unified helpers.
grep -v '^[[:space:]]*#' mlpstorage_py/storage_config.py | grep -c 'def _redact_secret'
# → 1
grep -v '^[[:space:]]*#' mlpstorage_py/storage_config.py | grep -c 'def _mask_credential_id'
# → 1

# ROADMAP wording reconciled (no sha256 reference, cites D-23 / D-24).
grep -n 'length-only sentinel' .planning/ROADMAP.md
# → 141:  2. ... with AWS_SECRET_ACCESS_KEY redacted as a length-only sentinel (per D-24) ...

# No live `_redact` references in production code (Option B clean delete).
grep -rn "_redact\b" mlpstorage_py/ tests/ 2>/dev/null | grep -v '^.*:[0-9]*:[[:space:]]*#' | grep -v 'docstring\|"""'
# → Only documentation / comment references remain (zero live code refs).
```

All three plan-level success criteria from `<verification>` satisfied. All six plan-level `<success_criteria>` satisfied.

## Deviations from Plan

### Rule 3 — Test contract update for D-25 KEY_ID shape change

**Found during:** Task 2 GREEN verification.

**Issue:** Two pre-existing tests asserted the legacy `[SET —` marker on the `aws_access_key_id_redacted` dict key / `AWS_ACCESS_KEY_ID:` summary row:

- `tests/unit/test_storage_config.py::TestCredentialRedaction::test_access_key_redacted_when_set` — asserted `'[SET —' in config['aws_access_key_id_redacted']` (passed under legacy `_redact`, fails under new `_mask_credential_id` which returns `AKIA****MPLE`).
- `tests/unit/test_run_summary.py::TestCredentialDisplay::test_credentials_never_plain_text` — asserted `'[SET —' in output` after setting `AWS_ACCESS_KEY_ID='secret123'` (legacy: KEY_ID row carried `[SET — 9 chars]`; new: KEY_ID row carries `secr****t123`, no `[SET —` marker because SECRET wasn't set).

**Fix:** Both tests updated in the GREEN commit (4f310cd) to assert the new contract:

- `test_access_key_redacted_when_set` now asserts `config['aws_access_key_id_redacted'] == 'AKIA****MPLE'` (the exact masked form per D-23 + D-25).
- `test_credentials_never_plain_text` now sets BOTH credentials and asserts (a) neither raw value appears in output, (b) the masked `secr****t123` appears, (c) the length-only `[SET —` SECRET sentinel appears. The "no plain text" contract is preserved; the marker shape contract is updated.

**Why Rule 3 (not Rule 1 or Rule 4):** These are blocking-task-completion issues caused directly by the D-25 contract change this plan ships. The plan's must_haves explicitly call out the run_summary.py side effect. The test updates are structurally part of the D-25 refactor (you cannot ship the contract change without updating its contract tests), so they belong in the GREEN commit alongside the production change rather than as a separate hygiene commit.

**Files modified:** tests/unit/test_storage_config.py (1 test body), tests/unit/test_run_summary.py (1 test body).
**Commit:** 4f310cd (same as GREEN production code).

No Rule 1 / Rule 2 / Rule 4 deviations triggered.

## Authentication Gates

None.

## Threat Flags

None — `collect_environment` reads `os.environ` (already in the process's address space, no new trust boundary crossed). Both credential dispatch paths flow through helpers explicitly designed to never emit the raw value. The MPI script duplication does not introduce a new code-injection surface beyond what Plans 03-02, 03-03, and 04-01 already shipped (the script body is a string literal compiled into the module and exec'd via subprocess; no user input flows into the script generation).

## Known Stubs

None — all production code paths shipped in this plan flow data end-to-end. The `result['environment']` key is populated by `collect_environment()` which reads `os.environ` directly; nothing is hardcoded to `[]` or placeholder text. The downstream YAML emit path (Plans 04-04 + 04-05) is the next phase's responsibility but does not block this plan's correctness.

## Forward Notes for Plan 04-03

Plan 04-03 ships the drives collector via `lsblk -J -d -o NAME,MODEL,VENDOR,SIZE,ROTA,TRAN,RM` (COLL-07; D-30/31/32/33/36). Pattern A/B discipline established in 04-01 and 04-02 carries forward:

- New collector lives next to `collect_sysctl` and `collect_environment` in the module.
- Pattern B twin inline in MPI_COLLECTOR_SCRIPT alongside the existing sysctl + environment twins.
- D-2 envelope at two scopes: outer subprocess.run try/except (lsblk absent → []) and per-row filter discipline (D-31).
- D-33 omit-key-when-empty behavior is implemented at the transform layer (Plan 04-04) via the splice-stub branch, NOT at the collector — `collect_drives()` returns `[]` for both "lsblk missing" and "lsblk returned but all rows filtered out" per the universal D-2 rule.

The Pattern B duplication risk in 04-02 (six new symbols inlined) is a useful precedent: when a script-side function needs imports the script can't make (`from storage_config import ...`), duplicating the helpers inline is the only correct path. Plan 04-03's drives collector should NOT need cross-module imports (lsblk parsing is self-contained), so its Pattern B surface should be smaller — only `collect_drives` and any local helpers.

## Self-Check: PASSED

- mlpstorage_py/storage_config.py: `_redact_secret` + `_mask_credential_id` defined (non-comment grep returns 1 each)
- mlpstorage_py/storage_config.py: legacy `_redact` deleted (grep returns only comment / docstring references)
- mlpstorage_py/cluster_collector.py: `_ENV_LITERALS`, `_ENV_PREFIXES`, `_env_allowlist_match`, `collect_environment` all defined; MPI script twins all defined (grep `def collect_environment` returns 2)
- mlpstorage_py/cluster_collector.py: `collect_local_system_info` wired with try/except (mirror of sysctl block)
- tests/unit/test_storage_config.py: 3 new test classes (10 tests) + 1 updated test green
- tests/unit/test_cluster_collector.py: 4 new test classes (30 tests) green
- tests/unit/test_run_summary.py: 1 contract-updated test green
- .planning/ROADMAP.md SC #2: "length-only sentinel" wording present (line 141)
- Commit 6c7a71c (RED): present in `git log --oneline -5`
- Commit 4f310cd (GREEN): present in `git log --oneline -5`
- Manual smoke `collect_environment()` returns sorted, redacted list end-to-end
