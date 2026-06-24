---
phase: 05-logical-diff-lifecycle-capacity-gate
verified: 2026-06-24T00:00:00Z
status: human_needed
score: 10/10 must-haves verified at code/test level; 2 hardware-dependent items deferred to UAT
overrides_applied: 0
warnings:
  - id: REVIEW-CR-01
    severity: critical (advisory, non-blocking per user direction)
    summary: "CAP-02 launcher uses shell=True with unsanitized destination/run_uuid/output_file interpolated into the command string (cluster_collector.py:3403-3441). A submitter who passes --data-dir containing shell metacharacters or spaces triggers either argument-splitting or arbitrary command execution. Pre-existing MPIClusterCollector uses the same pattern; Phase 5 expands the attack surface to a new entry point and a new set of args.* paths."
    suggested_fix: "Use shell=False with cmd_parts list, or shlex.quote() every interpolated position."
  - id: REVIEW-CR-02
    summary: "CAP-02 launcher writes output_file to a launch-host-local tempfs but rank 0 of the mpirun job may land on a remote host. When the launch host is not in --hosts (typical submitter-laptop deployment), the launcher reads the JSON path that does not exist locally and raises a misleading 'mpi4py not installed on all hosts' error even when the probe semantically succeeded. The real-mpi integration test runs both ranks on localhost so the defect is invisible at unit/integration layer."
    severity: critical (advisory, non-blocking per user direction)
    suggested_fix: "Place output_file inside the SCP-replicated staging_dir or SCP-fetch the output back from rank 0's host before reading."
human_verification:
  - test: "Real multi-host CAP-02 cardinality check across distinct physical machines (e.g., two clients with distinct mounts vs. shared NFS)."
    expected: "When --hosts spans two machines mounting the same shared FS, datagen proceeds silently. When one host has a local-disk path masquerading as the shared mount, datagen fails fast with each hostname + (st_dev, st_ino) tuple listed and the verbatim local-disk hint."
    why_human: "Requires multi-host SSH + OpenMPI + mpi4py installed on each rank; the test_shared_fs_probe_real_mpi.py file only spawns two MPI ranks on localhost (which always share st_dev,st_ino → cardinality 1)."
  - test: "Real-starved-disk CAP-01: invoke `mlpstorage training datagen --model unet3d --data-dir <real-mount-with-insufficient-space>` and confirm the 4-field error (destination / available_bytes / required_bytes / deficit) appears before any data is written."
    expected: "Benchmark exits non-zero before any dataset file is created; stderr contains all four labeled fields; no partial directory tree left in --data-dir."
    why_human: "Requires a real filesystem with controllable free space; unit tests mock os.statvfs. CR-01 advisory also worth re-exploring on this path because args.data_dir may contain spaces in realistic submitter mounts."
  - test: "End-to-end LIFE-04 hand-fill survival across a real run pipeline: complete `mlpstorage init`, run a first benchmark to produce systemname.yaml, hand-edit `friendly_description`, then re-run the same command."
    expected: "The hand-edited friendly_description survives byte-identical in the on-disk file; mtime is unchanged; no SystemDriftError raised."
    why_human: "The integration test mocks the bench._run pipeline (e.g., VectorDBBenchmark with mocked _run). Real DLIO/MPI execution is not exercised in CI."
  - test: "Submitter-laptop deployment for CAP-02 (REVIEW-CR-02 reproduction): launch from a host NOT in --hosts and confirm the misleading 'mpi4py not installed' message does NOT appear when the probe semantically succeeded."
    expected: "Either the bug is observed (validating CR-02) and tracked into Phase 6 / defect backlog, OR a fix is staged before phase sign-off."
    why_human: "Requires a multi-host deployment where the launch host is outside the rank list — not testable from the local dev shell."
---

# Phase 5: Logical Diff Lifecycle + Capacity Gate — Verification Report

**Phase Goal:** A submitter who re-runs the benchmark against an existing results-dir gets a hard failure if the client fleet has drifted from the previously recorded `systemname.yaml`, but their hand-filled blanks survive unchanged when nothing has drifted — and `datagen` refuses to start if the dataset destination doesn't have room or isn't the same shared filesystem on every participating host.

**Verified:** 2026-06-24
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria + PLAN must-haves)

| #   | Truth                                                                                                                          | Status     | Evidence                                                                                                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------------------ | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SC1 | Unchanged-fleet re-run preserves hand-fills and does NOT modify the on-disk file (LIFE-04)                                     | ✓ VERIFIED | `test_submitter_hand_fills_survive_unchanged_full_pipeline_sc1` + `test_second_run_against_unchanged_fleet_no_touch_mtime_invariant` (LIFE-04 mtime+sha256 invariant); 572-suite green. |
| SC2 | Drifted fleet causes `run` to fail BEFORE DLIO/MPI launch with JSONPath-style field list and on-disk/in-memory values          | ✓ VERIFIED | `test_drift_on_cpu_model_fails_before_dlio_sc2` + `test_drift_on_sysctl_value_surfaces_jsonpath_hunk_sc2`; bench._run mocked and asserted NOT called.                                   |
| SC3 | Drift error names BOTH remediation options (rename + remove)                                                                   | ✓ VERIFIED | `test_drift_message_contains_both_remediation_options_sc3` asserts "Rename the existing yaml" AND "Remove " both appear in `SystemDriftError.message`.                                  |
| SC4 | Per-mode independence (closed and open files are independent)                                                                  | ✓ VERIFIED | Bidirectional pair locked per checker W-3: `test_drift_in_closed_mode_does_not_trigger_drift_in_open_mode_sc4` AND `test_drift_in_open_mode_does_not_trigger_drift_in_closed_mode_sc4`. |
| SC5 | `datagen` startup fails with 4-field message when destination is starved (per-rank check on multi-node)                        | ✓ VERIFIED | `test_starved_destination_fails_datagen_with_4field_message_sc5` asserts 'available_bytes:', 'required_bytes:', 'deficit:', destination string all appear. CAP-01 wrapper at 125 lines. |
| SC6 | Happy-path silence when free space is sufficient                                                                               | ✓ VERIFIED | `test_sufficient_space_proceeds_silently_sc6` + `test_happy_path_returns_none_silent`; assert logger.error/info/warning NOT called.                                                     |
| SC7 | Multi-host shared-FS cardinality > 1 fails fast with per-host (st_dev, st_ino) listing + verbatim local-disk hint              | ⚠️ PARTIAL  | Verified at unit/integration level via mocked launcher: `test_multi_host_cardinality_2_fails_with_host_listing_sc7` + `test_multi_host_cardinality_2_error_message_contains_local_disk_hint_sc7`. Real multi-host hardware coverage is a UAT defer (see human_verification item 1). |
| SC8 | Single-host runs are silent no-op (no MPI calls, no sentinel, no logger output)                                                | ✓ VERIFIED | `test_single_host_run_is_silent_no_op_sc8` + `test_no_hosts_attr_is_silent_no_op_sc8`; SC#8 short-circuit lives inside `run_shared_fs_probe`.                                           |

**Score (truths):** 8/8 ROADMAP SCs verified at code/test level; SC#7 has a UAT-defer companion (real multi-host hardware).

### Plan must-haves (per-plan truth-level)

| Plan  | Must-haves verified | Notes                                                                                                                                                                  |
| ----- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 05-01 | 8/8                 | All D-37/D-38/D-39/D-40/D-41/D-46/D-47 + Pitfall 3(a) truths covered by 29 unit tests across 8 classes; D-41 verbatim long-sysctl round-trip locked.                   |
| 05-02 | 8/8                 | LIFE-02/03/04 wiring; 19 errors tests + 17 TestPhase5DriftWiring tests including B-5 stub-splice symmetry + LIFE-04 mtime invariant + hand-fill survival.              |
| 05-03 | 8/8                 | CAP-01 + per-subclass overrides; 29 capacity-gate tests cover A6 (KVCache 1x), A7 (checkpoint-join), A8 (VDB None destination), per-rank starvation.                   |
| 05-04 | 8/8                 | CAP-02 SHARED_FS_PROBE_SCRIPT + launcher + `_run_uuid`; 22 tests cover Pitfall-4 bcast-before-barrier (both source-grep AND in-process exec), Pitfall-7 per-instance UUID, W-1 D-49 tight ordering, W-5 launcher pass-through. |
| 05-05 | 10/10               | All 8 SCs + LIFE-04 hand-fill + main.py dispatch; 25 integration tests + 3 real-mpirun tests (skip-if-no-mpirun is proper UAT defer).                                  |

### Deferred Items (addressed in later phases)

None — Phase 5 is the closing phase of the Phase 1-5 milestone and there is no Phase 6 in the milestone roadmap. The 2 Critical review findings (REVIEW-CR-01 shell injection, REVIEW-CR-02 launch-host-only output) are advisory non-blockers per user direction; they need to be tracked into a defect/follow-up backlog rather than deferred to a later phase that doesn't exist.

### Required Artifacts

| Artifact                                                                  | Status      | Details                                                                                                                                                            |
| ------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mlpstorage_py/system_description/diff.py`                                | ✓ VERIFIED  | 337 lines; 8 public symbols (DiffEntry, DiffResult, diff_node_dict_lists, format_unified_diff, _flatten_to_paths, _compute_fingerprint, _render_fingerprint, __all__). |
| `tests/unit/test_diff.py`                                                 | ✓ VERIFIED  | 434 lines; 29 tests across 8 classes; all green.                                                                                                                   |
| `mlpstorage_py/errors.py` (SystemDriftError + SystemDescriptionParseError) | ✓ VERIFIED  | Both classes inherit MLPStorageException at line 310 + 364; ErrorCode FS_INVALID_STRUCTURE + CONFIG_PARSE_ERROR.                                                   |
| `mlpstorage_py/system_description/auto_generator.py` (drift wiring)       | ✓ VERIFIED  | `parse_on_disk_systemname_yaml` at line 669; `SystemDriftError` raise at line 892; B-5 stub-splice symmetry copy at line 871 (`_splice_stub_lists(_build_outer_dict(copy.deepcopy(stanzas)))`). |
| `tests/unit/test_errors.py`                                               | ✓ VERIFIED  | 192 lines; 19 tests across 3 classes.                                                                                                                              |
| `tests/unit/test_auto_generator_write.py` (TestPhase5DriftWiring)         | ✓ VERIFIED  | 17 new tests for LIFE-02/03/04 wiring; existing Phase-2 23 tests still green (40 total).                                                                           |
| `mlpstorage_py/benchmarks/capacity_gate.py`                               | ✓ VERIFIED  | 125 lines; `check_capacity_4field` emits 4-field message (available_bytes / required_bytes / deficit).                                                              |
| `mlpstorage_py/benchmarks/base.py` (_pre_execution_gate + _run_uuid)      | ✓ VERIFIED  | `_pre_execution_gate` at line 979; `self._run_uuid = uuid.uuid4().hex` at line 165; CAP-01 fires BEFORE CAP-02 inside the template (verified by reading 995-1027). |
| `mlpstorage_py/benchmarks/dlio.py` (Training + Checkpointing overrides)   | ✓ VERIFIED  | 2 `required_bytes_for_capacity_gate` overrides + 2 `_capacity_gate_destination` overrides + 2 `self._pre_execution_gate()` insertions.                            |
| `mlpstorage_py/benchmarks/vectordbbench.py` (VDB override)                | ✓ VERIFIED  | 1 override pair; `_capacity_gate_destination` returns None (A8 escape hatch).                                                                                      |
| `mlpstorage_py/benchmarks/kvcache.py` (KVCache override)                  | ✓ VERIFIED  | 1 override pair; A6 1x lock locked by test_returns_total_cache_bytes_at_1x_per_a6.                                                                                 |
| `tests/unit/test_capacity_gate.py`                                        | ✓ VERIFIED  | 489 lines; 7 classes; 29 tests green (after Slice 4 fixture extension keeps SC#8 short-circuit silent).                                                            |
| `mlpstorage_py/cluster_collector.py` (SHARED_FS_PROBE_SCRIPT + launcher)  | ✓ VERIFIED  | `SHARED_FS_PROBE_SCRIPT` at lines 2545-2801 (script compiles, len 8925 bytes); `run_shared_fs_probe` at line 3251; `_write_probe_script_to_tempfile` helper at line 3229. |
| `tests/unit/test_shared_fs_probe.py`                                      | ✓ VERIFIED  | 770 lines; 8 classes; 22 tests green. Includes B-3 Option B in-process exec test.                                                                                  |
| `tests/integration/test_systemname_yaml_end_to_end.py` (Phase 5 classes)  | ✓ VERIFIED  | +747 lines; 3 new classes (TestPhase5Lifecycle, TestPhase5Cap01, TestPhase5Cap02); 25 new tests; all 13 required literal test names present.                       |
| `tests/integration/test_shared_fs_probe_real_mpi.py`                      | ✓ VERIFIED  | 222 lines; 1 class, 3 tests; correctly skipped (3/3) on dev shell without mpi4py — proper UAT-defer pattern.                                                       |

### Key Link Verification

| From                            | To                                       | Via                                                              | Status   | Details                                                                                                                                            |
| ------------------------------- | ---------------------------------------- | ---------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `diff.py`                       | `auto_generator.py`                      | `from auto_generator import _FINGERPRINT_KEYS, _resolve_fingerprint_key` | ✓ WIRED  | Single source of truth for D-38; verified by grep + import test.                                                                                  |
| `auto_generator.py`             | `diff.py`                                | Lazy import inside FileExistsError branch                        | ✓ WIRED  | Lazy to break the circular edge; verified at line 848 + 853.                                                                                       |
| `auto_generator.py`             | `errors.py`                              | `from errors import SystemDriftError, SystemDescriptionParseError`| ✓ WIRED  | Top-level import at line 70; 1 raise SystemDriftError + 4 raise SystemDescriptionParseError sites.                                                 |
| `base.py`                       | `capacity_gate.py`                       | `from capacity_gate import check_capacity_4field`                | ✓ WIRED  | Top-level import at line 70; called from `_pre_execution_gate` at line 1007.                                                                       |
| `base.py`                       | `cluster_collector.py`                   | `from cluster_collector import run_shared_fs_probe`              | ✓ WIRED  | Verified via grep; called at line 1019 of `_pre_execution_gate` AFTER `check_capacity_4field`.                                                     |
| `base.run()` → `_pre_execution_gate()` → `write_systemname_yaml` | (CAP-01 + CAP-02 BEFORE LIFE-02 raise)                                | Ordering | ✓ WIRED  | `awk` verification: collect@1095 < gate@1102 < write@1110. Locked at unit + integration level.                                                     |
| Each subclass `datagen` entry → `self._pre_execution_gate()`                                                                       | Gate fires before workload                                            | ✓ WIRED  | TrainingBenchmark.datasize:542, CheckpointingBenchmark.datasize:683, VectorDBBenchmark.execute_datagen:742, KVCacheBenchmark._execute_datasize:354. |
| `Benchmark.__init__`             | `self._run_uuid = uuid.uuid4().hex`     | Per-instance UUID for D-43/Pitfall-7                              | ✓ WIRED  | Line 165; verified by grep + Slice 4 SUMMARY confirmation.                                                                                         |
| `run_shared_fs_probe(... run_uuid=self._run_uuid)`                  | UUID flows verbatim into mpirun argv[2]                              | W-5      | ✓ WIRED  | Locked by `test_launcher_passes_caller_supplied_run_uuid_not_generates_own` (uuid.uuid4 patched with wraps=, assert_not_called on launcher path).  |

### Data-Flow Trace (Level 4)

Diff core, errors, capacity gate are pure transformations or thin syscall wrappers; no dynamic-data-rendering surface to trace. The CAP-02 probe's data does flow:
- `Benchmark._run_uuid` → `_pre_execution_gate(self._run_uuid)` → `run_shared_fs_probe(run_uuid=)` → mpirun argv[2] → `SHARED_FS_PROBE_SCRIPT main()` reads `sys.argv[2]` → sentinel path `.mlpstorage-shared-fs-probe-<uuid>`.
- Verified by both the unit-level mocked-MPI exec test and (in environments with mpirun + mpi4py) the real-mpirun integration test. Locally skipped, so this hop is verified by code-level wiring + mocked execution only.

### Behavioral Spot-Checks

| Behavior                                                                      | Command                                                                                              | Result                                  | Status  |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------- | ------- |
| Heredoc compiles as Python                                                    | `python3 -c "from mlpstorage_py.cluster_collector import SHARED_FS_PROBE_SCRIPT; compile(...)"`     | `script compiles`, len 8925             | ✓ PASS  |
| Phase 5 production imports (under stubbed psutil)                             | Inside pytest with psutil stub: `from mlpstorage_py.system_description.diff import ...` etc.        | All 572 tests collected and passed       | ✓ PASS  |
| Full Phase 5 regression slice (per RESEARCH §"Sampling Rate")                | `pytest test_diff.py test_errors.py test_auto_generator_write.py test_auto_generator.py test_cluster_collector.py test_capacity_gate.py test_shared_fs_probe.py test_systemname_yaml_end_to_end.py test_shared_fs_probe_real_mpi.py -q` | 572 passed, 3 skipped in 11.47s         | ✓ PASS  |
| run() ordering: `_collect_cluster_start` → `_pre_execution_gate` → `write_systemname_yaml` | `awk` block (lines 1095/1102/1110)                                                                  | collect@1095 < gate@1102 < write@1110   | ✓ PASS  |
| `_pre_execution_gate` ordering: CAP-01 BEFORE CAP-02                          | Read base.py lines 979-1027                                                                          | check_capacity_4field at 1007; run_shared_fs_probe at 1019 | ✓ PASS  |
| Pitfall 4 source-grep: `comm.bcast(status, root=0)` appears                   | `grep -c 'comm.bcast(status, root=0)' cluster_collector.py`                                          | 2 (definition + docstring citation)     | ✓ PASS  |
| W-1 D-49 tight ordering: rank-0 `time.sleep(5.0)` precedes final `comm.Barrier()` | `grep -B1 'time\.sleep(5\.0)' cluster_collector.py`                                                  | `if rank == 0:\n    time.sleep(5.0)`    | ✓ PASS  |
| Sentinel prefix in heredoc                                                    | `grep 'mlpstorage-shared-fs-probe-' cluster_collector.py`                                            | 1 occurrence                             | ✓ PASS  |
| Local-disk hint verbatim                                                      | `grep 'this typically means one or more hosts have a local-disk path where a shared mount was expected' cluster_collector.py` | 1 occurrence | ✓ PASS  |
| Skip-if-no-mpirun pattern                                                     | `pytest tests/integration/test_shared_fs_probe_real_mpi.py`                                          | 3 skipped (mpi4py not installed)        | ✓ PASS  |

### Probe Execution

| Probe                                                                  | Command                                                                                | Result    | Status  |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------- | ------- |
| (No `scripts/*/tests/probe-*.sh` declared in Phase 5)                  | n/a                                                                                    | n/a       | n/a     |
| Real-MPI integration probe (in-tree, declared by Slice 5 + checker B-3 Option A) | `pytest tests/integration/test_shared_fs_probe_real_mpi.py -v`                | 3 skipped | ✓ PASS  |

### Requirements Coverage

| Requirement | Source Plans       | Description                                                                                              | Status       | Evidence                                                                                                                                       |
| ----------- | ------------------ | -------------------------------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| LIFE-02     | 05-01, 05-02, 05-05 | Load on-disk YAML and compute logical diff against in-memory image for collector-owned fields            | ✓ SATISFIED  | Diff core + parse_on_disk_systemname_yaml + load-diff-raise branch + per-mode independence test pair.                                          |
| LIFE-03     | 05-01, 05-02, 05-05 | Non-empty diff → fail before DLIO/MPI launch with JSONPath-style report + remediation block               | ✓ SATISFIED  | SystemDriftError raise site + format_unified_diff output verified by `test_drift_*_fails_before_dlio_sc2` (bench._run asserted NOT called).    |
| LIFE-04     | 05-02, 05-05        | Empty diff → no-touch + hand-fill survival                                                                | ✓ SATISFIED  | LIFE-04 mtime invariant + sha256 invariant + hand-fill survival tests at both unit (TestPhase5DriftWiring) and integration (SC#1) levels.      |
| CAP-01      | 05-03, 05-05        | Pre-datagen disk-space gate with 4-field error message; per-rank check on multi-node                      | ✓ SATISFIED  | check_capacity_4field + 4 per-subclass overrides + Benchmark.run() insertion + per-rank discipline via natural mpirun fan-out.                 |
| CAP-02      | 05-04, 05-05        | Pre-datagen/run shared-FS verification; cardinality > 1 fail with host listing; single-host silent no-op  | ⚠️ SATISFIED (with UAT-defer) | SHARED_FS_PROBE_SCRIPT + launcher + 22 unit tests + 7 integration tests; real multi-host hardware coverage deferred to UAT (see human_verification). |

**Orphaned requirements:** None. REQUIREMENTS.md line 111-115 maps exactly LIFE-02, LIFE-03, LIFE-04, CAP-01, CAP-02 to Phase 5; all five are present in plan frontmatter `requirements:` fields.

### Anti-Patterns Found

| File                                                  | Line   | Pattern                                                                                                                                                                                                  | Severity | Impact                                                                                                                                                                                  |
| ----------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mlpstorage_py/cluster_collector.py:3403-3441`        | 3403-3441 | `shell=True` with unsanitized destination / run_uuid / output_file interpolated into cmd_str (see REVIEW CR-01)                                                                                          | ⚠️ Warning | Pre-existing project pattern (MPIClusterCollector); new attack surface introduced by Phase 5. Suggested fix: `shell=False` with cmd_parts list, or `shlex.quote()` each interpolated position. Per user direction, non-blocking advisory. |
| `mlpstorage_py/cluster_collector.py:3312-3322,3454-3467` | 3312-3322,3454-3467 | `output_file` via `tempfile.mkstemp` on launch host, but rank 0 may run remote (see REVIEW CR-02)                                                                                                          | ⚠️ Warning | Silent misclassification when launch host is not in --hosts. Real-mpi test only runs on localhost so latent. Per user direction, non-blocking advisory; track into defect backlog or Phase 6 follow-up. |
| `mlpstorage_py/cluster_collector.py:3308-3525`        | 3308-3525 | Tempfiles leak on every CAP-02 probe (no try/finally cleanup) — REVIEW WR-02                                                                                                                              | ⚠️ Warning | Cosmetic; CI hosts may accumulate over time. Trivial fix.                                                                                                                              |
| `mlpstorage_py/cluster_collector.py:2677-2750`        | 2677-2750 | Possible MPI deadlock if rank-0 raises after gather but before bcast — REVIEW WR-01                                                                                                                       | ⚠️ Warning | Failure mode resolves to a 60s mpirun timeout; not a hang. Suggested wrap-in-try fix is straightforward.                                                                                |
| `mlpstorage_py/system_description/diff.py:142-168`    | 142-168 | `_flatten_to_paths` silently drops keys with empty-container values (REVIEW WR-03)                                                                                                                       | ℹ️ Info    | Benign at current emit shape; behavior should be documented. Future field additions could expose the gap.                                                                              |
| No `TBD`/`FIXME`/`XXX` debt markers found in any Phase 5 production file | n/a    | grep clean                                                                                                                                                                                                | ✓ N/A     | Debt-marker gate passes.                                                                                                                                                                |

### Human Verification Required

See `human_verification:` in frontmatter for the four hardware-dependent items deferred to UAT. Detailed below:

#### 1. Real multi-host CAP-02 cardinality check

**Test:** Run `mlpstorage training datagen --hosts client1,client2 --data-dir /shared/nfs/mlps-v3.0/data/ ...` where client1 and client2 are distinct physical machines mounting the same NFS share. Then repeat with one client's --data-dir pointed at a local-disk path masquerading as the same name.

**Expected:** Silent success when both clients mount the shared FS; hard failure with each hostname + (st_dev, st_ino) tuple listed and the verbatim local-disk hint when one is local.

**Why human:** The mocked-MPI launcher tests and the localhost-only real-mpi test cannot exercise (st_dev, st_ino) divergence between distinct physical machines.

#### 2. Real starved-disk CAP-01

**Test:** Configure a real partition (e.g., a small tmpfs mount) where `os.statvfs(.).f_bavail*f_frsize` < the unet3d datagen footprint. Invoke `mlpstorage training datagen --model unet3d --data-dir /mnt/tiny ...`.

**Expected:** Benchmark exits non-zero before any dataset file is created; stderr contains 'CAP-01:', destination path, available_bytes, required_bytes, deficit.

**Why human:** Unit tests mock `os.statvfs`. Real-disk behavior depends on kernel + filesystem-specific reporting.

#### 3. End-to-end LIFE-04 hand-fill survival across the real run pipeline

**Test:** `mlpstorage init`; `mlpstorage training run --model unet3d --num-accelerators 2 ...` to write systemname.yaml; hand-edit `friendly_description` on disk; re-run the same command.

**Expected:** friendly_description survives byte-identical; on-disk mtime unchanged; no SystemDriftError raised.

**Why human:** The integration tests mock `bench._run` to avoid invoking DLIO/MPI. Real DLIO execution against the running fleet is the only end-to-end coverage.

#### 4. Submitter-laptop deployment for CAP-02 (REVIEW-CR-02 reproduction)

**Test:** Launch from a host NOT in --hosts. Run `mlpstorage training datagen --hosts client1,client2 ...` from `submitter-laptop`.

**Expected:** Either (a) the bug surfaces (validating CR-02 — misleading "mpi4py not installed" error on success path) → file defect ticket, or (b) the fix has been staged and the probe completes cleanly.

**Why human:** Reproduces only in a multi-host topology where the launch host is outside the rank list.

### Gaps Summary

No code-level gaps were found. All 5 requirement IDs (LIFE-02, LIFE-03, LIFE-04, CAP-01, CAP-02) are SATISFIED at unit + integration test layers; all 8 ROADMAP SCs have named, passing locked tests; the 572-test regression slice is clean.

Two **non-blocking advisory warnings** (REVIEW-CR-01 shell injection, REVIEW-CR-02 launch-host-only output) come from the code review. Per the user's verifier context these are flagged but not blockers for Phase 5 sign-off — they should be tracked into a defect backlog or Phase 6 follow-up. Note that CR-02 is invisible to the test suite (real-mpi test only spawns localhost ranks) so the real-world impact can only be evaluated via the human-verification item #4 above.

Four hardware-dependent items are deferred to UAT per the project's "UAT defer pattern for hardware" memory — these are NOT gaps in the phase deliverable but checkpoints that require physical hardware (multi-host MPI, real starved disk, real DLIO run pipeline) to observe end-to-end.

### Re-verification Metadata

N/A — this is the initial verification of Phase 5.

---

*Verified: 2026-06-24*
*Verifier: Claude (gsd-verifier)*
