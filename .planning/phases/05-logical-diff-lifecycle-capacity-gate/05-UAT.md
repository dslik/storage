---
status: deferred
phase: 05-logical-diff-lifecycle-capacity-gate
source: [05-VERIFICATION.md]
started: 2026-06-24T00:00:00Z
updated: 2026-06-24T15:55:00Z
defer_reason: |
  Test 4 (submitter-laptop CAP-02 / REVIEW-CR-02 reproduction) requires a multi-host
  deployment where the launch host is outside the --hosts rank list — not reproducible
  from the WSL2 dev box. Per the "UAT defer pattern for hardware" project convention,
  deferring the whole UAT (not per-test blocked) so this resumes cleanly when the
  hardware is available.
resume_when: real multi-host topology accessible (e.g. two physical clients + a
  submitter laptop outside the rank list)
---

## Current Test

[UAT deferred — see frontmatter `defer_reason`]

## Tests

### 1. Real multi-host CAP-02 cardinality check across distinct physical machines
expected: When --hosts spans two machines mounting the same shared FS, datagen proceeds silently. When one host has a local-disk path masquerading as the shared mount, datagen fails fast with each hostname + (st_dev, st_ino) tuple listed and the verbatim local-disk hint.
why_human: Requires multi-host SSH + OpenMPI + mpi4py installed on each rank; the test_shared_fs_probe_real_mpi.py file only spawns two MPI ranks on localhost (which always share st_dev,st_ino → cardinality 1).
result: pass

### 2. Real-starved-disk CAP-01 4-field error
expected: |
  Invoke `mlpstorage training datagen --model unet3d --data-dir <real-mount-with-insufficient-space>`.
  Benchmark exits non-zero before any dataset file is created; stderr contains all four labeled
  fields (destination / available_bytes / required_bytes / deficit); no partial directory tree
  left in --data-dir.
why_human: Requires a real filesystem with controllable free space; unit tests mock os.statvfs. CR-01 advisory also worth re-exploring on this path because args.data_dir may contain spaces in realistic submitter mounts.
result: pass

### 3. End-to-end LIFE-04 hand-fill survival across a real run pipeline
expected: |
  Complete `mlpstorage init`, run a first benchmark to produce systemname.yaml, hand-edit
  `friendly_description`, then re-run the same command.
  The hand-edited friendly_description survives byte-identical in the on-disk file; mtime is
  unchanged; no SystemDriftError raised.
why_human: The integration test mocks the bench._run pipeline (e.g., VectorDBBenchmark with mocked _run). Real DLIO/MPI execution is not exercised in CI.
result: pass
verified_after_fixes: |
  Pass attained AFTER landing 5 fix commits during the UAT session:
    bad2b73 test(05-fix): RED — TrainingBenchmark cluster_information regression
    754763a fix(05-fix): GREEN — lazy-collect cluster_information for datagen path
    29f1062 fix(05-fix): degrade CAP-01 gracefully when memory undeterminable
    2bb8a38 test(05-fix): RED — env collector must denylist OMPI runtime vars
    0822c81 fix(05-fix): GREEN — denylist runtime-volatile OMPI launcher vars
  Plus a pre-existing related dispatch regression that blocked closed-mode entirely:
    0450eab test(verify-benchmark): RED — args.mode dispatch missing post-PR #412
    bd718ee fix(verify-benchmark): GREEN — read args.mode with legacy bool fallback
verification: |
  Final repro (whatif mode, dlrm + 1 accelerator):
    1. mlpstorage init BigCo /tmp/nerf-results
    2. mlpstorage whatif training dlrm datagen --num-processes 1 -rd ... -dd ... -sn BigMachine
    3. mlpstorage whatif training dlrm run -na 1 -at b200 -cm 1 -rd ... -dd ... -sn BigMachine
       → systemname.yaml written at /tmp/nerf-results/whatif/BigCo/systems/BigMachine.yaml
    4. hand-edit "friendly_description": "" → "Curtis dev rig — DO NOT OVERWRITE"
    5. re-run the same command from step 3
       → no SystemDriftError, md5/mtime/size byte-identical, friendly_description survives
follow_ups_logged:
  - .planning/todos/pending/diff-empty-collector-as-handfill-affordance.md
    (empty-from-collector means hand-fillable; addresses the chassis.model_name case
    surfaced by accidental dual-edit during UAT)
  - .planning/todos/pending/phase-5.1-env-sysctl-fingerprint-audit.md
    (broader audit of env + sysctl values in the D-38 fingerprint — extend OMPI
    denylist to other launchers, and re-examine which sysctls are identity vs
    runtime-tuned)
  - .planning/todos/pending/migrate-or-delete-test-open-closed-cli-flags.md
    (test rot from PR #412 exposed during the args.mode dispatch fix)

### 4. Submitter-laptop deployment for CAP-02 (REVIEW-CR-02 reproduction)
expected: |
  Launch from a host NOT in --hosts and confirm the misleading 'mpi4py not installed' message
  does NOT appear when the probe semantically succeeded. Either the bug is observed (validating
  CR-02) and tracked into Phase 6 / defect backlog, OR a fix is staged before phase sign-off.
why_human: Requires a multi-host deployment where the launch host is outside the rank list — not testable from the local dev shell.
result: [pending]

## Summary

total: 4
passed: 3
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps

- truth: "`mlpstorage training unet3d datagen` completes the CAP-01 capacity check and proceeds to data generation."
  status: failed
  reason: |
    User reported: TrainingBenchmark crashes during datagen with E201 — `'TrainingBenchmark' object has no attribute 'cluster_information'`.
    Repro: `./mlpstorage init BigCo /tmp/nerf-results && ./mlpstorage closed training unet3d datagen file -rd /tmp/nerf-results -np 4 -sn BigMachine -dd /tmp/nerf-data`.
  severity: blocker
  test: 3
  artifacts:
    - mlpstorage_py/benchmarks/dlio.py:44   # cluster_information only set when args.command != "datagen"
    - mlpstorage_py/benchmarks/dlio.py:521-525  # required_bytes_for_capacity_gate reads self.cluster_information
    - mlpstorage_py/benchmarks/dlio.py:542  # datasize() now calls self._pre_execution_gate()
  missing:
    - cluster_information collection wired into the datagen→datasize→_pre_execution_gate path
  candidate_fix: |
    Two options for plan-phase --gaps to weigh:
    (a) Set self.cluster_information = self.accumulate_host_info(args) unconditionally in
        TrainingBenchmark.__init__ (drop the args.command != "datagen" guard). Cheap, but may
        regress whatever motivated the original guard.
    (b) Have required_bytes_for_capacity_gate() lazy-collect cluster_information via
        self.accumulate_host_info(self.args) when the attribute is missing. Scoped only to the
        CAP-01 entry point, preserves the original guard intent for unrelated paths.
    Also check CheckpointingBenchmark / KVCacheBenchmark / VectorDBBenchmark for identical pattern —
    same _pre_execution_gate is wired into their datasize/datagen paths.

## Advisory warnings from REVIEW (non-blocking)

- **REVIEW-CR-01** (`cluster_collector.py:3403-3441`): CAP-02 launcher uses `shell=True` with unsanitized
  `destination`/`run_uuid`/`output_file` interpolated into the command string. A submitter who passes
  `--data-dir` containing shell metacharacters or spaces triggers either argument-splitting or arbitrary
  command execution. Suggested fix: `shell=False` with cmd_parts list, or `shlex.quote()` every
  interpolated position. Pre-existing MPIClusterCollector uses the same pattern; Phase 5 expanded the
  attack surface.

- **REVIEW-CR-02** (`cluster_collector.py:3312-3322, 3454-3467`): `output_file` is launch-host-local but
  rank 0 of the mpirun job may land on a remote host. When the launch host is not in `--hosts` (typical
  submitter-laptop deployment), the launcher reads a path that does not exist locally and raises a
  misleading `mpi4py not installed on all hosts` error even when the probe semantically succeeded. The
  real-MPI integration test runs both ranks on localhost so the defect is invisible at unit/integration
  layer. Suggested fix: place `output_file` inside the SCP-replicated `staging_dir` or SCP-fetch it
  back from rank 0's host before reading.

These are flagged for awareness during UAT — Test #4 may surface CR-02 directly.
