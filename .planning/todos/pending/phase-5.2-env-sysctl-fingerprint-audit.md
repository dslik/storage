---
title: Phase 5.2 — audit environment + kernel sysctl values in client-identity fingerprint
created: 2026-06-24
status: pending
severity: major
resolves_phase: ""
proposed_for: phase 5.2 (decimal/polish phase)
source: phase-05 UAT Test 3 (LIFE-04 hand-fill survival, OMPI env bug)
---

## What

A focused audit of which fields the auto-collector captures into the
"environment" stanza AND the "sysctl" stanza, with the goal of
distinguishing **identity-defining** values (legitimately part of D-38's
11-tuple fingerprint) from **transient/runtime/auto-tuned noise** (must
NOT be in the fingerprint or it would flag drift on every re-run).

## Why this is a phase, not a single fix

The OMPI-vars-as-identity bug surfaced during UAT (see sibling todo
`diff-empty-collector-as-handfill-affordance.md` for the empty-collector
nuance, and the patch in commits TBD for the immediate OMPI denylist).
But OMPI is only ONE example of a broader category. A proper audit
requires:

1. **Environment denylist patterns beyond OMPI:**
   - `PMI_*` (MPICH/Hydra launcher)
   - `SLURM_JOB_ID`, `SLURM_STEP_ID`, `SLURM_NODE_ALIASES`, …
   - `PALS_*` (HPE Cray launcher)
   - `HYDRA_*`
   - `PBS_JOBID`, `PBS_TASKNUM` (Torque/OpenPBS)
   - `LSB_JOBID`, `LSF_*` (IBM LSF)
   - `KUBE_*`, `K8S_*` (Kubernetes batch)
   - Generic: anything with `_JOBID`, `_PID`, `_PORT`, `_URI`, `_TOKEN`,
     `_TMP*`, `_SESSION*` substrings

2. **Sysctl values that are auto-tuned per-boot:**
   The collector currently captures the full `net.core.*`, `net.ipv4.*`,
   `vm.*` sysctl tree (see cluster_collector.py SYSCTL collector). Many
   of these are auto-tuned by the kernel based on memory, CPU count, NIC
   characteristics, or interactive workload pressure. Examples to
   investigate for volatility:
   - `net.ipv4.tcp_mem` — tied to total memory at boot, kernel may
     re-tune on memory pressure
   - `net.ipv4.tcp_rmem`, `net.ipv4.tcp_wmem` — auto-tuned per-NIC
   - `vm.dirty_bytes`, `vm.dirty_background_bytes` — runtime-tunable
   - `net.core.somaxconn` — admin-tunable but often varies
   - The whole `tcp_*` family for keepalive, retries, congestion control
     — fine-tuning differs between distro versions, not really client
     identity

3. **Same audit principle should apply to other captured fields:**
   - `chassis.cpu_model` — identity, KEEP in fingerprint
   - `chassis.memory_capacity` — identity, KEEP
   - `networking[].speed` — identity, KEEP
   - `drives[].vendor/model` — identity, KEEP
   - `OS, version` — identity but SHOULD it be? A package upgrade
     shouldn't flag drift on a stable cluster

## Proposed approach (phase 5.2 PLAN)

- Plan 05.2-01: enumerate every key captured by the collector + cross-reference
  to `_FINGERPRINT_KEYS`. Produce a categorization spreadsheet (identity / config /
  runtime / auto-tuned).
- Plan 05.2-02: write the environment denylist (extends the OMPI fix from
  the UAT session — pattern + literal denylist with cross-launcher
  coverage).
- Plan 05.2-03: write the sysctl filter (likely an allowlist of
  intentionally-tracked admin-set sysctls rather than a denylist —
  smaller surface, less subject to kernel evolution).
- Plan 05.2-04: revisit the D-38 11-tuple composition — should
  environment/sysctl be in the fingerprint at all, or only in the
  leaf-diff layer that drives the SystemDriftError detail report?
- Plan 05.2-05: integration tests that simulate (a) reboot, (b) kernel
  upgrade, (c) launcher change, (d) NIC swap — and assert which trigger
  drift vs which are silently absorbed.

## Out of scope

The empty-collector-as-handfill semantics (sibling todo
`diff-empty-collector-as-handfill-affordance.md`) is orthogonal but
related — both touch the diff layer. Phase 5.2 should bundle both
investigations.

## Why this is decimal-phase (5.2) not Phase 6

The auto-collector + diff form a tightly-coupled subsystem shipped as a
unit in Phase 5. Audit + correction should land as a polish phase
against the same milestone (v1.0) before any new feature work in Phase 6.
