# Phase 5: Logical Diff Lifecycle + Capacity Gate - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-23
**Phase:** 5-logical-diff-lifecycle-capacity-gate
**Areas discussed:** Drift scope definition, Drift error format, CAP-02 fsid mechanism, Version-skew on YAML

---

## Drift scope definition (LIFE-02 mechanism)

### Q1 — Which mechanism decides what fields the diff considers?

| Option | Description | Selected |
|--------|-------------|----------|
| Round-trip recompute | For each in-memory host, re-run `node_dict_from_host(host)` and compare ONLY keys in the recomputed subtree. Drift scope = "whatever the collector currently emits." Self-maintaining; auto-inherits SER-02 omit logic + Phase 4 7-key emit shape. | ✓ |
| Explicit JSONPath allowlist | Static tuple `_COLLECTOR_OWNED_PATHS`; diff masks paths NOT in the allowlist. Explicit but every new collector field requires updating two places. | |
| Implicit complement of SER-02 | Treat "collector-owned" as "everything except SER-02 blanks." Defines drift via what NOT to compare. Risks falsely-triggering drift on new submitter-owned schema fields. | |

**User's choice:** Round-trip recompute (D-37)
**Notes:** Self-maintaining was the key tiebreaker — adding a new collector field in a future phase shouldn't need a separate registry update.

### Q2 — How does the diff match in-memory hosts to on-disk stanzas?

| Option | Description | Selected |
|--------|-------------|----------|
| Match by fingerprint | Re-fingerprint each in-memory host via the 11-tuple `_FINGERPRINT_KEYS`, look up on-disk stanza by matching fingerprint. Symmetric across fleet expansion and host migration. | ✓ |
| Match by stanza index | Compare by D-7 sort position. Cheap but brittle — fleet growth shuffles positions and reports drift on every stanza. | |
| Match by cpu_model + hostname-set | Track hosts explicitly. Would require schema/emit change (YAML doesn't currently emit hostnames; SER-01 collapses them). | |

**User's choice:** Match by fingerprint (D-38)

### Q3 — Is a quantity change drift?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — quantity is drift | Fingerprint match required AND quantity match required for "no drift." Symmetric with D-38. | ✓ |
| No — quantity is informational only | Only per-host shape fields are drift. Convenient but masks fleet shrinkage. | |

**User's choice:** Yes — quantity is drift (D-39)

---

## Drift error format (LIFE-03 UX)

### Q1 — How should the drift report be laid out?

| Option | Description | Selected |
|--------|-------------|----------|
| Per-stanza grouped | One block per affected stanza with fingerprint header + bulleted field diffs. Reads like a doctor's report. | |
| Flat list of JSONPaths | One line per differing field, full JSONPath, then values. Greppable. | |
| Unified-diff style | `diff --git`-style format with `--- on-disk` / `+++ in-memory` header and `@@ <JSONPath> @@` hunk headers. Hunk header carries the LIFE-03-mandated JSONPath. | ✓ |

**User's choice:** Unified-diff style (D-40)
**Notes:** User picked from the preview pane after seeing the three rendered samples side-by-side. The familiar `diff --git` shape was the deciding factor — submitters reading benchmark output already pattern-match on `-`/`+` lines.

### Q2 — Mid-discussion clarification

User paused the discussion to confirm the intent of the drift feature. After re-reading
LIFE-02/03/04 + the milestone-core-value sentence in REQUIREMENTS.md, both sides confirmed
the same understanding: the diff fires on `run` re-invocation against an existing
systemname.yaml, compares the live-collected in-memory image against the on-disk file,
fails the run on any drift, and shows the user exactly what changed so they can pick one
of the two LIFE-03 remediation paths (rename or rm). D-37..D-41 unchanged after the
clarification.

### Q3 — How should long or sensitive values be shown?

| Option | Description | Selected |
|--------|-------------|----------|
| Full values, no truncation | Verbatim on both sides. Credentials already redacted upstream (D-23/D-24); diff compares redacted strings. | ✓ |
| Truncate at fixed width | 80-char cap with `… (N chars)` marker. Obscures exact differing character. | |
| Smart truncate: show diff window | 40 chars around first differing position. Best of both but most code. | |

**User's choice:** Full values, no truncation (D-41)

### Q4 — Where does the drift report go and what's the exit signal?

| Option | Description | Selected |
|--------|-------------|----------|
| Raise SystemDriftError, logger.error report | New exception class inheriting from `MlpStorageError`. Format report, `logger.error()` it, raise. `main.py` top-level handler prints + exits non-zero. Consistent with `ConfigurationError` / `FileSystemError` precedents. | ✓ |
| Write report to stderr + sys.exit | Bypass logger. Faster to wire but breaks `MlpStorageError` convention. | |

**User's choice:** Raise SystemDriftError, logger.error pipeline (D-42)

---

## CAP-02 shared-filesystem verification mechanism

### Q1 — Which mechanism collects the per-host filesystem identifier?

| Option | Description | Selected |
|--------|-------------|----------|
| Sentinel file + os.stat | Rank 0 writes `<data-dir>/.mlpstorage-shared-fs-probe-<run-uuid>`, peers `os.stat()` it and report `(st_dev, st_ino)`. Bulletproof against bind-mount, FUSE, overlay quirks. | ✓ |
| `os.statvfs(data_dir).f_fsid` | Pure Python, no extra files. REQUIREMENTS.md explicitly flags `f_fsid` as having edge cases with bind mounts and FUSE. | |
| Shell out to `stat -f -c '%i'` | Same `f_fsid` semantics as option 2, via subprocess. Extra parsing for no extra robustness. | |

**User's choice:** Sentinel file + os.stat (D-43)
**Notes:** REQUIREMENTS.md line 53 explicitly defers tool choice to discuss/plan. Locked here.

### Q2 — Sentinel cleanup contract?

| Option | Description | Selected |
|--------|-------------|----------|
| Rank 0 unlinks in finally block | Try/finally around the probe so unlink fires on error. `<run-uuid>` suffix prevents concurrent-run collisions. Unlink failure logs warning but doesn't fail. | ✓ |
| Leave sentinel, GC on next run | Scan + unlink stale probes older than 1 hour. Simpler error path; submitters might see stray files. | |
| Use tempfile in `<data-dir>`, anonymous | `tempfile.NamedTemporaryFile(dir=<data-dir>)` for auto-cleanup. Anonymous from peers' POV — doesn't fit the rank-0-writes-path-others-read model. | |

**User's choice:** Rank 0 unlinks in finally block (D-44)

### Q3 — Which CAP-02 error states are 'fail-the-run' vs 'log-and-degrade'?

| Option | Description | Selected |
|--------|-------------|----------|
| Fail on any collection error or cardinality > 1 | All per-host failures (rank 0 can't write, peer can't stat, MPI gather drops a rank) hard-fail. Cardinality > 1 hard-fails with the standard message. | ✓ |
| Treat collection failures as no-op | Only fail on cardinality > 1. Silent skip of a safety gate on broken NFS. | |

**User's choice:** Fail on any error (option 1) + 5-second post-unlink quiesce (D-45)
**Notes:** User added the 5-second quiesce after the unlink: "Since we're creating and unlinking a file in the --data-dir, that counts as 'load' on the storage system. We should put a 5-second delay between the unlink and the start of the benchmark run so that any consequences of the unlink have fully completed before we start putting load that we're measuring on the system-under-test." Locked as D-45.

### Q4 (follow-up post-deferral list discussion) — Quiesce coordination?

User added a refinement to the 5-second quiesce: "It would be good if we could use an
MPI barrier so that the rank 0 system did the sleep(5) and all the other instances
simply waited until rank 0 dropped the barrier. That reduces the smeared-start that
would happen if every node did their own sleep(5)."

Locked as D-49 — Rank-0-side `time.sleep(5)` followed by `MPI_Barrier` convergence;
all other ranks block on the barrier immediately after their `os.stat` gather completes.
Every rank exits the barrier within microseconds of each other, so the measured
workload starts simultaneously across the fleet.

---

## Toolchain version skew (LIFE-02 edge cases)

### Q1 — Disk-absent fields (new in-memory field not on disk)?

| Option | Description | Selected |
|--------|-------------|----------|
| Treat as drift — fail the run | Symmetric with D-37..D-39. Submitter upgrades mlpstorage; new collector field appears; run fails until submitter renames or rm's. Aligned with milestone-core-value sentence. | ✓ |
| Auto-merge new fields silently | Write the new field into the YAML in place, proceed. Breaks LIFE-04 no-touch invariant. | |
| Auto-merge but log a warning | Same merge with WARNING log. Same LIFE-04 break + surprise risk. | |

**User's choice:** Treat as drift — fail the run (D-46)

### Q2 — Disk-only fields (on-disk field no longer collected)?

| Option | Description | Selected |
|--------|-------------|----------|
| Treat as drift — fail | Symmetric with D-46. Submitter downgrades or removes a sysctl pattern; on-disk field disappears from new image; drift. | ✓ |
| Ignore on-disk-only fields | Asymmetric with D-46; adds an irregular edge case. | |

**User's choice:** Treat as drift — fail (D-47)

### Q3 — Structurally malformed on-disk YAML?

| Option | Description | Selected |
|--------|-------------|----------|
| Fail with a distinct error class | `SystemDescriptionParseError` separate from `SystemDriftError`. Names the path + line/column. Remediation hint: `rm <path> && re-run`. | ✓ |
| Treat malformed file as 'doesn't exist' — overwrite | Risk destroying submitter hand-fills on transient parse hiccup. | |
| Treat malformed file as drift — use SystemDriftError | Misleading error message. Less code, worse UX. | |

**User's choice:** Fail with a distinct error class (D-48)

---

## Claude's Discretion

The user explicitly confirmed these areas stay in Claude's Discretion (researcher /
planner picks during downstream agents' phases):

- **Diff implementation mechanism** — recursive dict-walk vs. flatten-to-paths-then-
  set-diff vs. off-the-shelf `deepdiff`-style library.
- **CAP-01 reuse of `validation_helpers.py:check_disk_space`** — wrap or augment;
  planner picks the integration shape.
- **CAP-01 required-bytes source per benchmark** — training uses existing
  `calculate_training_data_size`; checkpointing/vectordb/kvcache each have their own
  size knobs the planner reads from the benchmark classes.
- **Gate ordering** — CAP-01 + CAP-02 first (cheap, fast-fail), LIFE-02/03/04 after
  (only on `run`, requires loading + comparing). Implied by D-12 + cheap-first
  principle.
- **Pattern B vs. separate fast probe for CAP-02** — fold the CAP-02 fsid probe into
  the existing `MPI_COLLECTOR_SCRIPT` (D-36 Pattern B precedent) or ship a smaller
  separate probe. Planner picks based on the wave-ordering tradeoff.

## Deferred Ideas

- **Configurable quiesce duration** — D-49 hard-codes 5s. Future config knob if real
  shared-storage deployments need a longer wait.
- **JSON-formatted drift report** — Currently human-readable only. Future structured
  output for CI / submission-checker integration.
- **`mlpstorage validate` extension surfacing SER-02 blanks (SCH-01)** — v2 milestone.
- **Reportgen automatic inclusion of systemname.yaml (BUN-01)** — v2 milestone.
- **`mlpstorage init --adopt` (ADP-01)** — v2 milestone.
- **Drift with "soft" / "warning" severity** — v1 ships strict-fail; revisit if real
  demand surfaces.
- **Auto-suggested new `--systemname` value on drift** — generic "rename" hint in v1;
  auto-suggestion is a future UX polish.

---

*Phase 5 discussion: 2026-06-23*
*13 decisions locked (D-37 through D-49); 5 areas explicitly left to Claude's
discretion; 7 ideas captured for future phases.*
