# Phase 5: Logical Diff Lifecycle + Capacity Gate - Context

**Gathered:** 2026-06-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 5 closes the milestone with two distinct startup gates that together protect the
submission-package guarantee — a submitter can't accidentally mix results from one
client fleet (or one toolchain version) into another fleet's system description, and
they can't accidentally start a benchmark on a destination that can't hold the data or
isn't actually shared across the participating hosts.

1. **Drift lifecycle (LIFE-02/03/04)** — On every `mlpstorage <mode> <benchmark> [model]
   run` against an existing `<results-dir>/<mode>/<orgname>/systems/<systemname>.yaml`,
   load the on-disk file, recompute the in-memory image from the live MPI fleet
   collection, perform a logical (not textual) diff, and either fail with a unified-diff
   report (drift) or proceed without touching the file (no drift — submitter's hand-fills
   survive). Diff scope is the collector-owned fields only; SER-02 blanks are invisible
   to drift detection. Diff is per-mode (closed/open/whatif each own their own yaml).

2. **Capacity gate (CAP-01/02)** — At `datagen` and `run` startup, before any data is
   written and before any DLIO/MPI launch, refuse to proceed if (a) the dataset
   destination directory lacks free space for the computed dataset size (per-rank
   `os.statvfs` check), or (b) on multi-host operations, the participating hosts disagree
   on which filesystem the destination directory lives on (shared-FS verification via a
   sentinel file + per-host `os.stat`). On single-host runs, the shared-FS check is a
   no-op.

**Out of scope (deferred to v2 milestone):**
- `mlpstorage validate` extension surfacing SER-02 blanks (SCH-01)
- Reportgen automatic inclusion of systemname.yaml in submission bundle (BUN-01)
- `mlpstorage init --adopt` for migrating existing non-initialized results-dirs (ADP-01)

</domain>

<decisions>
## Implementation Decisions

Phase 5 carries forward all locked Phase 1/2/3/4 decisions D-1..D-36 verbatim
(Pydantic-driven emit, universal D-2 collection-failure rule scoped to collector
failures only, D-9 atomic write + OSError propagation, D-11 path derivation, D-12
`args.command == 'run'` write gate, D-22 callable fingerprint extractors with
`key=repr`, D-25 unified redaction in `storage_config.py`, D-33 omit-when-empty for
drives, D-34 11-tuple `_FINGERPRINT_KEYS`, D-36 Pattern B MPI script duplication).
Decisions D-37..D-49 below are Phase 5 additions only.

### Drift detection scope (LIFE-02 mechanism)

- **D-37 — Drift scope is computed via round-trip recompute.** For each in-memory host,
  re-run `node_dict_from_host(host)` to produce a node-shaped dict, then quantity-group
  via the existing `group_by_fingerprint` (D-4 generic, D-22 callable extractors). The
  resulting list-of-stanzas IS the comparison subject. Compare ONLY the keys that this
  recomputed subtree contains. Consequence: drift scope = "whatever fields the
  collector currently emits for this host, on this version of mlpstorage." Automatically
  inherits SER-02 omit logic (e.g., D-33 drives-omit when empty) and the Phase 4 7-key
  emit shape (`friendly_description, chassis, networking, sysctl, environment, drives,
  operating_system`). Self-maintaining — adding a new collector field in a future phase
  automatically expands the diff scope with no separate registry to update. No explicit
  JSONPath allowlist, no SER-02 complement scan.

- **D-38 — In-memory hosts match on-disk stanzas by fingerprint.** Each in-memory
  host's fingerprint is computed via `_FINGERPRINT_KEYS` (the 11-tuple Phase 4 settled
  on at D-34); the on-disk YAML's stanzas are indexed by the same fingerprint (recomputed
  from each stanza's fields). Diff pairs are matched fingerprint-to-fingerprint. A
  fingerprint with no matching counterpart on either side is drift (host shape changed,
  new fingerprint appeared, or fingerprint disappeared). Symmetric across additions and
  removals.

- **D-39 — Quantity change is drift.** A fingerprint that matches on both sides but
  whose `quantity` value differs (in-memory says qty=4, on-disk says qty=5) is drift.
  Fleet shrinkage AND fleet growth both surface as quantity diffs. Submitter who
  decommissions or adds a node sees an explicit signal rather than silent recompute.

### Drift error format (LIFE-03 UX)

- **D-40 — Drift report uses unified-diff style with JSONPath-style hunk headers.**
  Format the diff as a `diff --git`-style block: `--- on-disk` / `+++ in-memory` header,
  then per-hunk `@@ <JSONPath> @@` headers carrying the field location (e.g.,
  `@@ clients[0].chassis @@`), then `-`/`+` lines showing the differing field name +
  value. The hunk header satisfies LIFE-03's "JSONPath-style path" requirement; the
  `-`/`+` lines satisfy "on-disk value and in-memory value side-by-side." The standard
  drift report block carries the two LIFE-03 remediation hints at the end:
  ```
  Remediation:
    • Rename the existing yaml and re-run with --systemname <new>
      (a fresh systemname.yaml will be generated)
    • Remove <path>/systems/<sys>.yaml and re-run
      (you will lose hand-filled blanks)
  ```

- **D-41 — Full values, no truncation.** Both on-disk and in-memory values are emitted
  verbatim in the `-`/`+` lines. Long sysctl multi-tuples (e.g.,
  `net.ipv4.tcp_rmem = "4096\\t87380\\t16777216"`) appear in full so the submitter can
  see exactly which integer changed. Credential-shaped values (`AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`) are ALREADY redacted upstream in the collection step
  (D-23/D-24 in Phase 4) — the diff compares the redacted strings; no fresh leak risk.

- **D-42 — Drift surfaces as `SystemDriftError` through the logger pipeline.** Define a
  new exception class `SystemDriftError` inheriting from the project's `MlpStorageError`
  hierarchy. Format the unified-diff report as a single multi-line string, call
  `logger.error(report)` so it appears in both the on-disk run log AND the terminal via
  the existing logger handlers, then `raise SystemDriftError(...)`. `main.py`'s existing
  top-level error handler will format the error footer and exit non-zero. Consistent
  with how Phase 1's `ConfigurationError` and Phase 2's `FileSystemError` surface.

### Toolchain version skew (LIFE-02 edge cases)

- **D-46 — Disk-absent fields are drift.** If the in-memory image has a field the
  on-disk YAML doesn't have (e.g., the submitter upgraded mlpstorage between runs and
  the new version's sysctl allowlist added a pattern that now matches a key on this
  host), the diff treats that as drift. Forces an explicit submitter choice: rename or
  rm. Aligns with the milestone-core-value guarantee — a submitter can't accidentally
  ship one toolchain version's system description with another toolchain version's
  benchmark numbers.

- **D-47 — Disk-only fields are drift.** Symmetric with D-46. If the on-disk YAML
  has a field the new in-memory image doesn't produce (submitter removed a sysctl
  pattern from the allowlist file, or downgraded mlpstorage), that's also drift. Strict
  bidirectional comparison — no irregular edge cases for one direction vs. the other.

- **D-48 — Structurally malformed on-disk YAML raises `SystemDescriptionParseError`.**
  Distinct exception class from `SystemDriftError`. Fires when `yaml.safe_load` raises
  `yaml.YAMLError`, or when the parsed dict is missing the top-level keys the diff
  expects (`system_under_test.clients`). Message names the path, surfaces line/column
  from `yaml.YAMLError.problem_mark` when available, and points the submitter at
  `rm <path> && re-run` as the remediation. Distinct class so the top-level handler
  can format it differently and future tooling can distinguish "submitter has drifted
  hardware" from "file is corrupt." Refuses to overwrite a malformed file silently —
  treating malformed-as-drift would mislead the submitter; treating it as "doesn't
  exist" would risk destroying hand-fills on a transient parse hiccup.

### CAP-02 shared-filesystem verification mechanism

REQUIREMENTS.md line 53 explicitly defers tool choice to discuss/plan — locking it here:

- **D-43 — CAP-02 uses a sentinel file + `os.stat`.** Rank 0 writes a sentinel file at
  `<data-dir>/.mlpstorage-shared-fs-probe-<run-uuid>` (where `<run-uuid>` is a
  `uuid.uuid4().hex` value scoped to this single invocation), then each peer rank
  `os.stat()`s that exact path and reports `(st_dev, st_ino)` back via MPI gather.
  The set `{(st_dev, st_ino) for each rank}` must have cardinality 1; cardinality > 1
  means the hosts disagree on which filesystem holds `<data-dir>`. Bulletproof against
  bind-mount, FUSE, and overlay quirks that break `os.statvfs().f_fsid` (which
  REQUIREMENTS.md flags explicitly). No subprocess shell-out; pure Python.

- **D-44 — Rank 0 unlinks the sentinel in a `finally` block.** The unlink sits in the
  same try/finally that wraps the probe so the sentinel disappears even on collection
  error. The `<run-uuid>` suffix in the sentinel name guarantees concurrent runs from
  the same user against the same `<data-dir>` don't collide. If unlink itself fails
  (permission anomaly, NFS race), log a `WARNING` and continue — the sentinel file is
  small and the next run's probe uses a different name, so leftover sentinels are
  cosmetic, not load-bearing.

- **D-45 — Any CAP-02 collection error is a hard fail; cardinality > 1 is a hard fail.**
  Per-host failures (rank 0 EACCES/ENOSPC on sentinel create; peer ENOENT/EACCES/NFS
  stale handle on `os.stat`; MPI gather drops a rank) all raise with a specific message
  naming the failure mode and host. "I tried to verify shared storage and couldn't" must
  NOT be silently treated as "shared" — CAP-02 is a safety gate, and a silent skip would
  let broken NFS setups proceed to a benchmark whose numbers are meaningless. Cardinality
  > 1 raises with the standard "hosts disagree on filesystem" message listing each host
  and its reported `(st_dev, st_ino)`, plus the one-line "this typically means one or
  more hosts have a local-disk path where a shared mount was expected" hint per
  REQUIREMENTS.md CAP-02.

- **D-49 — Post-unlink quiesce is rank-0-side with MPI barrier convergence.** Sequence
  after a clean CAP-02 probe:
  1. All non-rank-0 ranks reach an `MPI_Barrier` immediately after their `os.stat`
     gather completes.
  2. Rank 0 unlinks the sentinel, then calls `time.sleep(5)`, then reaches the same
     `MPI_Barrier`.
  3. All ranks exit the barrier within microseconds of each other and proceed into
     the benchmark workload.

  Reasoning: the sentinel create + per-host `os.stat` + unlink count as load on the
  storage system. A 5-second quiesce window after the unlink lets any consequences of
  the unlink (NFS attribute-cache invalidation, lazy metadata flush, FUSE upcall, etc.)
  fully complete before the measured workload begins. Doing the `time.sleep(5)` only on
  rank 0 + using a barrier for convergence avoids the smeared-start that would happen
  if every rank slept independently — without coordination, each rank would resume at a
  slightly different wall-clock instant depending on when it finished its probe. With
  the barrier, every rank starts the measured work at approximately T+5 seconds after
  the unlink, simultaneously, which is the correct contract for "the benchmark starts
  now, fleet-wide."

### Claude's Discretion

- **Diff implementation mechanism.** Recursive dict-walk vs. flatten-to-paths-then-
  set-diff vs. an off-the-shelf `deepdiff`-style library. Pure code shape; researcher
  picks based on dependency budget and the round-trip-recompute structure (D-37).
- **CAP-01 reuse of `validation_helpers.py:check_disk_space`.** The existing helper
  (line 402) already does the `os.statvfs(path).f_bavail * f_frsize` math and raises
  `FileSystemError` with the canonical "Insufficient disk space" message. Phase 5
  CAP-01 mandates a specific error format ("destination path, available bytes,
  required bytes, deficit") — planner picks between thin-wrapper-around-existing or
  augmenting `check_disk_space` itself to emit the required four fields.
- **CAP-01 required-bytes source per benchmark type.** Training reuses
  `mlpstorage_py/rules/utils.py:calculate_training_data_size(args, ...)`.
  Checkpointing, vectordb, and kvcache each have benchmark-specific size knobs
  (`--checkpoint-folder` + workload yaml for checkpointing; engine config for vdb;
  workload yaml for kvcache). Planner reads each benchmark class's existing size
  computation and wires the resulting `required_bytes` into the CAP-01 check call.
- **Gate ordering.** CAP-01 + CAP-02 fire BEFORE LIFE-02/03/04 (cheap, fast-fail; both
  apply to `datagen` AND `run`, while LIFE-02/03/04 only applies to `run` per D-12).
  Implied by D-12 + cheap-first principle; no separate decision needed.
- **Pattern B vs. separate fast probe for CAP-02.** REQUIREMENTS.md says "before any
  work begins." Could fold the fsid collection into the existing `MPI_COLLECTOR_SCRIPT`
  Pattern B template, or land a separate smaller probe script. Planner picks based on
  the wave-ordering tradeoff (folded = one MPI roundtrip but the collector script gets
  bigger; separate = two roundtrips but cleaner separation).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 5 scope and requirements

- `.planning/ROADMAP.md` §"Phase 5: Logical Diff Lifecycle + Capacity Gate" — goal,
  success criteria SC #1–8, requirements list (LIFE-02/03/04, CAP-01/02).
- `.planning/REQUIREMENTS.md` lines 45–48 — LIFE-02 (logical diff), LIFE-03 (drift
  error format), LIFE-04 (no-touch when diff is empty).
- `.planning/REQUIREMENTS.md` line 52 — CAP-01 (per-rank free-space check).
- `.planning/REQUIREMENTS.md` line 53 — CAP-02 (shared-FS verification; explicitly
  defers tool choice to discuss/plan, locked here by D-43).
- `.planning/REQUIREMENTS.md` lines 6, 40 — milestone-core-value sentence + SER-02
  blanks list (which fields the drift detection MUST NOT consider).

### Carried from Phase 1/2/3/4 — must read for context

- `.planning/phases/04-sysctl-environment-and-drives-coverage/04-CONTEXT.md` — D-23..D-36
  locked decisions, especially D-33 (drives omit-when-empty), D-34 (11-tuple
  `_FINGERPRINT_KEYS`), D-35 (strict-split policy), D-36 (Pattern B MPI duplication).
- `.planning/phases/04-sysctl-environment-and-drives-coverage/04-VERIFICATION.md` —
  Phase 4 verified 8/8; the 7-key emit shape (`friendly_description, chassis,
  networking, sysctl, environment, drives, operating_system`) IS the round-trip target
  for D-37.
- `.planning/phases/03-chassis-model-networking-coverage/03-CONTEXT.md` — D-22 (callable
  fingerprint extractors with `key=repr`), D-17 (stub-vs-real splice — Phase 5 leaves
  alone), D-22 forward-notes section pointing at Phase 4.
- `.planning/phases/02-first-run-write-of-partial-systemname-yaml/02-CONTEXT.md` —
  D-1..D-16 baseline (Pydantic-driven emit, universal-failure rule scoped to collectors,
  D-3 stub splice, D-7 multi-stanza sort, D-9 atomic write, D-11 path, D-12 run-only
  gate, D-14 outer dict, D-16 HostInfo extension pattern). Especially D-9 docstring
  promise that the FileExistsError branch is "Phase 5 will replace this branch with
  diff-and-fail."
- `.planning/phases/01-canonical-layout-and-init/01-CONTEXT.md` — canonical
  `<results-dir>/<mode>/<orgname>/systems/<systemname>.yaml` path; Phase 5 reads from
  and (in the LIFE-04 no-touch case) leaves alone the file Phase 2 wrote.

### Code references (existing infrastructure Phase 5 reuses)

- `mlpstorage_py/system_description/auto_generator.py:295` — `node_dict_from_host(host)`
  IS the round-trip recompute target for D-37. Phase 5 calls this for each in-memory
  host and compares the result against the on-disk stanza.
- `mlpstorage_py/system_description/auto_generator.py:492` — `_splice_stub_lists(dump)`
  applies D-33 drives-omit + D-17 traffic splice; the round-trip recompute output
  passes through this same splice for symmetry with what was originally written.
- `mlpstorage_py/system_description/auto_generator.py:615-625` — comment block
  explicitly anticipates Phase 5: "FileExistsError → return None + logger.debug
  (**Phase 5 will replace this branch with diff-and-fail**)". This is the LIFE-02 hook
  point.
- `mlpstorage_py/system_description/auto_generator.py:655-720` — `write_systemname_yaml`
  current shape; Phase 5 extends with a "if file exists → load, recompute, diff, raise
  or no-op" branch before the O_CREAT|O_EXCL write call.
- `mlpstorage_py/system_description/schema_validator.py:487` — existing
  `yaml.safe_load(source)` pattern; reusable for loading the on-disk YAML during the
  LIFE-02 diff.
- `mlpstorage_py/rules/models.py:171-260` — `HostInfo` dataclass (extended through
  Phase 4 to its current shape); the diff operates on `HostInfo`-derived dicts via
  `node_dict_from_host`.
- `mlpstorage_py/rules/utils.py:49` — `calculate_training_data_size(args,
  cluster_information, dataset_params, reader_params, logger, ...)` is the
  required-bytes source for CAP-01 training-mode checks.
- `mlpstorage_py/validation_helpers.py:402-450` — `check_disk_space(path,
  required_bytes, logger) -> bool` already does the `os.statvfs` math + raises
  `FileSystemError`. Phase 5 CAP-01 wraps or augments this; planner's call.
- `mlpstorage_py/cluster_collector.py:1656` — `MPI_COLLECTOR_SCRIPT` template; CAP-02
  fsid collection joins this script under Pattern B, OR ships a separate smaller probe
  script (planner picks).
- `mlpstorage_py/benchmarks/base.py:155` — `self._cluster_info_start = None`
  initialization (Phase 2 02-06 gap-closure). Phase 5 reads `_cluster_info_start`
  through the same lifecycle hook Phase 2 wired (`write_systemname_yaml` call at
  base.py:1010).
- `mlpstorage_py/benchmarks/base.py:1005-1010` — current Phase 2 hook site where
  `write_systemname_yaml` runs after `_collect_cluster_start`. Phase 5 LIFE-02/03/04
  extends the existing hook; no new call sites.
- `mlpstorage_py/storage_config.py` — D-25 unified redactors (`_redact_secret`,
  `_mask_credential_id`). Phase 5 inherits the already-redacted environment values
  through `HostInfo.environment` — no fresh redaction logic in Phase 5.

### Schema files

- `mlpstorage_py/system_description/schema.yaml` — Yamale schema; Phase 5 does NOT
  modify. The schema's validation pass (D-15 NOT called at runtime) remains the
  submitter-facing surface for SER-02 blanks; Phase 5's drift detection is an
  earlier-stage guardrail.

### Existing exception classes Phase 5 extends

- `mlpstorage_py/errors.py` (or equivalent — researcher to confirm location) —
  `MlpStorageError` base class + existing `ConfigurationError`, `FileSystemError`. Phase
  5 adds `SystemDriftError` (D-42) and `SystemDescriptionParseError` (D-48) as
  siblings. Planner verifies the actual module path during pattern mapping.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`node_dict_from_host`** (`auto_generator.py:295`) — IS the round-trip recompute
  target for D-37. Phase 5 calls it for each in-memory host and matches the result
  against the on-disk stanza by fingerprint (D-38). No new emit code in Phase 5; the
  diff IS a comparison of already-Pydantic-shaped dicts.
- **`group_by_fingerprint`** (`auto_generator.py`) — Phase 2/3/4 generic helper with
  D-22 callable extractors. Phase 5 uses it on the in-memory side to produce the same
  stanza shape the on-disk YAML carries (recomputed quantity, recomputed sort order).
- **`_FINGERPRINT_KEYS`** (`auto_generator.py`) — Phase 4 11-tuple lock (D-34). Phase
  5's D-38 fingerprint match relies on this remaining stable across the read-from-disk
  → recompute-from-memory comparison.
- **`yaml.safe_load`** (existing in `schema_validator.py:487`) — the LIFE-02 on-disk
  loader. Phase 5 imports the same module-level `yaml` and uses `safe_load` with the
  same defaults to keep the round-trip symmetric with the D-10 emit policy.
- **`validation_helpers.py:check_disk_space`** — CAP-01 free-space check; Phase 5
  wraps or augments (planner's call) to emit the four-field error format LIFE-03 (...
  CAP-01) mandates.
- **MPI gather + `MPI_COLLECTOR_SCRIPT`** (`cluster_collector.py:1656`) — Phase 5 either
  folds the CAP-02 fsid collection into this script (Pattern B per D-36) or ships a
  smaller separate probe.

### Established Patterns

- **D-2 universal collection-failure rule** — applies to COLLECTOR failures only (Phase
  2 02-04 docstring explicitly excludes filesystem failures from this rule). Phase 5
  CAP-01/CAP-02 failures are filesystem failures, NOT collector failures — they raise
  per D-45 (CAP-02) and per existing FileSystemError semantics (CAP-01); D-2 does NOT
  apply.
- **D-9 atomic write + OSError propagation** — Phase 5 LIFE-04 no-touch path means the
  D-9 write is skipped when diff is empty; the FileExistsError branch (currently a
  no-op) becomes the LIFE-02 diff entry point.
- **D-12 `args.command == 'run'` write gate** — extended in Phase 5: the LIFE-02 diff
  also runs only on `run`. `datagen` does not load, diff, or touch the systemname.yaml
  (carried verbatim from Phase 2).
- **Pattern B MPI script duplication (D-36)** — applies to any new probe Phase 5 adds
  to `MPI_COLLECTOR_SCRIPT`. The CAP-02 probe (D-43 `os.stat`-based) duplicates inline
  under the same untyped-form convention.
- **`MlpStorageError` hierarchy** — D-42 `SystemDriftError` and D-48
  `SystemDescriptionParseError` join the existing error tree as siblings of
  `ConfigurationError` and `FileSystemError`. `main.py`'s top-level handler covers them
  via the base class.

### Integration Points

- **`Benchmark.run()` → `write_systemname_yaml(args, cluster_info_start, logger)`**
  (`base.py:1010`) — Phase 5 LIFE-02/03/04 lives inside the same call. The function's
  internal flow grows a new branch: "if file exists → load → fingerprint-match each
  in-memory stanza against on-disk → if any mismatch raise SystemDriftError, else
  return None (no-touch)" before the existing O_CREAT|O_EXCL write path.
- **`Benchmark.run()` and `Benchmark._run()`** (any benchmark class — DLIO, VectorDB,
  KVCache, Checkpointing) — CAP-01 and CAP-02 fire BEFORE any of the existing
  `_run()` payload work. Planner picks the precise hook point (likely a new
  pre-execution gate method on the base class so all benchmark classes inherit it).
- **`datagen` entry points** — `dlio.py:444 generate_datagen_benchmark_command`,
  `vectordbbench.py:694 execute_datagen`, `kvcache.py:_run`. CAP-01 + CAP-02 fire
  before these; LIFE-02/03/04 does NOT (D-12 + Phase 2 confirmed datagen never touches
  systemname.yaml).
- **`MPI_COLLECTOR_SCRIPT`** (`cluster_collector.py:1656`) — possible CAP-02 fsid probe
  integration point if planner picks the folded-into-existing-script option.
- **`Benchmark.__init__`** (`base.py`) — Phase 5 may add a `_run_uuid` instance attr
  (the `<run-uuid>` referenced by D-43) so the sentinel name is deterministic across
  rank 0's write and peers' stat within a single invocation.

</code_context>

<specifics>
## Specific Ideas

- **Reuse the Phase 2 → Phase 4 layered pattern.** Phase 5's diff sub-feature should
  mirror the Phase 2 → 3 → 4 layered shape — collector layer (already done — Phase 2-4
  produce the in-memory image), transform layer (already done — `node_dict_from_host`,
  `group_by_fingerprint`), and a new diff layer that compares two transform-layer
  outputs. The recommended slice ordering (planner has final say) is:
  - Slice 1 (LIFE-02 core diff function): `diff_node_dict_lists(on_disk_stanzas,
    in_memory_stanzas) → DiffResult` plus the unified-diff formatter. Pure-function,
    fully unit-testable without filesystem or MPI involvement.
  - Slice 2 (LIFE-02 wiring + `SystemDriftError` + `SystemDescriptionParseError`):
    Extend `write_systemname_yaml` with the load-then-diff branch; add the two new
    exception classes; wire `main.py`'s top-level handler.
  - Slice 3 (CAP-01): `check_disk_space` wrapper or augmentation + per-benchmark
    `required_bytes` sourcing + integration into `_run()` pre-execution gate.
  - Slice 4 (CAP-02): sentinel + per-rank `os.stat` + MPI gather + barrier + 5s quiesce
    + Pattern B (or separate-probe — planner picks).
  - Slice 5 (end-to-end integration tests): real-multi-host smoke (where feasible) or
    realistically-mocked multi-host smoke covering all five Phase 5 ROADMAP success
    criteria including the per-mode independence (SC #4), the multi-host fsid
    cardinality (SC #7), and the single-host no-op (SC #8).

- **The `<run-uuid>` mechanism (D-43, D-44).** A single `uuid.uuid4().hex` value
  generated in `Benchmark.__init__` (or once per `run` invocation) and made available
  to both rank 0 (sentinel write) and the peers (stat path construction). Researcher
  to confirm the right hook point — probably the same lifecycle as
  `_cluster_info_start` so a single MPI gather can pass the uuid alongside the fsid.

- **Per-mode independence (SC #4).** The drift check is per-mode because each mode
  owns its own systemname.yaml at its own path (D-11 stays). No special-case logic
  needed in the diff — the path derivation already separates them. Document this
  explicitly in the Phase 5 slice docstrings so verification doesn't miss the cross-
  mode independence test.

- **The 5-second quiesce constant (D-49).** Hard-code `5.0` seconds for v1. If real
  shared-storage deployments surface a need for a longer quiesce (slow NFS metadata
  invalidation, etc.), revisit as a future config knob — not in scope here.

</specifics>

<deferred>
## Deferred Ideas

- **Configurable quiesce duration** — D-49 hard-codes 5 seconds. If a real submitter
  deployment surfaces a need for a longer post-unlink wait (very slow NFS metadata,
  etc.), expose as a `--cap02-quiesce-seconds` flag or `MLPERF_CAP02_QUIESCE_SECONDS`
  env var. Not in scope for Phase 5.

- **JSON-formatted drift report alongside human-readable one** — Tooling (CI, the
  future submission-checker integration) might want a structured drift report.
  Currently the report is human-readable only via `SystemDriftError.message`. Future
  phase could add a `to_json()` method on `SystemDriftError` for machine consumption.
  Not in scope for v1.

- **`mlpstorage validate` extension surfacing SER-02 blanks (SCH-01)** — v2 milestone.

- **Reportgen automatic inclusion of systemname.yaml in submission bundle (BUN-01)** —
  v2 milestone.

- **`mlpstorage init --adopt` for migrating existing non-initialized results-dirs
  (ADP-01)** — v2 milestone.

- **Drift detection with "soft" / "warning" severity** — Some submitters might want
  a knob to demote certain field changes from hard fail to warning (e.g., sysctl
  drifts that don't affect benchmark numbers). v1 ships strict-fail (D-35 carries
  forward); revisit if real demand surfaces.

- **Auto-suggested new `--systemname` value** — When drift is detected, the remediation
  hint could suggest a candidate `--systemname` derived from the existing one (e.g.,
  `sys-v1` → `sys-v2`). v1 ships the generic "rename" hint; auto-suggestion is a
  future UX polish.

</deferred>

---

*Phase: 5-logical-diff-lifecycle-capacity-gate*
*Context gathered: 2026-06-23*
