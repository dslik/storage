---
phase: 05-logical-diff-lifecycle-capacity-gate
reviewed: 2026-06-24T00:00:00Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - mlpstorage_py/system_description/diff.py
  - mlpstorage_py/errors.py
  - mlpstorage_py/system_description/auto_generator.py
  - mlpstorage_py/benchmarks/capacity_gate.py
  - mlpstorage_py/benchmarks/base.py
  - mlpstorage_py/benchmarks/dlio.py
  - mlpstorage_py/benchmarks/vectordbbench.py
  - mlpstorage_py/benchmarks/kvcache.py
  - mlpstorage_py/cluster_collector.py
  - tests/unit/test_diff.py
  - tests/unit/test_errors.py
  - tests/unit/test_auto_generator_write.py
  - tests/unit/test_capacity_gate.py
  - tests/unit/test_shared_fs_probe.py
  - tests/integration/test_systemname_yaml_end_to_end.py
  - tests/integration/test_shared_fs_probe_real_mpi.py
findings:
  critical: 2
  warning: 6
  info: 4
  total: 12
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-06-24
**Depth:** standard
**Files Reviewed:** 16
**Status:** issues_found

## Summary

Phase 5 lands three new surfaces — the pure-function diff core, the
`SystemDriftError`/`SystemDescriptionParseError` exception siblings + drift wiring
in `write_systemname_yaml`, and the CAP-01/CAP-02 pre-execution gates with the
new `SHARED_FS_PROBE_SCRIPT` MPI heredoc. The diff core, exception classes, and
CAP-01 surface are clean and tightly tested. The drift wiring in
`auto_generator.py` is solid. The CAP-02 surface (`run_shared_fs_probe` launcher
in `cluster_collector.py`) has two correctness/security blockers and several
operational defects that the existing test suite does not exercise because the
real-MPI test only runs ranks on localhost.

Two BLOCKERS surfaced:

1. **CAP-02 launcher uses `shell=True` with unsanitized `destination`,
   `run_uuid`, and `output_file` interpolated into the command string** — a
   submitter who supplies `--data-dir` containing shell metacharacters (or a
   path with embedded spaces, which is common in shared-storage mount layouts)
   triggers either argument splitting or arbitrary command execution.
2. **CAP-02 launcher writes `output_file` to a `tempfile.mkstemp` path on the
   launching host, but rank 0 of the mpirun job may land on a remote host** —
   in the typical multi-host configuration this means rank 0 writes the JSON
   result to a path that does not exist on the launch host, and the launcher
   then raises a misleading "mpi4py not installed on all hosts" error even
   when the probe semantically succeeded.

Six warnings cover tempfile leaks, an MPI deadlock window in the probe script,
empty-container path-flattening in the diff, and a few quality issues.

## Critical Issues

### CR-01: CAP-02 launcher passes user-controlled paths through `shell=True` without quoting

**File:** `mlpstorage_py/cluster_collector.py:3393-3441`
**Issue:** `run_shared_fs_probe` builds the mpirun invocation by joining
`cmd_parts` with whitespace and then invoking `subprocess.run(cmd_str,
shell=True, ...)`. The following positions are user-influenced and
interpolated unquoted:

- `destination` — derived from `args.data_dir` (Training) /
  `args.checkpoint_folder + args.model` (Checkpointing) / `args.cache_dir`
  (KVCache). All three CLI flags accept arbitrary strings.
- `run_uuid` — currently from `uuid.uuid4().hex` (hex-only), but the launcher's
  public contract is "caller-supplied UUID, passed verbatim"; if a future
  caller passes a non-hex string the interpolation is unsafe.
- `output_file` — `tempfile.mkstemp` output (controlled), but still unquoted.
- `remote_script_path` — derived from `staging_results_dir`
  (`tempfile.mkdtemp`); controlled today but unquoted at the boundary.
- `mpi_executable` — when `mpi_bin != MPIRUN and mpi_bin != MPIEXEC`, this is
  the raw `args.mpi_bin` value from the CLI.

Concrete failure modes:

- A submitter who passes `--data-dir "/mnt/scratch foo"` (a path with a
  literal space — common in customer environments) sees mpirun receive
  `"/mnt/scratch"` as the data_dir argv and `"foo"` as the next positional,
  which silently scrambles the probe.
- A submitter who passes `--data-dir "/tmp; touch /tmp/pwn"` triggers
  arbitrary command execution on the launching host (the resulting cmd_str
  contains `... /tmp; touch /tmp/pwn unique-uuid /tmp/out.json`).

The existing `MPIClusterCollector` uses the same `shell=True` pattern, so
this is not novel — but Phase 5 expands the attack surface to a new entry
point and a new set of `args.*` paths that previously did not flow through
shell.

**Fix:** Use `shell=False` with a list of args, OR quote every interpolated
position with `shlex.quote`. Suggested patch:

```python
import shlex
cmd_parts = [
    mpi_executable,
    "-n", str(n),
    "-host", host_slots,
    "--bind-to", "none",
    "--map-by", "node",
]
if allow_run_as_root:
    cmd_parts.append("--allow-run-as-root")
cmd_parts += [
    "python3",
    remote_script_path,
    destination,
    run_uuid,
    output_file,
]
# Either drop shell=True:
result = subprocess.run(cmd_parts, capture_output=True, text=True,
                        timeout=timeout_seconds, env=env)
# Or quote when shell=True is required (e.g., to pick up PLM_RSH_AGENT
# expansion inside mpirun's argv parsing):
cmd_str = " ".join(shlex.quote(p) for p in cmd_parts)
```

If `shell=True` must stay for environmental reasons, add `shlex.quote()` on
EVERY interpolated string including `destination`, `run_uuid`,
`output_file`, `remote_script_path`, and `mpi_executable`.

### CR-02: CAP-02 `output_file` is on a launch-host-local tempfs but rank 0 may run on a remote host

**File:** `mlpstorage_py/cluster_collector.py:3312-3322, 3417, 3454-3467, 3469-3482`
**Issue:** The launcher creates `output_file` via
`tempfile.mkstemp(prefix="mlps_cap02_probe_out_", suffix=".json")`, which
returns a path on the LAUNCH HOST's filesystem (typically `/tmp/...`). It
then passes that path as the third positional to `python3 ... <destination>
<run_uuid> <output_file>` for every MPI rank.

Inside `SHARED_FS_PROBE_SCRIPT.main()`, line 2776:
```python
if rank == 0:
    try:
        with open(output_file, "w") as f:
            json.dump({...}, f, indent=2)
```

If mpirun assigns rank 0 to a remote host (which is the default for
`--map-by node` when `-host h1:1,h2:1,...` is used and h1 is not the launch
host, OR when the launching machine is not in the `unique_hosts` list at
all), rank 0 writes to `output_file` at that path on the REMOTE host. The
launcher then runs at line 3455:
```python
if not os.path.exists(output_file):
    msg = ("CAP-02: shared-FS probe failed to produce output; check "
           "mpi4py is installed on all hosts. ...")
```

This raises a misleading error claiming mpi4py is missing when the probe
actually succeeded — and silently misclassifies any CAP-02 result whenever
the destination IS a real shared filesystem but the launch host happens not
to be in the hosts list, OR rank 0 is mapped to a different node.

Concrete failure pattern (the user's actual deployment shape):

- Launch host is `submitter-laptop` (NOT a benchmark client).
- `--hosts client1,client2,client3` (none of which equal `submitter-laptop`).
- Probe staging succeeds via SCP.
- mpirun runs across 3 client hosts; rank 0 is `client1`.
- Rank 0 writes JSON to `/tmp/mlps_cap02_probe_out_xxx.json` on `client1`.
- Launcher on `submitter-laptop` reads `/tmp/mlps_cap02_probe_out_xxx.json`
  which doesn't exist there.
- Submitter sees "shared-FS probe failed; check mpi4py" on a working setup.

The existing `MPIClusterCollector` solves this by writing both the script
and the output to `<results_dir>/collector-staging/`, which is replicated
to every host via the same staging helper. The probe inherits the script
side of that pattern but NOT the output side.

The real-mpi integration test
(`tests/integration/test_shared_fs_probe_real_mpi.py`) runs both ranks on
localhost only, so this defect is invisible at the unit/integration layer.

**Fix:** Place `output_file` inside the same `staging_dir` that
`_stage_script_on_remote_hosts` replicates onto every host. Either:

```python
output_file = os.path.join(staging_dir, "cap02_probe_output.json")
# (and ensure staging_dir is created and SCP'd to every remote host
# alongside remote_script_path, OR require the destination to itself
# be a shared FS — which is exactly what CAP-02 is verifying so the
# argument is circular; the safe path is the SCP-replicated staging_dir).
```

Then on the launch host, read the file back from the same staging_dir on
the host where rank 0 ran — which requires either an SCP-fetch after the
mpirun OR a shared staging dir (mirroring `MPIClusterCollector`'s
`shared_staging_dir` parameter).

The minimum-correct fix: extend the launcher to discover which host rank 0
landed on (parse it out of mpirun's host-binding or write a hostname file
inside the staged dir from rank 0) and SCP the output back from there
before reading.

## Warnings

### WR-01: Possible MPI deadlock if rank-0 raises after gather but before bcast

**File:** `mlpstorage_py/cluster_collector.py:2677-2750`
**Issue:** Inside `SHARED_FS_PROBE_SCRIPT.main()`, Step E
(`if rank == 0: ... rank0_failure_summary = {...}`) is followed by Step F
`status = comm.bcast(status, root=0)` (line 2749) inside the same `try`
block. If rank 0 raises in Step E (for example, `all_payloads` is somehow
`None` because gather degraded, or `_build_cardinality_message` raises on a
malformed payload), control jumps to the `finally` block on rank 0 WITHOUT
executing the `bcast`. Non-rank-0 ranks have already completed their gather
and proceed into `bcast`, where they block waiting for a sender that never
calls bcast. The fleet hangs until mpirun's timeout fires.

The launcher's `timeout_seconds=60` does eventually fire, but the operator
sees an opaque "shared-FS probe timed out" error rather than the underlying
exception that caused rank 0 to abort.

**Fix:** Wrap Step E rank-0 analysis in its own try/except so any exception
sets `status='fail'` with a synthetic failure_summary, ensuring bcast is
always reached:

```python
if rank == 0:
    try:
        any_failure = any(p.get("failure") is not None for p in all_payloads)
        if any_failure:
            ...
        else:
            ids = set()
            for p in all_payloads:
                ids.add((p.get("st_dev"), p.get("st_ino")))
            ...
    except Exception as e:
        status = "fail"
        rank0_failure_summary = {
            "kind": "internal",
            "message": "CAP-02: rank-0 internal error: {0}".format(e),
        }
```

### WR-02: Tempfiles and staging dirs leak on every CAP-02 probe invocation

**File:** `mlpstorage_py/cluster_collector.py:3308-3344, 3433-3525`
**Issue:** `run_shared_fs_probe` creates:
1. `local_script_path` via `_write_probe_script_to_tempfile` (line 3308).
2. `output_file` via `tempfile.mkstemp` (line 3313).
3. `staging_results_dir` and `staging_dir` via `tempfile.mkdtemp` (line 3344).

None of these are cleaned up on either the happy path (status='ok' return)
or the failure path (any of the `raise FileSystemError` branches). Each
benchmark `run` leaves three tempfiles/dirs behind. On long-running CI or
nightly-regression hosts these accumulate without bound.

**Fix:** Wrap the body in `try/finally` and remove the three artifacts in
the finally block:

```python
local_script_path = _write_probe_script_to_tempfile(SHARED_FS_PROBE_SCRIPT)
out_fd, output_file = tempfile.mkstemp(...)
os.close(out_fd)
staging_results_dir = None
try:
    # ... existing body ...
finally:
    for p in (local_script_path, output_file):
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except OSError:
            pass
    if staging_results_dir:
        try:
            shutil.rmtree(staging_results_dir, ignore_errors=True)
        except Exception:
            pass
```

### WR-03: `_flatten_to_paths` silently drops keys whose value is an empty container

**File:** `mlpstorage_py/system_description/diff.py:142-168`
**Issue:** `_flatten_to_paths` returns nothing for empty dicts and empty
lists (lines 155-156, 161-162). This means a key whose VALUE is an empty
container does not appear in the path set at all. Consequence: if on-disk
has `chassis: {model: "X", optional_blob: {}}` and in-memory has
`chassis: {model: "X"}` (omitting `optional_blob` entirely), the diff
treats these as equal because neither yields a path for `optional_blob`.
Similarly for `field: []` versus omitted-`field`.

For the current 7-key emit shape (`friendly_description, chassis,
networking, sysctl, environment, drives, operating_system`) this is benign
because D-33 (drives omit-when-empty) plus the existing `_splice_stub_lists`
symmetry pass make both sides emit the same shape. But the behavior is a
silent invariant that a future field addition could break — D-37's
self-maintaining promise depends on the diff layer surfacing structural
asymmetry, which this case quietly does not.

**Fix:** Either (a) document this contract explicitly in the docstring, or
(b) emit a sentinel `(prefix, _EMPTY_CONTAINER)` for empty containers so
they participate in the path union:

```python
if isinstance(value, dict):
    if not value:
        yield (prefix, "<empty_dict>")  # or a module-level sentinel
        return
    ...
elif isinstance(value, list):
    if not value:
        yield (prefix, "<empty_list>")
        return
    ...
```

Today's tests would still pass with the sentinel approach because the
splice symmetry ensures both sides emit the same empty shape.

### WR-04: CAP-02 launcher does not validate `run_uuid` shape, accepting shell metacharacters

**File:** `mlpstorage_py/cluster_collector.py:3251-3296, 3393-3419`
**Issue:** `run_shared_fs_probe(run_uuid=...)` accepts any string and
interpolates it into the cmd_str (and into the sentinel filename via the
script body at line 2666: `sentinel = os.path.join(data_dir,
".mlpstorage-shared-fs-probe-" + run_uuid)`). The current production caller
sets `self._run_uuid = uuid.uuid4().hex` (32 hex chars — safe), but the
W-5 contract is "caller-supplied UUID passed verbatim." A future caller
that pulls the UUID from less trusted input (CLI flag, env var,
results-dir filename) could introduce path traversal in the sentinel name
(`..` segments) or shell injection at the cmd_str level (compounding
CR-01).

**Fix:** Add a one-line validation at the launcher entry:

```python
import re
if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_uuid):
    raise ValueError(
        f"run_uuid must be alphanumeric/underscore/dash only; got {run_uuid!r}"
    )
```

### WR-05: `CheckpointingBenchmark.required_bytes_for_capacity_gate` raises bare `ValueError` instead of project exception

**File:** `mlpstorage_py/benchmarks/dlio.py:661-663`
**Issue:** The capacity-gate hook contains:
```python
else:
    raise ValueError("Invalid zero_level")
```
Per the project's exception conventions (CLAUDE.md + `mlpstorage_py/errors.py`),
internal errors should route through `MLPStorageException` subclasses so the
top-level handler (`main.py:262`) produces a structured error footer with
error code, suggestion, and exit code. A raw `ValueError` here propagates up
through `_pre_execution_gate` to `Benchmark.run()` and trips the generic
"Failed to write systemname.yaml" except branch indirectly (or escapes via
the `datasize` method's try/except which catches `Exception` and returns
`EXIT_CODE.FAILURE` without surfacing the message structurally).

**Fix:**
```python
from mlpstorage_py.errors import ConfigurationError, ErrorCode
...
else:
    raise ConfigurationError(
        f"Invalid zero_level={zero_level} for model {self.args.model}",
        parameter="zero_level",
        code=ErrorCode.CONFIG_INVALID_VALUE,
    )
```

### WR-06: `Benchmark._pre_execution_gate` calls `self.logger.info` on every VectorDB run, defeating SC#6 silence

**File:** `mlpstorage_py/benchmarks/base.py:995-1005`
**Issue:** The A8 escape hatch logs at INFO level whenever the destination
is None:
```python
if destination is None:
    self.logger.info(
        "CAP-01 skipped: destination not local "
        "(e.g., remote vector-DB backend)"
    )
    return
```

This is intentional per the plan, but conflicts with the SC#6 happy-path
silence contract documented at the same call site. VectorDB runs ALWAYS
emit this INFO line (because `_capacity_gate_destination` is hard-coded to
return None per A8), so the SC#6 "no logger output on success" contract
only holds for non-VectorDB benchmarks. The SUMMARY's framing of "the gate
is always invoked; the destination decides what it does" is reasonable,
but the silence contract is loaded with a documented exception that the
README/operator-docs do not surface.

**Fix:** Either (a) drop the INFO line to debug (so SC#6 holds uniformly
and curious operators get the skip notice via `--verbose`), or (b) update
the SC#6 contract in REQUIREMENTS.md to acknowledge the A8 exception.
Recommend (a):

```python
if destination is None:
    self.logger.debug(
        "CAP-01 skipped: destination not local "
        "(e.g., remote vector-DB backend)"
    )
    return
```

## Info

### IN-01: `_MAX_READ_BYTES` constant in `capacity_gate.py` is unused

**File:** `mlpstorage_py/benchmarks/capacity_gate.py:36-39`
**Issue:** The module defines `_MAX_READ_BYTES = 8192` with a comment
explaining it is "not used today (we only call os.statvfs)" but kept for
"parallel evolution." Dead code with no enforced contract — a future
contributor reading this module sees an unused constant and may either
delete it or wire it incorrectly.

**Fix:** Remove the constant until it has a consumer, or convert the
docstring note into a `# TODO(CAP-03): ...` comment.

### IN-02: Duplicate model-cache estimates table in `kvcache.py`

**File:** `mlpstorage_py/benchmarks/kvcache.py:303-310, 358-364`
**Issue:** The model cache estimates dict is defined twice — once as
`_MODEL_CACHE_ESTIMATES` class attribute (line 303) for the CAP-01 hook,
and again as a local `model_cache_estimates` inside `_execute_datasize`
(line 358). The two dicts are identical today but can drift independently.
The plan SUMMARY notes the divergence intentionally (CAP-01 vs. user-facing
recommendation) but the same dict literal appears verbatim in both places.

**Fix:** Have `_execute_datasize` use `self._MODEL_CACHE_ESTIMATES`:

```python
def _execute_datasize(self) -> int:
    self._pre_execution_gate()
    self.logger.status("Calculating KV Cache memory requirements...")
    model_info = self._MODEL_CACHE_ESTIMATES.get(
        self.model, self._MODEL_CACHE_DEFAULT
    )
    # ... rest of the function ...
```

### IN-03: `parse_on_disk_systemname_yaml` repeats "Remediation: rm <path> && re-run" in four raise sites

**File:** `mlpstorage_py/system_description/auto_generator.py:714-741`
**Issue:** Four structural-validation raise sites each rebuild the same
remediation hint string in the message body. The error class itself already
generates the same hint via `_default_suggestion(code, path)`, so the
inline string in the message is redundant.

**Fix:** Drop the `Remediation: rm {path} && re-run` line from each raise's
message — the `_default_suggestion` path-aware default already renders it
via the structured error formatter. Reduces duplication and prevents drift
between the two messages.

### IN-04: `SystemDriftError` and `SystemDescriptionParseError` re-set `self.path` after `super().__init__` already stored it via context

**File:** `mlpstorage_py/errors.py:347-352, 399-404`
**Issue:** Both new exception classes do:
```python
super().__init__(..., path=path)  # path lands in self.error.context['path']
self.path = path                  # AND on self.path
```
The duplicated assignment is intentional (matches the `FileSystemError`
pattern at line 295) but the comment block redundantly explains the same
"so tests/handlers can inspect without poking the context dict" rationale
on both new classes. The structural-error hierarchy now has three sibling
classes (`FileSystemError`, `SystemDriftError`, `SystemDescriptionParseError`)
that each carry an explicit `self.path` shadow attribute; a future
maintainer adding a fourth sibling has no shared abstraction to inherit.

**Fix:** Extract a private mixin or move the `self.path = path` line into
`MLPStorageException.__init__` when `path` is in context. Low priority —
the explicit shadows are working as designed. Logged here for the next
refactoring pass.

---

_Reviewed: 2026-06-24_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
