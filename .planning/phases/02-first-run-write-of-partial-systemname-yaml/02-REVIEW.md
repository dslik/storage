---
phase: 02-first-run-write-of-partial-systemname-yaml
reviewed: 2026-06-19T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - mlpstorage_py/rules/models.py
  - mlpstorage_py/system_description/auto_generator.py
  - mlpstorage_py/benchmarks/base.py
  - tests/unit/test_cluster_collector.py
  - tests/unit/test_auto_generator.py
  - tests/unit/test_auto_generator_write.py
  - tests/integration/test_systemname_yaml_end_to_end.py
  - tests/unit/test_benchmarks_kvcache.py
  - tests/unit/test_benchmarks_vectordb.py
findings:
  critical: 1
  warning: 5
  info: 4
  total: 10
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-06-19
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Phase 02 adds `write_systemname_yaml(args, cluster_info, logger)` and wires it
into `Benchmark.run()` so a populated `systemname.yaml` lands at the canonical
Rules.md §2.1.8 path before DLIO launches. The pure-transformation core
(`node_dict_from_host`, `group_by_fingerprint`, `_splice_stub_lists`,
`_build_outer_dict`) is well-isolated and the atomic write recipe correctly
mirrors `sentinel.py`'s `O_CREAT|O_EXCL|O_WRONLY` pattern.

However, the hook in `Benchmark.run()` has a load-bearing assumption that
fails in production: it reads `self._cluster_info_start`, but
`_collect_cluster_start()` only sets that attribute when `args.hosts` is
populated. The integration tests mask this by always mocking
`_collect_cluster_start` to seed the attribute. Any user invoking a benchmark
without `--hosts` (the single-node dev path, which the writer's own D-8
fallback was *designed* to serve) will see `AttributeError` and never reach
DLIO. Tests pass; production paths break.

Secondary concerns: the documented "Phase 1 enforces well-formed systemname"
contract is overstated (Phase 1 only enforces non-empty), so a maliciously
crafted `--systemname` value can traverse out of the results-dir at write
time; `args.dry_run` is not honored at the writer (inconsistent with
`capture_code_image` WR-09 and `_start_timeseries_collection`); and
`HostInfo.to_dict()` silently drops the new `num_sockets` field so any
ClusterInformation re-loaded from disk loses CPU-quantity grouping signal.

## Critical Issues

### CR-01: `Benchmark.run()` crashes with `AttributeError` when `args.hosts` is unset

**File:** `mlpstorage_py/benchmarks/base.py:991`
**Issue:**
`run()` calls

```python
write_systemname_yaml(self.args, self._cluster_info_start, self.logger)
```

but `self._cluster_info_start` is set only inside the `else` branches of
`_collect_cluster_start()` (lines 644-651). That method returns early at
lines 634-636 when *both* `_should_collect_cluster_info()` and
`_should_use_ssh_collection()` return `False` — and both gating predicates
return `False` when `args.hosts` is `None`/empty (lines 473-474, 564-565).

So the common single-node invocation (no `--hosts`, no MPI, no SSH) reaches
line 991 with the attribute *never assigned*. Python raises
`AttributeError: 'XBenchmark' object has no attribute '_cluster_info_start'`.
The error is then caught by the generic `except Exception as e:` block at
line 997, logged as `"Failed to write systemname.yaml: <error>"`, and
re-raised — aborting the benchmark *before DLIO ever launches*.

This is exactly the path the writer's D-8 fallback (`_resolve_host_info_list`
with `cluster_info=None`) was designed to handle, but the caller in
`Benchmark.run()` never gives the writer the chance to take that path.

The integration tests in `tests/integration/test_systemname_yaml_end_to_end.py`
and the per-benchmark hook regression tests in
`tests/unit/test_benchmarks_kvcache.py::TestKVCacheSystemnameYamlHook` /
`test_benchmarks_vectordb.py::TestVectorDBSystemnameYamlHook` *all* mock
`_collect_cluster_start` with a `side_effect` that explicitly assigns
`bm._cluster_info_start = cluster_info_mock`. None of them exercises the
unmocked early-return path. The bug is invisible to the test suite as
written.

**Fix:**
Either (a) initialize `self._cluster_info_start = None` in `__init__` so the
attribute is always present, or (b) use `getattr` at the call site so the
D-8 None path is exercised:

```python
# Option A — preferred, also closes a similar AttributeError window in
# _collect_cluster_end() at line 662 which uses `hasattr` defensively.
def __init__(self, args, logger=None, run_datetime=None, run_number=0,
             cluster_collector=None, validator=None):
    ...
    self._cluster_info_start = None  # populated by _collect_cluster_start
    self._cluster_info_end = None
    self._collection_method = None

# Option B — minimal patch at the call site:
write_systemname_yaml(
    self.args,
    getattr(self, '_cluster_info_start', None),  # D-8 None path
    self.logger,
)
```

A regression test that runs the lifecycle WITHOUT mocking
`_collect_cluster_start` (letting the real early-return execute) would have
caught this. Add one alongside the existing hook coverage.

## Warnings

### WR-01: `--systemname` accepts path-traversal payloads

**File:** `mlpstorage_py/system_description/auto_generator.py:461-467`
**Issue:**
The docstring on `write_systemname_yaml` (lines 417-421) asserts that

> Phase 1's `generate_output_location` already enforces non-empty +
> well-formed systemname via the upstream `_reserve_run_directory` call

The "non-empty" part is true (`rules/utils.py:187-195`). The "well-formed"
part is **not**: there is no character allowlist, no regex check, no path-
traversal guard anywhere in `cli/common_args.py`, `cli_parser.py:274-307`,
or `rules/utils.py:122-220`. A user passing
`--systemname '../../../../tmp/pwned'` will see the writer compute the path

```
Path(args.results_dir) / args.mode / args.orgname / "systems" / "../../../../tmp/pwned.yaml"
```

and `os.open(..., O_CREAT|O_EXCL|O_WRONLY)` will happily create
`/tmp/pwned.yaml` (provided it does not already exist). `O_EXCL` blocks
*overwrites* but does not block creation outside the intended directory. The
same vulnerability applies to `args.mode` and `args.orgname` — neither is
syntactically validated.

In a CI/orchestration setting where `--systemname` is plumbed from an
environment variable or a job spec, this is reachable by a user without
filesystem access to the host running `mlpstorage`. The blast radius is
limited to wherever the running user can write.

**Fix:**
Either validate `systemname` (and `mode`, `orgname`) at the CLI / writer
entry, or canonicalize and bounds-check the resolved path:

```python
import re
_SYSTEMNAME_RE = re.compile(r'^[A-Za-z0-9._-]+$')

def write_systemname_yaml(args, cluster_info, logger):
    if getattr(args, "command", None) != "run":
        return None
    if not _SYSTEMNAME_RE.match(args.systemname):
        raise ConfigurationError(
            f"--systemname must match {_SYSTEMNAME_RE.pattern}; got {args.systemname!r}",
            code=ErrorCode.CONFIG_INVALID,
        )
    # …also validate args.mode in {"closed","open","whatif"} and args.orgname.
```

Either way, fix the misleading docstring claim in the same change — Phase 5
should not inherit the false premise that "Phase 1 already validated this".

### WR-02: `--dry-run` does not skip systemname.yaml write

**File:** `mlpstorage_py/benchmarks/base.py:990-1003`
**Issue:**
Phase 1's `WR-09` fix established the rule that pre-DLIO side effects skip
on `--dry-run`: `capture_code_image` is gated on
`getattr(self.args, 'dry_run', False)` at lines 171-181, and
`_should_collect_timeseries()` gates on the same flag at line 704. The
Phase 2 write hook is missing the corresponding gate — `write_systemname_yaml`
is called unconditionally as long as `args.command == 'run'`.

Result: `mlpstorage … run --dry-run` writes a real `systemname.yaml` to disk
even though no benchmark runs. On a second invocation (real run), the D-9
no-op-if-exists branch then refuses to overwrite the dry-run-generated file.
A user testing their setup with `--dry-run` poisons the subsequent real run
with a partial file based on whatever was collected at dry-run time.

**Fix:**
Skip the write in dry-run (mirroring `capture_code_image`):

```python
if getattr(self.args, 'dry_run', False) or getattr(self.args, 'what_if', False):
    self.logger.debug("Skipping systemname.yaml write (--dry-run / --what-if)")
else:
    try:
        write_systemname_yaml(self.args, self._cluster_info_start, self.logger)
    except Exception as e:
        self.logger.error(f"Failed to write systemname.yaml: {e}")
        raise
```

Alternatively, push the gate down into `write_systemname_yaml` itself —
either location is fine, but the inconsistency vs WR-09 / timeseries must be
resolved.

### WR-03: `HostInfo.to_dict()` drops the new `num_sockets` field

**File:** `mlpstorage_py/rules/models.py:271-277`
**Issue:**
Phase 02-01 added `num_sockets` to `HostCPUInfo` and wired it into
`from_collected_data` (line 221) and `from_dict` (line 166). The serializer
`to_dict()` was not updated:

```python
if self.cpu:
    result['cpu'] = {
        'num_cores': self.cpu.num_cores,
        'num_logical_cores': self.cpu.num_logical_cores,
        'model': self.cpu.model,
        'architecture': self.cpu.architecture,
        # MISSING: 'num_sockets': self.cpu.num_sockets,
    }
```

This is the dict used to write `metadata['system_info']` (base.py line 406)
into the per-run JSON metadata file. If anyone later reconstructs
`ClusterInformation` from that metadata (via `from_dict` → `HostInfo.from_dict`
which reads `'cpu_info'`, a separate path), `num_sockets` is zero.

Even more directly: the cluster_info JSON file (`write_cluster_info` line
449-462) is the canonical on-disk record of CPU topology. With `num_sockets`
missing, downstream submission-checker tools and reports lose
chassis.cpu_qty fidelity for re-runs that consume the JSON instead of
re-collecting.

**Fix:**
Add the field to `to_dict()` and round-trip it through `from_dict`/`from_collected_data`:

```python
if self.cpu:
    result['cpu'] = {
        'num_cores': self.cpu.num_cores,
        'num_logical_cores': self.cpu.num_logical_cores,
        'model': self.cpu.model,
        'architecture': self.cpu.architecture,
        'num_sockets': self.cpu.num_sockets,
    }
```

Add a unit test that round-trips a `HostInfo` through `to_dict()` → JSON →
`HostInfo.from_dict()` and asserts `num_sockets` survives. The existing
test_cluster_collector.py::TestHostCPUInfoNumSockets only covers
construction-time wiring, not the serialization roundtrip.

### WR-04: Generic `except Exception` in write hook masks bugs as I/O failures

**File:** `mlpstorage_py/benchmarks/base.py:997-1003`
**Issue:**
The catch-all clause

```python
except Exception as e:
    self.logger.error(f"Failed to write systemname.yaml: {e}")
    raise
```

is documented as catching filesystem errors (EACCES, ENOSPC) per D-9, but a
bare `except Exception` also swallows the *type* of programmer errors like
`AttributeError` (see CR-01), `TypeError`, `KeyError`, etc., relabeling them
as "Failed to write systemname.yaml" before re-raising. Combined with CR-01,
this is precisely why the underlying attribute bug would be misdiagnosed in
the field as a filesystem permission problem.

The docstring on `write_systemname_yaml` (lines 437-440) explicitly says
that only filesystem errors should propagate; programmer-error categories
should fail loudly. The base.py wrapper should mirror that intent.

**Fix:**
Narrow the catch to filesystem-level exceptions:

```python
try:
    write_systemname_yaml(self.args, self._cluster_info_start, self.logger)
except FileExistsError:
    raise  # D-9 in-writer handling; bubble-up is unexpected
except OSError as e:
    # EACCES / ENOSPC / IsADirectoryError etc.
    self.logger.error(f"Failed to write systemname.yaml: {e}")
    raise
# Any other exception (AttributeError, TypeError, KeyError, …) propagates
# without being relabeled as an I/O failure — these are programmer errors,
# not collector or filesystem failures.
```

### WR-05: D-8 fallback runs the local collector with no error handling

**File:** `mlpstorage_py/system_description/auto_generator.py:361-378`
**Issue:**
`_resolve_host_info_list` calls

```python
local_data = collect_local_system_info()
return [HostInfo.from_collected_data(local_data)]
```

with no try/except. `collect_local_system_info()` itself swallows per-file
errors into `result['errors'][…]` (cluster_collector.py lines 656-754), so
the top-level call generally won't raise — but `HostInfo.from_collected_data`
calls `summarize_cpuinfo(cpuinfo)` which can raise on malformed
`/proc/cpuinfo` content, and the subsequent `HostSystemInfo(...)`
construction has positional requirements that can `TypeError` if the data
shape drifts.

The "universal collection-failure rule" (CONTEXT.md D-2) demands that
collector failures degrade to empty strings, not raise. As written,
`_resolve_host_info_list` violates that rule for the D-8 fallback path: a
collector exception there aborts `Benchmark.run()` (via the WR-04
broad-except), instead of writing a YAML with blank fields. The unit test
`test_resolve_host_info_list_none_triggers_collector` only exercises the
happy path.

**Fix:**
Wrap the fallback in a try/except that falls through to a synthetic
empty-field HostInfo:

```python
try:
    local_data = collect_local_system_info()
    return [HostInfo.from_collected_data(local_data)]
except Exception as e:
    # D-2 universal collection-failure rule: degrade to a blank HostInfo
    # so write_systemname_yaml emits a YAML the submitter can fill in,
    # rather than aborting the benchmark.
    import socket
    return [HostInfo(hostname=socket.gethostname() or "unknown")]
```

Add a unit test that patches `collect_local_system_info` to raise and
asserts the writer still produces a YAML with a single blank stanza.

## Info

### IN-01: `import copy` inside `_apply_dotted_overrides` body

**File:** `mlpstorage_py/benchmarks/base.py:351`
**Issue:**
`_apply_dotted_overrides` performs `import copy` inside its body even though
the function is on a hot path (called via `metadata` property on every
write_metadata call). Top-of-file already imports `copy` implicitly nowhere
— but the rest of the module uses `import copy` style globally elsewhere
would be cleaner.

**Fix:**
Hoist to module-level imports (line 36-ish):

```python
import copy
```

Then drop `import copy` from line 351. Minor maintainability nit.

### IN-02: `_unique_run_datetime` produces non-canonical datetime strings

**File:** `tests/integration/test_systemname_yaml_end_to_end.py:117-126`
**Issue:**
`_unique_run_datetime` returns strings shaped `YYYYMMDD_HHMMSS_NNNN` (four
extra characters). `mlpstorage_py.config.DATETIME_STR` is shaped
`YYYYMMDD_HHMMSS`. The benchmark `run_datetime` flows into filenames and
metadata fields that downstream tools may parse with a strict format
matcher. Tolerated today only because the collision reserver bumps on any
mismatch.

This is a test-only concern, but if a future regression depends on parsing
`run_datetime` back into a `datetime` object, this fixture quietly breaks
that parse. Prefer simulating timestamp uniqueness by stubbing
`DATETIME_STR` or driving via `monkeypatch.setattr` rather than producing a
non-canonical shape.

**Fix:**
Either canonicalize the format and accept the 1-second-per-test resolution,
or rely on `time.monotonic()` / `time.time_ns()`-based timestamps within the
proper format.

### IN-03: Unused `types` import in `benchmarks/base.py`

**File:** `mlpstorage_py/benchmarks/base.py:42`
**Issue:**
`import types` is present but `types` is never referenced anywhere in the
file. Probably leftover from a prior refactor. Pyflakes would catch this.

**Fix:**
Remove the unused import.

### IN-04: Stale `if benchmark_result:` comment in BenchmarkRun gives misleading guidance

**File:** `mlpstorage_py/rules/models.py:907`
**Issue:**
At line 855 of models.py, `_from_metadata` has

```python
if 'system_info' in metadata and metadata['system_info']:
    pass  # TODO: Reconstruct system_info
```

This is a silent drop: when the metadata path is used (the
`ResultFilesExtractor` fast path for complete metadata), `system_info` is
discarded with no logging. Any submission-checker rule that reads
`run.system_info` will see `None` for runs reconstructed from metadata even
though the data was right there. This is not new in Phase 02 but lives in a
file Phase 02 touched, and Phase 02 increases the value of `system_info` by
populating `num_sockets`.

**Fix:**
Implement the reconstruction (call `ClusterInformation.from_dict`) or at
least emit a logger warning so the silent loss is visible:

```python
if 'system_info' in metadata and metadata['system_info']:
    try:
        system_info = ClusterInformation.from_dict(
            metadata['system_info'], None
        )
    except Exception:
        system_info = None
```

---

_Reviewed: 2026-06-19_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
