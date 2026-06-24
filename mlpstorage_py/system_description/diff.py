"""Pure-function diff core for systemname.yaml drift detection — Phase 05 / Plan 05-01.

This module is the comparison-subject layer between the on-disk systemname.yaml
(parsed into Python dicts by Slice 2's `parse_on_disk_systemname_yaml`) and the
freshly recomputed in-memory client stanzas (produced by Phase 02-04's
`write_systemname_yaml` → `_resolve_host_info_list` → `node_dict_from_host`).
It contains NO filesystem I/O, NO MPI involvement, and NO exception raises —
the `SystemDriftError` raise lives at the Slice 2 call site, not here. This
module is a pure transformation: list of dicts in, structured diff out.

Phase 05 / Plan 05-01 deliverables (Slice 1 of the lifecycle vertical):

- `DiffEntry` — frozen dataclass (path, old, new) representing one row of
  difference between the on-disk and in-memory views (D-37 / D-40).

- `DiffResult` — dataclass wrapping `entries: list[DiffEntry]` with a
  convenience `.empty` property. Returned by `diff_node_dict_lists`.

- `_flatten_to_paths(value, prefix='')` — generator yielding
  `(jsonpath, leaf_value)` for every leaf in the nested dict/list. Dict
  children use `prefix.key`, list children use `prefix[index]`. Empty
  containers yield nothing.

- `_compute_fingerprint(stanza)` — reuses `_FINGERPRINT_KEYS` and
  `_resolve_fingerprint_key` from `auto_generator.py` so the diff layer uses
  the exact same 11-tuple identity rule Phase 4 settled on for
  quantity-grouping. The D-38 round-trip-recompute contract depends on this:
  the same client must hash to the same fingerprint whether emitted into the
  on-disk YAML or recomputed in memory.

- `diff_node_dict_lists(on_disk, in_memory)` — the load-bearing public
  function. Indexes both sides by fingerprint, then for each fingerprint:
  - present only on-disk → emit a `<present>` / `<absent>` orphan entry (D-47);
  - present only in-memory → emit a `<absent>` / `<present>` orphan entry (D-46);
  - present on both → flatten both sides, walk the union of paths, emit one
    DiffEntry per differing leaf with the SER-02 Pitfall 3 direction (a)
    blank-preservation skip applied: if `in_memory_value == ''` and the
    on-disk value is filled, that path is skipped (submitter-filled values
    are sacred when the collector returned blank).

- `format_unified_diff(result, on_disk_path)` — converts a `DiffResult` to a
  human-readable unified-diff-style string (D-40 / D-41). Shape:
  `--- on-disk: <path>` / `+++ in-memory: <computed from live MPI fleet>`
  headers; `@@ <JSONPath> @@` hunk markers; `- <old>` / `+ <new>` lines;
  trailing `Remediation:` block listing both the rename and rm hints.
  Values are emitted verbatim (no truncation, no repr-wrapping) so long
  sysctl tuples like `4096\\t87380\\t16777216` round-trip cleanly (D-41).

Architecture notes:

- The `_SENTINEL_ABSENT` module-level object is used to distinguish "field
  absent on this side" from "field present but empty string". This matters
  because the SER-02 blank preservation rule (Pitfall 3 direction (a)) only
  fires when in-memory IS the empty string AND on-disk has a filled value —
  not when in-memory is absent entirely.

- Fingerprint orphan paths use `clients[fingerprint=<repr>]` rather than a
  positional index because the on-disk and in-memory sides may have
  different cardinalities and positional indices would not correspond.
  Sorting by `repr(fp)` defends against `TypeError: '<' not supported
  between instances of 'X' and 'Y'` on mixed-type fingerprint tuples (the
  same defense Phase 3-04 / Plan 04-04 settled on for `_network_signature`).

- DiffEntry is `@dataclass(frozen=True)` so individual entries are immutable
  once constructed; DiffResult.entries is a list to permit Slice 2's call
  site to extend or replace as needed (T-5-01-02 threat register accepted).

Slice 2 (next plan) will import `diff_node_dict_lists` and
`format_unified_diff` from this module — the public API is now LOCKED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from mlpstorage_py.system_description.auto_generator import _FINGERPRINT_KEYS, _resolve_fingerprint_key


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
__all__ = [
    "DiffEntry",
    "DiffResult",
    "diff_node_dict_lists",
    "format_unified_diff",
]


# ---------------------------------------------------------------------------
# Module-level sentinel used to disambiguate "field absent on this side"
# from "field present but empty string". `object()` is a unique identity
# distinct from every value the collector could ever emit.
# ---------------------------------------------------------------------------
_SENTINEL_ABSENT: Any = object()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffEntry:
    """One row of difference between on-disk and in-memory views.

    `path` is a JSONPath-style dotted/bracketed string (e.g.
    `clients[0].chassis.cpu_model` or `clients[fingerprint=(...)]`).
    `old` and `new` are the on-disk and in-memory leaf values respectively;
    sentinel strings `"<present>"` / `"<absent>"` are used at the
    fingerprint-orphan level when a whole stanza is present on only one side.
    """

    path: str
    old: Any
    new: Any


@dataclass
class DiffResult:
    """Wrapper around `entries: list[DiffEntry]` with a `.empty` property.

    Mutable on purpose: Slice 2's call site may want to extend the entries
    list with synthetic header entries before rendering. T-5-01-02 in the
    plan's threat register accepts this trade-off because the result is
    constructed per-call and not stored in shared state.
    """

    entries: list[DiffEntry] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return len(self.entries) == 0


# ---------------------------------------------------------------------------
# _flatten_to_paths — recursive generator over nested dict/list structures.
# ---------------------------------------------------------------------------


def _flatten_to_paths(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield (jsonpath, leaf_value) pairs for every leaf in `value`.

    Dict children: `f"{prefix}.{k}"` if prefix else `k`.
    List children: `f"{prefix}[{i}]"`.
    Empty dict/list: yields nothing.
    Scalars (str/int/float/bool/None) at any level: yields `(prefix, value)`.

    The scalar-at-root case yields `("", value)` so the caller can distinguish
    "empty container" (no entries) from "scalar input" (one entry with empty
    prefix).
    """
    if isinstance(value, dict):
        if not value:
            return
        for k, v in value.items():
            sub_prefix = f"{prefix}.{k}" if prefix else str(k)
            yield from _flatten_to_paths(v, sub_prefix)
    elif isinstance(value, list):
        if not value:
            return
        for i, v in enumerate(value):
            sub_prefix = f"{prefix}[{i}]"
            yield from _flatten_to_paths(v, sub_prefix)
    else:
        # Scalar leaf (str, int, float, bool, None, or any non-container type).
        yield (prefix, value)


# ---------------------------------------------------------------------------
# _compute_fingerprint — reuses Phase-4 11-tuple identity rule.
# ---------------------------------------------------------------------------


def _compute_fingerprint(stanza: dict) -> tuple:
    """Return the 11-tuple fingerprint per D-38 / auto_generator.py.

    This is the IDENTITY function for client stanzas: two stanzas with the
    same fingerprint are considered the same "client class" and their
    field-level differences will be surfaced by leaf comparison; two stanzas
    with different fingerprints surface as orphan entries (D-38 / Pitfall 2).
    """
    return tuple(_resolve_fingerprint_key(stanza, k) for k in _FINGERPRINT_KEYS)


# ---------------------------------------------------------------------------
# _render_fingerprint — verbatim-value fingerprint renderer.
# ---------------------------------------------------------------------------


def _render_fingerprint(fp: tuple) -> str:
    """Render a fingerprint tuple as a string with leaf values shown verbatim.

    The naive `repr(fp)` would escape control characters in string leaves
    (notably tabs in multi-value sysctl leaves like `4096\\t87380\\t16777216`
    per D-41), defeating the round-trip-verbatim contract. This helper walks
    the fingerprint structure and emits each leaf via plain `str()` so the
    bytes appear as-is in the rendered output.

    The rendering is purely cosmetic — fingerprints are still keyed and
    sorted by the tuple itself (which is hashable and ordered by Python's
    native tuple comparison after the `key=repr` defense applied by the
    caller).
    """
    def render(v: Any) -> str:
        if isinstance(v, tuple):
            return "(" + ", ".join(render(x) for x in v) + ")"
        return str(v)

    return render(fp)


# ---------------------------------------------------------------------------
# diff_node_dict_lists — public diff function.
# ---------------------------------------------------------------------------


def diff_node_dict_lists(on_disk: list[dict], in_memory: list[dict]) -> DiffResult:
    """Compare two lists of client stanzas and return a structured DiffResult.

    Algorithm:
      1. Index each side by fingerprint (`_compute_fingerprint`).
      2. Sort the union of fingerprints by `repr` (D-22 mixed-type defense).
      3. For each fingerprint:
         - on-disk only → DiffEntry(path=f"clients[fingerprint={fp!r}]",
           old="<present>", new="<absent>")  (D-47)
         - in-memory only → DiffEntry(path=f"clients[fingerprint={fp!r}]",
           old="<absent>", new="<present>")  (D-46)
         - present on both → flatten BOTH sides; for each path in the union:
            - Pitfall 3(a) SER-02: if mem_v == '' and disk_v is present and
              non-empty, skip (submitter-filled value is sacred).
            - else if disk_v != mem_v, emit DiffEntry.
    """
    on_disk_by_fp: dict[tuple, dict] = {_compute_fingerprint(s): s for s in on_disk}
    in_memory_by_fp: dict[tuple, dict] = {_compute_fingerprint(s): s for s in in_memory}

    all_fps = sorted(set(on_disk_by_fp) | set(in_memory_by_fp), key=repr)

    entries: list[DiffEntry] = []

    for fp in all_fps:
        if fp not in in_memory_by_fp:
            # D-47: present only on disk. The fingerprint tuple is rendered
            # via `_render_fingerprint` (not `repr(fp)`) so multi-value sysctl
            # leaves like `4096\t87380\t16777216` survive verbatim in the path
            # string — repr() would escape the literal tabs to `\\t` and trip
            # the D-41 round-trip lock.
            entries.append(DiffEntry(
                path=f"clients[fingerprint={_render_fingerprint(fp)}]",
                old="<present>",
                new="<absent>",
            ))
            continue
        if fp not in on_disk_by_fp:
            # D-46: present only in memory. See D-41 note above.
            entries.append(DiffEntry(
                path=f"clients[fingerprint={_render_fingerprint(fp)}]",
                old="<absent>",
                new="<present>",
            ))
            continue

        # Both sides have this fingerprint — flatten and walk path union.
        disk_paths = dict(_flatten_to_paths(on_disk_by_fp[fp]))
        mem_paths = dict(_flatten_to_paths(in_memory_by_fp[fp]))

        for path in sorted(set(disk_paths) | set(mem_paths), key=repr):
            disk_v = disk_paths.get(path, _SENTINEL_ABSENT)
            mem_v = mem_paths.get(path, _SENTINEL_ABSENT)

            # Pitfall 3(a) SER-02 blank preservation: in-memory empty string +
            # on-disk filled non-empty value means the submitter filled it in
            # by hand; the collector returning blank is NOT drift.
            if (
                mem_v == ""
                and disk_v is not _SENTINEL_ABSENT
                and disk_v != ""
            ):
                continue

            if disk_v != mem_v:
                entries.append(DiffEntry(
                    path=path,
                    old=("<absent>" if disk_v is _SENTINEL_ABSENT else disk_v),
                    new=("<absent>" if mem_v is _SENTINEL_ABSENT else mem_v),
                ))

    return DiffResult(entries=entries)


# ---------------------------------------------------------------------------
# format_unified_diff — human-readable rendering.
# ---------------------------------------------------------------------------


def format_unified_diff(result: DiffResult, on_disk_path: str) -> str:
    """Render a DiffResult as a unified-diff-style string per D-40 / D-41.

    Output shape:
        --- on-disk: <path>
        +++ in-memory: <computed from live MPI fleet>
        @@ <JSONPath_1> @@
        - <old_1>
        + <new_1>
        @@ <JSONPath_2> @@
        - <old_2>
        + <new_2>
        ...
        <blank line>
        Remediation:
          • Rename the existing yaml and re-run with --systemname <new>
            (a fresh systemname.yaml will be generated)
          • Remove <path> and re-run
            (you will lose hand-filled blanks)

    Values are emitted via plain `str()` (NOT `repr()`) so long sysctl tuples
    like `4096\\t87380\\t16777216` round-trip verbatim (D-41 lock).
    """
    lines: list[str] = [
        f"--- on-disk: {on_disk_path}",
        "+++ in-memory: <computed from live MPI fleet>",
    ]

    for entry in result.entries:
        lines.append(f"@@ {entry.path} @@")
        lines.append(f"- {entry.old}")
        lines.append(f"+ {entry.new}")

    lines.append("")
    lines.append("Remediation:")
    lines.append("  • Rename the existing yaml and re-run with --systemname <new>")
    lines.append("    (a fresh systemname.yaml will be generated)")
    lines.append(f"  • Remove {on_disk_path} and re-run")
    lines.append("    (you will lose hand-filled blanks)")

    return "\n".join(lines)
