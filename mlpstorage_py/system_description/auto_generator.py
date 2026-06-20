"""Pure-transformation core for systemname.yaml auto-generation.

This module is the in-memory transformation layer between the existing MPI
cluster collector (which produces `HostInfo` instances) and the eventual
on-disk `systemname.yaml` write step (Plan 02-04). It contains no I/O — every
function here is a pure transformation over Python dicts.

Phase 02 / Plan 02-02 deliverables (Slice 2 of the auto-collector vertical):

- `group_by_fingerprint(items, fingerprint_keys, count_field)` — generic
  quantity-grouping helper per CONTEXT.md D-4. Empty strings participate in
  the fingerprint as-is per D-5 (determinism over flattering output): failed-
  collection hosts group together as their own stanza instead of being hidden.
  Sorting (D-7) is the caller's responsibility — see Plan 02-04.

- `_get_dotted(d, dotted_key)` — internal dotted-key resolver. Missing keys
  return the empty string per D-5 so fingerprint determinism holds even on
  hosts where the collector returned partial data.

- `_FINGERPRINT_KEYS` — the six-key tuple per D-4 that defines what makes two
  hosts "the same" for quantity-grouping purposes.

Phase 02 / Plan 02-03 deliverables (Slice 3 — schema-aware blanks scaffolding):

- `_NETWORKING_STUB`, `_DRIVE_STUB` — `Final[dict]` stub literals carrying the
  empty-string / empty-list values the YAML emit needs for the `networking[]`
  and `drives[]` sublists. D-3 (CONTEXT.md): these intentionally bypass the
  StrictModel Pydantic classes since empty strings fail enum / `min_length=1`
  validation but ARE the desired emit shape (SER-02 visible to-do reminders).
  Field-name parity with `NetworkPort.model_fields` /
  `DriveInstance.model_fields` is asserted at TEST time
  (`test_stub_keys_match_pydantic_fields`) so any schema change forces the
  stubs to be updated in lockstep.

- `_splice_stub_lists(dump)` — post-dump mutator that injects exactly one
  stub `networking` entry and one stub `drives` entry into every client.
  Idempotent (re-runs REPLACE, do not append). Defensive on shape: missing
  `system_under_test` or `clients` returns the dump unchanged.

- `_build_outer_dict(stanzas)` — `{system_under_test: {clients: [...]}}`
  scaffolding. Per D-14, the top-level `solution`, `deployment`,
  `product_nodes`, `product_switches`, `total_rack_units`, and
  `rack_power_supplies` blocks are OMITTED — the auto-collector cannot supply
  legal enum values for `solution.architecture.storage_location` (Pitfall 1),
  so `schema_validator.validate_file()` will fail on the missing required
  fields, which IS the intended "submitter has work to do" UX (SER-02).

Symbols arriving in later slices of Phase 02 (NOT in this module yet):

- `write_systemname_yaml`, `_resolve_host_info_list`, atomic write,
  FileExistsError no-op → Plan 02-04.

Pitfall 2 lock: this module does NOT construct any leaf Pydantic instance
(Chassis / OperatingSystem / NodeDescription). Those models enforce
`min_length=1` on the very fields the universal collection-failure rule
demands we emit as empty strings; constructing them here would crash on any
partial-collection host. The Pydantic models live in `schema_validator.py`
and are exercised only at validation time and at test time (via
`.model_fields.keys()` reflection for schema-drift detection).
"""

import copy
from typing import Any, Final

from mlpstorage_py.rules.models import HostInfo


# ---------------------------------------------------------------------------
# D-4: locked fingerprint key set for quantity-grouping homogeneous client
# stanzas. Order matters only for human readability of the resulting tuples;
# group_by_fingerprint hashes them as a tuple so any consistent order works.
# ---------------------------------------------------------------------------
_FINGERPRINT_KEYS: tuple[str, ...] = (
    "chassis.cpu_model",
    "chassis.cpu_qty",
    "chassis.cpu_cores",
    "chassis.memory_capacity",
    "operating_system.name",
    "operating_system.version",
)


def _get_dotted(d: dict, dotted_key: str) -> Any:
    """Walk a dotted path through nested dicts.

    Examples:
        _get_dotted({"a": {"b": {"c": 42}}}, "a.b.c") == 42
        _get_dotted({"a": {}}, "a.b.c") == ""           # missing nested key
        _get_dotted({}, "x") == ""                       # missing top-level
        _get_dotted({"a": "leaf"}, "a") == "leaf"        # single segment
        _get_dotted({"a": "leaf"}, "a.b") == ""          # intermediate not a dict

    A miss at any depth returns the empty string. Per D-5, this is intentional:
    empty-string-on-miss makes the fingerprint deterministic even on hosts
    where the collector returned partial data — failed-collection hosts group
    together as their own stanza rather than crashing the grouping pass.
    """
    cur: Any = d
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return ""
    return cur


def group_by_fingerprint(
    items: list[dict],
    fingerprint_keys: tuple[str, ...],
    count_field: str,
) -> list[dict]:
    """Collapse items sharing all fingerprint_keys into one entry annotated
    with count_field: N.

    Properties:

    - Order: first-occurrence (deterministic on the input order). Sorting is
      D-7 territory and lives in `write_systemname_yaml` (Plan 02-04), not
      here — keeping this helper concerns-separated.
    - Empty strings participate in the fingerprint as-is per D-5. A host whose
      cpu_model collection failed (`""`) groups with other `""` hosts; it does
      NOT mysteriously absorb into a real-CPU stanza.
    - The input list and its dicts are NOT mutated: each accepted item is
      deep-copied before being annotated with the count field. Callers can
      safely pass the same list to repeated calls.
    - Dotted keys (`chassis.cpu_model`) are resolved by `_get_dotted`; missing
      keys → `""`, preserving determinism for partial collections.

    Args:
        items: list of dicts (e.g. node_description-shaped dicts from
            `node_dict_from_host`).
        fingerprint_keys: tuple of dotted keys defining the equivalence class.
            For Phase 2 host grouping, pass `_FINGERPRINT_KEYS`.
        count_field: name of the integer count field to inject on each
            returned stanza (e.g. `"quantity"` matching the schema).

    Returns:
        A new list of dicts, one per fingerprint equivalence class, each
        carrying `count_field=N`. Empty input → empty output.
    """
    groups: dict[tuple, dict] = {}
    for item in items:
        fp = tuple(_get_dotted(item, k) for k in fingerprint_keys)
        if fp not in groups:
            # Deep copy preserves the no-mutation invariant: callers' items
            # stay clean and re-grouping (e.g. in tests) is idempotent.
            groups[fp] = {**copy.deepcopy(item), count_field: 1}
        else:
            groups[fp][count_field] += 1
    return list(groups.values())


def node_dict_from_host(host: HostInfo) -> dict:
    """Map a `HostInfo` into a `NodeDescription`-shaped dict.

    Output shape (Plan 02-02 deliverable — `networking` and `drives` are
    spliced by Plan 02-03; `quantity` is injected by `group_by_fingerprint`):

        {
          "friendly_description": "",
          "chassis": {
            "model_name": "",          # SER-02 blank; Phase 3 fills via lshw
            "cpu_model": <str | "">,    # COLL-01
            "cpu_qty": <int | "">,      # COLL-01 via host.cpu.num_sockets (D-16)
            "cpu_cores": <int | "">,    # COLL-01 via host.cpu.num_cores
            "memory_capacity": <int | "">,  # COLL-01, GiB (D-6)
          },
          "operating_system": {
            "name": <str | "">,          # COLL-02 via os_release NAME
            "version": <str | "">,       # COLL-02 via os_release VERSION_ID
          },
        }

    Defensive on every field per the universal collection-failure rule
    (CONTEXT.md D-2 / Pitfall 9): if any source is missing, blank or zero,
    that single field becomes `""` — the function never raises.

    Memory rounding (D-6): `host.memory.total` is bytes (see
    `HostMemoryInfo.from_proc_meminfo_dict` which converts kB → bytes at
    `rules/models.py:117`). Dividing by `1024**3` yields binary GiB; Python's
    default round-half-to-even is acceptable per D-6 since real RAM sizes
    don't typically land on a half-GiB boundary.

    OS field mapping (COLL-02 / Pitfall 4): we select **only** the `NAME` and
    `VERSION_ID` keys from `/etc/os-release` — never `PRETTY_NAME`, `ID`,
    `VERSION`, or `VERSION_CODENAME`. The `.get(k, "") or ""` idiom collapses
    both missing keys and explicit `None` values to the empty string.

    Pitfall 2: this function deliberately does NOT construct any leaf Pydantic
    instance. The `Chassis` and `OperatingSystem` models enforce
    `min_length=1` on the very fields the universal collection-failure rule
    demands we emit as empty strings — constructing them here would crash on
    any partial-collection host. The dict shape is verified against the
    Pydantic schemas at TEST time via `.model_fields.keys()` reflection
    (`test_node_dict_field_names_match_pydantic_reflection`).
    """
    # COLL-01: CPU fields. Truthy guards preserve the 0 → "" mapping for the
    # `num_sockets == 0` case where summarize_cpuinfo couldn't determine the
    # socket count and emitted the dataclass default rather than a real value.
    cpu_model = host.cpu.model if (host.cpu and host.cpu.model) else ""
    cpu_qty = host.cpu.num_sockets if (host.cpu and host.cpu.num_sockets) else ""
    cpu_cores = host.cpu.num_cores if (host.cpu and host.cpu.num_cores) else ""

    # D-6: memory_capacity in binary GiB. host.memory.total is bytes; dividing
    # by 1024**3 yields GiB; round() is round-half-to-even (Python default).
    if host.memory and host.memory.total:
        memory_capacity: Any = round(host.memory.total / (1024 ** 3))
    else:
        memory_capacity = ""

    # COLL-02 / Pitfall 4: NAME → name, VERSION_ID → version, only.
    # Pitfall 9: `.get(k, "") or ""` collapses missing-key and explicit-None.
    os_name = ""
    os_version = ""
    if host.system and host.system.os_release:
        os_name = host.system.os_release.get("NAME", "") or ""
        os_version = host.system.os_release.get("VERSION_ID", "") or ""

    return {
        "friendly_description": "",  # SER-02 blank — human declaration
        "chassis": {
            "model_name": "",        # SER-02 blank — Phase 3 fills via lshw
            # rack_units OMITTED per D-2 row 4 (optional + non-derivable)
            "cpu_model": cpu_model,
            "cpu_qty": cpu_qty,
            "cpu_cores": cpu_cores,
            "memory_capacity": memory_capacity,
            # power OMITTED per D-2 row 4 (optional + non-derivable)
        },
        # networking / drives: spliced by Plan 02-03's _splice_stub_lists
        "operating_system": {
            "name": os_name,
            "version": os_version,
        },
        # environment / sysctl: Phase 4 territory — not emitted here
        # quantity: injected by group_by_fingerprint downstream
    }


# ---------------------------------------------------------------------------
# Plan 02-03 — D-3 stub literals for the networking[] and drives[] sublists.
#
# These intentionally bypass the NetworkPort / DriveInstance StrictModel
# constructors: those models enforce enum membership and `min_length=1`
# constraints which fail on empty strings, but empty-string stubs ARE the
# desired emit shape — the schema_validator failures at submission time are
# the SER-02 "submitter has work to do" UX, not a bug.
#
# Field-name parity with NetworkPort.model_fields / DriveInstance.model_fields
# is asserted at TEST time by
# tests/unit/test_auto_generator.py::test_stub_keys_match_pydantic_fields.
# Any future schema change failing that test forces the stub to be updated.
# ---------------------------------------------------------------------------
_NETWORKING_STUB: Final[dict] = {
    "unit_count": "",   # NetworkPort.unit_count: int(ge=1) — '' fails validation
    "type":       "",   # NetworkPort.type: NetworkType enum — '' fails enum check
    "speed":      "",   # NetworkPort.speed: int(ge=1)
    "traffic":    [],   # NetworkPort.traffic: List[TrafficType], min_length=1
}

_DRIVE_STUB: Final[dict] = {
    "unit_count":     "",   # DriveInstance.unit_count: int(ge=1)
    "vendor_name":    "",
    "model_name":     "",
    "interface":      "",   # DriveInstance.interface: DriveInterface enum
    "media_type":     "",   # DriveInstance.media_type: DriveMediaType enum
    "capacity_in_GB": "",   # DriveInstance.capacity_in_GB: int(ge=1)
    # performance: OMITTED per D-2 row 4 — optional + non-derivable spec-sheet fact
}


def _splice_stub_lists(dump: dict) -> dict:
    """Splice stub networking[] and drives[] entries into every client.

    D-3 (CONTEXT.md): stub entries intentionally bypass the StrictModel
    Pydantic classes — empty-string fields fail enum / min=1 / list-non-empty
    checks at construction time but ARE the desired emit shape (visible
    to-do reminders for the submitter; SER-02). Each stub is a fresh dict
    (`dict(_NETWORKING_STUB)`) so callers can safely mutate without
    aliasing the module-level constant.

    Idempotent: re-running on the same dict REPLACES the spliced entries,
    not appends. Plan 02-04 relies on this so callers can chain
    `_splice_stub_lists(_build_outer_dict(stanzas))` even after re-grouping.

    Defensive on shape: if `dump` has no `system_under_test` or no
    `clients`, the function returns the dump unchanged (no `KeyError`).

    Returns the input dict (mutated in place). Callers can write
    `dump = _splice_stub_lists(dump)` for clarity even though the return
    value is identity.
    """
    clients = dump.get("system_under_test", {}).get("clients", [])
    for client in clients:
        client["networking"] = [dict(_NETWORKING_STUB)]
        client["drives"]     = [dict(_DRIVE_STUB)]
    return dump


def _build_outer_dict(stanzas: list[dict]) -> dict:
    """Construct {system_under_test: {clients: [...]}} per D-14.

    Top-level optional and required blocks are OMITTED:
      - solution         (required at schema level; submitter fills via D-14)
      - deployment       (required at schema level; submitter fills via D-14)
      - product_nodes    (optional)
      - product_switches (optional)
      - total_rack_units (optional)
      - rack_power_supplies (optional)

    Rationale (Pitfall 1 / D-14): `Solution.architecture.storage_location` is
    an enum and `""` is not a legal enum value. `Architecture.check_na_pairing`
    and `Capabilities.check_remap_time` are `model_validator`s that fire at
    construction time. Whole-file `schema_validator.validate_file()` will fail
    on the missing required fields — that IS the intended "submitter has work
    to do" UX (SER-02). Stubbing these blocks would just shift the failure
    from "missing block" to "invalid enum value", which is a worse signal.
    """
    return {
        "system_under_test": {
            # solution: OMITTED per D-14 (submitter fills)
            # deployment: OMITTED per D-14 (submitter fills)
            # product_nodes: OMITTED (optional)
            # product_switches: OMITTED (optional)
            # total_rack_units: OMITTED (optional)
            # rack_power_supplies: OMITTED (optional)
            "clients": stanzas,
        }
    }
