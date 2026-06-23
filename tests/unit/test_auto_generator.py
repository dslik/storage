"""Unit tests for mlpstorage_py.system_description.auto_generator.

Phase 02 / Plan 02-02 — RED→GREEN coverage for the pure transformation core
of the systemname.yaml auto-generator. Tests exercise:

- D-4 : group_by_fingerprint signature (items, fingerprint_keys, count_field)
- D-5 : empty strings participate in the fingerprint as-is (determinism on
        failed-collection hosts)
- D-6 : memory_capacity = round(bytes / (1024**3)) — binary GiB
- D-16: chassis.cpu_qty sourced from host.cpu.num_sockets (landed in 02-01)
- COLL-01: cpu_model / cpu_qty / cpu_cores / memory_capacity extraction
- COLL-02: operating_system.name / version extraction (NAME → name,
           VERSION_ID → version per Pitfall 4)
- SER-01: quantity-grouping for homogeneous and heterogeneous fleets

Pitfall 2 lock: this module exercises the adapter and grouping helpers as
PURE dict transformations. The leaf Pydantic models (Chassis,
OperatingSystem) are imported only for `.model_fields.keys()` reflection in
test_node_dict_field_names_match_pydantic_reflection — never constructed.
"""

import copy

import pytest

from mlpstorage_py.rules.models import (
    HostCPUInfo,
    HostInfo,
    HostMemoryInfo,
)
from mlpstorage_py.cluster_collector import HostSystemInfo
from mlpstorage_py.system_description.auto_generator import (
    _DRIVE_STUB,
    _FINGERPRINT_KEYS,
    _NETWORKING_STUB,
    _build_outer_dict,
    _get_dotted,
    _splice_stub_lists,
    group_by_fingerprint,
    node_dict_from_host,
)


# ---------------------------------------------------------------------------
# Task 1 — _get_dotted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, dotted_key, expected",
    [
        ({"a": {"b": {"c": 42}}}, "a.b.c", 42),
        ({"a": {}}, "a.b.c", ""),
        ({}, "x", ""),
        ({"a": "leaf"}, "a", "leaf"),
    ],
)
def test_get_dotted_walks_nested_dicts(d, dotted_key, expected):
    """Dotted-key walker handles nested dicts, missing keys, and single segments."""
    assert _get_dotted(d, dotted_key) == expected


def test_get_dotted_missing_key_returns_empty_string():
    """Missing top-level key returns empty string (D-5 determinism)."""
    assert _get_dotted({}, "missing") == ""


def test_get_dotted_handles_non_dict_intermediate():
    """If an intermediate value is not a dict, return empty string (not crash)."""
    assert _get_dotted({"a": "leaf"}, "a.b.c") == ""


# ---------------------------------------------------------------------------
# Task 1 — group_by_fingerprint
# ---------------------------------------------------------------------------


def _homogeneous_host_dict(cpu_model: str = "X") -> dict:
    """Helper: build a NodeDescription-shaped dict for grouping tests."""
    return {
        "friendly_description": "",
        "chassis": {
            "model_name": "",
            "cpu_model": cpu_model,
            "cpu_qty": 2,
            "cpu_cores": 64,
            "memory_capacity": 256,
        },
        "operating_system": {"name": "Rocky", "version": "9.5"},
    }


def test_group_by_fingerprint_homogeneous():
    """Three identical dicts collapse to one stanza with quantity=3 (SER-01)."""
    items = [_homogeneous_host_dict() for _ in range(3)]
    result = group_by_fingerprint(items, _FINGERPRINT_KEYS, "quantity")
    assert len(result) == 1
    assert result[0]["quantity"] == 3


def test_group_by_fingerprint_heterogeneous():
    """Differing cpu_model produces multiple stanzas summing to N (SER-01)."""
    items = [
        _homogeneous_host_dict("X"),
        _homogeneous_host_dict("X"),
        _homogeneous_host_dict("Y"),
    ]
    result = group_by_fingerprint(items, _FINGERPRINT_KEYS, "quantity")
    assert len(result) == 2
    # First-occurrence order: X stanza first with qty=2, then Y with qty=1.
    assert result[0]["chassis"]["cpu_model"] == "X"
    assert result[0]["quantity"] == 2
    assert result[1]["chassis"]["cpu_model"] == "Y"
    assert result[1]["quantity"] == 1


def test_empty_strings_participate_in_fingerprint():
    """Empty-string fields group together (D-5: determinism over flattering output)."""
    items = [
        _homogeneous_host_dict(""),
        _homogeneous_host_dict(""),
        _homogeneous_host_dict("X"),
    ]
    result = group_by_fingerprint(items, _FINGERPRINT_KEYS, "quantity")
    assert len(result) == 2
    # Empty-string hosts arrive first in input order, so they group first.
    assert result[0]["chassis"]["cpu_model"] == ""
    assert result[0]["quantity"] == 2
    assert result[1]["chassis"]["cpu_model"] == "X"
    assert result[1]["quantity"] == 1


def test_group_by_fingerprint_preserves_first_occurrence_order():
    """Input order [X, Y, X, Y] → output order [X(qty=2), Y(qty=2)]."""
    items = [
        _homogeneous_host_dict("X"),
        _homogeneous_host_dict("Y"),
        _homogeneous_host_dict("X"),
        _homogeneous_host_dict("Y"),
    ]
    result = group_by_fingerprint(items, _FINGERPRINT_KEYS, "quantity")
    assert [s["chassis"]["cpu_model"] for s in result] == ["X", "Y"]
    assert [s["quantity"] for s in result] == [2, 2]


def test_group_by_fingerprint_does_not_mutate_input():
    """The helper must deepcopy items so callers' inputs are untouched."""
    items = [_homogeneous_host_dict("X") for _ in range(3)]
    snapshot = copy.deepcopy(items)
    group_by_fingerprint(items, _FINGERPRINT_KEYS, "quantity")
    assert items == snapshot
    # No 'quantity' key sneaked into the originals.
    for original in items:
        assert "quantity" not in original


def test_group_by_fingerprint_empty_input():
    """Empty input → empty output, no crash."""
    assert group_by_fingerprint([], _FINGERPRINT_KEYS, "quantity") == []


# ---------------------------------------------------------------------------
# Task 2 — node_dict_from_host adapter
# ---------------------------------------------------------------------------


def _make_host(
    *,
    cpu=None,
    memory=None,
    os_release=None,
    system_present=True,
):
    """Build a HostInfo with sensible Phase 2 defaults for testing the adapter."""
    if memory is None:
        memory = HostMemoryInfo(total=274_877_906_944)  # 256 GiB exact
    system = None
    if system_present:
        system = HostSystemInfo(
            hostname="h",
            os_release=os_release if os_release is not None else {
                "NAME": "Rocky Linux",
                "VERSION_ID": "9.5",
            },
        )
    return HostInfo(hostname="h", cpu=cpu, memory=memory, system=system)


def test_node_dict_cpu_fields():
    """COLL-01 happy path: cpu_model/qty/cores/memory_capacity + OS fields."""
    host = _make_host(
        cpu=HostCPUInfo(
            model="Intel(R) Xeon Platinum 8480+",
            num_cores=56,
            num_logical_cores=112,
            num_sockets=2,
            architecture="x86_64",
        ),
    )
    result = node_dict_from_host(host)

    assert result["chassis"]["cpu_model"] == "Intel(R) Xeon Platinum 8480+"
    assert result["chassis"]["cpu_qty"] == 2  # from num_sockets (D-16)
    assert result["chassis"]["cpu_cores"] == 56
    assert result["chassis"]["memory_capacity"] == 256
    assert result["operating_system"]["name"] == "Rocky Linux"
    assert result["operating_system"]["version"] == "9.5"

    # SER-02 blanks (Phase 3 fills model_name; friendly_description is human-only).
    assert result["chassis"]["model_name"] == ""
    assert result["friendly_description"] == ""

    # D-2 row 4: optional fields are OMITTED, not blanked.
    assert "rack_units" not in result["chassis"]
    assert "power" not in result["chassis"]

    # Stub-splice / Phase 4 / quantity-injection territory — not here.
    assert "networking" not in result
    assert "drives" not in result
    assert "environment" not in result
    assert "sysctl" not in result
    assert "quantity" not in result


@pytest.mark.parametrize(
    "total_bytes, expected_capacity",
    [
        (274_877_906_944, 256),  # 256 GiB exact
        (270_582_939_648, 252),  # ~252.0 GiB after kernel reservation
        (271_652_882_432, 253),  # ~252.9965 GiB → rounds to 253
        (1_073_741_824, 1),      # 1 GiB exact
        (0, ""),                 # universal collection-failure rule
    ],
)
def test_memory_capacity_rounding(total_bytes, expected_capacity):
    """D-6: round(bytes / (1024**3)); zero bytes maps to empty string."""
    host = _make_host(
        cpu=HostCPUInfo(model="X", num_cores=4, num_sockets=1),
        memory=HostMemoryInfo(total=total_bytes),
    )
    result = node_dict_from_host(host)
    assert result["chassis"]["memory_capacity"] == expected_capacity


def test_node_dict_empty_cpuinfo():
    """Universal collection-failure rule: cpu=None → blank cpu fields, no raise."""
    host = _make_host(cpu=None)
    result = node_dict_from_host(host)
    assert result["chassis"]["cpu_model"] == ""
    assert result["chassis"]["cpu_qty"] == ""
    assert result["chassis"]["cpu_cores"] == ""
    # OS and memory still real because only cpu was missing.
    assert result["chassis"]["memory_capacity"] == 256
    assert result["operating_system"]["name"] == "Rocky Linux"


def test_node_dict_empty_os_release():
    """Pitfall 9: empty os_release dict → blank name/version, not None."""
    host = _make_host(
        cpu=HostCPUInfo(model="X", num_cores=4, num_sockets=1),
        os_release={},
    )
    result = node_dict_from_host(host)
    assert result["operating_system"]["name"] == ""
    assert result["operating_system"]["version"] == ""
    # Never None, never null — strict empty string per the universal rule.
    assert result["operating_system"]["name"] is not None
    assert result["operating_system"]["version"] is not None


def test_node_dict_missing_system():
    """HostInfo.system=None must not raise; OS fields blank."""
    host = _make_host(
        cpu=HostCPUInfo(model="X", num_cores=4, num_sockets=1),
        system_present=False,
    )
    result = node_dict_from_host(host)
    assert result["operating_system"]["name"] == ""
    assert result["operating_system"]["version"] == ""


def test_node_dict_empty_memory():
    """HostMemoryInfo() default total=0 → memory_capacity blank."""
    host = _make_host(
        cpu=HostCPUInfo(model="X", num_cores=4, num_sockets=1),
        memory=HostMemoryInfo(),
    )
    result = node_dict_from_host(host)
    assert result["chassis"]["memory_capacity"] == ""


def test_os_field_mapping():
    """COLL-02 / Pitfall 4: only NAME → name and VERSION_ID → version.

    Even when the os_release dict carries PRETTY_NAME / ID / VERSION /
    VERSION_CODENAME, the adapter must select NAME and VERSION_ID exclusively.
    """
    host = _make_host(
        cpu=HostCPUInfo(model="X", num_cores=4, num_sockets=1),
        os_release={
            "NAME": "Rocky Linux",
            "PRETTY_NAME": "Rocky Linux 9.5 (Blue Onyx)",
            "ID": "rocky",
            "VERSION": "9.5 (Blue Onyx)",
            "VERSION_ID": "9.5",
            "VERSION_CODENAME": "blue onyx",
        },
    )
    result = node_dict_from_host(host)
    assert result["operating_system"]["name"] == "Rocky Linux"
    assert result["operating_system"]["version"] == "9.5"


def test_node_dict_no_extra_keys():
    """Schema discipline: top-level keys are exactly the three Phase 2 emits."""
    host = _make_host(cpu=HostCPUInfo(model="X", num_cores=4, num_sockets=1))
    result = node_dict_from_host(host)

    assert set(result.keys()) == {"friendly_description", "chassis", "operating_system"}
    assert set(result["chassis"].keys()) == {
        "model_name",
        "cpu_model",
        "cpu_qty",
        "cpu_cores",
        "memory_capacity",
    }


def test_node_dict_field_names_match_pydantic_reflection():
    """D-1: leaf-level field names must match the Pydantic StrictModel schemas.

    chassis: issubset (rack_units/power intentionally omitted per D-2).
    operating_system: ==     (both name and version are emitted).
    """
    from mlpstorage_py.system_description.schema_validator import (
        Chassis,
        OperatingSystem,
    )

    host = _make_host(cpu=HostCPUInfo(model="X", num_cores=4, num_sockets=1))
    result = node_dict_from_host(host)

    chassis_keys = set(result["chassis"].keys())
    chassis_schema_keys = set(Chassis.model_fields.keys())
    assert chassis_keys.issubset(chassis_schema_keys), (
        f"chassis dict has fields Chassis Pydantic does not: "
        f"{chassis_keys - chassis_schema_keys}"
    )

    os_keys = set(result["operating_system"].keys())
    os_schema_keys = set(OperatingSystem.model_fields.keys())
    assert os_keys == os_schema_keys, (
        f"operating_system dict drift from OperatingSystem: "
        f"missing={os_schema_keys - os_keys}, extra={os_keys - os_schema_keys}"
    )


# ---------------------------------------------------------------------------
# Plan 02-03 / Task 1 — _NETWORKING_STUB, _DRIVE_STUB module constants
# ---------------------------------------------------------------------------


def test_networking_stub_shape():
    """_NETWORKING_STUB has the five NetworkPort field names with empty-string /
    empty-list values per RESEARCH.md Pattern 3 (D-3 seam) plus the Phase 3
    `state` key added in lockstep with D-20.

    `traffic` is a List[TrafficType] with min_length=1 in NetworkPort
    (pre-Phase-3) and Optional[List[TrafficType]] (post-Phase-3); the stub
    uses `[]` either way because Pydantic rejects `[]` at validation time
    (intended SER-02 to-do UX).

    `state` is `""` (D-3 option (a) per RESEARCH § "_NETWORKING_STUB Redesign
    Tradeoff Analysis"): the empty string means "collector has no information
    at all" and is distinct from a real `"down"` (which would be a positive
    statement about NIC state). Pydantic rejects `""` against
    `Literal["up","down"]`, which is the SER-02 signal that the collector
    failed and the submitter must fill the NIC entry by hand.
    """
    assert _NETWORKING_STUB == {
        "unit_count": "",
        "type": "",
        "state": "",
        "speed": "",
        "traffic": [],
    }
    # traffic specifically must be the empty LIST, not empty string.
    assert _NETWORKING_STUB["traffic"] == []
    assert isinstance(_NETWORKING_STUB["traffic"], list)
    # state is the empty STRING (Pydantic-bypass per D-3 option (a)), NOT "down".
    assert _NETWORKING_STUB["state"] == ""
    assert isinstance(_NETWORKING_STUB["state"], str)


def test_drive_stub_shape():
    """_DRIVE_STUB has the six expected DriveInstance field names with
    empty-string values. `performance` is deliberately OMITTED per D-2 row 4
    (optional + non-derivable spec-sheet fact).
    """
    assert _DRIVE_STUB == {
        "unit_count": "",
        "vendor_name": "",
        "model_name": "",
        "interface": "",
        "media_type": "",
        "capacity_in_GB": "",
    }
    # D-2 row 4: performance is optional and non-derivable. Not stubbed.
    assert "performance" not in _DRIVE_STUB


def test_stub_keys_match_pydantic_fields():
    """D-3 single-source-of-truth: stub literals must match the Pydantic
    StrictModel field names exactly (minus the optional `performance` field
    deliberately omitted from _DRIVE_STUB per D-2 row 4).

    This is the load-bearing schema-drift discipline test. If NetworkPort or
    DriveInstance gains a field in a later phase, this test fires — forcing
    the stub literal to be updated in lockstep (D-1).
    """
    from mlpstorage_py.system_description.schema_validator import (
        DriveInstance,
        NetworkPort,
    )

    assert set(_NETWORKING_STUB.keys()) == set(NetworkPort.model_fields.keys()), (
        f"_NETWORKING_STUB drift from NetworkPort: "
        f"missing={set(NetworkPort.model_fields.keys()) - set(_NETWORKING_STUB.keys())}, "
        f"extra={set(_NETWORKING_STUB.keys()) - set(NetworkPort.model_fields.keys())}"
    )

    # `performance` is optional on DriveInstance and intentionally omitted per D-2 row 4.
    drive_pydantic_keys = set(DriveInstance.model_fields.keys()) - {"performance"}
    assert set(_DRIVE_STUB.keys()) == drive_pydantic_keys, (
        f"_DRIVE_STUB drift from DriveInstance (minus optional `performance`): "
        f"missing={drive_pydantic_keys - set(_DRIVE_STUB.keys())}, "
        f"extra={set(_DRIVE_STUB.keys()) - drive_pydantic_keys}"
    )


def test_stub_constants_are_module_level_not_mutated_by_callers():
    """`dict(_NETWORKING_STUB)` yields a fresh dict so the splice helper can
    hand out copies without aliasing the module-level constant. Mutating a
    copy must not affect the original.
    """
    original_net = copy.deepcopy(_NETWORKING_STUB)
    original_drive = copy.deepcopy(_DRIVE_STUB)

    # Simulate a downstream caller mutating a returned stub.
    mutated_net = dict(_NETWORKING_STUB)
    mutated_net["unit_count"] = 16
    mutated_drive = dict(_DRIVE_STUB)
    mutated_drive["vendor_name"] = "Acme"

    # Module-level constants remain pristine.
    assert _NETWORKING_STUB == original_net
    assert _DRIVE_STUB == original_drive


# ---------------------------------------------------------------------------
# Plan 02-03 / Task 2 — _splice_stub_lists
# ---------------------------------------------------------------------------


def _bare_client_dump(num_clients: int = 1) -> dict:
    """Build a dump shaped like _build_outer_dict's output but without
    networking/drives — the shape 02-02's node_dict_from_host produces before
    Plan 02-03's splice step.
    """
    clients = [
        {
            "friendly_description": "",
            "chassis": {
                "model_name": "",
                "cpu_model": f"CPU-{i}",
                "cpu_qty": 2,
                "cpu_cores": 64,
                "memory_capacity": 256,
            },
            "operating_system": {"name": "Rocky", "version": "9.5"},
        }
        for i in range(num_clients)
    ]
    return {"system_under_test": {"clients": clients}}


def test_splice_stub_lists_adds_to_every_client():
    """Single-client dump: networking and drives lists each get one stub entry."""
    dump = _bare_client_dump(num_clients=1)
    result = _splice_stub_lists(dump)

    client = result["system_under_test"]["clients"][0]
    assert client["networking"] == [_NETWORKING_STUB]
    assert client["drives"] == [_DRIVE_STUB]

    # The spliced entries are FRESH dicts (not aliases of the module constants).
    # Mutating one must not mutate the other.
    assert client["networking"][0] is not _NETWORKING_STUB
    assert client["drives"][0] is not _DRIVE_STUB


def test_splice_stub_lists_multiple_clients():
    """All clients in a multi-client dump receive their own stub lists."""
    dump = _bare_client_dump(num_clients=3)
    result = _splice_stub_lists(dump)

    for client in result["system_under_test"]["clients"]:
        assert client["networking"] == [_NETWORKING_STUB]
        assert client["drives"] == [_DRIVE_STUB]


def test_splice_stub_lists_idempotent():
    """Calling twice REPLACES the spliced entries — does not append.

    Idempotence is a contractual property: callers (Plan 02-04) may chain
    `_splice_stub_lists(_build_outer_dict(stanzas))` and trust that re-running
    on the same dict yields the same final state.
    """
    dump = _bare_client_dump(num_clients=1)
    _splice_stub_lists(dump)
    _splice_stub_lists(dump)

    client = dump["system_under_test"]["clients"][0]
    assert len(client["networking"]) == 1
    assert len(client["drives"]) == 1
    assert client["networking"] == [_NETWORKING_STUB]
    assert client["drives"] == [_DRIVE_STUB]


def test_splice_stub_lists_empty_clients():
    """Empty clients list → no crash, dump returned unchanged."""
    dump = {"system_under_test": {"clients": []}}
    result = _splice_stub_lists(dump)
    assert result == {"system_under_test": {"clients": []}}


def test_splice_stub_lists_missing_system_under_test():
    """Defensive `dict.get` chain: missing system_under_test → no KeyError."""
    dump: dict = {}
    result = _splice_stub_lists(dump)
    # The contract is "no crash"; the exact return shape is whatever the
    # defensive chain produces (here, the unchanged input).
    assert result == {}


def test_splice_stub_lists_returns_same_object():
    """The helper mutates in place and returns the input dict — 02-04 callers
    rely on this so they can chain `dump = _splice_stub_lists(_build_outer_dict(...))`.
    """
    dump = _bare_client_dump(num_clients=1)
    assert _splice_stub_lists(dump) is dump


# ---------------------------------------------------------------------------
# Plan 02-03 / Task 2 — _build_outer_dict
# ---------------------------------------------------------------------------


def test_build_outer_dict_shape():
    """D-14: outer dict has exactly `system_under_test` at top level, and
    `system_under_test` contains exactly `clients` — nothing else.
    """
    stanzas = [
        {
            "friendly_description": "",
            "chassis": {"cpu_model": "X"},
            "operating_system": {"name": "Rocky", "version": "9.5"},
            "quantity": 4,
        }
    ]
    result = _build_outer_dict(stanzas)

    assert set(result.keys()) == {"system_under_test"}
    sut = result["system_under_test"]
    assert set(sut.keys()) == {"clients"}
    assert sut["clients"] == stanzas


def test_build_outer_dict_omits_solution_deployment():
    """D-14 explicit lock: solution, deployment, product_nodes,
    product_switches, total_rack_units, rack_power_supplies are ALL absent.

    Per Pitfall 1, those top-level blocks would require enum values and
    model-validator-satisfying inputs which the auto-collector cannot supply.
    Omitting them lets schema_validator.validate_file() surface "submitter
    has work to do" as the intended UX (SER-02).
    """
    result = _build_outer_dict([])
    sut = result["system_under_test"]
    for forbidden in (
        "solution",
        "deployment",
        "product_nodes",
        "product_switches",
        "total_rack_units",
        "rack_power_supplies",
    ):
        assert forbidden not in sut, f"D-14 violation: {forbidden} present in outer dict"


def test_build_outer_dict_empty_stanzas():
    """Empty stanzas list → empty clients list under system_under_test."""
    result = _build_outer_dict([])
    assert result == {"system_under_test": {"clients": []}}


def test_outer_dict_with_spliced_stubs_yaml_roundtrip():
    """End-to-end sanity: build outer dict + splice stubs + YAML safe_dump +
    safe_load reproduces the structure intact.

    This is a sanity check that the combined output is a valid YAML data
    structure. Full formatting tests (block style, default_style, etc.) live
    in Plan 02-04 — only `yaml.safe_*` is used here (T-2-04 mitigation).
    """
    import yaml

    stanzas = [
        {
            "friendly_description": "",
            "chassis": {
                "model_name": "",
                "cpu_model": "Intel(R) Xeon Platinum 8480+",
                "cpu_qty": 2,
                "cpu_cores": 56,
                "memory_capacity": 256,
            },
            "operating_system": {"name": "Rocky Linux", "version": "9.5"},
            "quantity": 4,
        }
    ]
    dump = _build_outer_dict(stanzas)
    dump = _splice_stub_lists(dump)

    yaml_text = yaml.safe_dump(dump, default_flow_style=False, sort_keys=False)
    reloaded = yaml.safe_load(yaml_text)

    client = reloaded["system_under_test"]["clients"][0]
    assert client["chassis"]["cpu_qty"] == 2
    assert client["operating_system"]["name"] == "Rocky Linux"
    assert client["networking"] == [
        # D-20 (Phase 3): _NETWORKING_STUB now carries state="" per D-3 option (a)
        {"unit_count": "", "type": "", "state": "", "speed": "", "traffic": []}
    ]
    assert client["drives"] == [
        {
            "unit_count": "",
            "vendor_name": "",
            "model_name": "",
            "interface": "",
            "media_type": "",
            "capacity_in_GB": "",
        }
    ]
    # D-14: top-level forbidden blocks remain absent after YAML roundtrip.
    assert "solution" not in reloaded["system_under_test"]
    assert "deployment" not in reloaded["system_under_test"]
