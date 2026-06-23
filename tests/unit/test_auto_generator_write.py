"""Unit tests for write_systemname_yaml — Phase 02 / Plan 02-04.

This file owns the on-disk side of the auto-generator vertical: the atomic
write orchestrator that composes 02-02 (adapter + grouping), 02-03 (stub
splice + outer dict), and adds the D-7 sort, D-11 path derivation, D-12
command gate, D-9 atomic O_CREAT|O_EXCL|O_WRONLY write + FileExistsError
no-op, D-8 empty-fleet fallback, and D-10 YAML formatting.

Test discipline:
- All filesystem work happens under pytest's `tmp_path` fixture.
- The race test uses `threading.Barrier(2)` per RESEARCH.md Code Example
  lines 676-700 to synchronize concurrent entry into `os.open`.
- Logger is a `MagicMock` so `logger.debug` / `logger.info` assertions
  catch the no-op-if-exists path.
- `args` is a `SimpleNamespace` (not a `MagicMock`) so attribute access
  is strict — missing attributes raise `AttributeError`, which catches
  any drift in the function's expected `args.*` surface.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mlpstorage_py.rules.models import (
    HostCPUInfo,
    HostInfo,
    HostMemoryInfo,
)
from mlpstorage_py.cluster_collector import HostSystemInfo
from mlpstorage_py.system_description.auto_generator import (
    _SYSTEMNAME_YAML_MODE,
    _resolve_host_info_list,
    write_systemname_yaml,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_host(
    *,
    cpu_model: str = "Intel(R) Xeon Platinum 8480+",
    num_cores: int = 56,
    num_sockets: int = 2,
    mem_bytes: int = 274_877_906_944,  # 256 GiB exact
    os_name: str = "Rocky Linux",
    os_version: str = "9.5",
    hostname: str = "h1",
) -> HostInfo:
    """Build a HostInfo with sensible Phase 2 defaults."""
    return HostInfo(
        hostname=hostname,
        cpu=HostCPUInfo(
            model=cpu_model,
            num_cores=num_cores,
            num_logical_cores=num_cores * 2,
            num_sockets=num_sockets,
            architecture="x86_64",
        ),
        memory=HostMemoryInfo(total=mem_bytes),
        system=HostSystemInfo(
            hostname=hostname,
            os_release={"NAME": os_name, "VERSION_ID": os_version},
        ),
    )


def _make_cluster_info(num_hosts: int = 3, **host_kwargs) -> MagicMock:
    """MagicMock with `host_info_list = [HostInfo, ...]`."""
    ci = MagicMock()
    ci.host_info_list = [
        _make_host(hostname=f"h{i}", **host_kwargs) for i in range(num_hosts)
    ]
    return ci


@pytest.fixture
def args(tmp_path) -> SimpleNamespace:
    """Default `args` for write_systemname_yaml — `command='run'`, D-11 path triples."""
    return SimpleNamespace(
        command="run",
        results_dir=str(tmp_path),
        mode="closed",
        orgname="Acme",
        systemname="sys-v1",
    )


@pytest.fixture
def cluster_info() -> MagicMock:
    """Default 3-host homogeneous fleet."""
    return _make_cluster_info(num_hosts=3)


@pytest.fixture
def target_path(tmp_path) -> Path:
    """Expected D-11 canonical path for the default `args`."""
    return tmp_path / "closed" / "Acme" / "systems" / "sys-v1.yaml"


# ---------------------------------------------------------------------------
# LIFE-01 / D-11 — canonical path + happy path
# ---------------------------------------------------------------------------


def test_writes_at_canonical_path(args, cluster_info, target_path):
    """LIFE-01: file appears at `<rd>/<mode>/<org>/systems/<sys>.yaml`."""
    # Sanity: target dir does NOT pre-exist (we want to prove mkdir works).
    assert not target_path.parent.exists()

    returned = write_systemname_yaml(args, cluster_info, MagicMock())

    assert returned == str(target_path)
    assert target_path.exists()

    data = yaml.safe_load(target_path.read_text())
    assert data["system_under_test"]["clients"][0]["quantity"] == 3
    assert (
        data["system_under_test"]["clients"][0]["chassis"]["cpu_model"]
        == "Intel(R) Xeon Platinum 8480+"
    )
    # mkdir created `systems/` on demand.
    assert target_path.parent.exists()


def test_path_parent_mkdir_creates_systems_dir(args, cluster_info, tmp_path):
    """`<rd>/<mode>/<org>/` exists but `systems/` does not → mkdir creates it."""
    # Pre-create everything except systems/.
    (tmp_path / "closed" / "Acme").mkdir(parents=True)
    assert not (tmp_path / "closed" / "Acme" / "systems").exists()

    write_systemname_yaml(args, cluster_info, MagicMock())

    assert (tmp_path / "closed" / "Acme" / "systems").is_dir()


# ---------------------------------------------------------------------------
# D-9 — no-op-if-exists
# ---------------------------------------------------------------------------


def test_no_op_if_exists(args, cluster_info, target_path):
    """Pre-existing file → return None, file content unchanged, logger.debug called."""
    target_path.parent.mkdir(parents=True)
    target_path.write_text("existing: content\n")

    logger = MagicMock()
    returned = write_systemname_yaml(args, cluster_info, logger)

    assert returned is None
    assert target_path.read_text() == "existing: content\n"
    # logger.debug fired at least once with a message about the existing file.
    assert logger.debug.called
    debug_messages = " ".join(str(c) for c in logger.debug.call_args_list)
    assert "exists" in debug_messages.lower() or "no-op" in debug_messages.lower()


# ---------------------------------------------------------------------------
# D-9 — atomic concurrent-write race (T-2-01)
# ---------------------------------------------------------------------------


def test_concurrent_writers_one_wins(args, cluster_info, target_path):
    """T-2-01: two simultaneous writers → exactly one wins, the other returns None.

    Uses `threading.Barrier(2)` to synchronize both threads' entry into
    `os.open(..., O_CREAT|O_EXCL|O_WRONLY)` so the kernel-level race is
    actually exercised. Per RESEARCH.md Code Example lines 676-700.
    """
    barrier = threading.Barrier(2)
    results: list = []

    def worker():
        barrier.wait()  # Synchronize both threads' entry.
        results.append(write_systemname_yaml(args, cluster_info, MagicMock()))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    paths = [r for r in results if r is not None]
    nones = [r for r in results if r is None]

    assert len(paths) == 1, f"expected exactly one winner, got {results}"
    assert len(nones) == 1, f"expected exactly one loser, got {results}"
    assert target_path.exists()
    assert paths[0] == str(target_path)


# ---------------------------------------------------------------------------
# D-12 — command gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd", ["datagen", "configview", "datasize", "validate", "history", "reportgen"],
)
def test_non_run_commands_skip_write(args, cluster_info, target_path, cmd):
    """D-12: only `command='run'` writes; all other commands skip."""
    args.command = cmd
    returned = write_systemname_yaml(args, cluster_info, MagicMock())
    assert returned is None
    assert not target_path.exists()


def test_run_command_writes(args, cluster_info, target_path):
    """D-12 positive: `command='run'` writes."""
    args.command = "run"
    returned = write_systemname_yaml(args, cluster_info, MagicMock())
    assert returned == str(target_path)
    assert target_path.exists()


# ---------------------------------------------------------------------------
# D-8 — empty-fleet fallback
# ---------------------------------------------------------------------------


_FAKE_LOCAL_COLLECTED = {
    "hostname": "local-h",
    "meminfo": {"MemTotal": 274_877_906_944 // 1024},  # kB → bytes via from_proc_meminfo_dict
    "cpuinfo": [
        {"processor": "0", "model name": "Local CPU", "cpu cores": "4",
         "physical id": "0", "siblings": "8"},
        {"processor": "1", "model name": "Local CPU", "cpu cores": "4",
         "physical id": "0", "siblings": "8"},
    ],
    "os_release": {"NAME": "Local OS", "VERSION_ID": "1.0"},
}


def test_empty_fleet_fallback_writes_single_stanza(args, target_path):
    """D-8: cluster_info=None → collect_local_system_info called → 1 stanza, qty=1."""
    with patch(
        "mlpstorage_py.system_description.auto_generator.collect_local_system_info",
        return_value=_FAKE_LOCAL_COLLECTED,
    ) as mock_collect:
        returned = write_systemname_yaml(args, None, MagicMock())

    assert returned == str(target_path)
    mock_collect.assert_called_once()
    data = yaml.safe_load(target_path.read_text())
    clients = data["system_under_test"]["clients"]
    assert len(clients) == 1
    assert clients[0]["quantity"] == 1


def test_empty_cluster_info_host_list_falls_back(args, target_path):
    """D-8 edge: cluster_info.host_info_list = [] also triggers fallback."""
    ci = MagicMock()
    ci.host_info_list = []
    with patch(
        "mlpstorage_py.system_description.auto_generator.collect_local_system_info",
        return_value=_FAKE_LOCAL_COLLECTED,
    ) as mock_collect:
        returned = write_systemname_yaml(args, ci, MagicMock())

    assert returned == str(target_path)
    mock_collect.assert_called_once()
    data = yaml.safe_load(target_path.read_text())
    assert data["system_under_test"]["clients"][0]["quantity"] == 1


def test_resolve_host_info_list_passthrough():
    """`_resolve_host_info_list` returns the existing list when populated."""
    hosts = [_make_host(hostname="a"), _make_host(hostname="b")]
    ci = MagicMock()
    ci.host_info_list = hosts
    assert _resolve_host_info_list(ci) is hosts


def test_resolve_host_info_list_none_triggers_collector():
    """`_resolve_host_info_list(None)` calls `collect_local_system_info`."""
    with patch(
        "mlpstorage_py.system_description.auto_generator.collect_local_system_info",
        return_value=_FAKE_LOCAL_COLLECTED,
    ) as mock_collect:
        result = _resolve_host_info_list(None)
    assert mock_collect.called
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], HostInfo)


# ---------------------------------------------------------------------------
# D-10 — YAML formatting
# ---------------------------------------------------------------------------


def test_yaml_formatting_document_marker(args, cluster_info, target_path):
    """D-10: emitted bytes start with `---\\n` (explicit_start=True)."""
    write_systemname_yaml(args, cluster_info, MagicMock())
    assert target_path.read_text().startswith("---\n")


def test_yaml_formatting_strings_double_quoted(args, cluster_info, target_path):
    """D-10: with default_style='"' PyYAML double-quotes ALL scalars and KEYS.

    Surprise vs. PLAN: `default_style='"'` quotes keys too (not just values),
    so the emitted text contains `"cpu_model": "Intel..."` (both sides quoted),
    not `cpu_model: "Intel..."`. Semantic intent (D-10: strings round-trip as
    strings via yaml.safe_load, no plain-scalar misinterpretation) is still
    locked — the round-trip test below proves it.
    """
    write_systemname_yaml(args, cluster_info, MagicMock())
    text = target_path.read_text()
    # cpu_model value is a string and must be double-quoted on the value side.
    assert re.search(r'"cpu_model":\s*"[^"]+"', text), (
        f"cpu_model not double-quoted in:\n{text}"
    )
    # operating_system.name value is a string and must be double-quoted.
    assert re.search(r'"name":\s*"[^"]+"', text), (
        f"name not double-quoted in:\n{text}"
    )
    # Round-trip: strings load back as Python strings.
    data = yaml.safe_load(text)
    assert isinstance(
        data["system_under_test"]["clients"][0]["chassis"]["cpu_model"], str
    )


def test_yaml_formatting_integers_round_trip_as_int(args, cluster_info, target_path):
    """D-10 / Pitfall 6 (corrected): integers must round-trip as Python `int`.

    Surprise vs. PLAN: modern PyYAML with `default_style='"'` emits integers
    as `!!int "N"` (tagged double-quoted), NOT as bare unquoted ints. The
    PLAN claim that "PyYAML emits int natively even with default_style='\"'"
    is incorrect for this PyYAML version. What MATTERS for the schema
    validator and submission checker is that `quantity`, `cpu_qty`,
    `cpu_cores`, and `memory_capacity` round-trip as Python `int` — which
    the `!!int` tag guarantees. This test locks the round-trip type, not
    the on-disk byte pattern.
    """
    write_systemname_yaml(args, cluster_info, MagicMock())
    data = yaml.safe_load(target_path.read_text())
    client = data["system_under_test"]["clients"][0]
    assert isinstance(client["quantity"], int)
    assert isinstance(client["chassis"]["cpu_qty"], int)
    assert isinstance(client["chassis"]["cpu_cores"], int)
    assert isinstance(client["chassis"]["memory_capacity"], int)
    assert client["quantity"] == 3


def test_yaml_formatting_integers_tagged_not_string(args, cluster_info, target_path):
    """Lock the `!!int` tag emission so a PyYAML version-bump that drops it
    (and silently turns ints into strings) is caught at test time.

    `!!int "N"` (tagged) round-trips as int; bare `"N"` (no tag) would
    round-trip as str and break the schema validator.
    """
    write_systemname_yaml(args, cluster_info, MagicMock())
    text = target_path.read_text()
    # quantity must be either bare-unquoted (`quantity: 3`) OR `!!int`-tagged
    # (`"quantity": !!int "3"`). Both round-trip as int. Forbid the
    # untagged-double-quoted form (`"quantity": "3"`) which would round-trip
    # as str.
    assert re.search(r'"quantity":\s+!!int\s+"\d+"', text) or re.search(
        r"quantity:\s+\d+\s*$", text, re.MULTILINE
    ), f"quantity must be int-tagged or bare int, got:\n{text}"


def test_yaml_block_style(args, cluster_info, target_path):
    """D-10: no `{` flow markers. `[` only appears in legitimate empty-list cases.

    Allowed empty-list keys (kept as explicit `key: []` for self-documenting output
    so readers see "nothing here" rather than a silent omission):
      - traffic        (Phase 2 D-10 precedent)
      - sysctl         (Phase 4 — allowlist-driven, empty is meaningful)
      - environment    (Phase 4 — allowlist-driven, empty is meaningful)
    Drives is omitted entirely when empty per D-33 (client nodes commonly have none).
    """
    write_systemname_yaml(args, cluster_info, MagicMock())
    text = target_path.read_text()
    assert "{" not in text, f"flow-style {{ leaked in:\n{text}"
    stripped = text
    for key in ("traffic", "sysctl", "environment"):
        stripped = re.sub(rf'"{key}":\s*\[\]', "", stripped)
        stripped = re.sub(rf"{key}:\s*\[\]", "", stripped)
    assert "[" not in stripped, (
        f"flow-style [ leaked outside allowed empty-list keys in:\n{stripped}"
    )


# ---------------------------------------------------------------------------
# D-7 — stanza ordering
# ---------------------------------------------------------------------------


def test_stanza_ordering_homogeneous_passthrough(args, cluster_info, target_path):
    """Single-stanza fleet: 3 identical hosts → 1 stanza, quantity=3."""
    write_systemname_yaml(args, cluster_info, MagicMock())
    data = yaml.safe_load(target_path.read_text())
    clients = data["system_under_test"]["clients"]
    assert len(clients) == 1
    assert clients[0]["quantity"] == 3


def test_stanza_ordering_largest_quantity_first(args, target_path):
    """D-7: input [qty=1 X, qty=3 Y] → output [qty=3 Y, qty=1 X]."""
    ci = MagicMock()
    ci.host_info_list = [
        _make_host(cpu_model="X", hostname="x1"),
        _make_host(cpu_model="Y", hostname="y1"),
        _make_host(cpu_model="Y", hostname="y2"),
        _make_host(cpu_model="Y", hostname="y3"),
    ]
    write_systemname_yaml(args, ci, MagicMock())
    data = yaml.safe_load(target_path.read_text())
    clients = data["system_under_test"]["clients"]
    assert len(clients) == 2
    assert clients[0]["chassis"]["cpu_model"] == "Y"
    assert clients[0]["quantity"] == 3
    assert clients[1]["chassis"]["cpu_model"] == "X"
    assert clients[1]["quantity"] == 1


def test_stanza_ordering_alphabetical_tiebreak(args, target_path):
    """D-7: input [qty=2 Zen, qty=2 Atom] → output [qty=2 Atom, qty=2 Zen]."""
    ci = MagicMock()
    ci.host_info_list = [
        _make_host(cpu_model="Zen", hostname="z1"),
        _make_host(cpu_model="Zen", hostname="z2"),
        _make_host(cpu_model="Atom", hostname="a1"),
        _make_host(cpu_model="Atom", hostname="a2"),
    ]
    write_systemname_yaml(args, ci, MagicMock())
    data = yaml.safe_load(target_path.read_text())
    clients = data["system_under_test"]["clients"]
    assert len(clients) == 2
    assert clients[0]["chassis"]["cpu_model"] == "Atom"
    assert clients[0]["quantity"] == 2
    assert clients[1]["chassis"]["cpu_model"] == "Zen"
    assert clients[1]["quantity"] == 2


# ---------------------------------------------------------------------------
# D-14 round-trip — outer-dict omissions survive yaml.safe_dump
# ---------------------------------------------------------------------------


def test_systemname_yaml_omits_solution_deployment_in_emitted_file(
    args, cluster_info, target_path,
):
    """D-14 integration: `solution`, `deployment`, etc. absent in emitted YAML."""
    write_systemname_yaml(args, cluster_info, MagicMock())
    data = yaml.safe_load(target_path.read_text())
    sut = data["system_under_test"]
    for forbidden in (
        "solution",
        "deployment",
        "product_nodes",
        "product_switches",
        "total_rack_units",
        "rack_power_supplies",
    ):
        assert forbidden not in sut, f"D-14 violation: {forbidden} present"


# ---------------------------------------------------------------------------
# T-2-08 — symlink attack
# ---------------------------------------------------------------------------


def test_symlink_attack_at_target_path_returns_none(args, cluster_info, tmp_path):
    """T-2-08: pre-existing symlink at target path → O_EXCL refuses to create.

    Verifies POSIX guarantee that O_CREAT|O_EXCL fails if the path resolves to
    anything pre-existing — including a symlink. The symlink's target file
    MUST remain unchanged.
    """
    innocent = tmp_path / "innocent.txt"
    innocent.write_text("innocent")

    target_dir = tmp_path / "closed" / "Acme" / "systems"
    target_dir.mkdir(parents=True)
    target = target_dir / "sys-v1.yaml"
    os.symlink(str(innocent), str(target))

    returned = write_systemname_yaml(args, cluster_info, MagicMock())

    assert returned is None
    # The symlink target is unchanged.
    assert innocent.read_text() == "innocent"


# ---------------------------------------------------------------------------
# D-9 — filesystem errors propagate (not swallowed)
# ---------------------------------------------------------------------------


def test_filesystem_error_propagates_eacces(args, cluster_info, tmp_path):
    """D-9: non-FileExistsError filesystem errors propagate as exceptions."""
    if os.geteuid() == 0:
        pytest.skip("root bypasses chmod restrictions")

    # Pre-create the org directory and make it non-writable so mkdir of
    # `systems/` (or the os.open) fails.
    org_dir = tmp_path / "closed" / "Acme"
    org_dir.mkdir(parents=True)
    os.chmod(str(org_dir), 0o555)
    try:
        with pytest.raises((PermissionError, OSError)):
            write_systemname_yaml(args, cluster_info, MagicMock())
    finally:
        # Restore so pytest can clean up tmp_path.
        os.chmod(str(org_dir), 0o755)


# ---------------------------------------------------------------------------
# D-15 — no validate_file call inside writer
# ---------------------------------------------------------------------------


def test_writer_does_not_call_schema_validator_validate_file(args, cluster_info):
    """D-15: writer must NOT call schema_validator.validate_file post-write."""
    with patch(
        "mlpstorage_py.system_description.schema_validator.validate_file",
        side_effect=AssertionError("must not be called per D-15"),
    ):
        write_systemname_yaml(args, cluster_info, MagicMock())


# ---------------------------------------------------------------------------
# Mode constant sanity
# ---------------------------------------------------------------------------


def test_systemname_yaml_mode_is_0o644():
    """`_SYSTEMNAME_YAML_MODE` mirrors `sentinel.py:_SENTINEL_MODE` (LAY-03 parity)."""
    assert _SYSTEMNAME_YAML_MODE == 0o644
