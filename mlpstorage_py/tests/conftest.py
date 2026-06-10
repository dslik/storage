"""
Pytest fixture factory for SubmissionStructureCheck tests.

Provides:
  - ``MockLogger`` — capture-mode logger that stores ``error()`` and
    ``warning()`` calls as formatted strings.
  - ``mock_logger`` fixture — yields a fresh ``MockLogger`` per test.
  - ``build_submission(tmp_path, **overrides)`` — factory that builds a
    minimal but valid closed-only submission tree under ``tmp_path``.

Run with:
    pytest mlpstorage_py/tests/ -v
"""

import json
import os
import shutil
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# MockLogger — capture-mode logger
# ---------------------------------------------------------------------------

class MockLogger:
    """Capture-mode mock logger for STRUCT-* tests.

    Stores ``error()`` and ``warning()`` calls as fully-formatted strings
    (``msg % args`` applied) so tests can assert on the locked
    ``[<rule_id> <rule_name>] ...`` prefix without pytest caplog plumbing.

    All other logging methods (``info``, ``debug``, ``verbose``,
    ``verboser``, ``ridiculous``) are no-ops.
    """

    def __init__(self):
        self.errors = []    # list[str] — formatted error messages
        self.warnings = []  # list[str] — formatted warning messages

    def error(self, msg, *args):
        self.errors.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        pass

    def debug(self, msg, *args):
        pass

    def verbose(self, msg, *args):
        pass

    def verboser(self, msg, *args):
        pass

    def ridiculous(self, msg, *args):
        pass


@pytest.fixture
def mock_logger():
    """Return a fresh MockLogger per test."""
    return MockLogger()


# ---------------------------------------------------------------------------
# Default submission tree constants
# ---------------------------------------------------------------------------

_SUBMITTER = "Acme"
_SYSNAME = "acme-storage-v1"

# Six run timestamps (1 warm-up + 5 measured per Rules.md 2.1.17)
_RUN_TIMESTAMPS = [
    "20250111_140001",
    "20250111_150001",
    "20250111_160001",
    "20250111_170001",
    "20250111_180001",
    "20250111_190001",
]
# One datagen timestamp
_DATAGEN_TIMESTAMPS = ["20250111_130000"]

# Ten checkpointing timestamps
_CHKPT_TIMESTAMPS = [
    "20250112_100001",
    "20250112_110001",
    "20250112_120001",
    "20250112_130001",
    "20250112_140001",
    "20250112_150001",
    "20250112_160001",
    "20250112_170001",
    "20250112_180001",
    "20250112_190001",
]

# Stable 3-file code/ tree content (binary-stable, deterministic)
_CODE_FILES = {
    "mod.py": b"# mod\ndef hello():\n    return 'hello'\n",
    "helper.py": b"# helper\ndef util():\n    return 42\n",
    "README.md": b"# Submission Code\n\nThis is the reference implementation.\n",
}

# Default summary.json content per run timestamp (feeds STRUCT-09)
_DEFAULT_SUMMARY = {
    "num_hosts": 2,
    "host_memory_GB": [256, 256],
}


def _build_system_yaml(submission_name: str, multi_host: bool = True) -> dict:
    """Build a schema-valid system YAML dict for the given submission_name."""
    return {
        "system_under_test": {
            "solution": {
                "submission_name": submission_name,
                "friendly_description": "Test NAS system",
                "architecture": {
                    "storage_location": "remote",
                    "benchmark_API": "file",
                    "product_API": "file",
                    "client_footprint": "open_source",
                    "client_installation": "in_box",
                },
                "capabilities": {
                    "multi_host": multi_host,
                    "simultaneous_write": True,
                    "simultaneous_read": True,
                    "remap_time_in_seconds": 0,
                },
            },
            "deployment": "onprem",
            "clients": [
                {
                    "friendly_description": "Benchmark client",
                    "quantity": 2,
                    "chassis": {
                        "model_name": "TestServer-A",
                        "rack_units": 2,
                        "cpu_model": "Xeon Gold 6338",
                        "cpu_qty": 2,
                        "cpu_cores": 64,
                        "memory_capacity": 256,
                        "power": {
                            "min_psus_active": 1,
                            "psus_configured": [
                                {
                                    "power_capacity_watts": 800,
                                    "efficiency": "Platinum",
                                    "unit_count": 2,
                                }
                            ],
                        },
                    },
                    "operating_system": {
                        "name": "RHEL",
                        "version": "9.2",
                    },
                }
            ],
        }
    }


# ---------------------------------------------------------------------------
# build_submission — fixture factory
# ---------------------------------------------------------------------------

def build_submission(tmp_path, **overrides) -> Path:
    """Build a minimal valid MLPerf Storage submission tree under *tmp_path*.

    Returns the root path (``tmp_path`` itself).  Instantiate
    ``SubmissionStructureCheck(log, config, str(tmp_path))`` to walk from
    there.

    Default tree (closed-only, one system, one training workload, one
    checkpointing workload):

    .. code-block:: text

        tmp_path/
          closed/
            Acme/
              code/
                mod.py
                helper.py
                README.md
              systems/
                acme-storage-v1.yaml   (schema-valid)
                acme-storage-v1.pdf    (1-byte placeholder)
              results/
                acme-storage-v1/
                  training/
                    unet3d/
                      datagen/
                        20250111_130000/
                      run/
                        20250111_140001/   (+ summary.json)
                        ... (6 total)
                  checkpointing/
                    llama3-8b/
                      20250112_100001/   (+ summary.json)
                      ... (10 total)

    Mutation kwargs (sealed — unknown kwargs raise ``TypeError``):

    * ``submitter_name_with_space`` (bool)  — name becomes "Acme Storage"
    * ``top_level_capitalcase`` (bool)      — "closed" → "CLOSED"
    * ``extra_top_level`` (str)             — adds an extra top-level dir
    * ``no_top_level_dirs`` (bool)          — removes closed/ entirely
    * ``open_mismatches_closed`` (bool)     — adds open/ missing one subdir
    * ``wrong_submitter_in_closed`` (bool)  — closed/OtherAcme/ instead
    * ``multiple_submitters_in_closed`` (bool) — two submitter dirs under closed/
    * ``missing_required_subdir`` (str)     — removes code/results/systems
    * ``extra_submitter_subdir`` (str)      — adds a stray dir under submitter
    * ``mutate_code`` (bool)                — adds extra file → hash differs
    * ``set_reference_checksum`` (str)      — unused in tree; caller passes to Config
    * ``code_with_symlink`` (bool)          — adds a symlink in code/
    * ``code_with_pycache`` (bool)          — adds code/pkg/__pycache__/mod.pyc
    * ``unpaired_yaml`` (bool)              — systems/ yaml without pdf
    * ``extra_systems_file`` (str)          — adds a stray file in systems/
    * ``unpaired_results_system`` (bool)    — adds results/no-yaml-for-this/
    * ``missing_systems_pdf`` (bool)        — drops systems/<sysname>.pdf
    * ``submission_name_mismatch`` (bool)   — YAML submission_name ≠ <name>
    * ``num_hosts_mismatch`` (bool)         — summary.json num_hosts mismatch
    * ``memory_mismatch`` (bool)            — summary.json host_memory_GB mismatch
    * ``multi_host_capability_inconsistent`` (bool) — multi_host=False + num_hosts>1
    * ``missing_summary_field`` (str)       — drops a field from summary.json
    * ``extra_workload_category`` (str)     — adds a stray workload category dir
    * ``wrong_training_workload`` (str)     — adds invalid training workload dir
    * ``wrong_training_phase`` (str)        — adds invalid training phase dir
    * ``datagen_timestamps`` (int)          — overrides datagen timestamp count
    * ``bad_datagen_timestamp_format`` (bool) — uses non-timestamp datagen dir name
    * ``wrong_checkpointing_workload`` (str) — adds invalid checkpointing workload
    """
    # -----------------------------------------------------------------------
    # Pop all known overrides before the sealed-enum guard runs
    # -----------------------------------------------------------------------
    submitter_name_with_space = overrides.pop("submitter_name_with_space", False)
    top_level_capitalcase = overrides.pop("top_level_capitalcase", False)
    extra_top_level = overrides.pop("extra_top_level", None)
    no_top_level_dirs = overrides.pop("no_top_level_dirs", False)
    open_mismatches_closed = overrides.pop("open_mismatches_closed", False)
    wrong_submitter_in_closed = overrides.pop("wrong_submitter_in_closed", False)
    multiple_submitters_in_closed = overrides.pop("multiple_submitters_in_closed", False)
    missing_required_subdir = overrides.pop("missing_required_subdir", None)
    extra_submitter_subdir = overrides.pop("extra_submitter_subdir", None)
    mutate_code = overrides.pop("mutate_code", False)
    set_reference_checksum = overrides.pop("set_reference_checksum", None)  # caller uses this
    code_with_symlink = overrides.pop("code_with_symlink", False)
    code_with_pycache = overrides.pop("code_with_pycache", False)
    unpaired_yaml = overrides.pop("unpaired_yaml", False)
    extra_systems_file = overrides.pop("extra_systems_file", None)
    unpaired_results_system = overrides.pop("unpaired_results_system", False)
    missing_systems_pdf = overrides.pop("missing_systems_pdf", False)
    submission_name_mismatch = overrides.pop("submission_name_mismatch", False)
    num_hosts_mismatch = overrides.pop("num_hosts_mismatch", False)
    memory_mismatch = overrides.pop("memory_mismatch", False)
    multi_host_capability_inconsistent = overrides.pop("multi_host_capability_inconsistent", False)
    missing_summary_field = overrides.pop("missing_summary_field", None)
    extra_workload_category = overrides.pop("extra_workload_category", None)
    wrong_training_workload = overrides.pop("wrong_training_workload", None)
    wrong_training_phase = overrides.pop("wrong_training_phase", None)
    datagen_timestamps_count = overrides.pop("datagen_timestamps", None)
    bad_datagen_timestamp_format = overrides.pop("bad_datagen_timestamp_format", False)
    wrong_checkpointing_workload = overrides.pop("wrong_checkpointing_workload", None)

    # Sealed-enum guard — any leftover key is unknown
    if overrides:
        raise TypeError(f"unknown override: {sorted(overrides)}")

    # -----------------------------------------------------------------------
    # Determine submitter name
    # -----------------------------------------------------------------------
    submitter = _SUBMITTER
    if submitter_name_with_space:
        submitter = "Acme Storage"

    # -----------------------------------------------------------------------
    # Determine top-level division name
    # -----------------------------------------------------------------------
    division = "closed"
    if top_level_capitalcase:
        division = "CLOSED"

    # -----------------------------------------------------------------------
    # Build the tree
    # -----------------------------------------------------------------------
    # STRUCT-04 (D-277) requires the input root basename to equal the submitter
    # name. Nest the build inside tmp_path/Acme/ so that
    # os.path.basename(root) == _SUBMITTER for the default tree.
    root = Path(tmp_path) / _SUBMITTER
    root.mkdir(parents=True, exist_ok=True)

    if not no_top_level_dirs:
        # Division directory
        div_path = root / division
        div_path.mkdir(parents=True)

        # Submitter directory inside division
        if wrong_submitter_in_closed:
            # Use a different submitter name inside closed/
            sub_path = div_path / "OtherAcme"
        else:
            sub_path = div_path / submitter
        sub_path.mkdir()

        if multiple_submitters_in_closed:
            (div_path / "AlsoAcme").mkdir()

        # Required subdirectories
        required_subdirs = {"code", "results", "systems"}
        if missing_required_subdir:
            required_subdirs.discard(missing_required_subdir)

        for sd in required_subdirs:
            (sub_path / sd).mkdir()

        if extra_submitter_subdir:
            (sub_path / extra_submitter_subdir).mkdir()

        code_path = sub_path / "code"
        results_path = sub_path / "results"
        systems_path = sub_path / "systems"

        # ---------------------------------------------------------------
        # code/ tree
        # ---------------------------------------------------------------
        if code_path.exists():
            for fname, content in _CODE_FILES.items():
                (code_path / fname).write_bytes(content)

            if mutate_code:
                (code_path / "extra_unexpected.py").write_bytes(b"# extra\n")

            if code_with_symlink:
                target = code_path / "mod.py"
                link = code_path / "link_to_mod.py"
                os.symlink(str(target), str(link))

            if code_with_pycache:
                pkg_dir = code_path / "pkg"
                pkg_dir.mkdir()
                pycache_dir = pkg_dir / "__pycache__"
                pycache_dir.mkdir()
                (pycache_dir / "mod.pyc").write_bytes(b"\x00\x00\x00\x00")

        # ---------------------------------------------------------------
        # systems/ directory
        # ---------------------------------------------------------------
        if systems_path.exists():
            # Determine system YAML content
            yaml_submission_name = _SYSNAME
            if submission_name_mismatch:
                yaml_submission_name = "wrong-name-here"

            multi_host_val = True
            if multi_host_capability_inconsistent:
                multi_host_val = False

            sys_yaml_dict = _build_system_yaml(
                yaml_submission_name,
                multi_host=multi_host_val,
            )
            yaml_content = yaml.dump(sys_yaml_dict, default_flow_style=False)

            if not unpaired_yaml:
                (systems_path / f"{_SYSNAME}.yaml").write_text(yaml_content, encoding="utf-8")
            else:
                # No .pdf — yaml without matching pdf
                (systems_path / f"{_SYSNAME}.yaml").write_text(yaml_content, encoding="utf-8")
                # deliberately do NOT write the pdf

            if not missing_systems_pdf and not unpaired_yaml:
                # Write PDF placeholder
                (systems_path / f"{_SYSNAME}.pdf").write_bytes(b"%PDF")

            if extra_systems_file:
                (systems_path / extra_systems_file).write_text("stray content\n")

        # ---------------------------------------------------------------
        # results/ subtree
        # ---------------------------------------------------------------
        if results_path.exists():
            sys_results = results_path / _SYSNAME
            sys_results.mkdir()

            # Determine summary.json content
            base_summary = dict(_DEFAULT_SUMMARY)
            if num_hosts_mismatch:
                base_summary["num_hosts"] = 3  # system YAML has quantity:2 → mismatch
            if memory_mismatch:
                base_summary["host_memory_GB"] = [128, 128]  # system YAML has 256
            if missing_summary_field:
                base_summary.pop(missing_summary_field, None)
            if multi_host_capability_inconsistent:
                base_summary["num_hosts"] = 2  # capabilities.multi_host=False but num_hosts=2

            # training/unet3d
            training_path = sys_results / "training"
            training_path.mkdir()

            unet3d_path = training_path / "unet3d"
            unet3d_path.mkdir()

            # datagen timestamps
            datagen_dir = unet3d_path / "datagen"
            datagen_dir.mkdir()
            if bad_datagen_timestamp_format:
                dg_timestamps = ["not-a-timestamp"]
            elif datagen_timestamps_count is not None:
                dg_timestamps = [
                    f"2025011{i}_130000" for i in range(1, datagen_timestamps_count + 1)
                ]
            else:
                dg_timestamps = _DATAGEN_TIMESTAMPS
            for ts in dg_timestamps:
                ts_dir = datagen_dir / ts
                ts_dir.mkdir()

            # run timestamps
            run_dir = unet3d_path / "run"
            run_dir.mkdir()
            for ts in _RUN_TIMESTAMPS:
                ts_dir = run_dir / ts
                ts_dir.mkdir()
                (ts_dir / "summary.json").write_text(
                    json.dumps(base_summary), encoding="utf-8"
                )

            if wrong_training_workload:
                (training_path / wrong_training_workload).mkdir()

            if wrong_training_phase:
                (unet3d_path / wrong_training_phase).mkdir()

            if extra_workload_category:
                (sys_results / extra_workload_category).mkdir()

            # checkpointing/llama3-8b
            chkpt_path = sys_results / "checkpointing"
            chkpt_path.mkdir()

            llama_path = chkpt_path / "llama3-8b"
            llama_path.mkdir()
            for ts in _CHKPT_TIMESTAMPS:
                ts_dir = llama_path / ts
                ts_dir.mkdir()
                (ts_dir / "summary.json").write_text(
                    json.dumps(base_summary), encoding="utf-8"
                )

            if wrong_checkpointing_workload:
                (chkpt_path / wrong_checkpointing_workload).mkdir()

            if unpaired_results_system:
                (results_path / "no-yaml-for-this").mkdir(parents=True)

    # -----------------------------------------------------------------------
    # open/ mirror (only when open_mismatches_closed=True)
    # -----------------------------------------------------------------------
    if open_mismatches_closed:
        open_div = root / "open"
        open_div.mkdir(parents=True)
        open_sub = open_div / submitter
        open_sub.mkdir()
        # Add some but not all required subdirs to trigger STRUCT-03
        (open_sub / "results").mkdir()
        # Deliberately omit code/ and systems/

    if extra_top_level:
        (root / extra_top_level).mkdir()

    return root
