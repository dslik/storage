"""
Comprehensive Tests for the MLPerf Storage CLI parser.
Validates structural boundaries, subcommand availability, value constraints,
YAML overrides, post-parse argument updates, and 'closed' vs 'open' parity.
"""

import sys
import pytest
import argparse
from unittest.mock import patch, mock_open
from mlpstorage_py.cli_parser import parse_arguments, update_args, apply_yaml_config_overrides
from mlpstorage_py.config import EXIT_CODE

# =====================================================================
# 1. Open vs. Closed Equivalence & Constraints Tests
# =====================================================================

class TestOpenClosedEquivalence:

    def test_kvcache_open_closed_defaults_match(self):
        """Verify hardcoded defaults in KVCache closed mode match open mode defaults.

        In closed mode, --model and --num-users are not accepted (model is fixed by
        the benchmark; users must be specified via set_defaults). In open mode, they
        are required flags. We verify the common defaults match.
        """
        # Closed mode: model/num-users are set_defaults; only --results-dir required
        with patch('sys.argv', ['mlpstorage', 'closed', 'kvcache', 'run', '-rd', '/tmp',
                                 '-sn', 'sys-v1']):
            args_closed = parse_arguments()

        # Open mode: model and num-users are required flags
        with patch('sys.argv', ['mlpstorage', 'open', 'kvcache', 'run', '-rd', '/tmp',
                                 '-m', 'llama3.1-8b', '-nu', '100', '-sn', 'sys-v1']):
            args_open = parse_arguments()

        # Check common defaults that should match between modes
        assert args_closed.gpu_mem_gb == args_open.gpu_mem_gb == 16.0
        assert args_closed.duration == args_open.duration == 60
        assert args_closed.loops == args_open.loops == 1
        assert args_closed.disable_multi_turn == args_open.disable_multi_turn == False

    def test_checkpointing_open_closed_defaults_match(self):
        """Verify Checkpointing 'closed' forces read/write checkpoint counts to match 'open' defaults."""
        base_args = ['checkpointing', 'run', '-cm', '1024', '-m', 'llama3-8b', '-np', '2', '-cf', '/tmp/ckpt', '-rd', '/tmp', '-sn', 'sys-v1', 'file']

        with patch('sys.argv', ['mlpstorage', 'closed'] + base_args):
            args_closed = parse_arguments()

        with patch('sys.argv', ['mlpstorage', 'open'] + base_args):
            args_open = parse_arguments()

        assert args_closed.num_checkpoints_read == args_open.num_checkpoints_read == 10
        assert args_closed.num_checkpoints_write == args_open.num_checkpoints_write == 10

    def test_closed_mode_strips_open_args(self):
        """Open-mode arguments should trigger an unrecognized argument error if passed in closed mode."""
        test_args = ['mlpstorage', 'closed', 'kvcache', 'run', '-rd', '/tmp', '-sn', 'sys-v1', '--allow-invalid-params']
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code != 0


# =====================================================================
# 2. Structural & Subcommand Combinations (Positive Cases)
# =====================================================================

class TestCLIStructureAndCombinations:

    @pytest.mark.parametrize("test_name, cmd_list, expected_mode_or_benchmark, expected_command", [
        # Training — model is now a positional (no --model flag); storage type is positional
        # closed mode: only 'unet3d' and 'retinanet' are valid model choices
        # NOTE: every emitting command now requires --systemname per D-10/LAY-04.
        ("01", ['training', 'retinanet', 'run', '-cm', '1024', '-at', 'b200', '-na', '4', '-rd', '/tmp', '-sn', 'sys-v1', 'file'], 'training', 'run'),
        ("02", ['training', 'unet3d', 'datasize', '-cm', '1024', '-at', 'b200', '-ma', '4', '-rd', '/tmp', '-sn', 'sys-v1'], 'training', 'datasize'),
        ("03", ['training', 'unet3d', 'datagen', '-np', '4', 'file', '-rd', '/tmp', '-sn', 'sys-v1'], 'training', 'datagen'),
        ("04", ['training', 'unet3d', 'configview', '-na', '4', '-cm', '64', '-at', 'b200', '-rd', '/tmp', '-sn', 'sys-v1', 'file'], 'training', 'configview'),

        # Checkpointing — --model stays as a flag; storage type is positional
        ("05", ['checkpointing', 'run', '-cm', '1024', '-m', 'llama3-8b', '-np', '4', '-cf', '/tmp/ckpt', '-rd', '/tmp', '-sn', 'sys-v1', 'file'], 'checkpointing', 'run'),
        ("06", ['checkpointing', 'datasize', '-cm', '1024', '-m', 'llama3-8b', '-np', '4', '-sn', 'sys-v1'], 'checkpointing', 'datasize'),

        # KVCache closed mode: model/num-users are not accepted in closed mode
        ("07", ['kvcache', 'run', '-rd', '/tmp', '-sn', 'sys-v1'], 'kvcache', 'run'),
        ("08", ['kvcache', 'datasize'], 'kvcache', 'datasize'),

        # VectorDB
        ("09", ['vectordb', 'run', '-rd', '/tmp', '-sn', 'sys-v1', 'file'], 'vectordb', 'run'),
        ("10", ['vectordb', 'datagen', 'file', '-rd', '/tmp', '-sn', 'sys-v1'], 'vectordb', 'datagen'),
        ("11", ['vectordb', 'datasize'], 'vectordb', 'datasize'),

        # Utilities — top-level siblings, no mode prefix needed (they are their own mode)
        ("12", ['reports', 'reportgen', '-rd', '/tmp', '-sn', 'sys-v1'], 'reports', 'reportgen'),
        ("13", ['history', 'show', '-rd', '/tmp', '-sn', 'sys-v1'], 'history', 'show'),
        ("14", ['lockfile', 'generate', '-rd', '/tmp'], 'lockfile', 'generate'),
        ("15", ['lockfile', 'verify', '-rd', '/tmp'], 'lockfile', 'verify'),
    ])
    def test_all_program_subcommand_combinations(self, test_name, cmd_list, expected_mode_or_benchmark, expected_command):
        """Ensure all benchmarks and subcommands can parse their minimum required arguments."""
        # Benchmark commands run under 'closed'; utility commands are top-level
        benchmark_benchmarks = {'training', 'checkpointing', 'vectordb', 'kvcache'}
        if expected_mode_or_benchmark in benchmark_benchmarks:
            test_args = ['mlpstorage', 'closed'] + cmd_list
        else:
            test_args = ['mlpstorage'] + cmd_list

        with patch('sys.argv', test_args):
            args = parse_arguments()

        if expected_mode_or_benchmark in benchmark_benchmarks:
            assert args.mode == "closed", f"[{test_name}] expected mode==closed, got {args.mode}"
            assert args.benchmark == expected_mode_or_benchmark, f"[{test_name}] expected benchmark=={expected_mode_or_benchmark}, got {args.benchmark}"
        else:
            assert args.mode == expected_mode_or_benchmark, f"[{test_name}] expected mode=={expected_mode_or_benchmark}, got {args.mode}"

        cmd_val = getattr(args, 'command', getattr(args, 'lockfile_command', None))
        assert cmd_val == expected_command, f"[{test_name}] expected command=={expected_command}, got {cmd_val}"

    def test_missing_required_results_dir(self):
        """Omitting -rd when req_results=True (e.g., training run) should fail."""
        test_args = ['mlpstorage', 'closed', 'training', 'unet3d', 'run', '-cm', '1024', '-at', 'b200', '-na', '4', 'file']
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code != 0

    def test_data_access_protocol_positional(self):
        """Test that the data_access_protocol positional is set correctly."""
        # Use 'unet3d' — a valid model in closed mode
        test_args = ['mlpstorage', 'closed', 'training', 'unet3d', 'datagen', '-np', '4', 'file', '-rd', '/tmp', '-sn', 'sys-v1']
        with patch('sys.argv', test_args):
            args = parse_arguments()
            assert args.data_access_protocol == 'file'
            # Positional means no separate 'file' or 'object' attributes
            assert not hasattr(args, 'file')
            assert not hasattr(args, 'object')


# =====================================================================
# 3. Validation Rules
# =====================================================================

class TestCustomValidation:

    def test_kvcache_rejects_object_storage(self):
        """KVCache validate_args should reject object storage."""
        # kvcache has no storage type positional; 'object' would be unrecognized
        test_args = ['mlpstorage', 'closed', 'kvcache', 'run', '-rd', '/tmp', 'object']
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code != 0

    def test_checkpointing_rejects_negative_checkpoints(self):
        """Checkpointing validate_args should reject negative checkpoint counts."""
        test_args = [
            'mlpstorage', 'closed', 'checkpointing', 'run',
            '-cm', '1024', '-m', 'llama3-8b', '-np', '2', '-cf', '/tmp/ckpt', '-rd', '/tmp',
            '-sn', 'sys-v1', 'file',
            '--num-checkpoints-read', '-5'
        ]
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code == EXIT_CODE.INVALID_ARGUMENTS


# =====================================================================
# 4. Post-Parse Configuration (update_args & YAML)
# =====================================================================

class TestUpdateArgsAndConfig:

    def test_update_args_normalizes_hosts(self):
        """update_args should handle messy host strings and normalize them to a clean list."""
        args = argparse.Namespace(hosts="host1, host2   host3,host4")
        update_args(args)
        assert args.hosts == ['host1', 'host2', 'host3', 'host4']
        assert args.num_client_hosts == 4

    def test_update_args_empty_hosts_fails(self):
        """update_args should exit if host normalization results in an empty list."""
        args = argparse.Namespace(hosts=" , , ")
        with pytest.raises(SystemExit) as exc_info:
            update_args(args)
        assert exc_info.value.code == EXIT_CODE.INVALID_ARGUMENTS

    def test_update_args_process_nomenclature_mapping(self):
        """update_args should unify 'num_accelerators' or 'max_accelerators' into 'num_processes'."""
        args = argparse.Namespace(num_accelerators=8)
        update_args(args)
        assert args.num_processes == 8

    def test_update_args_flattens_params(self):
        """update_args should flatten lists of lists for params/mpi_params resulting from multiple append actions."""
        args = argparse.Namespace(params=[['key=val1'], ['key=val2']])
        update_args(args)
        assert args.params == ['key=val1', 'key=val2']

    def test_yaml_config_overrides(self):
        """apply_yaml_config_overrides should update namespace attributes safely."""
        mock_yaml_content = """
        duration: 999
        hosts: "node1,node2"
        params:
            batch_size: 32
        """
        # Create a namespace simulating a parsed output
        initial_args = argparse.Namespace(
            config_file="dummy.yaml",
            duration=100,
            hosts=['default_host'],
            params=None
        )

        with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
            updated_args = apply_yaml_config_overrides(initial_args)

        assert updated_args.duration == 999
        assert updated_args.hosts == ['node1', 'node2']  # Special yaml handling for hosts


# =====================================================================
# 5. args.mode and args.benchmark attributes
# =====================================================================

class TestModeAndBenchmarkAttributes:
    """Verify the new args.mode and args.benchmark namespace attributes."""

    def test_closed_training_sets_mode_and_benchmark(self):
        """closed training unet3d run should set mode='closed' and benchmark='training'."""
        test_args = [
            'mlpstorage', 'closed', 'training', 'unet3d', 'run',
            '--data-dir', '/tmp', '--results-dir', '/tmp',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', '--systemname', 'sys-v1', 'file'
        ]
        with patch('sys.argv', test_args):
            args = parse_arguments()
        assert args.mode == 'closed'
        assert args.benchmark == 'training'
        assert args.model == 'unet3d'
        assert args.command == 'run'
        assert args.data_access_protocol == 'file'

    def test_open_mode_allows_loops(self):
        """open mode should accept --loops flag for training."""
        test_args = [
            'mlpstorage', 'open', 'training', 'unet3d', 'run',
            '--data-dir', '/tmp', '--results-dir', '/tmp',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64',
            '--systemname', 'sys-v1', 'file', '--loops', '3'
        ]
        with patch('sys.argv', test_args):
            args = parse_arguments()
        assert args.mode == 'open'
        assert args.loops == 3

    def test_reports_sets_mode(self):
        """reports subcommand should set mode='reports' (no benchmark attribute)."""
        with patch('sys.argv', ['mlpstorage', 'reports', 'reportgen',
                                '--results-dir', '/tmp', '--systemname', 'sys-v1']):
            args = parse_arguments()
        assert args.mode == 'reports'
        assert not hasattr(args, 'benchmark')

    def test_history_sets_mode(self):
        """history subcommand should set mode='history'."""
        with patch('sys.argv', ['mlpstorage', 'history', 'show',
                                '--results-dir', '/tmp', '--systemname', 'sys-v1']):
            args = parse_arguments()
        assert args.mode == 'history'
        assert not hasattr(args, 'benchmark')

    def test_lockfile_sets_mode(self):
        """lockfile subcommand should set mode='lockfile'."""
        with patch('sys.argv', ['mlpstorage', 'lockfile', 'generate', '--results-dir', '/tmp']):
            args = parse_arguments()
        assert args.mode == 'lockfile'
        assert not hasattr(args, 'benchmark')

    def test_no_file_object_consolidation_needed(self):
        """data_access_protocol is set directly by positional; no 'file' or 'object' attrs remain."""
        test_args = [
            'mlpstorage', 'closed', 'training', 'unet3d', 'run',
            '--data-dir', '/tmp', '--results-dir', '/tmp',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', '--systemname', 'sys-v1', 'file'
        ]
        with patch('sys.argv', test_args):
            args = parse_arguments()
        assert args.data_access_protocol == 'file'
        assert not hasattr(args, 'file')
        assert not hasattr(args, 'object')


# =====================================================================
# 6. --systemname / MLPERF_SYSTEMNAME plumbing (LAY-04, D-10)
# =====================================================================

class TestSystemname:
    """Tests for the --systemname flag and MLPERF_SYSTEMNAME env-var plumbing.

    Per CONTEXT.md D-10, --systemname is required on every emitting subcommand:
    training {datagen, run, configview, datasize}, checkpointing {datagen, run,
    configview, validate}, vectordb {datagen, run}, kvcache {run, datagen}, plus
    reports reportgen and history (show, rerun).

    Pure utility commands (lockfile, version, init, rules-coverage) are exempt
    and continue to parse without --systemname.
    """

    # Sets of full argv (without --systemname) that should each accept the flag.
    EMITTING_COMMANDS = [
        # (label, argv tail)
        ('training-run', [
            'closed', 'training', 'unet3d', 'run',
            '--data-dir', '/d', '--results-dir', '/r',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', 'file',
        ]),
        ('training-datagen', [
            'closed', 'training', 'unet3d', 'datagen', 'file',
            '--data-dir', '/d', '--results-dir', '/r',
            '--num-processes', '1',
        ]),
        ('training-configview', [
            'closed', 'training', 'unet3d', 'configview',
            '--data-dir', '/d', '--results-dir', '/r',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', 'file',
        ]),
        ('training-datasize', [
            'closed', 'training', 'unet3d', 'datasize',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64',
            '--data-dir', '/d', '--results-dir', '/r',
        ]),
        ('checkpointing-run', [
            'closed', 'checkpointing', 'run',
            '-cm', '1024', '-m', 'llama3-8b', '-np', '2',
            '-cf', '/tmp/ckpt', '-rd', '/r', 'file',
        ]),
        ('vectordb-run', [
            'closed', 'vectordb', 'run',
            '-rd', '/r', 'file',
        ]),
        ('kvcache-run', [
            'closed', 'kvcache', 'run', '-rd', '/r',
        ]),
        ('reports-reportgen', [
            'reports', 'reportgen', '--results-dir', '/r',
        ]),
        ('history-show', [
            'history', 'show', '--results-dir', '/r',
        ]),
    ]

    @pytest.mark.parametrize(
        "label,argv_tail",
        EMITTING_COMMANDS,
        ids=[label for label, _ in EMITTING_COMMANDS],
    )
    def test_systemname_on_emitting_commands(self, label, argv_tail, monkeypatch):
        """Every emitting subcommand accepts --systemname and binds it to args.systemname."""
        monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
        full = ['mlpstorage'] + argv_tail + ['--systemname', 'sys-v1']
        with patch('sys.argv', full):
            args = parse_arguments()
        assert getattr(args, 'systemname', None) == 'sys-v1', (
            f"{label}: expected args.systemname=='sys-v1' got "
            f"{getattr(args, 'systemname', None)!r}"
        )

    @pytest.mark.parametrize(
        "label,argv_tail",
        EMITTING_COMMANDS,
        ids=[label for label, _ in EMITTING_COMMANDS],
    )
    def test_empty_systemname_errors(self, label, argv_tail, monkeypatch):
        """Omitting --systemname on emitting commands raises SystemExit (argparse required-flag)."""
        # Ensure env var is unset so the default falls back to '' (which argparse
        # accepts as a value, but required=True without --systemname on argv fails).
        monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
        full = ['mlpstorage'] + argv_tail  # no --systemname
        with patch('sys.argv', full):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_lockfile_does_not_require_systemname(self, monkeypatch):
        """Pure utility subcommands (lockfile) parse without --systemname."""
        monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
        with patch('sys.argv', ['mlpstorage', 'lockfile', 'generate',
                                '-o', '/tmp/x', '--results-dir', '/r']):
            args = parse_arguments()
        assert args.mode == 'lockfile'

    def test_init_does_not_require_systemname(self, monkeypatch):
        """`mlpstorage init` parses without --systemname (init is bootstrap, not emitting)."""
        monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
        with patch('sys.argv', ['mlpstorage', 'init', 'Acme', '/tmp/r']):
            args = parse_arguments()
        assert args.mode == 'init'

    def test_systemname_env_var_default(self, monkeypatch):
        """When MLPERF_SYSTEMNAME is set, --systemname is satisfied via the env-var default."""
        monkeypatch.setenv('MLPERF_SYSTEMNAME', 'env-sys')
        # Reload config + common_args so the new env value flows into argparse defaults.
        import importlib
        import mlpstorage_py.config as cfg_mod
        import mlpstorage_py.cli.common_args as common_args_mod
        importlib.reload(cfg_mod)
        importlib.reload(common_args_mod)
        # Reload modules that close over add_universal_arguments via from-import.
        for mod_name in (
            'mlpstorage_py.cli.training_args',
            'mlpstorage_py.cli.checkpointing_args',
            'mlpstorage_py.cli.vectordb_args',
            'mlpstorage_py.cli.kvcache_args',
            'mlpstorage_py.cli.utility_args',
            'mlpstorage_py.cli_parser',
        ):
            try:
                mod = importlib.import_module(mod_name)
                importlib.reload(mod)
            except ImportError:
                pass
        from mlpstorage_py.cli_parser import parse_arguments as parse_args_reloaded
        try:
            with patch('sys.argv', ['mlpstorage', 'reports', 'reportgen',
                                    '--results-dir', '/r']):
                args = parse_args_reloaded()
            assert args.systemname == 'env-sys'
        finally:
            monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
            importlib.reload(cfg_mod)
            importlib.reload(common_args_mod)
            for mod_name in (
                'mlpstorage_py.cli.training_args',
                'mlpstorage_py.cli.checkpointing_args',
                'mlpstorage_py.cli.vectordb_args',
                'mlpstorage_py.cli.kvcache_args',
                'mlpstorage_py.cli.utility_args',
                'mlpstorage_py.cli_parser',
            ):
                try:
                    importlib.reload(importlib.import_module(mod_name))
                except ImportError:
                    pass
