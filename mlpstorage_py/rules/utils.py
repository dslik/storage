"""
Utility functions for rules validation.

This module contains helper functions used by rules checkers and other
components for calculating requirements and generating output paths.
"""

import os
import sys
from typing import Tuple, List, Optional

from mlpstorage_py.config import BENCHMARK_TYPES, DATETIME_STR
from mlpstorage_py.errors import ConfigurationError, ErrorCode


def calculate_training_data_size(args, cluster_information, dataset_params, reader_params, logger,
                                 num_processes=None) -> Tuple[int, int, int]:
    """
    Calculate the required training data size for closed submission.

    Requirements:
      - Dataset needs to be 5x the amount of total memory
      - Training needs to do at least 500 steps per epoch

    Memory Ratio:
      - Collect "Total Memory" from /proc/meminfo on each host
      - Sum it up
      - Multiply by 5
      - Divide by sample size
      - Divide by batch size

    500 steps:
      - 500 steps per epoch
      - Multiply by max number of processes
      - Multiply by batch size

    Args:
        args: Command-line arguments (optional, can be None).
        cluster_information: ClusterInformation instance with system info.
        dataset_params: Dataset parameters from benchmark config.
        reader_params: Reader parameters from benchmark config.
        logger: Logger instance.
        num_processes: Number of processes (optional).

    Returns:
        Tuple of (required_file_count, required_subfolders_count, total_disk_bytes)
    """
    required_file_count = 1
    required_subfolders_count = 0

    # Find the amount of memory in the cluster via args or measurements
    if not args:
        total_mem_bytes = cluster_information.total_memory_bytes
    elif hasattr(args, 'client_host_memory_in_gb') and args.client_host_memory_in_gb and \
         hasattr(args, 'num_client_hosts') and args.num_client_hosts:
        per_host_memory_in_bytes = args.client_host_memory_in_gb * 1024 * 1024 * 1024
        num_hosts = args.num_client_hosts
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
        num_processes = args.num_processes
    elif hasattr(args, 'clienthost_host_memory_in_gb') and args.clienthost_host_memory_in_gb and \
         not (hasattr(args, 'num_client_hosts') and args.num_client_hosts):
        per_host_memory_in_bytes = args.clienthost_host_memory_in_gb * 1024 * 1024 * 1024
        num_hosts = len(args.hosts)
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
        num_processes = args.num_processes
    else:
        raise ValueError('Either args or cluster_information is required')

    # Required Minimum Dataset size is 5x the total client memory
    dataset_size_bytes = 5 * total_mem_bytes

    # Calculate record length
    if 'record_length_bytes' in dataset_params:
        record_length_bytes = dataset_params['record_length_bytes']
    elif dataset_params.get('format') == 'parquet' and 'parquet' in dataset_params:
        # Calculate record length from parquet columns
        record_length_bytes = 0
        columns = dataset_params['parquet'].get('columns', [])
        for col in columns:
            dtype = col.get('dtype', 'float32')
            size = int(col.get('size', 1))
            
            if dtype == 'float64' or dtype == 'int64':
                record_length_bytes += size * 8
            elif dtype == 'uint8' or dtype == 'bool':
                record_length_bytes += size * 1
            else:
                # Default to float32/int32 (4 bytes)
                record_length_bytes += size * 4
    else:
        record_length_bytes = 0
        logger.warning("Could not determine record_length_bytes. Defaulting to 0.")

    file_size_bytes = dataset_params['num_samples_per_file'] * record_length_bytes

    if file_size_bytes > 0:
        min_num_files_by_bytes = dataset_size_bytes // file_size_bytes
    else:
        min_num_files_by_bytes = 0
    num_samples_by_bytes = min_num_files_by_bytes * dataset_params['num_samples_per_file']
    min_samples = 500 * num_processes * reader_params['batch_size']
    min_num_files_by_samples = min_samples // dataset_params['num_samples_per_file']

    required_file_count = max(min_num_files_by_bytes, min_num_files_by_samples)
    total_disk_bytes = required_file_count * file_size_bytes

    logger.ridiculous(f'Required file count: {required_file_count}')
    logger.ridiculous(f'Required sample count: {min_samples}')
    logger.ridiculous(f'Min number of files by samples: {min_num_files_by_samples}')
    logger.ridiculous(f'Min number of files by size: {min_num_files_by_bytes}')
    logger.ridiculous(f'Required dataset size: {required_file_count * file_size_bytes / 1024 / 1024}MiB')
    logger.ridiculous(f'Number of Samples by size: {num_samples_by_bytes}')

    if min_num_files_by_bytes > min_num_files_by_samples:
        logger.result(f'Minimum file count dictated by dataset size to memory size ratio.')
    else:
        logger.result(f'Minimum file count dictated by 500 step requirement of given accelerator count and batch size.')

    return int(required_file_count), int(required_subfolders_count), int(total_disk_bytes)


def generate_output_location(benchmark, datetime_str=None, **kwargs) -> str:
    """
    Generate the canonical Rules.md §2.1-shaped output path for benchmark results.

    Canonical shape (LAY-05, Phase 1 Plan 01-03):

        <results-dir>/<mode>/<orgname>/results/<systemname>/<benchmark>/<model>/<command>/<datetime>/

    Checkpointing intentionally omits the <command> segment to preserve the
    pre-refactor layout of checkpointing runs:

        <results-dir>/<mode>/<orgname>/results/<systemname>/checkpointing/<model>/<datetime>/

    This function is PURE with respect to args.{mode, orgname, systemname} —
    it does NOT resolve orgname from the sentinel or read MLPERF_SYSTEMNAME
    here. Per RESEARCH.md Pitfall 1, orgname resolution lives upstream in
    main._main_impl()'s sentinel-resolution gate (Slice 4); the universal
    --systemname plumbing (Slice 3 / this plan) populates args.systemname.

    Args:
        benchmark: Benchmark instance. Expected attributes:
            - benchmark.BENCHMARK_TYPE — one of BENCHMARK_TYPES enum values.
            - benchmark.args.results_dir, args.mode, args.orgname, args.systemname.
            - benchmark.args.{model | vdb_engine}, args.command (per BENCHMARK_TYPE).
        datetime_str: Optional datetime string for the run; defaults to
            mlpstorage_py.config.DATETIME_STR.
        **kwargs: Reserved for forward compatibility; currently unused.

    Returns:
        Full path to the output location, no trailing slash.

    Raises:
        ConfigurationError: If args.systemname is empty (T-1-02 mitigation —
            empty post-resolution systemname would silently produce
            "<rd>/closed/Acme/results//training/..." which subsequent
            os.makedirs collapses to a different shape that breaks
            submission-checker layout invariants). Same for empty orgname
            (Pitfall 1 defense-in-depth: orgname must be resolved upstream).
        ValueError: If a per-benchmark-type required field is missing:
            - training/checkpointing: args.model.
            - vector_database: args.vdb_engine.
            - kv_cache: args.model.
    """
    if datetime_str is None:
        datetime_str = DATETIME_STR

    args = benchmark.args

    # Defense-in-depth empty-string guards (T-1-02 + Pitfall 1).
    # Use getattr per Pitfall 2: args may not have the attribute if
    # _apply_yaml_config_overrides() dropped it via key-not-in-dict skip.
    orgname = getattr(args, 'orgname', '')
    systemname = getattr(args, 'systemname', '')
    if not orgname:
        raise ConfigurationError(
            "Cannot generate output location: orgname is empty "
            "(sentinel not resolved).",
            suggestion=(
                "Internal error: the upstream orgname-resolution gate in "
                "main._main_impl() must populate args.orgname before "
                "benchmark instantiation. If you reached this from a non-init "
                "command, run `mlpstorage init <orgname> <results-dir>` first."
            ),
            code=ErrorCode.CONFIG_MISSING_REQUIRED,
        )
    if not systemname:
        raise ConfigurationError(
            "Cannot generate output location: --systemname is empty.",
            suggestion=(
                "Pass --systemname <name> on the CLI or set the "
                "MLPERF_SYSTEMNAME environment variable before re-running."
            ),
            code=ErrorCode.CONFIG_MISSING_REQUIRED,
        )

    # Shared Rules.md §2.1 prefix for every benchmark type.
    base = os.path.join(
        args.results_dir,
        args.mode,                 # closed | open | whatif
        orgname,
        "results",
        systemname,
    )

    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.training:
        if not hasattr(args, "model"):
            raise ValueError("Model name is required for training benchmark output location")
        return os.path.join(
            base,
            benchmark.BENCHMARK_TYPE.name,
            args.model,
            args.command,
            datetime_str,
        )

    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database:
        engine = getattr(args, "vdb_engine", None)
        if not engine:
            raise ValueError(
                "VectorDB engine is required for output location "
                "(set --vdb-engine on the CLI)."
            )
        return os.path.join(
            base,
            benchmark.BENCHMARK_TYPE.name,
            engine,
            args.command,
            datetime_str,
        )

    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.kv_cache:
        model = getattr(args, "model", None)
        if not model:
            raise ValueError(
                "Model is required for kv_cache output location: set "
                "args.model before calling generate_output_location "
                "(KVCacheBenchmark.__init__ defaults this from KVCACHE_MODEL_DEFAULT)."
            )
        return os.path.join(
            base,
            benchmark.BENCHMARK_TYPE.name,
            model,
            args.command,
            datetime_str,
        )

    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.checkpointing:
        if not hasattr(args, "model"):
            raise ValueError("Model name is required for checkpointing benchmark output location")
        # Checkpointing intentionally omits the <command> segment; preserves
        # the pre-refactor layout shape that downstream submission-checkers
        # already validate against.
        return os.path.join(
            base,
            benchmark.BENCHMARK_TYPE.name,
            args.model,
            datetime_str,
        )

    print('The given benchmark is not supported by mlpstorage_py.rules.generate_output_location()')
    sys.exit(1)


def get_runs_files(results_dir: str, logger=None) -> List:
    """
    Find all benchmark run directories in a results directory.

    Args:
        results_dir: Path to the results directory.
        logger: Optional logger instance.

    Returns:
        List of BenchmarkRun instances.
    """
    from mlpstorage_py.rules.models import BenchmarkRun

    runs = []

    if not os.path.exists(results_dir):
        if logger:
            logger.warning(f"Results directory not found: {results_dir}")
        return runs

    # Walk the directory tree looking for run directories. followlinks=True
    # lets users symlink previously-completed run directories into a fresh
    # results-dir to accumulate them — a common workflow when stitching
    # together results from multiple machines or earlier runs.
    for root, dirs, files in os.walk(results_dir, followlinks=True):
        # Check if this directory contains a summary.json (DLIO run) or metadata file
        has_summary = 'summary.json' in files
        metadata_files = [f for f in files if f.endswith('_metadata.json')]
        has_metadata = len(metadata_files) == 1

        if len(metadata_files) > 1:
            if logger:
                logger.warning(f"Skipping {root}: multiple metadata files found ({len(metadata_files)})")
            continue

        if has_summary or has_metadata:
            try:
                run = BenchmarkRun.from_result_dir(root, logger)
                runs.append(run)
                if logger:
                    logger.debug(f"Found run: {run.run_id}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load run from {root}: {e}")

    return runs
