
from .base import BaseCheck
from ..constants import *
from ..configuration.configuration import Config
from ..loader import SubmissionLogs
from ..rule_registry import rule
from ..utils import *

import os
import re
from datetime import datetime


class DirectoryCheck(BaseCheck):
    """
    A check class for validating directory structure and related properties.
    Inherits from BaseCheck and receives a config and loader instance.
    """

    def __init__(self, log, config: Config, submissions_logs: SubmissionLogs):
        """
        Initialize DirectoryChecks with configuration and loader.

        Args:
            config: A Config instance containing submission configuration.
            loader: A SubmissionLogs instance for accessing submission logs.
        """
        # Call parent constructor with the loader's log and submission path
        super().__init__(log=log, path=submissions_logs.loader_metadata.folder)
        self.config = config
        self.submissions_logs = submissions_logs
        self.name = "directory checks"
        self.datagen_path = os.path.join(self.path, "datagen")
        self.run_path = os.path.join(self.path, "run")
        self.checkpointing_path = self.path
        self.init_checks()

    def init_checks(self):
        """Register §2 directory checks for the current submission's mode.

        Per CR-01 (review 2026-06-10): the previous binary if/else routed any
        non-training mode into the checkpointing branch, which would emit
        false 2.1.22..2.1.26 violations against vdb/kvcache submission trees.
        DirectoryCheck owns §2 rules for training and checkpointing only;
        vdb/kvcache directory rules (if added) belong to their own Check
        classes. Unknown modes register no checks and log at debug level.
        """
        self.checks = []
        mode = getattr(self.submissions_logs.loader_metadata, 'mode', 'training')
        if mode == "training":
            # Training mode checks
            self.checks.extend([
                self.datagen_files_check,
                self.datagen_dlio_config_check,
                self.run_results_json_check,
                self.run_files_check,
                self.run_files_timestamp_check,
                self.run_dlio_config_check,
                self.run_duration_valid_check,
            ])
        elif mode == "checkpointing":
            # Checkpointing mode checks
            self.checks.extend([
                self.checkpointing_results_json_check,
                self.checkpointing_timestamps_check,
                self.checkpointing_timestamp_gap_check,
                self.checkpointing_files_check,
                self.checkpointing_dlio_config_check,
            ])
        else:
            # vdb / kvcache / unknown — DirectoryCheck has no §2 rules for
            # these modes yet; emit nothing and let the per-mode Check class
            # own its directory rules when they land.
            self.log.debug(
                "DirectoryCheck: no §2 checks registered for mode=%r", mode
            )
    
    
    @rule("2.1.14", "datagenFiles")
    def datagen_files_check(self):
        """
        Check that each datagen timestamp directory contains:
        - training_datagen.stdout.log
        - training_datagen.stderr.log
        - *output.json
        - *per_epoch_stats.json
        - *summary.json
        - dlio.log
        - dlio_config/ (subdirectory)

        (Rules.md 2.1.14 datagenFiles)
        """
        valid = True
        for _, _, timestamp in self.submissions_logs.datagen_files:
            timestamp_path = os.path.join(self.datagen_path, timestamp)
            files = list_files(timestamp_path)
            for required_file in self.config.get_datagen_required_files():
                if self.config.skip_output_file and required_file == "*output.json":
                    continue
                if not regex_matches_any(required_file, files):
                    self.log_violation(
                        "2.1.14", "datagenFiles", timestamp_path,
                        "%s not found", required_file,
                    )
                    valid = False

            # Check for dlio_config directory
            for required_folder in self.config.get_datagen_required_folders():
                if required_folder not in list_dir(timestamp_path):
                    self.log_violation(
                        "2.1.14", "datagenFiles", timestamp_path,
                        "%s directory not found", required_folder,
                    )
                    valid = False

        return valid
    
    @rule("2.1.15", "datagenDlioConfig")
    def datagen_dlio_config_check(self):
        """
        Check that the dlio_config subdirectory in each datagen timestamp directory
        contains exactly: config.yaml, hydra.yaml, and overrides.yaml (case-sensitive).

        (Rules.md 2.1.15 datagenDlioConfig)
        """
        valid = True
        required_files = {"config.yaml", "hydra.yaml", "overrides.yaml"}

        for _, _, timestamp in self.submissions_logs.datagen_files:
            dlio_config_path = os.path.join(self.datagen_path, timestamp, "dlio_config")

            if not os.path.exists(dlio_config_path):
                self.log_violation(
                    "2.1.15", "datagenDlioConfig", dlio_config_path,
                    "dlio_config directory not found",
                )
                valid = False
                continue

            files = set(list_files(dlio_config_path))

            # Check for exact match
            if files != required_files:
                self.log_violation(
                    "2.1.15", "datagenDlioConfig", dlio_config_path,
                    "dlio_config has incorrect files. Expected %s, got %s",
                    required_files,
                    files,
                )
                valid = False

        return valid

    @rule("2.1.16", "runResultsJson")
    def run_results_json_check(self):
        """
        Check that there is exactly one results.json file in the run phase directory.

        (Rules.md 2.1.16 runResultsJson)
        """
        valid = True
        results_files = list_files(self.run_path)
        results_json_count = sum(1 for f in results_files if f == "results.json")

        if results_json_count != 1:
            self.log_violation(
                "2.1.16", "runResultsJson", self.run_path,
                "Expected exactly 1 results.json file, found %d",
                results_json_count,
            )
            valid = False

        return valid
    
    @rule("2.1.19", "runFiles")
    def run_files_check(self):
        """
        Check that each run timestamp directory contains:
        - training_run.stdout.log
        - training_run.stderr.log
        - *output.json
        - *per_epoch_stats.json
        - *summary.json
        - dlio.log
        - dlio_config/ (subdirectory)

        (Rules.md 2.1.19 runFiles)
        """
        valid = True
        for _, _, timestamp in self.submissions_logs.run_files:
            timestamp_path = os.path.join(self.run_path, timestamp)
            files = list_files(timestamp_path)
            for required_file in self.config.get_run_required_files():
                if self.config.skip_output_file and required_file == "*output.json":
                    continue
                if not regex_matches_any(required_file, files):
                    self.log_violation(
                        "2.1.19", "runFiles", timestamp_path,
                        "%s not found", required_file,
                    )
                    valid = False

            # Check for dlio_config directory
            for required_folder in self.config.get_run_required_folders():
                if required_folder not in list_dir(timestamp_path):
                    self.log_violation(
                        "2.1.19", "runFiles", timestamp_path,
                        "%s directory not found", required_folder,
                    )
                    valid = False

        return valid
    
    @rule("2.1.17", "runTimestamps")
    def run_files_timestamp_check(self):
        """
        Check that all run_files have timestamps matching format "YYYYMMDD_HHmmss"
        and that there are exactly RUN_TIMESTAMP_COUNT of them.

        Per Rules.md 2.1.17 (runTimestamps): exactly 6 timestamp directories are
        required — 1 warm-up run plus 5 measured runs.
        """
        valid = True
        timestamp_pattern = r"^\d{8}_\d{6}$"
        timestamps = []

        for _, _, timestamp in self.submissions_logs.run_files:
            timestamps.append(timestamp)
            if not re.match(timestamp_pattern, timestamp):
                self.log_violation(
                    "2.1.17", "runTimestamps", self.run_path,
                    "Invalid timestamp format '%s'. Expected format: YYYYMMDD_HHmmss",
                    timestamp,
                )
                valid = False

        if len(timestamps) != RUN_TIMESTAMP_COUNT:
            self.log_violation(
                "2.1.17", "runTimestamps", self.run_path,
                "Expected %d run files, but found %d. Timestamps: %s",
                RUN_TIMESTAMP_COUNT,
                len(timestamps),
                timestamps,
            )
            valid = False

        return valid
    
    @rule("2.1.20", "runDlioConfig")
    def run_dlio_config_check(self):
        """
        Check that the dlio_config subdirectory in each run timestamp directory
        contains exactly: config.yaml, hydra.yaml, and overrides.yaml (case-sensitive).

        (Rules.md 2.1.20 runDlioConfig)
        """
        valid = True
        required_files = {"config.yaml", "hydra.yaml", "overrides.yaml"}

        for _, _, timestamp in self.submissions_logs.run_files:
            dlio_config_path = os.path.join(self.run_path, timestamp, "dlio_config")

            if not os.path.exists(dlio_config_path):
                self.log_violation(
                    "2.1.20", "runDlioConfig", dlio_config_path,
                    "dlio_config directory not found",
                )
                valid = False
                continue

            files = set(list_files(dlio_config_path))

            # Check for exact match
            if files != required_files:
                self.log_violation(
                    "2.1.20", "runDlioConfig", dlio_config_path,
                    "dlio_config has incorrect files. Expected %s, got %s",
                    required_files,
                    files,
                )
                valid = False

        return valid
    
    @rule("2.1.18", "runTimestampGap")
    def run_duration_valid_check(self):
        """
        Check that the gap between consecutive timestamp directories is less than
        the duration of a single run. The gap must be short enough to ensure there
        was no benchmark activity between consecutive runs.

        Compares the time delta between consecutive run directory names with the
        duration of each individual run (from start to end time).

        (Rules.md 2.1.18 runTimestampGap)
        """
        valid = True

        # Parse all run data: (run_dict, _, timestamp_dir_name)
        run_dir_time = []
        max_gap = float("inf")
        time_factor = 2
        for run_dict, _, timestamp_dir in self.submissions_logs.run_files:
            try:
                # Parse timestamps from run_dict
                start_time = datetime.fromisoformat(run_dict["start"])
                end_time = datetime.fromisoformat(run_dict["end"])

                # Parse the directory timestamp (YYYYMMDD_HHmmss format)
                dir_time = datetime.strptime(timestamp_dir, "%Y%m%d_%H%M%S")

                run_duration = (end_time - start_time).total_seconds() * time_factor
                if run_duration < max_gap:
                    max_gap = run_duration

                run_dir_time.append(dir_time)
            except (ValueError, KeyError, TypeError) as e:
                self.log_violation(
                    "2.1.18", "runTimestampGap", timestamp_dir,
                    "Failed to parse timestamp data: %s",
                    str(e),
                )
                valid = False
                continue

        # Check gaps between consecutive runs
        for i in range(len(run_dir_time) - 1):
            current_run = run_dir_time[i]
            next_run = run_dir_time[i + 1]

            # Calculate gap between end of current run and start of next run
            gap = (next_run - current_run).total_seconds()

            # Gap should be less than the max gap
            if gap >= max_gap:
                self.log_violation(
                    "2.1.18", "runTimestampGap", self.run_path,
                    "Gap between runs is %s, which is >= the run duration %s. "
                    "Benchmark activity between runs can't be discarted.",
                    gap,
                    max_gap,
                )
                valid = False

        return valid
    
    @rule("2.1.22", "checkpointingResultsJson")
    def checkpointing_results_json_check(self):
        """
        Check that there is exactly one results.json file in each workload directory
        within the checkpointing directory hierarchy.

        (Rules.md 2.1.22 checkpointingResultsJson)
        """
        valid = True

        if not hasattr(self.submissions_logs, 'checkpoint_files') or not self.submissions_logs.checkpoint_files:
            self.log.warning("No checkpointing files found in submission logs")
            return valid

        # Get workload directories
        workload_dirs = list_dir(self.checkpointing_path)

        for workload_dir in workload_dirs:
            workload_path = os.path.join(self.checkpointing_path, workload_dir)
            results_files = list_files(workload_path)
            results_json_count = sum(1 for f in results_files if f == "results.json")

            if results_json_count != 1:
                self.log_violation(
                    "2.1.22", "checkpointingResultsJson", workload_path,
                    "Expected exactly 1 results.json, found %d",
                    results_json_count,
                )
                valid = False

        return valid
    
    @rule("2.1.23", "checkpointingTimestamps")
    def checkpointing_timestamps_check(self):
        """
        Check that there are exactly 10 timestamp directories in YYYYMMDD_HHmmss format
        within the workload directories in the checkpointing hierarchy.

        (Rules.md 2.1.23 checkpointingTimestamps)
        """
        valid = True
        timestamp_pattern = r"^\d{8}_\d{6}$"

        if not hasattr(self.submissions_logs, 'checkpoint_files') or not self.submissions_logs.checkpoint_files:
            self.log.warning("No checkpointing files found in submission logs")
            return valid

        # Get workload directories
        workload_dirs = list_dir(self.checkpointing_path)

        for workload_dir in workload_dirs:
            workload_path = os.path.join(self.checkpointing_path, workload_dir)
            timestamp_dirs = list_dir(workload_path)

            # Validate format of each timestamp directory
            for timestamp_dir in timestamp_dirs:
                if not re.match(timestamp_pattern, timestamp_dir):
                    self.log_violation(
                        "2.1.23", "checkpointingTimestamps", workload_path,
                        "Invalid timestamp format '%s'. Expected format: YYYYMMDD_HHmmss",
                        timestamp_dir,
                    )
                    valid = False

            # Check count
            if len(timestamp_dirs) != 10:
                self.log_violation(
                    "2.1.23", "checkpointingTimestamps", workload_path,
                    "Expected 10 timestamp directories, found %d",
                    len(timestamp_dirs),
                )
                valid = False

        return valid
    
    @rule("2.1.24", "checkpointingTimestampGap")
    def checkpointing_timestamp_gap_check(self):
        """
        Check that the gap between consecutive timestamp directories is less than
        the duration of a single checkpoint run.

        (Rules.md 2.1.24 checkpointingTimestampGap)
        """
        valid = True

        if not hasattr(self.submissions_logs, 'checkpoint_files') or not self.submissions_logs.checkpoint_files:
            self.log.warning("No checkpointing files found in submission logs")
            return valid

        # Parse all checkpoint run data
        checkpoint_run_data = []
        max_gap = float("inf")

        for checkpoint_dict, _, timestamp_dir in self.submissions_logs.checkpoint_files:
            try:
                # Parse timestamps from checkpoint_dict
                start_time = datetime.fromisoformat(checkpoint_dict["start"])
                end_time = datetime.fromisoformat(checkpoint_dict["end"])

                # Parse the directory timestamp (YYYYMMDD_HHmmss format)
                dir_time = datetime.strptime(timestamp_dir, "%Y%m%d_%H%M%S")

                run_duration = end_time - start_time
                if run_duration < max_gap:
                    max_gap = run_duration

                checkpoint_run_data.append(dir_time)
            except (ValueError, KeyError, TypeError) as e:
                self.log_violation(
                    "2.1.24", "checkpointingTimestampGap", timestamp_dir,
                    "Failed to parse timestamp data for checkpointing: %s",
                    str(e),
                )
                valid = False
                continue

        # Sort timestamps to check gaps
        checkpoint_run_data.sort()

        # Check gaps between consecutive checkpoints
        for i in range(len(checkpoint_run_data) - 1):
            gap = checkpoint_run_data[i + 1] - checkpoint_run_data[i]

            if gap >= max_gap:
                self.log_violation(
                    "2.1.24", "checkpointingTimestampGap", self.checkpointing_path,
                    "Gap between checkpoints is %s, which is >= the checkpoint duration %s. "
                    "Benchmark activity between checkpoints can't be discarded.",
                    gap,
                    max_gap,
                )
                valid = False

        return valid
    
    @rule("2.1.25", "checkpointingFiles")
    def checkpointing_files_check(self):
        """
        Check that each checkpointing timestamp directory contains:
        - checkpointing_run.stdout.log
        - checkpointing_run.stderr.log
        - *output.json
        - *per_epoch_stats.json
        - *summary.json
        - dlio.log
        - dlio_config/ (subdirectory)

        (Rules.md 2.1.25 checkpointingFiles)
        """
        valid = True

        if not hasattr(self.submissions_logs, 'checkpoint_files') or not self.submissions_logs.checkpoint_files:
            self.log.warning("No checkpointing files found in submission logs")
            return valid

        for _, _, timestamp in self.submissions_logs.checkpoint_files:
            timestamp_path = os.path.join(self.checkpointing_path, timestamp)
            files = list_files(timestamp_path)
            dirs = list_dir(timestamp_path)

            for required_file in self.config.get_checkpoint_required_files():
                if not regex_matches_any(required_file, files):
                    self.log_violation(
                        "2.1.25", "checkpointingFiles", timestamp_path,
                        "%s not found", required_file,
                    )
                    valid = False

            # Check for dlio_config directory
            for required_folder in self.config.get_checkpoint_required_folders():
                if required_folder not in dirs:
                    self.log_violation(
                        "2.1.25", "checkpointingFiles", timestamp_path,
                        "%s directory not found", required_folder,
                    )
                    valid = False

        return valid
    
    @rule("2.1.26", "checkpointingDlioConfig")
    def checkpointing_dlio_config_check(self):
        """
        Check that the dlio_config subdirectory in each checkpointing timestamp directory
        contains exactly: config.yaml, hydra.yaml, and overrides.yaml (case-sensitive).

        (Rules.md 2.1.26 checkpointingDlioConfig)
        """
        valid = True
        required_files = {"config.yaml", "hydra.yaml", "overrides.yaml"}

        if not hasattr(self.submissions_logs, 'checkpoint_files') or not self.submissions_logs.checkpoint_files:
            self.log.warning("No checkpointing files found in submission logs")
            return valid

        for _, _, timestamp in self.submissions_logs.checkpoint_files:
            dlio_config_path = os.path.join(self.checkpointing_path, timestamp, "dlio_config")

            if not os.path.exists(dlio_config_path):
                self.log_violation(
                    "2.1.26", "checkpointingDlioConfig", dlio_config_path,
                    "dlio_config directory not found",
                )
                valid = False
                continue

            files = set(list_files(dlio_config_path))

            # Check for exact match
            if files != required_files:
                self.log_violation(
                    "2.1.26", "checkpointingDlioConfig", dlio_config_path,
                    "dlio_config has incorrect files. Expected %s, got %s",
                    required_files,
                    files,
                )
                valid = False

        return valid

    @rule("2.1.27", "directoryDiagram")
    def directory_diagram_check(self):
        """No-op binding for Rules.md 2.1.27 (directoryDiagram).

        Rules.md 2.1.27 is a pictorial illustration of the submission directory
        layout (see Rules.md line 117: "Pictorially, here is what this looks
        like:"). There is no programmatic check that maps to the diagram
        itself — the rules the diagram depicts are enforced by the structural
        and file-content checks elsewhere:

        - 2.1.1..2.1.13 (top-level / submitter / code / systems / results
          hierarchy) are covered by SubmissionStructureCheck (Phase 1).
        - 2.1.14..2.1.26 (per-workload datagen/run/checkpointing layout)
          are covered by the other 12 methods on this class.

        This method exists only so discover_rules(DirectoryCheck) can report
        2.1.27 as bound (Phase 3 D-A1 aggressive-retrofit choice — prefer an
        @rule binding over an OUT_OF_SCOPE_RULES entry). It is NOT registered
        in init_checks and does NOT contribute to the per-submission pass/fail
        accumulator.

        Returns:
            True — emits no logging and never participates in run_checks().
        """
        return True
