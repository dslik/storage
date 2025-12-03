# MLPerf™ Storage V2.0 Benchmark Validation
——————————————————————————————————————————

- [MLPerf Storage Benchmark Submission Guidelines v2.0](#mlperf-storage-benchmark-submission-guidelines-v20)
  - [1. Introduction](#1-introduction)
  - [2. Directory Structure for All Submissions](#2-directory-structure-for-all-submissions)
  - [3. Sanity Checking the Training Options](#3-sanity-checking-the-training-options)
    - [3.1. CLOSED Versus OPEN Options](#31-closed-versus-open-options)
    - [3.2. Benchmark Dataset Generation Options](#32-benchmark-dataset-generation-options)
    - [3.3. Benchmark Run Options](#33-benchmark-run-options)
  - [4. Sanity Checking the Checkpointing Options](#3-sanity-checking-the-checkpointing-options)
    - [4.1. CLOSED Versus OPEN Options](#41-closed-versus-open-options)
    - [4.2. Benchmark Run Options](#42-benchmark-run-options)

# 1. Introduction

These are the requirements for the *submission validation checker* for version 2.0 of the MLPerf™ Storage benchmark,
but since the `mlpstorage` tool will be responsible for generating the vast majority (if not all) of the contents of a submission, it is also a spec for what `mlpstorage` should generate.

The *submission validation checker* should check that the tested directory hierarachy matches the below requirements and output messages for all cases where it does not match.
The tool should make it's best effort to continue testing all the other aspects of the directory hierarchy after any given failure.
If the tested directory hierarchy does not meet all of the below requirements, then it should be labelled as invalid and the validation check should fail.

Even if the structure of a submission package matches the spec, the options that were used to run the benchmark may not fall within acceptable bounds,
so we need the *submission validation checker* to check for illegal/inapproriate option settings,
and for semantic mismatches between different options that were used.

# 2. Directory Structure for All Submissions

**2.1.**  The submission structure must start from a single directory whose name is the name of the submitter.  This can be any string, possibly including blanks.

**2.2.**  Within the top-level directory of the submission structure there must be a directory named "closed" and/or one named "open", and nothing more.  These names are case-sensitive.

**2.3.**  The "open" directory hierarchy should be constructed identically to the "closed" directory hierarchy describe just below.

**2.4.**  Within the "closed" directory there must be a single directory whose name is the name of the submitter (the same as the top-level directory).

**2.5.**  Within the submitter directory mentioned just above, there must be exactly three directories: "code", "results", and "systems".  These names are case-sensitive.

**2.6.**  The "code" directory must include a complete copy of the MLPerf Storage github repo that was used to run the test that resulted in the "results" directory's contents.
If this is in the "open" hierarchy, any modifications made to the benchmark code must be included here, and if this is in the "closed" hierarchy, there must be no changes to the benchmark code.
Note that in both cases this must be the code that was actually run to generate those results.

**2.7.**  The "systems" directory must contain two files for each "system name", a .yaml file and a .pdf file, and nothing more.  Each of those files must be named with the "system name".
Eg: for a system-under-test named "Big_and_Fast_4000_buffered", there must be a "Big_and_Fast_4000_buffered.yaml" and a "Big_and_Fast_4000_buffered.pdf" file.  These names are case-sensitive.

**2.8.**  The "results" directory, whether it is within the "closed' or "open" hierarchies, must include one or more directories that are the names of the systems-under-test.  Eg: a system name could be "Big_and_Fast_4000_buffered".
This name can be anything the submitter wants, it is just a name to both idenfity the set of results that were collected from a given	
configuration of storage system and to link together those results with the .pdf and .yaml files that describe the system-under-test.

**2.9.**  All the configuration parameters and hardware and software components of the system-under-test that are part of a given *system name* must be identical.  Any changes to those configuration parameters or hardware or software must be submitted as a separate *system name*, so we should compare the configuration parameters and hardware and software components to verify that they're the same across all the tests and runs within the given *system name* directory hierarchy, to the extent that we can.  The *system names*  are case-sensitive.

**2.10.**  Within a *system name* directory in the "results" directory, there must be one or both of the following directories, and nothing else: "training", and/or "checkpointing".  These names are case-sensitive.

**2.11.**  Within the "training" directory, there must be one or more of the following *workload directories*, and nothing else: "unet3d", "resnet50" and/or "cosmoflow".  These names are case-sensitive.

**2.12.**  Within the *workload directories* in the "training" hierarchy, there must exist *phase directories* named "datagen" and "run", and nothing else.  These names are case-sensitive.

**2.13.**  Within the "datagen" *phase directory* within the "training" directory hierarchy, there must be exactly one *timestamp directory* named *YYYYMMDD_HHmmss" that represent a *timestamp* of when that part of the test run was completed.  Where Y's are replaced with the year the run was performed, M's are replaced with the month, D's with the day, H's with the hour (in 24-hour format), m's with the minute, and s's with the second.  The timestamps should be relative to the local timezone where the test was actually run.

**2.14.**  Within the *timestamp directory* within the "datagen" *phase*, there must exist the following files: "training_datagen.stdout.log", "training_datagen.stderr.log" file, "*output.json, "*per_epoch_stats.json", "*summary.json", and "dlio.log", plus a subdirectory named "dlio_config".  These names are case-sensitive.

**2.15.**  The "dlio_config" subdirectory in each *timestamp directory*  must contain the following list of files, and nothing else: "config.yaml", "hydra.yaml", and "overrides.yaml".  These names are case-sensitive.

**2.16.**  Within the "run" *phase directory* within the "training" directory hierarchy, there must be one "results.json" file.  This name is case-sensitive.

**2.17.**  Within the "run" *phase directory* within the "training" directory hierarchy, there must also be exactly 5 subdirectories named *YYYYMMDD_HHmmss" that represent a *timestamp* of when that part of the test run was completed.  Where Y's are replaced with the year the run was performed, M's are replaced with the month, D's with the day, H's with the hour (in 24-hour format), m's with the minute, and s's with the second.  The timestamps should be relative to the local timezone where the test was actually run.

**2.18.**  Within each *timestamp directory* within the "run" *phase*, there must exist the following files: "training_run.stdout.log", "training_run.stderr.log" file, "*output.json, "*per_epoch_stats.json", "*summary.json", and "dlio.log", plus a subdirectory named "dlio_config".  These names are case-sensitive.

**2.19.**  The "dlio_config" subdirectory in each *timestamp directory* must contain the following list of files, and nothing else: "config.yaml", "hydra.yaml", and "overrides.yaml".  These names are case-sensitive.

**2.20.**  Within the "checkpointing" directory, there must be one or more of the following *workload directories*, and nothing else: "llama3-8b", "llama3-70b", "llama3-405b", and/or "llama3-1t".  These names are case-sensitive.

**2.21.**  Within the *workload directories* within the "checkpointing" directory hierarchy, there must be one "results.json" file.  This name is case-sensitive.

**2.22.**  Within the *workload directories* within the "checkpointing" directory hierarchy, there must also be exactly ten *timestamp directories* named *YYYYMMDD_HHmmss" that represent a *timestamp* of when that part of the test run was completed.  Where Y's are replaced with the year the run was performed, M's are replaced with the month, D's with the day, H's with the hour (in 24-hour format), m's with the minute, and s's with the second.  The timestamps should be relative to the local timezone where the test was actually run.

**2.23.**  Within the *timestamp directories* within the "checkpointing" directory hierarchy, there must exist the following files: "checkpointing_run.stdout.log", "checkpointing_run.stderr.log" file, "*output.json, "*per_epoch_stats.json", "*summary.json", and "dlio.log", plus a subdirectory named "dlio_config".  These names are case-sensitive.

**2.24.**  The "dlio_config" subdirectory in each *timestamp directory* must contain the following list of files, and nothing else: "config.yaml", "hydra.yaml", and "overrides.yaml".  These names are case-sensitive.

**2.25.**  Pictorially, here is what this looks like:
```
root_folder (or any name you prefer)
├── Closed
│ 	└──<submitter_org>
│	  	├── code
│	  	├── results
│	  	│	└──system-name-1
│	  	│	 	├── training
│	  	│	 	│	├── unet3d
│	  	│		│	│	├── datagen
│	  	│		│	│	│	└── YYYYMMDD_HHmmss
│	  	│		│	│	│		└── dlio_config
│	  	│		│	│	└── run
│	  	│		│	│		├──results.json
│	  	│		│	│		├── YYYYMMDD_HHmmss
│	  	│		│	│		│	└── dlio_config 
│	  	│		│	│		... (5x Runs per Emulated Accelerator Type)
│	  	│		│	│		└── YYYYMMDD_HHmmss
│	  	│		│	│			└── dlio_config
│	  	│	 	│	├── resnet50
│	  	│		│	│	├── datagen
│	  	│		│	│	│	└── YYYYMMDD_HHmmss
│	  	│		│	│	│		└── dlio_config
│	  	│		│	│	└── run
│	  	│		│	│		├──results.json
│	  	│		│	│		├── YYYYMMDD_HHmmss
│	  	│		│	│		│	└── dlio_config 
│	  	│		│	│		... (5x Runs per Emulated Accelerator Type)
│	  	│		│	│		└── YYYYMMDD_HHmmss
│	  	│		│	│			└── dlio_config
│	  	│	 	│	└── cosmoflow
│	  	│		│	 	├── datagen
│	  	│		│	 	│	└── YYYYMMDD_HHmmss
│	  	│		│	 	│		└── dlio_config
│	  	│		│	 	└── run
│	  	│		│			├──results.json
│	  	│		│	 		├── YYYYMMDD_HHmmss
│	  	│		│	 		│	└── dlio_config 
│	  	│		│	 		... (5x Runs per Emulated Accelerator Type)
│	  	│		│	 		└── YYYYMMDD_HHmmss
│	  	│		│	 			└── dlio_config
│	  	│	 	└── checkpointing
│	  	│	 		├── llama3-8b
│	  	│			│	├──results.json
│	  	│			│	├── YYYYMMDD_HHmmss
│	  	│			│	│	└── dlio_config 
│	  	│			 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│			│	└── YYYYMMDD_HHmmss
│	  	│			│		└── dlio_config
│	  	│	 		├── llama3-70b
│	  	│			│	├──results.json
│	  	│			│	├── YYYYMMDD_HHmmss
│	  	│			│	│	└── dlio_config 
│	  	│			 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│			│	└── YYYYMMDD_HHmmss
│	  	│			│		└── dlio_config
│	  	│	 		├── llama3-405b
│	  	│			│	├──results.json
│	  	│			│	├── YYYYMMDD_HHmmss
│	  	│			│	│	└── dlio_config 
│	  	│			 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│			│	└── YYYYMMDD_HHmmss
│	  	│			│		└── dlio_config
│	  	│	 		└── llama3-1t
│	  	│				├──results.json
│	  	│			 	├── YYYYMMDD_HHmmss
│	  	│			 	│	└── dlio_config 
│	  	│			 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│				└── YYYYMMDD_HHmmss
│	  	│			 		└── dlio_config
│	  	└── systems
│	  		├──system-name-1.yaml
│	  		├──system-name-1.pdf
│	  		├──system-name-2.yaml
│	  		└──system-name-2.pdf
│
└── Open
 	└──<submitter_org>
		├── code
		├── results
		│	└──system-name-1
		│	 	├── training
		│	 	│	├── unet3d
		│		│	│	├── datagen
		│		│	│	│	└── YYYYMMDD_HHmmss
		│		│	│	│		└── dlio_config
		│		│	│	└── run
		│		│	|		├──results.json
		│		│	│		├── YYYYMMDD_HHmmss
		│		│	│		│	└── dlio_config 
		│		│	│		... (5x Runs per Emulated Accelerator Type)
		│		│	│		└── YYYYMMDD_HHmmss
		│		│	│			└── dlio_config
		│	 	│	├── resnet50
		│		│	│	├── datagen
		│		│	│	│	└── YYYYMMDD_HHmmss
		│		│	│	│		└── dlio_config
		│		│	│	└── run
		│		│	|		├──results.json
		│		│	│		├── YYYYMMDD_HHmmss
		│		│	│		│	└── dlio_config 
		│		│	│		... (5x Runs per Emulated Accelerator Type)
		│		│	│		└── YYYYMMDD_HHmmss
		│		│	│			└── dlio_config
		│	 	│	└── cosmoflow
		│		│	 	├── datagen
		│		│	 	│	└── YYYYMMDD_HHmmss
		│		│	 	│		└── dlio_config
		│		│	 	└── run
		│		│			├──results.json
		│		│	 		├── YYYYMMDD_HHmmss
		│		│	 		│	└── dlio_config 
		│		│	 		... (5x Runs per Emulated Accelerator Type)
		│		│	 		└── YYYYMMDD_HHmmss
		│		│	 			└── dlio_config
		│	 	└── checkpointing
		│	 		├── llama3-8b
		│			|	├──results.json
		│			│	├── YYYYMMDD_HHmmss
		│			│	│	└── dlio_config 
		│			│	... (10x Runs for Read and Write. May be combined in a single run)
		│			│	└── YYYYMMDD_HHmmss
		│			│		└── dlio_config
		│	 		├── llama3-70b
		│			|	├──results.json
		│			│	├── YYYYMMDD_HHmmss
		│			│	│	└── dlio_config 
		│			│	... (10x Runs for Read and Write. May be combined in a single run)
		│			│	└── YYYYMMDD_HHmmss
		│			│		└── dlio_config
		│	 		├── llama3-405b
		│			|	├──results.json
		│			│	├── YYYYMMDD_HHmmss
		│			│	│	└── dlio_config 
		│			│	... (10x Runs for Read and Write. May be combined in a single run)
		│			│	└── YYYYMMDD_HHmmss
		│			│		└── dlio_config
		│	 		└── llama3-1t
		│				├──results.json
		│			 	├── YYYYMMDD_HHmmss
		│			 	│	└── dlio_config 
		│				... (10x Runs for Read and Write. May be combined in a single run)
		│				└── YYYYMMDD_HHmmss
		│			 		└── dlio_config
		└── systems
			├──system-name-1.yaml
			├──system-name-1.pdf
			├──system-name-2.yaml
			└──system-name-2.pdf
```
**2.26.**  Since the "dlio_log" subdirectory has a similar structure in all cases, it is describe pictorially just below:
```
└── YYYYMMDD_HHmmss
    ├── [training|checkpointing]_[datagen|run].stdout.log
    ├── [training|checkpointing]_[datagen|run].stderr.log
    ├── *[output|per_epoch_stats|summary].json
    ├── dlio.log
    └── dlio_config
        ├── config.yaml
        ├── hydra.yaml
        └── overrides.yaml
```

# 3. Sanity Checking the Training Options

dfg

## 3.1.  CLOSED Versus OPEN Options

dfg

## 3.2.  Dataset Generation Options

Minimum dataset size. The MLPerf Storage benchmark script must be used to run the benchmarks since it calculates the minimum dataset size for each benchmark. It does so using the provided number of simulated accelerators and the size of all of the host node’s memory in GB. The minimum dataset size computation is as follows:

Calculate required minimum samples given number of steps per epoch (NB: num_steps_per_epoch is a minimum of 500):
   min_samples_steps_per_epoch = num_steps_per_epoch * batch_size * num_accelerators_across_all_nodes
Calculate required minimum samples given host memory to eliminate client-side caching effects; (NB: HOST_MEMORY_MULTIPLIER = 5):
   min_samples_host_memory_across_all_nodes = number_of_hosts * memory_per_host_in_GB * HOST_MEMORY_MULTIPLIER * 1024 * 1024 * 1024 / record_length
Ensure we meet both constraints:
   min_samples = max(min_samples_steps_per_epoch, min_samples_host_memory_across_all_nodes)
Calculate minimum files to generate
   min_total_files= min_samples / num_samples_per_file
   min_files_size = min_samples * record_length / 1024 / 1024 / 1024
A minimum of min_total_files files are required which will consume min_files_size GB of storage.

## 3.3.  Benchmark Run Options

The benchmark performance metric for Training workloads (3D-Unet, ResNet-50, and Cosmflow) is samples per second, subject to a minimum accelerator utilization (AU) defined for that workload. Higher samples per second is better.

To pass a benchmark run, the AU should be equal to or greater than the minimum value, and is computed as follows:

AU (percentage) = (total_compute_time/total_benchmark_running_time) * 100
All the I/O operations from the first step are excluded from the AU calculation in order to avoid the disturbance in the averages caused by the startup costs of the data processing pipeline, allowing the AU to more-quickly converge on the steady-state performance of the pipeline. The I/O operations that are excluded from the AU calculation are included in the samples/second reported by the benchmark, however.

If all I/O operations are hidden by compute time, then the total_compute_time will equal the total_benchmark_running_time and the AU will be 100%.

The total compute time can be derived from the batch size, total dataset size, number of simulated accelerators, and sleep time:

total_compute_time = (records_per_file * total_files) / simulated_accelerators / batch_size * computation_time * epochs.




8. Single-host Submissions
This section only applies to Training workloads, the equivalent topic is covered in section 2.2.2, "subset mode".

Submitters can add load to the storage system in two orthogonal ways: (1) increase the number of simulated accelerators inside one host node (i.e., one machine), and/or (2) increase the number of host nodes connected to the storage system.

For single-host submissions, increase the number of simulated accelerators by changing the --num-accelerators parameter to the benchmark.sh script. Note that the benchmarking tool requires approximately 0.5GB of host memory per simulated accelerator.

For single-host submissions, CLOSED and OPEN division results must include benchmark runs for the maximum simulated accelerators that can be run on ONE HOST NODE, in ONE MLPerf Storage job, without going below the 90% accelerator utilization threshold.

9. Distributed Training Submissions
This setup simulates distributed training of a single training task, spread across multiple host nodes, on a shared dataset. The current version of the benchmark only supports data parallelism, not model parallelism.

Submitters must respect the following for multi-host node submissions:

All the data must be accessible to all the host nodes.
The number of simulated accelerators in each host node must be identical.
While it is recommended that all host nodes be as close as possible to identical, that is not required by these Rules. The fact that distributed training uses a pool-wide common barrier to synchronize the transition from one step to the next of all host nodes results in the overall performance of the cluster being determined by the slowest host node.

Here are a few practical suggestions on how to leverage a set of non-identical hardware, but these are not requirements of these Rules. It is possible to leverage very large physical nodes by using multiple Containers or VM guest images per node, each with dedicated affinity to given CPUs cores and where DRAM capacity and NUMA locality have been configured. Alternatively, larger physical nodes that have higher numbers of cores or additional memory than the others may have those additional cores or memory disabled.

For distributed training submissions, CLOSED and OPEN division results must include benchmark runs for the maximum number of simulated accelerators across all host nodes that can be run in the distributed training setup, without going below the 90% accelerator utilization threshold. Each host node must run the same number of simulated accelerators for the submission to be valid.



For CLOSED submissions of this benchmark, the MLPerf Storage codebase takes the place of the AI/ML algorithms and framework, and therefore cannot be changed. The sole exception to this rule is if the submitter decides to apply the code change identified in PR#299 of the DLIO repo in github, the resulting codebase will be considered "unchanged" for the purposes of this rule. 

A small number of parameters can be configured in CLOSED submissions; listed in the tables below.

**Table: Training Workload Tunable Parameters for CLOSED**

| Parameter                    | Description                                                                                                                         | Default  |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|----------|
| *Dataset parameters*         |                                                                                                                                     |          |
| dataset.num_files_train      | Number of files for the training set                                                                                                | --       |
| dataset.num_subfolders_train | Number of subfolders that the training set is stored                                                                                | 0        |
| dataset.data_folder          | The path where dataset is stored                                                                                                    | --       |
|                              |                                                                                                                                     |          |
| *Reader parameters*          |                                                                                                                                     |          |
| reader.read_threads          | Number of threads to load the data                                                                                                  | --       |
| reader.computation_threads   | Number of threads to preprocess the data (only for resnet)                                                                          | --       |
| reader.transfer_size         | An int64 scalar representing the number of bytes in the read buffer. (only supported for Tensorflow models -- Resnet and Cosmoflow) |          |
| reader.prefetch_size         | An int64 scalar representing the amount of prefetching done, with values of 0, 1, or 2.                                             |          |
| reader.odirect               | Enable ODIRECT mode for Unet3D Training                                                                                             | False    |
|                              |                                                                                                                                     |          |
| *Checkpoint parameters*      |                                                                                                                                     |          |
| checkpoint.checkpoint_folder | The folder to save the checkpoints                                                                                                  | --       |
|                              |                                                                                                                                     |          |
| *Storage parameters*         |                                                                                                                                     |          |
| storage.storage_root         | The storage root directory                                                                                                          | ./       |
| storage.storage_type         | The storage type                                                                                                                    | local_fs |

In addition to what can be changed in the CLOSED submission, the following parameters can be changed in the benchmark.sh script:

| Parameter                    | Description                                | Default                                                             |
|------------------------------|--------------------------------------------|---------------------------------------------------------------------|
| framework                    | The machine learning framework.            | 3D U-Net: PyTorch<br>ResNet-50: Tensorflow<br>Cosmoflow: Tensorflow |
|                              |                                            |                                                                     |
| *Dataset parameters*         |                                            |                                                                     |
| dataset.format               | Format of the dataset.                     | 3D U-Net: .npz<br>ResNet-50: .tfrecord<br>Cosmoflow: .tfrecord      |
| dataset.num_samples_per_file |                                            | 3D U-Net: 1<br>ResNet-50: 1251<br>Cosmoflow: 1                      |
|                              |                                            |                                                                     |
| *Reader parameters*          |                                            |                                                                     |
| reader.data_loader           | Supported options: Tensorflow or PyTorch.  | 3D U-Net: PyTorch<br>ResNet-50: Tensorflow<br>Cosmoflow: Tensorflow |

# 4. Sanity Checking the Checkpointing Options

dgh

## 4.1.  CLOSED Versus OPEN Options

dgh

## 4.2.  Benchmark Run Options

The checkpoints that are written are quite large. If the checkpoint size per client node is less than 3x the client node's memory capacity, then the filesystem cache needs to be cleared between the write and read phases.

We enforce fsync to be applied during checkpoint writes to ensure data is flushed to persistent storage. fsync is enabled by default in all workload configuration files.

A checkpoint workload submission must include 10 checkpoints written and 10 checkpoints read as well as the logs for any optional processes as outlined in section 2.2.5 (clearing caches, storage remapping, etc)

Benchmark results may be submitted for the following four model configurations. The associated model architectures and parallelism settings are listed below. The number of MPI processes must be set to 8, 64, 512, and 1024 for the respective models for CLOSED submission. 

For CLOSED submissions, participants are not permitted to change the total number of simulated accelerators. However, they may adjust the number of simulated accelerators per host, as long as each host uses more than 4 simulated accelerators. This allows the use of nodes with higher simulated accelerator density and fewer total nodes. Note: the aggregate simulated accelerator memory across all nodes must be sufficient to accommodate the model’s checkpoint size.

**Table 2 LLM models**

| Model                  | 8B     | 70B    | 405B    | 1T     |
|------------------------|--------|--------|---------|--------|
| Hidden dimension       | 4096   | 8192   | 16384   | 25872  |
| FFN size               | 14336  | 28672  | 53248   | 98304  |
| num_attention_heads    | 32     | 128    | 128     | 192    |
| num_kv_heads           | 8      | 8      | 8       | 32     |
| Num layers             | 32     | 80     | 126     | 128    |
| Parallelism (TPxPPxDP) | 1×1×8  | 8×1x8  | 8×32×2  | 8×64×2 |
| Total Processes        | 8      | 64     | 512     | 1024   |
| ZeRO                   | 3      | 3      | 1       | 1      |
| Checkpoint size        | 105 GB | 912 GB | 5.29 TB | 18 TB  |
| Subset: 8-Process Size | 105 GB | 114 GB | 94 GB   | 161 GB |


**Table: Checkpoint Workload Tunable Parameters for CLOSED**

| Parameter                        | Description                                                 | Default               |
|----------------------------------|-------------------------------------------------------------|-----------------------|
| checkpoint.checkpoint_folder     | The storage directory for writing and reading checkpoints   | ./checkpoints/<model> |
| checkpoint.num_checkpoints_write | The number of checkpoint writes to do in a single dlio call | 10                    |
| checkpoint.num_checkpoints_read  | The number of checkpoint reads to do in a single dlio call  | 10                    |


For OPEN submissions, the total number of processes may be increased in multiples of (TP×PP) to showcase the scalability of the storage solution.

**Table 3: Configuration parameters and their mutability in CLOSED and OPEN divisions**

| Parameter                          | Meaning                                      | Default value                                 | Changeable in CLOSED | Changeable in OPEN |
|------------------------------------|----------------------------------------------|-----------------------------------------------|----------------------|--------------------|
| --ppn hostname:slotcount           | Number of processes per node                 | N/A                                           | YES (minimal 4)      | YES (minimal 4)    |
| --num-processes                    | Total number of processes                    | Node local: 8<br>Global: the value in Table 1 | NO                   | YES                |
| --checkpoint-folder                | The folder to save the checkpoint data       | checkpoint/{workload}                         | YES                  | YES                |
| --num-checkpoints-write            | Number of write checkpoints                  | 10 or 0**                                     | NO                   | NO                 |
| --num-checkpoints-read             | Number of write checkpoints                  | 10 or 0**                                     | NO                   | NO                 |

**In the ``--ppn`` syntax above, the ``slotcount`` value has the same meaning as the ``ppn`` value, the number of processes per node to run.**

** By default, --num-checkpoints-read and --num-checkpoints-write are set to be 10. To perform write only, one has to turn off read by explicitly setting ``--num-checkpoints-read=0``; to perform read only, one has to turn off write by explicitly set  ``--num-checkpoints-write=0``

### 4.2.  Storage System Must Be Simultaneously R/W or _Remappable_

For storage systems where 1 host has write access to a volume but all hosts have read access, the above process also satisfies the requirements so long as reads can be fulfilled immediately following a write.

For storage systems where 1 host has write access to a volume and a "remapping" process is required for other hosts to read the same data, the time to remap must be measured and included in the submission.

When a checkpoint is taken/written, it must be written to stable storage, but that checkpoint does not need to be readable by other other hosts yet. If it is not readable by other hosts immediately after the checkpoint write is complete, if it requires some additional processing or reconfiguration before the checkpoint is readable by other hosts, the time duration between the checkpoint being completed and the earliest time that that checkpoint could be read by a different host node must be reported in the SystemDescription.yaml file. That duration between write completion and availability for reading will be added to the time to read/recover from the benchmark.

Any processes between the write and read phases of checkpointing that are required before data can be read by a different host than wrote the data must be measured and included in the submission. The time for these processes will be added to the recovery time and throughput calculation for submitted scores

The system_configuration.yaml document must list whether the solution support simultaneous reads and/or writes as such:

System:
  shared_capabilities:
    multi_host_support: True            # False is used for local storage
    simultaneous_write_support: False   # Are simultaneous writes by multiple hosts supported in the submitted configuration
    simultaneous_read__support: True    # Are simultaneous reads by multiple hosts supported in the submitted configuration


## 5.  Validating The Phases

The MLPerf Storage working group provides a benchmark implementation which includes:

* Scripts to determine the minimum dataset size required for your system, for a given benchmark.
* Scripts for data generation.
* Benchmark tool, based on DLIO, with configuration files for the benchmarks.
* A script for running the benchmark on one host (additional setup is required if you are running a distributed training benchmark – see Section 5).
* A script for generating the results report (additional scripting and setup may be required if you are running a distributed training benchmark – see Section 5), and potentially additional supporting scripts.

Each of the benchmarks described in this document have a requirement for multiple runs. This is to ensure consistency of operation of the system under test as well as ensure statistical significance of the measurements.  Unless otherwise noted, the multiple runs for a workload need to be run consecutively. To ensure this requirement is met, the time between runs (from the stop time of one run and the start time to the next run) needs to be less than the time to execute a single run. This is to discourage cherry-picking of results which is expressly forbidden and against the spirit of the rules.



















