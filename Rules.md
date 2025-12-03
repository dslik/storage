# MLPerf™ Storage V2.0 Benchmark Validation
——————————————————————————————————————————

- [MLPerf Storage Benchmark Submission Guidelines v2.0](#mlperf-storage-benchmark-submission-guidelines-v20)
  - [1. Introduction](#1-introduction)
  - [2. Directory Structure for All Submissions](#2-directory-structure-for-all-submissions)
  - [3. Sanity Checking the Training Options](#3-sanity-checking-the-training-options)
  - [4. Sanity Checking the Checkpointing Options](#3-sanity-checking-the-checkpointing-options)

## 1. Introduction

These are the requirements for the *submission validation checker* for version 2.0 of the MLPerf™ Storage benchmark,
but since the `mlpstorage` tool will be responsible for generating the vast majority (if not all) of the contents of a submission, it is also a spec for what `mlpstorage` should generate.

The *submission validation checker* should check that the tested directory hierarachy matches the below requirements and output messages for all cases where it does not match.
The tool should make it's best effort to continue testing all the other aspects of the directory hierarchy after any given failure.
If the tested directory hierarchy does not meet all of the below requirements, then it should be labelled as invalid and the validation check should fail.

Even if the structure of a submission package matches the spec, the options that were used to run the benchmark may not fall within acceptable bounds,
so we need the *submission validation checker* to check for illegal/inapproriate option settings,
and for semantic mismatches between different options that were used.

### 2. Directory Structure for All Submissions

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

**2.9.**  All the configuration parameters and hardware and software components of the system-under-test that are part of a given *system name* must be identical.  Any changes to those configuration parameters or hardware or software must be submitted as a separate *system name*.  These names are case-sensitive.

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

### 3. Sanity Checking the Training Options

dfg

#### 3.1.  CLOSED Versus OPEN Options

dfg

#### 3.2.  Dataset Generation Options

dfh

#### 3.3.  Benchmark Run Options

dfg

### 4. Sanity Checking the Checkpointing Options

dgh

#### 4.1.  CLOSED Versus OPEN Options

dgh

#### 4.2.  Benchmark Run Options

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
























