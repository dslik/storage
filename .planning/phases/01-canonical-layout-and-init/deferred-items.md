
## 2026-06-19 (Plan 01-03)

- `tests/unit/test_version.py::test_version_matches_pyproject` and
  `::test_version_fallback_reads_pyproject` were already failing on `main`
  (version assertion expects `3.0.13`, sees `3.0.12`). Not caused by this
  plan's --systemname plumbing. Out of scope per execute-plan scope-boundary.
- `tests/unit/test_benchmarks_base.py`, `test_parquet_reader.py`,
  `test_vdb_modular_fake_backend.py` fail to import in this dev shell because
  `psutil`, `pyarrow`, and `numpy` are not installed. Pre-existing environment
  gap. Run `pip install -e ".[test,full]"` to pick them up.
