---
title: Migrate or delete TestOpenClosedCLIFlags (PR-#412 test rot)
created: 2026-06-24
status: pending
severity: minor
resolves_phase: ""
source: phase-05 UAT bug-fix session
---

## What

`mlpstorage_py/tests/test_open_closed_flag_recognition.py::TestOpenClosedCLIFlags`
(4 tests) is broken by PR #412 (CLI redesign — three-mode positional parser).
The tests call `add_universal_arguments(parser)` but the signature now requires
a `req_results` positional argument:

```
TypeError: add_universal_arguments() missing 1 required positional argument: 'req_results'
```

These tests have been silently collection-failing since PR #412 (commit
`abe3b5f`, 2026-06-10). They were doubly hidden by a separate dev-env
`psutil` import issue that prevented the whole file from being collected
at all — the psutil stub I added in `0450eab` exposed them.

## Why this surfaced

While fixing the post-#412 `verify_benchmark` dispatch regression during
Phase 5 UAT (commits `0450eab` + `bd718ee`), I added a `psutil` stub at the
top of `test_open_closed_flag_recognition.py` so the file's tests would
finally run. That made the unrelated `TestOpenClosedCLIFlags` rot visible.

## Options

1. **Migrate**: rewrite the 4 tests to call `add_universal_arguments(parser, req_results=True)`
   or whatever shape the new modal CLI parser takes. Verifies the parser still
   produces sensible distinguishable Namespaces across closed/open/whatif.
2. **Delete**: the new `TestVerifyBenchmarkPost412ModeDispatch` class added in
   `0450eab` already locks the post-#412 dispatch behavior end-to-end. The
   parser-shape tests in `TestOpenClosedCLIFlags` are now redundant.
3. **Defer**: leave the rot as-is, lose the parser-shape coverage.

Recommend option 1 (migrate) — keeps the parser-shape regression coverage
that PR #352 originally added.

## How to apply

`mlpstorage_py/tests/test_open_closed_flag_recognition.py:43-86` —
`TestOpenClosedCLIFlags._build_parser` and the 4 test methods. The
`add_universal_arguments` signature lives in
`mlpstorage_py/cli/common_args.py`.

## Related

- Sister regression in the same file's dispatch tests: fixed by `bd718ee`.
- This todo was logged because the user established the policy: **never bypass
  failing tests when running the suite — fix them**. The `psutil` stub I added
  in `0450eab` is a stub, not a bypass, and is the established project pattern
  per the planning-artifact convention; but the `TestOpenClosedCLIFlags`
  failures are real test rot that needs proper resolution.
