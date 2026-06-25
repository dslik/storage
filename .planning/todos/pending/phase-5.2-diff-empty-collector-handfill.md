---
title: Treat empty-from-collector fields as hand-fill affordances in diff_node_dict_lists
created: 2026-06-24
status: pending
severity: enhancement
resolves_phase: ""
source: phase-05 UAT Test 3 (LIFE-04 hand-fill survival)
proposed_by: curtis.anderson@hammerspace.com
---

## What

Add asymmetric "empty-from-collector means hand-fillable" semantics to
`mlpstorage_py/system_description/diff.py::diff_node_dict_lists` and the
fingerprint construction in `_compute_fingerprint`.

## Proposal

> "If the auto-collector returns a value as '' then we could treat the field
> in the yaml as user-fillable. If the user filled that field and then in
> the future the auto-collector returned something other than '', we could
> treat that changed field as drift."

Concretely, the rule is asymmetric per-field:

| recomputed | on-disk | verdict |
|------------|---------|---------|
| `""` | `""` | no drift (initial state) |
| `""` | `"<user-fill>"` | **no drift** — adopt the on-disk value; field is hand-fillable |
| `"X"` | `""` | no drift, but emit INFO log that collector finally learned the value (or just adopt silently) |
| `"X"` | `"X"` | no drift (steady state) |
| `"X"` | `"Y"` (both non-empty) | **drift** — collector now disagrees with the user's hand-fill |

## Why

Phase 5 LIFE-04 spec says "the hand-edited friendly_description survives
byte-identical in the on-disk file." `friendly_description` is not in the
fingerprint, so the current diff handles it correctly.

But several other YAML fields **default to `""` in the auto-generated file**
because the collector couldn't determine them (e.g. `chassis.model_name`
on a generic Linux box without dmidecode parseable model strings). These
fields are *de facto* hand-fillable — the empty string is a sentinel
meaning "please tell me what this is." Yet these same fields ARE in the
11-tuple fingerprint per D-38, so any hand-fill changes the client's
identity and triggers SystemDriftError on the next run.

The user surfaced this during UAT Test 3 by hand-filling both
`friendly_description` (correct LIFE-04 case) AND `chassis.model_name`
(inadvertently broke the diff). The on-disk vs recomputed delta:

```
on-disk:    (..., chassis.model_name="User-defined model name here", ...)
recomputed: (..., chassis.model_name="",                              ...)
```

Today: SystemDriftError E404, no overwrite (LIFE-04 file no-touch ✓), but
the operator gets a confusing drift report for a field they were
explicitly invited to fill.

Proposed: adopt-on-empty makes hand-fill a one-way enrichment, while real
hardware drift (collector returning a NEW non-empty value that differs
from the on-disk hand-fill) still raises legitimately.

## How to apply (sketch)

1. In `_compute_fingerprint(stanza)` (diff.py:176), add a sibling helper
   `_compute_effective_fingerprint(recomputed_stanza, on_disk_stanza)`
   that returns the recomputed tuple with `""` positions replaced by the
   on-disk value at the same key. Used only when pairing recomputed→on-disk
   clients, not for general dedup.

2. In `diff_node_dict_lists` (diff.py:~220), before the fingerprint orphan
   logic, attempt a "soft pair": for each recomputed client, find an
   on-disk client whose non-empty positions match the recomputed's
   non-empty positions. If a unique pair exists, treat them as the same
   client and let leaf-level comparison handle the remaining fields with
   the asymmetric rule above.

3. Leaf comparison stays as-is — `friendly_description` keeps working,
   plus newly-hand-fillable fingerprint fields like `chassis.model_name`
   become hand-fill-survivable too.

4. New tests in `tests/unit/test_diff.py`:
   - hand-filled fingerprint field survives a re-run with empty collector
   - real drift still fires when collector returns a DIFFERENT non-empty value
   - multi-client topology still pairs correctly (don't conflate genuinely
     different machines)

## Related

- LIFE-04 spec in `.planning/REQUIREMENTS.md`
- D-38 fingerprint composition in Phase 4 / Phase 5 RESEARCH.md
- Phase 5 UAT Test 3 result (passes for `friendly_description` after
  resetting `chassis.model_name`); SystemDriftError on the model_name
  edit is the cited example.

## Out of scope for this todo

Changing what fields the auto-collector emits (e.g., teaching it to read
dmidecode for model_name). The proposal is purely a diff-layer
enhancement; the collector keeps its current "" sentinel.
