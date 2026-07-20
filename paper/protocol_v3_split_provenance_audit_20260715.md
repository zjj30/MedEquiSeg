# Protocol V3 Split-Provenance Audit

Audit date: 2026-07-15

## Finding

The five public Protocol V3 manifests inherit the MedCLIPSeg repository's
`Train_Folder`, `Val_Folder`, and `Test_Folder` layout. Their `patient_id`
columns are empty, so the main manuscript must not claim universal patient-level
separation.

A reproducible all-public-manifest audit found no exact image SHA-256 value
crossing a split. BUSI has three and BRISC has one cross-split mask SHA-256
value, but pixel inspection shows that every repeated value is an all-zero mask;
no nonempty mask hash crosses a split. For BUSI, ClinicDB, BUS-BRA, and BRISC,
numeric image identifiers are not treated as patient IDs because the released
metadata provide no validated patient or acquisition-sequence mapping. Detailed
output is stored in
`paper/results/protocol_v3_public_grouping_audit_20260715/`.

COVID-19 filenames expose a recoverable `sub-<ID>` token for 6,307 of 9,258
images, representing 2,130 unique subjects. Under the frozen main Protocol V3
manifest:

- recovered train/test subject overlap: 0;
- recovered validation/test subject overlap: 0;
- recovered train/validation subject overlap: 434.

Thus, the reported COVID-19 test set remains subject-disjoint for all subjects
whose IDs can be recovered, but validation checkpoint selection is not fully
subject-grouped. The remaining 2,951 rows do not expose a recoverable subject ID
and cannot support a patient-level claim.

## Sensitivity protocol

The sensitivity analysis preserves every original test row and all associated
image, mask, prompt, and SHA-256 fields. The original train and validation rows
are pooled, then deterministically reassigned using SHA-256 of
`split_seed=123` plus the recovered subject ID. Rows without a recoverable ID are
treated as singleton case groups. The target validation fraction is 0.20.

Resulting counts:

| Split | Images |
|---|---:|
| Train | 5,706 |
| Validation | 1,439 |
| Test | 2,113 |

The realized validation fraction among train+validation rows is 20.14%. No
recoverable subject crosses any split. The frozen test fingerprint is
`c10378e377f50d25f26a1682b0a9dc41bc69108d82c3385e73fef687b1e1d255`.
The separately hashed sensitivity protocol is
`c0e828a220a5ce116fbe40fac2f113df4b198e1c3c7ca57171bee4b7702adfe3`.

This analysis is a checkpoint-selection sensitivity test. It does not replace
the main Protocol V3 table and does not establish patient-level independence for
datasets whose source metadata lack patient or sequence identifiers.

## Reproducible artifacts

- `smoke_tests/build_protocol_v3_covid_groupval_sensitivity.py`
- `smoke_tests/protocol_v3/manifests/medclipseg_covid19_groupval_sensitivity.csv`
- `smoke_tests/protocol_v3/protocol_lock_covid_groupval_sensitivity.yaml`
- `paper/results/protocol_v3_covid_groupval_sensitivity/manifest_audit.json`
- `paper/analysis/summarize_covid_groupval_sensitivity.py`
- `paper/analysis/audit_protocol_v3_public_grouping.py`
- `paper/results/protocol_v3_public_grouping_audit_20260715/public_grouping_audit.csv`
