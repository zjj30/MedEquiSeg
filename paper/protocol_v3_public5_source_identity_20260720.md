# Protocol V3 Public-5 Source Identity

Date: 2026-07-20

## Frozen result identity

- Protocol ID: `MEDSEG_TEXT_V3_20260710`
- Data/evaluation hash: `abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718`
- Recorded aggregate training-code SHA-256:
  `2732eeac3bfcb99163752ad95b1037e36db1fdf3e829b0963a908c2a67ad4b6e`
- Recorded Git commit: `f3afe35189889bad2f2a3896461deb8d385892cc`
- Ninety unique run-configuration hashes and ninety unique checkpoint hashes.
- One manifest hash per public dataset across all eighteen configuration/seed combinations.
- Three hundred sixty prompt-control exports bound to checkpoint, manifest,
  prediction threshold, and protocol hash.

The machine-readable identities are stored in
`paper/results/bmc_submission_audit_20260719/factorial_run_lineage_deep.json`.

## Reproducibility boundary

This archive contains the current audited implementation, analysis code,
sanitized public manifests, result tables, manuscript source, and audit outputs.
It supports inspection and regeneration of the released tables from stored
result artifacts.

The available commit and dated source copies do not reproduce the recorded
aggregate training-code digest bit for bit. Four files included in that digest
are absent from the recorded commit. The archive therefore does not establish
clean-clone bitwise retraining of the 90-cell matrix.

The R11 shared sampler is a separately frozen training-policy amendment. The
Protocol V3 hash identifies the data and evaluation specification, not the full
training policy. Per-run identities additionally retain recipe, augmentation,
code, manifest, cache, checkpoint, and command-log evidence.

## Execution disclosures

Eighty-nine of ninety training logs contain a clean 100-epoch sequence. One
intermediate COVID-19 seed-789 log contains duplicate-write damage, while its
run metadata, four controls, and recomputed checkpoint hash consistently bind
the reported numerical cell to one checkpoint.

Designated runs registered four ATConv replacements. Static graph inspection
and a real-test-case forward-hook audit found two active projector replacements
and two dormant neck members. Architectural claims use the effective two-projector
graph; the parameter count still includes all instantiated modules.

## Release scope

The archive contains only BUSI, ClinicDB, BUS-BRA, BRISC, and COVID-19 manuscript
and reproducibility artifacts. It contains no non-public dataset material.
