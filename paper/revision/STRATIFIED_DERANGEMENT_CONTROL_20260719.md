# Presence/Class-Stratified Prompt Derangement Control

## Material Passport

- Artifact ID: `STRATIFIED_DERANGEMENT_V1_20260719`
- Verification status: `VERIFIED`
- Protocol: `MEDSEG_TEXT_V3_20260710`
- Protocol hash: `abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718`
- Experiment type: post hoc frozen-checkpoint sensitivity analysis
- Retraining: none
- Checkpoints: 5 datasets x 3 seeds = 15 complete-model checkpoints
- Analysis: case-first three-seed mean, 10,000 paired bootstrap replicates, Holm-adjusted Wilcoxon tests

## Question

The original true-versus-shuffled control changes prompt identity but also mixes
target presence, diagnostic class, and wording templates. This control asks
whether true prompts still outperform mismatched prompts after preserving coarse
target presence and diagnostic class.

## Construction

For each public test set, prompts are permuted one-to-one within strata. BUSI uses
normal/benign/malignant; BUS-BRA and BRISC use their released diagnostic classes;
ClinicDB and COVID-19 use their single task classes. The builder reads released
case identifiers and prompt fields only. It does not read masks or image pixels.

The assignment preserves the complete within-stratum prompt multiset and
maximizes changed strings. Preflight checks enforce case closure, donor closure,
original and assigned text hashes, donor-text equality, and stratum equality.
The control changes 3,413/3,426 prompts. The 13 unavoidable unchanged prompts are
BUSI normal cases whose within-stratum strings are identical.

## Results

| Dataset | Changed/N | True Dice | Stratified Dice | Delta Dice pp (95% CI) | Holm p |
|---|---:|---:|---:|---:|---:|
| BUSI | 65/78 | 85.78 | 79.87 | +5.91 (+2.62, +9.86) | .038 |
| ClinicDB | 61/61 | 92.07 | 88.71 | +3.37 (+0.54, +6.96) | .142 |
| BUS-BRA | 282/282 | 90.24 | 90.16 | +0.08 (-0.10, +0.26) | .311 |
| BRISC | 892/892 | 87.18 | 86.81 | +0.37 (+0.13, +0.65) | .011 |
| COVID-19 | 2113/2113 | 84.08 | 65.63 | +18.45 (+17.68, +19.22) | <1e-6 |
| Public-5 macro | 3413/3426 | 87.87 | 82.24 | +5.63 (+4.71, +6.67) | -- |

Changed-only Public-5 Dice is `+5.87 pp` (95% CI `+4.87` to
`+7.03`). For 474 recoverable COVID-19 subjects, equal-subject Dice is
`+22.42 pp` with a subject-cluster 95% interval of `+21.12` to `+23.80`.

## Interpretation

The Public-5 effect attenuates from `+14.66 pp` under unstratified shuffling to
`+5.63 pp` after coarse presence/class matching. Class and template mismatch
therefore explain a substantial share of the original gap. A heterogeneous
residual remains, concentrated in BUSI and COVID-19, while BUS-BRA is neutral.

This is evidence of sensitivity to privileged case-specific prompt content. It
is not evidence of clinical-report understanding: prompts and strata still carry
released target-derived information, and the control does not match morphology,
location, wording template, or every other prompt attribute.

## Reproducibility Evidence

- Matrix status: 15/15 PASS, 0 failures
- Matrix audit: `paper/results/protocol_v3_stratified_derangement_20260719/matrix_audit.json`
- Matrix audit SHA256: `6d473eee3081bc498c8068b599d4bdca3ba763e38a0ad767603513059b94ac42`
- Analysis code SHA256: `5c18f19644e6ce29982cde54f1cf576d62414c2474f3b39d939f45d5f9ca8738`
- Analysis metadata: `paper/results/protocol_v3_stratified_derangement_20260719/analysis_meta.json`
- LaTeX table: `paper/latex/bmc_work/bmc_stratified_derangement_table.tex`

