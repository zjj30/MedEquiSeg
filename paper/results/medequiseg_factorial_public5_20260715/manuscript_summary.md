# Strict MedEquiSeg Factorial Ablation

- Source SHA-256: `597ecca46f0a00ee1988012dfa6f72898f626bf95227b07246cb310e57330ce2`
- Completeness gate: `90/90` unique dataset-model-seed cells
- Seeds: `123, 456, 789`
- Datasets: BUSI, ClinicDB, BUS-BRA, BRISC, COVID-19

| Configuration | Public-5 Dice (%) | Public-5 IoU (%) |
|---|---:|---:|
| Base model | 86.41 +/- 0.35 | 79.22 +/- 0.34 |
| Base + BioMedCLIP | 86.67 +/- 0.08 | 79.52 +/- 0.06 |
| Base + ATConv | 86.63 +/- 0.32 | 79.42 +/- 0.31 |
| Base + BioMedCLIP + ATConv | 87.03 +/- 0.15 | 79.85 +/- 0.23 |
| MedEquiSeg w/o Prompt Rewrite | 87.96 +/- 0.05 | 81.10 +/- 0.08 |
| MedEquiSeg | 87.87 +/- 0.17 | 80.97 +/- 0.13 |

## Exact-plan rewrite contrast

- Public-5 macro DICE: MedEquiSeg - no rewrite = -0.09 pp.
- Public-5 macro IOU: MedEquiSeg - no rewrite = -0.13 pp.
