# BMC Statistical Table Consistency Audit

Status: **PASS**

- Displayed-value and caption checks: 223
- COVID-19 subject-cluster rows: 6
- R11-LR diagnostic dataset-metric rows: 4
- R11/R11NR dataset-metric rows: 10
- R11/R11NR macro rows: 2
- Semantic-control Dice rows: 15
- Semantic shuffled-control Holm-significant datasets: BRISC, BUS-BRA, BUSI, COVID-19, ClinicDB
- Errors: 0

The audit recomputes macro sample standard deviations from equal-dataset seed-level macro means and checks all displayed means, standard deviations, deltas, confidence bounds, adjusted p-values, and the semantic-control caption.
