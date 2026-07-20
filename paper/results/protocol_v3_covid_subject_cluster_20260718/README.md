# COVID-19 Subject-Cluster Sensitivity

The Protocol V3 COVID-19 test set contains repeated images from recovered
subject tokens. Main-table means remain image weighted. This sensitivity
analysis resamples subjects and training seeds for confidence intervals and
performs Wilcoxon/paired t-tests on equal-subject mean differences.

- Cases: 2113
- Recovered test subjects: 474
- Bootstrap replicates per comparison: 10000

See `clustered_statistics.csv` for compact results and
`clustered_statistics.json` for source paths and bootstrap seeds.
