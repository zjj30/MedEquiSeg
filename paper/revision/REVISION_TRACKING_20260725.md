# Submission revision tracking — 2026-07-25

This file is the auditable change ledger for the grouped-split and submission
revision. Experimental values are changed only when a named machine-readable
source is regenerated. The invalid `Shared plan (no rewrite)` configuration is
excluded from every primary, supplementary, and release result.

| Priority | Issue | Required change | Verification | Status |
|---|---|---|---|---|
| P0 | CVC-ClinicDB frames from the same video sequence crossed splits | Rebuild by the documented 29-sequence mapping; retrain all five retained configurations for three seeds | Group-overlap audit and 15 completed run records | Manifest fixed; training pending |
| P0 | BUS-BRA images from the same patient crossed splits | Rebuild by the numeric patient prefix; retrain all five retained configurations for three seeds | Group-overlap audit and 15 completed run records | Manifest fixed; training pending |
| P0 | Manuscript confidence intervals and released CSVs use inconsistent estimands | Use one case/subject-after-seed-reduction estimator and one canonical generated source | Table-to-source audit and regenerated PDF | Pending new runs |
| P0 | Stale 90/90 and `Shared plan (no rewrite)` artifacts remain | Purge from manuscript, release manifest, summaries, archives, and source tree | Recursive forbidden-term audit | Pending |
| P0 | Clean clone cannot reproduce the release | Pin the upstream CausalCLIPSeg source, fix script entry order, regenerate manifest, and run clean-clone smoke checks | Clean-clone log and hashes | Pending |
| P1 | Bundle comparison is described as component causality | Limit claims to the tested bundled configurations; use prior 90-run evidence only for the valid five configurations | Claim/evidence audit | Pending |
| P1 | Prompt controls overstate reliability and rely on one derangement | Describe privileged/oracle prompt sensitivity; add fixed stratified derangements without causal overclaim | Control-source table and text audit | Pending |
| P1 | Transform and data/reference-standard descriptions are incomplete | State exactly which transforms rewrite directional terms and document grouping, masks, and reference standards | Methods/Data availability audit | Pending |
| P1 | Ethics, funder role, author approval, originality, and competing interests need submission-ready wording | Insert the confirmed declarations without placeholders | Front/back-matter audit | Pending |
| P2 | Engineering vocabulary appears in scientific prose | Remove protocol IDs, run IDs, gate language, cache jargon, and internal bundle labels from prose/captions | Forbidden-term audit | Pending |
| P2 | Layout and reporting details need cleanup | Reduce large tables/whitespace, qualify selected-model comparisons, replace “ground truth” with “reference mask,” and correct reference count | Rendered-PDF inspection | Pending |

## Locked corrective experiment

- Lock: `smoke_tests/protocol_v3/protocol_lock_grouped_public5.yaml`
- Affected datasets: CVC-ClinicDB and BUS-BRA
- Configurations: base, BioMedCLIP, ATConv, BioMedCLIP+ATConv, and full MedEquiSeg
- Seeds: 123, 456, and 789
- Planned training runs: 5 × 2 × 3 = 30
- Selection: validation split only; final reporting: held-out test split
- Resampling unit: video sequence for CVC-ClinicDB and patient for BUS-BRA
