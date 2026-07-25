# CLAIM 2024 checklist: MedEquiSeg

Guideline: Tejani et al., *Checklist for Artificial Intelligence in Medical
Imaging (CLAIM): 2024 update*, Radiology: Artificial Intelligence 2024;
6:e240300. <https://doi.org/10.1148/ryai.240300>

The locations below use stable manuscript section, table, and figure names so
they remain valid if pagination changes during typesetting. “No” and “N/A”
entries include an explanation, as required by CLAIM.

Final line-numbered review PDF map: Abstract, pp. 1–2; Background, pp. 2–5;
Methods, pp. 6–17; Results, pp. 18–27; Discussion, pp. 28–32; Conclusions,
p. 33; Declarations, pp. 34–35; References, pp. 36–40. Supplementary Tables
S1–S13 are on pp. 3–12 of the supplement, S14 on p. 13, S15 on p. 14, and S16
on p. 15. Visible continuous line numbers are included in both PDFs.

| Item | CLAIM 2024 topic | Status | Manuscript location / explanation |
|---:|---|:---:|---|
| 1 | Identify the AI methodology | Yes | Title; Abstract Methods; Methods, “MedEquiSeg architecture.” |
| 2 | Structured summary | Yes | Structured Abstract: Background, Methods, Results, Conclusions. |
| 3 | Scientific and clinical background / intended use | Yes | Background; “Prompt-evidence scope and real-text testing.” The work is explicitly a benchmark mechanism study, not a clinical tool. |
| 4 | Study objectives and hypotheses | Yes | End of Background; Methods, “Endpoints and statistical analysis.” |
| 5 | Prospective or retrospective design | Yes | Abstract, Methods, and Limitations identify the ordered analysis as retrospective and post hoc. |
| 6 | Study goal (model creation, evaluation, or both) | Yes | Methods describes model development and fixed-holdout evaluation. |
| 7 | Data sources | Yes | Methods, “Dataset and prompt provenance”; Table 1; cited source datasets. |
| 8 | Eligibility and exclusion criteria | Yes | Methods, “Data partitioning and evaluation protocol”; BRISC duplicate-resolution rule; Table 1. |
| 9 | Data preprocessing | Yes | Methods, architecture, training, augmentation, and native-grid evaluation subsections. |
| 10 | Selection of data subsets | Yes | Fixed Train/Val/Test folders and public manifests are described; splits do not change across seeds. |
| 11 | De-identification | Yes | Declarations, “Ethics approval and consent to participate”; only public de-identified benchmark data were used. |
| 12 | Missing data handling | Yes | Required image--mask--prompt pairing and duplicate sanitation are described; incomplete pairings are not imputed. |
| 13 | Image acquisition protocol | No | Acquisition information is limited to that reported by the five source publications; scanner-level metadata were not uniformly available in the released benchmark files. This is acknowledged as a limitation. |
| 14 | Definition of the reference standard | Yes | Methods, “Reference standards.” Binary masks distributed with each source dataset are the reference standards. |
| 15 | Rationale for the reference standard | Yes | Methods, “Reference standards.” Released benchmark masks enable like-for-like comparison. |
| 16 | Source of the reference standard | Yes | Methods, “Dataset and prompt provenance” and “Reference standards”; original dataset papers are cited. |
| 17 | Test-set annotation process | Yes | Methods, “Reference standards.” No new test annotation, editing, or adjudication was performed; dataset-provided masks were retained. |
| 18 | Inter- and intra-rater variability | No | Not consistently reported in the machine-readable public releases and not re-measured; stated in “Reference standards” and Limitations. |
| 19 | Partition assignment method | Yes | Methods, “Data partitioning and evaluation protocol”; fixed released partitions and the COVID-19 grouped sensitivity are described. |
| 20 | Level of partition disjointness | Yes | Methods and Limitations distinguish image-level disjointness from unavailable patient/sequence identities; COVID-19 subject recovery is reported separately. |
| 21 | Sample-size determination | Yes | Table 1 reports all available fixed test cases. No prospective power calculation was used because the study evaluates complete released benchmark partitions. |
| 22 | Model architecture detail | Yes | Methods; Figs. 1–2; Supplementary layer/operator figures; effective two-projector ATConv graph. |
| 23 | Software, dependencies, and hardware | Yes | Methods and Supplementary Table S11; environment record; Data availability. |
| 24 | Model initialization | Yes | Methods describes pretrained CLIP/BioMedCLIP initialization and three training seeds. |
| 25 | Training procedure | Yes | Methods reports optimization, loss, augmentation, epochs, checkpoint selection, and seeds. |
| 26 | Final model selection | Yes | Validation split selects the best checkpoint; the held-out test split is used only for final prediction/evaluation. |
| 27 | Ensembling | Yes | Seed-level means and qualitative three-seed majority votes are explicitly distinguished; the deployed forward graph itself is not a seed ensemble. |
| 28 | Evaluation metrics | Yes | Methods defines Dice, IoU, NSD, HD95, ASSD, and empty-mask handling. |
| 29 | Statistical uncertainty | Yes | Methods, “Endpoints and statistical analysis”; paired bootstrap CIs, Holm adjustment, hierarchical and clustered sensitivities. |
| 30 | Robustness / sensitivity analyses | Yes | Prompt controls, mask-presence analysis, COVID-19 subject analyses, matched BUSI rewriting, and Supplementary Table S16. |
| 31 | Explainability | N/A | No post hoc saliency or explainability method is claimed; the work evaluates segmentation and prompt sensitivity. |
| 32 | Internal testing results | Yes | Results; main tables/figures and Supplementary Tables S1–S16. |
| 33 | External testing | No | The five datasets are evaluated as separate fixed benchmarks with dataset-specific training; no model is transferred to an independent institution without retraining. This is stated in Limitations. |
| 34 | Trial registration | N/A | Retrospective secondary computational benchmark study; no clinical trial or prospective participant recruitment. |
| 35 | Included and excluded case counts | Yes | Table 1, BRISC duplicate-resolution description, prompt-control changed/N counts, and public manifests. |
| 36 | Demographic and clinical characteristics | No | Age, sex, and other patient-level characteristics are not consistently available in the released benchmark metadata; this limits subgroup analysis. |
| 37 | Performance estimates with uncertainty | Yes | Main accuracy and prompt-control results report point estimates, seed variability, and paired confidence intervals. |
| 38 | Diagnostic-accuracy measures | N/A | The primary task is pixel-wise segmentation, not patient-level diagnostic classification; segmentation metrics are reported instead. |
| 39 | Failure and error analysis | Yes | Results, qualitative error analysis; boundary/empty-mask tables and one-empty failure rates. |
| 40 | Limitations | Yes | Dedicated “Limitations and next evidence” subsection covers causal, prompt, source-identity, grouping, annotation, and generalizability limits. |
| 41 | Implications for practice and future work | Yes | Discussion and Conclusions explicitly limit clinical interpretation and specify clinician-authored text and independent cross-domain testing as future work. |
| 42 | Protocol detail sufficient for replication | Yes with limitation | Public code, manifests, lock files, statistics, and environment are released. The manuscript explicitly states that the exact historical executed training snapshot is unavailable. |
| 43 | Availability of data, code, model, and protocol | Yes | Declarations, “Availability of data and materials”; GitHub repository and archived version v1.1.0; dataset sources cited. Medical images and third-party weights are not redistributed. |
| 44 | Funding source and funder role | Partial | Funding sources and grant numbers are stated. The funders’ role still requires explicit author confirmation before the submission checklist is finalized. |
