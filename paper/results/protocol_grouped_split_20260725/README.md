# Group-disjoint split audit

The corrected manifests preserve the original image counts while assigning
ClinicDB by video sequence and BUS-BRA by patient. No image, mask, prompt,
or content hash field changes.

| Dataset | Grouping unit | Train images/groups | Val images/groups | Test images/groups | Cross-split group overlap |
|---|---|---:|---:|---:|---:|
| ClinicDB | video_sequence | 490/23 | 61/3 | 61/3 | 0 |
| BUS-BRA | patient | 1311/744 | 282/160 | 282/160 | 0 |

Split seed: 123.

ClinicDB uses the published contiguous frame-range mapping for 29 video
sequences. BUS-BRA uses the numeric filename prefix; its 1,064 unique
prefixes and benign/malignant patient counts exactly match the dataset
article's 1,064 patients (722 benign and 342 malignant).

The JSON audit records every holdout group, source/output hash, class
allocation, and overlap check.
