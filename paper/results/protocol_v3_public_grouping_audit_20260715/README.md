# Protocol V3 Public Grouping Audit

All five public manifests inherit MedCLIPSeg Train/Val/Test folders. Exact
image and mask SHA-256 values are checked across split labels. A recovered
group is reported only when a documented filename token has an unambiguous
subject interpretation; numeric image indices are not treated as patients.

| Dataset | Patient-ID rows | Image-hash overlap | Mask-hash overlap (nonempty/total) | Recoverable groups | Train/val | Train/test | Val/test |
|---|---:|---:|---:|---:|---:|---:|---:|
| BUSI | 0 | 0 | 0/3 | 0 | 0 | 0 | 0 |
| ClinicDB | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 |
| BUS-BRA | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 |
| BRISC | 0 | 0 | 0/1 | 0 | 0 | 0 | 0 |
| COVID-19 | 0 | 0 | 0/0 | 2130 | 434 | 0 | 0 |

Only COVID-19 exposes a validated recoverable subject token. For BUSI,
ClinicDB, BUS-BRA, and BRISC, the released package does not provide a
patient or sequence mapping; zero reported recoverable overlap therefore
means unavailable grouping evidence, not proof of patient independence.

All cross-split mask-hash repetitions are all-zero masks with matching
dimensions; no nonempty mask hash crosses a split. They therefore reflect
the common empty target representation rather than duplicated images.

The audit supports image-disjoint held-out terminology. It does not support
a universal patient-level split claim.
