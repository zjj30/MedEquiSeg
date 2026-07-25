# COVID-19 grouped train--validation sensitivity

This directory contains only the retained MedEquiSeg and U-Net++ sensitivity
results used in Supplementary Table S10. The public test set is unchanged. A
deterministic grouped split assigns each recoverable subject token to only one
training or validation partition; rows without recoverable subject tokens stay
as case-level groups.

The original-split values in `original_split_reference.csv` are fixed reference
means from separately trained, configuration-matched runs. The grouped-split
models are distinct from the checkpoints in the main 75-cell ordered analysis.
No prompt-rewriting contrast is included in this result family.
