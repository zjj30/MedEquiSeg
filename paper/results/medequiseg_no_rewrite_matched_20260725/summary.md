# Targeted matched prompt-rewrite control (BUSI)

The two arms passed the metadata audit. They share the same current code snapshot, protocol, manifest, BioMedCLIP cache, recipe, epoch count, seeds, and deterministic image/mask augmentation plans. The only intended intervention is prompt rewriting after discrete flips or right-angle rotations.

- code SHA-256: `db558aa0431c002c937c4348e8e82fe00a01946a06453bbd2d777e85eee2b5c4`
- manifest SHA-256: `95b553c061065294ac39d648f634b7f7e8e1d182b99eb7fedc897ddff3fef36f`
- cache SHA-256: `8ad6798b2cac30bef6bd1234390a6c0c6d854113d668660fcde6a2bf8c918abd`
- recipe: `biomed_lcaug_v2_atconv4`
- seeds: `123, 456, 789`

## Seed-level metrics

| dataset | arm | seed | n_cases | dice | iou | nsd | hd95 | assd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BUSI | rewrite | 123 | 78 | 0.851020 | 0.784195 | 0.419539 | 39.495816 | 12.494863 |
| BUSI | no_rewrite | 123 | 78 | 0.854851 | 0.788405 | 0.441285 | inf | inf |
| BUSI | rewrite | 456 | 78 | 0.862418 | 0.799900 | 0.449349 | 34.875897 | 10.625204 |
| BUSI | no_rewrite | 456 | 78 | 0.822180 | 0.753434 | 0.413853 | 51.055941 | 16.066033 |
| BUSI | rewrite | 789 | 78 | 0.857933 | 0.796836 | 0.459910 | 37.512173 | 11.369612 |
| BUSI | no_rewrite | 789 | 78 | 0.859365 | 0.795447 | 0.453709 | 38.033937 | 11.671153 |

## Paired statistics

| dataset | metric | n_cases | n_seeds | rewrite_mean | rewrite_std_across_seeds | no_rewrite_mean | no_rewrite_std_across_seeds | delta_rewrite_minus_no_rewrite | hierarchical_ci_low | hierarchical_ci_high | wilcoxon_case_mean_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BUSI | dice | 78 | 3 | 0.857123 | 0.005742 | 0.845465 | 0.020292 | 0.011658 | -0.015836 | 0.041675 | 0.012616 |
| BUSI | iou | 78 | 3 | 0.793643 | 0.008325 | 0.779095 | 0.022500 | 0.014548 | -0.015533 | 0.047852 | 0.013599 |

The hierarchical confidence interval resamples both test cases and training seeds. Statistical results describe association under this controlled intervention and do not by themselves establish a broader causal mechanism.
