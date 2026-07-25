# Public-5 Audited Source-and-Results Package

This package contains the public, non-identifying artifacts needed to audit the
Public-5 version of the MedEquiSeg manuscript. It is a source-and-results
package, not a copy of the underlying medical datasets, third-party checkpoints,
or an exact snapshot of the source tree that executed every training run.

## Scientific scope

The released manuscript uses only BUSI, ClinicDB, BUS-BRA, BRISC, and COVID-19.
Its principal evidence is a retrospective ordered configuration study with five
retained configurations, five datasets, and three seeds (75 training cells),
together with four frozen-checkpoint prompt controls per cell. The shared
augmentation sampler is a separately recorded training-policy amendment on the
fixed data and evaluation specification. Because that training policy appears only
in the complete configuration, the ordered matrix supports conditional
configuration contrasts rather than a standalone augmentation-policy main
effect or complete component interactions. A separately trained, matched
three-seed BUSI experiment compares prompt rewriting with no rewriting while
holding the model architecture and shared augmentation policy fixed.

Four ATConv targets were registered in designated runs. Runtime hooks found two
forward-active projector replacements and two dormant legacy neck members. The
manuscript and release therefore describe the effective two-projector graph.

## Source-identity limitation

The retained 75-cell matrix recorded aggregate training-code SHA-256
`2732eeac3bfcb99163752ad95b1037e36db1fdf3e829b0963a908c2a67ad4b6e`
and Git commit `f3afe35189889bad2f2a3896461deb8d385892cc`. The available
commit and dated source packages do not reconstruct that aggregate digest bit
for bit. This archive releases the current audited implementation plus immutable
run, manifest, checkpoint, and control-export identities; it must not be
described as the exact executed training snapshot.

## Included

- Protocol lock, registry, and sanitized manifests with case-level text
  prompts for the five public sets.
- Retained augmentation, training, prediction, evaluation, and statistical code.
- Three-seed overlap, boundary, semantic-control, complexity, and qualitative results.
- The retained 75-cell summaries, ATConv activation, stratified-derangement
  audits, and the separately matched BUSI prompt-rewriting control.
- Public-5 BMC Medical Imaging LaTeX source, figures, manuscript, and supplement.
- Repository README, citation metadata, license, notice, and third-party notices.
- A file-level release manifest containing byte counts and SHA-256 checksums.

## Excluded

- Medical image and mask files; obtain the public datasets from their cited sources.
- Third-party pretrained weights and model repositories.
- Training checkpoints, embedding caches, raw logs, credentials, and machine paths.
- Any non-public dataset, prompt, path, split, checkpoint, or result artifact.

## Protocol summary

- Fixed train/validation/test roles; only validation Dice selects checkpoints.
- Training seeds 123, 456, and 789; fixed split seed 123.
- Native-grid Dice, IoU, NSD at two pixels, HD95, and ASSD evaluation.
- Both-empty and one-empty cases follow the locked explicit policy.
- The matched BUSI control uses identical architecture, training budget, and
  shared augmentation policy while varying prompt rewriting.
- Statistics first average seeds by case, then use dataset-stratified bootstrap and Holm correction.

## Repository identifier

The public code repository is
<https://github.com/zjj30/MedEquiSeg>. Publishing this source-and-results
package does not release medical image data or any non-public cohort artifact.

From a clean clone, verify the 75 retained cells and rebuild the deterministic
archive with:

```bash
python paper/analysis/generate_medequiseg_factorial_manuscript_assets.py \
  --seed-metrics paper/results/medequiseg_factorial_public5_20260715/seed_metrics.csv \
  --check-only
python paper/analysis/build_protocol_v3_public5_release_package.py \
  --project-root .
```
