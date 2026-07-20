# Protocol V3 Public-5 Audited Source-and-Results Package

This package contains the public, non-identifying artifacts needed to audit the
Public-5 version of the MedEquiSeg manuscript. It is a source-and-results
package, not a copy of the underlying medical datasets, third-party checkpoints,
or an exact snapshot of the source tree that executed every training run.

## Scientific scope

The released manuscript uses only BUSI, ClinicDB, BUS-BRA, BRISC, and COVID-19.
Its principal evidence is a retrospective ordered ablation with six
configurations, five datasets, and three seeds (90 training cells), together
with four frozen-checkpoint prompt controls per cell. The shared R11 sampler is
a separately frozen training-policy amendment on the fixed Protocol V3 data and
evaluation specification; it is not represented as the original July 10
augmentation route.

Four ATConv targets were registered in designated runs. Runtime hooks found two
forward-active projector replacements and two dormant legacy neck members. The
manuscript and release therefore describe the effective two-projector graph.

## Source-identity limitation

The 90-cell matrix recorded aggregate training-code SHA-256
`2732eeac3bfcb99163752ad95b1037e36db1fdf3e829b0963a908c2a67ad4b6e`
and Git commit `f3afe35189889bad2f2a3896461deb8d385892cc`. The available
commit and dated source packages do not reconstruct that aggregate digest bit
for bit. This archive releases the current audited implementation plus immutable
run, manifest, checkpoint, and control-export identities; it must not be
described as the exact executed training snapshot.

## Included

- Protocol V3 lock, registry, and sanitized manifests with case-level text
  prompts for the five public sets.
- R11/R11NR augmentation, training, prediction, evaluation, and statistical code.
- Three-seed overlap, boundary, semantic-control, complexity, and qualitative results.
- The 90-cell lineage, ATConv activation, and stratified-derangement audits.
- Public-5 BMC Medical Imaging LaTeX source, figures, manuscript, and supplement.
- A file-level release manifest containing byte counts and SHA-256 checksums.
- A validated submission-metadata template and deterministic finalizer for the
  title page, Declarations, cover letter, PDF rebuild, archive rebuild, and
  final readiness audit.

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
- R11 versus R11NR uses identical image-mask plans and changes only text rewriting.
- Statistics first average seeds by case, then use dataset-stratified bootstrap and Holm correction.

## Repository identifier

The permanent public code repository is
<https://github.com/zjj30/MedEquiSeg>. Publishing this source-and-results
package does not release medical image data or any non-public cohort artifact.

After author-confirmed facts and the permanent repository URL are entered in a
copy of `paper/submission_metadata_public5_template.yaml`, validate without
changing the manuscript:

```bash
python paper/analysis/finalize_public5_submission.py \
  --project-root . \
  --metadata paper/submission_metadata_public5.yaml
```

Review the generated preview, then add `--apply` to update the submission files,
compile both PDFs, rebuild this archive, and require a zero-blocker readiness audit.
