# MedEquiSeg

Audited source, public-benchmark results, and manuscript artifacts for:

> **MedEquiSeg: Shared Augmentation and Privileged-Prompt Reliability in
> Multimodal Medical Image Segmentation**

Authors, in manuscript order: Xianjun Ye, Ruxue Xing, Mengmeng Zhang, Ye Zhang,
Yongsheng Luo, Yu Kang, and Jianjun Zhu. Xianjun Ye and Ruxue Xing contributed
equally. Yu Kang and Jianjun Zhu are the corresponding authors.

This repository accompanies the public-benchmark submission route. It contains
only non-identifying protocol, analysis, result, figure, and manuscript
artifacts. It is research software and is not intended for clinical use.

## Scientific scope

The retained ordered analysis covers BUSI, ClinicDB, BUS-BRA, BRISC, and
COVID-19 with five configurations and three seeds (75 training cells), plus
four frozen-checkpoint prompt controls per cell. Because shared augmentation
and prompt rewriting enter together in the complete ordered configuration,
that sequence does not identify either component as an isolated causal effect.

A separately trained, matched three-seed BUSI experiment holds the model
architecture, training budget, and shared image--mask augmentation policy fixed
while varying prompt rewriting. Its hierarchical confidence interval includes
zero. This targeted control is reported separately and is not pooled into the
75-cell ordered analysis.

Runtime hooks found two forward-active projector ATConv replacements and two
dormant legacy neck members. The manuscript therefore describes the effective
two-projector forward graph. Prompt controls evaluate sensitivity to
target-derived text; they do not establish clinical-report understanding or
deployment readiness.

## Repository contents

- `smoke_tests/`: retained protocol, augmentation, training, prediction, and
  evaluation code.
- `paper/results/`: public-only aggregate, seed-level, boundary, prompt-control,
  complexity, and statistical outputs used by the manuscript.
- `paper/analysis/`: deterministic analysis, audit, figure, and release builders.
- `paper/revision/`: active stratified-derangement and forward-activation audits.
- `paper/latex/bmc_work_public5/`: BMC Medical Imaging LaTeX source.
- `output/pdf/`: compiled manuscript and supplement.
- `release_manifest.csv`: byte counts and SHA-256 values for every release file.

The release is built from an explicit whitelist. Historical result families,
superseded manuscript sources, backup files, checkpoints, caches, raw logs,
credentials, and private-cohort artifacts are excluded. The aggregate and
seed-level ordered-analysis files contain exactly the five retained
configurations (30 aggregate rows and 75 seed--dataset rows).

For complete scope and exclusions, see
[`paper/protocol_v3_public5_release_readme.md`](paper/protocol_v3_public5_release_readme.md).

## Environment and data

The principal environment used Python 3.11.15, PyTorch 2.5.1+cu121, MONAI
1.5.1, transformers 4.36.2, and open-clip-torch 3.3.0 on NVIDIA RTX 4090 GPUs.
Recorded environments are listed in
[`paper/protocol_v3_environment_record.md`](paper/protocol_v3_environment_record.md).

The sanitized manifests publish case-level prompts and split assignments for
the five public benchmarks. This repository does not redistribute medical
images, reference masks, pretrained weights, checkpoints, embedding caches, or
raw training logs. Obtain datasets and third-party model assets from the
sources cited in the manuscript and comply with their licenses and terms.

## Verification

From a clean clone, verify the retained matrix and rebuild a deterministic
release archive:

```bash
python paper/analysis/generate_medequiseg_factorial_manuscript_assets.py \
  --seed-metrics paper/results/medequiseg_factorial_public5_20260715/seed_metrics.csv \
  --check-only

python paper/analysis/build_protocol_v3_public5_release_package.py \
  --project-root . \
  --output output/submission/medequiseg_public5_reproducibility_release_20260725.zip
```

The builder verifies the whitelist, rejects obsolete configurations and backup
paths, checks all manifest hashes, and compares the ZIP payload byte-for-byte
with its verified staging tree.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite
the final journal article once bibliographic details are available.

Repository: <https://github.com/zjj30/MedEquiSeg>

## License

Project-authored software is released under the Apache License 2.0. Files
identified in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) retain their
original licenses. Dataset and model-weight licenses are not granted by this
repository.
