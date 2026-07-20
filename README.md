# MedEquiSeg

Audited source, public-benchmark results, and manuscript artifacts for:

> **MedEquiSeg: Shared Augmentation and Privileged-Prompt Reliability in
> Multimodal Medical Image Segmentation**

Authors: Xianjun Ye, Jianjun Zhu, Mengmeng Zhang, Ye Zhang, Yu Kang, and
Yongsheng Luo. Xianjun Ye and Jianjun Zhu contributed equally. Yu Kang and
Yongsheng Luo are the corresponding authors.

This repository accompanies the Public-5 submission route. It contains only
public, non-identifying protocol and result artifacts. It is research software
and is not intended for clinical use.

The sanitized manifests publish the case-level text prompts for the five public
benchmarks. No private-cohort image, mask, prompt, patient metadata, source
path, checkpoint, or result artifact is included.

## Scientific scope

The release covers BUSI, ClinicDB, BUS-BRA, BRISC, and COVID-19. Its principal
evidence is a retrospective ordered ablation with six configurations, five
datasets, and three seeds (90 training cells), plus four frozen-checkpoint
prompt controls per cell.

The richer R11 shared sampler is a separately frozen training-policy amendment
on the fixed Protocol V3 data and evaluation specification. Runtime hooks found
two forward-active projector ATConv replacements and two dormant legacy neck
members. The manuscript therefore describes the effective two-projector graph.

The shared-plan no-rewrite control receives the same image-mask transforms as
the rewrite condition. The results support a reliability-audited benchmark;
they do not establish clinical-report understanding or deployment readiness.

## Repository contents

- `smoke_tests/`: protocol, augmentation, training, prediction, and evaluation code.
- `paper/results/`: public-only aggregate, seed-level, control, boundary, and audit outputs.
- `paper/analysis/`: deterministic analysis and manuscript-asset generators.
- `paper/revision/`: lineage, ATConv, and stratified-derangement audit code and reports.
- `paper/latex/bmc_work_public5/`: BMC Medical Imaging LaTeX source.
- `output/pdf/`: compiled Public-5 manuscript and supplement.
- `release_manifest.csv`: byte counts and SHA-256 values for the audited package payload.

For the complete scope and exclusions, see
[`paper/protocol_v3_public5_release_readme.md`](paper/protocol_v3_public5_release_readme.md).

## Environment

The principal environment used Python 3.11.15, PyTorch 2.5.1+cu121, MONAI
1.5.1, transformers 4.36.2, and open-clip-torch 3.3.0 on NVIDIA RTX 4090 GPUs.
The exact recorded environments for the principal, MedCLIPSeg/nnU-Net, and
U-Mamba paths are listed in
[`paper/protocol_v3_environment_record.md`](paper/protocol_v3_environment_record.md).

The release does not redistribute datasets, pretrained weights, checkpoints,
embedding caches, or raw training logs. Obtain the public datasets and
third-party model assets from the sources cited in the manuscript.

## Verification

The corresponding Public-5 ZIP contains 172 manifest entries, is 6,049,081
bytes, and has SHA-256:

```text
7f2ee1ce54350f465ffe6ca395b05d45e56a741ac0e3719a017a184191ac2b44
```

Run the included finalizer tests with an environment containing PyYAML:

```bash
python paper/analysis/test_finalize_public5_submission.py -v
```

The file-level release manifest can be checked independently by recomputing the
SHA-256 digest and byte count for every listed relative path. The source-identity
boundary and the non-bitwise-reconstruction limitation are documented in
[`paper/protocol_v3_public5_source_identity_20260720.md`](paper/protocol_v3_public5_source_identity_20260720.md).

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite
the final journal article once bibliographic details are available.

Repository: <https://github.com/zjj30/MedEquiSeg>

## License

Project-authored software is released under the Apache License 2.0. Files
identified in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) retain their
original licenses. Dataset and model-weight licenses are not granted by this
repository.
