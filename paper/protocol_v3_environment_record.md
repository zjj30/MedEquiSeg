# Public-5 Environment Record

Recorded: 2026-07-15

The training and evaluation launchers reference three project environments.
Versions below were read directly from those environments on the experiment
server. CUDA values are the versions reported by PyTorch; all benchmark GPUs
were NVIDIA GeForce RTX 4090 devices with driver 580.119.02.

## Principal MedEquiSeg and image-baseline environment

Launcher path placeholder: `<PROJECT_ROOT>/../envs/rmtfd/bin/python`

| Component | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA | 12.1 |
| cuDNN | 9.1.0 |
| torchvision | 0.20.1 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| Pillow | 12.2.0 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| MONAI | 1.5.1 |
| transformers | 4.36.2 |
| open-clip-torch | 3.3.0 |
| OpenCV | 4.13.0 |
| PyYAML | 6.0.3 |

## MedCLIPSeg and nnU-Net environment

Launcher path placeholder: `<PROJECT_ROOT>/../envs/sota_baselines/bin/python`

| Component | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA | 12.1 |
| torchvision | 0.20.1 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| Pillow | 12.2.0 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| MONAI | 1.3.0 |
| nnU-Net v2 | 2.8.0 |
| transformers | 4.36.2 |
| open-clip-torch | 3.3.0 |

## Official U-Mamba environment

Launcher path placeholder: `<PROJECT_ROOT>/../envs/umamba_py310/bin/python`

| Component | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA | 12.1 |
| torchvision | 0.20.1+cu121 |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| Pillow | 12.2.0 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| MONAI | 1.3.0 |
| nnU-Net v2 | 2.1.1 |
| transformers | 4.39.3 |

## Reproduction notes

- The public release archive records code and result checksums but does not
  redistribute third-party model weights or datasets.
- Public dataset download locations and model citations are given in the
  manuscript.
- Any cohort not listed among the five public benchmarks, together with its
  images, masks, paths, manifests, checkpoints, logs, and results, is excluded
  from the public archive.
