# OptiFusionNet


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

**OptiFusionNet** is a deep learning model for **low-light image enhancement** using a novel two-resolution training strategy that combines meta-optimization (PSO + ACO) with a multi-stage training pipeline for superior image quality.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Problem Statement

Low-light images suffer from poor visibility, excessive noise, and color distortion. Traditional enhancement methods often fail to generalise across different lighting conditions or produce visually pleasant results. OptiFusionNet addresses this by:

1. **Automatically tuning hyperparameters** using bio-inspired meta-optimizers (PSO + ACO) so the model adapts to the dataset without manual trial-and-error.
2. **Training in two resolutions** — a fast 128×128 pretraining stage for speed, followed by a high-quality 256×256 stage for fidelity.
3. **Combining residual denoising blocks with additive skip connections** to simultaneously suppress noise and restore structure.

---

## Architecture

OptiFusionNet is a **U-Net-style encoder-decoder** with the following key components:

```
Input (Low-light Image)
        │
    ┌───▼───┐
    │ Enc-1 │  ConvBlock  → 64  ch
    └───┬───┘
    ┌───▼───┐
    │ Enc-2 │  MaxPool + ConvBlock → 128 ch
    └───┬───┘
    ┌───▼───┐
    │ Enc-3 │  MaxPool + ConvBlock → 256 ch
    └───┬───┘
    ┌───▼───┐
    │ Enc-4 │  MaxPool + ConvBlock → 512 ch
    └───┬───┘
    ┌───▼───┐
    │Bottlnk│  ConvBlock + 3×ResidualDenoiseBlock (512 ch)
    └───┬───┘
    ┌───▼───┐
    │ Dec-3 │  UpAddBlock + ResidualDenoiseBlock → 256 ch
    └───┬───┘
    ┌───▼───┐
    │ Dec-2 │  UpAddBlock + ResidualDenoiseBlock → 128 ch
    └───┬───┘
    ┌───▼───┐
    │ Dec-1 │  UpAddBlock + ResidualDenoiseBlock → 64  ch
    └───┬───┘
    ┌───▼───┐
    │ Fuse  │  Additive fusion with Enc-1 skip, FusionConv
    └───┬───┘
    ┌───▼───┐
    │Refiner│  ResidualDenoiseBlock → Residual add → clamp(0,1)
    └───────┘
      Output (Enhanced Image)
```

### Key Modules

| Module | Description |
|---|---|
| `ConvBlock` | Conv2D → BN → ReLU |
| `ResidualDenoiseBlock` | Identity-shortcut residual block for denoising |
| `UpAddBlock` | Bilinear upsample + additive skip-connection fusion |
| `HybridLoss` | L1 + gradient loss for perceptual sharpness |
| `PSO` | Particle Swarm Optimization for hyperparameter search |
| `ACO` | Ant Colony Optimization for hyperparameter search |

### Multi-Stage Training Pipeline

| Stage | Resolution | Optimizer | Purpose |
|---|---|---|---|
| 0 — Meta-opt | 128×128 (subset) | PSO + ACO | Find optimal lr / momentum / weight decay |
| 1 — Fast Pretrain | 128×128 | Adam | Quick convergence from scratch |
| 2 — HQ Adam | 256×256 | Adam (halved lr) | High-quality feature learning |
| 3 — SGD Fine-tune | 256×256 | SGD (momentum) | Final polishing with meta-opt momentum |

---

## Folder Structure

```
IPD_OPTIFUSIONNET/
├── notebooks/
│   └── OptifusionNet.ipynb       # Full Colab notebook — training, eval & inference
├── configs/
│   └── config.yaml               # All tuneable hyperparameters
├── data/
│   └── README.md                 # Dataset download & setup instructions
├── checkpoints/                  # Saved model weights (git-ignored)
├── results/                      # Enhanced output images (git-ignored)
├── requirements.txt              # Python dependencies
├── setup.sh                      # One-shot environment setup script
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (strongly recommended; CPU is supported but very slow)
- `pip` or `conda`

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/OptiFusionNet.git
cd OptiFusionNet

# 2. Install dependencies (CUDA 11.8 example — adjust for your CUDA version)
bash setup.sh

# Or manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Dataset

OptiFusionNet has been evaluated on:

- **LOL Dataset** — paired low/high-light images (`eval15` split for quick experiments, `our485` split for full training).
- **Custom ISRO PSR images** — unpaired dark images from planetary surface data.

See [`data/README.md`](data/README.md) for download links and the expected directory layout.

Update `configs/config.yaml` (or the `CONFIG` dict in the notebook) to point to your local dataset paths before training.

---

## Usage

### Training

```bash
# Using the notebook (recommended for Colab)
jupyter notebook notebooks/OptifusionNet.ipynb

# Using the Python scripts (local GPU)
python src/train.py --config configs/config.yaml
```

The training script will:
1. Run PSO + ACO meta-optimization on a small subset.
2. Perform fast Adam pretraining at 128×128.
3. Continue training at 256×256 with Adam.
4. Fine-tune with SGD using the meta-optimized momentum.
5. Save the best checkpoint to `checkpoints/optifusionnet_multistage_best.pth`.

### Inference

```python
from src.inference import enhance_folder

enhance_folder(
    model_path="checkpoints/optifusionnet_multistage_best.pth",
    input_folder="data/test/low",
    output_folder="results/enhanced",
)
```

Or run the standalone inference script:

```bash
python src/inference.py \
    --model checkpoints/optifusionnet_multistage_best.pth \
    --input  data/test/low \
    --output results/enhanced
```

### Evaluation

The notebook includes full evaluation with PSNR, SSIM, and BRISQUE metrics, plus side-by-side visualisation of input vs. enhanced images.

---

## Configuration

All hyperparameters live in `configs/config.yaml`:

```yaml
# Dataset
low_dir:  "data/lol_dataset/our485/low"
high_dir: "data/lol_dataset/our485/high"

# Resolutions
fast_size: 128
hq_size:   256

# Training schedule
fast_adam_epochs: 6
hq_adam_epochs:   20
sgd_epochs:       8
batch_size_fast:  12
batch_size_hq:    8

# Meta-optimization
use_meta_opt:   true
meta_subset:    300
pso_iters:      5
pso_particles:  6
pso_eval_epochs: 1
aco_iters:      3
aco_ants:       6

# Misc
seed:      42
save_path: "checkpoints/optifusionnet_multistage_best.pth"
```

---

## Results

| Dataset | PSNR (dB) | SSIM |
|---|---|---|
| LOL eval15 | ~30+ | ~0.85+ |
| LOL our485 | ~29+ | ~0.83+ |

*(Results are indicative and may vary with hardware/random seed.)*

---

## Future Improvements

> **Note:** The following are suggestions only — they have **not** been implemented in this repository.

- **Perceptual / VGG loss** — adding a feature-space loss could further improve texture fidelity.
- **Attention mechanisms** — channel attention (SE blocks) or spatial attention (CBAM) inside the bottleneck.
- **Larger training sets** — training on MIT-Adobe FiveK or SICE would improve generalisation.
- **Mixed-precision training** — `torch.cuda.amp` to reduce memory and speed up training.
- **Export to ONNX / TorchScript** — for deployment on edge devices.
- **Better SSIM implementation** — the current SSIM is per-sample rather than batched; a batched version would be faster.

---
