<div align="center">

<h2><nobr>Faster-WAM: Efficient Inference-Time Future Conditioning</nobr><br><nobr>for Robust World Action Models</nobr></h2>

<b>Weiheng Zhao</b><sup>1</sup> &middot; <b>Haoyi Jiang</b><sup>1</sup> &middot; <b>Xin Shi</b><sup>2</sup> &middot; <b>Liu Liu</b><sup>3</sup> &middot; <b>Zhizhong Su</b><sup>3</sup> &middot; <b>Wei Sui</b><sup>2</sup> &middot; <b>Fan Huang</b><sup>4</sup> &middot; <b>Xinggang Wang</b><sup>1</sup>

Huazhong University of Science and Technology<sup>1</sup> &middot; D-Robotics<sup>2</sup> &middot; Horizon Robotics<sup>3</sup> &middot; Xiamen University<sup>4</sup>

<a href="https://arxiv.org/pdf/2608.04404"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper arXiv"></a>

</div>

The key insight behind **Faster-WAM** is that future representations are not merely an auxiliary training signal, but essential inference-time context for robust action prediction under distribution shifts. Guided by this principle, Faster-WAM computes future representations once and selectively reuses them during action denoising, reducing redundant video-action interaction. It achieves state-of-the-art in-distribution performance and robust OOD generalization across simulated and real-world manipulation, while substantially reducing inference latency.

<div align="center">
  <img src="assets/framework_r.png" alt="Faster-WAM framework" width="85%">
</div>

---

## Index

- [Release](#release)
- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Dataset Download](#dataset-download)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgments](#acknowledgments)

## Release

Training and inference code ✅

LIBERO, LIBERO-Plus, and RoboTwin evaluation code ✅

Model checkpoints

## File Structure

```text
FasterWAM/
├── configs/
│   ├── data/                 # LIBERO and RoboTwin dataset configs
│   ├── model/                # FastWAM, JointWAM, and FasterWAM models
│   ├── task/                 # Benchmark-specific training configs
│   ├── sim_libero.yaml       # LIBERO evaluation defaults
│   ├── sim_libero_plus.yaml  # LIBERO-Plus evaluation defaults
│   └── sim_robotwin.yaml     # RoboTwin evaluation defaults
├── environments/             # Independent benchmark uv projects and locks
├── scripts/
│   ├── train.py
│   ├── train_zero1.sh
│   ├── preprocess_sparse_action_dit_backbone.py
│   ├── precompute_text_embeds.py
│   └── eval_fasterwam_*.sh
├── experiments/
│   ├── libero/               # LIBERO evaluation manager and worker
│   └── robotwin/             # RoboTwin evaluation manager and policy adapter
├── src/fasterwam/            # Core model, dataset, and training code
├── third_party/RoboTwin/     # RoboTwin evaluation integration
├── checkpoints/              # Wan components and model checkpoints
├── data/                     # Preprocessed training datasets
├── runs/                     # Training outputs
└── evaluate_results/         # Evaluation outputs
```

The final system is `FasterWAM`. The repository also keeps the `FastWAM` and
`JointWAM` baselines for controlled comparisons.

## Environment Setup

Run all commands below from the repository root.

Install [uv](https://docs.astral.sh/uv/) first. The core training environment
uses Python 3.10 and the PyTorch 2.7.1 CUDA 12.8 wheels locked in `uv.lock`:

```bash
bash scripts/setup/install_core.sh
source .venv/bin/activate
```

## Model Preparation

FasterWAM uses Wan2.2-TI2V-5B. By default, missing components are downloaded
from Hugging Face and stored under `./checkpoints`. Set the directory explicitly
before model preparation, training, or evaluation:

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

To use ModelScope instead, additionally set:

```bash
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
```

### FasterWAM ActionDiT initialization

Before training FasterWAM from scratch, generate its SparseActionDiT
initialization from the Wan2.2 video DiT:

```bash
python scripts/preprocess_sparse_action_dit_backbone.py \
  --model-config configs/model/fasterwam.yaml \
  --output checkpoints/SparseActionDiT_cond_0_4_8_12_16_20_24_28_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

### Baseline ActionDiT initialization

FastWAM and JointWAM use the dense ActionDiT initialization:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## Dataset Download

### LIBERO

FasterWAM uses the same preprocessed MuJoCo 3.3.2 LIBERO dataset as FastWAM:

- [yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)

Download the four archives and extract them under `data/libero_mujoco3.3.2`:

```bash
mkdir -p data/libero_mujoco3.3.2

huggingface-cli download yuanty/LIBERO-fastwam \
  --repo-type dataset \
  --local-dir data/libero_mujoco3.3.2

cd data/libero_mujoco3.3.2
for f in *.tar.gz; do tar -xzf "$f"; done
cd ../..
```

The resulting layout must be:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin

The preprocessed RoboTwin dataset is available from:

- [yuanty/robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

Download all split archives, concatenate them, and extract them as described in
the FastWAM release:

```bash
mkdir -p data/robotwin2.0

huggingface-cli download yuanty/robotwin2.0-fastwam \
  --repo-type dataset \
  --local-dir data/robotwin2.0

cd data/robotwin2.0
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd ../..
```

The expected layout is:

```text
data/robotwin2.0/
├── dataset_stats.json
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

## Training

### 1. Precompute instruction embeddings

Training reads cached T5 instruction embeddings. Generate them once after the
dataset has been extracted:

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_fasterwam_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_fasterwam_3cam_384_1e-4
```

For multi-GPU preprocessing:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_text_embeds.py \
  task=libero_fasterwam_2cam224_1e-4
```

The caches are written to `data/text_embeds_cache_fasterwam/libero` and
`data/text_embeds_cache_fasterwam/robotwin`.

### 2. Launch training

```bash
NPROC_PER_NODE=8 bash scripts/train_fasterwam_libero.sh

NPROC_PER_NODE=8 bash scripts/train_fasterwam_robotwin.sh
```

Both wrappers accept additional Hydra overrides. For example:

```bash
NPROC_PER_NODE=8 bash scripts/train_fasterwam_libero.sh \
  batch_size=8 \
  num_epochs=1 \
  wandb.enabled=true
```

## Evaluation

LIBERO, LIBERO-Plus, and RoboTwin are managed in three separate uv environments
to isolate their simulator dependencies. Run the corresponding setup command
once before evaluation; each evaluation launcher automatically uses the matching
environment.

### LIBERO

```bash
bash scripts/setup/install_libero.sh

TASK_NAME=libero_fasterwam_2cam224_1e-4 \
CKPT_PATH=checkpoints/fasterwam_release/libero/step_021700.pt \
DATASET_STATS_PATH=checkpoints/fasterwam_release/libero/dataset_stats.json \
NUM_GPUS=8 \
bash scripts/eval_fasterwam_libero.sh
```

### LIBERO-Plus

```bash
bash scripts/setup/install_libero_plus.sh

TASK_NAME=libero_fasterwam_2cam224_1e-4 \
CKPT_PATH=checkpoints/fasterwam_release/libero/step_021700.pt \
DATASET_STATS_PATH=checkpoints/fasterwam_release/libero/dataset_stats.json \
NUM_GPUS=8 \
bash scripts/eval_fasterwam_libero_plus.sh
```

### RoboTwin

```bash
bash scripts/setup/install_robotwin.sh

TASK_NAME=robotwin_fasterwam_3cam_384_1e-4 \
CKPT_PATH=checkpoints/fasterwam_release/robotwin/step_029355.pt \
DATASET_STATS_PATH=checkpoints/fasterwam_release/robotwin/dataset_stats.json \
NUM_GPUS=8 \
bash scripts/eval_fasterwam_robotwin.sh
```

## Acknowledgments

Our codebase is built upon:

- FastWAM: https://github.com/yuantianyuan01/FastWAM
- Wan2.2: https://github.com/Wan-Video/Wan2.2
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- LIBERO-Plus: https://github.com/sylvestf/LIBERO-plus
- RoboTwin: https://github.com/RoboTwin-Platform/RoboTwin

We thank these teams for contributing their impressive code and models to the
community.
