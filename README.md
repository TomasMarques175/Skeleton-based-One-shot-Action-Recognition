# Skeleton-based One-shot Action Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-shot-action-recognition-towards-novel/one-shot-3d-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/one-shot-3d-action-recognition-on-ntu-rgbd?p=one-shot-action-recognition-towards-novel)

[[Paper](https://arxiv.org/abs/2102.08997)] [[Supplementary video](https://drive.google.com/file/d/1NmY0vw78YwJ0ciKlUKwGU6Wrl9XVkTbO/view?usp=sharing)]

A PyTorch implementation for **one-shot and few-shot action recognition** from 3D skeleton sequences, based on the paper [One-Shot Action Recognition in Challenging Therapy Scenarios](https://arxiv.org/abs/2102.08997) (CVPR Workshops 2021). The model encodes skeleton pose sequences using a **Temporal Convolutional Network (TCN)** with triplet loss, enabling recognition of new action classes from a single example — with no retraining required.

---

## Contributors

| Name | GitHub |
|---|---|
| Tomás Marques | [@TomasMarques175](https://github.com/TomasMarques175) |

---

## Overview

The system is designed for **real-world therapy scenarios** where patients perform rehabilitation exercises that must be recognised on-the-fly from a single reference example. Given a query skeleton sequence, the model computes an embedding and compares it against one (or few) reference action embeddings using a distance metric — no softmax classifier or class-specific training is needed at inference time.

Key properties:
- **One-shot & few-shot** inference using cosine or Jensen-Shannon distance
- **Dynamic thresholding** support for robust detection
- **Online & offline** inference modes for real-time or batch evaluation
- Evaluated on both **NTU-120** (large-scale benchmark) and a **clinical therapy dataset**

---

## Architecture

### Temporal Convolutional Network (TCN)

The backbone is a **causal dilated TCN** with residual blocks, built from scratch in PyTorch:

```
Input: skeleton sequence (N, L, F)
  └─> EncoderTCN
        └─> TemporalConvNet
              └─> ResidualBlock × (nb_stacks × len(dilations))
                    ├─ CausalPad → Conv1d (dilated) → Norm → Activation → Dropout
                    ├─ CausalPad → Conv1d (dilated) → Norm → Activation → Dropout
                    └─ Residual skip connection (1×1 conv if channel mismatch)
  └─> Last time-step → L2-normalized embedding (N, nb_filters)
```

- **Causal padding** ensures no future frames leak into the current prediction — critical for online/real-time use
- **Dilations:** `[1, 2, 4, 8, 16, 32]` to capture multi-scale temporal patterns
- **Weight normalization** on convolutional layers for training stability
- Optional intermediate dense layer + final classification head for supervised pre-training with **triplet loss + cross-entropy**

### Skeleton Feature Representations

Raw 3D skeleton joints (25 joints × 3D) are pre-processed into rich feature vectors, configurable via flags:

| Feature | Description |
|---|---|
| `use_coords` | Torso-centred 3D joint coordinates |
| `use_coords_raw` | Raw (uncentred) 3D joint coordinates |
| `use_jcd_features` | Joint-centroid distance (JCD) — pairwise Euclidean distances |
| `use_jcd_diff` | Frame-to-frame JCD differences (speed of joint distances) |
| `use_speeds` | Per-joint 3D velocity vectors |
| `use_bone_angles` | Spherical bone angles (elevation + azimuth per bone) |
| `use_bone_angles_cent` | Centred spherical bone angles |

---

## Datasets

**[NTU RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/)** — large-scale 3D action recognition benchmark, 120 action classes. Store under `./datasets/NTU-120/raw_npy/`.

**[Therapy Dataset](https://doi.org/10.5281/zenodo.4700564)** — clinical rehabilitation exercise dataset with 14 gesture classes (e.g., *big*, *high*, *happy*, *waving*, *giving*, *pointing*). Store under `./datasets/therapies_dataset/`.

---

## Project Structure

```
├── models/
│   └── TCN_classifier.py        # PyTorch TCN encoder + classifier (EncoderTCN, TCN_clf)
├── pytorch_tcn.py               # TemporalConvNet and ResidualBlock implementations
├── pytorch_dataset.py           # PyTorch Dataset classes (TripletPoseDataset, TherapyDataset)
├── data_generator.py            # Skeleton feature extraction (JCD, angles, speeds, scaling)
├── train.py                     # Training loop (triplet + classification loss, TensorBoard)
├── prediction_utils.py          # Model loading and weight selection utilities
├── demo_therapies_benchmark.py  # One-shot / few-shot / dynamic threshold evaluation
├── demo_ntu_one_shot_benchmark.py # NTU-120 one-shot benchmark evaluation
├── demo_speed.py                # Online vs. offline inference speed benchmarking
├── curves_comparison.py         # Hyperparameter sweep and threshold optimisation
├── convert_skeletons_to_npy.py  # Raw skeleton data conversion to .npy format
└── requirements.txt             # Python dependencies
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** Python 3.6+, PyTorch, scikit-learn, scipy, pandas, numpy, keras-tcn (for Keras baseline comparison)

### Download Pre-trained Models

Store under `./pretrained_models/`:

- [NTU Benchmark model](https://unizares-my.sharepoint.com/:u:/g/personal/asabater_unizar_es/EamXVPDPFtFKtn1z26n5qhMBRMGWS8mDSXL-wfORQoHdLQ?e=YgLlcD)
- [Therapies model](https://unizares-my.sharepoint.com/:u:/g/personal/asabater_unizar_es/EVeQYXBP5dZNv0pD3-2485MB47RrMz7tA5KnfdJVnJLCqA?e=mNjK7C)

---

## Usage

### Training

```bash
python train.py
```

Training supports:
- **Triplet loss** for metric learning (embedding space)
- **Cross-entropy** for supervised classification pre-training
- **WeightedRandomSampler** for class imbalance
- **Mixed precision** (AMP) with `GradScaler`
- **TensorBoard** logging
- Left-right skeleton **data augmentation** via joint flip correspondences

### Evaluation — Therapy Dataset

```bash
# Evaluate using pre-computed best parameters
python demo_therapies_benchmark.py --path_model ./pretrained_models/therapies_model_7/

# Re-compute and store optimal parameters (threshold, distance metric)
python curves_comparison.py --path_model ./pretrained_models/therapies_model_7/ --force_all
```

Evaluation modes: **one-shot** (m=1), **few-shot** (m=3), **dynamic few-shot** (adaptive threshold).  
Distance metrics: cosine (`cos`), Jensen-Shannon (`js`).  
Reports per-class and overall **Precision / Recall / F1**.

### Evaluation — NTU-120 Benchmark

```bash
python demo_ntu_one_shot_benchmark.py --path_model ./pretrained_models/ntu_benchmark_model/
```

### Speed Benchmarking

```bash
# Therapy dataset — GPU, online + offline
python demo_speed.py --use_therapies --use_gpu --test_online --test_offline --max_clips 1000 \
    --path_model './pretrained_models/ntu_benchmark_model/' \
    --path_ntu_anns './ntu_annotations/one_shot_aux_set_full.txt'

# NTU-120 dataset — CPU only
python demo_speed.py --use_ntu --test_online --test_offline --max_clips 1000 \
    --path_model './pretrained_models/ntu_benchmark_model/' \
    --path_ntu_anns './ntu_annotations/one_shot_aux_set_full.txt'
```

---

## Citation

```bibtex
@InProceedings{Sabater_2021_CVPR,
    author    = {Sabater, Alberto and Santos, Laura and Santos-Victor, Jose and Bernardino, Alexandre and Montesano, Luis and Murillo, Ana C.},
    title     = {One-Shot Action Recognition in Challenging Therapy Scenarios},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {2777-2785}
}
```
