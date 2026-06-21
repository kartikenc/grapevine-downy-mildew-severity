# Grapevine Downy Mildew Severity Estimation Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: Zenodo](https://img.shields.io/badge/Dataset-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.20701851)

A two-stage computational pipeline for automated severity estimation of grapevine downy mildew (*Plasmopara viticola*) from field-captured leaf imagery.

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Input Image    │────▶│  Stage 1: U-Net   │────▶│  Segmented Leaf    │
│  (1024×1024)    │     │  (ResNet-34)      │     │  Binary Mask       │
└─────────────────┘     └──────────────────┘     └────────┬───────────┘
                                                          │
                         ┌──────────────────┐             │
                         │  Stage 2: Feature │◀────────────┘
                         │  Extraction       │
                         │  (75 features)    │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐     ┌────────────────────┐
                         │  Random Forest   │────▶│  Severity Class    │
                         │  Classifier      │     │  S0–S4 (0–100%)    │
                         └──────────────────┘     └────────────────────┘
```

**Stage 1** — U-Net with pre-trained ResNet-34 encoder segments the leaf from complex vineyard backgrounds (Dice = 0.992).

**Stage 2** — Extracts 75 interpretable features (HSV color, GLCM texture, morphology) from the segmented region and classifies severity using a Random Forest ensemble (87.61% accuracy, κ_qw = 0.922).

## Key Results

| Metric | Value |
|--------|-------|
| Segmentation Dice | 0.992 |
| Classification Accuracy | 87.61 ± 2.24% |
| Balanced Accuracy | 85.09% |
| Quadratic-weighted κ | 0.922 |
| Spearman ρ | 0.942 |

## Dataset

The dataset is publicly available on Zenodo:

**DOI:** [10.5281/zenodo.20701851](https://doi.org/10.5281/zenodo.20701851)

- 920 field-captured images (1024×1024, JPEG)
- 5 severity classes (S0–S4) annotated by expert plant pathologists
- Polygon segmentation annotations (Pascal VOC XML)
- Class-balanced augmented version (1,750 images)
- Fixed train/val/test splits with provenance tracking

## Installation

```bash
git clone https://github.com/kartikenc/grapevine-downy-mildew-severity.git
cd grapevine-downy-mildew-severity
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- CUDA ≥ 12.1 (for GPU training)

## Usage

### 1. Data Preparation

```bash
python scripts/prepare_dataset.py --data_dir /path/to/images --output_dir /path/to/output
```

### 2. Train Segmentation Model (Stage 1)

```bash
python src/segmentation/train.py --data_dir /path/to/data --epochs 50 --batch_size 8
```

### 3. Run Full Pipeline (Stage 2)

```bash
python src/classification/hybrid_severity.py --model_path /path/to/best_unet_resnet34.pt --data_dir /path/to/data
```

### 4. CNN Baseline Comparison

```bash
python src/baseline/efficientnet_baseline.py --data_dir /path/to/data
```

## Repository Structure

```
├── configs/
│   └── config.yaml          # Hyperparameters and settings
├── scripts/
│   └── prepare_dataset.py   # Data splitting and augmentation
├── src/
│   ├── segmentation/
│   │   └── train.py          # U-Net (ResNet-34) training
│   ├── classification/
│   │   └── hybrid_severity.py # Feature extraction + RF classification
│   ├── baseline/
│   │   ├── hsv_baseline.py   # HSV-only severity estimation
│   │   └── efficientnet_baseline.py # End-to-end CNN comparison
│   ├── evaluation/
│   │   └── statistical_tests.py # McNemar's tests
│   └── utils/
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{cholachgudda2026automated,
  title={Automated severity grading of grapevine downy mildew in the field: a hybrid segmentation and feature-based machine learning approach},
  author={Cholachgudda, Kartik E. and Biradar, Rajashekhar C. and Kiran, B.M. and Prasannakumar, M.K.},
  journal={Journal of Agriculture and Food Research},
  year={2026},
  note={Submitted}
}

@misc{cholachgudda2026dataset,
  title={Grapevine Downy Mildew Severity Dataset},
  author={Cholachgudda, Kartik E. and Biradar, Rajashekhar C. and Kiran, B.M. and Prasannakumar, M.K.},
  year={2026},
  doi={10.5281/zenodo.20701851},
  publisher={Zenodo}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Authors

- **Kartik E. Cholachgudda** — RECA Labs, REVA University, Bangalore
- **Rajashekhar C. Biradar** — RECA Labs, REVA University, Bangalore
- **Kiran B.M.** — AgriHawk Technologies (Fyllo), Bengaluru
- **M.K. Prasannakumar** — PathoGenOmics Lab, UAS GKVK, Bangalore
