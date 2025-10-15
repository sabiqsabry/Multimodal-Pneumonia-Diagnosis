# Multimodal Pneumonia Diagnosis (CXR + EMR)

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Made for FYP](https://img.shields.io/badge/Masters-FYP-orange)](#) [![Repo](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/sabiqsabry/Multimodal-Pneumonia-Diagnosis)



A research-grade, end-to-end pipeline to detect pneumonia from chest X-rays (CXR) and structured electronic medical records (EMR) using multimodal deep learning. Built as part of my Masters Final Year Project (FYP).

## Key Features
- Multimodal fusion: CNN-based image encoder + MLP-based EMR encoder
- Trains and evaluates: CXR-only, EMR-only, and CXR+EMR fusion models
- Reproducible data processing for RSNA CXR and clinical EMR (synthetic + real)
- CLI scripts for training, evaluation, and prediction (batch and single)
- Explainability: Grad-CAM style visualizations and EMR feature attributions
- Detailed logs, figures, and checkpoints outputs

## Repository Structure
```
├── data/
│   ├── raw/                 # Place RSNA and EMR source data here (not versioned)
│   └── processed/           # Generated datasets, label maps, splits, scalers
├── outputs/
│   ├── checkpoints/         # Model weights (ignored)
│   ├── figures/             # Plots/visualizations for reports
│   └── logs/                # Training/eval logs
├── scripts/                 # CLI utilities for train/eval/predict/explain
└── src/                     # Core library: models, datamodules, preprocessing, utils
```

## Getting Started

### 1) Environment
- Python 3.13 (see `venv/` in this repo if you prefer a local venv)
- Install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Data Setup
- RSNA Pneumonia dataset paths expected under `data/raw/rsna/`
- EMR CSVs (synthetic and/or real) expected under `data/synthetic_emr/` and `data/raw/real_emr/`
- Run preprocessing to generate processed arrays/splits and scalers:

```bash
python scripts/prepare_data.py
```

### 3) Training
- CXR-only:
```bash
python scripts/train_cxr_only.py
```
- EMR-only:
```bash
python scripts/train_real_emr.py
```
- CXR+EMR fusion:
```bash
python scripts/train_cxr_real_emr.py
```

### 4) Evaluation
- Unified evaluation entrypoint:
```bash
python scripts/evaluate.py
```
- Modality-specific:
```bash
python scripts/evaluate_cxr_only.py
python scripts/evaluate_real_emr.py
python scripts/evaluate_cxr_real_emr.py
python scripts/evaluate_cxr_real_emr_weighted.py
```

### 5) Prediction
- Single case prediction:
```bash
python scripts/predict.py --image path/to/cxr.png --emr_csv path/to/row.csv
```
- Batch prediction over a directory/CSV:
```bash
python scripts/predict_all.py --images_dir data/raw/rsna/Test --emr_csv data/synthetic_emr/emr_data.csv
```
- See `PREDICTION_GUIDE.md`, `INTERACTIVE_PREDICTION_GUIDE.md`, and `LOGGING_PREDICTION_GUIDE.md` for more.

### 6) Explainability
- Simple demo:
```bash
python scripts/explain_simple.py
```
- Batch explanations:
```bash
python scripts/explain_batch.py
```
- Details in `EXPLAINABILITY_README.md`.

## Reproducibility & Logging
- Seeds and config are managed in `src/config.py`
- All runs log metrics to CSV under `outputs/logs/`
- Figures and reports are written under `outputs/figures/`

## Model Checkpoints
Checkpoints are saved to `outputs/checkpoints/` and are git-ignored. Sample names:
- `best_model.pt` (multimodal)
- `cxr_only_best.pt`
- `real_emr_best.pt`

## Notes on Data & Licensing
- RSNA dataset licensing applies; do not redistribute raw data via this repo
- Synthetic EMR is included only as CSV examples; real EMR should be handled securely

## Citation
If you use this code or build upon it, please cite:

```
Sabiq Sabry. Multimodal Pneumonia Diagnosis (CXR + EMR). Masters Final Year Project (FYP), 2025.
```

## Acknowledgements
- RSNA Pneumonia Detection Challenge organizers
- Open-source PyTorch and scientific Python community

---
Built for my Masters FYP.
