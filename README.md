# Amazon ML Challenge 2025 — Smart Product Pricing

This repository contains a complete, reproducible multimodal solution for the Amazon ML Challenge 2025 Smart Product Pricing task. The pipeline fuses textual product content, product images, and engineered numerical features; trains strong base models; and stacks them with a simple meta-learner to generate final predictions optimized for SMAPE.

## Highlights

- Multimodal fusion: CLIP text embeddings, CLIP image embeddings, and engineered numeric features.
- Base models: Fusion MLP (PyTorch), LightGBM, and XGBoost with Optuna tuning.
- Meta-learner: Ridge regression stacking of OOF predictions for robust generalization.
- Efficient preprocessing and caching; compatible with macOS MPS, CPU, and CUDA if available.
- Outputs a submission-ready `test_out.csv` in the required format.

## Repository Structure

- `preprocess.py` — Builds features from `catalog_content` (value/unit/IPQ, brand, content stats) and writes `preprocessed.parquet` combining train/test.
- `embed_text.py` — Generates CLIP text embeddings `text_embeddings.npy` from `content_clean`.
- `embed_images.py` — Generates CLIP image embeddings `image_embeddings.npy`; caches images in `cached_images/`.
- `train.py` — Trains Fusion MLP, LightGBM, XGBoost, and a Ridge meta-learner; saves artifacts and `test_out.csv`.
- `src/utils.py` — Utilities for bulk image download if you prefer local files.
- `dataset/` — Place `train.csv` and `test.csv` here.
- Artifacts — Saved during training: `pca_text.pkl`, `pca_img.pkl`, `scaler_text.pkl`, `scaler_img.pkl`, `scaler_num.pkl`, `tfidf_vectorizer.pkl`, `tfidf_svd.pkl`, `imputer.pkl`, `scaler.pkl`, `oof_mlp_log.pkl`, `oof_lgb_log.pkl`, `oof_xgb_log.pkl`, `test_preds_mlp_log.pkl`.

## Environment Setup

Use Python 3.10+ and a virtual environment.

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip

# Core dependencies
pip install pandas numpy scikit-learn lightgbm xgboost optuna tqdm pillow requests joblib pyarrow

# PyTorch (choose one that matches your OS/GPU)
pip install torch torchvision torchaudio  # standard CPU/MPS

# OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git
```

macOS with Apple Silicon (M1/M2/M3) automatically uses MPS if available; CUDA GPUs are also supported if your environment provides them.

## Data Preparation

- Place the challenge files in `dataset/train.csv` and `dataset/test.csv`.
- Ensure columns include `sample_id`, `catalog_content`, `image_link`, and `price` (train only).

## Quick Start

1) Preprocess train and test into a combined Parquet with engineered features:

```
python preprocess.py --train dataset/train.csv --test dataset/test.csv --out preprocessed.parquet
```

2) Generate CLIP text embeddings from `content_clean`:

```
python embed_text.py
```

3) Generate CLIP image embeddings from `image_link` (with caching):

```
python embed_images.py
```

4) Train base models, stack, and produce final predictions:

```
python train.py --trials 20 --mlp_epochs 12 --pca_text 128 --pca_img 128
```

Outputs:

- `test_out.csv` — submission file with `sample_id` and `price`.
- Model artifacts — PCA/scalers/TF-IDF SVD, OOF logs, meta-model inputs saved to disk.

## Approach Overview

- Text: CLIP `ViT-B/32` encodes `content_clean`; normalized rows; reduced via PCA; additional TF‑IDF bigram features reduced by TruncatedSVD for tree models.
- Images: CLIP `ViT-B/32` encodes images; normalized rows; reduced via PCA.
- Numeric features: IPQ, normalized value/unit, content length, word counts, ratios/log transforms, simple brand heuristic, target encoding for `brand` and `unit_norm` via OOF strategies.
- Base learners:
  - Fusion MLP over `[text_pca, image_pca, numeric]` blocks with KFold OOF.
  - LightGBM and XGBoost over dense features + `tfidf_svd` with Optuna tuning.
- Stacking: Concatenate OOF predictions and train a Ridge meta‑model; predict on test stacked features; clip to positive values; write `test_out.csv`.

## Evaluation

- Metric: SMAPE (Symmetric Mean Absolute Percentage Error). Training prints OOF and per‑fold SMAPE for MLP, LightGBM, and XGBoost.
- Submission: Ensure `test_out.csv` contains predictions for all test `sample_id`s.

## Reproducibility

- Seeds: `RANDOM_SEED=42` for NumPy and PyTorch. Optuna uses TPE with a fixed seed; minor nondeterminism may remain.
- Hardware: Results can differ across CPU/MPS/CUDA; use consistent environment for comparison.

## Troubleshooting

- CLIP install issues: Ensure `git` is available and re‑run `pip install git+https://github.com/openai/CLIP.git`.
- Image download throttling: `embed_images.py` caches files in `cached_images/`; you can also pre‑download with `src/utils.py`.
- MPS precision: The image encoder uses autocast on macOS; reduce `BATCH_SIZE` if you encounter memory errors.

## Acknowledgements

- Amazon ML Challenge 2025 organizers and dataset providers.
- OpenAI CLIP for text and image encoders.
- LightGBM and XGBoost libraries.
- Optuna for hyperparameter optimization.

## License

Project license is currently unspecified. Please set licensing terms before public distribution.
