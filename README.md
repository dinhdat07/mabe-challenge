# MABe Mouse Behavior Detection

Notebooks have been refactored into Python modules under `src/mabe` plus bash helpers in `../scripts`.

## Layout (relative to this README)
- `data/`: generators, labels.
- `features/`: feature engineering for single/pair tracks.
- `training/`: trainers for XGB, CNN (ResNet-1D), causal TCN, ensemble weight optimizer, and a section/mode train runner.
- `inference/`: inference for XGB/CNN/TCN/ensemble and a runner to assemble submissions per section/mode.
- `notebooks/`: original notebooks (EDA + models).
- `../scripts/`: bash scripts to train + infer each model.

## Requirements
- Python 3.10+, PyTorch, scikit-learn, optuna, joblib, pandas, numpy, koolbox, xgboost.
- Env vars for Kaggle data paths:
  - `TRAIN_CSV`, `TEST_CSV`
  - `TRAIN_ANNO` (train_annotation dir)
  - `TRAIN_TRACK` (train_tracking dir)
  - `TEST_TRACK` (test_tracking dir)
  - `MODEL_DIR` (output dir for the current script)

## Quick run (from repo root, Bash)
```bash
# XGB
export TRAIN_CSV=... TEST_CSV=... TRAIN_ANNO=... TRAIN_TRACK=... TEST_TRACK=...
bash scripts/run_xgb.sh

# CNN
export MODEL_DIR=models/cnn
bash scripts/run_cnn.sh

# TCN
export MODEL_DIR=models/tcn
bash scripts/run_tcn.sh

# Ensemble (needs thresholds/weights JSON and model_roots)
export THR_JSON=models/ensemble/thresholds.json
export WEIGHT_JSON=models/ensemble/weights.json
export MODEL_ROOTS_JSON='{"xgb":"models/xgb","cnn":"models/cnn","tcn":"models/tcn"}'
bash scripts/run_ensemble.sh
```
Scripts train per section/mode (single, pair) and write `submission_*.csv`.

## Key modules
- `training/train_runner.py`: builds features/labels per section/mode and calls the trainers; returns XGB thresholds and OOF blobs if needed for ensemble.
- `training/ensemble_trainer.py`: Optuna-based weight/threshold tuning from OOF of base models.
- `inference/runner.py`: `run_xgb_loop`, `run_cnn_loop`, `run_tcn_loop`, `run_ensemble_loop` to assemble final submissions.

## Notes
- Ensemble scripts assume you already have OOF and tuned thresholds/weights (they do not auto-generate OOF yet).
- CNN/TCN training is GPU-friendly; expect slowdowns on CPU.
