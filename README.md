# MABe Mouse Behavior Detection

[![Kaggle](https://img.shields.io/badge/Kaggle-MABe%20Challenge-20BEFF?style=for-the-badge\&logo=kaggle\&logoColor=white)](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection)
![Medal](https://img.shields.io/badge/Medal-Bronze-CD7F32?style=for-the-badge)
![Ranking](https://img.shields.io/badge/Ranking-Top%209%25-blue?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Multi--Agent%20Behavior%20Detection-purple?style=for-the-badge)

---

## Overview

This repository presents a **Bronze Medal (Top 9%) solution** for the Kaggle **MABe Mouse Behavior Detection Challenge**.

The objective is to classify **30+ social and non-social behaviors** from multi-agent pose estimation data, requiring robust modeling of **temporal dynamics**, **agent interactions**, and **cross-domain generalization** across different experimental settings.

---

## Problem Characteristics

![Multi-Agent](https://img.shields.io/badge/Setting-Multi--Agent-black?style=flat-square)
![Temporal](https://img.shields.io/badge/Data-Sequential%20Time%20Series-blue?style=flat-square)
![Challenge](https://img.shields.io/badge/Challenge-Cross--Lab%20Generalization-orange?style=flat-square)

* Multi-agent interaction modeling (pairwise + group behaviors)
* Strong temporal dependencies across sequences
* Distribution shift across different labs and environments
* Highly imbalanced behavior classes

---

## Approach

![Models](https://img.shields.io/badge/Models-XGBoost%20%7C%20ResNet1D%20%7C%20TCN-black?style=flat-square)
![Optimization](https://img.shields.io/badge/Optimization-Optuna-green?style=flat-square)

## Pipeline

### Feature Engineering

* Extracted **spatial features** from keypoints
* Built **pairwise interaction features** between agents
* Designed **temporal features** (velocity, motion patterns, windows)

### Modeling

* **XGBoost** baseline with per-behavior binary classification
* **ResNet-1D CNN** on windowed sequences
* **Causal Temporal Convolutional Network (TCN)** for sequence modeling

### Ensemble Strategy

* Combined predictions across three XGBoost models with different hyperparameters and seeds
* Optimized **weights and thresholds using Optuna**
* Improved robustness across sections and modes (single / pair)

---

## Results

* Bronze Medal on Kaggle leaderboard
* Top **9%** overall ranking
* Built a robust pipeline for **multi-agent behavior recognition**

---

## Repository Structure

```
data/        # generators, labels
features/    # feature engineering (single / pair)
training/    # model trainers (XGB, CNN, TCN, ensemble)
inference/   # inference pipelines and submission builders
notebooks/   # EDA and modeling notebooks
scripts/     # training & inference bash scripts
```

---

## Key Components

* `training/train_runner.py`
  Orchestrates feature building, label preparation, and model training

* `training/ensemble_trainer.py`
  Performs Optuna-based optimization for ensemble weights and thresholds

* `inference/runner.py`
  Runs inference loops and assembles final submission outputs

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square\&logo=pytorch\&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC0000?style=flat-square)
![Optuna](https://img.shields.io/badge/Optuna-3C3C3C?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square\&logo=numpy\&logoColor=white)

---

## Requirements

* Python 3.10+
* PyTorch, scikit-learn, optuna, joblib
* pandas, numpy, xgboost

Environment variables:

```
TRAIN_CSV
TEST_CSV
TRAIN_ANNO
TRAIN_TRACK
TEST_TRACK
MODEL_DIR
```

---

## Quick Run

```bash
# XGBoost
bash scripts/run_xgb.sh

# CNN
export MODEL_DIR=models/cnn
bash scripts/run_cnn.sh

# TCN
export MODEL_DIR=models/tcn
bash scripts/run_tcn.sh

# Ensemble
export THR_JSON=models/ensemble/thresholds.json
export WEIGHT_JSON=models/ensemble/weights.json
export MODEL_ROOTS_JSON='{"xgb":"models/xgb","cnn":"models/cnn","tcn":"models/tcn"}'
bash scripts/run_ensemble.sh
```

---

## Notes

* Ensemble requires precomputed OOF predictions
* CNN/TCN training is significantly faster on GPU
* Thresholds and weights must be tuned before final submission

---

## Summary

This project demonstrates a complete pipeline for:

* Multi-agent sequence modeling
* Feature engineering for structured time-series data
* Ensemble optimization for competitive ML performance
* Robust handling of distribution shift in real-world datasets
