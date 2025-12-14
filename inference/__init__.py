"""Inference runners for XGB, CNN, TCN, and ensemble models."""

from mabe.inference.xgb import submit as submit_xgb
from mabe.inference.cnn import submit as submit_cnn
from mabe.inference.tcn import submit as submit_tcn
from mabe.inference.ensemble import submit as submit_ensemble
