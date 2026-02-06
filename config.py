# config.py

import numpy as np
import tensorflow as tf

# ---------------- Reproducibility ----------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------- RAN Topology ----------------
NUM_CELLS = 12
TIME_STEPS = 1440           # 24h at 1-min granularity

# ---------------- KPI Set ----------------
FEATURES = [
    "load",
    "latency",
    "throughput",
    "packet_loss",
    "handover"
]

# ---------------- Attack Model ----------------
POISONED_CELLS = [4, 9]
ATTACK_WINDOWS = [(400, 650), (900, 1150)]

# Attack intensity
LOAD_BIAS = 1.08
THROUGHPUT_DROP = 0.90
LATENCY_DRIFT_MAX = 4.0
LOSS_SPIKE = 1.5

# ---------------- Ensemble Parameters ----------------
ISO_CONTAMINATION = 0.08
LOF_CONTAMINATION = 0.08
ENSEMBLE_WEIGHTS = {
    "if": 0.35,
    "lof": 0.40,
    "lstm": 0.25
}
ENSEMBLE_PERCENTILE = 92

# ---------------- LSTM Parameters ----------------
LSTM_UNITS = 64
LSTM_EPOCHS = 10
LSTM_BATCH = 128
