# utils.py

import numpy as np
from sklearn.metrics import confusion_matrix

# ---------------- Normalization ----------------
def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

# ---------------- Event-level Detection ----------------
def event_level_detection(df, pred, poisoned_cells, attack_windows):
    total = detected = 0
    for cell in poisoned_cells:
        for (s, e) in attack_windows:
            total += 1
            if np.any(pred[(df.cell == cell) &
                            (df.time >= s) &
                            (df.time <= e)]):
                detected += 1
    return detected / total if total > 0 else 0

# ---------------- Error Rates ----------------
def error_rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "FPR": fp / (fp + tn + 1e-8),
        "FNR": fn / (fn + tp + 1e-8)
    }
