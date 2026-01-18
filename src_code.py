"""
=====================================================================
KPI POISONING ATTACK DETECTION USING ENSEMBLE ANOMALY DETECTION
Near-RT RIC | O-RAN | IEEE Research-Grade Implementation
=====================================================================

PART 1 : Realistic KPI Dataset Generation + Stealthy Attack Injection
PART 2 : Data Ingestion, Statistics & Feature Engineering
PART 3 : Individual Models, Ensemble & Performance Evaluation
PART 4 : Extensive Visualization & Diagnostic Analysis

This file is intentionally verbose and diagnostic-rich.
=====================================================================
"""

# =====================================================================
# PART 0 — IMPORTS & GLOBAL CONFIGURATION
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

np.random.seed(42)
tf.random.set_seed(42)

# =====================================================================
# PART 1 — DATA GENERATION (REALISTIC KPIs + ALL ATTACK TYPES)
# =====================================================================

NUM_CELLS = 12
TIME_STEPS = 1440
POISONED_CELLS = [4, 9]
ATTACK_WINDOWS = [(400, 650), (900, 1150)]

def diurnal(t):
    return np.sin(2*np.pi*t/TIME_STEPS - np.pi/2) + 1

def generate_kpis(base):
    t = np.arange(TIME_STEPS)

    load = base + 25*diurnal(t) + np.random.normal(0,3,TIME_STEPS)
    load = np.clip(load, 15, 95)

    latency = 8 + 0.45*load + np.random.normal(0,2,TIME_STEPS)
    latency = np.clip(latency, 5, 60)

    throughput = 110 - 0.7*load + np.random.normal(0,6,TIME_STEPS)
    throughput = np.clip(throughput, 5, 120)

    loss = 0.015*load + np.random.normal(0,0.3,TIME_STEPS)
    loss = np.clip(loss, 0, 6)

    ho = 100 - 0.05*load + np.random.normal(0,0.4,TIME_STEPS)
    ho = np.clip(ho, 94, 100)

    return load, latency, throughput, loss, ho

def inject_attacks(kpis):
    load, lat, thr, loss, ho = kpis
    for (s,e) in ATTACK_WINDOWS:
        # Linear bias
        load[s:e] *= 1.08
        # Temporal drift
        lat[s:e] += np.linspace(0,4,e-s)
        # Correlated KPI poisoning
        thr[s:e] *= 0.9
        # On–off packet loss
        loss[s:e:20] += 1.5
    return load, lat, thr, loss, ho

def label(cell, t):
    if cell in POISONED_CELLS:
        for (s,e) in ATTACK_WINDOWS:
            if s <= t <= e:
                return 1
    return 0

rows = []
for cell in range(NUM_CELLS):
    base = 55 if cell%4==0 else 35 if cell%5==0 else 45
    kpis = generate_kpis(base)
    if cell in POISONED_CELLS:
        kpis = inject_attacks(kpis)

    for t in range(TIME_STEPS):
        rows.append([
            t, cell,
            kpis[0][t], kpis[1][t], kpis[2][t],
            kpis[3][t], kpis[4][t],
            label(cell,t)
        ])

df = pd.DataFrame(rows, columns=[
    "time","cell","load","latency",
    "throughput","packet_loss","handover","label"
])

print("DATASET SUMMARY")
print(df.describe())
print("Poisoned samples:", df.label.sum())

# =====================================================================
# PART 2 — INGESTION, FEATURE ENGINEERING & DATA STATISTICS
# =====================================================================

FEATURES = ["load","latency","throughput","packet_loss","handover"]

X = df[FEATURES].values
y = df["label"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_clean = X_scaled[y==0]

print("\nCLASS DISTRIBUTION")
print(pd.Series(y).value_counts(normalize=True))

# =====================================================================
# PART 3 — MODELS, ENSEMBLE & FULL PERFORMANCE EVALUATION
# =====================================================================

# ---------------- Isolation Forest ----------------
iso = IsolationForest(
    n_estimators=300,
    contamination=0.08,
    random_state=42
)
iso.fit(X_clean)
iso_score = -iso.score_samples(X_scaled)

# ---------------- LOF ----------------
lof = LocalOutlierFactor(
    n_neighbors=25,
    contamination=0.08,
    novelty=True
)
lof.fit(X_clean)
lof_score = -lof.score_samples(X_scaled)

# ---------------- LSTM Autoencoder ----------------
X_lstm = X_scaled.reshape((-1,1,X_scaled.shape[1]))
X_lstm_clean = X_clean.reshape((-1,1,X_clean.shape[1]))

lstm = Sequential([
    LSTM(64, activation="relu"),
    Dense(X_scaled.shape[1])
])
lstm.compile("adam","mse")
lstm.fit(X_lstm_clean, X_clean, epochs=10, batch_size=128, verbose=0)

recon = np.mean(np.abs(lstm.predict(X_lstm, verbose=0)-X_scaled), axis=1)

# ---------------- Normalization ----------------
def norm(x): return (x-x.min())/(x.max()-x.min())
iso_n, lof_n, lstm_n = map(norm,[iso_score, lof_score, recon])

# ---------------- Ensemble ----------------
ensemble_score = 0.35*iso_n + 0.40*lof_n + 0.25*lstm_n
thr = np.percentile(ensemble_score[y==0], 92)

iso_pred = (iso_n>thr).astype(int)
lof_pred = (lof_n>thr).astype(int)
lstm_pred = (lstm_n>thr).astype(int)
ens_pred = (ensemble_score>thr).astype(int)

def metrics(name, pred, score):
    tn,fp,fn,tp = confusion_matrix(y,pred).ravel()
    return {
        "Model": name,
        "Accuracy": accuracy_score(y,pred),
        "Precision": precision_score(y,pred,zero_division=0),
        "Recall": recall_score(y,pred),
        "F1": f1_score(y,pred),
        "FPR": fp/(fp+tn),
        "FNR": fn/(fn+tp),
        "AUC": roc_auc_score(y,score)
    }

results = pd.DataFrame([
    metrics("Isolation Forest", iso_pred, iso_n),
    metrics("LOF", lof_pred, lof_n),
    metrics("LSTM", lstm_pred, lstm_n),
    metrics("Ensemble", ens_pred, ensemble_score)
])

print("\nMODEL PERFORMANCE")
print(results)

# =====================================================================
# PART 4 — EXTENSIVE VISUALIZATION & DIAGNOSTICS
# =====================================================================

# ---- Time-series with attack windows ----
plt.figure(figsize=(12,4))
cell = POISONED_CELLS[0]
plt.plot(df[df.cell==cell].time, df[df.cell==cell].load, label="Load")
for (s,e) in ATTACK_WINDOWS:
    plt.axvspan(s,e,color='red',alpha=0.2)
plt.title("Traffic Load with Attack Windows")
plt.legend(); plt.grid(); plt.show()

# ---- Temporal Drift ----
plt.figure(figsize=(10,4))
plt.plot(df[df.cell==cell].latency)
plt.title("Latency Drift under KPI Poisoning")
plt.grid(); plt.show()

# ---- Cross-KPI Relationship ----
plt.figure(figsize=(6,5))
plt.scatter(df.load, df.latency, c=df.label, alpha=0.3)
plt.xlabel("Load"); plt.ylabel("Latency")
plt.title("Cross-KPI Semantic Violation")
plt.grid(); plt.show()

# ---- KPI Distributions ----
for kpi in FEATURES:
    plt.figure(figsize=(5,3))
    sns.histplot(df[df.label==0][kpi], label="Normal", kde=True)
    sns.histplot(df[df.label==1][kpi], label="Poisoned", kde=True)
    plt.title(f"{kpi} Distribution")
    plt.legend(); plt.show()

# ---- ROC Curves ----
plt.figure(figsize=(7,6))
for name,score in {
    "IF":iso_n, "LOF":lof_n,
    "LSTM":lstm_n, "ENS":ensemble_score
}.items():
    fpr,tpr,_ = roc_curve(y,score)
    plt.plot(fpr,tpr,label=name)
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Comparison")
plt.legend(); plt.grid(); plt.show()

# ---- Confusion Matrix (Ensemble) ----
cm = confusion_matrix(y, ens_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Ensemble Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# ---- Event-level Detection ----
def event_level(pred):
    total=detected=0
    for c in POISONED_CELLS:
        for (s,e) in ATTACK_WINDOWS:
            total+=1
            if np.any(pred[(df.cell==c)&(df.time>=s)&(df.time<=e)]):
                detected+=1
    return detected/total

print("\nEVENT-LEVEL DETECTION RATE")
print("IF :",event_level(iso_pred))
print("LOF:",event_level(lof_pred))
print("LSTM:",event_level(lstm_pred))
print("ENS:",event_level(ens_pred))
