# ensemble_xapp.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def run_ensemble(X, y):
    X_clean = X[y==0]

    iso = IsolationForest(n_estimators=300, contamination=0.08, random_state=42)
    iso.fit(X_clean)
    iso_s = -iso.score_samples(X)

    lof = LocalOutlierFactor(n_neighbors=25, contamination=0.08, novelty=True)
    lof.fit(X_clean)
    lof_s = -lof.score_samples(X)

    X_lstm = X.reshape((-1,1,X.shape[1]))
    X_clean_lstm = X_clean.reshape((-1,1,X.shape[1]))

    lstm = Sequential([LSTM(64, activation="relu"), Dense(X.shape[1])])
    lstm.compile("adam","mse")
    lstm.fit(X_clean_lstm, X_clean, epochs=10, batch_size=128, verbose=0)

    recon = np.mean(np.abs(lstm.predict(X_lstm, verbose=0)-X), axis=1)

    def norm(x): return (x-x.min())/(x.max()-x.min())
    iso_n, lof_n, lstm_n = map(norm,[iso_s, lof_s, recon])

    ensemble = 0.35*iso_n + 0.40*lof_n + 0.25*lstm_n
    thr = np.percentile(ensemble[y==0], 92)

    return {
        "iso": iso_n, "lof": lof_n, "lstm": lstm_n,
        "ensemble": ensemble,
        "pred": (ensemble>thr).astype(int),
        "threshold": thr
    }
