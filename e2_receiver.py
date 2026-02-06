# e2_receiver.py
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["load","latency","throughput","packet_loss","handover"]

def ingest_kpis(df):
    X = df[FEATURES].values
    y = df["label"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
