from e2_kpi_generator import generate_dataset
from e2_receiver import ingest_kpis
from ensemble_xapp import run_ensemble
from evaluation_and_plots import *

df = generate_dataset()
X, y, _ = ingest_kpis(df)
out = run_ensemble(X, y)

results = [
    performance(y, out["ensemble"], out["pred"], "Ensemble")
]
print(results)

plot_roc(y, {
    "Ensemble": out["ensemble"]
})
