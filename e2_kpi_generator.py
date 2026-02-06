# e2_kpi_generator.py
import numpy as np
import pandas as pd
from config import *

def diurnal(t):
    return np.sin(2*np.pi*t/TIME_STEPS - np.pi/2) + 1

def generate_kpis(base):
    t = np.arange(TIME_STEPS)
    load = np.clip(base + 25*diurnal(t) + np.random.normal(0,3,TIME_STEPS), 15, 95)
    latency = np.clip(8 + 0.45*load + np.random.normal(0,2,TIME_STEPS), 5, 60)
    throughput = np.clip(110 - 0.7*load + np.random.normal(0,6,TIME_STEPS), 5, 120)
    loss = np.clip(0.015*load + np.random.normal(0,0.3,TIME_STEPS), 0, 6)
    ho = np.clip(100 - 0.05*load + np.random.normal(0,0.4,TIME_STEPS), 94, 100)
    return load, latency, throughput, loss, ho

def inject_poisoning(kpis):
    load, lat, thr, loss, ho = kpis
    for s,e in ATTACK_WINDOWS:
        load[s:e] *= 1.08
        lat[s:e] += np.linspace(0,4,e-s)
        thr[s:e] *= 0.9
        loss[s:e:20] += 1.5
    return load, lat, thr, loss, ho

def generate_dataset():
    rows = []
    for cell in range(NUM_CELLS):
        base = 55 if cell%4==0 else 35 if cell%5==0 else 45
        kpis = generate_kpis(base)
        if cell in POISONED_CELLS:
            kpis = inject_poisoning(kpis)

        for t in range(TIME_STEPS):
            label = int(
                cell in POISONED_CELLS and
                any(s<=t<=e for s,e in ATTACK_WINDOWS)
            )
            rows.append([t, cell, *[k[t] for k in kpis], label])

    return pd.DataFrame(rows, columns=[
        "time","cell","load","latency","throughput",
        "packet_loss","handover","label"
    ])
