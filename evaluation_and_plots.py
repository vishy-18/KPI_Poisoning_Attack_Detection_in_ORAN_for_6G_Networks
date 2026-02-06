# evaluation_and_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *

def performance(y, score, pred, name):
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

def plot_roc(y, scores):
    plt.figure(figsize=(7,6))
    for name,score in scores.items():
        fpr,tpr,_ = roc_curve(y,score)
        plt.plot(fpr,tpr,label=name)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Comparison")
    plt.legend(); plt.grid()
    plt.show()
