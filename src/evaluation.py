"""Metrics, threshold selection, risk tiers and evaluation plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

RISK_ORDER = ["Low", "Medium", "High"]


def classification_metrics(y, p, threshold) -> dict:
    y = np.asarray(y); pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "threshold": float(threshold), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "accuracy": (tp + tn) / len(y),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "base_rate": float(y.mean())}


def best_f1_threshold(y, p) -> float:
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    return float(th[np.argmax(f1)])


def risk_thresholds(y, p, min_lift=2.0, max_low_churn=0.15, min_share=0.03, return_info=False):
    """
    Risk tiers anchored on VALIDATION outcomes (not on arbitrary 30/70 cut-offs):
      High   : lowest score threshold at which the flagged group churns at >= `min_lift` x the base rate
      Low    : highest score threshold below which <= `max_low_churn` of customers really churned
      Medium : in between
    """
    y = np.asarray(y); p = np.asarray(p); n = len(y); floor = max(30, int(min_share * n))
    min_precision = min(0.9, min_lift * y.mean())
    o = np.argsort(-p); ys, ps = y[o], p[o]
    prec = np.cumsum(ys) / np.arange(1, n + 1)
    ok = np.where((prec >= min_precision) & (np.arange(1, n + 1) >= floor))[0]
    high_fb = not len(ok)
    t_high = float(ps[ok.max()]) if len(ok) else float(np.quantile(p, .90))
    oa = np.argsort(p); ya, pa = y[oa], p[oa]
    rate = np.cumsum(ya) / np.arange(1, n + 1)
    ok = np.where((rate <= max_low_churn) & (np.arange(1, n + 1) >= floor))[0]
    low_fb = not len(ok)
    t_low = float(pa[ok.max()]) if len(ok) else float(np.quantile(p, .33))
    crossed = t_low >= t_high
    if crossed:
        t_low, t_high = float(np.quantile(p, .33)), float(np.quantile(p, .67))
    if return_info:
        return t_low, t_high, {"high_fallback_to_quantile": high_fb, "low_fallback_to_quantile": low_fb, "thresholds_crossed_used_terciles": crossed,
                               "required_high_churn_rate": float(min_precision)}
    return t_low, t_high


def assign_risk(p, t_low, t_high) -> pd.Categorical:
    p = np.asarray(p)
    lab = np.where(p >= t_high, "High", np.where(p <= t_low, "Low", "Medium"))
    return pd.Categorical(lab, categories=RISK_ORDER, ordered=True)


def risk_table(y, p, t_low, t_high) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "risk": assign_risk(p, t_low, t_high)})
    t = d.groupby("risk", observed=False).agg(customers=("y", "size"), observed_churn_rate=("y", "mean"),
                                              mean_predicted_prob=("p", "mean"), churners=("y", "sum"))
    t["share_of_customers"] = t.customers / t.customers.sum()
    t["share_of_all_churners"] = t.churners / t.churners.sum()
    return t


def decile_lift(y, p, bins=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), bins, labels=range(1, bins + 1))
    t = d.groupby("decile", observed=True).agg(customers=("y", "size"), churn_rate=("y", "mean"))
    t["lift"] = t.churn_rate / d.y.mean()
    return t


# ----------------------------- plots -----------------------------
def plot_roc_pr(y, probs: dict):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p); ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p); ax[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(np.mean(y), color="k", ls=":", lw=1, label=f"no-skill (base rate {np.mean(y):.2f})")
    ax[1].set(title="Precision-Recall curve (test)", xlabel="Recall", ylabel="Precision")
    for a in ax: a.legend(fontsize=8)
    fig.tight_layout(); return fig


def plot_confusion(m: dict, title="Confusion matrix (test)"):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                                                color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"],
           xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout(); return fig


def plot_calibration(y, probs: dict, bins=10):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for name, p in probs.items():
        fy, fp = calibration_curve(y, p, n_bins=bins, strategy="quantile"); ax.plot(fp, fy, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set(title="Calibration (test)", xlabel="Predicted probability", ylabel="Observed churn rate")
    ax.legend(fontsize=8); fig.tight_layout(); return fig


def plot_risk_tiers(tab: pd.DataFrame, title="Observed churn rate by risk tier (test)"):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    colors = ["#59a14f", "#edc948", "#e15759"]
    bars = ax.bar(tab.index.astype(str), tab.observed_churn_rate, color=colors)
    for b, n in zip(bars, tab.customers):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .01, f"{b.get_height():.0%}\n(n={n:,})", ha="center", fontsize=9)
    ax.set(ylim=(0, min(1.05, tab.observed_churn_rate.max() + .2)), ylabel="Share who really churned", title=title)
    fig.tight_layout(); return fig
