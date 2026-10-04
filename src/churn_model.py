"""Model definitions, validation-based tuning and persistence."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config as C
from .config import SEED
from . import evaluation as ev
from .feature_engineering import FEATURES

# Heavy-tailed, non-negative variables get a log1p before the linear model.
LOG_COLS = ["frequency", "monetary", "aov", "purchase_frequency_30d", "orders_30d", "orders_90d", "orders_prev90d",
            "spend_90d", "spend_prev90d", "avg_units_per_order", "n_unique_products"]


def logistic_pipeline(features, C=0.3):
    logc = [c for c in features if c in LOG_COLS]
    rest = [c for c in features if c not in LOG_COLS]
    prep = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), logc), ("rest", "passthrough", rest)],
        verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([("prep", prep), ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                     ("scale", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=2000))])


def boosting(**kw):
    p = dict(learning_rate=0.05, max_depth=4, max_iter=250, min_samples_leaf=80, l2_regularization=1.0, random_state=SEED)
    p.update(kw)
    return HistGradientBoostingClassifier(**p)


def tune_boosting(Xtr, ytr, Xva, yva):
    """Small grid, selected on VALIDATION PR-AUC (never on test)."""
    rows = []
    for lr in (0.03, 0.06):
        for depth in (3, 5):
            for leaf in (40, 120):
                m = boosting(learning_rate=lr, max_depth=depth, min_samples_leaf=leaf).fit(Xtr, ytr)
                rows.append({"learning_rate": lr, "max_depth": depth, "min_samples_leaf": leaf,
                             "val_pr_auc": average_precision_score(yva, m.predict_proba(Xva)[:, 1])})
    grid = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    best = grid.iloc[0]
    return boosting(learning_rate=best.learning_rate, max_depth=int(best.max_depth),
                    min_samples_leaf=int(best.min_samples_leaf)), grid


def save(obj, name):
    C.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, C.MODELS / name)


def load(name):
    return joblib.load(C.MODELS / name)


# ------------------------------------------------------------------------------------------
# Training + selection (uses train and validation ONLY) and one-shot test evaluation
# ------------------------------------------------------------------------------------------
SELECTION_METRICS = ("pr_auc", "roc_auc")


def run_modeling(train, val, features, metric="pr_auc", seed=SEED, target="churned"):
    """
    Fit the candidate models on `train`, score them on `val`, and pick the best by the VALIDATION metric.
    This function never receives the test set, so test results cannot influence model choice.
    """
    if metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
    Xtr, ytr, Xva, yva = train[features], train[target], val[features], val[target]
    lr_rec = logistic_pipeline(["recency_days"]).fit(Xtr[["recency_days"]], ytr)
    lr_all = logistic_pipeline(features).fit(Xtr, ytr)
    gb, grid = tune_boosting(Xtr, ytr, Xva, yva)
    gb.fit(Xtr, ytr)
    models = {"Logistic (recency only)": (lr_rec, ["recency_days"]), "Logistic (all features)": (lr_all, list(features)),
              "Gradient boosting": (gb, list(features))}
    val_probs = {n: m.predict_proba(val[cols])[:, 1] for n, (m, cols) in models.items()}
    rows = []
    for n, p in val_probs.items():
        rows.append({"model": n, "val_pr_auc": float(average_precision_score(yva, p)),
                     "val_roc_auc": float(ev.roc_auc_score(yva, p)), "threshold": ev.best_f1_threshold(yva, p)})
    val_table = pd.DataFrame(rows)
    best = val_table.sort_values(f"val_{metric}", ascending=False).iloc[0].model
    return {"models": models, "val_probs": val_probs, "val_table": val_table, "best_name": best, "metric": metric, "grid": grid,
            "thresholds": dict(zip(val_table.model, val_table.threshold))}


def evaluate_on_test(result, test, target="churned"):
    """The only place the test set is used: score every candidate once and report. Does not change the selection."""
    y = test[target]
    rows, probs = [], {}
    vt = result["val_table"].set_index("model")
    for n, (m, cols) in result["models"].items():
        p = m.predict_proba(test[cols])[:, 1]
        probs[n] = p
        rows.append({"model": n, "val_pr_auc": vt.loc[n, "val_pr_auc"], "val_roc_auc": vt.loc[n, "val_roc_auc"],
                     **ev.classification_metrics(y, p, result["thresholds"][n])})
    base = float(y.mean())
    rows.append({"model": "Flag everyone as churn (naive)", "roc_auc": 0.5, "pr_auc": base, "threshold": 0.0,
                 "precision": base, "recall": 1.0, "f1": 2 * base / (1 + base), "accuracy": base})
    return pd.DataFrame(rows), probs
