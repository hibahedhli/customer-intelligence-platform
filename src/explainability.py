"""Global and local explanations. SHAP explains the MODEL's prediction, not the cause of behaviour."""
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

TEMPLATES = {
    "recency_days": "{v:.0f} days since last purchase",
    "frequency": "{v:.0f} orders placed so far",
    "monetary": "{v:,.0f} total spend",
    "aov": "average order value of {v:,.0f}",
    "purchase_frequency_30d": "{v:.2f} orders per 30 days of life",
    "days_since_registration": "{v:.0f} days since registration",
    "days_since_first_order": "{v:.0f} days since first order",
    "avg_gap_days": "usually {v:.0f} days between orders",
    "gap_std_days": "order spacing varies by {v:.0f} days",
    "gap_cv": "irregular ordering rhythm (variation {v:.2f})",
    "recency_to_gap_ratio": "silent {v:.1f}x longer than their usual gap",
    "orders_30d": "{v:.0f} orders in the last 30 days",
    "orders_90d": "{v:.0f} orders in the last 90 days",
    "orders_prev90d": "{v:.0f} orders in the 90 days before that",
    "frequency_change": "order count changed by {v:+.0f} vs the previous 90 days",
    "spend_90d": "{v:,.0f} spent in the last 90 days",
    "spend_prev90d": "{v:,.0f} spent in the 90 days before that",
    "spend_change": "spend changed by {v:+,.0f} vs the previous 90 days",
    "active_months": "bought in {v:.0f} different months",
    "n_unique_products": "{v:.0f} distinct products bought",
    "n_unique_categories": "{v:.0f} product categories explored",
    "avg_discount": "average discount of {v:.0%}",
    "discount_order_share": "{v:.0%} of orders used a discount",
    "avg_units_per_order": "{v:.1f} units per order",
    "return_rate": "{v:.0%} of spend returned",
}


def describe(feature, value):
    if pd.isna(value):
        return f"{feature}: not enough history"
    return TEMPLATES.get(feature, feature + " = {v:.2f}").format(v=value)


def _is_tree(model) -> bool:
    return hasattr(model, "_predictors") or hasattr(model, "estimators_") or hasattr(model, "tree_")


def shap_values(model, X: pd.DataFrame, seed=42, background_n=100):
    """
    SHAP values in LOG-ODDS for whichever model was selected. Returns (values, base_value).
    Trees -> TreeExplainer (exact). Anything else (e.g. the logistic pipeline) -> model-agnostic permutation
    explainer on decision_function. In both cases base + sum(values) = the model's own log-odds output.
    """
    if _is_tree(model):
        exp = shap.TreeExplainer(model)(X)
        vals = exp.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        return np.asarray(vals), float(np.ravel(exp.base_values)[0])
    cols = list(X.columns)
    f = lambda a: model.decision_function(pd.DataFrame(a, columns=cols))
    bg = X.sample(min(background_n, len(X)), random_state=seed).to_numpy(dtype=float)
    expl = shap.Explainer(f, shap.maskers.Independent(bg, max_samples=background_n), algorithm="permutation", seed=seed)
    exp = expl(X.to_numpy(dtype=float), max_evals=2 * len(cols) + 1, silent=True)
    return np.asarray(exp.values), float(np.ravel(exp.base_values)[0])


def explain_row(sv_row, x_row: pd.Series, top_n=3):
    """Return (risk_increasing, risk_reducing) lists of human-readable strings."""
    order = np.argsort(sv_row)
    up = [i for i in order[::-1] if sv_row[i] > 0][:top_n]
    down = [i for i in order if sv_row[i] < 0][:top_n]
    f = lambda i: describe(x_row.index[i], x_row.iloc[i])
    return [f(i) for i in up], [f(i) for i in down]


def explanation_table(sv, X: pd.DataFrame, top_n=3):
    ups, downs = [], []
    for k in range(len(X)):
        u, d = explain_row(sv[k], X.iloc[k], top_n)
        ups.append(" | ".join(u)); downs.append(" | ".join(d))
    return pd.DataFrame({"top_risk_factors": ups, "top_protective_factors": downs}, index=X.index)


def global_importance(sv, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values(
        "mean_abs_shap", ascending=False).reset_index(drop=True)


def permutation_table(model, X, y, n_repeats=5, seed=42) -> pd.DataFrame:
    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return pd.DataFrame({"feature": X.columns, "pr_auc_drop": r.importances_mean, "std": r.importances_std}).sort_values(
        "pr_auc_drop", ascending=False).reset_index(drop=True)
