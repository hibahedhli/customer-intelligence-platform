"""
Feature engineering and the churn labelling design.

Snapshot design (prevents data leakage)
---------------------------------------
For a cut-off date T and a churn window H (days):

    <------ observation period ------>|<---- prediction period ---->
    all history up to and including T |   (T, T + H]
    -> features are computed here     |   -> the label is computed here

* Eligible customers: bought at least once, and their last purchase was <= H days before T
  ("active"). Customers silent for more than H days are already churned by our own definition,
  so scoring them would be pointless.
* Label churned = 1 if the customer makes NO purchase in (T, T + H].
* Every feature uses only rows dated <= T. Nothing from the prediction period is ever used.
* The first/last date of the data, the churn window H and all cut-offs come from the dataset itself.
"""
import numpy as np
import pandas as pd


FEATURES = [
    "recency_days", "frequency", "monetary", "aov", "purchase_frequency_30d",
    "days_since_registration", "days_since_first_order", "avg_gap_days", "gap_std_days", "gap_cv",
    "recency_to_gap_ratio", "orders_30d", "orders_90d", "orders_prev90d", "frequency_change",
    "spend_90d", "spend_prev90d", "spend_change", "active_months", "n_unique_products",
    "n_unique_categories", "avg_discount", "discount_order_share", "avg_units_per_order", "return_rate",
]


# Features that need an optional input field: they are dropped (never faked) when the field is unavailable.
FEATURE_REQUIRES = {"n_unique_categories": "category", "n_unique_products": "product_id", "avg_discount": "discount",
                    "discount_order_share": "discount", "days_since_registration": "registration_date_real"}


class InsufficientHistoryError(Exception):
    """The dataset is too short for a time-based churn evaluation at the chosen horizon."""


def select_features(train: pd.DataFrame, availability: dict, candidates=None):
    """
    Choose the model features that this dataset can support.
    Returns (used, excluded) where excluded maps feature -> reason.
    """
    used, excluded = [], {}
    av = dict(availability)
    av["registration_date_real"] = av.get("registration_date", False)
    for f in (candidates or FEATURES):
        need = FEATURE_REQUIRES.get(f)
        if need and not av.get(need, False):
            excluded[f] = f"needs '{need.replace('_real', '')}', which is unavailable in this dataset"
        elif f not in train or train[f].notna().sum() == 0:
            excluded[f] = "no values"
        elif train[f].dropna().nunique() <= 1:
            excluded[f] = "constant in the training data (carries no information)"
        else:
            used.append(f)
    return used, excluded


def build_orders(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per purchase order (returns excluded)."""
    p = tx[~tx.is_return]
    o = p.groupby("order_id").agg(
        customer_id=("customer_id", "first"), order_date=("date", "min"),
        order_value=("transaction_value", "sum"), n_lines=("transaction_id", "count"),
        n_units=("quantity", "sum"), n_categories=("category", "nunique"),
        discount=("discount", "mean"))
    o["has_discount"] = o.discount > 0
    return o.reset_index().sort_values(["customer_id", "order_date"]).reset_index(drop=True)


def interpurchase_gaps(orders: pd.DataFrame) -> pd.Series:
    """Days between consecutive orders of the same customer (same-day orders ignored)."""
    g = orders.sort_values(["customer_id", "order_date"]).groupby("customer_id").order_date.diff().dt.days.dropna()
    return g[g > 0]


def choose_churn_window(gaps: pd.Series, q=0.95, step=15, lo=30, hi=180):
    """
    Pick H so that a *normal* customer rarely stays silent longer than H:
    H = q-quantile of observed inter-purchase gaps, rounded up to a multiple of `step`.
    """
    qs = gaps.quantile([.5, .75, .9, .95, .99])
    H = int(np.clip(np.ceil(gaps.quantile(q) / step) * step, lo, hi))
    return H, qs


def compute_features(orders, lines, returns, customers, cutoff) -> pd.DataFrame:
    """Customer-level features using ONLY data dated <= cutoff. Index = customer_id."""
    cutoff = pd.Timestamp(cutoff)
    o = orders[orders.order_date <= cutoff].sort_values(["customer_id", "order_date"]).copy()
    o["gap"] = o.groupby("customer_id").order_date.diff().dt.days
    o.loc[o["gap"] <= 0, "gap"] = np.nan
    o["age_days"] = (cutoff - o.order_date).dt.days
    o["ym"] = o.order_date.dt.year * 12 + o.order_date.dt.month
    g = o.groupby("customer_id")

    f = pd.DataFrame({"first_order": g.order_date.min(), "last_order": g.order_date.max(),
                      "frequency": g.size(), "monetary": g.order_value.sum(),
                      "active_months": g.ym.nunique(),
                      "avg_gap_days": g.gap.mean(), "gap_std_days": g.gap.std(),
                      "discount_order_share": g.has_discount.mean()})
    f["recency_days"] = (cutoff - f.last_order).dt.days
    f["days_since_first_order"] = (cutoff - f.first_order).dt.days
    f["aov"] = f.monetary / f.frequency
    f["purchase_frequency_30d"] = f.frequency / f.days_since_first_order.clip(lower=30) * 30
    f["gap_cv"] = (f.gap_std_days / f.avg_gap_days.replace(0, np.nan))
    f["recency_to_gap_ratio"] = f.recency_days / f.avg_gap_days.replace(0, np.nan)

    for name, lo, hi in (("30d", 0, 30), ("90d", 0, 90), ("prev90d", 90, 180)):
        sub = o[(o.age_days >= lo) & (o.age_days < hi)].groupby("customer_id")
        f[f"orders_{name}"] = sub.size().reindex(f.index).fillna(0)
        if name != "30d":
            f[f"spend_{name}"] = sub.order_value.sum().reindex(f.index).fillna(0)
    f["frequency_change"] = f.orders_90d - f.orders_prev90d
    f["spend_change"] = f.spend_90d - f.spend_prev90d

    ld = lines[lines.date <= cutoff]
    l = ld.groupby("customer_id")
    f["n_unique_products"] = l.product_id.nunique().reindex(f.index) if lines.product_id.notna().any() else np.nan
    f["n_unique_categories"] = l.category.nunique().reindex(f.index) if lines.category.notna().any() else np.nan
    f["avg_discount"] = l.discount.mean().reindex(f.index)
    f["avg_units_per_order"] = l.quantity.sum().reindex(f.index) / f.frequency

    r = returns[returns.date <= cutoff].groupby("customer_id").transaction_value.sum()
    f["return_rate"] = (-r.reindex(f.index).fillna(0) / f.monetary).clip(0, 1)

    reg = customers.set_index("customer_id").customer_registration_date
    f["days_since_registration"] = (cutoff - reg.reindex(f.index)).dt.days
    return f.replace([np.inf, -np.inf], np.nan)


def build_snapshot(orders, lines, returns, customers, cutoff, H, label=True, data_end=None) -> pd.DataFrame:
    """Features at `cutoff` (past only) and, if label=True, the churn label from (cutoff, cutoff+H]."""
    cutoff = pd.Timestamp(cutoff)
    data_end = pd.Timestamp(data_end) if data_end is not None else orders.order_date.max()
    f = compute_features(orders, lines, returns, customers, cutoff)
    f = f[f.recency_days <= H].copy()
    f["cutoff"] = cutoff
    if label:
        end = cutoff + pd.Timedelta(days=H)
        assert end <= data_end, "label window runs past the end of the data (right-censoring)"
        future = orders[(orders.order_date > cutoff) & (orders.order_date <= end)].customer_id.unique()
        f["churned"] = (~f.index.isin(future)).astype(int)
    return f


def required_history_days(H, first=120, step=30, min_snapshots=3):
    """Days of history needed for: >= `first` observation days, `min_snapshots` training cut-offs, then 3 churn windows."""
    return int(first + 3 * H + (min_snapshots - 1) * step)


def cutoff_schedule(H, data_start, data_end, step=30, first=120, min_snapshots=3):
    """
    Chronological split derived from the dataset's own dates (no fixed calendar).

        train cut-offs ... | last_train + H = val | val + H = test | test + H = data_end

    Label windows never overlap the period being predicted next. Raises InsufficientHistoryError if the
    history is too short for at least `min_snapshots` training cut-offs.
    """
    start, end = pd.Timestamp(data_start), pd.Timestamp(data_end)
    test = end - pd.Timedelta(days=H)
    val = test - pd.Timedelta(days=H)
    last_train = val - pd.Timedelta(days=H)
    first_cut = start + pd.Timedelta(days=first)
    train = list(pd.date_range(first_cut, last_train, freq=f"{step}D")) if last_train >= first_cut else []
    if len(train) < min_snapshots:
        have = (end - start).days
        need = required_history_days(H, first, step, min_snapshots)
        raise InsufficientHistoryError(
            f"The dataset contains only {have / 30.4:.0f} months of history ({start.date()} -> {end.date()}). "
            f"At least approximately {need / 30.4:.0f} months are recommended for a churn horizon of {H} days "
            f"({first} days of observation + {min_snapshots} training cut-offs + 3 consecutive {H}-day windows for train/validation/test). "
            f"Options: provide more history, or set analysis.churn_window_days to a shorter horizon if that is meaningful for this business.")
    return train, val, test


def make_snapshots(orders, lines, returns, customers, cutoffs, H, data_end=None) -> pd.DataFrame:
    parts = [build_snapshot(orders, lines, returns, customers, c, H, data_end=data_end).reset_index() for c in cutoffs]
    return pd.concat(parts, ignore_index=True)


def window_sensitivity(orders, lines, returns, customers, data_start, data_end, H, grid=(30, 60, 90, 120, 150, 180)):
    """How the churn rate and eligible population change with H (evaluated at one common cut-off)."""
    start, end = pd.Timestamp(data_start), pd.Timestamp(data_end)
    windows = sorted(set(grid) | {int(H)})
    while windows and (end - pd.Timedelta(days=max(windows))) < start + pd.Timedelta(days=60):
        windows = windows[:-1]          # short dataset: drop the longest windows
    if not windows:
        return pd.DataFrame(columns=["H_days", "eligible_customers", "churn_rate"])
    cutoff = end - pd.Timedelta(days=max(windows))
    rows = []
    for w in windows:
        s_ = build_snapshot(orders, lines, returns, customers, cutoff, w, data_end=end)
        rows.append({"H_days": w, "eligible_customers": len(s_), "churn_rate": s_.churned.mean() if len(s_) else np.nan})
    return pd.DataFrame(rows)
