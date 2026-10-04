"""EDA / analysis figures. Each function has ONE analytical purpose, stated in its docstring."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid", context="notebook")
PAL = sns.color_palette("deep")


def save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.png", dpi=130, bbox_inches="tight")
    return fig


def monthly_summary(tx, orders) -> pd.DataFrame:
    p = tx[~tx.is_return]
    m = pd.DataFrame({
        "gross_revenue": p.groupby(p.date.dt.to_period("M")).transaction_value.sum(),
        "returns": -tx[tx.is_return].groupby(tx[tx.is_return].date.dt.to_period("M")).transaction_value.sum(),
        "orders": orders.groupby(orders.order_date.dt.to_period("M")).size(),
        "active_customers": orders.groupby(orders.order_date.dt.to_period("M")).customer_id.nunique()}).fillna(0)
    m["net_revenue"] = m.gross_revenue - m.returns
    m["aov"] = m.gross_revenue / m.orders
    m.index = m.index.to_timestamp()
    return m


def fig_revenue_trends(m):
    """Is growth driven by more customers or bigger baskets? Shows seasonality and the customer base over time."""
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].bar(m.index, m.net_revenue, width=22, color=PAL[0]); ax[0].set_title("Net revenue per month (after returns)")
    ax[1].plot(m.index, m.aov, marker="o", color=PAL[1]); ax[1].set_title("Average order value")
    ax[2].plot(m.index, m.active_customers, marker="o", color=PAL[2]); ax[2].set_title("Customers who ordered in the month")
    fig.tight_layout(); return save(fig, "01_revenue_trends")


def fig_customer_distributions(f):
    """Shape of customer value: heavy right tails mean a small group drives revenue (motivates log-scaling)."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    sns.histplot(f.monetary, log_scale=True, bins=40, ax=ax[0, 0], color=PAL[0]); ax[0, 0].set_title("Total spend per customer (log x)")
    sns.countplot(x=f.frequency.clip(upper=15), ax=ax[0, 1], color=PAL[1]); ax[0, 1].set_title("Orders per customer (15 = 15+)")
    sns.histplot(f.aov, log_scale=True, bins=40, ax=ax[1, 0], color=PAL[2]); ax[1, 0].set_title("Average order value (log x)")
    sns.histplot(f.recency_days, bins=40, ax=ax[1, 1], color=PAL[3]); ax[1, 1].set_title("Recency: days since last purchase")
    fig.tight_layout(); return save(fig, "02_customer_distributions")


def fig_gap_distribution(gaps, H, qs):
    """Evidence for the churn window: how long do ACTIVE customers normally stay silent between orders?"""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.histplot(gaps[gaps <= 300], bins=60, ax=ax, color=PAL[0])
    for q, v in qs.items():
        ax.axvline(v, color="grey", ls=":", lw=1); ax.text(v, ax.get_ylim()[1] * .92, f"P{int(q * 100)}={v:.0f}d", rotation=90, fontsize=8, va="top")
    ax.axvline(H, color="crimson", lw=2, label=f"chosen churn window H = {H} days")
    ax.set(title="Days between consecutive orders (same customer)", xlabel="days"); ax.legend()
    fig.tight_layout(); return save(fig, "03_interpurchase_gaps")


def fig_window_sensitivity(s, H):
    """Is the churn rate stable around the chosen H? A plateau means the definition is not an artefact."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s.H_days, s.churn_rate, marker="o", color=PAL[3]); ax.axvline(H, color="crimson", ls="--", label=f"chosen H = {H}")
    ax.set(title="Churn rate vs churn window", xlabel="window H (days)", ylabel="churn rate among active customers"); ax.legend()
    fig.tight_layout(); return save(fig, "04_window_sensitivity")


def fig_category_revenue(tx):
    """Where does revenue come from? Category mix."""
    r = tx[~tx.is_return].groupby("category").transaction_value.sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4)); r.plot.barh(ax=ax, color=PAL[0]); ax.set_title("Revenue by product category"); ax.set_xlabel("")
    fig.tight_layout(); return save(fig, "05_category_revenue")


def fig_cohort_retention(orders, max_m=12):
    """Do customers keep ordering after their first month? Each row = acquisition cohort (share ordering in month k)."""
    o = orders.copy()
    o["m"] = o.order_date.dt.to_period("M")
    first = o.groupby("customer_id").m.min().rename("cohort")
    o = o.join(first, on="customer_id")
    o["k"] = (o.m - o.cohort).apply(lambda x: x.n)
    size = first.value_counts()
    t = o[o.k <= max_m].groupby(["cohort", "k"]).customer_id.nunique().unstack().div(size, axis=0)
    last = o.m.max()
    for c in t.index:
        for k in t.columns:
            if (c + k) > last: t.loc[c, k] = np.nan
    t = t[size.reindex(t.index) >= 30]
    if t.empty:
        return None
    t.index = t.index.astype(str)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(t, annot=True, fmt=".0%", cmap="YlGnBu", ax=ax, annot_kws={"size": 7}, cbar_kws={"label": "share of cohort ordering"})
    ax.set(title="Cohort retention (rows: first-order month)", xlabel="months since first order", ylabel="")
    fig.tight_layout(); return save(fig, "06_cohort_retention")


def fig_repeat_vs_onetime(f):
    """How much do one-time buyers matter for revenue vs repeat buyers?"""
    g = np.where(f.frequency == 1, "One-time", "Repeat")
    d = f.groupby(g).agg(customers=("monetary", "size"), revenue=("monetary", "sum"))
    d = d / d.sum()
    fig, ax = plt.subplots(figsize=(5.5, 3.8)); d.T.plot.bar(ax=ax, color=[PAL[1], PAL[0]], rot=0)
    ax.set(title="One-time vs repeat customers", ylabel="share of total", ylim=(0, 1.05)); ax.legend(title="")
    for c in ax.containers: ax.bar_label(c, fmt="%.0%%", labels=[f"{v:.0%}" for v in c.datavalues])
    fig.tight_layout(); return save(fig, "07_repeat_vs_onetime")


def fig_recency_vs_future(snap, H):
    """Relationship between recency and future activity: the probability of buying again falls as silence grows."""
    d = snap.copy(); d["bin"] = pd.cut(d.recency_days, np.arange(0, H + 1, 10), include_lowest=True)
    t = d.groupby("bin", observed=True).agg(p_return=("churned", lambda s: 1 - s.mean()), n=("churned", "size"))
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(range(len(t)), t.p_return, marker="o", color=PAL[2])
    ax.set_xticks(range(len(t))); ax.set_xticklabels([str(i.right).rstrip("0").rstrip(".") if False else int(i.right) for i in t.index], rotation=45)
    ax.set(title=f"Chance of buying again within {H} days, by recency", xlabel="days since last purchase (upper bound of bin)", ylabel="P(buy again)", ylim=(0, 1))
    fig.tight_layout(); return save(fig, "08_recency_vs_future")


def fig_churn_vs_retained(snap, available=None):
    """Which behaviours separate churned from retained customers? (training snapshots)"""
    wanted = ["recency_days", "frequency", "monetary", "aov", "discount_order_share", "n_unique_categories", "frequency_change", "avg_gap_days"]
    cols = [c for c in wanted if (available is None or c in available)]
    cols = (cols + [c for c in ["n_unique_products", "orders_90d", "spend_90d", "active_months"] if (available is None or c in available) and c not in cols])[:8]
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.5))
    lab = snap.churned.map({0: "Retained", 1: "Churned"})
    for a in ax.ravel()[len(cols):]:
        a.axis("off")
    for a, c in zip(ax.ravel(), cols):
        sns.boxplot(x=lab, y=snap[c], ax=a, hue=lab, palette=[PAL[2], PAL[3]], showfliers=False, legend=False); a.set_title(c); a.set_xlabel(""); a.set_ylabel("")
    fig.suptitle("Churned vs retained customers at the cut-off date (outliers hidden)", y=1.0)
    fig.tight_layout(); return save(fig, "09_churn_vs_retained")


def fig_corr(snap, features):
    """Redundancy between predictors (matters for the linear model and for reading importances)."""
    fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(snap[features].corr(), cmap="RdBu_r", center=0, ax=ax, square=True, cbar_kws={"shrink": .7})
    ax.set_title("Feature correlation (Pearson)"); fig.tight_layout(); return save(fig, "10_feature_correlation")


def fig_rfm(seg):
    """Segment sizes vs revenue: who really pays the bills?"""
    order = ["Champions", "Loyal customers", "Potential loyalists", "New customers", "At-risk customers", "Hibernating"]
    d = seg.reindex(order).dropna(how="all")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    d[["customer_share", "revenue_share"]].plot.bar(ax=ax[0], rot=30, color=[PAL[1], PAL[0]]); ax[0].set_title("Share of customers vs share of revenue")
    ax[1].bar(d.index, d.lapsed_share, color=PAL[3]); ax[1].set_title("Already lapsed (silent for longer than the churn window)"); ax[1].tick_params(axis="x", rotation=30); ax[1].set_ylim(0, 1.05)
    fig.tight_layout(); return save(fig, "11_rfm_segments")


def fig_k_selection(ev, k):
    """Elbow + silhouette to choose k (together with business interpretability)."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ev.k, ev.inertia, marker="o"); ax[0].set(title="Elbow (inertia)", xlabel="k")
    ax[1].plot(ev.k, ev.silhouette, marker="o", color=PAL[1]); ax[1].set(title="Silhouette score", xlabel="k")
    for a in ax: a.axvline(k, color="crimson", ls="--", label=f"chosen k = {k}")
    ax[1].legend(); fig.tight_layout(); return save(fig, "12_k_selection")


def fig_clusters_pca(X, labels, names):
    """Do the clusters occupy distinct regions of behaviour space? (2-D PCA projection)"""
    from sklearn.decomposition import PCA
    p = PCA(2, random_state=0).fit(X); Z = p.transform(X)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in sorted(set(labels)):
        ax.scatter(Z[labels == c, 0], Z[labels == c, 1], s=8, alpha=.5, label=names[c])
    ax.set(title="K-Means clusters (PCA projection)", xlabel=f"PC1 ({p.explained_variance_ratio_[0]:.0%})", ylabel=f"PC2 ({p.explained_variance_ratio_[1]:.0%})")
    ax.legend(markerscale=3); fig.tight_layout(); return save(fig, "13_clusters_pca")


def fig_cluster_heatmap(profile, names):
    """Cluster fingerprints: mean of each feature relative to the cluster average (z-score across clusters)."""
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    z.index = [names[i] for i in z.index]
    fig, ax = plt.subplots(figsize=(8, 3.2)); sns.heatmap(z, annot=profile.round(1).values, fmt="", cmap="RdBu_r", center=0, ax=ax, cbar=False)
    ax.set_title("Cluster profiles (colour = relative level, numbers = raw means)"); fig.tight_layout(); return save(fig, "14_cluster_profiles")
