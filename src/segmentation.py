"""RFM scoring (rule-based segments) and K-Means behavioural clustering (unsupervised)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------------------
# RFM
# ----------------------------------------------------------------------------------------

def _quintile(s: pd.Series) -> pd.Series:
    """1..5 by percentile rank. Ties share the same score (no arbitrary tie-breaking)."""
    pct = s.rank(pct=True, method="average")
    return pd.cut(pct, [0, .2, .4, .6, .8, 1.0], labels=[1, 2, 3, 4, 5], include_lowest=True).astype(int)


def rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """df needs recency_days, frequency, monetary. Recency is inverted: recent = 5."""
    out = df.copy()
    out["R"] = _quintile(-out.recency_days)
    out["F"] = _quintile(out.frequency)
    out["M"] = _quintile(out.monetary)
    out["RFM_score"] = out.R.astype(str) + out.F.astype(str) + out.M.astype(str)
    out["RFM_sum"] = out.R + out.F + out.M
    return out


def rfm_quality(df: pd.DataFrame) -> dict:
    """Does the data support meaningful RFM labels? (Warnings are shown in the dashboard / README.)"""
    one_time = float((df.frequency <= 1).mean())
    warnings = []
    if one_time > 0.7:
        warnings.append(f"{one_time:.0%} of customers bought only once: the Frequency score is weakly informative, so segments are driven mostly by Recency and Monetary value.")
    if df.frequency.nunique() < 4:
        warnings.append("Fewer than 4 distinct order counts: the Frequency score has little resolution.")
    return {"share_one_time": one_time, "distinct_frequencies": int(df.frequency.nunique()), "warnings": warnings}


SEGMENT_RULES = """
Segments are assigned top-to-bottom; the first rule that matches wins. FM = (F + M) / 2.
 1. New customers        first order <= 60 days ago AND <= 2 orders (too little history to judge value)
 2. Champions            R >= 4 AND FM >= 4          (recent, frequent, high spend)
 3. Loyal customers      R >= 3 AND FM >= 3.5        (solid repeat buyers, reasonably recent)
 4. At-risk customers    R <= 2 AND FM >= 3          (used to be good, now quiet -> valuable but slipping)
 5. Potential loyalists  R >= 3                      (recent, but not yet frequent/high-spend)
 6. Hibernating          everything else (R <= 2, low FM)
"""


def rfm_segments(df: pd.DataFrame, new_days=60) -> pd.DataFrame:
    out = df.copy()
    fm = (out.F + out.M) / 2
    conds = [
        (out.days_since_first_order <= new_days) & (out.frequency <= 2),
        (out.R >= 4) & (fm >= 4),
        (out.R >= 3) & (fm >= 3.5),
        (out.R <= 2) & (fm >= 3),
        (out.R >= 3),
    ]
    names = ["New customers", "Champions", "Loyal customers", "At-risk customers", "Potential loyalists"]
    out["rfm_segment"] = np.select(conds, names, default="Hibernating")
    return out


# ----------------------------------------------------------------------------------------
# K-Means
# ----------------------------------------------------------------------------------------
CLUSTER_FEATURES = ["recency_days", "frequency", "monetary", "aov", "purchase_frequency_30d", "n_unique_categories"]


def prepare_matrix(df: pd.DataFrame, features=CLUSTER_FEATURES):
    """
    1. log1p      : spend/frequency are right-skewed (a few huge customers); K-Means uses Euclidean
                    distance, so raw values would let those customers dominate the geometry.
    2. winsorise  : clip to the 1st-99th percentile so a handful of extreme points cannot drag centroids.
    3. standardise: put features on a common scale; otherwise 'monetary' (hundreds) would outweigh
                    'n_unique_categories' (units) purely because of its units.
    """
    X = np.log1p(df[features].astype(float))
    X = X.clip(X.quantile(.01), X.quantile(.99), axis=1)
    return StandardScaler().fit_transform(X), X.columns.tolist()


MIN_CUSTOMERS_FOR_CLUSTERING = 100


def evaluate_k(X, ks=range(2, 9), seed=42) -> pd.DataFrame:
    rows = []
    for k in [k for k in ks if k < len(X) // 20]:   # keep >= ~20 customers per cluster on average
        km = KMeans(k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def choose_k(ev: pd.DataFrame, lo=3, hi=6, tol=0.10) -> int:
    """
    Not simply argmax(silhouette): k=2 almost always wins that contest and is useless for marketing.
    Rule: restrict to the actionable range [lo, hi]; take the SMALLEST k whose silhouette is within
    `tol` of the best in that range (parsimony).
    """
    sub = ev[(ev.k >= lo) & (ev.k <= hi)]
    if sub.empty:                      # small dataset: fall back to the best of whatever was evaluated
        return int(ev.loc[ev.silhouette.idxmax(), "k"])
    return int(sub[sub.silhouette >= sub.silhouette.max() * (1 - tol)].k.min())


def fit_kmeans(X, k, seed=42) -> KMeans:
    return KMeans(k, n_init=20, random_state=seed).fit(X)


def stability(X, k, seeds=(1, 2, 3, 4, 5)) -> float:
    """Mean adjusted Rand index between clusterings from different seeds (1 = identical)."""
    labs = [KMeans(k, n_init=5, random_state=s).fit_predict(X) for s in seeds]
    return float(np.mean([adjusted_rand_score(labs[0], l) for l in labs[1:]]))


def name_clusters(profile: pd.DataFrame, population: pd.DataFrame) -> dict:
    """Readable names: cluster means expressed in standard deviations of the WHOLE customer base."""
    cols = ["recency_days", "frequency", "monetary"]
    z = (profile[cols] - population[cols].mean()) / population[cols].std(ddof=0)
    names = {}
    for c in profile.index:
        act = "Active" if z.loc[c, "recency_days"] < -0.4 else ("Dormant" if z.loc[c, "recency_days"] > 0.4 else "Cooling")
        val = z.loc[c, ["frequency", "monetary"]].mean()
        tier = "high-value" if val > 0.4 else ("low-value" if val < -0.4 else "mid-value")
        names[c] = f"C{c}: {act}, {tier}"
    return names
