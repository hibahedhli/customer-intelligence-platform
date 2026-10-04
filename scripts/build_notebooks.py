"""Generates and executes the six analysis notebooks:  python scripts/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

import os, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = r'''
import sys, json, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import Markdown, display
from src.config import *
from src import eda, evaluation as ev
from src.feature_engineering import *
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
'''
CORE = r'''
customers = pd.read_csv(DATA_PROC / "customers_clean.csv", parse_dates=["customer_registration_date"])
tx = pd.read_csv(DATA_PROC / "transactions_clean.csv.gz", parse_dates=["date"])
orders = build_orders(tx); lines, returns = tx[~tx.is_return], tx[tx.is_return]
M = json.load(open(DATA_PROC / "metrics.json")); H = M["H"]
meta = json.load(open(DATA_PROC / "run_metadata.json"))
START, END = pd.Timestamp(M["data_start"]), pd.Timestamp(M["data_end"])
FEATURES_USED = meta["features_used"]
train_c = [pd.Timestamp(c) for c in M["train_cutoffs"]]; val_c = pd.Timestamp(M["val_cutoff"]); test_c = pd.Timestamp(M["test_cutoff"])
display(Markdown(f"> **Dataset: {meta['name']}** ({'SIMULATED' if meta['kind'] == 'synthetic' else 'real'}) | {M['data_start']} -> {M['data_end']} | churn window **{H} days** ({M['churn_window_source']}). "
                 "All numbers in the commentary below are computed from this dataset when the notebook runs."))
'''

md = lambda s: ("md", s.strip())
code = lambda s: ("code", s.strip())

NOTEBOOKS = {}

# ---------------------------------------------------------------- 01
NOTEBOOKS["01_data_understanding"] = [
md("""# 01 - Data understanding & cleaning
**Business question:** can we trust the data enough to measure customer behaviour?

The dataset, its source and kind (real / simulated) are shown in the banner of each notebook and described in the README. For the bundled synthetic demo, realistic quality problems were injected on purpose so that the cleaning decisions below are real decisions."""),
code(SETUP),
code(r'''
from src.adapter import load_config, standardize, mapping_table
from src.data_processing import inspect_data, clean_data
meta = json.load(open(DATA_PROC / "run_metadata.json"))
cfg = load_config(meta["config"])
print(f"Dataset: {meta['name']} ({'SIMULATED' if meta['kind'] == 'synthetic' else 'real'})")
display(mapping_table(cfg))                                   # original column -> canonical column
cust_raw, tx_raw, av, notes = standardize(cfg, save=False)    # adapter: raw file(s) -> canonical schema
print(*notes, sep="\n")
rep = inspect_data(cust_raw, tx_raw, cfg)
display(rep["shapes"]); display(rep["transactions_profile"]); display(rep["customers_profile"])
display(rep["issues"].to_frame())
'''),
md("""## Investigate before deleting
Unusual values are not automatically errors. We look at each one first."""),
code(r'''
neg = tx_raw[tx_raw.quantity < 0].copy()
neg["kind"] = np.where(neg.is_return, "flagged as return by the configured rule", "NOT flagged (treated as invalid)")
display(neg.kind.value_counts().to_frame())
print("return rule:", av["return_rule"])
display(neg[~neg.is_return].head())
'''),
code(r'''
bad = tx_raw[tx_raw.unit_price.fillna(0) <= 0].copy()
print(f"{len(bad):,} rows with price <= 0 or missing; configured repair: {cfg['cleaning']['nonpositive_price']}")
if av["product_id"] and len(bad):
    med = tx_raw[tx_raw.unit_price > 0].groupby("product_id").unit_price.agg(["median", "std"])
    print("Within-product price spread (std) - a small value means one list price per product, so the product median is a sound repair:")
    display(med["std"].describe().round(3).to_frame().T)
'''),
md("## Cleaning (every decision is logged)"),
code(r'''
cust, tx_clean, log, summ = clean_data(cust_raw, tx_raw, cfg, av)
display(log)
print(f"rows: {summ['rows_in']:,} in -> {summ['rows_out']:,} kept | purchase lines {summ['normal_lines_kept']:,} | returns kept {summ['return_lines_kept']:,}")
print("removed by reason:", summ["removed_by_reason"])
print(f"customers in table: {len(cust_raw):,} -> {len(cust):,}")
'''),
md("""### So what?
* Check the "removed by reason" line above: it states exactly how many rows were dropped and why. Anything that can be repaired from other information (dates, spellings, units) is repaired; only unverifiable rows are dropped.
* Returns / cancellations are kept and flagged by the **configured** rule: they matter for *net* revenue but are **not purchases** and never count as customer activity in the churn definition. If the "NOT flagged" count above is large, the return rule in the config is probably wrong.
* Cleaned tables are the only input to every later notebook."""),
]

# ---------------------------------------------------------------- 02
NOTEBOOKS["02_eda"] = [
md("""# 02 - Exploratory data analysis
**Business questions:** who creates the revenue, how do customers behave over time, and what separates customers who leave from customers who stay?"""),
code(SETUP), code(CORE),
code(r'''
monthly = eda.monthly_summary(tx, orders)
eda.fig_revenue_trends(monthly); plt.show()
g = monthly.net_revenue
display(Markdown(f"**So what?** Net revenue was {g.iloc[0]:,.0f} in the first month and {g.iloc[-1]:,.0f} in the last; the strongest month is {g.idxmax():%Y-%m} ({g.max():,.0f}). "
                 "Compare it with the customers-per-month panel: growth that only reflects a larger customer base is different from growth in spend per customer. "
                 "Strong seasonal peaks also mean churn rates measured in different seasons are not directly comparable."))
'''),
code(r'''
f = compute_features(orders, lines, returns, customers, END)
eda.fig_customer_distributions(f); plt.show()
s = f.monetary.sort_values(ascending=False)
top20 = s.head(int(.2 * len(s))).sum() / s.sum()
display(Markdown(f"**So what?** Spending is extremely right-skewed: the top 20% of customers generate **{top20:.0%}** of revenue. "
                 "Averages are therefore misleading, and any distance-based method (K-Means) needs log-scaling."))
'''),
code(r'''
eda.fig_repeat_vs_onetime(f); plt.show()
if meta["availability"]["category"]:
    eda.fig_category_revenue(tx); plt.show()
else:
    print("Revenue by category: Not available in this dataset")
eda.fig_cohort_retention(orders); plt.show()
rep = (f.frequency >= 2).mean()
display(Markdown(f"**So what?** {rep:.0%} of customers bought at least twice. Read the cohort heatmap row by row: the first column is 100% by construction, the second shows how many customers come back in the month after their first purchase, and the right-hand side shows how fast activity fades. "
                 "Retention work should consider both the first months after acquisition and long-tenured customers who go quiet."))
'''),
md("""## Retention and churn
We now need the churn definition (developed fully in notebook 05). Here we only use it to compare behaviour. Features are computed at each cut-off using **past data only**."""),
code(r'''
train = make_snapshots(orders, lines, returns, customers, train_c, H, data_end=END)
eda.fig_recency_vs_future(train, H); plt.show()
display(Markdown(f"**So what?** Read the curve: it shows how the chance of buying again within {H} days changes with the time already silent. A steep decline means recency is a strong signal of future activity; a flat curve would mean it is not."))
'''),
code(r'''
eda.fig_churn_vs_retained(train, FEATURES_USED); plt.show()
cols = [c for c in ["recency_days","frequency","monetary","aov","avg_gap_days","discount_order_share","n_unique_categories","n_unique_products","frequency_change"] if c in FEATURES_USED]
med = train.groupby(train.churned.map({0:"Retained",1:"Churned"}))[cols].median()
display(med.T.round(2))
eda.fig_corr(train, FEATURES_USED); plt.show()
excl = meta["features_excluded"]
if excl: print("Features not available / not used in this dataset:", {k: v for k, v in excl.items()})
'''),
code(r'''
bul = []
for c in med.columns:
    ch, rt = med.loc["Churned", c], med.loc["Retained", c]
    bul.append(f"* `{c}`: median {ch:,.2f} for churned vs {rt:,.2f} for retained ({'higher' if ch > rt else 'lower' if ch < rt else 'equal'} among churned)")
display(Markdown("### So what?\n" + "\n".join(bul) + "\n\n"
    "* These are *associations at a point in time*, not causes: e.g. a different discount share between groups does not show that discounts cause churn.\n"
    "* Many predictors are correlated (frequency, spend, 90-day orders...). That is fine for tree models, but individual linear coefficients and importances must be read with care."))
'''),
]

# ---------------------------------------------------------------- 03
NOTEBOOKS["03_rfm_analysis"] = [
md("""# 03 - RFM analysis
**Question:** which customers are the most valuable, and which of them have gone quiet?"""),
code(SETUP), code(CORE),
code(r'''
from src.segmentation import rfm_scores, rfm_segments, SEGMENT_RULES
f = compute_features(orders, lines, returns, customers, END)
rfm = rfm_segments(rfm_scores(f))
rfm["status"] = np.where(rfm.recency_days <= H, "Active", "Lapsed")
print(SEGMENT_RULES)
display(rfm[["recency_days","frequency","monetary","R","F","M","RFM_score","rfm_segment"]].head())
'''),
md("""**Scoring logic.** Each of R, F, M gets a 1-5 score from percentile rank, so ties share a score (no arbitrary tie-breaking: the many one-order customers all get the same F). Recency is inverted so that 5 = most recent. Segment names are *labels for rules*, so we verify afterwards that the data supports them."""),
code(r'''
seg = pd.read_csv(DATA_PROC / "rfm_segment_summary.csv").set_index("rfm_segment")
display(seg.round(2))
eda.fig_rfm(seg); plt.show()
rq = meta["rfm_quality"]
bul = [f"* **{n}**: {r.customers:,.0f} customers ({r.customer_share:.0%}), **{r.revenue_share:.0%} of revenue**, median recency {r.median_recency:.0f} d, median {r.median_frequency:.0f} orders, {r.lapsed_share:.0%} already beyond the {H}-day churn window" for n, r in seg.iterrows()]
display(Markdown("**So what?**\n" + "\n".join(bul) + f"\n\n* Segments with no customers do not appear: a business label is only shown when the data supports it.\n"
    "* RFM is *descriptive*: segments defined by poor recency are mostly already past the churn window. Predicting who is *about to* leave needs the churn model (notebook 05), applied to customers who are still active.\n"
    + "".join(f"* WARNING: {w}\n" for w in rq["warnings"])))
'''),
]

# ---------------------------------------------------------------- 04
NOTEBOOKS["04_customer_segmentation"] = [
md("""# 04 - Behavioural segmentation (K-Means)
**Question:** beyond hand-written rules, what natural groups exist in customer behaviour?"""),
code(SETUP), code(CORE),
code(r'''
from src.segmentation import *
f = compute_features(orders, lines, returns, customers, END)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
for a, c in zip(ax, ["monetary", "frequency", "aov"]):
    sns.histplot(f[c], ax=a, bins=40); a.set_title(f"{c}: skew = {f[c].skew():.1f}")
plt.show()
cl_feats = [c for c in CLUSTER_FEATURES if f[c].notna().any() and f[c].nunique() > 1]
print("clustering features available in this dataset:", cl_feats)
assert len(f) >= MIN_CUSTOMERS_FOR_CLUSTERING, "too few customers for clustering"
X, cols = prepare_matrix(f, cl_feats)
print("after log1p -> winsorise -> standardise: skew =", pd.DataFrame(X, columns=cols).skew().round(2).to_dict())
'''),
md("""**Why scale?** K-Means groups points by Euclidean distance. Without scaling, `monetary` (hundreds/thousands) would dominate `n_unique_categories` (1-8) purely because of units. Log-transform + winsorising tame the long right tail so a few huge customers do not drag the centroids."""),
code(r'''
ev_k = evaluate_k(X); k = choose_k(ev_k)
display(ev_k.round(3)); eda.fig_k_selection(ev_k, k); plt.show()
print(f"chosen k = {k}; stability across seeds (ARI) = {stability(X, k):.3f}")
'''),
md("""**Why not simply the highest silhouette?** k = 2 always scores best (it just splits active from inactive), which is useless for marketing. We restrict to an actionable range (3-6) and take the smallest k within 10% of the best silhouette in that range. The silhouette value printed above tells how separated the groups are: below ~0.25 means weak structure (behaviour is closer to a continuum than to distinct groups), 0.25-0.5 means real but overlapping groups, above 0.5 clearly separated groups."""),
code(r'''
km = fit_kmeans(X, k); f["cluster"] = km.labels_
prof = f.groupby("cluster")[cl_feats].mean()
names = name_clusters(prof, f)
prof_out = prof.copy(); prof_out["customers"] = f.groupby("cluster").size(); prof_out["revenue_share"] = f.groupby("cluster").monetary.sum() / f.monetary.sum()
prof_out["lapsed_share"] = f.groupby("cluster").recency_days.apply(lambda s: (s > H).mean()); prof_out.index = [names[i] for i in prof_out.index]
display(prof_out.round(2))
eda.fig_clusters_pca(X, km.labels_, names); plt.show()
eda.fig_cluster_heatmap(prof, names); plt.show()
'''),
code(r'''
cl = prof_out
hi = cl.revenue_share.idxmax(); lo = cl.lapsed_share.idxmax()
sil = float(ev_k.set_index("k").silhouette[k])
display(Markdown(f"""**So what? (business reading of the clusters, computed from this dataset)**
* Silhouette for k = {k}: **{sil:.2f}**.
* **{hi}** generates the largest revenue share (**{cl.loc[hi,'revenue_share']:.0%}**) with {cl.loc[hi,'customers']:,.0f} customers ({cl.loc[hi,'customers']/cl.customers.sum():.0%}); on average {cl.loc[hi,'frequency']:.1f} orders and {cl.loc[hi,'recency_days']:.0f} days since the last purchase. Check these two numbers before calling it 'engaged'.
* **{lo}** has the highest lapsed share ({cl.loc[lo,'lapsed_share']:.0%}) and {cl.loc[lo,'revenue_share']:.0%} of revenue.
* Names such as 'Active, high-value' are generated automatically from each cluster's recency / frequency / spend relative to all customers; read the table, not just the label.

Clusters complement RFM: RFM is a transparent rulebook; clusters are learned from the behaviours listed above and need no hand-set thresholds."""))
'''),
]

# ---------------------------------------------------------------- 05
NOTEBOOKS["05_churn_modeling"] = [
md("""# 05 - Defining churn, building and evaluating the model
**Question:** which currently-active customers are likely to stop buying, and how well can we really tell?"""),
code(SETUP), code(CORE),
md("""## 1. Churn definition - derived from the data, not assumed"""),
code(r'''
gaps = interpurchase_gaps(orders); H_, qs = choose_churn_window(gaps)
print(qs.round(0).to_dict()); print("derived H =", H_, "| used in the pipeline:", H, f"({M['churn_window_source']})")
eda.fig_gap_distribution(gaps, H, qs); plt.show()
sens = pd.read_csv(DATA_PROC / "churn_window_sensitivity.csv"); display(sens.round(3)); eda.fig_window_sensitivity(sens, H); plt.show()
'''),
code(r'''
display(Markdown(f"""**Reasoning.** We want a window long enough that a *normally active* customer rarely stays silent for that long. For this dataset the 95th percentile of observed gaps is **{qs.loc[0.95]:.0f} days**, rounded up to a multiple of 15 and kept inside the safety range: **H = {H} days**. This is *recalculated for every dataset*: it is not a constant of the project.
The sensitivity table shows the churn rate for neighbouring windows: a stable region around H means the label is not an artefact of one arbitrary number; steep changes mean the definition is sensitive (the validation report warns about this).
One limitation: gaps are only observable for customers who did come back, so the true gap distribution is somewhat longer than measured (right-censoring)."""))
'''),
md("""## 2. Observation vs prediction period (no leakage)
```
 features: everything up to cut-off T  |  label: any purchase in (T, T+H] ?
```
Eligible customers are those still active at T (last order <= H days before T). Customers silent for more than H days are already churned by definition, so scoring them would be pointless."""),
code(r'''
train = make_snapshots(orders, lines, returns, customers, train_c, H, data_end=END)
val = build_snapshot(orders, lines, returns, customers, val_c, H, data_end=END).reset_index()
test = build_snapshot(orders, lines, returns, customers, test_c, H, data_end=END).reset_index()
tbl = pd.DataFrame({"split": ["train", "validation", "test"], "cut-off": [f"{train_c[0].date()} ... {train_c[-1].date()} ({len(train_c)} snapshots)", val_c.date(), test_c.date()],
                    "rows": [len(train), len(val), len(test)], "churn rate": [train.churned.mean(), val.churned.mean(), test.churned.mean()]})
display(tbl.round(3))
'''),
md("""**Why a time-based split?** Random splitting would put the same customer's future behaviour into training and its past into the test set, and would let the model learn from time periods it is supposed to predict. Here training labels are fully resolved before validation starts, and validation labels before test starts: the test set is a genuine *future*.

The cut-offs above come from the dataset's own first/last dates. If the churn rate differs noticeably between train, validation and test, that is *dataset shift* (e.g. seasonality) and shows up later in the calibration plot."""),
md("## 3. Models and metrics"),
code(r'''
comp = pd.read_csv(DATA_PROC / "model_comparison.csv")
display(comp[["model","val_pr_auc","roc_auc","pr_auc","brier","precision","recall","f1","accuracy"]].round(3))
'''),
md("""Three meaningful approaches only: (1) logistic regression on **recency alone** - what a simple "silent customers" rule would capture; (2) logistic regression on all features - interpretable; (3) gradient boosting - can capture interactions/non-linearity. The model is **selected on the validation metric configured for the run** (see the banner / `selection_metric`) and the test set is only reported afterwards.

How to read the metrics for retention:
* **Accuracy** is misleading (predicting the majority class for everyone already scores 1 - base rate).
* **PR-AUC** (compare with the base rate = what a random ranking gets) measures how well high scores concentrate real churners.
* **Recall** = share of real churners we reach; **precision** = share of contacted customers who were really about to leave. High recall with low precision means contacting many customers who would have stayed.
* **Brier score** checks that probabilities are usable as probabilities."""),
code(r'''
from src.churn_model import load
model = load("churn_model.joblib"); cfg = json.load(open(DATA_PROC / "model_config.json")); feats = cfg["features"]
p = model.predict_proba(test[feats])[:, 1]; y = test.churned.values
thr = M["t_f1"]; m = ev.classification_metrics(y, p, thr)
print({k: round(v, 3) for k, v in m.items() if k in ["roc_auc", "pr_auc", "brier", "precision", "recall", "f1"]}, "| base rate", round(y.mean(), 3))
fig = ev.plot_roc_pr(y, {M["best_model"]: p}); plt.show()
ev.plot_confusion(m, f"Confusion matrix (test, threshold {thr:.2f})"); plt.show()
ev.plot_calibration(y, {M["best_model"]: p}); plt.show()
'''),
md("""## 4. From probability to risk tiers
Tiers are set on **validation** outcomes (not on fixed 30/70 cut-offs). The rule used for this run:

> {RULE}

The 2x-base-rate criterion is computed from this dataset's own base churn rate. Tiers are then checked on the untouched test set."""),
code(r'''
t_low, t_high = M["t_low"], M["t_high"]
display(Markdown("**Rule used for this run:** " + meta["risk_rule"]))
tiers = ev.risk_table(y, p, t_low, t_high); display(tiers.round(3))
ev.plot_risk_tiers(tiers); plt.show()
display(ev.decile_lift(y, p).round(3).T)
'''),
code(r'''
lo, hi = tiers.loc["Low"], tiers.loc["High"]
display(Markdown(f"""**So what?**
* Observed churn in the test period: **{hi.observed_churn_rate:.0%}** of High-tier customers vs **{lo.observed_churn_rate:.0%}** of Low-tier ones (overall {y.mean():.0%}). The top decile of scores churns at ~{ev.decile_lift(y,p).lift.iloc[0]:.1f}x the base rate.
* It is **not** a crystal ball: test ROC-AUC {m['roc_auc']:.2f}, PR-AUC {m['pr_auc']:.2f} (a random ranking gets {y.mean():.2f}). Part of churn is not predictable from purchase history alone.
* Compare the models in the table above: if 'Logistic (recency only)' is close to the others, a simple "silent for N days" rule captures most of the signal and is the benchmark the model must beat."""))
'''),
md("""## 5. Threshold and business trade-off
The F1-optimal threshold is a statistical convention. The right threshold depends on what an intervention costs and what a saved customer is worth. The numbers below are **illustrative assumptions**, to be replaced by real costs and measured save rates (from an A/B test)."""),
code(r'''
margin = (test.monetary / test.days_since_first_order.clip(lower=90) * 365 * meta["gross_margin"]).values
scen = {"Cheap email (cost 1, 5% saved)": (1, .05), "Phone call (cost 20, 25% saved)": (20, .25)}
rows = []
for th in np.arange(0.0, 0.75, 0.05):
    sel = p >= th
    r = {"threshold": round(th, 2), "contacted": int(sel.sum()), "churners reached": int(y[sel].sum()), "recall": y[sel].sum() / y.sum()}
    for name, (cost, save) in scen.items():
        r[name] = (y[sel] * margin[sel]).sum() * save - cost * sel.sum()
    rows.append(r)
trade = pd.DataFrame(rows).set_index("threshold"); display(trade.round(2))
best = {n: float(trade[n].idxmax()) for n in scen}
display(Markdown(f"**So what?** Under these ILLUSTRATIVE costs, net benefit peaks at different thresholds for different interventions ({'; '.join(f'{k}: {v:.2f}' for k, v in best.items())}; threshold 0.00 = contact everyone). A cheap, automated email can be sent broadly (low threshold, high recall); an expensive personal call should be reserved for the highest-risk, highest-value customers. This is why the final system adds a *value* dimension (notebook 06)."))
'''),
]

# ---------------------------------------------------------------- 06
NOTEBOOKS["06_model_explainability"] = [
md("""# 06 - Explainability, prioritisation and CLV proxy
**Questions:** why does the model flag a customer, and whom should the company contact first?

> Explanations describe how the **model** reaches a score. They do **not** show what causes a customer to leave."""),
code(SETUP), code(CORE),
code(r'''
import shap
from src.churn_model import load
from src import explainability as xai
model = load("churn_model.joblib"); feats = json.load(open(DATA_PROC / "model_config.json"))["features"]
print("selected model:", meta["model_selected"], "| features:", feats)
test = build_snapshot(orders, lines, returns, customers, test_c, H, data_end=END).reset_index()
X, y = test[feats], test.churned
sv, base = xai.shap_values(model, X)
# SHAP must reproduce the model: base value + sum of contributions = the model's own log-odds
logit = np.log(model.predict_proba(X)[:, 1] / (1 - model.predict_proba(X)[:, 1]))
print("max |base + sum(SHAP) - model log-odds| =", float(np.abs(base + sv.sum(1) - logit).max()))
gi = xai.global_importance(sv, X); perm = pd.read_csv(DATA_PROC / "permutation_importance_test.csv")
display(pd.concat([gi.head(8), perm.head(8)], axis=1))
'''),
code(r'''
shap.summary_plot(sv, X, show=False, plot_size=(9, 6)); plt.show()
'''),
code(r'''
top_shap, top_perm = gi.feature.iloc[0], perm.feature.iloc[0]
display(Markdown(f"""**Reading the importances.** SHAP ranks `{top_shap}` first; permutation importance (drop in test PR-AUC when a feature is shuffled) ranks `{top_perm}` first. Agreement between two independent methods increases confidence; disagreement is a reason to look closer.
Features that share information (frequency, 90-day orders, spend...) split the credit among themselves. Importances describe what the *model* uses on *this* dataset; they are not universal laws and not causes."""))
'''),
md("## Individual explanations"),
code(r'''
p = model.predict_proba(X)[:, 1]
for label, i in [("Highest-risk customer", int(np.argmax(p))), ("A low-risk customer", int(np.argsort(p)[len(p)//10]))]:
    up, down = xai.explain_row(sv[i], X.iloc[i], 3)
    print(f"{label}: {test.customer_id[i]} | churn probability {p[i]:.0%} | actual outcome: {'churned' if y.iloc[i] else 'retained'}")
    print("  Factors increasing risk:"); [print("   +", s) for s in up]
    print("  Factors reducing risk:");   [print("   -", s) for s in down]; print()
shap.plots.waterfall(shap.Explanation(sv[int(np.argmax(p))], base, X.iloc[int(np.argmax(p))].to_numpy(), feature_names=feats), show=False); plt.show()
'''),
md("""## Prioritisation: risk x value
Churn probability alone is not a plan. Two High-risk customers with very different annual value deserve different responses. The playbook below maps (risk tier, value tier) to a *suggested* action. It supports a human decision, it does not make it."""),
code(r'''
from src.prioritization import PLAYBOOK, WINBACK
cs = pd.read_csv(DATA_PROC / "customers_scored.csv.gz")
pb = pd.DataFrame([(r, v, *a) for (r, v), a in PLAYBOOK.items()], columns=["risk", "value", "priority", "action"]); display(pb)
act = cs[cs.status == "Active"]
display(pd.crosstab(act.risk_level, act.value_tier).reindex(index=["High","Medium","Low"], columns=["High","Medium","Low"]))
summary = act.groupby("priority").agg(customers=("customer_id", "size"), expected_margin_at_risk=("margin_at_risk", "sum"), mean_churn_prob=("churn_prob", "mean")).round(2); display(summary)
'''),
md("""## CLV proxy - assumptions stated plainly
`clv_12m_proxy = annual margin run-rate x average survival over the next 12 months`, where run-rate = historical spend / observed life (at least 90 days) x 365, margin = a flat assumed share of revenue (`business.gross_margin` in the config, shown in the banner of the dashboard), and the model's churn probability for one H-day window is assumed constant in each future window. This is a **heuristic**. A rigorous CLV would use a probabilistic purchase model (e.g. BG/NBD + Gamma-Gamma) validated on a hold-out period."""),
code(r'''
top = act.sort_values("priority_score", ascending=False).head(10)[["customer_id","rfm_segment","churn_prob","risk_level","annual_margin_run_rate","clv_12m_proxy","margin_at_risk","priority"]]
display(top.round(2))
p1 = act[act.priority.str.startswith("P1")]
display(Markdown(f"""**So what?** Of {len(act):,} active customers, **{len(p1)}** are both High-risk and High-value (P1) with ~{p1.margin_at_risk.sum():,.0f} of expected annual margin at stake: compare this with the size of the other groups to judge whether personal outreach is feasible. Lower-value groups get automated, low-cost messages. {int((act.risk_level=='Low').sum()):,} Low-risk customers are deliberately **left alone**: contacting them wastes budget and discounting them gives away margin."""))
'''),
]


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in cells]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path = NB_DIR / f"{name}.ipynb"
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(Path(__file__).resolve().parents[1] / "notebooks")}})
    nbf.write(nb, path)
    print("built", path.name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build and execute the analysis notebooks for one run")
    ap.add_argument("--run", default=None, help="run name (data/runs/<name>); default = the default run")
    ap.add_argument("--out", default=None, help="output folder (default: notebooks/)")
    a = ap.parse_args()
    if a.run:
        os.environ["CIP_RUN"] = a.run
    if a.out:
        NB_DIR = Path(a.out).resolve()
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for n, c in NOTEBOOKS.items():
        build(n, c)
