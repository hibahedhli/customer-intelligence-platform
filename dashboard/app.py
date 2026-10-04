"""
Customer Intelligence & Retention Analytics dashboard (dataset-independent).

    streamlit run dashboard/app.py

It shows whichever run was produced by `python -m src.pipeline [--config ...]`. Pick the run in the sidebar.
Everything (dates, thresholds, window, available fields) is read from the run's output files.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

st.set_page_config(page_title="Customer Retention Analytics", page_icon=":bar_chart:", layout="wide")
RISK_COLORS = {"Low": "#59a14f", "Medium": "#edc948", "High": "#e15759", "Lapsed": "#9e9e9e"}
NA = "Not available in this dataset"


def _opt(path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else None


@st.cache_data
def load(run):
    P = C.paths_for(None if run == "default" else run)["processed"]
    D = dict(
        cust=pd.read_csv(P / "customers_scored.csv.gz", parse_dates=["last_order", "first_order"]),
        tx=pd.read_csv(P / "transactions_clean.csv.gz", parse_dates=["date"]),
        M=json.load(open(P / "metrics.json")), meta=json.load(open(P / "run_metadata.json")),
        monthly=pd.read_csv(P / "monthly_summary.csv", parse_dates=[0], index_col=0),
        seg=pd.read_csv(P / "rfm_segment_summary.csv"), comp=pd.read_csv(P / "model_comparison.csv"),
        tiers=pd.read_csv(P / "risk_tiers_test.csv"), shap=pd.read_csv(P / "shap_global.csv"),
        sens=pd.read_csv(P / "churn_window_sensitivity.csv"), log=pd.read_csv(P / "cleaning_log.csv"),
        mapping=_opt(P / "column_mapping.csv"), clu=_opt(P / "cluster_profile.csv"))
    return D


runs = C.list_runs()
if not runs:
    st.error("No results found. Run `python -m src.generate_data` and `python -m src.pipeline` (demo), or "
             "`python -m src.pipeline --config configs/<your_dataset>.yaml`, then reload.")
    st.stop()
default = os.environ.get("CIP_RUN") or "default"
run = st.sidebar.selectbox("Dataset run", runs, index=runs.index(default) if default in runs else 0,
                           help="Each run is one dataset processed by the same pipeline.")
D = load(run)
cust, tx, M, meta = D["cust"], D["tx"], D["M"], D["meta"]
av, H = meta["availability"], meta["churn_window"]
active = cust[cust.status == "Active"]
has = lambda k: bool(av.get(k))

st.sidebar.markdown(f"**{meta['name']}**  \n{meta['data_start']} -> {meta['data_end']}  \nchurn window: **{H} days**")
if meta["kind"] == "synthetic":
    st.sidebar.warning("SIMULATED data: demonstrates the method, says nothing about real customers.")
st.title("Customer Intelligence & Retention Analytics")
st.caption(f"Dataset: **{meta['name']}** ({'simulated' if meta['kind'] == 'synthetic' else 'real'}) | data {meta['data_start']} -> {meta['data_end']} | "
           f"churn = no purchase in the {H} days after the cut-off ({meta['churn_window_source']}).")
if meta["warnings"]:
    with st.expander(f"{len(meta['warnings'])} data warnings for this dataset"):
        for w in meta["warnings"]:
            st.markdown(f"- {w}")

tab_over, tab_seg, tab_churn, tab_risk, tab_cust, tab_about = st.tabs(
    ["Overview", "Customer segmentation", "Churn analytics", "At-risk customers", "Individual customer", "Method & data"])

# ------------------------------------------------------------------ Overview
with tab_over:
    c = st.columns(6)
    c[0].metric("Customers (with orders)", f"{M['n_customers_with_orders']:,}")
    c[1].metric("Net revenue", f"{M['net_revenue']:,.0f}", help="After returns, identified customers only")
    c[2].metric("Active customers", f"{M['n_active']:,}", help=f"Bought within the last {H} days")
    c[3].metric("Lapsed customers", f"{M['lapsed_share']:.0%}", help=f"No purchase for more than {H} days: already churned by definition")
    c[4].metric("Avg customer value", f"{M['avg_customer_value']:,.0f}", help="Historical spend per customer")
    c[5].metric("High-risk (active)", f"{M['n_high_risk']:,}", help="Active customers in the High risk tier")
    st.caption(f"Orders: **{M['n_orders']:,}** | order lines: {M['n_lines']:,}. Observed churn rate among active customers in the test window: "
               f"**{M['churn_rate_test']:.1%}** (training windows: {M['churn_rate_train']:.1%}). Mean predicted churn today: **{M['mean_pred_churn_active']:.1%}**.")
    m = D["monthly"].reset_index(); m.columns = ["month"] + list(m.columns[1:])
    a, b = st.columns(2)
    a.plotly_chart(px.bar(m, x="month", y="net_revenue", title="Net revenue per month"), width="stretch")
    b.plotly_chart(px.line(m, x="month", y="active_customers", markers=True, title="Customers ordering per month"), width="stretch")
    st.plotly_chart(px.histogram(active, x="churn_prob", nbins=40, color="risk_level", color_discrete_map=RISK_COLORS,
                                 category_orders={"risk_level": ["Low", "Medium", "High"]}, title="Distribution of churn probability (active customers)"), width="stretch")

# ------------------------------------------------------------------ Segmentation
with tab_seg:
    st.subheader("RFM segments (rule-based)")
    for w in meta["rfm_quality"]["warnings"]:
        st.warning(w)
    order = ["Champions", "Loyal customers", "Potential loyalists", "New customers", "At-risk customers", "Hibernating"]
    seg = D["seg"].set_index("rfm_segment").reindex(order).dropna(how="all").reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(seg.melt("rfm_segment", ["customer_share", "revenue_share"]), x="rfm_segment", y="value", color="variable", barmode="group",
                          title="Share of customers vs share of revenue"), width="stretch")
    b.plotly_chart(px.bar(seg, x="rfm_segment", y="lapsed_share", title=f"Share already lapsed (> {H} days silent)", range_y=[0, 1]), width="stretch")
    st.dataframe(seg.round(2), width="stretch", hide_index=True)
    st.info("Segments that are defined by poor recency are mostly past the churn window already: RFM describes who has gone quiet; "
            "the churn model scores the customers who are still active. Segments with 0 customers do not appear.")
    st.subheader("Behavioural clusters (K-Means)")
    clu = D["clu"]
    if not meta["clustering"]["ran"] or clu is None:
        st.info(f"Clustering was not run: {meta['clustering']['reason']}")
    else:
        st.caption(f"k = {M['k']} chosen from the data (silhouette {M['silhouette']:.2f}, stability ARI {M['cluster_stability_ari']:.2f}); features: {', '.join(meta['clustering']['features'])}.")
        a, b = st.columns(2)
        a.plotly_chart(px.pie(clu, names="name", values="customers", title="Cluster sizes"), width="stretch")
        b.plotly_chart(px.bar(clu, x="name", y="revenue_share", title="Revenue share by cluster"), width="stretch")
        st.dataframe(clu.round(2), width="stretch", hide_index=True)
        pc = active.groupby("cluster_name").agg(active_customers=("customer_id", "size"), mean_churn_prob=("churn_prob", "mean")).reset_index()
        st.plotly_chart(px.bar(pc, x="cluster_name", y="mean_churn_prob", title="Mean predicted churn probability, active customers by cluster"), width="stretch")

# ------------------------------------------------------------------ Churn analytics
with tab_churn:
    st.subheader("Model evaluation (time-based test set)")
    st.dataframe(D["comp"][["model", "val_pr_auc", "roc_auc", "pr_auc", "precision", "recall", "f1"]].round(3), width="stretch", hide_index=True)
    st.caption(f"Model **{meta['model_selected']}** was selected on validation {meta['selection_metric'].replace('_', '-').upper()} only; the test set was scored once afterwards. "
               f"Test base rate = {M['churn_rate_test']:.1%}. Precision/recall use the F1-optimal validation threshold ({M['t_f1']:.2f}).")
    a, b = st.columns(2)
    t = D["tiers"]
    a.plotly_chart(px.bar(t, x="risk", y="observed_churn_rate", color="risk", color_discrete_map=RISK_COLORS, text=t.observed_churn_rate.map("{:.0%}".format),
                          title="Observed churn rate by risk tier (test)", category_orders={"risk": ["Low", "Medium", "High"]}), width="stretch")
    a.caption(f"Tiers for this run: Low <= {M['t_low']:.2f} < Medium < {M['t_high']:.2f} <= High. {meta['risk_rule']}")
    if meta["risk_rule_info"].get("high_fallback_to_quantile") or meta["risk_rule_info"].get("thresholds_crossed_used_terciles"):
        a.warning("The validation data could not support the standard tier rule, so quantile-based cut-offs were used for part of it.")
    top = D["shap"].head(12).iloc[::-1]
    b.plotly_chart(px.bar(top, x="mean_abs_shap", y="feature", orientation="h", title="Global importance (mean |SHAP|)"), width="stretch")
    b.caption("Importance describes the model, not the causes of customer behaviour.")
    if meta["features_excluded"]:
        with st.expander("Features not used in this dataset"):
            st.dataframe(pd.DataFrame(meta["features_excluded"].items(), columns=["feature", "reason"]), hide_index=True, width="stretch")
    st.subheader("Churn by customer characteristics (lapsed share = already churned)")
    ch = cust.copy()
    dims = ["rfm_segment"] + (["cluster_name"] if meta["clustering"]["ran"] else [])
    if has("location"): dims.insert(0, "customer_location")
    if has("category"): dims.insert(0, "favorite_category")
    if has("age"):
        ch["age_group"] = pd.cut(ch.customer_age, [0, 24, 34, 44, 54, 100], labels=["<=24", "25-34", "35-44", "45-54", "55+"]).astype(str).replace("nan", "Unknown")
        dims.append("age_group")
    if has("gender"): dims.append("customer_gender")
    missing = [n for n, k in (("location", "location"), ("category", "category"), ("age", "age"), ("gender", "gender")) if not has(k)]
    if missing:
        st.caption(f"Breakdowns by {', '.join(missing)}: {NA}.")
    dim = st.selectbox("Break down by", dims)
    g = ch.groupby(dim).agg(customers=("customer_id", "size"), lapsed_share=("status", lambda s: (s == "Lapsed").mean())).reset_index()
    st.plotly_chart(px.bar(g, x=dim, y="lapsed_share", hover_data=["customers"], range_y=[0, 1]), width="stretch")
    st.caption("Differences between groups are descriptive. They do not show that the characteristic causes churn.")
    st.subheader(f"Why H = {H} days?")
    st.plotly_chart(px.line(D["sens"], x="H_days", y="churn_rate", markers=True, title="Churn rate vs window length"), width="stretch")

# ------------------------------------------------------------------ At-risk table
with tab_risk:
    st.subheader("Retention worklist")
    f1, f2, f3, f4 = st.columns(4)
    rs = f1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium"])
    sg = f2.multiselect("RFM segment", sorted(active.rfm_segment.unique()))
    vt = f3.multiselect("Value tier", ["High", "Medium", "Low"])
    cl = f4.multiselect("Cluster", sorted(active.cluster_name.unique())) if meta["clustering"]["ran"] else []
    if not meta["clustering"]["ran"]:
        f4.caption(f"Cluster filter: {NA}")
    g1, g2 = st.columns(2)
    loc = g1.multiselect("Location", sorted(active.customer_location.dropna().unique())) if has("location") else []
    if not has("location"): g1.caption(f"Location filter: {NA}")
    cat = g2.multiselect("Favourite category", sorted(active.favorite_category.dropna().unique())) if has("category") else []
    if not has("category"): g2.caption(f"Category filter: {NA}")
    view = active[active.risk_level.isin(rs)]
    for col, vals in (("rfm_segment", sg), ("value_tier", vt), ("cluster_name", cl), ("customer_location", loc), ("favorite_category", cat)):
        if vals:
            view = view[view[col].isin(vals)]
    view = view.sort_values("priority_score", ascending=False)
    k = st.columns(3)
    k[0].metric("Customers in view", f"{len(view):,}")
    k[1].metric("Expected annual margin at risk", f"{view.margin_at_risk.sum():,.0f}")
    k[2].metric("Mean churn probability", f"{view.churn_prob.mean():.0%}" if len(view) else "-")
    show = view[["customer_id", "rfm_segment", "cluster_name", "churn_prob", "risk_level", "monetary", "clv_12m_proxy", "orders_90d", "recency_days", "last_order", "frequency", "priority", "suggested_action"]].rename(
        columns={"rfm_segment": "RFM segment", "cluster_name": "cluster", "churn_prob": "churn prob.", "monetary": "customer value (spend)", "clv_12m_proxy": "12m CLV proxy",
                 "orders_90d": "orders (last 90 d)", "recency_days": "days since last purchase", "last_order": "last purchase", "frequency": "orders"})
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "churn prob.": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        "customer value (spend)": st.column_config.NumberColumn(format="%.0f"), "12m CLV proxy": st.column_config.NumberColumn(format="%.0f"),
        "last purchase": st.column_config.DateColumn(format="YYYY-MM-DD")})
    st.download_button("Download worklist (CSV)", show.to_csv(index=False).encode(), "retention_worklist.csv", "text/csv")
    st.caption(f"Priority = churn tier x customer value tier (value = annualised margin run-rate, margin ASSUMED at {meta['gross_margin']:.0%}). "
               "Suggested actions are starting points for a human team, not automatic decisions.")
    lapsed = cust[cust.status == "Lapsed"].sort_values("annual_margin_run_rate", ascending=False)
    with st.expander(f"Already lapsed customers ({len(lapsed):,}) - win-back list"):
        st.dataframe(lapsed[["customer_id", "rfm_segment", "monetary", "recency_days", "frequency", "priority", "suggested_action"]].head(500), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Individual customer
with tab_cust:
    ids = list(cust.sort_values(["status", "priority_score"], ascending=[True, False]).customer_id)
    cid = st.selectbox("Customer", ids, index=0)
    r = cust[cust.customer_id == cid].iloc[0]
    h = tx[(tx.customer_id == cid)].sort_values("date")
    a, b, c3, d, e = st.columns(5)
    a.metric("Status", r.status); b.metric("Orders", f"{int(r.frequency)}"); c3.metric("Total spend", f"{r.monetary:,.0f}")
    d.metric("Days since last purchase", f"{int(r.recency_days)}"); e.metric("Avg days between orders", "-" if pd.isna(r.avg_gap_days) else f"{r.avg_gap_days:.0f}")
    bits = [f"**RFM segment:** {r.rfm_segment} (R{int(r.R)} F{int(r.F)} M{int(r.M)})", f"**Cluster:** {r.cluster_name}",
            f"**Location:** {r.customer_location if has('location') else NA}", f"**Favourite category:** {r.favorite_category if has('category') else NA}"]
    st.write("  |  ".join(bits))
    left, right = st.columns([1, 2])
    if r.status == "Active":
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=r.churn_prob * 100, number={"suffix": "%"}, title={"text": f"Churn probability ({r.risk_level} risk)"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[r.risk_level]},
                                              "steps": [{"range": [0, M["t_low"] * 100], "color": "#e8f3e6"}, {"range": [M["t_low"] * 100, M["t_high"] * 100], "color": "#fbf4d4"},
                                                        {"range": [M["t_high"] * 100, 100], "color": "#f9dada"}]}))
        gauge.update_layout(height=260, margin=dict(t=60, b=0))
        left.plotly_chart(gauge, width="stretch")
        right.markdown(f"**Priority:** {r.priority}  \n**Suggested action:** {r.suggested_action}  \n"
                       f"**12-month value proxy:** {r.clv_12m_proxy:,.0f}  |  **Expected margin at risk:** {r.margin_at_risk:,.0f}")
        x, y = st.columns(2)
        x.markdown("**Factors increasing risk**")
        for s in str(r.top_risk_factors).split(" | "):
            if s and s != "nan": x.markdown(f"- :red[+] {s}")
        y.markdown("**Factors reducing risk**")
        for s in str(r.top_protective_factors).split(" | "):
            if s and s != "nan": y.markdown(f"- :green[-] {s}")
        st.caption(f"Explanations show what drove the MODEL's score ({meta['model_selected']}, SHAP). They do not prove why the customer behaves this way.")
    else:
        left.warning(f"No purchase for {int(r.recency_days)} days (> {H}): already churned by definition.")
        right.markdown(f"**Priority:** {r.priority}  \n**Suggested action:** {r.suggested_action}")
    pur = h[~h.is_return]
    if len(pur):
        mm = pur.groupby(pur.date.dt.to_period("M").dt.to_timestamp()).transaction_value.sum().reset_index()
        st.plotly_chart(px.bar(mm, x="date", y="transaction_value", title="Spend per month"), width="stretch")
    st.markdown("**Purchase history**")
    hist_cols = ["date", "order_id", "product_id", "category", "quantity", "unit_price", "discount", "transaction_value"]
    hist_cols = [c for c in hist_cols if c in h and (h[c].notna().any()) and not (c == "discount" and not has("discount"))]
    st.dataframe(h[hist_cols].sort_values("date", ascending=False), hide_index=True, width="stretch")

# ------------------------------------------------------------------ About
with tab_about:
    tr = M["churn_rate_train"]; te = M["churn_rate_test"]
    shift = (f" The churn rate differs between training ({tr:.1%}) and test ({te:.1%}) periods; probabilities may therefore be somewhat miscalibrated on new data."
             if abs(tr - te) > 0.03 else "")
    st.markdown(f"""
**Dataset.** {meta['description'] or meta['name']}  Kind: **{'simulated' if meta['kind'] == 'synthetic' else 'real'}**.

**Churn definition.** An *active* customer (last purchase <= {H} days before the cut-off) is *churned* if they place no order in the next {H} days.
Window: {meta['churn_window_source']}.

**No leakage.** Features use only data up to each cut-off. Chronological cut-offs: {len(M['train_cutoffs'])} training, validation {M['val_cutoff']}, test {M['test_cutoff']}.
The model ({meta['model_selected']}) was **selected on validation {meta['selection_metric']}**; the test set was used once for the final report.

**Risk tiers.** {meta['risk_rule']}

**Value / CLV proxy.** Historical run-rate x assumed margin ({meta['gross_margin']:.0%}) x average survival implied by the churn probability: a heuristic, not a probabilistic CLV model.

**Registration date:** {meta['availability'].get('registration_date_source', 'n/a').replace('_', ' ')}.
**Return rule:** `{meta['return_rule'].get('method_used')}`{' ' + str(meta['return_rule'].get('prefixes', '')) if meta['return_rule'].get('prefixes') else ''}.
{shift}
""")
    st.markdown("**Features used by the model**: " + ", ".join(meta["features_used"]))
    st.markdown("**Data validation checks**")
    st.dataframe(pd.DataFrame(meta["validation_checks"]), hide_index=True, width="stretch")
    if D["mapping"] is not None:
        st.markdown("**Column mapping (original -> canonical)**")
        st.dataframe(D["mapping"], hide_index=True, width="stretch")
    st.markdown("**Data cleaning log**")
    st.dataframe(D["log"], hide_index=True, width="stretch")
