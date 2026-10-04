"""
End-to-end pipeline (dataset-agnostic):

    python -m src.pipeline                                   # synthetic demo (configs/synthetic.yaml)
    python -m src.pipeline --config configs/my_data.yaml     # any dataset described by a YAML adapter config
    python -m src.pipeline --config configs/my_data.yaml --validate-only

raw file(s) -> adapter (canonical schema) -> cleaning -> validation -> churn window -> RFM -> clusters
-> time-based snapshots -> models (selected on VALIDATION) -> test once -> SHAP -> prioritisation -> files
"""
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import eda, evaluation as ev, explainability as xai
from .adapter import DatasetValidationError, load_config, mapping_table, standardize
from .churn_model import run_modeling, evaluate_on_test, save
from .data_processing import clean_data, inspect_data
from .feature_engineering import (FEATURES, build_orders, build_snapshot, compute_features, cutoff_schedule,
                                  make_snapshots, select_features, window_sensitivity)
from .prioritization import add_value_and_priority
from .segmentation import (CLUSTER_FEATURES, MIN_CUSTOMERS_FOR_CLUSTERING, choose_k, evaluate_k, fit_kmeans, name_clusters,
                           prepare_matrix, rfm_quality, rfm_scores, rfm_segments, stability)
from .validation import check_target, validate_dataset

np.random.seed(C.SEED)


def _wipe_outputs():
    for d in (C.DATA_PROC, C.FIGURES):
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.name != ".gitkeep":
                    f.unlink()


def main(config=None, run=None, validate_only=False, input_files=None):
    cfg = load_config(config)
    C.set_run(run if run is not None else cfg.get("run"))
    if input_files:
        cfg["source"]["transactions"] = list(input_files)
    C.ensure_dirs()
    _wipe_outputs()
    P, name = C.DATA_PROC, cfg["name"]
    an, md_cfg = cfg["analysis"], cfg["modeling"]
    seed = md_cfg.get("seed", C.SEED)
    M = {}
    print(f"=== {name} ({cfg['kind']}) | run: {C.RUN or 'default'} | config: {cfg['_path']}")

    # 1. adapter -> canonical tables -> cleaning -------------------------------------------------------------
    customers_std, tx_std, av, notes = standardize(cfg)
    inspect_data(customers_std, tx_std, cfg)["issues"].to_csv(P / "raw_data_issues.csv")
    customers, tx, log, summary = clean_data(customers_std, tx_std, cfg, av)
    ex = av.get("n_lines_excluded_non_product", 0)          # removed by the adapter's configured non-product filter
    if ex:
        summary["removed_by_reason"] = {"non-product line (configured filter, e.g. postage / fees)": ex, **summary["removed_by_reason"]}
        summary["rows_in"] += ex; summary["rows_removed"] += ex
    if an.get("data_start") or an.get("data_end"):                       # optional manual date window
        lo = pd.Timestamp(an["data_start"]) if an.get("data_start") else tx.date.min()
        hi = pd.Timestamp(an["data_end"]) if an.get("data_end") else tx.date.max()
        out = ~tx.date.between(lo, hi)
        if out.any():
            summary["removed_by_reason"]["outside configured date window"] = int(out.sum())
            summary["rows_removed"] += int(out.sum()); summary["rows_out"] -= int(out.sum())
            tx = tx[~out].reset_index(drop=True)
    orders = build_orders(tx)
    lines, returns = tx[~tx.is_return], tx[tx.is_return]

    rep = validate_dataset(name, cfg, customers_std, tx_std, customers, tx, orders, av, summary, notes)
    print(rep.render())
    if validate_only or rep.errors:
        rep.save(P)
        rep.raise_if_errors()
        return rep

    customers.to_csv(P / "customers_clean.csv", index=False)
    tx.to_csv(P / "transactions_clean.csv.gz", index=False)
    log.to_csv(P / "cleaning_log.csv", index=False)
    mapping_table(cfg).to_csv(P / "column_mapping.csv", index=False)
    json.dump(summary, open(P / "cleaning_summary.json", "w"), indent=2, default=int)
    monthly = eda.monthly_summary(tx, orders); monthly.to_csv(P / "monthly_summary.csv")
    f = rep.facts
    start, end = pd.Timestamp(f["data_start"]), pd.Timestamp(f["data_end"])
    H = int(f["churn_window"])
    M.update(dataset_name=name, dataset_kind=cfg["kind"], data_start=f["data_start"], data_end=f["data_end"],
             n_customers_total=len(customers), n_customers_with_orders=int(orders.customer_id.nunique()),
             n_orders=len(orders), n_lines=len(lines), gross_revenue=float(lines.transaction_value.sum()),
             net_revenue=float(tx.transaction_value.sum()), return_value=float(-returns.transaction_value.sum()))

    # 2. churn window (derived in validation from the data) + chronological cut-offs ------------------------------
    sens = window_sensitivity(orders, lines, returns, customers, start, end, H)
    sens.to_csv(P / "churn_window_sensitivity.csv", index=False)
    rep.facts["window_sensitivity"] = sens.to_dict("records")
    M.update(H=H, gap_quantiles=f.get("gap_quantiles", {}), churn_window_source=f["churn_window_source"])
    train_c, val_c, test_c = cutoff_schedule(H, start, end, an["snapshot_step_days"], an["min_observation_days"])
    M.update(val_cutoff=str(val_c.date()), test_cutoff=str(test_c.date()), train_cutoffs=[str(c.date()) for c in train_c])

    # 3. customer table at the end of the data (RFM + clusters) ---------------------------------------------------------
    f_all = compute_features(orders, lines, returns, customers, end)
    cust = rfm_segments(rfm_scores(f_all), new_days=an["new_customer_days"])
    cust["status"] = np.where(cust.recency_days <= H, "Active", "Lapsed")
    rq = rfm_quality(cust)
    seg = cust.groupby("rfm_segment").agg(customers=("R", "size"), revenue=("monetary", "sum"), median_recency=("recency_days", "median"),
                                          median_frequency=("frequency", "median"), median_spend=("monetary", "median"),
                                          lapsed_share=("status", lambda s: (s == "Lapsed").mean()))
    seg["customer_share"] = seg.customers / seg.customers.sum(); seg["revenue_share"] = seg.revenue / seg.revenue.sum()
    seg.to_csv(P / "rfm_segment_summary.csv")

    clustering = {"ran": False, "reason": None, "features": []}
    cust["cluster"], cust["cluster_name"] = -1, "Not clustered"
    cl_feats = [c for c in CLUSTER_FEATURES if cust[c].notna().any() and cust[c].nunique() > 1]
    if len(cust) < MIN_CUSTOMERS_FOR_CLUSTERING or len(cl_feats) < 3:
        clustering["reason"] = f"only {len(cust)} customers / {len(cl_feats)} usable features (need >= {MIN_CUSTOMERS_FOR_CLUSTERING} / 3)"
        print(f"[WARN] clustering skipped: {clustering['reason']}")
        X = None
    else:
        X, _ = prepare_matrix(cust, cl_feats)
        kev = evaluate_k(X, seed=seed); k = choose_k(kev); km = fit_kmeans(X, k, seed)
        cust["cluster"] = km.labels_
        prof = cust.groupby("cluster")[cl_feats].mean()
        cnames = name_clusters(prof, cust)
        cust["cluster_name"] = cust.cluster.map(cnames)
        cprof = prof.copy(); cprof["customers"] = cust.groupby("cluster").size(); cprof["revenue_share"] = cust.groupby("cluster").monetary.sum() / cust.monetary.sum()
        cprof["lapsed_share"] = cust.groupby("cluster").status.apply(lambda s: (s == "Lapsed").mean()); cprof["name"] = cprof.index.map(cnames)
        cprof.to_csv(P / "cluster_profile.csv"); kev.to_csv(P / "k_selection.csv", index=False)
        M.update(k=int(k), silhouette=float(kev.set_index("k").silhouette[k]), cluster_stability_ari=stability(X, k))
        clustering.update(ran=True, features=cl_feats)

    # 4. supervised snapshots (time-based split) -----------------------------------------------------------------------------
    train = make_snapshots(orders, lines, returns, customers, train_c, H, data_end=end)
    val = build_snapshot(orders, lines, returns, customers, val_c, H, data_end=end).reset_index()
    test = build_snapshot(orders, lines, returns, customers, test_c, H, data_end=end).reset_index()
    check_target(rep, train, val, test, H)
    rep.save(P)
    rep.raise_if_errors()
    features, excluded = select_features(train, av)
    M.update(n_train=len(train), n_val=len(val), n_test=len(test), churn_rate_train=float(train.churned.mean()),
             churn_rate_val=float(val.churned.mean()), churn_rate_test=float(test.churned.mean()))
    yte, yva = test.churned, val.churned

    # 5. models: fit + SELECT on train/validation only; test is scored once afterwards -----------------------------------
    result = run_modeling(train, val, features, metric=md_cfg["selection_metric"], seed=seed)
    result["grid"].to_csv(P / "gb_tuning_grid.csv", index=False)
    comp, pt = evaluate_on_test(result, test)
    comp.to_csv(P / "model_comparison.csv", index=False)
    best_name = result["best_name"]; M["best_model"] = best_name; M["selection_metric"] = f"validation {md_cfg['selection_metric']}"
    final, fcols = result["models"][best_name]
    save(final, "churn_model.joblib")

    rt = cfg["risk_tiers"]
    pv = result["val_probs"][best_name]
    t_low, t_high, tinfo = ev.risk_thresholds(yva, pv, rt["high_min_lift"], rt["low_max_churn"], return_info=True)
    t_f1 = result["thresholds"][best_name]
    tiers_val = ev.risk_table(yva, pv, t_low, t_high); tiers_test = ev.risk_table(yte, pt[best_name], t_low, t_high)
    tiers_test.to_csv(P / "risk_tiers_test.csv"); tiers_val.to_csv(P / "risk_tiers_val.csv")
    ev.decile_lift(yte, pt[best_name]).to_csv(P / "decile_lift_test.csv")
    M.update(t_low=t_low, t_high=t_high, t_f1=t_f1, test_metrics=ev.classification_metrics(yte, pt[best_name], t_f1))
    json.dump({"H": H, "t_low": t_low, "t_high": t_high, "features": fcols}, open(P / "model_config.json", "w"))

    # 6. figures --------------------------------------------------------------------------------------------------------------
    gaps_fig = None
    from .feature_engineering import interpurchase_gaps
    gaps = interpurchase_gaps(orders)
    qs = gaps.quantile([.5, .75, .9, .95, .99])
    eda.fig_revenue_trends(monthly); eda.fig_customer_distributions(cust); eda.fig_gap_distribution(gaps, H, qs)
    if len(sens):
        eda.fig_window_sensitivity(sens, H)
    if av.get("category"):
        eda.fig_category_revenue(tx)
    eda.fig_cohort_retention(orders); eda.fig_repeat_vs_onetime(cust); eda.fig_recency_vs_future(train, H)
    eda.fig_churn_vs_retained(train, features); eda.fig_corr(train, features); eda.fig_rfm(seg)
    if clustering["ran"]:
        eda.fig_k_selection(kev, k); eda.fig_clusters_pca(X, km.labels_, cnames); eda.fig_cluster_heatmap(prof, cnames)
    for fig, nm in ((ev.plot_roc_pr(yte, pt if len(pt) < 8 else {best_name: pt[best_name]}), "15_roc_pr"),
                    (ev.plot_confusion(M["test_metrics"], f"Confusion matrix (test, thr={t_f1:.2f})"), "16_confusion"),
                    (ev.plot_calibration(yte, pt), "17_calibration"), (ev.plot_risk_tiers(tiers_test), "18_risk_tiers")):
        eda.save(fig, nm); plt.close(fig)
    plt.close("all")

    # 7. explainability (whichever model was selected) ---------------------------------------------------------------------------
    perm = xai.permutation_table(final, test[fcols], yte, seed=seed); perm.to_csv(P / "permutation_importance_test.csv", index=False)

    # 8. score today's active customers -----------------------------------------------------------------------------------------------
    cust["churn_prob"] = np.nan; cust["risk_level"] = pd.Categorical([None] * len(cust), categories=ev.RISK_ORDER, ordered=True)
    cust["top_risk_factors"] = ""; cust["top_protective_factors"] = ""
    act = cust.status == "Active"
    Xa = cust.loc[act, fcols]
    cust.loc[act, "churn_prob"] = final.predict_proba(Xa)[:, 1]
    cust.loc[act, "risk_level"] = ev.assign_risk(cust.loc[act, "churn_prob"], t_low, t_high)
    import shap
    sv, base = xai.shap_values(final, Xa, seed=seed)
    gi = xai.global_importance(sv, Xa); gi.to_csv(P / "shap_global.csv", index=False)
    et = xai.explanation_table(sv, Xa); cust.loc[act, ["top_risk_factors", "top_protective_factors"]] = et.values
    plt.figure(); shap.summary_plot(sv, Xa, show=False, plot_size=(9, 7)); eda.save(plt.gcf(), "19_shap_beeswarm"); plt.close("all")
    plt.figure(); shap.summary_plot(sv, Xa, plot_type="bar", show=False, plot_size=(8, 6)); eda.save(plt.gcf(), "20_shap_bar"); plt.close("all")
    i = int(np.argmax(cust.loc[act, "churn_prob"].to_numpy()))
    plt.figure(); shap.plots.waterfall(shap.Explanation(sv[i], base, Xa.iloc[i].to_numpy(), feature_names=list(Xa.columns)), show=False)
    eda.save(plt.gcf(), "21_shap_waterfall_example"); plt.close("all")
    cust["risk_level"] = cust.risk_level.astype(object).where(act, "Lapsed")
    cust = cust.join(customers.set_index("customer_id")[["customer_age", "customer_gender", "customer_location"]])
    if av.get("category"):
        fav = lines.groupby(["customer_id", "category"]).transaction_value.sum().reset_index().sort_values("transaction_value").groupby("customer_id").tail(1)
        cust = cust.join(fav.set_index("customer_id").category.rename("favorite_category"))
    else:
        cust["favorite_category"] = "Not available"
    cust = add_value_and_priority(cust, H, margin=cfg["business"]["gross_margin"])
    cust.reset_index().to_csv(P / "customers_scored.csv.gz", index=False)

    M.update(n_active=int(act.sum()), n_lapsed=int((~act).sum()), lapsed_share=float((~act).mean()),
             n_high_risk=int((cust.risk_level == "High").sum()), avg_customer_value=float(cust.monetary.mean()),
             mean_pred_churn_active=float(cust.churn_prob.mean()))
    json.dump(M, open(P / "metrics.json", "w"), indent=2, default=float)

    # 9. run metadata: everything the dashboard / README need to describe THIS run -------------------------------------------------------
    base_rate = float(yva.mean())
    meta = dict(
        name=name, kind=cfg["kind"], description=cfg["description"].strip(), config=cfg["_path"], run=C.RUN or "default",
        availability={k: v for k, v in av.items() if isinstance(v, (bool, str))}, return_rule=av["return_rule"],
        cleaning=summary, data_start=f["data_start"], data_end=f["data_end"], history_days=f["history_days"],
        churn_window=H, churn_window_source=f["churn_window_source"], features_used=fcols, features_excluded=excluded,
        selection_metric=md_cfg["selection_metric"], model_selected=best_name, seed=seed,
        gross_margin=cfg["business"]["gross_margin"], clustering=clustering, rfm_quality=rq,
        risk_rule=(f"High = churn probability >= {t_high:.2f}: on validation data, customers at or above it churned at >= "
                   f"{rt['high_min_lift']:g}x the validation base rate ({base_rate:.1%}). "
                   f"Low = probability <= {t_low:.2f}: validation churn rate <= {rt['low_max_churn']:.0%}. Medium = in between."),
        risk_rule_info=tinfo, validation_checks=rep.checks,
        warnings=[c["title"] + (": " + c["detail"] if c["detail"] else "") for c in rep.warnings] + rq["warnings"],
    )
    json.dump(meta, open(P / "run_metadata.json", "w"), indent=2, default=str)
    print(f"\nModel comparison (test set; model chosen on validation {md_cfg['selection_metric']}): selected = {best_name}")
    print(comp[["model", "val_pr_auc", "roc_auc", "pr_auc", "precision", "recall", "f1"]].round(3).to_string())
    print(f"Outputs: {P}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Customer retention analytics pipeline")
    ap.add_argument("--config", default=None, help="YAML dataset config (default: configs/synthetic.yaml)")
    ap.add_argument("--run", default=None, help="output run name (default: the config's `run`, i.e. default folders for the demo)")
    ap.add_argument("--input", nargs="+", default=None, help="override source.transactions with these file(s)")
    ap.add_argument("--validate-only", action="store_true", help="only standardize, clean and print the validation report")
    a = ap.parse_args()
    try:
        main(a.config, a.run, a.validate_only, a.input)
    except DatasetValidationError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
