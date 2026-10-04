"""
Dataset validation report: clear errors / warnings BEFORE the expensive modelling steps.
Run alone with:  python -m src.pipeline --config configs/<yours>.yaml --validate-only
"""
import json

import numpy as np
import pandas as pd

from .adapter import DatasetValidationError, mapping_table
from .feature_engineering import (InsufficientHistoryError, choose_churn_window, cutoff_schedule, interpurchase_gaps,
                                  required_history_days)

ICON = {"ok": "[OK]  ", "warn": "[WARN]", "error": "[FAIL]"}


class ValidationReport:
    def __init__(self, name):
        self.name, self.checks, self.facts = name, [], {}

    def add(self, level, title, detail=""):
        self.checks.append({"level": level, "title": title, "detail": detail})

    @property
    def errors(self):
        return [c for c in self.checks if c["level"] == "error"]

    @property
    def warnings(self):
        return [c for c in self.checks if c["level"] == "warn"]

    def render(self) -> str:
        f = self.facts
        L = [f"Dataset validation: {self.name}", "-" * 60]
        L += [f"{ICON[c['level']]} {c['title']}" + (f"  ({c['detail']})" if c["detail"] else "") for c in self.checks]
        if "data_start" in f:
            L += ["", "Date range:", f"  {f['data_start']} -> {f['data_end']}  ({f['history_days']} days, {f['history_days'] / 30.4:.0f} months)"]
            L += ["", f"Customers (with >= 1 purchase): {f['n_customers']:,}",
                  f"Purchase orders: {f['n_orders']:,}   order lines: {f['n_lines']:,}",
                  f"Returns / cancellations kept: {f['n_returns']:,} ({f['return_share']:.1%} of lines)   [rule: {f['return_rule']}]",
                  f"Rows read: {f['rows_in']:,}  ->  rows kept: {f['rows_out']:,}  (removed {f['rows_removed']:,})"]
            for k, v in f["removed_by_reason"].items():
                L.append(f"    removed {v:,}  because: {k}")
        miss = {k: v for k, v in (f.get("missing_values") or {}).items() if v > 0}
        L += (["", "Missing values in the mapped fields (before cleaning):"] + [f"  {k}: {v:.1f}%" for k, v in miss.items()]) if miss else ["", "Missing values in the mapped fields: none"]
        if "churn_window" in f:
            L += ["", f"Recommended churn window: {f['churn_window']} days   [{f['churn_window_source']}]"]
            if f.get("gap_quantiles"):
                L.append("  gap quantiles (days between orders): " + ", ".join(f"P{int(float(k) * 100)}={v:.0f}" for k, v in f["gap_quantiles"].items()) + f"   from {f['n_gaps']:,} gaps")
        if "sufficient_history" in f:
            L += ["", f"Sufficient history: {'YES' if f['sufficient_history'] else 'NO'}   "
                      f"(have {f['history_days']} days, need about {f['required_history_days']} for a {f['churn_window']}-day horizon)"]
        return "\n".join(L)

    def raise_if_errors(self):
        if self.errors:
            raise DatasetValidationError("\n\n" + self.render() + "\n\nThe dataset cannot be analysed as configured:\n" +
                                         "\n".join(f"  - {e['title']}: {e['detail']}" for e in self.errors))

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "dataset_validation.md").write_text("```\n" + self.render() + "\n```\n")
        json.dump({"name": self.name, "checks": self.checks, "facts": self.facts}, open(folder / "dataset_validation.json", "w"), indent=2, default=str)


def validate_dataset(name, cfg, customers_std, tx_std, customers, tx, orders, availability, summary, notes=()):
    rep = ValidationReport(name)
    cols = cfg["columns"]
    required = {"customer_id": "Customer ID", "order_id": "Order / invoice ID", "date": "Transaction date",
                "quantity": "Quantity", "unit_price": "Unit price"}
    for k, label in required.items():
        src = cols.get(k) or (cols.get("transaction_id") if k == "order_id" else None)
        rep.add("ok", f"{label} found", f"'{src}' -> {k}")
    labels = {"discount": "Discount", "category": "Product category", "product_id": "Product id", "payment_method": "Payment method"}
    for k, label in labels.items():
        if not availability.get(k):
            rep.add("warn", f"{label} unavailable", "related features are excluded, not invented")
    if not availability.get("registration_date"):
        rep.add("warn", "Customer registration date unavailable", "approximated by first observed purchase")
    if not (availability.get("age") or availability.get("gender") or availability.get("location")):
        rep.add("warn", "Demographic information unavailable", "age / gender / location views show 'Not available'")
    else:
        for k in ("age", "gender", "location"):
            if not availability.get(k):
                rep.add("warn", f"Customer {k} unavailable")
    for n in notes:
        if "non-numeric" in n:
            rep.add("warn", n)

    f = rep.facts
    f["return_rule"] = availability["return_rule"].get("detected_by") or availability["return_rule"]["method_used"]
    f.update(rows_in=summary["rows_in"], rows_out=summary["rows_out"], rows_removed=summary["rows_removed"],
             removed_by_reason=summary["removed_by_reason"], n_returns=summary["return_lines_kept"], n_lines=summary["normal_lines_kept"])
    f["return_share"] = summary["return_lines_kept"] / max(summary["rows_out"], 1)
    f["missing_values"] = {c: float(tx_std[c].isna().mean() * 100) for c in ["customer_id", "order_id", "date", "quantity", "unit_price", "product_id", "category"] if availability.get(c, True)}

    if summary["rows_removed"] / max(summary["rows_in"], 1) > 0.5:
        rep.add("warn", "More than half of the rows were removed during cleaning", "check the column mapping and the return rule")
    if (summary["removed_by_reason"].get("invalid quantity (zero / missing / negative but not a return)", 0) / max(summary["rows_in"], 1)) > 0.05:
        rep.add("warn", "Many negative quantities are not covered by the return rule", "if they are returns, set returns.method (negative_quantity / id_prefix / flag_column)")
    if f["missing_values"].get("customer_id", 0) > 10:
        rep.add("warn", f"{f['missing_values']['customer_id']:.0f}% of rows have no customer id", "excluded from customer analytics (guest checkouts)")

    if orders.empty or orders.customer_id.nunique() < 50:
        rep.add("error", "Too few customers", f"{0 if orders.empty else orders.customer_id.nunique()} customers with a purchase; at least ~50 are required")
        return rep
    start, end = tx["date"].min(), tx["date"].max()
    f.update(data_start=str(start.date()), data_end=str(end.date()), history_days=int((end - start).days),
             n_customers=int(orders.customer_id.nunique()), n_orders=int(len(orders)))
    an = cfg["analysis"]
    if an.get("data_start") or an.get("data_end"):
        f["data_start"] = str(pd.Timestamp(an.get("data_start") or start).date())
        f["data_end"] = str(pd.Timestamp(an.get("data_end") or end).date())
        f["history_days"] = int((pd.Timestamp(f["data_end"]) - pd.Timestamp(f["data_start"])).days)
        rep.add("warn", "Date range overridden manually in the config", f"{f['data_start']} -> {f['data_end']}")

    gaps = interpurchase_gaps(orders)
    f["n_gaps"] = int(len(gaps))
    H = None
    if an.get("churn_window_days"):
        H = int(an["churn_window_days"])
        f["churn_window_source"] = "manual override (analysis.churn_window_days)"
    elif len(gaps) < an["min_gaps_for_window"]:
        rep.add("error", "Not enough repeat purchases to derive a churn window",
                f"only {len(gaps)} gaps between consecutive orders (need >= {an['min_gaps_for_window']}); set analysis.churn_window_days manually if you know the purchase cycle")
    else:
        H, qs = choose_churn_window(gaps, q=an["churn_gap_quantile"], step=15, lo=an["churn_window_min"], hi=an["churn_window_max"])
        raw_q = float(gaps.quantile(an["churn_gap_quantile"]))
        f["gap_quantiles"] = {str(k): float(v) for k, v in qs.items()}
        f["churn_window_source"] = f"P{int(an['churn_gap_quantile'] * 100)} of purchase gaps = {raw_q:.0f} days, rounded up to a multiple of 15"
        if raw_q > an["churn_window_max"] or raw_q < an["churn_window_min"] - 15:
            rep.add("warn", f"Derived window ({raw_q:.0f} days) is outside the safety range [{an['churn_window_min']}, {an['churn_window_max']}] and was clipped to {H} days")
    if H is not None:
        f["churn_window"] = H
        f["required_history_days"] = required_history_days(H, an["min_observation_days"], an["snapshot_step_days"])
        try:
            train, val, test = cutoff_schedule(H, f["data_start"], f["data_end"], an["snapshot_step_days"], an["min_observation_days"])
            f["sufficient_history"] = True
            f["n_train_cutoffs"], f["val_cutoff"], f["test_cutoff"] = len(train), str(val.date()), str(test.date())
            rep.add("ok", "Enough history for a time-based train / validation / test split", f"{len(train)} training cut-offs")
        except InsufficientHistoryError as e:
            f["sufficient_history"] = False
            rep.add("error", "Not enough history", str(e))
    return rep


def check_target(rep, train, val, test, H):
    """Is the churn target meaningful and stable enough to model? Adds checks to the report."""
    for nm, d, mn in (("train", train, 100), ("validation", val, 30), ("test", test, 30)):
        pos, neg = int(d.churned.sum()), int((1 - d.churned).sum())
        rep.facts[f"{nm}_rows"], rep.facts[f"{nm}_churn_rate"] = int(len(d)), float(d.churned.mean()) if len(d) else None
        if min(pos, neg) < mn:
            rep.add("error", f"Too few churned or retained customers in the {nm} set", f"{pos} churned / {neg} retained (need >= {mn} of each); try another churn window")
    r = rep.facts.get("train_churn_rate")
    if r is not None and not (0.05 <= r <= 0.60):
        rep.add("warn", f"Churn rate {r:.0%} is extreme for window H={H}", "very imbalanced targets give unstable metrics")
    sens = rep.facts.get("window_sensitivity")
    if sens:
        s = pd.DataFrame(sens).set_index("H_days").churn_rate
        near = [w for w in s.index if w != H]
        if H in s.index and near:
            nb = min(near, key=lambda w: abs(w - H))
            if abs(s[H] - s[nb]) > 0.10:
                rep.add("warn", "Churn rate changes a lot between neighbouring windows", f"{s[H]:.0%} at {H} d vs {s[nb]:.0%} at {nb} d: the definition is sensitive")
            else:
                rep.add("ok", "Churn rate is stable around the chosen window", f"{s[H]:.0%} at {H} d vs {s[nb]:.0%} at {nb} d")
    return rep
