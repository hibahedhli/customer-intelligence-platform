"""Inspection and cleaning of the STANDARDIZED tables (canonical names). Every decision is logged."""
import numpy as np
import pandas as pd

from . import config as C
from .adapter import DEFAULT_CFG, load_config
from .schema import TX_COLUMNS


def _parse_dates(s: pd.Series, fmt=None, dayfirst=False) -> pd.Series:
    """Parse dates robustly; returns midnight timestamps (NaT where unparseable)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    else:
        s = s.astype("string").str.strip()
        if fmt:
            out = pd.to_datetime(s, format=fmt, errors="coerce")
        else:
            out = pd.to_datetime(s, format="ISO8601", errors="coerce")
            order = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]
            us = ["%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y"]
            for f in (order + us if dayfirst else us + order):
                miss = out.isna() & s.notna()
                if not miss.any():
                    break
                out = out.fillna(pd.to_datetime(s.where(miss), format=f, errors="coerce"))
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def _canonical(s: pd.Series) -> pd.Series:
    """Map spelling variants (case/whitespace) to the most frequent surface form."""
    key = s.str.strip().str.lower()
    surface = s.str.strip().groupby(key).agg(lambda x: x.value_counts().idxmax())
    return key.map(surface)


def inspect_data(customers, tx, cfg=None):
    """Small tables/numbers describing data quality on the standardized (still uncleaned) frames."""
    cfg = cfg or DEFAULT_CFG
    rep = {"shapes": pd.DataFrame({"table": ["customers", "transactions"], "rows": [len(customers), len(tx)],
                                   "columns": [customers.shape[1], tx.shape[1]]})}
    for name, df in (("customers", customers), ("transactions", tx)):
        rep[f"{name}_profile"] = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                                               "missing_pct": (df.isna().mean() * 100).round(2), "n_unique": df.nunique()})
    d = _parse_dates(tx["date"], cfg.get("date_format"), cfg.get("dayfirst", False))
    age = pd.to_numeric(customers.customer_age, errors="coerce")
    q = tx.quantity
    issues = {
        "rows without customer id": int(tx.customer_id.isna().sum()),
        "exact duplicate rows": int(tx.duplicated(subset=[c for c in TX_COLUMNS if c != "transaction_id"]).sum()),
        "dates that could not be parsed": int(d.isna().sum()),
        "returns / cancellations (per configured rule)": int(tx.is_return.sum()),
        "negative quantity NOT flagged as return": int(((q < 0) & ~tx.is_return).sum()),
        "zero or missing quantity": int((q.fillna(0) == 0).sum()),
        "unit_price <= 0 or missing": int((tx.unit_price.fillna(0) <= 0).sum()),
        "discount outside [0, 1]": int(((tx.discount < 0) | (tx.discount > 1)).sum()),
        "impossible ages (<16 or >100)": int(((age < 16) | (age > 100)).sum()),
        "missing ages": int(age.isna().sum()),
    }
    if tx.category.notna().any():
        issues["category spellings (raw -> normalised)"] = f"{tx.category.nunique()} -> {tx.category.str.strip().str.lower().nunique()}"
    rep["issues"] = pd.Series(issues, name="count")
    return rep


def clean_data(customers, tx, cfg=None, availability=None):
    """
    Return (customers_clean, transactions_clean, cleaning_log, summary).
    `summary` counts normal / return / removed rows and the reason for each removal.
    """
    cfg = cfg or DEFAULT_CFG
    av = availability or {}
    log, removed = [], {}
    n_in = len(tx)

    def note(issue, n, decision, why):
        log.append({"issue": issue, "rows_affected": int(n), "decision": decision, "rationale": why})

    def drop(mask, reason):
        nonlocal tx
        n = int(mask.sum())
        if n:
            removed[reason] = removed.get(reason, 0) + n
            tx = tx[~mask]
        return n

    customers, tx = customers.copy(), tx.copy()
    fmt, dayfirst = cfg.get("date_format"), cfg.get("dayfirst", False)

    # ---------------- transactions ----------------
    iso = pd.to_datetime(tx["date"].astype("string"), format="ISO8601", errors="coerce") if not pd.api.types.is_datetime64_any_dtype(tx["date"]) else tx["date"]
    parsed = _parse_dates(tx["date"], fmt, dayfirst)
    non_iso = int((iso.isna() & parsed.notna()).sum())
    tx["date"] = parsed
    note("Dates in several formats", non_iso, "Parsed (ISO first, then day-first/month-first fallbacks)" if non_iso else "All ISO / native dates",
         "Time of day is dropped. Ambiguous formats should be fixed with `date_format` / `dayfirst` in the config.")
    n = drop(tx["date"].isna(), "unparseable date")
    if n:
        note("Unparseable dates", n, "Dropped", "A purchase without a date cannot be placed in time.")

    n = drop(tx.customer_id.isna(), "missing customer id")
    note("Rows without customer id", n, "Dropped (reported)" if n else "None found",
         "Cannot be attributed to a customer, so they cannot enter customer-level analytics (guest checkouts). Revenue totals therefore cover identified customers only.")

    for col in ("category", "payment_method"):
        if tx[col].notna().any():
            before = tx[col].nunique()
            tx[col] = _canonical(tx[col])
            note(f"Inconsistent spelling in {col}", before - tx[col].nunique(), "Normalised case/whitespace",
                 f"{before} spellings collapsed to {tx[col].nunique()} real values.")

    if cfg["cleaning"].get("drop_exact_duplicates", True):
        subset = [c for c in TX_COLUMNS if c != "transaction_id" or av.get("transaction_id", True)]
        n = drop(tx.duplicated(subset=subset), "exact duplicate row")
        note("Exact duplicate rows", n, "Dropped (kept first)",
             "Identical in every field: double-posted records. (Disable with cleaning.drop_exact_duplicates if repeated lines are legitimate.)")
    if av.get("transaction_id", True):
        n = drop(tx.transaction_id.duplicated(), "repeated transaction_id")
        note("Repeated transaction_id with different content", n, "Dropped (kept first)", "transaction_id must be unique.")

    note("Returns / cancellations (per configured rule)", tx.is_return.sum(), "Kept, flagged is_return=True",
         "Returns reduce net revenue but are NOT purchases (excluded from orders/frequency/churn activity).")

    q = tx.quantity
    bad_q = ((~tx.is_return) & ~(q > 0)) | (tx.is_return & ~(q < 0))
    n_neg = int(((q < 0) & ~tx.is_return).sum())
    n = drop(bad_q, "invalid quantity (zero / missing / negative but not a return)")
    note("Invalid quantity (zero, missing, or negative and NOT a return)", n, "Dropped",
         f"{n_neg:,} rows are negative but not flagged as returns by the return rule: the true value is unknown, so imputing would invent data. "
         "If these are real returns, change `returns.method` in the config.")

    mode = cfg["cleaning"].get("nonpositive_price", "drop")
    bad_p = ~(tx.unit_price > 0)
    if mode == "impute_product_median" and tx.product_id.notna().any():
        valid = tx[tx.unit_price > 0].groupby("product_id").unit_price.median()
        tx.loc[bad_p, "unit_price"] = tx.loc[bad_p, "product_id"].map(valid)
        note("unit_price <= 0 or missing", bad_p.sum(), "Imputed with the product's median valid price",
             "Products have stable list prices, so the product median is a defensible reconstruction.")
        n = drop(~(tx.unit_price > 0), "unrecoverable price")
        if n:
            note("Price unrecoverable", n, "Dropped", "No valid price exists for the product.")
    else:
        n = drop(bad_p, "invalid price (<= 0 or missing)")
        note("unit_price <= 0 or missing", n, "Dropped", "Zero/negative prices are free items, adjustments or errors; imputing would invent revenue.")

    m = (tx.discount > 1) & (tx.discount <= 100)
    tx.loc[m, "discount"] = tx.loc[m, "discount"] / 100
    if av.get("discount", True):
        note("Discount stored as a percentage (e.g. 15 instead of 0.15)", m.sum(), "Divided by 100", "Integers in 1..100 while valid discounts are fractions.")
    m = (tx.discount < 0) | (tx.discount > 1)
    tx.loc[m, "discount"] = 0.0
    if m.sum():
        note("Discount still out of range", m.sum(), "Set to 0", "Cannot be interpreted.")

    tx["transaction_value"] = tx.quantity * tx.unit_price * (1 - tx.discount)

    pur = tx[~tx.is_return]
    cut = pur.transaction_value.quantile(.999)
    tx["suspicious_value"] = (~tx.is_return) & (tx.transaction_value > cut)
    note("Unusually large line values (> 99.9th percentile)", tx.suspicious_value.sum(), "Kept, flagged suspicious_value",
         "Large baskets are plausible; they are real revenue and are handled by log-scaling, not deletion.")

    # ---------------- customers ----------------
    customers["customer_registration_date"] = _parse_dates(customers.customer_registration_date, fmt, dayfirst) \
        if customers.customer_registration_date.notna().any() else pd.NaT
    customers = customers.drop_duplicates("customer_id")
    unknown = ~tx.customer_id.isin(customers.customer_id)
    if unknown.any():
        extra = pd.DataFrame({"customer_id": tx.loc[unknown, "customer_id"].unique()})
        customers = pd.concat([customers, extra], ignore_index=True)
    note("Transactions of customers missing from the customer table", unknown.sum(),
         "Customers added to the table (demographics unknown)" if unknown.any() else "None found", "")

    first = tx[~tx.is_return].groupby("customer_id").date.min()
    reg = customers.customer_registration_date
    fill = reg.isna()
    customers.loc[fill, "customer_registration_date"] = customers.loc[fill, "customer_id"].map(first)
    if not av.get("registration_date", False):
        note("Registration date unavailable", len(customers), "Approximated by the first observed purchase date",
             "APPROXIMATION: customers may have registered earlier. Features that depend on it are excluded.")
    elif fill.any():
        note("Missing registration dates", fill.sum(), "Filled with the first purchase date", "Approximation.")

    if av.get("age", True) and customers.customer_age.notna().any():
        age = pd.to_numeric(customers.customer_age, errors="coerce")
        m = (age < 16) | (age > 100)
        customers["customer_age"] = age.mask(m)
        note("Impossible ages (<16 or >100)", m.sum(), "Set to missing (not deleted)",
             "The customer's purchases are still valid; only the age field is unreliable. Age is not used by the model.")
        note("Missing age", customers.customer_age.isna().sum(), "Left missing", "Age is descriptive only; no imputation.")
    else:
        customers["customer_age"] = np.nan
    if av.get("gender", True) and customers.customer_gender.notna().any():
        g = customers.customer_gender.astype("string").str.strip().str.lower().map(
            {"f": "Female", "female": "Female", "m": "Male", "male": "Male"})
        note("Gender spellings / missing", (customers.customer_gender.astype("string") != g).sum(), "Normalised to Female/Male/Unknown", "")
        customers["customer_gender"] = g.fillna("Unknown")
    else:
        customers["customer_gender"] = "Unknown"
    if av.get("location", True) and customers.customer_location.notna().any():
        loc = customers.customer_location.astype("string").str.strip()
        canon = loc.dropna().groupby(loc.dropna().str.lower()).agg(lambda x: x.value_counts().idxmax())
        customers["customer_location"] = loc.str.lower().map(canon).fillna("Unknown")
        note("Location spellings / missing", loc.isna().sum(), "Normalised; missing -> 'Unknown'", "")
    else:
        customers["customer_location"] = "Unknown"

    if av.get("registration_date", False):
        before_reg = (tx.merge(customers[["customer_id", "customer_registration_date"]], on="customer_id")
                      .eval("date < customer_registration_date").sum())
        note("Transactions dated before registration", before_reg, "None found" if before_reg == 0 else "Kept", "Consistency check.")

    tx = tx.sort_values(["date", "transaction_id"]).reset_index(drop=True)
    summary = {"rows_in": n_in, "rows_out": len(tx), "rows_removed": n_in - len(tx),
               "normal_lines_kept": int((~tx.is_return).sum()), "return_lines_kept": int(tx.is_return.sum()),
               "removed_by_reason": removed, "invalid_rows_removed": int(sum(v for k, v in removed.items() if k.startswith(("invalid", "unrecoverable", "unparseable", "repeated"))))}
    return customers.reset_index(drop=True), tx, pd.DataFrame(log), summary
