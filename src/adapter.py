"""
Input adapter:  raw file(s) + YAML column mapping  ->  canonical standardized tables.

Everything dataset-specific (column names, file format, return convention, discount scale,
non-product codes) is declared in a YAML file under configs/. The modelling pipeline never
sees raw column names.
"""
import copy
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config as C
from .schema import CUST_COLUMNS, OPTIONAL_CUST, REQUIRED_TX, TX_COLUMNS


class DatasetValidationError(Exception):
    """Raised with a human-readable message when a dataset cannot be used as configured."""


DEFAULT_CFG = {
    "name": "dataset", "kind": "real", "description": "", "run": None, "extends": None,
    "source": {"transactions": [], "customers": None, "returns": None, "read_options": {}},
    "columns": {k: None for k in ["customer_id", "order_id", "transaction_id", "date", "product_id", "product_name",
                                  "category", "quantity", "unit_price", "discount", "payment_method"]},
    "customer_columns": {k: None for k in CUST_COLUMNS},
    "returns_columns": {},
    "date_format": None, "dayfirst": False,
    "discount": {"scale": "fraction"},
    "returns": {"method": "negative_quantity", "column": None, "prefixes": [], "values": [], "require_negative_quantity": True},
    "filters": {"exclude_product_ids": []},
    "cleaning": {"drop_exact_duplicates": True, "nonpositive_price": "drop"},   # or impute_product_median
    "analysis": {"data_start": None, "data_end": None, "churn_window_days": None, "churn_gap_quantile": 0.95,
                 "churn_window_min": 30, "churn_window_max": 180, "min_observation_days": 120,
                 "snapshot_step_days": 30, "new_customer_days": 60, "min_gaps_for_window": 200},
    "modeling": {"selection_metric": "pr_auc", "seed": C.SEED},
    "risk_tiers": {"high_min_lift": C.HIGH_RISK_MIN_LIFT, "low_max_churn": C.LOW_RISK_MAX_CHURN},
    "business": {"gross_margin": C.GROSS_MARGIN},
}


def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path=None) -> dict:
    """Load a YAML dataset config (supports ``extends: other.yaml``) merged over the defaults."""
    if path is None:
        path = C.ROOT / "configs" / "synthetic.yaml"
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = C.ROOT / path
    raw = yaml.safe_load(path.read_text()) or {}
    if raw.get("extends"):
        base = load_config(path.parent / raw["extends"])
        raw = _deep_merge({k: v for k, v in base.items() if k != "extends"}, {k: v for k, v in raw.items() if k != "extends"})
    cfg = _deep_merge(DEFAULT_CFG, raw)
    cfg["_path"] = str(path)
    return cfg


# ----------------------------------------------------------------------------- reading
def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else C.ROOT / p


def read_table(paths, read_options=None) -> pd.DataFrame:
    """Read one or several csv / csv.gz / tsv / xlsx / xls / parquet files and stack them."""
    read_options = read_options or {}
    frames = []
    for p in ([paths] if isinstance(paths, (str, Path)) else paths):
        fp = _resolve(p)
        if not fp.exists():
            raise DatasetValidationError(f"Input file not found: {fp}\n  -> check source.transactions in your config "
                                         "(paths are relative to the project root).")
        suf = "".join(fp.suffixes).lower()
        if suf.endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(fp, sheet_name=None, **read_options)   # every sheet is stacked
            frames.extend(sheets.values())
        elif suf.endswith(".parquet"):
            frames.append(pd.read_parquet(fp))
        else:
            opts = dict(read_options)
            if suf.endswith(".tsv"):
                opts.setdefault("sep", "\t")
            frames.append(pd.read_csv(fp, low_memory=False, **opts))
    if not frames:
        raise DatasetValidationError("No input files configured (source.transactions is empty).")
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- helpers
def _clean_id(s: pd.Series) -> pd.Series:
    """'12346.0' -> '12346'; blanks / 'nan' -> missing."""
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.0+$", "", regex=True)
    return out.mask(out.str.lower().isin(["", "nan", "none", "null", "<na>"]))


def _suggest(name, columns):
    m = difflib.get_close_matches(str(name), [str(c) for c in columns], n=3, cutoff=0.5)
    return f" Did you mean: {m}?" if m else ""


def _check_mapping(mapping, raw_cols, canonical_required, table):
    missing, absent = [], []
    for k in canonical_required:
        src = mapping.get(k)
        if not src:
            missing.append(k)
        elif src not in raw_cols:
            absent.append((k, src))
    msgs = []
    if missing:
        msgs.append(f"[{table}] required fields not mapped in the config: {missing}")
    for k, src in absent:
        msgs.append(f"[{table}] column '{src}' (mapped to '{k}') not found in the file.{_suggest(src, raw_cols)}")
    if msgs:
        msgs.append(f"Columns available in the file: {list(raw_cols)}")
        raise DatasetValidationError("\n".join(msgs))


# ----------------------------------------------------------------------------- returns
def detect_returns(raw: pd.DataFrame, qty: pd.Series, rcfg: dict, order_col=None):
    """
    Flag return / cancellation lines according to the configured strategy.
    Returns (is_return Series[bool], info dict). Never drops anything.

    methods:  negative_quantity | id_prefix | flag_column | none | auto
    """
    method = rcfg.get("method", "negative_quantity")
    info = {"method": method, "detected_by": None}
    neg = qty < 0

    if method == "auto":
        col = rcfg.get("column") or order_col
        if neg.sum() == 0:
            method = "negative_quantity"
        else:
            s = raw[col].astype("string").str.strip().str.upper()
            pref = s[neg].str.extract(r"^([A-Z]+)", expand=False).dropna()
            if len(pref) >= 0.8 * neg.sum():
                top = pref.value_counts()
                rcfg = {**rcfg, "prefixes": [top.index[0]], "column": col}
                method = "id_prefix"
            else:
                method = "negative_quantity"
        info["detected_by"] = f"auto -> {method}" + (f" {rcfg.get('prefixes')}" if method == "id_prefix" else "")

    if method == "negative_quantity":
        flag = neg
    elif method == "id_prefix":
        col = rcfg.get("column") or order_col
        if col not in raw.columns:
            raise DatasetValidationError(f"returns.column '{col}' not found in the file.{_suggest(col, raw.columns)}")
        prefixes = [str(p).upper() for p in rcfg.get("prefixes") or []]
        if not prefixes:
            raise DatasetValidationError("returns.method 'id_prefix' needs returns.prefixes, e.g. ['C'].")
        s = raw[col].astype("string").str.strip().str.upper()
        flag = s.str.match("^(" + "|".join(re.escape(p) for p in prefixes) + ")").fillna(False).astype(bool)
        if rcfg.get("require_negative_quantity", True):
            flag = flag & neg
        info["prefixes"] = prefixes
    elif method == "flag_column":
        col = rcfg.get("column")
        if not col or col not in raw.columns:
            raise DatasetValidationError(f"returns.column '{col}' not found in the file.{_suggest(col, raw.columns)}")
        vals = rcfg.get("values") or []
        s = raw[col]
        if vals:
            flag = s.astype("string").str.strip().str.lower().isin([str(v).lower() for v in vals]).fillna(False).astype(bool)
        else:
            flag = s.astype("string").str.strip().str.lower().isin(["1", "true", "yes", "y", "t"]).fillna(False).astype(bool)
    elif method == "none":
        flag = pd.Series(False, index=raw.index)
    else:
        raise DatasetValidationError(f"Unknown returns.method '{method}'. Use negative_quantity, id_prefix, flag_column, none or auto.")
    info["method_used"] = method
    return flag.astype(bool), info


# ----------------------------------------------------------------------------- standardize
def standardize_frames(tx_raw: pd.DataFrame, cust_raw, cfg: dict, returns_raw: pd.DataFrame = None):
    """
    Pure function: raw DataFrames + config -> (customers, transactions, availability, notes).
    Output uses canonical names only. Values are NOT yet cleaned (dates still unparsed).
    """
    cols = cfg["columns"]
    _check_mapping(cols, tx_raw.columns, [c for c in REQUIRED_TX if c != "order_id"], "transactions")
    if not cols.get("order_id") and not cols.get("transaction_id"):
        raise DatasetValidationError("[transactions] map your order/invoice column to 'order_id' (or, if every row is a separate "
                                     "order, to 'transaction_id').")
    for k in ("order_id", "transaction_id", "product_id", "product_name", "category", "discount", "payment_method"):
        if cols.get(k) and cols[k] not in tx_raw.columns:
            raise DatasetValidationError(f"[transactions] column '{cols[k]}' (mapped to '{k}') not found.{_suggest(cols[k], tx_raw.columns)}")

    notes, av = [], {}
    n_in = len(tx_raw)
    tx = pd.DataFrame(index=tx_raw.index)
    pick = lambda k: tx_raw[cols[k]] if cols.get(k) else pd.Series(np.nan, index=tx_raw.index)

    tx["customer_id"] = _clean_id(pick("customer_id"))
    order_src = cols.get("order_id") or cols["transaction_id"]
    if not cols.get("order_id"):
        notes.append("order_id not mapped: each row is treated as its own order (transaction_id used as order id).")
    tx["order_id"] = tx_raw[order_src].astype("string").str.strip()
    av["transaction_id"] = bool(cols.get("transaction_id"))
    tx["transaction_id"] = (tx_raw[cols["transaction_id"]].astype("string").str.strip() if av["transaction_id"]
                            else pd.Series([f"L{i:08d}" for i in range(n_in)], index=tx_raw.index, dtype="string"))
    tx["date"] = tx_raw[cols["date"]]
    qty_raw, price_raw = tx_raw[cols["quantity"]], tx_raw[cols["unit_price"]]
    tx["quantity"], tx["unit_price"] = pd.to_numeric(qty_raw, errors="coerce"), pd.to_numeric(price_raw, errors="coerce")
    bad_num = int((tx.quantity.isna() & qty_raw.notna()).sum() + (tx.unit_price.isna() & price_raw.notna()).sum())
    if bad_num:
        notes.append(f"{bad_num} non-numeric quantity/price values were set to missing (rows removed later as invalid).")
    for k in ("product_id", "product_name", "category", "payment_method"):
        tx[k] = pick(k).astype("string").str.strip() if cols.get(k) else pd.Series(pd.NA, index=tx_raw.index, dtype="string")
    av["product_id"], av["product_name"] = bool(cols.get("product_id")), bool(cols.get("product_name"))
    av["category"], av["payment_method"] = bool(cols.get("category")), bool(cols.get("payment_method"))
    if cols.get("discount"):
        d = pd.to_numeric(tx_raw[cols["discount"]], errors="coerce").fillna(0.0)
        tx["discount"] = d / 100.0 if cfg["discount"].get("scale") == "percent" else d
        av["discount"] = True
    else:
        tx["discount"] = 0.0
        av["discount"] = False
        notes.append("discount not mapped: set to 0 meaning 'discount information unavailable' (customers may still have received discounts).")

    is_ret, rinfo = detect_returns(tx_raw, tx["quantity"], cfg["returns"], order_col=order_src)
    tx["is_return"] = is_ret
    tx.loc[is_ret, "quantity"] = -tx.loc[is_ret, "quantity"].abs()
    notes.append(f"Return rule '{rinfo['method_used']}'" + (f" ({rinfo['detected_by']})" if rinfo.get("detected_by") else "") +
                 f": {int(is_ret.sum()):,} of {n_in:,} rows flagged as returns/cancellations.")

    # optional separate returns table (same canonical mapping, or returns_columns overrides)
    if returns_raw is not None and len(returns_raw):
        rcols = {**cols, **{k: v for k, v in (cfg.get("returns_columns") or {}).items() if v}}
        _check_mapping(rcols, returns_raw.columns, [c for c in REQUIRED_TX if c != "order_id"], "returns")
        r = pd.DataFrame(index=returns_raw.index)
        has_col = lambda k: bool(rcols.get(k)) and rcols[k] in returns_raw.columns     # optional fields may be absent from the returns table
        g = lambda k: returns_raw[rcols[k]] if has_col(k) else pd.Series(np.nan, index=returns_raw.index)
        r["customer_id"] = _clean_id(g("customer_id"))
        osrc = next((rcols[k] for k in ("order_id", "transaction_id") if has_col(k)), None)
        r["order_id"] = ("R" + returns_raw[osrc].astype("string").str.strip()) if osrc else pd.NA
        r["transaction_id"] = [f"RL{i:08d}" for i in range(len(returns_raw))]
        r["date"] = returns_raw[rcols["date"]]
        r["quantity"] = -pd.to_numeric(returns_raw[rcols["quantity"]], errors="coerce").abs()
        r["unit_price"] = pd.to_numeric(returns_raw[rcols["unit_price"]], errors="coerce")
        for k in ("product_id", "product_name", "category", "payment_method"):
            r[k] = g(k).astype("string").str.strip() if has_col(k) else pd.Series(pd.NA, index=returns_raw.index, dtype="string")
        r["discount"] = 0.0
        r["is_return"] = True
        tx = pd.concat([tx, r[tx.columns]], ignore_index=True)
        notes.append(f"{len(r):,} rows added from the separate returns table.")

    excl = [str(x).upper() for x in cfg["filters"].get("exclude_product_ids") or []]
    n_excl = 0
    if excl and av["product_id"]:
        m = tx["product_id"].astype("string").str.strip().str.upper().isin(excl)
        n_excl = int(m.sum())
        tx = tx[~m]
        notes.append(f"{n_excl:,} lines removed because their product id is a configured non-product code (postage, fees, adjustments...).")

    tx = tx[TX_COLUMNS + ["is_return"]].reset_index(drop=True)

    # ---- customers
    if cust_raw is not None:
        cc = cfg["customer_columns"]
        _check_mapping(cc, cust_raw.columns, ["customer_id"], "customers")
        cu = pd.DataFrame({"customer_id": _clean_id(cust_raw[cc["customer_id"]])})
        for k in OPTIONAL_CUST:
            src = cc.get(k)
            if src and src in cust_raw.columns:
                cu[k] = cust_raw[src].values
            else:
                cu[k] = np.nan
        av["customer_table"] = True
    else:
        cu = pd.DataFrame({"customer_id": tx["customer_id"].dropna().unique()})
        for k in OPTIONAL_CUST:
            cu[k] = np.nan
        av["customer_table"] = False
        notes.append("No customer table: customers derived from transactions; demographics unavailable.")
    for k, label in (("customer_registration_date", "registration_date"), ("customer_age", "age"),
                     ("customer_gender", "gender"), ("customer_location", "location")):
        av[label] = bool(av["customer_table"] and cu[k].notna().any())
    cu = cu[CUST_COLUMNS]
    av["registration_date_source"] = "provided" if av["registration_date"] else "derived_from_first_purchase"
    av["n_rows_in"], av["n_lines_excluded_non_product"], av["return_rule"] = n_in, n_excl, rinfo
    return cu, tx, av, notes


def standardize(cfg: dict, save=True):
    """Read the configured files, standardize, and (optionally) write data/.../standardized/."""
    src = cfg["source"]
    tx_raw = read_table(src["transactions"], src.get("read_options"))
    cust_raw = read_table(src["customers"], src.get("read_options")) if src.get("customers") else None
    ret_raw = read_table(src["returns"], src.get("read_options")) if src.get("returns") else None
    cu, tx, av, notes = standardize_frames(tx_raw, cust_raw, cfg, ret_raw)
    if save:
        C.DATA_STD.mkdir(parents=True, exist_ok=True)
        cu.to_csv(C.DATA_STD / "customers.csv", index=False)
        tx.to_csv(C.DATA_STD / "transactions.csv.gz", index=False)
    return cu, tx, av, notes


def mapping_table(cfg: dict) -> pd.DataFrame:
    """Human-readable 'original column -> canonical column' table (used in the validation report)."""
    rows = [(src, k, "transactions") for k, src in cfg["columns"].items()]
    rows += [(src, k, "customers") for k, src in cfg["customer_columns"].items()]
    return pd.DataFrame([(a if a else "(unavailable)", b, t) for a, b, t in rows], columns=["original_column", "canonical_column", "table"])
