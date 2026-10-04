"""Data-quality tests: missing values, duplicates, invalid dates, quantities, prices, and return conventions."""
import numpy as np
import pandas as pd
import pytest

from conftest import TINY_CFG, make_raw
from src.adapter import DEFAULT_CFG, _deep_merge, detect_returns, standardize_frames
from src.data_processing import clean_data


def _base(n=8):
    return pd.DataFrame({"cust": [1.0] * n, "inv": [f"A{i}" for i in range(n)], "dt": ["01.03.2020"] * n, "sku": ["S1"] * n, "dept": ["x"] * n,
                         "qty": [2] * n, "px": [10.0] * n, "disc_pct": [0.0] * n, "kind": ["SALE"] * n})


def _run(raw, cfg=TINY_CFG):
    cu, tx, av, _ = standardize_frames(raw, None, cfg)
    return clean_data(cu, tx, cfg, av)


def test_each_problem_is_handled_and_counted():
    raw = _base(12)
    raw["px"] = raw["px"].astype(object)
    raw.loc[0, "cust"] = np.nan                          # no customer id
    raw.loc[1, "dt"] = "not a date"                      # invalid date
    raw.loc[2, "qty"] = 0                                # zero quantity
    raw.loc[3, "qty"] = -4                               # negative but NOT a return
    raw.loc[4, "px"] = -5.0                              # invalid price
    raw.loc[5, "disc_pct"] = 25.0                        # percent scale -> 0.25
    raw.loc[6, "px"] = "abc"                             # non-numeric price
    raw = pd.concat([raw, raw.iloc[[7]]], ignore_index=True)  # exact duplicate of row 7
    _, tx, log, summ = _run(raw)
    r = summ["removed_by_reason"]
    assert r["missing customer id"] == 1 and r["unparseable date"] == 1 and r["exact duplicate row"] == 1
    assert r["invalid quantity (zero / missing / negative but not a return)"] == 2
    assert r["invalid price (<= 0 or missing)"] == 2                 # the negative and the non-numeric price
    assert summ["rows_in"] == summ["rows_out"] + summ["rows_removed"] == len(raw)
    assert tx.discount.max() == pytest.approx(0.25)


def test_returns_are_kept_for_every_configured_convention():
    n = 6
    raw = _base(n)
    # 1) flag column, refunds with POSITIVE quantity
    raw.loc[[0, 1], "kind"] = "REFUND"
    _, tx, _, summ = _run(raw)
    assert tx.is_return.sum() == 2 and (tx[tx.is_return].quantity < 0).all() and summ["return_lines_kept"] == 2
    # 2) id prefix + negative quantity (Online-Retail style)
    cfg = _deep_merge(TINY_CFG, {"returns": {"method": "id_prefix", "column": "inv", "prefixes": ["C"]}})
    raw2 = _base(n); raw2.loc[[0, 1], "inv"] = ["C1", "c2"]; raw2.loc[[0, 1], "qty"] = -2
    _, tx, _, summ = _run(raw2, cfg)
    assert tx.is_return.sum() == 2 and summ["removed_by_reason"].get("invalid quantity (zero / missing / negative but not a return)", 0) == 0
    # 3) plain negative quantity
    cfg = _deep_merge(TINY_CFG, {"returns": {"method": "negative_quantity"}})
    raw3 = _base(n); raw3.loc[[0, 1, 2], "qty"] = -1
    _, tx, _, summ = _run(raw3, cfg)
    assert tx.is_return.sum() == 3
    # 4) auto-detection picks the prefix convention when negatives all carry a letter prefix
    cfg = _deep_merge(TINY_CFG, {"returns": {"method": "auto", "column": "inv"}})
    raw4 = _base(n); raw4.loc[[0, 1], "inv"] = ["C10", "C11"]; raw4.loc[[0, 1], "qty"] = -3
    cu, tx, av, _ = standardize_frames(raw4, None, cfg)
    assert tx.is_return.sum() == 2 and "id_prefix" in av["return_rule"]["detected_by"]


def test_returns_are_not_silently_dropped_when_rule_is_wrong_they_are_reported():
    raw = _base(10); raw.loc[:3, "qty"] = -1
    cfg = _deep_merge(TINY_CFG, {"returns": {"method": "none"}})
    _, tx, log, summ = _run(raw, cfg)
    assert summ["return_lines_kept"] == 0
    row = log[log.issue.str.startswith("Invalid quantity")].iloc[0]
    assert row.rows_affected == 4 and "4 rows are negative but not flagged as returns" in row.rationale


def test_returns_do_not_count_as_purchases():
    from src.feature_engineering import build_orders
    raw = _base(5); raw.loc[0, "kind"] = "REFUND"
    _, tx, _, _ = _run(raw)
    assert len(build_orders(tx)) == 4                    # the refund line is not an order


def test_price_policy_drop_vs_impute():
    raw = _base(6); raw.loc[0, "px"] = 0.0
    _, tx, _, s1 = _run(raw)                             # default: drop
    assert s1["removed_by_reason"]["invalid price (<= 0 or missing)"] == 1
    cfg = _deep_merge(TINY_CFG, {"cleaning": {"nonpositive_price": "impute_product_median"}})
    _, tx, _, s2 = _run(raw, cfg)
    assert s2["rows_removed"] == 0 and (tx.unit_price == 10.0).all()


def test_registration_date_fallback_and_unknown_demographics():
    customers, tx, log, _ = _run(make_raw(n_cust=20, days=150))
    first = tx[~tx.is_return].groupby("customer_id").date.min()
    assert (customers.set_index("customer_id").customer_registration_date.reindex(first.index) == first).all()
    assert (customers.customer_gender == "Unknown").all() and (customers.customer_location == "Unknown").all()
    assert customers.customer_age.isna().all()           # never fabricated
    assert "Registration date unavailable" in set(log.issue)
