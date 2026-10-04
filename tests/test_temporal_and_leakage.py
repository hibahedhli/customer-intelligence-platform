"""Temporal tests + leakage prevention: the evaluation is only trustworthy if none of these can fail."""
import numpy as np
import pandas as pd
import pytest

from conftest import TINY_CFG, make_raw, prepare
from src.feature_engineering import (FEATURES, InsufficientHistoryError, build_orders, build_snapshot, choose_churn_window,
                                     compute_features, cutoff_schedule, interpurchase_gaps, required_history_days)
from src.validation import validate_dataset
from src.adapter import standardize_frames
from src.data_processing import clean_data


# ----------------------------------------------------------------------------- dynamic dates / window / split
def test_dates_are_inferred_not_hard_coded(tiny):
    assert tiny["start"].year == 2015 and tiny["end"].year == 2017            # the tiny data lives in 2015-2017; the demos use 2024-25 and 2009-11
    for c in tiny["train_c"] + [tiny["val_c"], tiny["test_c"]]:
        assert tiny["start"] <= c <= tiny["end"]


def test_churn_window_is_recomputed_for_each_dataset():
    slow = prepare(make_raw(seed=1, rate=1 / 60, n_cust=500, days=900))
    fast = prepare(make_raw(seed=1, rate=1 / 12, n_cust=300, days=500))
    assert fast["H"] < slow["H"]
    for d in (slow, fast):                                                  # and it equals the documented rule
        raw_q = interpurchase_gaps(d["orders"]).quantile(.95)
        assert d["H"] == int(np.clip(np.ceil(raw_q / 15) * 15, 30, 180))


@pytest.mark.parametrize("H,start,end", [(30, "2018-01-01", "2019-06-30"), (90, "2009-12-01", "2011-12-09"), (150, "2000-03-05", "2003-01-01")])
def test_split_is_chronological_for_any_dates(H, start, end):
    train, val, test = cutoff_schedule(H, start, end)
    assert max(train) + pd.Timedelta(days=H) <= val             # train labels resolved before validation starts
    assert val + pd.Timedelta(days=H) <= test                   # validation labels resolved before test starts
    assert test + pd.Timedelta(days=H) == pd.Timestamp(end)     # test labels end exactly at the last day of data
    assert min(train) >= pd.Timestamp(start) + pd.Timedelta(days=120) and train == sorted(train)


def test_too_short_history_fails_with_clear_message():
    with pytest.raises(InsufficientHistoryError, match=r"only \d+ months of history.*At least approximately \d+ months are recommended for a churn horizon of 90 days"):
        cutoff_schedule(90, "2020-01-01", "2020-08-31")
    assert required_history_days(90) == 120 + 270 + 60


def test_no_labels_beyond_the_available_data(tiny):
    with pytest.raises(AssertionError, match="runs past the end of the data"):
        build_snapshot(tiny["orders"], tiny["lines"], tiny["returns"], tiny["customers"], tiny["end"] - pd.Timedelta(days=5), tiny["H"], data_end=tiny["end"])


def test_validation_report_states_window_and_history(tiny):
    raw = make_raw()
    cu, tx_std, av, notes = standardize_frames(raw, None, TINY_CFG)
    customers, tx, log, summ = clean_data(cu, tx_std, TINY_CFG, av)
    rep = validate_dataset("tiny", TINY_CFG, cu, tx_std, customers, tx, build_orders(tx), av, summ, notes)
    f = rep.facts
    assert not rep.errors and f["sufficient_history"] is True and f["churn_window"] == tiny["H"]
    assert f["data_start"] == str(tiny["start"].date()) and "P95" in f["churn_window_source"]
    text = rep.render()
    assert "Recommended churn window" in text and "Sufficient history: YES" in text and "[WARN] Product id" not in text


def test_validation_blocks_short_or_tiny_datasets():
    raw = make_raw(n_cust=300, days=250)
    cu, tx_std, av, notes = standardize_frames(raw, None, TINY_CFG)
    customers, tx, log, summ = clean_data(cu, tx_std, TINY_CFG, av)
    rep = validate_dataset("short", TINY_CFG, cu, tx_std, customers, tx, build_orders(tx), av, summ, notes)
    assert any("history" in e["title"].lower() for e in rep.errors)
    with pytest.raises(Exception, match="cannot be analysed"):
        rep.raise_if_errors()


# ----------------------------------------------------------------------------- leakage
@pytest.mark.parametrize("which", ["val_c", "test_c"])
def test_future_transactions_cannot_influence_past_features(tiny, which):
    """Delete / rewrite EVERYTHING after the cut-off: features at the cut-off must be identical."""
    T = tiny[which]
    base = compute_features(tiny["orders"], tiny["lines"], tiny["returns"], tiny["customers"], T)
    rng = np.random.default_rng(0)
    o, l, r = tiny["orders"].copy(), tiny["lines"].copy(), tiny["returns"].copy()
    fut = o.order_date > T
    o.loc[fut, "order_value"] = rng.uniform(1, 1e6, fut.sum())               # tamper with the future
    o.loc[fut, "customer_id"] = rng.choice(o.customer_id.unique(), fut.sum())
    l.loc[l.date > T, ["quantity", "discount"]] = [999, 0.9]
    r.loc[r.date > T, "transaction_value"] = -1e6
    after = compute_features(o, l, r, tiny["customers"], T)
    pd.testing.assert_frame_equal(base[FEATURES].sort_index(), after[FEATURES].sort_index())
    cut = compute_features(o[o.order_date <= T], l[l.date <= T], r[r.date <= T], tiny["customers"], T)   # and deleting it entirely
    pd.testing.assert_frame_equal(base[FEATURES].sort_index(), cut[FEATURES].sort_index())


def test_labels_depend_only_on_the_prediction_window(tiny):
    H, T, o = tiny["H"], tiny["val_c"], tiny["orders"]
    snap = build_snapshot(o, tiny["lines"], tiny["returns"], tiny["customers"], T, H, data_end=tiny["end"])
    future = o[(o.order_date > T) & (o.order_date <= T + pd.Timedelta(days=H))].customer_id.unique()
    assert (snap.churned == (~snap.index.isin(future)).astype(int)).all()
    o2 = o.copy()                                                              # move every order after T+H somewhere else
    o2.loc[o2.order_date > T + pd.Timedelta(days=H), "order_date"] += pd.Timedelta(days=30)
    snap2 = build_snapshot(o2, tiny["lines"], tiny["returns"], tiny["customers"], T, H, data_end=tiny["end"] + pd.Timedelta(days=30))
    assert (snap.churned == snap2.churned.reindex(snap.index)).all()


def test_labels_never_enter_the_feature_set(tiny):
    assert "churned" not in FEATURES and not any("churn" in f for f in FEATURES)
    assert set(tiny["train"].columns) - set(FEATURES) >= {"churned", "cutoff"}


def test_training_labels_are_resolved_before_validation_starts(tiny):
    H = tiny["H"]
    assert all(c + pd.Timedelta(days=H) <= tiny["val_c"] for c in tiny["train_c"])
    assert tiny["train"].cutoff.max() + pd.Timedelta(days=H) <= tiny["val_c"]


def test_default_run_features_ignore_the_future():
    """Same property on the real outputs of the default (synthetic demo) run, if it has been generated."""
    from src import config as C
    p = C.paths_for(None)["processed"]
    if not (p / "transactions_clean.csv.gz").exists():
        pytest.skip("default run not generated")
    c = pd.read_csv(p / "customers_clean.csv", parse_dates=["customer_registration_date"])
    t = pd.read_csv(p / "transactions_clean.csv.gz", parse_dates=["date"])
    T = pd.Timestamp("2025-03-01")
    full = compute_features(build_orders(t), t[~t.is_return], t[t.is_return], c, T)
    tc = t[t.date <= T]
    cut = compute_features(build_orders(tc), tc[~tc.is_return], tc[tc.is_return], c, T)
    pd.testing.assert_frame_equal(full[FEATURES].sort_index(), cut[FEATURES].sort_index())
