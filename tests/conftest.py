"""
Shared test helpers. The unit tests use a small THIRD schema (different column names, date format,
discount scale and return convention from both the synthetic demo and the retail-style fixture), so they
cannot pass by accident just because the project's own synthetic data is used.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C                                              # noqa: E402
from src.adapter import DEFAULT_CFG, _deep_merge, standardize_frames     # noqa: E402
from src.data_processing import clean_data                               # noqa: E402
from src.feature_engineering import (build_orders, build_snapshot, choose_churn_window, cutoff_schedule,  # noqa: E402
                                     interpurchase_gaps, make_snapshots)

TINY_CFG = _deep_merge(DEFAULT_CFG, {
    "name": "tiny",
    "columns": {"customer_id": "cust", "order_id": "inv", "date": "dt", "product_id": "sku", "category": "dept",
                "quantity": "qty", "unit_price": "px", "discount": "disc_pct"},
    "date_format": "%d.%m.%Y",
    "discount": {"scale": "percent"},
    "returns": {"method": "flag_column", "column": "kind", "values": ["REFUND"]},
})


def make_raw(seed=0, n_cust=400, start="2015-01-01", days=760, rate=1 / 30, hazard=0.002):
    """Raw transactions in the tiny schema: refunds have POSITIVE quantity and kind == 'REFUND'."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    skus = [f"S{i:03d}" for i in range(25)]
    dept = {s: ["toys", "food", "tools", "garden", "books"][i % 5] for i, s in enumerate(skus)}
    price = {s: round(float(rng.uniform(2, 60)), 2) for s in skus}
    rows, inv = [], 0
    for i in range(n_cust):
        join = int(rng.integers(0, days // 2))
        churn = join + rng.exponential(1 / hazard)
        r = rate * rng.lognormal(0, .3)
        t, order_days = float(join), [join]
        while True:
            t += rng.exponential(1 / r)
            if t > days or t > churn:
                break
            order_days.append(int(t))
        for d in order_days:
            inv += 1
            for _ in range(1 + rng.integers(0, 3)):
                s = skus[rng.integers(len(skus))]
                q = int(1 + rng.integers(0, 4))
                disc = float(rng.choice([0, 0, 0, 5, 10, 20]))
                rows.append((1000.0 + i, f"A{inv}", (t0 + pd.Timedelta(days=d)).strftime("%d.%m.%Y"), s, dept[s], q, price[s], disc, "SALE"))
                if rng.random() < .03 and d + 5 <= days:
                    rows.append((1000.0 + i, f"R{inv}", (t0 + pd.Timedelta(days=d + 5)).strftime("%d.%m.%Y"), s, dept[s], q, price[s], disc, "REFUND"))
    return pd.DataFrame(rows, columns=["cust", "inv", "dt", "sku", "dept", "qty", "px", "disc_pct", "kind"])


def prepare(raw, cfg=None, step=30):
    """adapter -> cleaning -> orders -> churn window -> chronological cut-offs -> snapshots."""
    cfg = cfg or TINY_CFG
    cu, tx, av, notes = standardize_frames(raw, None, cfg)
    customers, tx, log, summ = clean_data(cu, tx, cfg, av)
    orders = build_orders(tx)
    lines, returns = tx[~tx.is_return], tx[tx.is_return]
    H, _ = choose_churn_window(interpurchase_gaps(orders))
    start, end = tx.date.min(), tx.date.max()
    train_c, val_c, test_c = cutoff_schedule(H, start, end, step=step)
    train = make_snapshots(orders, lines, returns, customers, train_c, H, data_end=end)
    val = build_snapshot(orders, lines, returns, customers, val_c, H, data_end=end).reset_index()
    test = build_snapshot(orders, lines, returns, customers, test_c, H, data_end=end).reset_index()
    return dict(customers=customers, tx=tx, orders=orders, lines=lines, returns=returns, H=H, start=start, end=end,
                train_c=train_c, val_c=val_c, test_c=test_c, train=train, val=val, test=test, av=av, summary=summ, log=log)


@pytest.fixture(scope="session")
def tiny():
    return prepare(make_raw())


@pytest.fixture(scope="session")
def retail_run():
    """Run the COMPLETE pipeline on the second dataset (structurally different schema) once per test session."""
    run = "pytest_retail"
    shutil.rmtree(ROOT / "data" / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-W", "ignore", "-m", "src.pipeline", "--config", "configs/retail_style_fixture.yaml", "--run", run]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    yield C.paths_for(run)["processed"]
    shutil.rmtree(ROOT / "data" / "runs" / run, ignore_errors=True)
