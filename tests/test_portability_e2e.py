"""
Second-dataset tests: the COMPLETE pipeline and the dashboard run on a dataset whose schema differs from the
synthetic demo, and nothing synthetic-specific leaks into the code.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import ROOT
from src.feature_engineering import choose_churn_window, interpurchase_gaps, build_orders


def test_second_dataset_pipeline_outputs(retail_run):
    meta = json.load(open(retail_run / "run_metadata.json")); M = json.load(open(retail_run / "metrics.json"))
    assert meta["name"] == "retail_style_fixture" and meta["return_rule"]["method_used"] == "id_prefix" and meta["return_rule"]["prefixes"] == ["C"]
    assert (M["data_start"], M["data_end"]) == ("2009-12-01", "2011-12-08")                 # inferred from the data, not the demo's 2024-2025
    av = meta["availability"]
    assert not av["category"] and not av["discount"] and not av["customer_table"] and av["registration_date_source"] == "derived_from_first_purchase"
    for f in ("n_unique_categories", "avg_discount", "discount_order_share", "days_since_registration"):
        assert f in meta["features_excluded"] and f not in meta["features_used"]
    assert M["best_model"] == meta["model_selected"] and M["churn_window_source"].startswith("P95")
    tx = pd.read_csv(retail_run / "transactions_clean.csv.gz")
    assert tx.is_return.sum() > 0 and not tx.order_id.astype(str).str.startswith("C").any() or tx[tx.is_return].order_id.astype(str).str.startswith("C").all()
    assert "POST" not in set(tx.product_id.astype(str))                                       # non-product lines filtered by config
    for f in ("dataset_validation.md", "model_comparison.csv", "customers_scored.csv.gz", "shap_global.csv", "risk_tiers_test.csv"):
        assert (retail_run / f).exists()


def test_churn_window_for_second_dataset_is_derived_from_its_own_gaps(retail_run):
    tx = pd.read_csv(retail_run / "transactions_clean.csv.gz", parse_dates=["date"])
    H, _ = choose_churn_window(interpurchase_gaps(build_orders(tx)))
    meta = json.load(open(retail_run / "run_metadata.json"))
    assert meta["churn_window"] == H
    from src import config as C
    default = C.paths_for(None)["processed"] / "run_metadata.json"
    if default.exists():                                                                      # different data -> different window
        assert json.load(open(default))["churn_window"] != H


def test_risk_tiers_use_this_datasets_base_rate(retail_run):
    meta = json.load(open(retail_run / "run_metadata.json"))
    base = json.load(open(retail_run / "metrics.json"))["churn_rate_val"]
    assert f"{base:.1%}" in meta["risk_rule"] and "2x" in meta["risk_rule"]


def test_dashboard_runs_on_second_dataset(retail_run):
    from streamlit.testing.v1 import AppTest
    os.environ["CIP_RUN"] = "pytest_retail"
    try:
        at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180).run()
        assert not at.exception, [e.value for e in at.exception]
        assert at.sidebar.selectbox[0].value == "pytest_retail" and len(at.tabs) == 6
        text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        assert "retail_style_fixture" in text and "Not available in this dataset" in text   # missing fields are announced, not crashed on
        dims = [s for s in at.selectbox if s.label == "Break down by"][0]
        assert "customer_location" not in dims.options and "favorite_category" not in dims.options
        cs = [s for s in at.selectbox if s.label == "Customer"][0]
        for o in (cs.options[0], cs.options[-1]):
            cs.set_value(o).run(); assert not at.exception
    finally:
        os.environ.pop("CIP_RUN", None)


def test_broken_config_fails_with_a_readable_message(tmp_path):
    f = tmp_path / "t.csv"
    pd.DataFrame({"Invoice": ["1"], "InvoiceDate": ["2020-01-01"], "Quantity": [1], "Customer ID": [5]}).to_csv(f, index=False)   # no price column
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "src.pipeline", "--config", "configs/online_retail_ii.yaml", "--input", str(f),
                        "--run", "pytest_broken", "--validate-only"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "Traceback" not in r.stderr
    assert "'Price'" in r.stderr and "not found in the file" in r.stderr and "Columns available in the file" in r.stderr
    import shutil; shutil.rmtree(ROOT / "data" / "runs" / "pytest_broken", ignore_errors=True)


def test_no_synthetic_specific_values_in_the_pipeline_code():
    forbidden = [r"transaction_date", r"product_category", r"\bDATA_END\b", r"\bDATA_START\b", r"\b87\b", r"churn_window\s*=\s*90", r"2024-01-01", r"2025-12-31", r"\bR\d*\"\)"]
    files = [p for p in (ROOT / "src").glob("*.py") if p.name not in ("generate_data.py", "config.py")] + [ROOT / "dashboard" / "app.py"]
    hits = [(p.name, pat) for p in files for pat in forbidden if re.search(pat, p.read_text())]
    assert not hits, hits
