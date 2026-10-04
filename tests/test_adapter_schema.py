"""Schema tests: different column names can be mapped, required fields are detected, optional fields are handled."""
import numpy as np
import pandas as pd
import pytest

from conftest import TINY_CFG, make_raw
from src.adapter import DEFAULT_CFG, DatasetValidationError, _deep_merge, load_config, read_table, standardize_frames
from src.schema import REQUIRED_TX, TX_COLUMNS


def test_arbitrary_column_names_map_to_canonical_schema():
    cu, tx, av, _ = standardize_frames(make_raw(n_cust=30, days=200), None, TINY_CFG)
    assert set(TX_COLUMNS) <= set(tx.columns) and "is_return" in tx
    assert not ({"cust", "inv", "dt", "qty", "px"} & set(tx.columns))            # raw names never leak downstream
    assert tx.customer_id.str.fullmatch(r"\d+").all()                             # 1000.0 -> "1000"


@pytest.mark.parametrize("missing", ["customer_id", "date", "quantity", "unit_price"])
def test_required_field_not_mapped_is_detected(missing):
    cfg = _deep_merge(TINY_CFG, {"columns": {missing: None}})
    with pytest.raises(DatasetValidationError, match="required fields not mapped"):
        standardize_frames(make_raw(n_cust=20, days=100), None, cfg)


def test_wrong_column_name_gives_suggestion():
    cfg = _deep_merge(TINY_CFG, {"columns": {"unit_price": "price_"}})
    raw = make_raw(n_cust=20, days=100).rename(columns={"px": "price"})
    with pytest.raises(DatasetValidationError, match="Did you mean"):
        standardize_frames(raw, None, cfg)


def test_optional_fields_absent_are_flagged_not_invented():
    cfg = _deep_merge(TINY_CFG, {"columns": {"category": None, "discount": None, "product_id": None}})
    cu, tx, av, notes = standardize_frames(make_raw(n_cust=30, days=200), None, cfg)
    assert not av["category"] and not av["discount"] and not av["product_id"]
    assert not av["customer_table"] and not av["age"] and not av["gender"] and not av["location"]
    assert av["registration_date_source"] == "derived_from_first_purchase"
    assert tx.category.isna().all() and (tx.discount == 0).all()
    assert cu[["customer_registration_date", "customer_age", "customer_gender", "customer_location"]].isna().all().all()
    assert any("discount information unavailable" in n for n in notes)


def test_customer_table_derived_when_missing_and_used_when_given():
    raw = make_raw(n_cust=30, days=200)
    cu, *_ = standardize_frames(raw, None, TINY_CFG)
    assert set(cu.customer_id) == set(raw.cust.astype(int).astype(str))
    cust_raw = pd.DataFrame({"id": [1000.0, 1001.0], "age": [30, 41], "city": ["A", "B"]})
    cfg = _deep_merge(TINY_CFG, {"customer_columns": {"customer_id": "id", "customer_age": "age", "customer_location": "city"}})
    cu2, _, av, _ = standardize_frames(raw, cust_raw, cfg)
    assert av["customer_table"] and av["age"] and av["location"] and not av["gender"]
    assert list(cu2.customer_id) == ["1000", "1001"]


def test_order_id_falls_back_to_transaction_id():
    cfg = _deep_merge(TINY_CFG, {"columns": {"order_id": None, "transaction_id": "inv"}})
    _, tx, _, notes = standardize_frames(make_raw(n_cust=10, days=100), None, cfg)
    assert (tx.order_id == tx.transaction_id).all() and any("treated as its own order" in n for n in notes)


def test_separate_returns_table():
    raw = make_raw(n_cust=20, days=150)
    raw = raw[raw.kind == "SALE"]
    ret = pd.DataFrame({"cust": [1000.0], "inv": ["X1"], "dt": ["01.02.2015"], "qty": [2], "px": [10.0]})
    _, tx, _, notes = standardize_frames(raw, None, TINY_CFG, ret)
    r = tx[tx.is_return]
    assert len(r) == 1 and r.quantity.iloc[0] == -2 and any("separate returns table" in n for n in notes)


def test_extends_and_file_formats(tmp_path):
    cfg = load_config("configs/retail_style_fixture.yaml")
    assert cfg["columns"]["customer_id"] == "Customer ID" and cfg["name"] == "retail_style_fixture"   # inherited + overridden
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(tmp_path / "x.csv", index=False)
    with pd.ExcelWriter(tmp_path / "y.xlsx") as w:                   # two sheets are stacked (as in Online Retail II)
        df.to_excel(w, sheet_name="s1", index=False); df.to_excel(w, sheet_name="s2", index=False)
    assert len(read_table(str(tmp_path / "x.csv"))) == 2 and len(read_table(str(tmp_path / "y.xlsx"))) == 4
    with pytest.raises(DatasetValidationError, match="not found"):
        read_table(str(tmp_path / "nope.csv"))


def test_template_config_loads_and_every_shipped_config_is_valid():
    from src.adapter import load_config
    for name in ("template", "synthetic", "online_retail_ii", "retail_style_fixture"):
        cfg = load_config(f"configs/{name}.yaml")
        assert set(REQUIRED_TX) <= set(cfg["columns"])
        assert cfg["returns"]["method"] in ("negative_quantity", "id_prefix", "flag_column", "none", "auto")
