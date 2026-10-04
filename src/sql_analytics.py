"""Builds a small SQLite analytical database and runs the queries in sql/queries.sql."""
import re
import sqlite3

import pandas as pd

from . import config as C
from .config import ROOT


def build_db(path=None):
    path = path or C.DATA_PROC / "retention.db"
    tx = pd.read_csv(C.DATA_PROC / "transactions_clean.csv.gz", parse_dates=["date"])
    tx["date"] = tx.date.dt.strftime("%Y-%m-%d")
    tx["is_return"] = tx.is_return.astype(int); tx["suspicious_value"] = tx.suspicious_value.astype(int)
    cs = pd.read_csv(C.DATA_PROC / "customers_scored.csv.gz")
    cols = ["customer_id", "rfm_segment", "cluster_name", "status", "monetary", "frequency", "recency_days", "churn_prob",
            "risk_level", "value_tier", "annual_margin_run_rate", "margin_at_risk", "priority"]
    con = sqlite3.connect(path)
    tx.to_sql("transactions", con, if_exists="replace", index=False)
    cs[cols].to_sql("customers_scored", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_tx_customer ON transactions(customer_id)")
    return con


def run_queries(con):
    text = (ROOT / "sql" / "queries.sql").read_text()
    out = {}
    for name, body in re.findall(r"-- name: (\w+)\n(.*?)(?=\n-- name:|\Z)", text, flags=re.S):
        out[name] = pd.read_sql_query(body, con)
    return out


if __name__ == "__main__":
    con = build_db()
    for name, df in run_queries(con).items():
        print(f"\n=== {name} ===\n{df.to_string(index=False)}")
