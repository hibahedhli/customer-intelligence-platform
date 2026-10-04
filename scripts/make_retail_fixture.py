"""
Builds examples/retail_style_fixture/transactions.csv.gz

THIS IS SIMULATED DATA. It reproduces only the STRUCTURE of the public UCI "Online Retail II" file
(a UK gift wholesaler, 2009-12-01 -> 2011-12-09) so that the adapter can be tested on a schema that is
genuinely different from the project's own synthetic data:

  * 8 columns with other names (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country)
  * ONE table: no customer table, no registration date, no demographics, no category, no discount
  * cancellations = invoice numbers starting with "C" with negative quantities
  * ~15-20% of rows without a Customer ID (guest checkouts), float-formatted ids ("13085.0")
  * date-times ("2009-12-01 07:45:00"), a different date range, a different number of customers
  * non-product lines (POST = postage, M = manual, ...), zero-price rows, exact duplicate rows

None of the values are real. For real results download the real dataset (see README) and use
configs/online_retail_ii.yaml.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "examples" / "retail_style_fixture" / "transactions.csv.gz"
START, END = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09")
N_CUST = 2000
TYPES = {  # share, order rate/day, churn hazard/day, mean lines per invoice
    "wholesale_big": (.15, 1 / 22, .0009, 11), "wholesale_mid": (.35, 1 / 45, .0014, 6),
    "small_buyer": (.35, 1 / 90, .0025, 3), "one_off": (.15, 1 / 400, .0120, 2)}
SEASON = {1: .7, 2: .8, 3: .9, 4: .9, 5: 1.0, 6: 1.0, 7: .95, 8: 1.0, 9: 1.3, 10: 1.5, 11: 1.7, 12: .9}
W1 = "WHITE PINK RED BLUE GREEN VINTAGE RETRO HANGING HEART STAR LARGE SMALL CERAMIC GLASS WOODEN METAL PAPER LACE FLORAL SPOTTY STRIPED".split()
W2 = "MUG CANDLE HOLDER BOX BAG CARD BOWL JAR TRAY CUSHION LANTERN FRAME BOTTLE NOTEBOOK BUNTING SIGN CLOCK HOOK COASTER".split()


def main(seed=2011):
    rng = np.random.default_rng(seed)
    n_days = (END - START).days
    cids = rng.choice(np.arange(12346, 18288), N_CUST, replace=False)
    types = rng.choice(list(TYPES), N_CUST, p=[v[0] for v in TYPES.values()])
    existing = rng.random(N_CUST) < .55
    join = np.where(existing, rng.integers(0, 50, N_CUST), rng.integers(0, 640, N_CUST))
    country = np.where(rng.random(N_CUST) < .91, "United Kingdom", rng.choice(["France", "Germany", "EIRE", "Spain", "Netherlands"], N_CUST))
    # product catalogue
    n_prod = 1500
    codes = [str(rng.integers(10000, 99999)) + (rng.choice(list("ABCW")) if rng.random() < .2 else "") for _ in range(n_prod)]
    codes = list(dict.fromkeys(codes))
    desc = [f"{rng.choice(W1)} {rng.choice(W1)} {rng.choice(W2)}" for _ in codes]
    price = np.round(np.exp(rng.normal(np.log(2.5), .8, len(codes))), 2).clip(0.2, 60)
    pop = rng.pareto(1.2, len(codes)) + 1; pop /= pop.sum()
    packs = np.array([1, 2, 3, 4, 6, 8, 10, 12, 24, 48]); pw = np.array([.12, .08, .06, .06, .16, .08, .1, .16, .1, .08])
    month = [(START + pd.Timedelta(days=d)).month for d in range(n_days + 1)]

    rows, inv = [], 489434
    def add_invoice(cust, day, n_lines, country_, cancel_of=None):
        nonlocal inv
        inv += 1
        ts = START + pd.Timedelta(days=int(day), hours=int(rng.integers(8, 18)), minutes=int(rng.integers(0, 60)))
        no = f"C{inv}" if cancel_of is not None else str(inv)
        if cancel_of is not None:
            for (j, q) in cancel_of:
                rows.append((no, codes[j], desc[j], -q, ts, price[j], cust, country_))
            return
        lines = []
        for j in rng.choice(len(codes), n_lines, p=pop):
            q = int(rng.choice(packs, p=pw)); lines.append((j, q))
            rows.append((no, codes[j], desc[j], q, ts, price[j], cust, country_))
        if rng.random() < .12:
            rows.append((no, "POST", "POSTAGE", 1, ts, 18.0, cust, country_))
        if rng.random() < .005:
            rows.append((no, "M", "Manual", 1, ts, float(np.round(rng.uniform(1, 40), 2)), cust, country_))
        if cust is not None and rng.random() < .022:                       # a later cancellation of part of the invoice
            cd = day + int(rng.integers(2, 40))
            if cd <= n_days:
                k = rng.choice(len(lines), max(1, len(lines) // 2), replace=False)
                add_invoice(cust, cd, 0, country_, cancel_of=[lines[i] for i in k])
        return

    for i in range(N_CUST):
        rate, haz, mean_l = TYPES[types[i]][1:]
        rate *= rng.lognormal(0, .4)
        churn = join[i] + rng.exponential(1 / haz)
        t = float(join[i]) + rng.integers(0, 5)
        days = [int(t)] if t <= n_days else []
        while True:
            t += rng.exponential(1 / (rate * 1.7))
            if t > n_days or t > churn:
                break
            fade = 1.0 if churn - t >= 45 else .3 + .7 * (churn - t) / 45
            if rng.random() < fade * SEASON[month[int(t)]] / 1.7:
                days.append(int(t))
        for d in days:
            add_invoice(float(cids[i]), d, 1 + rng.poisson(mean_l - 1), country[i])
    n_valid = len({r[0] for r in rows})
    for _ in range(int(.2 * n_valid)):                                       # guest checkouts: no Customer ID
        add_invoice(None, rng.integers(0, n_days + 1), 1 + rng.poisson(4), "United Kingdom")
    for _ in range(40):                                                      # zero-price / adjustment rows without customer
        j = rng.integers(len(codes)); inv += 1
        rows.append((str(inv), codes[j], "damaged" if rng.random() < .5 else "?", int(rng.integers(1, 30)),
                     START + pd.Timedelta(days=int(rng.integers(0, n_days)), hours=11), 0.0, None, "United Kingdom"))
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])
    df = pd.concat([df, df.sample(frac=.008, random_state=1)], ignore_index=True)   # exact duplicates
    df = df.sort_values(["InvoiceDate", "Invoice"], kind="stable").reset_index(drop=True)
    df["InvoiceDate"] = df.InvoiceDate.dt.strftime("%Y-%m-%d %H:%M:%S")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(df.shape, "invoices:", df.Invoice.nunique(), "missing customer %:", round(df["Customer ID"].isna().mean() * 100, 1),
          "cancel rows:", int(df.Invoice.str.startswith("C").sum()), "size MB:", round(OUT.stat().st_size / 1e6, 2))


if __name__ == "__main__":
    main()
