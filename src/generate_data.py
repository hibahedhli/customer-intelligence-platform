"""
Synthetic e-commerce data generator.

IMPORTANT: this dataset is SIMULATED. It is NOT real customer data and must never be
presented as such. It exists so the full workflow can be reproduced without licensing
or privacy constraints. The generator is a transparent mechanistic simulation:

* every customer has a latent behavioural type (loyal / regular / occasional / bargain / fading)
  that sets an order rate, a churn hazard, basket size and discount sensitivity;
* orders arrive as a seasonal Poisson process; activity fades during the ~60 days
  before a customer permanently leaves (this is what makes "declining engagement"
  detectable);
* realistic data-quality problems are injected on purpose (duplicates, mixed date formats,
  inconsistent categories, impossible ages, invalid quantities/prices, legitimate returns...)
  so that the cleaning step has real work to do.

Because the churn mechanism is known, results should be read as a demonstration of the
METHOD, not as evidence about any real business. Swap in a real transaction file
(same schema) to run the identical pipeline.
"""
import numpy as np
import pandas as pd

from .config import DATA_RAW, SEED, ensure_dirs
from .config import SIM_END as DATA_END, SIM_N_CUSTOMERS as N_CUSTOMERS, SIM_START as DATA_START

CATEGORIES = {  # price range (low, high) and purchase popularity
    "Electronics": ((40, 400), 0.10), "Home & Kitchen": ((15, 150), 0.16),
    "Fashion": ((12, 120), 0.22), "Beauty": ((8, 60), 0.14), "Sports": ((15, 200), 0.09),
    "Books": ((5, 40), 0.10), "Toys": ((8, 80), 0.07), "Grocery": ((3, 30), 0.12),
}
CITIES = ["Tunis", "Sfax", "Sousse", "Ariana", "Ben Arous", "Nabeul", "Bizerte", "Monastir", "Gabes", "Kairouan"]
CITY_P = [.22, .14, .11, .12, .09, .07, .06, .06, .06, .07]
PAYMENTS = ["Credit card", "Cash on delivery", "Bank transfer", "Mobile wallet"]
PAY_P = [.45, .30, .10, .15]
DISC_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]
DISC_P = [.30, .30, .20, .15, .05]

TYPES = {  # order rate (orders/day), daily churn hazard, mean items/order, mean qty/item, P(order has discount)
    "loyal": dict(share=.12, rate=1 / 16, hazard=.0006, items=3.0, qty=1.6, disc_p=.15),
    "regular": dict(share=.30, rate=1 / 32, hazard=.0014, items=2.2, qty=1.4, disc_p=.25),
    "occasional": dict(share=.28, rate=1 / 65, hazard=.0022, items=1.8, qty=1.3, disc_p=.30),
    "bargain": dict(share=.15, rate=1 / 40, hazard=.0020, items=2.0, qty=1.3, disc_p=.85),
    "fading": dict(share=.15, rate=1 / 55, hazard=.0065, items=1.6, qty=1.2, disc_p=.35),
}
SEASON = {1: 1.1, 2: .95, 3: 1.0, 4: 1.0, 5: 1.0, 6: .9, 7: .9, 8: .9, 9: 1.05, 10: 1.1, 11: 1.4, 12: 1.35}
MAX_MULT = 1.4
FADE_DAYS = 60


def _catalog(rng, n_products=240):
    cats = list(CATEGORIES)
    rows = []
    for i in range(n_products):
        cat = cats[i % len(cats)]
        lo, hi = CATEGORIES[cat][0]
        rows.append((f"P{i + 1:04d}", cat, round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 2)))
    return pd.DataFrame(rows, columns=["product_id", "product_category", "unit_price"])


def simulate(seed=SEED):
    rng = np.random.default_rng(seed)
    cat = _catalog(rng)
    cat_names = list(CATEGORIES)
    cat_w = np.array([CATEGORIES[c][1] for c in cat_names]); cat_w = cat_w / cat_w.sum()
    pid = {c: cat.loc[cat.product_category == c, "product_id"].to_numpy() for c in cat_names}
    price = {c: cat.loc[cat.product_category == c, "unit_price"].to_numpy() for c in cat_names}

    start = pd.Timestamp(DATA_START)
    n_days = (pd.Timestamp(DATA_END) - start).days
    months = [(start + pd.Timedelta(days=d)).month for d in range(n_days + 1)]

    tnames = list(TYPES)
    types = rng.choice(tnames, size=N_CUSTOMERS, p=[TYPES[t]["share"] for t in tnames])
    reg_day = (rng.beta(1.0, 1.35, N_CUSTOMERS) * (n_days - 15)).astype(int)

    lines, oc, lc = [], 0, 0
    for i in range(N_CUSTOMERS):
        T = TYPES[types[i]]
        lam = T["rate"] * rng.lognormal(0, 0.35)
        churn_day = reg_day[i] + rng.exponential(1 / T["hazard"])
        fav = rng.choice(cat_names, p=cat_w)
        pay = rng.choice(PAYMENTS, p=PAY_P)
        t = float(reg_day[i] + (0 if rng.random() < .85 else rng.integers(1, 8)))
        days = [int(t)] if t <= n_days else []
        while True:
            t += rng.exponential(1 / (lam * MAX_MULT))
            if t > n_days or t > churn_day:
                break
            remaining = churn_day - t
            fade = 1.0 if remaining >= FADE_DAYS else 0.25 + 0.75 * remaining / FADE_DAYS
            if rng.random() < fade * SEASON[months[int(t)]] / MAX_MULT:
                days.append(int(t))
        for d in days:
            oc += 1
            n_items = 1 + rng.poisson(T["items"] - 1)
            disc = float(rng.choice(DISC_LEVELS, p=DISC_P)) if rng.random() < T["disc_p"] else 0.0
            pm = pay if rng.random() < .75 else rng.choice(PAYMENTS)
            for _ in range(n_items):
                c = fav if rng.random() < .5 else rng.choice(cat_names, p=cat_w)
                k = rng.integers(len(pid[c]))
                lc += 1
                lines.append((f"T{lc:07d}", f"O{oc:07d}", f"C{i + 1:05d}", d, pid[c][k], c,
                              int(1 + rng.poisson(T["qty"] - 1)), price[c][k], disc, pm))
    tx = pd.DataFrame(lines, columns=["transaction_id", "order_id", "customer_id", "day", "product_id",
                                      "product_category", "quantity", "unit_price", "discount", "payment_method"])
    tx["transaction_date"] = start + pd.to_timedelta(tx.day, unit="D")

    # legitimate returns: ~2.5% of lines, negative quantity, id/order prefixed with "R"
    ret = tx[rng.random(len(tx)) < 0.025].copy()
    ret["transaction_id"] = "R" + ret.transaction_id
    ret["order_id"] = "R" + ret.order_id
    ret["quantity"] = -ret.quantity
    ret["transaction_date"] = ret.transaction_date + pd.to_timedelta(rng.integers(3, 22, len(ret)), unit="D")
    ret = ret[ret.transaction_date <= pd.Timestamp(DATA_END)]
    tx = pd.concat([tx, ret], ignore_index=True).drop(columns="day")

    n = N_CUSTOMERS
    age = np.clip(np.round(rng.normal(34, 11, n)), 18, 70)
    customers = pd.DataFrame({
        "customer_id": [f"C{i + 1:05d}" for i in range(n)],
        "customer_age": age,
        "customer_gender": rng.choice(["Female", "Male"], n, p=[.52, .48]),
        "customer_location": rng.choice(CITIES, n, p=CITY_P),
        "customer_registration_date": start + pd.to_timedelta(reg_day, unit="D"),
    })
    return customers, tx, rng


def inject_quality_issues(customers, tx, rng):
    """Corrupt the clean simulation the way real exports are corrupted."""
    tx, customers = tx.copy(), customers.copy()
    n = len(tx)
    is_ret = tx.transaction_id.str.startswith("R").to_numpy()

    # dates: 6% exported as dd/mm/YYYY
    dstr = tx.transaction_date.dt.strftime("%Y-%m-%d")
    alt = rng.random(n) < .06
    dstr[alt] = tx.transaction_date[alt].dt.strftime("%d/%m/%Y")
    tx["transaction_date"] = dstr

    # inconsistent categorical spellings
    u = rng.random(n)
    c = tx.product_category.copy()
    c[u < .04] = c[u < .04].str.lower()
    m = (u >= .04) & (u < .07); c[m] = c[m] + " "
    m = (u >= .07) & (u < .085); c[m] = c[m].str.upper()
    tx["product_category"] = c
    low = rng.random(n) < .03
    tx.loc[low, "payment_method"] = tx.loc[low, "payment_method"].str.lower()

    # invalid values on non-return lines
    pur = np.where(~is_ret)[0]
    q = rng.choice(pur, int(.0025 * n), replace=False); tx.loc[tx.index[q], "quantity"] *= -1
    z = rng.choice(pur, int(.001 * n), replace=False); tx.loc[tx.index[z], "quantity"] = 0
    p = rng.choice(pur, int(.003 * n), replace=False)
    tx.loc[tx.index[p[::2]], "unit_price"] = 0.0
    tx.loc[tx.index[p[1::2]], "unit_price"] *= -1
    d = rng.choice(pur, int(.001 * n), replace=False)
    tx.loc[tx.index[d], "discount"] = (tx.loc[tx.index[d], "discount"] * 100).replace(0, 10)

    # exact duplicate rows (double-posted transactions)
    tx = pd.concat([tx, tx.sample(frac=.012, random_state=SEED)], ignore_index=True)
    tx = tx.sort_values("transaction_id", kind="stable").reset_index(drop=True)

    # customers
    k = len(customers)
    age = customers.customer_age.copy()
    age[rng.random(k) < .03] = np.nan
    bad = rng.random(k) < .005
    age[bad] = rng.choice([0, -1, 150, 999], bad.sum())
    customers["customer_age"] = age
    g = customers.customer_gender.copy().astype(object)
    for lab, short in (("Female", "F"), ("Male", "M")):
        m = (g == lab).to_numpy()
        v = rng.choice([lab, short, lab.lower(), " " + lab], m.sum(), p=[.8, .1, .07, .03])
        g[m] = v
    g[rng.random(k) < .02] = np.nan
    customers["customer_gender"] = g
    loc = customers.customer_location.copy().astype(object)
    lw = rng.random(k) < .03; loc[lw] = loc[lw].str.lower()
    loc[rng.random(k) < .015] = np.nan
    customers["customer_location"] = loc
    r = customers.customer_registration_date.dt.strftime("%Y-%m-%d")
    ra = rng.random(k) < .03
    r[ra] = customers.customer_registration_date[ra].dt.strftime("%d/%m/%Y")
    customers["customer_registration_date"] = r
    return customers, tx


def main():
    ensure_dirs()
    customers, tx, rng = simulate()
    customers, tx = inject_quality_issues(customers, tx, rng)
    customers.to_csv(DATA_RAW / "customers_raw.csv", index=False)
    tx.to_csv(DATA_RAW / "transactions_raw.csv", index=False)
    print(f"customers_raw: {customers.shape} | transactions_raw: {tx.shape}")


if __name__ == "__main__":
    main()
