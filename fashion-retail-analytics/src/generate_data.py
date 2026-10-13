"""Generate a reproducible SIMULATED weekly apparel sales dataset.

Replace data/raw_sales.csv with your own Week 1-5 dataset to re-run the pipeline
(keep the same column names or edit src/analysis.py).
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
weeks = pd.date_range("2023-01-02", periods=104, freq="W-MON")
cats = {"Tops": (25, 1.0), "Dresses": (60, 0.9), "Outerwear": (120, 0.6),
        "Denim": (70, 0.8), "Activewear": (45, 0.95), "Accessories": (20, 0.7)}
regions = ["North", "South", "East", "West"]
channels = ["Online", "Store"]
rows = []
for w_i, d in enumerate(weeks):
    m = d.month
    for c, (price, base) in cats.items():
        season = {"Outerwear": 1 + 0.8*np.cos((m-12)/12*2*np.pi),
                  "Dresses": 1 + 0.6*np.cos((m-6)/12*2*np.pi),
                  "Activewear": 1 + 0.3*np.cos((m-1)/12*2*np.pi)}.get(c, 1 + 0.15*np.cos((m-11)/12*2*np.pi))
        for r in regions:
            for ch in channels:
                holiday = int(d.isocalendar().week in (47, 48, 51))
                discount = float(rng.choice([0, 0, 0, 0.1, 0.2, 0.3]))
                if holiday: discount = max(discount, 0.2)
                mkt = max(0, rng.normal(900, 250)) * (1.3 if ch == "Online" else 1.0)
                trend = 1 + 0.002 * w_i
                lam = 40*base*season*trend*(1.15 if ch == "Online" else 1.0)
                lam *= (1 + 1.8*discount) * (1 + mkt/6000) * (1.35 if holiday else 1.0)
                units = rng.poisson(max(lam, 1))
                unit_price = round(price*(1-discount), 2)
                ret_p = np.clip(0.08 + (0.07 if ch == "Online" else 0) + (0.05 if c == "Dresses" else 0)
                                + 0.10*discount + rng.normal(0, 0.02), 0.01, 0.5)
                returns = rng.binomial(units, ret_p)
                rows.append([d, c, r, ch, price, discount, unit_price, round(mkt, 2), holiday,
                             units, returns])
df = pd.DataFrame(rows, columns=["week", "category", "region", "channel", "list_price", "discount",
                                 "unit_price", "marketing_spend", "holiday_week", "units_sold", "returns"])
# inject realistic data-quality problems for the cleaning step
idx = rng.choice(df.index, 90, replace=False); df.loc[idx, "marketing_spend"] = np.nan
idx = rng.choice(df.index, 40, replace=False); df.loc[idx, "region"] = df.loc[idx, "region"].str.lower()
df = pd.concat([df, df.sample(60, random_state=1)], ignore_index=True)
df.to_csv("data/raw_sales.csv", index=False)
print("raw rows:", len(df))
