import sqlite3
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
dates = pd.date_range("2026-06-01", periods=90)

# channel: (daily spend, cost per click, conversion rate, avg order value)
channels = {
    "Paid Search": (500, 1.5, 0.040, 120),
    "Social": (400, 1.0, 0.020, 90),
    "Display": (300, 0.8, 0.008, 80),
    "Email": (100, 0.3, 0.050, 70),
}

rows = []
for ch, (spend, cpc, cvr, aov) in channels.items():
    for d in dates:
        s = rng.normal(spend, spend * 0.1)
        clicks = rng.poisson(s / cpc)
        conv = rng.binomial(clicks, cvr)
        rev = conv * rng.normal(aov, aov * 0.1)
        rows.append([d.strftime("%Y-%m-%d"), ch, round(s, 2), clicks, conv, round(rev, 2)])

df = pd.DataFrame(rows, columns=["date", "channel", "spend", "clicks", "conversions", "revenue"])

# Injected anomaly: Social spend jumps 2.5x in the last 5 days with no extra clicks
mask = (df["channel"] == "Social") & (df["date"] >= dates[-5].strftime("%Y-%m-%d"))
df.loc[mask, "spend"] = (df.loc[mask, "spend"] * 2.5).round(2)

with sqlite3.connect("marketing.db") as con:
    df.to_sql("daily_spend", con, if_exists="replace", index=False)

print(df.groupby("channel")[["spend", "revenue"]].sum().round(0))
