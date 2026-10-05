import sqlite3
import numpy as np
import pandas as pd
from scipy import stats


def load(db="marketing.db"):
    with sqlite3.connect(db) as con:
        df = pd.read_sql("SELECT * FROM daily_spend", con, parse_dates=["date"])
    df["cpc"] = df["spend"] / df["clicks"]
    df["roas"] = df["revenue"] / df["spend"]
    return df


def detect_anomalies(df, metric="cpc", window=28, threshold=3.5):
    """Robust z-score (median/MAD) against the trailing window, excluding the current day."""
    out = []
    for _, g in df.sort_values("date").groupby("channel"):
        past = g[metric].shift(1).rolling(window, min_periods=14)
        med = past.median()
        mad = past.apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
        g = g.assign(z=(0.6745 * (g[metric] - med) / mad).round(2))
        out.append(g.loc[g["z"].abs() > threshold, ["date", "channel", "spend", metric, "z"]])
    res = pd.concat(out)
    res["date"] = res["date"].dt.strftime("%Y-%m-%d")
    return res.round(2).to_dict("records")


def test_channels(df):
    """Per-channel ROAS with 95% CI and a one-sided test of ROAS < 1 (losing money)."""
    rows = []
    for ch, g in df.groupby("channel"):
        r = g["roas"]
        lo, hi = stats.t.interval(0.95, len(r) - 1, loc=r.mean(), scale=stats.sem(r))
        rows.append({
            "channel": ch,
            "spend": round(g["spend"].sum()),
            "revenue": round(g["revenue"].sum()),
            "mean_daily_roas": round(r.mean(), 2),
            "roas_ci_low": round(lo, 2),
            "roas_ci_high": round(hi, 2),
            "p_roas_below_1": round(stats.ttest_1samp(r, 1, alternative="less").pvalue, 4),
        })
        rows[-1]["losing_money"] = bool(rows[-1]["p_roas_below_1"] < 0.05)
    # Kruskal-Wallis instead of ANOVA: ROAS variances differ a lot across channels
    p = stats.kruskal(*[g["roas"] for _, g in df.groupby("channel")]).pvalue
    return {"kruskal_p": float(p), "channels": rows}
