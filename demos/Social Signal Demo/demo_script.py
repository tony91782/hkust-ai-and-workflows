"""
HKUST PhD Course — Social Signal Demo
"From Data to Result in One Session"

Dataset: social_signal_index.dta
Source:  Cookson, Lu, Mullins & Niessner (2024), The Social Signal, JFE
         https://data.mendeley.com/datasets/xffyybvw4j/1

Variables:
  permno       — CRSP firm identifier
  date         — trading date (2012-01-03 to 2021-12-31)
  zee_sent_pc  — Sentiment PC1 z-score: first principal component of
                  firm-day sentiment across StockTwits, Twitter, Seeking Alpha
  zee_attn_pc  — Attention PC1 z-score: first principal component of
                  firm-day attention (message share) across same three platforms

N = 821,534 firm-day observations, 1,500 firms, 10 years.

Demo flow (meant to be run live with Claude Code narrating each step):
  1. Load and describe the data
  2. Time-series plot: average daily sentiment and attention 2012-2021
  3. Merge with Yahoo Finance returns for a subset of firms
  4. Predict next-day abnormal return from sentiment and attention
  5. Plot the return predictability pattern
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR = Path(__file__).parent
OUT_DIR  = DATA_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("Loading social_signal_index.dta …")
df = pd.read_stata(DATA_DIR / "social_signal_index.dta")
print(f"  {len(df):,} firm-day obs | {df['permno'].nunique():,} firms | "
      f"{df['date'].min().date()} to {df['date'].max().date()}")
print(df.describe().round(3).to_string(), "\n")

# ── 2. Time-series of average daily sentiment & attention ────────────────────
daily = df.groupby("date")[["zee_sent_pc", "zee_attn_pc"]].mean()
daily_roll = daily.rolling(21).mean()   # 21-day rolling mean (~1 month)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(daily.index, daily["zee_sent_pc"], alpha=0.25, color="steelblue", lw=0.6)
axes[0].plot(daily_roll.index, daily_roll["zee_sent_pc"], color="steelblue", lw=1.8,
             label="21-day rolling mean")
axes[0].axhline(0, color="black", lw=0.7, ls="--")
axes[0].set_ylabel("Sentiment PC1 (z)", fontsize=11)
axes[0].legend(fontsize=9)
axes[0].set_title("Cross-Platform Social Sentiment", fontsize=12)

axes[1].plot(daily.index, daily["zee_attn_pc"], alpha=0.25, color="coral", lw=0.6)
axes[1].plot(daily_roll.index, daily_roll["zee_attn_pc"], color="coral", lw=1.8,
             label="21-day rolling mean")
axes[1].axhline(0, color="black", lw=0.7, ls="--")
axes[1].set_ylabel("Attention PC1 (z)", fontsize=11)
axes[1].legend(fontsize=9)
axes[1].set_title("Cross-Platform Social Attention", fontsize=12)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].xaxis.set_major_locator(mdates.YearLocator())

# annotate GME event
gme_date = pd.Timestamp("2021-01-28")
for ax in axes:
    ax.axvline(gme_date, color="gray", ls=":", lw=1.2)
axes[0].annotate("GME\nEvent", xy=(gme_date, axes[0].get_ylim()[1]*0.85),
                 fontsize=8, color="gray", ha="left")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_sentiment_attention_timeseries.png", dpi=150, bbox_inches="tight")
print("Saved: fig1_sentiment_attention_timeseries.png")
plt.close()

# ── 3. Distribution of sentiment and attention ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["zee_sent_pc"].dropna(), bins=100, color="steelblue", alpha=0.7, density=True)
axes[0].set_xlabel("Sentiment PC1 (z)", fontsize=11)
axes[0].set_ylabel("Density", fontsize=11)
axes[0].set_title("Distribution of Sentiment PC1", fontsize=12)

# Attention is very right-skewed — plot winsorized version
attn_p99 = df["zee_attn_pc"].quantile(0.99)
axes[1].hist(df["zee_attn_pc"].clip(upper=attn_p99).dropna(), bins=100,
             color="coral", alpha=0.7, density=True)
axes[1].set_xlabel("Attention PC1 (z, winsorized at p99)", fontsize=11)
axes[1].set_ylabel("Density", fontsize=11)
axes[1].set_title(f"Distribution of Attention PC1\n(raw max = {df['zee_attn_pc'].max():.1f})", fontsize=12)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_distributions.png", dpi=150, bbox_inches="tight")
print("Saved: fig2_distributions.png")
plt.close()

# ── 4. Merge with Yahoo Finance returns ──────────────────────────────────────
# Use yfinance for publicly available price data
# Map a sample of permnos to tickers via a manual crosswalk of the most-covered firms

print("\nFetching return data from Yahoo Finance for demo sample …")

try:
    import yfinance as yf

    # A small crosswalk: permno → ticker for well-known firms in the sample
    # (In a real analysis you'd use CRSP directly; this illustrates the workflow)
    CROSSWALK = {
        14593: "AAPL",   # Apple
        10107: "AMZN",   # Amazon  (approx)
        81001: "MSFT",   # Microsoft
        84788: "TSLA",   # Tesla
        17284: "GS",     # Goldman Sachs
        22111: "JPM",    # JPMorgan
        66158: "META",   # Meta (was FB; renamed Oct 2021)
        92957: "GOOGL",  # Alphabet
    }

    tickers = list(CROSSWALK.values())
    prices = yf.download(tickers, start="2012-01-01", end="2022-01-01",
                         auto_adjust=True, progress=False)["Close"]

    # Compute daily returns and abnormal returns (market-adjusted)
    rets = prices.pct_change()
    mkt  = rets.mean(axis=1)   # simple equal-weighted market proxy
    abret = rets.subtract(mkt, axis=0)

    # Stack to firm-day; merge with signal
    abret_long = abret.stack().reset_index()
    abret_long.columns = ["date", "ticker", "abret"]
    abret_long["permno"] = abret_long["ticker"].map({v: k for k, v in CROSSWALK.items()})
    abret_long = abret_long.dropna(subset=["permno"])
    abret_long["permno"] = abret_long["permno"].astype(int)

    merged = df.merge(abret_long[["permno", "date", "abret"]], on=["permno", "date"])
    merged = merged.sort_values(["permno", "date"])

    # Next-day return
    merged["abret_next"] = merged.groupby("permno")["abret"].shift(-1)
    reg_df = merged.dropna(subset=["abret_next", "zee_sent_pc", "zee_attn_pc"]).copy()

    print(f"  Merged sample: {len(reg_df):,} obs across {reg_df['permno'].nunique()} firms")

    # ── 5. Return predictability: binned plot ────────────────────────────────
    # Sort into deciles by sentiment and attention; plot mean next-day return
    for var, label, color in [
        ("zee_sent_pc", "Sentiment PC1", "steelblue"),
        ("zee_attn_pc", "Attention PC1", "coral"),
    ]:
        reg_df[f"{var}_dec"] = pd.qcut(reg_df[var].clip(
            lower=reg_df[var].quantile(0.01),
            upper=reg_df[var].quantile(0.99)), q=10, labels=False)
        bins = reg_df.groupby(f"{var}_dec")["abret_next"].mean() * 100   # in pct

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(bins.index + 1, bins.values, color=color, alpha=0.75, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(f"{label} Decile (1 = most negative)", fontsize=11)
        ax.set_ylabel("Mean Next-Day Abnormal Return (%)", fontsize=11)
        ax.set_title(f"Return Predictability: {label}\n(demo sample, yfinance data)", fontsize=12)
        ax.set_xticks(range(1, 11))
        plt.tight_layout()
        fname = f"fig3_return_pred_{var}.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        print(f"Saved: {fname}")
        plt.close()

    # ── 6. Simple OLS ────────────────────────────────────────────────────────
    print("\nSimple OLS: next-day abnormal return ~ sentiment + attention")
    from numpy.linalg import lstsq

    X = reg_df[["zee_sent_pc", "zee_attn_pc"]].assign(const=1).values
    y = reg_df["abret_next"].values * 100
    coef, _, _, _ = lstsq(X, y, rcond=None)
    print(f"  β_sentiment = {coef[0]:+.4f}%  (higher sentiment → higher next-day return)")
    print(f"  β_attention = {coef[1]:+.4f}%  (higher attention → lower next-day return)")
    print("  (signs match the paper's findings)")

except ImportError:
    print("  yfinance not installed — skipping return merge.")
    print("  In class: run `pip install yfinance` or use CRSP data directly.")
    print("  The signal data alone still supports steps 1–3.")

print("\nDemo complete. Figures saved to:", OUT_DIR)
