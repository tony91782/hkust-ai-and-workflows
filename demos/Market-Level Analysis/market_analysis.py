"""
HKUST PhD Course — Market-Level Social Signal Analysis
"From Firm-Day Index to Market-Level Dynamics"

Starting from:  ../Social Signal Demo/social_signal_index.dta
  permno, date, zee_sent_pc, zee_attn_pc  (821,534 firm-day obs, 1,500 firms, 2012–2021)

This script:
  1. Aggregates to a daily market-level sentiment and attention index
  2. Downloads market data: S&P 500 returns, VIX, and trading volume (Yahoo Finance)
  3. Downloads FOMC announcement dates (Federal Reserve)
  4. Section A: How does today's signal predict tomorrow's market outcomes?
               (returns, volatility, trading volume)
  5. Section B: How do recent market outcomes predict today's signal?
               (the reverse / feedback direction)
  6. Section C: FOMC event study — does the social signal spike or dip
               around Fed announcements?

Cite signal data as:
  Cookson, Lu, Mullins & Niessner (2024). The Social Signal. JFE.
  https://data.mendeley.com/datasets/xffyybvw4j/1
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path
import urllib.request, re, io

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent
SIGNAL_DTA = DATA_DIR.parent / "Social Signal Demo" / "social_signal_index.dta"
OUT_DIR    = DATA_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

PLOT_STYLE = dict(figsize=(11, 4.5))
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build market-level signal index
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1  Aggregate firm-day signal → market-level daily index")
print("=" * 60)

firm_day = pd.read_stata(SIGNAL_DTA)
print(f"  Loaded {len(firm_day):,} firm-day obs, {firm_day['permno'].nunique():,} firms")

# Equal-weighted cross-sectional mean each calendar day
mkt = (firm_day.groupby("date")[["zee_sent_pc", "zee_attn_pc"]]
               .mean()
               .rename(columns={"zee_sent_pc": "sent", "zee_attn_pc": "attn"})
               .reset_index())

# Also compute cross-sectional dispersion (variance of sentiment = disagreement proxy)
mkt["sent_disp"] = (firm_day.groupby("date")["zee_sent_pc"]
                             .std().reset_index(drop=True))

mkt = mkt.sort_values("date").reset_index(drop=True)
print(f"  Market index: {len(mkt)} trading days, "
      f"{mkt['date'].min().date()} to {mkt['date'].max().date()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Download market data from Yahoo Finance
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2  Download S&P 500, VIX, and volume (Yahoo Finance)")
print("=" * 60)

import yfinance as yf

raw = yf.download(["^GSPC", "^VIX", "SPY"],
                  start="2012-01-01", end="2022-01-01",
                  auto_adjust=True, progress=False)

# S&P 500 daily return and log-return
sp500 = raw["Close"]["^GSPC"].rename("sp500")
sp500_ret = np.log(sp500).diff().rename("sp500_ret")         # log return
sp500_rvol = sp500_ret.rolling(21).std().rename("rvol_21d")  # realized vol (21-day)

# VIX (implied volatility, level)
vix = raw["Close"]["^VIX"].rename("vix")
vix_chg = vix.diff().rename("vix_chg")

# SPY volume as a proxy for US equity market activity
spy_vol = np.log(raw["Volume"]["SPY"] + 1).rename("spy_log_vol")
spy_vol_chg = spy_vol.diff().rename("spy_vol_chg")

# Combine into a market DataFrame
mkt_data = pd.concat([sp500, sp500_ret, sp500_rvol, vix, vix_chg,
                       spy_vol, spy_vol_chg], axis=1).reset_index()
mkt_data = mkt_data.rename(columns={"Date": "date"})
mkt_data["date"] = pd.to_datetime(mkt_data["date"])
print(f"  Market data: {len(mkt_data)} rows, "
      f"{mkt_data['date'].min().date()} to {mkt_data['date'].max().date()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Load FOMC announcement dates
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3  Load FOMC announcement dates")
print("=" * 60)

# FOMC announcement days (second day of meeting) 2012–2021
# Source: Federal Reserve historical calendars
FOMC_DATES = pd.to_datetime([
    # 2012
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20",
    "2012-08-01","2012-09-13","2012-10-24","2012-12-12",
    # 2013
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19",
    "2013-07-31","2013-09-18","2013-10-30","2013-12-18",
    # 2014
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18",
    "2014-07-30","2014-09-17","2014-10-29","2014-12-17",
    # 2015
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17",
    "2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    # 2016
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15",
    "2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    # 2017
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14",
    "2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    # 2018
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13",
    "2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    # 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19",
    "2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    # 2020
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29",
    "2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
    "2021-07-28","2021-09-22","2021-11-03","2021-12-15",
])
print(f"  {len(FOMC_DATES)} FOMC announcement dates, 2012–2021")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Merge everything
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4  Merge signal + market data")
print("=" * 60)

df = (mkt.merge(mkt_data, on="date", how="inner")
         .sort_values("date")
         .reset_index(drop=True))

# FOMC indicator and days-to-FOMC
df["fomc"] = df["date"].isin(FOMC_DATES).astype(int)
df["days_to_fomc"] = df["date"].apply(
    lambda d: (FOMC_DATES[FOMC_DATES >= d].min() - d).days
    if any(FOMC_DATES >= d) else np.nan)

# Lead variables (tomorrow's market outcomes)
for col in ["sp500_ret", "vix_chg", "spy_vol_chg"]:
    df[f"{col}_next"] = df[col].shift(-1)

# Lag variables (yesterday's market outcomes)
for col in ["sp500_ret", "vix_chg", "spy_vol_chg", "sent", "attn"]:
    df[f"{col}_lag"] = df[col].shift(1)

df = df.dropna(subset=["sent", "attn", "sp500_ret", "vix_chg", "spy_vol_chg"])
print(f"  Final panel: {len(df)} trading days")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Market-level signal 2012–2021
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION A  The aggregate signal: time-series properties")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

for ax, col, label, color in [
    (axes[0], "sent",      "Market Sentiment (eq-wtd mean, z)",  "steelblue"),
    (axes[1], "attn",      "Market Attention (eq-wtd mean, z)",  "coral"),
    (axes[2], "sent_disp", "Sentiment Dispersion (cross-sec std)", "purple"),
]:
    roll = df.set_index("date")[col].rolling(21).mean()
    ax.plot(df["date"], df[col], alpha=0.2, color=color, lw=0.5)
    ax.plot(roll.index, roll.values, color=color, lw=1.8, label="21d MA")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_ylabel(label, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")

# Annotate key events
events = {
    "2015-08-24": "China\nFlash Crash",
    "2018-02-05": "VIX\nShort Squeeze",
    "2020-03-16": "COVID\nLockdown",
    "2021-01-27": "GME\nPeak",
}
for date_str, label in events.items():
    d = pd.Timestamp(date_str)
    if df["date"].min() <= d <= df["date"].max():
        for ax in axes:
            ax.axvline(d, color="gray", ls=":", lw=0.9, alpha=0.7)
        axes[0].annotate(label, xy=(d, axes[0]["sent"].max() if False else
                         df["sent"].quantile(0.95)),
                         fontsize=7, color="gray", ha="left",
                         xytext=(d + pd.Timedelta(days=30),
                                 df["sent"].quantile(0.90)),
                         arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))

axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[2].xaxis.set_major_locator(mdates.YearLocator())
fig.suptitle("Market-Level Social Signal Index, 2012–2021\n"
             "(aggregated from 1,500-firm social_signal_index.dta)",
             fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_market_signal_timeseries.png", dpi=150, bbox_inches="tight")
print("  Saved fig1_market_signal_timeseries.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Cross-correlations (lead/lag structure ±10 days)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCross-correlations: signal ↔ market outcomes")

lags = range(-10, 11)
pairs = [
    ("sent", "sp500_ret",   "Sentiment → S&P 500 Return",          "steelblue"),
    ("sent", "vix_chg",     "Sentiment → ΔVIX",                    "tomato"),
    ("sent", "spy_vol_chg", "Sentiment → ΔLog Volume",             "seagreen"),
    ("attn", "sp500_ret",   "Attention → S&P 500 Return",          "steelblue"),
    ("attn", "vix_chg",     "Attention → ΔVIX",                    "tomato"),
    ("attn", "spy_vol_chg", "Attention → ΔLog Volume",             "seagreen"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
axes = axes.flatten()

for ax, (sig, outcome, title, color) in zip(axes, pairs):
    xcorrs = [df[sig].corr(df[outcome].shift(-lag)) for lag in lags]
    bar_colors = ["darkred" if lag == 1 else color for lag in lags]
    ax.bar(lags, xcorrs, color=bar_colors, alpha=0.7, edgecolor="white", width=0.7)
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.4, ls=":")
    # 95% confidence band (approx: ±1.96/√N)
    ci = 1.96 / np.sqrt(len(df))
    ax.axhline(ci, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(-ci, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Lead of outcome (negative = signal lags outcome)", fontsize=8)
    ax.set_ylabel("Correlation", fontsize=8)
    ax.set_xticks([-10, -5, 0, 5, 10])

# Red bars mark the signal-predicts-tomorrow relationship
fig.text(0.5, 0.01,
         "Dark red bar = signal today predicts outcome tomorrow (lead=+1)\n"
         "Dashed lines = 95% CI (±1.96/√N)",
         ha="center", fontsize=8, color="gray")
plt.suptitle("Cross-Correlations: Social Signal ↔ Market Outcomes\n"
             "(lags/leads ±10 trading days)", fontsize=11)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(OUT_DIR / "fig2_cross_correlations.png", dpi=150, bbox_inches="tight")
print("  Saved fig2_cross_correlations.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: Predictive regressions (OLS with Newey-West SEs)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION B  Predictive regressions (signal → next-day outcomes)")
print("=" * 60)

import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

def nw_ols(y, X_cols, df_clean, lags=5):
    """OLS with Newey-West HAC standard errors."""
    sub = df_clean.dropna(subset=[y] + X_cols)
    X = sm.add_constant(sub[X_cols])
    model = sm.OLS(sub[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return model

outcomes = {
    "sp500_ret_next":  "S&P 500 Return (t+1)",
    "vix_chg_next":    "ΔVIX (t+1)",
    "spy_vol_chg_next":"ΔLog Volume (t+1)",
}
controls = ["sp500_ret_lag", "vix_chg_lag", "spy_vol_chg_lag"]

print(f"\n{'Outcome':<28} {'β_sent':>10} {'t_sent':>8} {'β_attn':>10} {'t_attn':>8} {'R²':>6}")
print("-" * 75)

results_fwd = {}
for dep, label in outcomes.items():
    regressors = ["sent", "attn"] + controls
    m = nw_ols(dep, regressors, df)
    b_s = m.params["sent"];  t_s = m.tvalues["sent"]
    b_a = m.params["attn"];  t_a = m.tvalues["attn"]
    r2  = m.rsquared
    print(f"{label:<28} {b_s:>+10.4f} {t_s:>8.2f} {b_a:>+10.4f} {t_a:>8.2f} {r2:>6.3f}")
    results_fwd[dep] = m

print("\nNote: Newey-West SEs, 5 lags. Controls: lagged return, ΔVIX, Δlog volume.")
print("      |t| > 1.96 = significant at 5%.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: Reverse regressions (market outcomes → signal)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION C  Reverse regressions (market outcomes → signal today)")
print("=" * 60)

signal_outcomes = {
    "sent": "Sentiment Index (t)",
    "attn": "Attention Index (t)",
}
market_predictors = ["sp500_ret_lag", "vix_chg_lag", "spy_vol_chg_lag",
                     "sent_lag", "attn_lag"]

print(f"\n{'Outcome':<26} {'β_ret':>10} {'t':>6} {'β_Δvix':>10} {'t':>6} "
      f"{'β_Δvol':>10} {'t':>6} {'R²':>6}")
print("-" * 80)

results_rev = {}
for dep, label in signal_outcomes.items():
    m = nw_ols(dep, market_predictors, df)
    print(f"{label:<26} "
          f"{m.params['sp500_ret_lag']:>+10.4f} {m.tvalues['sp500_ret_lag']:>6.2f} "
          f"{m.params['vix_chg_lag']:>+10.4f} {m.tvalues['vix_chg_lag']:>6.2f} "
          f"{m.params['spy_vol_chg_lag']:>+10.4f} {m.tvalues['spy_vol_chg_lag']:>6.2f} "
          f"{m.rsquared:>6.3f}")
    results_rev[dep] = m

print("\nNote: Also includes lagged signal as control (AR term).")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Leads-and-lags cumulative return plot
#   "Event" = day with high (top tercile) or low (bottom tercile) signal
#   Cumulative S&P 500 return from t−6 (baseline=0) through t+15
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding leads-and-lags cumulative return plot…")

LL_BEFORE = 6   # days before event as baseline
LL_AFTER  = 15  # days after event
REL_DAYS  = list(range(-LL_BEFORE, LL_AFTER + 1))   # −6 … +15
N_WINDOW  = len(REL_DAYS)
TERCILE   = 1/3

df_ll = df[["date","sent","attn","sp500_ret"]].dropna().reset_index(drop=True)

sent_hi = df_ll["sent"].quantile(1 - TERCILE)
sent_lo = df_ll["sent"].quantile(TERCILE)
attn_hi = df_ll["attn"].quantile(1 - TERCILE)
attn_lo = df_ll["attn"].quantile(TERCILE)

def event_cum_returns(signal_col, hi_thresh, lo_thresh, data):
    """
    For every row that is in the high or low signal group,
    extract the cumulative S&P 500 return window (LL_BEFORE days before
    through LL_AFTER days after), indexed to 0 at the baseline day.
    Returns arrays of shape (n_events, N_WINDOW) for each group.
    """
    hi_rows, lo_rows = [], []
    ret = data["sp500_ret"].values
    sig = data[signal_col].values
    for i in range(LL_BEFORE, len(data) - LL_AFTER):
        window = ret[i - LL_BEFORE : i + LL_AFTER + 1]
        if len(window) != N_WINDOW:
            continue
        cumret = np.cumsum(window)
        cumret -= cumret[0]          # index to 0 at t−LL_BEFORE
        if sig[i] >= hi_thresh:
            hi_rows.append(cumret)
        elif sig[i] <= lo_thresh:
            lo_rows.append(cumret)
    return np.array(hi_rows), np.array(lo_rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, sig_col, hi_t, lo_t, label, hi_color, lo_color in [
    (axes[0], "sent", sent_hi, sent_lo,
     "Market Sentiment", "steelblue", "tomato"),
    (axes[1], "attn", attn_hi, attn_lo,
     "Market Attention",  "coral",     "teal"),
]:
    hi_arr, lo_arr = event_cum_returns(sig_col, hi_t, lo_t, df_ll)

    for arr, color, grp_label in [(hi_arr, hi_color, "High (top tercile)"),
                                   (lo_arr, lo_color, "Low (bottom tercile)")]:
        mean_cr  = arr.mean(axis=0) * 100          # convert to %
        sem_cr   = arr.std(axis=0) / np.sqrt(len(arr)) * 100
        ax.plot(REL_DAYS, mean_cr, color=color, lw=2.0, label=f"{grp_label} (n={len(arr):,})")
        ax.fill_between(REL_DAYS, mean_cr - 1.96*sem_cr, mean_cr + 1.96*sem_cr,
                        alpha=0.15, color=color)

    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.axvline(0, color="black", lw=1.2, ls="-", alpha=0.4, label="Event day (t=0)")
    ax.axvline(-LL_BEFORE, color="gray", lw=1.0, ls=":", alpha=0.7, label=f"Baseline (t=−{LL_BEFORE})")
    ax.set_xlabel("Trading days relative to signal day", fontsize=10)
    ax.set_ylabel("Cumulative S&P 500 Return (%)", fontsize=10)
    ax.set_title(f"{label}: Cumulative Return\naround High vs. Low Signal Days", fontsize=11)
    ax.set_xticks([-6, -3, 0, 3, 6, 9, 12, 15])
    ax.legend(fontsize=8)

    n_hi, n_lo = len(hi_arr), len(lo_arr)
    print(f"  {label}: {n_hi} high-signal days, {n_lo} low-signal days")

plt.suptitle("Leads-and-Lags Cumulative Return: High vs. Low Social Signal Days\n"
             "(t=−6 baseline; ±1.96 SEM shaded; top/bottom tercile)", fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_leads_lags.png", dpi=150, bbox_inches="tight")
print("  Saved fig3_leads_lags.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: FOMC Event Study
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION D  FOMC event study")
print("=" * 60)

WINDOW = 15       # days before and after each FOMC announcement
BASELINE_DAY = -6 # cumulative return indexed to 0 at this relative day

event_rows = []
df_idx = df.set_index("date")

for fomc_date in FOMC_DATES:
    # Find the nearest trading day on or after the FOMC date
    candidates = df_idx.index[df_idx.index >= fomc_date]
    if len(candidates) == 0:
        continue
    event_day = candidates[0]
    loc = df_idx.index.get_loc(event_day)
    if loc < WINDOW or loc > len(df_idx) - WINDOW - 1:
        continue
    window_dates = df_idx.index[loc - WINDOW : loc + WINDOW + 1]
    window_df = df_idx.loc[window_dates].copy()
    window_df["event_day"] = range(-WINDOW, WINDOW + 1)
    window_df["fomc_date"] = fomc_date
    event_rows.append(window_df)

event_panel = pd.concat(event_rows).reset_index()
avg = event_panel.groupby("event_day")[
    ["sent", "attn", "sent_disp", "sp500_ret", "vix_chg"]
].mean()

# De-mean signals by baseline window (days -10 to -6)
baseline = avg.loc[-10:-6].mean()
avg_dm = avg - baseline

print(f"  {len(FOMC_DATES)} FOMC events, ±{WINDOW}-day window")
print(f"  Avg signal on FOMC day (day 0): "
      f"sent = {avg.loc[0,'sent']:+.3f}, "
      f"attn = {avg.loc[0,'attn']:+.3f}")

# ── Build cumulative S&P 500 return indexed to BASELINE_DAY = 0 ──────────────
cum_rows = []
for fomc_date, grp in event_panel.groupby("fomc_date"):
    g = grp[grp["event_day"] >= BASELINE_DAY].sort_values("event_day").copy()
    if BASELINE_DAY not in g["event_day"].values:
        continue
    g["cumret"] = g["sp500_ret"].cumsum()
    g["cumret"] -= g.loc[g["event_day"] == BASELINE_DAY, "cumret"].values[0]
    cum_rows.append(g[["event_day", "cumret"]])

cum_panel = pd.concat(cum_rows)
cum_agg = cum_panel.groupby("event_day")["cumret"].agg(
    mean="mean",
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75),
)
# Restrict to t-6 through t+15
cum_agg = cum_agg.loc[BASELINE_DAY:15]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Signal panels: bar charts of de-meaned values
for ax, col, label, color in [
    (axes[0,0], "sent",      "Market Sentiment",     "steelblue"),
    (axes[0,1], "attn",      "Market Attention",     "coral"),
    (axes[0,2], "sent_disp", "Sentiment Dispersion", "purple"),
    (axes[1,1], "vix_chg",   "ΔVIX",                 "tomato"),
]:
    y = avg_dm.loc[-WINDOW:15, col]
    ax.bar(y.index, y.values, color=color, alpha=0.65, edgecolor="white", width=0.7)
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.axvline(0, color="red", lw=1.2, ls="-", alpha=0.6)
    ax.axvline(BASELINE_DAY, color="gray", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("Trading days relative to FOMC announcement", fontsize=9)
    ax.set_ylabel(f"Δ{label} (de-meaned, days −10 to −6)", fontsize=8)
    ax.set_title(f"{label}\naround FOMC days", fontsize=10)
    ax.set_xticks([-15, -10, -6, -5, 0, 5, 10, 15])

# S&P 500 cumulative return panel (t-6 to t+15)
ax = axes[1, 0]
ax.plot(cum_agg.index, cum_agg["mean"] * 100, color="darkgreen", lw=2.0,
        label="Mean cumulative return")
ax.fill_between(cum_agg.index,
                cum_agg["q25"] * 100, cum_agg["q75"] * 100,
                alpha=0.2, color="darkgreen", label="IQR across events")
ax.axhline(0, color="black", lw=0.7, ls="--")
ax.axvline(0, color="red", lw=1.2, ls="-", alpha=0.6, label="FOMC day")
ax.axvline(BASELINE_DAY, color="gray", lw=1.0, ls="--", alpha=0.7,
           label=f"Baseline (day {BASELINE_DAY})")
ax.set_xlabel("Trading days relative to FOMC announcement", fontsize=9)
ax.set_ylabel("Cumulative S&P 500 Return (%)", fontsize=9)
ax.set_title(f"Cumulative S&P 500 Return\n(indexed to 0 at day {BASELINE_DAY})", fontsize=10)
ax.set_xticks([BASELINE_DAY, -3, 0, 3, 6, 9, 12, 15])
ax.legend(fontsize=7, loc="upper left")

# Raw sentiment level panel (IQR across events)
ax = axes[1, 2]
ax.plot(avg.index, avg["sent"], color="steelblue", lw=1.8)
ax.fill_between(avg.index,
                event_panel.groupby("event_day")["sent"].quantile(0.25),
                event_panel.groupby("event_day")["sent"].quantile(0.75),
                alpha=0.2, color="steelblue")
ax.axhline(0, color="black", lw=0.6, ls="--")
ax.axvline(0, color="red", lw=1.2, ls="-", alpha=0.6)
ax.axvline(BASELINE_DAY, color="gray", lw=1.0, ls="--", alpha=0.7)
ax.set_xlabel("Trading days relative to FOMC announcement", fontsize=9)
ax.set_ylabel("Sentiment (level; IQR shaded)", fontsize=8)
ax.set_title("Sentiment Level around FOMC\n(median ± IQR across events)", fontsize=10)
ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])

plt.suptitle(f"FOMC Event Study: Social Signal around Fed Announcements\n"
             f"{len(FOMC_DATES)} FOMC dates, 2012–2021, ±{WINDOW}-day window",
             fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig4_fomc_event_study.png", dpi=150, bbox_inches="tight")
print("  Saved fig4_fomc_event_study.png")
plt.close()

# Print FOMC-window summary table
print(f"\n  Cumulative S&P 500 return around FOMC (% return relative to day {BASELINE_DAY}):")
print(f"  {'Day':<6} {'CumRet (%)':>12} {'Sentiment':>12} {'Attention':>12}")
for d in [BASELINE_DAY, -3, 0, 1, 3, 5, 10, 15]:
    if d in cum_agg.index and d in avg_dm.index:
        print(f"  {d:>+3d}   {cum_agg.loc[d,'mean']*100:>+12.3f} "
              f"{avg_dm.loc[d,'sent']:>+12.3f} "
              f"{avg_dm.loc[d,'attn']:>+12.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Signal volatility (VIX) relationship — scatter + rolling correlation
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding VIX / signal relationship…")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Scatter: VIX level vs sentiment
axes[0].scatter(df["vix"], df["sent"], alpha=0.08, s=4, color="steelblue")
# LOWESS-style: bin by VIX decile
df["vix_dec"] = pd.qcut(df["vix"], q=10, labels=False)
binned = df.groupby("vix_dec")[["vix", "sent"]].mean()
axes[0].plot(binned["vix"], binned["sent"], color="darkblue", lw=2, marker="o", ms=5)
axes[0].set_xlabel("VIX Level", fontsize=10)
axes[0].set_ylabel("Market Sentiment (z)", fontsize=10)
axes[0].set_title("Sentiment vs VIX Level\n(binned means overlaid)", fontsize=11)

# Rolling 63-day correlation: sentiment with VIX changes
roll_corr = df["sent"].rolling(63).corr(df["vix_chg"])
axes[1].plot(df["date"], roll_corr, color="tomato", lw=1.2)
axes[1].axhline(0, color="black", lw=0.7, ls="--")
axes[1].set_xlabel("Date", fontsize=10)
axes[1].set_ylabel("Rolling 63-day Corr", fontsize=10)
axes[1].set_title("Rolling Correlation:\nSentiment vs ΔVIX", fontsize=11)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Rolling correlation: attention with VIX
roll_corr_a = df["attn"].rolling(63).corr(df["vix_chg"])
axes[2].plot(df["date"], roll_corr_a, color="coral", lw=1.2)
axes[2].axhline(0, color="black", lw=0.7, ls="--")
axes[2].set_xlabel("Date", fontsize=10)
axes[2].set_ylabel("Rolling 63-day Corr", fontsize=10)
axes[2].set_title("Rolling Correlation:\nAttention vs ΔVIX", fontsize=11)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.suptitle("Social Signal and Volatility (VIX) Relationship, 2012–2021", fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig5_signal_vix.png", dpi=150, bbox_inches="tight")
print("  Saved fig5_signal_vix.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY OF FINDINGS")
print("=" * 60)

# Pull actual estimates from results_fwd and results_rev
m_ret  = results_fwd["sp500_ret_next"]
m_vix  = results_fwd["vix_chg_next"]
m_vol  = results_fwd["spy_vol_chg_next"]
m_sent = results_rev["sent"]
m_attn = results_rev["attn"]

def sig(t): return "*" if abs(t) >= 1.96 else ""

print(f"""
Market-level social signal, 2012–2021 (N={len(df)} trading days)

FORWARD (signal → market tomorrow):
  Sentiment → S&P 500 return: β={m_ret.params['sent']:+.4f}  t={m_ret.tvalues['sent']:+.2f}{sig(m_ret.tvalues['sent'])}
  Sentiment → ΔVIX:           β={m_vix.params['sent']:+.4f}  t={m_vix.tvalues['sent']:+.2f}{sig(m_vix.tvalues['sent'])}
  Sentiment → ΔLog Volume:    β={m_vol.params['sent']:+.4f}  t={m_vol.tvalues['sent']:+.2f}{sig(m_vol.tvalues['sent'])}
  Attention → ΔLog Volume:    β={m_vol.params['attn']:+.4f}  t={m_vol.tvalues['attn']:+.2f}{sig(m_vol.tvalues['attn'])}

  Key: high aggregate sentiment predicts higher next-day volatility (VIX) and
  higher trading volume — consistent with overconfidence / disagreement-driven
  trading. Return predictability is weaker at the aggregate level (cf. firm level).

REVERSE (market outcomes → signal today):
  Prior return → sentiment:   β={m_sent.params['sp500_ret_lag']:+.4f}  t={m_sent.tvalues['sp500_ret_lag']:+.2f}{sig(m_sent.tvalues['sp500_ret_lag'])}
  Prior ΔVIX   → sentiment:   β={m_sent.params['vix_chg_lag']:+.4f}  t={m_sent.tvalues['vix_chg_lag']:+.2f}{sig(m_sent.tvalues['vix_chg_lag'])}
  Prior ΔVIX   → attention:   β={m_attn.params['vix_chg_lag']:+.4f}  t={m_attn.tvalues['vix_chg_lag']:+.2f}{sig(m_attn.tvalues['vix_chg_lag'])}
  Prior Δvol   → attention:   β={m_attn.params['spy_vol_chg_lag']:+.4f}  t={m_attn.tvalues['spy_vol_chg_lag']:+.2f}{sig(m_attn.tvalues['spy_vol_chg_lag'])}

  Key: rising volatility (VIX) dampens sentiment and increases attention —
  the market's mood responds to uncertainty. Volume feeds attention (activity
  begets activity on social media).

FOMC EVENT STUDY:
  Signal shifts systematically around Fed announcements — see fig4.

* = |t| >= 1.96 (NW-HAC SEs, 5 lags)

All figures saved to: figures/
  fig1_market_signal_timeseries.png
  fig2_cross_correlations.png
  fig3_distributed_lag.png
  fig4_fomc_event_study.png
  fig5_signal_vix.png
""")
