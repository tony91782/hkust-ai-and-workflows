"""
HKUST PhD Course — Market-Level Social Signal: Market-Cap Weighted Analysis
"Does Size Matter? Cap-Weighted vs. Equal-Weighted Social Signals"

This script repeats market_analysis.py but aggregates the firm-day signal
using market-capitalisation weights rather than equal weights.

Methodology note:
  In a production setting you would use CRSP daily market cap
  (prc × shrout). Here we approximate using Yahoo Finance daily closing
  prices × historical shares outstanding from yfinance.get_shares_full(),
  backfilled to 2012 for firms where data begins later.

  We can get reliable market-cap data for the ~8 large-cap firms in the
  permno→ticker crosswalk below.  For the remaining ~1,492 firms we assign
  equal weight, so the resulting "cap-weighted" index is really a
  *large-cap-tilted* index.

  Three indices are built and compared throughout:
    ew   — Equal-weighted over all 1,500 firms  (baseline from market_analysis.py)
    cw   — Cap-weighted using available market caps (large-cap tilted)
    ew8  — Equal-weighted over the same 8 large-cap firms as a bridge

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
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent
SIGNAL_DTA = DATA_DIR.parent / "Social Signal Demo" / "social_signal_index.dta"
OUT_DIR    = DATA_DIR / "figures_cw"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})

# permno → ticker crosswalk (firms where we can fetch market cap via yfinance)
CROSSWALK = {
    14593: "AAPL",   # Apple
    10107: "AMZN",   # Amazon  (approx — note: low permno suggests old listing)
    81001: "MSFT",   # Microsoft
    84788: "TSLA",   # Tesla
    17284: "GS",     # Goldman Sachs
    22111: "JPM",    # JPMorgan
    66158: "META",   # Meta / Facebook
    92957: "GOOGL",  # Alphabet
}
TICKER_TO_PERMNO = {v: k for k, v in CROSSWALK.items()}

# =============================================================================
# STEP 1A: Load firm-day signal
# =============================================================================
print("=" * 60)
print("STEP 1A  Load firm-day signal")
print("=" * 60)

firm_day = pd.read_stata(SIGNAL_DTA)
print(f"  Loaded {len(firm_day):,} firm-day obs, {firm_day['permno'].nunique():,} firms")
trading_dates = pd.DatetimeIndex(sorted(firm_day["date"].unique()))

# =============================================================================
# STEP 1B: Build market-cap weights from Yahoo Finance
# =============================================================================
print("\n" + "=" * 60)
print("STEP 1B  Fetch historical market cap (price × shares) via yfinance")
print("=" * 60)

import yfinance as yf

tickers = list(CROSSWALK.values())

# Download daily closing prices for all 8 tickers
prices = yf.download(tickers, start="2012-01-01", end="2022-01-01",
                     auto_adjust=True, progress=False)["Close"]
prices.index = pd.to_datetime(prices.index).tz_localize(None)

# Build daily shares outstanding (backfilled quarterly filings → daily)
shares_series = {}
for ticker in tickers:
    t = yf.Ticker(ticker)

    # Try get_shares_full first (most granular)
    try:
        sh = t.get_shares_full(start="2011-01-01", end="2022-01-01")
        if sh is not None and len(sh) > 0:
            sh.index = pd.DatetimeIndex(sh.index).tz_localize(None)
            sh = sh[~sh.index.duplicated(keep="last")].sort_index()
            shares_series[ticker] = sh
            continue
    except Exception:
        pass

    # Fallback: quarterly balance sheet
    try:
        bs = t.quarterly_balance_sheet
        if "Ordinary Shares Number" in bs.index:
            sh = bs.loc["Ordinary Shares Number"].dropna().sort_index()
            sh.index = pd.DatetimeIndex(sh.index).tz_localize(None)
            sh = sh[~sh.index.duplicated(keep="last")].sort_index()
            shares_series[ticker] = sh
            continue
    except Exception:
        pass

    # Last resort: use fast_info (current only — constant backfill)
    try:
        n = t.fast_info.get("shares", None)
        if n:
            shares_series[ticker] = pd.Series(
                [n], index=[pd.Timestamp("2012-01-01")])
    except Exception:
        pass

# Reindex to trading dates with backfill then forward-fill
shares_df = pd.DataFrame(shares_series)
shares_df = (shares_df
             .reindex(shares_df.index.union(trading_dates))
             .sort_index()
             .bfill()       # carry earliest back to 2012
             .ffill()       # carry latest forward
             .reindex(trading_dates))

prices_aligned = prices.reindex(trading_dates)

mcap = prices_aligned * shares_df          # daily market cap (USD)
mcap_long = (mcap.stack()
               .reset_index()
               .rename(columns={"level_0": "date", "level_1": "ticker", 0: "mcap"}))
mcap_long["permno"] = mcap_long["ticker"].map(TICKER_TO_PERMNO)
mcap_long = mcap_long.dropna(subset=["permno", "mcap"])
mcap_long["permno"] = mcap_long["permno"].astype(int)

print(f"  Market-cap data: {len(mcap_long):,} firm-day obs, "
      f"{mcap_long['permno'].nunique()} firms with known cap")
print(f"  Date range: {mcap_long['date'].min().date()} to {mcap_long['date'].max().date()}")

# Avg market cap per firm over sample period (for reference)
avg_caps = (mcap_long.groupby("ticker")["mcap"].mean() / 1e9).sort_values(ascending=False)
print("\n  Average market cap ($B) by firm:")
for t_name, cap in avg_caps.items():
    print(f"    {t_name:<6} ${cap:,.0f}B")

# =============================================================================
# STEP 1C: Merge market cap into firm-day; build all three indices
# =============================================================================
print("\n" + "=" * 60)
print("STEP 1C  Build equal-weighted and cap-weighted indices")
print("=" * 60)

fd = firm_day.merge(mcap_long[["permno", "date", "mcap"]],
                    on=["permno", "date"], how="left")

# Flag which firms have cap data
fd["has_cap"] = fd["mcap"].notna()

# ── EW over all 1,500 firms ──────────────────────────────────────────────────
ew = (fd.groupby("date")[["zee_sent_pc", "zee_attn_pc"]]
        .mean()
        .rename(columns={"zee_sent_pc": "sent_ew", "zee_attn_pc": "attn_ew"}))
ew["disp_ew"] = fd.groupby("date")["zee_sent_pc"].std()

# ── EW over 8 large-cap firms only ───────────────────────────────────────────
fd8 = fd[fd["has_cap"]]
ew8 = (fd8.groupby("date")[["zee_sent_pc", "zee_attn_pc"]]
          .mean()
          .rename(columns={"zee_sent_pc": "sent_ew8", "zee_attn_pc": "attn_ew8"}))

# ── Cap-weighted over 8 large-cap firms ──────────────────────────────────────
def wavg(g, val, wt):
    w = g[wt]
    v = g[val]
    total = w.sum()
    return (v * w).sum() / total if total > 0 else np.nan

cw = (fd8.groupby("date")
         .apply(lambda g: pd.Series({
             "sent_cw": wavg(g, "zee_sent_pc", "mcap"),
             "attn_cw": wavg(g, "zee_attn_pc", "mcap"),
         })))

# Combine
mkt = (ew.join(ew8, how="outer")
          .join(cw,  how="outer")
          .reset_index()
          .sort_values("date")
          .reset_index(drop=True))

print(f"  EW (1500 firms):  {mkt['sent_ew'].notna().sum()} days")
print(f"  EW (8 large caps): {mkt['sent_ew8'].notna().sum()} days")
print(f"  CW (8 large caps): {mkt['sent_cw'].notna().sum()} days")

# Pairwise correlations between the three sentiment series
corr_cols = ["sent_ew", "sent_ew8", "sent_cw"]
print("\n  Pairwise correlations (daily sentiment):")
print(mkt[corr_cols].corr().round(3).to_string())

# =============================================================================
# STEP 2: Download S&P 500, VIX, SPY volume
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2  Download market outcomes (Yahoo Finance)")
print("=" * 60)

raw = yf.download(["^GSPC", "^VIX", "SPY"],
                  start="2012-01-01", end="2022-01-01",
                  auto_adjust=True, progress=False)
raw.index = pd.to_datetime(raw.index).tz_localize(None)

sp500_ret  = np.log(raw["Close"]["^GSPC"]).diff().rename("sp500_ret")
vix_chg    = raw["Close"]["^VIX"].diff().rename("vix_chg")
spy_vol_chg= np.log(raw["Volume"]["SPY"] + 1).diff().rename("spy_vol_chg")
vix        = raw["Close"]["^VIX"].rename("vix")

mkt_data = pd.concat([sp500_ret, vix_chg, spy_vol_chg, vix], axis=1).reset_index()
mkt_data = mkt_data.rename(columns={"Date": "date"})
mkt_data["date"] = pd.to_datetime(mkt_data["date"])

# =============================================================================
# STEP 3: FOMC dates
# =============================================================================
FOMC_DATES = pd.to_datetime([
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20",
    "2012-08-01","2012-09-13","2012-10-24","2012-12-12",
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19",
    "2013-07-31","2013-09-18","2013-10-30","2013-12-18",
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18",
    "2014-07-30","2014-09-17","2014-10-29","2014-12-17",
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17",
    "2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15",
    "2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14",
    "2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13",
    "2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19",
    "2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29",
    "2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
    "2021-07-28","2021-09-22","2021-11-03","2021-12-15",
])

# =============================================================================
# STEP 4: Merge and create leads/lags for each index variant
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4  Merge signal + market data")
print("=" * 60)

df = (mkt.merge(mkt_data, on="date", how="inner")
         .sort_values("date")
         .reset_index(drop=True))

# Leads / lags
for col in ["sp500_ret", "vix_chg", "spy_vol_chg"]:
    df[f"{col}_next"] = df[col].shift(-1)
    df[f"{col}_lag"]  = df[col].shift(1)

for sig in ["sent_ew", "attn_ew", "sent_ew8", "attn_ew8", "sent_cw", "attn_cw"]:
    df[f"{sig}_lag"] = df[sig].shift(1)

df = df.dropna(subset=["sent_ew", "sp500_ret", "vix_chg", "spy_vol_chg"])
print(f"  Final panel: {len(df)} trading days")

# =============================================================================
# FIGURE 1: Side-by-side comparison of the three sentiment indices
# =============================================================================
print("\n" + "=" * 60)
print("FIGURE 1  Comparing EW-1500, EW-8, and CW-8 sentiment indices")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

specs = [
    ("sent_ew",  "EW (all 1,500 firms)",  "steelblue"),
    ("sent_ew8", "EW (8 large caps)",     "darkorange"),
    ("sent_cw",  "CW (8 large caps)",     "darkgreen"),
]
roll = 21  # one-month rolling mean

for ax, (col, label, color) in zip(axes, specs):
    s = df[col].dropna()
    r = s.rolling(roll).mean()
    ax.plot(df["date"], df[col], alpha=0.2, color=color, lw=0.5)
    ax.plot(df["date"], df[col].rolling(roll).mean(), color=color, lw=1.8,
            label=f"{roll}-day rolling mean")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_ylabel("Sentiment (z)", fontsize=9)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    # Annotate key events
    for ev_date, ev_name in [
        ("2015-08-24", "China\nCrash"),
        ("2018-02-05", "VIX\nSquz"),
        ("2020-03-16", "COVID"),
        ("2021-01-27", "GME"),
    ]:
        ax.axvline(pd.Timestamp(ev_date), color="gray", ls=":", lw=0.9)
    ax.annotate("COVID", xy=(pd.Timestamp("2020-03-16"), ax.get_ylim()[1] * 0.85),
                fontsize=7, color="gray")

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator())

plt.suptitle("Social Sentiment: Equal-Weighted vs. Market-Cap Weighted, 2012–2021",
             fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "cw_fig1_comparison.png", dpi=150, bbox_inches="tight")
print("  Saved cw_fig1_comparison.png")
plt.close()

# =============================================================================
# FIGURE 2: Scatter — EW-1500 vs CW-8 (daily)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sub = df.dropna(subset=["sent_ew", "sent_cw", "attn_ew", "attn_cw"])

for ax, ew_col, cw_col, label, color in [
    (axes[0], "sent_ew", "sent_cw", "Sentiment", "steelblue"),
    (axes[1], "attn_ew", "attn_cw", "Attention", "coral"),
]:
    ax.scatter(sub[ew_col], sub[cw_col], alpha=0.15, s=4, color=color)
    # Add regression line
    m = np.polyfit(sub[ew_col].dropna(), sub[cw_col].dropna(), 1)
    x_line = np.linspace(sub[ew_col].min(), sub[ew_col].max(), 100)
    ax.plot(x_line, np.polyval(m, x_line), color="black", lw=1.2)
    r = sub[[ew_col, cw_col]].corr().iloc[0, 1]
    ax.set_xlabel(f"EW-1500 {label}", fontsize=10)
    ax.set_ylabel(f"CW-8 {label}", fontsize=10)
    ax.set_title(f"{label}: EW-1500 vs CW-8\nr = {r:.3f}", fontsize=11)

plt.suptitle("Correlation: Equal-Weighted (1,500 firms) vs. Cap-Weighted (8 large caps)",
             fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "cw_fig2_scatter.png", dpi=150, bbox_inches="tight")
print("  Saved cw_fig2_scatter.png")
plt.close()

# =============================================================================
# SECTION B: Predictive regressions — compare EW vs CW
# =============================================================================
print("\n" + "=" * 60)
print("SECTION B  Predictive regressions: EW-1500 vs CW-8 (signal → next-day market)")
print("=" * 60)

import statsmodels.api as sm

def nw_ols(y, X_cols, data, lags=5):
    sub = data.dropna(subset=[y] + X_cols)
    X = sm.add_constant(sub[X_cols])
    return sm.OLS(sub[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

outcomes = {
    "sp500_ret_next":   "S&P 500 Return (t+1)",
    "vix_chg_next":     "ΔVIX (t+1)",
    "spy_vol_chg_next": "ΔLog Volume (t+1)",
}
controls = ["sp500_ret_lag", "vix_chg_lag", "spy_vol_chg_lag"]

def sig(t): return "*" if abs(t) >= 1.96 else " "

print("\n── EW-1500 ─────────────────────────────────────────────────────────────────")
print(f"{'Outcome':<28} {'β_sent':>9} {'t':>6} {'β_attn':>9} {'t':>6}  {'R²':>5}")
print("-" * 68)
results_ew = {}
for dep, label in outcomes.items():
    m = nw_ols(dep, ["sent_ew", "attn_ew"] + controls, df)
    b_s, t_s = m.params["sent_ew"], m.tvalues["sent_ew"]
    b_a, t_a = m.params["attn_ew"], m.tvalues["attn_ew"]
    print(f"{label:<28} {b_s:>+9.4f} {t_s:>5.2f}{sig(t_s)} {b_a:>+9.4f} {t_a:>5.2f}{sig(t_a)}  {m.rsquared:>5.3f}")
    results_ew[dep] = m

print("\n── CW-8 ────────────────────────────────────────────────────────────────────")
print(f"{'Outcome':<28} {'β_sent':>9} {'t':>6} {'β_attn':>9} {'t':>6}  {'R²':>5}")
print("-" * 68)
results_cw = {}
for dep, label in outcomes.items():
    sub = df.dropna(subset=["sent_cw", "attn_cw"])
    m = nw_ols(dep, ["sent_cw", "attn_cw"] + controls, sub)
    b_s, t_s = m.params["sent_cw"], m.tvalues["sent_cw"]
    b_a, t_a = m.params["attn_cw"], m.tvalues["attn_cw"]
    print(f"{label:<28} {b_s:>+9.4f} {t_s:>5.2f}{sig(t_s)} {b_a:>+9.4f} {t_a:>5.2f}{sig(t_a)}  {m.rsquared:>5.3f}")
    results_cw[dep] = m

print("\n* = |t| ≥ 1.96 (NW-HAC, 5 lags). Controls: lagged return, ΔVIX, Δlog volume.")

# =============================================================================
# FIGURE 3: β-coefficient comparison bar chart (EW vs CW)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

outcome_labels = {
    "sp500_ret_next":   "S&P 500\nReturn (t+1)",
    "vix_chg_next":     "ΔVIX\n(t+1)",
    "spy_vol_chg_next": "ΔLog Volume\n(t+1)",
}

for ax, (dep, olabel) in zip(axes, outcome_labels.items()):
    betas_ew = [results_ew[dep].params["sent_ew"],
                results_ew[dep].params["attn_ew"]]
    betas_cw = [results_cw[dep].params["sent_cw"],
                results_cw[dep].params["attn_cw"]]
    ses_ew   = [results_ew[dep].bse["sent_ew"],
                results_ew[dep].bse["attn_ew"]]
    ses_cw   = [results_cw[dep].bse["sent_cw"],
                results_cw[dep].bse["attn_cw"]]

    x = np.array([0, 1])
    width = 0.35
    ax.bar(x - width/2, betas_ew, width, color="steelblue", alpha=0.75,
           label="EW-1500", edgecolor="white",
           yerr=1.96 * np.array(ses_ew), capsize=4, error_kw={"lw": 1.2})
    ax.bar(x + width/2, betas_cw, width, color="darkgreen", alpha=0.75,
           label="CW-8", edgecolor="white",
           yerr=1.96 * np.array(ses_cw), capsize=4, error_kw={"lw": 1.2})
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["Sentiment", "Attention"], fontsize=9)
    ax.set_ylabel("β (NW-HAC 95% CI)", fontsize=9)
    ax.set_title(olabel, fontsize=10)
    if ax is axes[0]:
        ax.legend(fontsize=8)

plt.suptitle("Predictive Coefficients: Equal-Weighted vs. Cap-Weighted Signal\n"
             "(next-day market outcomes)", fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "cw_fig3_beta_comparison.png", dpi=150, bbox_inches="tight")
print("\n  Saved cw_fig3_beta_comparison.png")
plt.close()

# =============================================================================
# SECTION C: Distributed-lag return predictability — EW vs CW
# =============================================================================
print("\n" + "=" * 60)
print("SECTION C  Distributed-lag predictability (1–10 days ahead)")
print("=" * 60)

LL_BEFORE = 6
LL_AFTER  = 15
REL_DAYS  = list(range(-LL_BEFORE, LL_AFTER + 1))
N_WINDOW  = len(REL_DAYS)
TERCILE   = 1/3

df_ll = df[["date","sent_ew","attn_ew","sent_cw","attn_cw","sp500_ret"]].dropna().reset_index(drop=True)

def event_cum_returns_cw(ew_col, cw_col, hi_pct, lo_pct, data):
    """
    Partition days into high/low by EW signal; compute cumulative S&P 500
    return window for each group for both EW and CW signal variants.
    Returns dict of arrays (n_events × N_WINDOW).
    """
    hi_thresh = data[ew_col].quantile(1 - hi_pct)
    lo_thresh = data[ew_col].quantile(lo_pct)
    ret = data["sp500_ret"].values
    sig = data[ew_col].values
    hi_rows, lo_rows = [], []
    for i in range(LL_BEFORE, len(data) - LL_AFTER):
        window = ret[i - LL_BEFORE : i + LL_AFTER + 1]
        if len(window) != N_WINDOW:
            continue
        cumret = np.cumsum(window)
        cumret -= cumret[0]
        if sig[i] >= hi_thresh:
            hi_rows.append(cumret)
        elif sig[i] <= lo_thresh:
            lo_rows.append(cumret)
    return np.array(hi_rows), np.array(lo_rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, ew_col, cw_col, slabel in [
    (axes[0], "sent_ew", "sent_cw", "Sentiment"),
    (axes[1], "attn_ew", "attn_cw", "Attention"),
]:
    hi_arr, lo_arr = event_cum_returns_cw(ew_col, cw_col, TERCILE, TERCILE, df_ll)

    for arr, color, grp_label in [(hi_arr, "steelblue", f"High EW (n={len(hi_arr):,})"),
                                   (lo_arr, "tomato",    f"Low EW  (n={len(lo_arr):,})")]:
        mean_cr = arr.mean(axis=0) * 100
        sem_cr  = arr.std(axis=0) / np.sqrt(len(arr)) * 100
        ax.plot(REL_DAYS, mean_cr, color=color, lw=2.0, label=grp_label)
        ax.fill_between(REL_DAYS, mean_cr - 1.96*sem_cr, mean_cr + 1.96*sem_cr,
                        alpha=0.15, color=color)

    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.axvline(0, color="black", lw=1.2, ls="-", alpha=0.4, label="Event day (t=0)")
    ax.axvline(-LL_BEFORE, color="gray", lw=1.0, ls=":", alpha=0.7,
               label=f"Baseline (t=−{LL_BEFORE})")
    ax.set_xlabel("Trading days relative to signal day", fontsize=10)
    ax.set_ylabel("Cumulative S&P 500 Return (%)", fontsize=10)
    ax.set_title(f"{slabel}: Cumulative Return\naround High vs. Low Signal Days", fontsize=11)
    ax.set_xticks([-6, -3, 0, 3, 6, 9, 12, 15])
    ax.legend(fontsize=8)

plt.suptitle("Leads-and-Lags Cumulative Return: EW-1500 Signal (High vs. Low)\n"
             "(t=−6 baseline; ±1.96 SEM shaded; top/bottom tercile)", fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "cw_fig4_leads_lags.png", dpi=150, bbox_inches="tight")
print("  Saved cw_fig4_leads_lags.png")
plt.close()

# =============================================================================
# SECTION D: FOMC event study — EW vs CW
# =============================================================================
print("\n" + "=" * 60)
print("SECTION D  FOMC event study")
print("=" * 60)

WINDOW = 15
BASELINE_DAY = -6   # cumulative return indexed to 0 at this relative day
df_idx = df.set_index("date")

event_rows_cw = []
for fomc_date in FOMC_DATES:
    cands = df_idx.index[df_idx.index >= fomc_date]
    if len(cands) == 0: continue
    ev = cands[0]
    loc = df_idx.index.get_loc(ev)
    if loc < WINDOW or loc > len(df_idx) - WINDOW - 1: continue
    win = df_idx.index[loc - WINDOW: loc + WINDOW + 1]
    w = df_idx.loc[win, ["sent_ew", "attn_ew", "sent_cw", "attn_cw",
                          "sp500_ret", "vix_chg"]].copy()
    w["rel_day"] = range(-WINDOW, WINDOW + 1)
    w["fomc_date"] = fomc_date
    event_rows_cw.append(w)

ep = pd.concat(event_rows_cw).reset_index()
avg = ep.groupby("rel_day")[["sent_ew", "attn_ew", "sent_cw", "attn_cw",
                               "sp500_ret", "vix_chg"]].mean()
baseline = avg.loc[-10:-6].mean()
avg_dm = avg - baseline

# ── Cumulative S&P 500 return indexed to BASELINE_DAY = 0 ────────────────────
cum_rows_cw = []
for fomc_date, grp in ep.groupby("fomc_date"):
    g = grp[grp["rel_day"] >= BASELINE_DAY].sort_values("rel_day").copy()
    if BASELINE_DAY not in g["rel_day"].values:
        continue
    g["cumret"] = g["sp500_ret"].cumsum()
    g["cumret"] -= g.loc[g["rel_day"] == BASELINE_DAY, "cumret"].values[0]
    cum_rows_cw.append(g[["rel_day", "cumret"]])

cum_panel_cw = pd.concat(cum_rows_cw)
cum_agg_cw = cum_panel_cw.groupby("rel_day")["cumret"].agg(
    mean="mean",
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75),
).loc[BASELINE_DAY:15]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Signal panels: EW vs CW bar charts
for ax, col_ew, col_cw, label in [
    (axes[0,0], "sent_ew", "sent_cw", "Sentiment"),
    (axes[0,1], "attn_ew", "attn_cw", "Attention"),
]:
    d = avg_dm.loc[-WINDOW:15]
    ax.bar(d.index - 0.2, d[col_ew], 0.35,
           color="steelblue", alpha=0.7, label="EW-1500", edgecolor="white")
    ax.bar(d.index + 0.2, d[col_cw], 0.35,
           color="darkgreen",  alpha=0.7, label="CW-8",    edgecolor="white")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.axvline(0, color="red", lw=1.2, alpha=0.5)
    ax.axvline(BASELINE_DAY, color="gray", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("Trading days relative to FOMC", fontsize=9)
    ax.set_title(f"{label} (de-meaned, days −10 to −6)", fontsize=10)
    ax.set_xticks([-15, -10, -6, 0, 5, 10, 15])

# Cumulative S&P 500 return (t-6 to t+15)
ax = axes[1, 0]
ax.plot(cum_agg_cw.index, cum_agg_cw["mean"] * 100,
        color="darkgreen", lw=2.0, label="Mean cumulative return")
ax.fill_between(cum_agg_cw.index,
                cum_agg_cw["q25"] * 100, cum_agg_cw["q75"] * 100,
                alpha=0.2, color="darkgreen", label="IQR across events")
ax.axhline(0, color="black", lw=0.7, ls="--")
ax.axvline(0, color="red", lw=1.2, alpha=0.5, label="FOMC day")
ax.axvline(BASELINE_DAY, color="gray", lw=1.0, ls="--", alpha=0.7,
           label=f"Baseline (day {BASELINE_DAY})")
ax.set_xlabel("Trading days relative to FOMC", fontsize=9)
ax.set_ylabel("Cumulative S&P 500 Return (%)", fontsize=9)
ax.set_title(f"Cumulative S&P 500 Return\n(indexed to 0 at day {BASELINE_DAY})", fontsize=10)
ax.set_xticks([BASELINE_DAY, -3, 0, 3, 6, 9, 12, 15])
ax.legend(fontsize=7, loc="upper left")

# ΔVIX panel
ax = axes[1, 1]
d = avg_dm.loc[-WINDOW:15]
ax.bar(d.index, d["vix_chg"], 0.7, color="tomato", alpha=0.7, edgecolor="white")
ax.axhline(0, color="black", lw=0.7, ls="--")
ax.axvline(0, color="red", lw=1.2, alpha=0.5)
ax.axvline(BASELINE_DAY, color="gray", lw=1.0, ls="--", alpha=0.7)
ax.set_xlabel("Trading days relative to FOMC", fontsize=9)
ax.set_title("ΔVIX (de-meaned, days −10 to −6)", fontsize=10)
ax.set_xticks([-15, -10, -6, 0, 5, 10, 15])

plt.suptitle(f"FOMC Event Study: EW vs. CW Signal\n"
             f"{len(FOMC_DATES)} FOMC dates, 2012–2021, ±{WINDOW}-day window",
             fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "cw_fig5_fomc.png", dpi=150, bbox_inches="tight")
print("  Saved cw_fig5_fomc.png")
plt.close()

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# correlations
r_sent = df["sent_ew"].corr(df["sent_cw"])
r_attn = df["attn_ew"].corr(df["attn_cw"])

m_ew_ret = results_ew["sp500_ret_next"]
m_cw_ret = results_cw["sp500_ret_next"]
m_ew_vix = results_ew["vix_chg_next"]
m_cw_vix = results_cw["vix_chg_next"]

print(f"""
EW-1500 vs. CW-8 daily signal correlation:
  Sentiment:  r = {r_sent:.3f}
  Attention:  r = {r_attn:.3f}

Predictability of next-day S&P 500 return:
  EW sentiment: β = {m_ew_ret.params['sent_ew']:+.4f}  t = {m_ew_ret.tvalues['sent_ew']:+.2f}{sig(m_ew_ret.tvalues['sent_ew'])}
  CW sentiment: β = {m_cw_ret.params['sent_cw']:+.4f}  t = {m_cw_ret.tvalues['sent_cw']:+.2f}{sig(m_cw_ret.tvalues['sent_cw'])}

Predictability of next-day ΔVIX:
  EW sentiment: β = {m_ew_vix.params['sent_ew']:+.4f}  t = {m_ew_vix.tvalues['sent_ew']:+.2f}{sig(m_ew_vix.tvalues['sent_ew'])}
  CW sentiment: β = {m_cw_vix.params['sent_cw']:+.4f}  t = {m_cw_vix.tvalues['sent_cw']:+.2f}{sig(m_cw_vix.tvalues['sent_cw'])}

Key question: do the results change when you weight by size?
See cw_fig3_beta_comparison.png for the full comparison.

Figures saved to: figures_cw/
  cw_fig1_comparison.png      — EW vs CW time series
  cw_fig2_scatter.png         — EW-1500 vs CW-8 daily scatter
  cw_fig3_beta_comparison.png — β coefficients with 95% CI
  cw_fig4_distributed_lag.png — multi-horizon return predictability
  cw_fig5_fomc.png            — FOMC event study
""")
