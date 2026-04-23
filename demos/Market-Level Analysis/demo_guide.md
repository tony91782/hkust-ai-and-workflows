# Market-Level Social Signal — In-Class Demo Guide

**Course:** HKUST PhD — AI Workflows and Publishing in Finance  
**Block:** 2.1 — From Empty Folder to Research Result  
**Data:** Cookson, Lu, Mullins & Niessner (2024), *The Social Signal*, JFE  
**Source:** https://data.mendeley.com/datasets/xffyybvw4j/1

---

## What this demo illustrates

Starting from a single Stata file (`social_signal_index.dta`, 821,534 firm-day
observations), a sequence of four plain-English prompts produces a complete
market-level analysis: time-series properties, predictive regressions,
a leads-and-lags event study, a FOMC event study, and a cap-weighted
robustness check. The full session took under 15 minutes.

**Key pedagogical point:** the researcher supplied the economic questions;
Claude Code supplied the mechanics. Every substantive choice — what to
aggregate, how to weight, what the benchmark date should be, whether to
show leads and lags — came from the researcher.

---

## The prompt sequence

### Prompt 1 — Build the market-level index and explore predictive properties

> *"Can you do a demo of how you would produce an aggregate index and explore
> its market-level predictive properties for trading, volatility, and returns?
> Also, consider the reverse relation — how do recent trading, vol and returns
> predict this aggregate index? Does this have anything to do with market news
> events like FOMC dates? Make a separate sub-demo folder with this
> market-level analysis."*

**Produced:** `market_analysis.py` — a self-contained script that:
- Aggregates 821,534 firm-day observations to a daily equal-weighted
  market sentiment and attention index
- Downloads S&P 500, VIX, and SPY volume from Yahoo Finance
- Runs forward regressions (signal → next-day returns, VIX, volume)
  with Newey-West HAC standard errors
- Runs reverse regressions (market outcomes → signal)
- Builds an FOMC event study (±10-day window around Fed announcement days)
- Produces five publication-style figures (fig1–fig5)

**Key findings at this step:**
- High aggregate sentiment predicts higher next-day VIX (β = +0.46, t = 2.73)
  and higher trading volume — consistent with overconfidence / disagreement
- Rising VIX dampens sentiment and raises attention within one trading day
- Social signal shifts systematically in the window around FOMC announcements

---

### Prompt 2 — Redo with market-cap weights

> *"Can you aggregate in a market value way and redo the analysis?"*

**Produced:** `market_analysis_capweighted.py` — repeats the full analysis
using market-capitalisation weights (daily price × shares outstanding from
yfinance, backfilled for the pre-2015 period where filings are sparse).

**Methodology note shown in class:** In a production setting you would use
CRSP daily market cap (prc × shrout). Here we approximate using Yahoo Finance
data for 8 large-cap firms, producing a large-cap-tilted index. Three variants
are built and compared:

| Index | Coverage | Weighting |
|---|---|---|
| EW-1500 | All 1,500 firms | Equal weight |
| EW-8 | 8 large caps only | Equal weight |
| CW-8 | 8 large caps only | Market-cap weight |

**Key finding at this step:** EW-1500 and CW-8 sentiment are only weakly
correlated (r = 0.39), while attention is more aligned (r = 0.83). Cap-weighting
sharpens the return predictability signal (CW β = −0.0007, t = −2.13*) even
as the coefficient magnitude shrinks — suggesting large-cap sentiment is a
cleaner risk signal than breadth-weighted sentiment.

---

### Prompt 3 — Cumulative return plot, leads and lags

> *"Can you show the return plots as cumulative plots from a benchmark return
> date of t−6 through t+15?"*

> *(Correction)* *"I meant to do this in the distributed lag plot. Could be a
> leads and lags plot."*

**Produced:** Replaced the regression-coefficient distributed-lag chart with
an event-study style leads-and-lags cumulative return plot:
- **Event** = day when the market signal is in the top vs. bottom tercile
- **Baseline** = t−6 (cumulative return indexed to 0)
- **Window** = t−6 through t+15 (~800 events per group)
- **Bands** = ±1.96 SEM across events

This visualises the pre-event return trajectory (potential pre-trend /
momentum contamination) and the post-event predictive path together in one
panel — the standard leads-and-lags format used in DiD and event studies.

**FOMC event study** was simultaneously updated to WINDOW = 15 with the
same cumulative return format for the S&P 500 return panel.

---

### Prompt 4 — Package the demo

> *"Can you package up this illustration of the market level signals, along
> with the prompts I used for the in-class illustration?"*

**Produced:** This file.

---

## How to run

```bash
cd "demos/Market-Level Analysis"

# Equal-weighted baseline analysis
python3 market_analysis.py

# Cap-weighted comparison
python3 market_analysis_capweighted.py
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `statsmodels`, `yfinance`  
**Data:** `../Social Signal Demo/social_signal_index.dta` (must be present)  
**Output:** figures saved to `figures/` and `figures_cw/`

---

## Output figures

### Equal-weighted (`figures/`)

| File | What it shows |
|---|---|
| `fig1_market_signal_timeseries.png` | Daily EW sentiment, attention, and dispersion 2012–2021 with annotated events (GME, COVID, VIX squeeze, China crash) |
| `fig2_cross_correlations.png` | Signal ↔ market outcomes at lags −10 to +10; dark bar = signal predicts tomorrow |
| `fig3_leads_lags.png` | **Leads-and-lags cumulative return** around high vs. low signal days (t−6 baseline → t+15); top/bottom tercile |
| `fig4_fomc_event_study.png` | Signal and cumulative S&P 500 return in ±15-day window around FOMC announcements |
| `fig5_signal_vix.png` | Signal vs. VIX level (scatter + binned means) and rolling 63-day correlation |

### Cap-weighted comparison (`figures_cw/`)

| File | What it shows |
|---|---|
| `cw_fig1_comparison.png` | EW-1500 vs. EW-8 vs. CW-8 sentiment time series |
| `cw_fig2_scatter.png` | Daily scatter: EW-1500 vs. CW-8 sentiment and attention |
| `cw_fig3_beta_comparison.png` | Regression β with 95% CI: EW-1500 vs. CW-8 side by side for all three outcomes |
| `cw_fig4_leads_lags.png` | Leads-and-lags cumulative return, EW signal partition |
| `cw_fig5_fomc.png` | FOMC event study: EW vs. CW signal panels + cumulative return |

---

## Discussion questions for class

1. **The aggregation question:** Does the signal look different when you weight
   by market cap vs. equal weight? What does it tell you that the correlation
   between EW-1500 and CW-8 sentiment is only 0.39?

2. **Pre-trends:** Look at the leads-and-lags plot. Do returns already differ
   before the signal day (t < 0)? If so, what are the interpretations —
   momentum, endogeneity, or something else?

3. **The reverse regression:** We showed that prior VIX decreases sentiment
   and prior volume increases attention. Does that make the forward-predictive
   results harder or easier to interpret?

4. **The FOMC study:** The signal shifts in the window around Fed announcements.
   Does that mean the signal is informationally driven by macro news, or is
   it consistent with investors using social media to process the news?

5. **Limitations of this demo:** We used only 8 firms for market-cap weights
   (no CRSP access in the demo environment) and the event windows overlap
   (SEM bands are too tight). How would you fix these for a real paper?

---

## What the researcher did vs. what Claude did

| Researcher | Claude Code |
|---|---|
| Chose the economic question (market-level predictability) | Wrote all Python code |
| Specified cap-weighting as a robustness check | Fetched yfinance data, built shares × price market cap |
| Identified the FOMC event study as the right design | Hardcoded 81 FOMC dates, built event panel |
| Corrected "cumulative" → "leads and lags" framing | Rewrote distributed-lag plot as event-study |
| Decided the benchmark date (t−6) and window (t+15) | Implemented the cumulative indexing |
| Evaluated whether results make economic sense | Produced all figures and regression tables |

The bottleneck was always the researcher's judgment — what question to ask,
what specification to use, whether the output is economically sensible.
The code was never the bottleneck.
