# Forecasting Sovereign Bond Spread Widening from IMF Fundamentals and Regional Spillovers

*Trading@GT | Research proposal | October 2026*

**Hypothesis.** Deteriorating macroeconomic fundamentals may predict increases in emerging-market **USD sovereign bond spreads** before the full repricing occurs. This builds on my IMF sovereign-vulnerability project, which combined macro-fiscal indicators, stress scenarios, and probability-weighted risk scores. The new question is whether those indicators offer **out-of-sample predictive information**, rather than simply explaining existing spreads.

**Regional spillover hypothesis.** Study neighboring countries together—for example, the Balkans. Test whether rising debt, larger budget deficits, weaker growth, or widening bond spreads in neighboring countries predict a later spread increase in a country whose bonds have not yet reacted. If that delayed reaction is predictable, we could trade ahead of it.

## Signal: Country-level macro and market data

**Indicators to track in each country and its neighbors:**

- **Government debt (% of GDP):** Is the debt burden rising?
- **Fiscal balance (% of GDP):** Is the budget deficit getting larger?
- **Real GDP growth (%):** Is economic growth slowing?
- **Consumer price inflation (%):** Is inflation accelerating?
- **Current account balance (% of GDP):** Is the external deficit widening?
- **Gross government financing needs (% of GDP):** Is the amount the government must borrow to cover deficits and maturing debt increasing?
- **External debt (% of GDP):** Is debt owed to foreign creditors rising?
- **Sovereign bond spread (basis points):** Have neighboring countries' spreads already widened over the past one or three months while the target country's spread stayed flat?

These are candidate warning signals; test which ones actually predict later spread moves. Use both the latest available levels and changes since the previous release. For each indicator, start with a simple average across the target country's neighbors, excluding the target itself.

**Data sources and controls:**

- **IMF:** Use historical versions of the [World Economic Outlook](https://data.imf.org/Datasets/WEO) for debt, fiscal balance, growth, inflation, and current account data. Add financing needs and external debt from IMF *Article IV* reports where consistently available.
- **Sovereign spreads:** Start with Brazil, Mexico, Colombia, Chile, and Peru. The [Central Reserve Bank of Peru](https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/tasas-de-interes-internacionales) publishes daily country-level **EMBIG spreads (basis points)** with an [API](https://estadisticas.bcrp.gob.pe/estadisticas/series/ayuda/api). Verify continuity and redistribution permissions; underlying data are attributed to JPMorgan/Reuters.
- **Balkans extension:** Verify coverage of comparable USD sovereign bonds. If using euro bonds, analyze them separately with a consistent euro benchmark. Define the neighboring-country groups before testing.
- **Global controls:** [VIX](https://fred.stlouisfed.org/series/VIXCLS), [10-year US Treasury yield](https://fred.stlouisfed.org/series/DGS10), and regional EMBIG spreads, to separate country-specific deterioration from global risk-off moves.

Use macro **levels and changes**, with each observation available only after its actual publication date. Predict the spread change over the following ~63 trading days:

\[
\Delta s_{i,t\rightarrow t+63}=s_{i,t+63}-s_{i,t}.
\]

## Model: Macro-informed sovereign risk

Start with an interpretable **country fixed-effects panel regression**:

\[
\widehat{\Delta s}_{i,t\rightarrow t+63}=\alpha_i+\beta^\top X^{IMF}_{i,t}+\theta^\top N_{i,t}+\gamma^\top G_t+\delta s_{i,t}.
\]

Here, \(X^{IMF}\) represents country fundamentals, \(N_{i,t}\) neighboring-country signals available at the forecast date, and \(G_t\) global risk factors. The neighbor terms test whether regional developments lead subsequent changes in the target country’s spread, after accounting for its own fundamentals and global conditions. Next, test whether adding a **probability-weighted sovereign vulnerability score**, inspired by my IMF scenario framework, improves forecasts. An optional two-state **HMM** on global market factors could test whether fundamentals matter more during stressed regimes; it should only be kept if it improves out-of-sample results.

## Test: Do IMF data and regional signals add predictive value?

Make forecasts **monthly** using expanding-window training and only information known at each date. Compare **(M0)** no change in spreads, **(M1)** lagged spreads + global controls, **(M2)** M1 + IMF macro variables, **(M3)** M2 + neighboring-country signals, and **(M4)** M3 + vulnerability score/HMM. Evaluate **MAE/RMSE (basis points)**, spread-widening **directional accuracy**, and cross-country **ranking accuracy**; separately test large widening episodes. Account for overlapping three-month forecast errors and correlated errors across countries. Compare M3 against M2 to isolate the added value of neighboring-country information, including episodes when neighbors have deteriorated or repriced but the target country has not. Test whether these signals predict later adjustment rather than merely reflecting simultaneous regional moves.

**Success:** IMF features consistently outperform market-only baselines out of sample, and neighboring-country signals improve on the own-country model. For signals that pass these tests, paper-test relative-value positioning: underweight countries forecast to widen, overweight those forecast to tighten, controlling for duration, liquidity, and trading costs. For the regional hypothesis, paper-test reducing exposure to or shorting bonds of a country forecast to widen after its neighbors, potentially against a comparable regional bond position. Test the reverse for predicted tightening. Only treat the apparent delay as tradeable if subsequent price moves support the forecast and returns survive hedging, financing, and transaction costs.

## Why it might fail

Markets may already price the macro fundamentals. US rates, global liquidity, and political shocks may dominate. IMF releases are infrequent and revised, making historical vintages essential to avoid look-ahead bias. Neighboring economies may differ enough that an unchanged spread is justified, and apparent lead–lag relationships may reflect stale or illiquid bond prices rather than an executable opportunity. Regional signals may also duplicate global risk factors. EMBIG indices are not directly tradeable. A negative out-of-sample result is a valid research finding.

**Data setup:** `pip install yfinance fredapi pandas`; verify `^GSPC` and `^VIX` downloads since 2000, then test the IMF and BCRP data pipelines.

**Reading:** [Two Sigma (2021)](https://www.twosigma.com/wp-content/uploads/2021/10/Machine-Learning-Approach-to-Regime-Modeling_.pdf) · [Brunnermeier & Pedersen](https://www.nber.org/papers/w12939) · [Moreira & Muir](https://www.nber.org/papers/w22208).
