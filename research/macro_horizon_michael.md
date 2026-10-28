# At What Horizon Does Macro Data Beat VIX?
**Michael | Trading @ GT**

## Signal

Two weekly FRED series, both back to the 1970s:

- **`NFCI`** (Chicago Fed financial conditions): composite of 105 credit, leverage, and risk indicators. Positive = tighter than average.
- **`ICSA`** (initial jobless claims): an early sign of a weakening economy.

Features: NFCI level and 13-week change; log claims vs. their 1-year average, and 13-week change. Each value enters only on its release date. ALFRED first-release vintages for NFCI exist only from 2011, so post-2011 results are the cleanest test.

**Why longer horizons?** Volatility mean-reverts toward a long-run level $\bar\sigma^2$:

```math
\mathbb{E}_t\left[\sigma^2_{t+h}\right] = \bar\sigma^2 + \phi^h\left(\sigma^2_t - \bar\sigma^2\right)
```

With $\phi \approx 0.98$ (daily), today's volatility keeps weight $0.98^{21} \approx 0.65$ at one month but only $0.98^{126} \approx 0.08$ at six months. Far out, the forecast is mostly $\bar\sigma^2$. **Hypothesis:** $\bar\sigma^2$ moves with the economy, so macro data should matter most where VIX matters least.

## Model

Target: log realized S&P 500 volatility over the next $h$ days, for $h \in \{21, 63, 126, 252\}$:

```math
y_{t,h} = \log \sqrt{\tfrac{252}{h}\textstyle\sum_{i=1}^{h} r_{t+i}^2}
```

One linear regression per horizon:

```math
y_{t,h} = \beta_0 + \beta_1 \log RV_{t-21,t} + \beta_2 \log RV_{t-252,t} + \beta_3 \log VIX_t + \gamma^\top x_t + \epsilon_{t,h}
```

$x_t$ holds the macro features. The model stays linear on purpose, so any gain comes from the data, not model flexibility.

## Test

- **Baselines:** GARCH(1,1), and the regression without macro ($\gamma = 0$).
- **Setup:** 1990–present, expanding window, monthly forecasts (daily forecasts with long targets overlap too much).
- **Metric:** out-of-sample $R^2$ of the macro model vs. the no-macro model:

```math
R^2_{OOS}(h) = 1 - \frac{\sum_t (y_{t,h} - \hat y^{\text{macro}}_{t,h})^2}{\sum_t (y_{t,h} - \hat y^{\text{no macro}}_{t,h})^2}
```

Significance via the Clark–West test with Newey–West errors (lag $h$).

**Success:** $R^2_{OOS} \approx 0$ at 1 month, rising clearly above 0 at 6–12 months. One chart of $R^2_{OOS}$ vs. horizon tells the team where macro signals are worth building around.

## Why It Might Fail

- **Few cycles:** only ~4 recessions since 1990, so long-horizon results rest on a handful of events. Paye (2012) found macro's power concentrates near recession onsets and rarely beats simple benchmarks out of sample.
- **Sudden shocks:** COVID hit volatility *before* claims spiked. Slow data can't see exogenous events.
- **VIX overlap:** VIX is one of NFCI's inputs. Check with claims alone.

## References

[Two Sigma (2021)](https://www.twosigma.com/wp-content/uploads/2021/10/Machine-Learning-Approach-to-Regime-Modeling_.pdf) · [Brunnermeier & Pedersen (2009)](https://www.nber.org/papers/w12939) · [Moreira & Muir (2017)](https://www.nber.org/papers/w22208) · Engle, Ghysels & Sohn (2013), *REStat* · Paye (2012), *JFE* · Clark & West (2007), *J. Econometrics* · FRED: [NFCI](https://fred.stlouisfed.org/series/NFCI), [ICSA](https://fred.stlouisfed.org/series/ICSA)
