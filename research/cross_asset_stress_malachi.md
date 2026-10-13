# Volatility

## Signal

I'd like to test whether stress starts showing up in credit and rates markets before it is fully reflected in equity volatility. I would start with two main signals: the BAA10Y credit spread, $C_t$, from FRED and the MOVE index, $M_t$, from Yahoo Finance. BAA10Y is the spread between Moody's Baa corporate bond yield and the 10-year Treasury yield, so a wider spread generally means investors are demanding more compensation for holding corporate debt. MOVE measures implied volatility in the Treasury market, so it gives a similar measure of uncertainty/stress on the rates side.

I would look at both the level of each signal and how much it has changed recently:

```math
\Delta_{20} C_t = C_t - C_{t-20}
```

```math
\Delta_{20} M_t = M_t - M_{t-20}
```

The reason for looking at changes too is that a market that has already been stressed for a few months may be different from one where conditions are suddenly getting worse. For an early warning signal, the second case is probably more interesting.

## Model

One model I would test is a two-state Gaussian Hidden Markov Model. The inputs would be standardized versions of the credit spread, MOVE, and their recent changes:

```math
x_t =
\begin{bmatrix}
z(C_t) \\
z(M_t) \\
z(\Delta_{20} C_t) \\
z(\Delta_{20} M_t)
\end{bmatrix}
```

The HMM would try to identify two hidden states from these variables. I would expect them to roughly correspond to a lower-stress and higher-stress environment, although the states would be determined by the model rather than labeled beforehand. The main output I would use is the probability that the market is currently in the stressed state:

```math
p_t = P(S_t = \text{stress} \mid x_1,\ldots,x_t)
```

I would use filtered probabilities here so that the model only uses information that would actually have been available at time $t$.

The thing I want to predict is S&P 500 realized volatility over the next 20 trading days:

```math
RV_{t,t+20}
=
\sqrt{
\frac{252}{20}
\sum_{i=1}^{20} r_{t+i}^{2}
}
```

A simple first forecasting model could be:

```math
RV_{t,t+20}
=
\beta_0
+
\beta_1 RV_{t-19,t}
+
\beta_2 p_t
+
\epsilon_t
```

Recent volatility should already be useful because volatility tends to cluster, so the main thing I care about is whether $p_t$ gives us anything extra after controlling for that. I would also try the same general test without the HMM, using the MOVE and credit variables directly. That would give us a way to tell whether the regime model itself is useful or whether most of the signal is already in the raw variables.

## Test

I would test this out of sample using an expanding window. For example, fit everything using data through year $T$, forecast during year $T+1$, then add that year to the training data and continue forward. The standardization and HMM would also have to be refit using only past data so we are not accidentally giving the model future information.

The main comparisons I would start with are:

```math
M_0:
\qquad
RV_{t,t+20}
\sim
RV_{t-19,t}
```

```math
M_1:
\qquad
\mathrm{GARCH}(1,1)
```

```math
M_2:
\qquad
RV_{t,t+20}
\sim
RV_{t-19,t} + p_t
```

The first model is basically asking how much we can do with recent volatility alone. GARCH gives a stronger standard volatility baseline. The third model then tells us whether adding the credit/rates stress state actually improves anything.

I would also want to do a harder comparison against VIX. Since VIX already reflects the equity options market's expectation of future volatility, this is a good test of whether rates and credit are giving us information that equities have not already priced in:

```math
M_3:
\qquad
RV_{t,t+20}
\sim
VIX_t
```

```math
M_4:
\qquad
RV_{t,t+20}
\sim
VIX_t + p_t
```

For basic forecast accuracy I would start with MAE and RMSE:

```math
MAE
=
\frac{1}{N}
\sum_{t=1}^{N}
\left|
\widehat{RV}_t - RV_t
\right|
```

```math
RMSE
=
\sqrt{
\frac{1}{N}
\sum_{t=1}^{N}
\left(
\widehat{RV}_t - RV_t
\right)^2
}
```

I would also look at the highest-volatility periods separately. It is possible that the signal does not improve average forecasts by much but is still useful specifically before large volatility increases, which would probably matter more from a risk-management perspective.

If the volatility forecast ends up being useful, a later extension would be to turn it into a volatility-targeting strategy:

```math
w_t
=
\frac{\sigma^*}
{\widehat{\sigma}_{t,t+20}}
```

with some kind of position limit in practice. The basic idea would just be to cut exposure when the model expects volatility to be higher.

## Why It Might Fail

The biggest issue is that MOVE and credit spreads may not actually lead equity volatility. A common shock could just cause all three markets to react at roughly the same time:

```math
MOVE \uparrow,
\qquad
Credit\ Spread \uparrow,
\qquad
Equity\ Volatility \uparrow
```

If that is what is happening, the model could do a good job identifying that markets are currently stressed without actually helping us forecast what happens next.

Another possibility is that VIX already absorbs this information quickly enough that the credit/rates signal does not add much once VIX is included. There is also no guarantee that the relationship stays the same across different crises. 2008 was very heavily tied to credit and funding stress, while other volatility events can start for completely different reasons. Because of that, an HMM could fit past crises pretty well and still not generalize out of sample.

## References

[1] Two Sigma. "A Machine Learning Approach to Regime Modeling." 2021.

[2] Brunnermeier, M. K. and Pedersen, L. H. "Market Liquidity and Funding Liquidity." Review of Financial Studies, 2009.

[3] Moreira, A. and Muir, T. "Volatility-Managed Portfolios." Journal of Finance, 2017.

[4] Federal Reserve Bank of St. Louis, FRED. BAA10Y: Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity.
