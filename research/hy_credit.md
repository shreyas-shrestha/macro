# High-yield credit spreads (TRACE via WRDS)

Issue: #24

## Question

Do granular high-yield corporate bond spreads contain information about future S&P 500 realized variance beyond recent realized variance, VIX, and the simpler Baa credit-spread signal in #11?

The main development sample is pre-2018. The 2018-present holdout is not inspected until the final frozen run.

## Data currently available

Georgia Tech WRDS access includes TRACE BTDS/TRACE Enhanced and the TRACE corporate master file. Mergent FISD is not subscribed. Legacy Compustat North America Ratings is available through February 2017 and contains the historical S&P long-term issuer rating (`splticrm`) plus industry identifiers.

Raw WRDS exports remain in `data/cache/` and are never committed.

## Initial construction

1. Clean TRACE corporate-bond transactions.
2. Collapse trades to one observation per CUSIP/day using median reported yield and price.
3. Join bond characteristics from the TRACE master file.
4. Join each issuer to the contemporaneous S&P long-term issuer rating.
5. Compute a corporate credit spread relative to a maturity-matched Treasury yield.
6. Collapse multiple bonds to one issuer/day median so issuers with many outstanding bonds do not receive more cross-sectional weight.
7. Aggregate issuer spreads into daily high-yield and investment-grade medians.

Initial features:

- HY spread level
- 20-trading-day change in HY spread
- point-in-time expanding z-score of HY spread
- HY minus IG spread
- 20-trading-day change in HY minus IG spread

A later extension will measure cross-sectional residual stress after controlling for rating, maturity and sector.

## Pre-registered threshold

At h = 21, the TRACE high-yield signal must deliver at least 5% lower QLIKE than the VIX + RV no-signal model with Clark-West p < 0.10. The incremental result with #11's Baa-spread features included will also be reported to determine whether TRACE adds information beyond the simpler aggregate credit signal.

## Open implementation questions

- Exact TRACE cancellation/correction filters after inspecting the observed status codes.
- Best issuer crosswalk between TRACE CUSIPs/tickers and Compustat GVKEY.
- Treasury curve interpolation convention.
- Whether the legacy S&P rating coverage is large enough for a stable daily panel.
- Whether TRACE `grade` should be used only as a robustness classification or to extend coverage beyond the S&P-rated sample.
