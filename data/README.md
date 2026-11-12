Datasets.
## MOVE
Use the unadjusted daily close for Yahoo Finance's ^MOVE. The development
request starts in 2003 and ends in 2017. ICE bases the end-of-day value on
mid-implied volatilities as of 3 p.m. ET. Yahoo's historical date labels do
not establish when its daily closes became available, so use a conservative
one-NYSE-session release lag for the 4 p.m. ET forecast cutoff.
ICE publication announcement: https://ir.theice.com/press/news-details/2019/Intercontinental-Exchange-Completes-Acquisition-of-the-Merrill-Lynch-Option-Volatility-Estimate-MOVE-Index-from-Bank-of-America-Merrill-Lynch/default.aspx

Align closes to NYSE session dates, discarding off-calendar observations.
After applying the release lag, carry the last available level forward for
at most one missing session. Leave leading gaps and the rest of longer gaps
missing. Compute the 20-session point change from the aligned series. For the
z-score, wait for a full 252-session history and require at least 245 available
levels within it. Leave zero-variance windows missing.
