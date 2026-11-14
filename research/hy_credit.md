# High-yield credit spreads (TRACE via WRDS)

Issue: #24

## Question

Do granular high-yield corporate bond spreads contain information about future S&P 500 realized variance beyond recent realized variance, VIX, and the simpler Baa credit-spread signal in #11?

The main development sample is pre-2018. The 2018-present holdout is not inspected until the final frozen run.

## Data sources (WRDS discovery, 2026-11-07)

| Purpose | WRDS table | Notes |
|---|---|---|
| Trades | `trace_enhanced.trace_enhanced` | Enhanced historic file: all reported trades with linking fields (`msg_seq_nb`, `orig_msg_seq_nb`, `trd_rpt_dt`, `pr_trd_dt`), uncapped `entrd_vol_qt`, `rpt_side_cd`, `cntra_mp_id`, `dissem_fl`, `trdg_mkt_cd`, `trd_mod_3/4`. No `chng_cd`/`frmt_cd`: those belong to `trace_standard.trace` (disseminated BTDS), which we do not use. |
| Bond master | `trace_enhanced.camasterfile` | Versions per CUSIP with `stdt`/`enddt`; `sym_cd` (not `bond_sym_id`); `cpn_rt` is text. |
| Ratings | `comp.adsprate` | `gvkey`, `datadate`, `splticrm`, `spsdrm`, `spsticrm` only; identifiers and sector come from `comp.security`, `comp.names`, `comp.company`. Legacy file; the last `datadate` is 2017-02-28. |
| Issuer link (used) | `comp.security` | issuer CUSIP6 -> `gvkey` (`comp_cusip6`). |
| Issuer link (used) | `wrdsapps_link_crsp_bond.bondcrsp_link` -> `crsp.stocknames` -> `comp.security` | bond CUSIP -> `permno` (`link_startdt`/`link_enddt`) -> historical CRSP CUSIP6 (`namedt`/`nameenddt`) -> `gvkey` (`crsp_ncusip6`). |
| Issuer link (no access) | `contrib_bond_firm_link.fang_link`, `crsp.ccmxpf_lnkhist` | Permission denied for this account; code paths exist but are not used. |
| Benchmark (no access) | `wrdsapps_bondret.bondret` | WRDS Bond Returns. Permission denied, so no benchmark comparison was made. |
| Treasury curve | FRED `DGS1MO` ... `DGS30` (H.15) | Public CSV endpoint, cached in `data/cache/fred/`, through 2017-12-29. |

Mergent FISD is not subscribed. Raw WRDS rows stay in `data/cache/` (git-ignored) and are never committed; `results/hy_credit/` holds aggregated diagnostics only.

## Construction (code: `macro/data/hy_dataset.py`, runner `macro/data/hy_runs.py`)

Fixed before looking at any predictive result:

1. **Sample start 2006-01-09.** FINRA disseminated about 99% of trades from Phase III (2005-02-07), but large trades in infrequently traded high-yield bonds stayed on delayed dissemination until 2006-01-09, from when all public TRACE-eligible transactions were disseminated immediately (FINRA TRACE Fact Books). Starting then keeps every HY input public when it printed.
2. **Lifecycle cleaning** (`clean_trace_lifecycle`): pre-2012 same-day C/W, post-2012 prior-reference links, then As-Of reversals (rules below).
3. **Condition filters**: when-issued, special price, weighted-average price (`sale_cndtn_cd` W pre-2012, `trd_mod_4` W post-2012), primary-market P1, inter-dealer buy side.
4. **Disseminated trades only** (`dissem_fl` Y), so the inputs are what the market could see.
5. **Bond-days**: median signed yield and median price per CUSIP and execution date; trade counts kept.
6. **Bond universe** from FINRA master-file codes (TRACE Corporate and Agency master file spec, Appendix 2): `sub_prdct_type` CORP; `cnvrb_fl` not Y; `cpn_type_cd` FXPV (plain vanilla fixed); `ind_144a` not Y. The master version in force on the trade date is chosen by `stdt` <= date <= `enddt` (missing end = open); duplicates are never resolved by picking one.
7. **Maturity and spread screens** (Gilchrist and Zakrajsek 2012): remaining maturity 1-30 years; spread 5-3,500 bp.
8. **Treasury spread**: `spread_bp = 100 * (yield_pct - treasury_yield_pct)`, where the Treasury yield is linearly interpolated in maturity on the same day's H.15 curve, flat beyond the end nodes. `DGS30` is ignored inside its documented discontinuation (2002-02-18 to 2006-02-08).
9. **Issuer (GVKEY)**: consensus of the exact-identifier methods available to us (`comp_cusip6` and `crsp_ncusip6`; both were used in the development build). A bond-day gets a GVKEY only if every method that finds a candidate agrees on exactly one; disagreement leaves it unlinked and counted. Links use only information dated before the bond-day (the `fang_link` month-end before the trade). Ticker, name and Compustat CUSIP6 matches are diagnostics only.
10. **Rating**: latest `comp.adsprate.splticrm` with `datadate` <= bond-day for that GVKEY, expiring after 93 days. BBB- and better = IG, BB+ and worse = HY, unrated stays missing. The HY/IG panel therefore ends about three months after the last Legacy Compustat rating date.
11. **Aggregation**: issuer-day median spread, then the cross-issuer median for HY and for IG. A day needs at least 10 issuers in a class, or that class is missing that day.

Features (`HYCredit` in `macro/features/hy_credit.py`, release lag 1): `hy_spread`, `hy_spread_change_20`, `hy_spread_expanding_z`, `hy_ig_gap`, `hy_ig_gap_change_20`. Changes are over 20 NYSE sessions; the z-score is expanding (past and current values only, at least 60 observations). Gaps of up to 5 sessions are forward-filled with the last published value.

The residual-stress extension (descriptive, development only) is described in its own section below.

## Pre-registered threshold

At h = 21, the TRACE high-yield signal must deliver at least 5% lower QLIKE than the VIX + RV no-signal model with Clark-West p < 0.10. The incremental result with #11's Baa-spread features included will also be reported to determine whether TRACE adds information beyond the simpler aggregate credit signal.

## TRACE record semantics (FINRA documentation)

Sources: FINRA, "Historic TRACE Data: Enhanced Historic Time and Sales Trade Record File Layout". There are two versions: the pre-migration layout (FINRA `p120989`, November 2010) and the layout used after TRACE moved to a new platform on 2012-02-06 (FINRA `p353157`). Both describe the **Enhanced** historic file. Neither describes the standard disseminated BTDS file, whose `chng_cd`, `frmt_cd` and capped `ascii_rptd_vol_tx` therefore have no documented meaning here.

Implemented in `macro/data/trace.py`, because the documents state these rules:

| Rule | Documented basis |
|---|---|
| Pre-2012 `trc_st`: T = trade, C = cancellation, W = correction | p120989, Trade Status |
| C and W records point at the original via `orig_msg_seq_nb` = original `msg_seq_nb`; `msg_seq_nb` is unique within a trade report date | p120989, ORIG_MSG_SEQ_NB and MSG_SEQ_NB |
| C removes itself and its original; W replaces its original | follows from the two rows above |
| Reversal = `asof_cd` 'R', a cancellation of a trade reported on an earlier day; match it on CUSIP, quantity, price, execution date, execution time, buy/sell and contra party against earlier non-reversal records | both layouts, As Of Indicator note |
| `yld_sign_cd` '-' = negative yield; '+' (pre-2012) or blank (post-2012) = positive or zero | both layouts |
| `wis_fl` 'Y' = when-issued; `spcl_trd_fl` 'Y' = special condition that impacted the price; `sale_cndtn_cd` 'W' = weighted average price; `trdg_mkt_cd` 'P1' = primary-market list/fixed-price or takedown trade | both layouts |
| Inter-dealer buys (`cntra_mp_id` 'D', `rpt_side_cd` 'B') and sells are two sides of one trade; only the sell side is disseminated | both layouts, Contra Party note |
| Post-2012 (see "Validated on real rows"): X/C remove the record with the same CUSIP, `msg_seq_nb` and execution date; R/Y remove the record at (`pr_trd_dt`, `orig_msg_seq_nb`); X, C, Y and As-Of 'R' records are not trades | p353157 status codes and prior fields; WRDS encoding established on 2013 rows |
| Post-2012 weighted-average price: `trd_mod_4` 'W' | p353157, Trade Modifier 4 |

Further documented facts that affect the pipeline:

- Yield was reported by dealers until 2008-10-31. From 2008-11-03 FINRA calculates it for all dissemination-eligible trades. The development sample spans this change.
- Quantity units change. Pre-2012 `entrd_vol_qt` is in $1,000-par units; post-2012 Quantity is uncapped par value. Volume is a diagnostic only, so no conversion is applied yet.
- `trdg_mkt_cd` became a required field on 2010-03-01; earlier trades default to 'S1'.
- Pre-2012 price includes any commission (`cmsn_trd` flags it).
- Post-2012 status codes: T = trade, X = cancel, C = cancelled correction, R = new correction, Y = reversal. Cancels, corrections and reversals carry a Prior Trade Report Date and Prior Reference Number.

### Check against real rows (run `python -m macro.data.hy_runs validate-sample`)

The sample pull (`trace_enhanced_sample.csv`, plus `trace_enhanced_sample_2013.csv` for the post-2012 rules) reports each of these in `results/hy_credit/sample_validation.json`:

1. Same-day C/W linking (pre-2012 has no prior-report-date field): `cw_n_unmatched_references` should be about zero.
2. That W rows carry complete corrected trade details.
3. Reversal match rate. Unmatched reversals are expected only for originals reported before the extract.
4. Whether `yld_pt` is stored signed (`yield_sign_vs_value`).
5. `stdt`/`enddt` behaviour: master match with and without the validity interval.
6. Post-2012 links: the WRDS mapping Prior Reference Number -> `orig_msg_seq_nb` and Prior Trade Report Date -> `pr_trd_dt` is inferred from the names; `post_n_unmatched_references` tests it.
7. Whether keeping only disseminated rows before lifecycle cleaning gives the same trades (`dissem_prefilter_equivalence`). If it does, the development pull can filter on the server.

## Validated on real rows (2010-01-04..08 and 2013-01-07..08 samples)

Full output: `results/hy_credit/sample_validation.json` (aggregates only).

- **Pre-2012 C/W**: 7,305 of 7,312 references matched (99.9%), all to the same CUSIP; the 7 unmatched are lookback-pull artefacts. Same-day linking holds. 1,236 of 2,724 W records change the price, so W carries corrected details.
- **Pre-2012 reversals**: 2,549 of 2,890 matched (88%): 2,394 exactly and 155 by a same-minute pass. WRDS reversal records often carry `HH:MM:00` execution times while the original has seconds, so a still-unmatched reversal with zero seconds may match an original in that minute, all other details exact.
- **Post-2012 encoding differs from a literal reading of the layout**: X and C records carry the *original's* reference number in `msg_seq_nb`, with prior fields mostly blank. R and Y carry `pr_trd_dt`/`orig_msg_seq_nb` (1,759 of 1,765 matched, 99.7%). A correction is a C copy of the old version plus an R with the new values (399 of 401 C have their R). (report date, `msg_seq_nb`) is unique only among non-X/C records.
- **Yield sign**: `yld_pt` is stored unsigned; the sign is only in `yld_sign_cd` (4,191 '-').
- **Master versions**: inclusive, back-to-back (`next stdt = enddt + 1`). Without `stdt`/`enddt`, 95% of bond-days are ambiguous; with them 96.5% are unique. When overlapping versions agree on the fields the pipeline uses they count as `consistent`, giving 98.0% usable.
- **Issuer links**: `contrib_bond_firm_link.fang_link`, `crsp.ccmxpf_lnkhist` (CCM) and `wrdsapps_bondret` are **permission denied**. Production uses the exact issuer-CUSIP6 match to `comp.security` (`comp_cusip6`), plus bond -> PERMNO -> CRSP historical CUSIP6 -> `comp.security` if CRSP stock names are readable. Sample: 41% of spread bond-days linked, 6.7% conflicting (often listed-debt GVKEYs sharing the issuer's CUSIP6), 52% unmatched. Where both exist, CUSIP6 agrees with the normalized-name match 96% of the time; ticker agrees only 64%, so ticker stays diagnostic.
- **Ratings**: 85.8% of linked bond-days rated (IG 5,731, HY 1,865, unrated 1,254); rating age median 6 days; re-running the join with ratings truncated at each date changes 0 rows.
- **Daily aggregates**: about 185 HY and 335 IG issuers a day; HY about 490-515 bp, IG about 142-156 bp.
- **Server-side reductions** for the development pull, each reproducing the full lifecycle-cleaned set exactly on both samples: keep `dissem_fl = 'Y' or trc_st <> 'T' or asof_cd = 'R'` (a blanket `dissem_fl = 'Y'` would be wrong by 2 trades in 2010 and 186 in 2013), and keep CUSIPs with any master version in the bond universe (0 of 9,064 lifecycle links cross CUSIPs).
- **Sanity gates** (`macro.data.hy_runs.GATES`, fixed before any development data): all nine pass.

## Development results (2006-01-09 to 2017-12-29 pulled; usable through 2017-06-01)

Outputs: `results/hy_credit/` (`dev_report.json`, `qa_report.json`, `attrition.csv`, `attrition_by_month.csv`, `daily_coverage.csv`, `monthly_summary.csv`, three plots). All are aggregates; no bond-level rows.

| Stage | Rows | Share |
|---|---|---|
| Raw TRACE records (144 months) | 106,860,029 | |
| Cleaned trades (lifecycle, conditions, disseminated, sanity) | 99,763,741 | 93.4% of raw |
| Bond-days | 14,967,163 | |
| Master usable (unique or consistent version) | 14,948,114 | 99.9% of bond-days |
| Valid spread (universe, 1-30y, 5-3,500 bp) | 12,354,156 | 82.5% of bond-days |
| Issuer-linked (one GVKEY) | 7,491,671 | 60.6% of valid-spread bond-days |
| Rated | 6,466,105 | 86.3% of linked bond-days |
| Issuer-days (rated) | 1,680,109 | HY 655,358; IG 1,024,751 |
| Daily rows (HY and IG both present) | 2,852 | every session 2006-01-09 to 2017-06-01 |

- **Issuer links** (denominator: 12,354,156 valid-spread bond-days): linked 60.6%, conflict 24.3%, unmatched 15.0%. Linked bond-days come from both methods (51.7% of linked), `crsp_ncusip6` only (38.8%) and `comp_cusip6` only (9.5%). The sample's 41% used `comp_cusip6` alone. A per-month decomposition (2007-06, 2010-01, 2016-06) shows most conflicts arise inside the CRSP route (one bond-day reaching several GVKEYs: 15-23% of bond-days), then CUSIP6 shared by several GVKEYs (6-8%); the two methods each giving one different GVKEY is 4-5%. Conflicts stay unlinked by design.
- **Ratings end**: the last Legacy Compustat rating is 2017-02-28, so with the 93-day expiry no bond-day after 2017-06-01 is rated. The 490,505 linked bond-days from 2017-06-02 to 2017-12-29 are unrated and drop out; the last decision date with features is 2017-06-09 (5-session forward fill).
- **Daily issuer counts**: HY median 235 (min 11), IG median 364 (min 31). Six days have fewer than 50 HY issuers: Good Fridays and other partial sessions (2012-04-06, 2015-04-03, 2007-11-23, 2008-11-28) and two thin days (2011-09-01, 2012-07-23). No day falls below the 10-issuer minimum.
- **Spreads** (bp; median, 5th-95th percentile, max): HY 410 (258-900, max 1,595 on 2008-12-16), IG 140 (91-382, max 594 on 2008-12-15), HY-IG 270 (162-510, max 1,009).
- **QA** (`qa_report.json`): no failures. No chunk, reference table, rating or FRED value is dated 2018 or later; no duplicate days, bond-days or issuer-days; spreads within the screen; every decision date is after its observation date. `assert_no_lookahead` passes on the real series at seven cutoffs (2006-06-30 to 2017-05-31).
- **Predictive test**: not run. The harness (#8, #9, #10) and #11's features are not in the repository, so no QLIKE or Clark-West result exists.

## Reproducing

Scripts in `scripts/hy_credit/` (WRDS steps prompt for the password once; set `WRDS_USERNAME`):

- `wrds_pull_sample.py`: the 2010/2013 validation sample (`data/cache/*_sample*.csv`).
- `run_hy_once.py`: local preflight, one login, monthly pulls into `data/cache/hy_credit/` (resumable), then the local build, report, plots and QA. Logs go to `data/cache/logs/`.
- `preflight_build.py`, `plot_hy.py`, `qa_dev.py`: the local stages, also runnable alone.
- `test_run_hy_once.py`: fake-connection reliability tests for the orchestrator (no WRDS).
- `residual_research.py`: the residual-stress extension (local only; reads `data/cache/hy_credit/`, writes `data/cache/hy_residual/` and `results/hy_credit/residual/`).

## Timing

- TRACE: a bond-day dated *s* enters at the first NYSE session after *s* (`release_lag = 1`). This covers prints after the 16:00 ET close and dissemination delays. Bond-days on non-NYSE dates roll to the next session; the later observation wins.
- Treasury curve: FRED/H.15 daily values for *s* are published the next morning (L = 1, the CONTRIBUTING convention). The spread for *s* pairs same-day yields and enters with the TRACE lag, so neither input is used before it is published.
- Ratings: latest `datadate` <= *s*.
- Issuer links: `bondcrsp_link` and CRSP names are date-bounded at *s*. `comp.security` is a current snapshot (CUSIP -> GVKEY), so the `comp_cusip6` mapping is not point-in-time; it maps identifiers only and carries no spread or rating information.
- Known limitation: the Enhanced file contains final values after later corrections and reversals. A cancel or correction reported after *s* + 1 changes a historical bond-day slightly. Most corrections are same-day; the effect is not modelled.

## Extension: residual (within-rating) credit stress — descriptive, development only

Code: `macro/features/hy_residual.py` (`HYResidualStress`), runner `scripts/hy_credit/residual_research.py`, tests `tests/test_hy_residual.py`. Outputs (aggregates only): `results/hy_credit/residual/`. Issuer-level residuals stay in `data/cache/hy_residual/` (git-ignored). **This is not evidence of predictive improvement; no forecasting test has been run.**

**Final decision (production, decision B).** The production residual is

log(spread) = rating effects + maturity function + residual

fitted on each day's bonds (spec `M1`, `CHOSEN_SPEC = "M1"`; `HYResidualStress` reads `data/cache/hy_residual/residual_daily.parquet`). Production features: `residual_p90`, `residual_iqr`, `residual_positive_share`. Secondary diagnostics (reported, never fed to a model): `residual_p95`, `mean_positive_residual`, MAD, SD, the market-implied notch gap and the sector tail shares. **No sector premium is subtracted.** Reasons:

- past-fitted sector premiums are point-in-time but unstable;
- normal sector levels drift on the same horizon as persistent stress (the financials premium absorbed the 2008-10 crisis and stayed high);
- sector adjustment removes genuine stress episodes such as the 2011 financials stress;
- rolling sector premiums also absorb much of the 2007-09 and 2015-16 repricing;
- choosing a sector estimator from later forecasting results would create researcher degrees of freedom.

Point-in-time GICS sector (`comp.co_hgic`) is therefore used for **interpretation** (who is in the upper residual tail; whether an episode is broad or sector-concentrated), not for residualization. The M2 (same-day sector) and M2P (past-fitted sector) code and results are kept as documented research only.

**Motivation.** The basic signal measures how wide HY spreads are. This extension asks whether some issuers trade unusually wide for their agency rating and maturity, i.e. whether the bond market prices them as riskier than their rating says, and whether that differs from "HY spreads are high".

**Model (spec M1, fixed before looking at any outcome).** For each day *t*, on that day's rated bond-days only:

log(spread_bp)_b,t = rating-bucket effect_r(b),t + a_t·x_b + c_t·x_b² + e_b,t, with x = log(remaining maturity / 5y)

- Rating buckets: S&P notches AAA ... B- (16 buckets) plus CCC+ and below pooled; D excluded (defaulted-bond yields are not spreads); unrated bonds never enter. IG/HY boundary unchanged (BBB-/BB+).
- Log spreads: rating effects are roughly proportional, and the residual reads as a percentage (0.10 log points ~ 10% wider than peers). Spreads are already screened to 5-3,500 bp.
- Weighted least squares with weight 1/(issuer's bonds that day): every issuer has total weight one in the fit. Bond residuals are then collapsed to one issuer-day residual (median over the issuer's bonds), and every daily feature is computed across issuers.
- A bucket with fewer than 3 issuers that day is dropped (its dummy would fit it exactly); a day needs at least 100 issuers (typical day: 590; 4 of 2,852 days fail: 2011-09-01, 2012-04-06, 2012-07-23, 2015-04-03).

**Why a same-day cross-section (not a past-fitted model).** A daily fit uses only bonds observable on *t*, so it cannot leak future data and needs no refit schedule; the rating effects absorb the day's market level, so the residual measures the *shape* of the cross-section rather than its level, which `HYCredit` already captures. A past-fitted model would mostly measure the level shift since the fit. Features keep the one-session release lag of `HYCredit`.

**Specification ladder** (median daily issuer-weighted R², all 2,848 fitted days; rule fixed in advance: add a component only if it raises the median by at least 0.02):

| Model | Terms | Median R² | Median residual IQR |
|---|---|---|---|
| M0 | rating buckets | 0.679 | 0.458 |
| M1 | + quadratic in log maturity | 0.731 | 0.423 |
| M3 | + separate maturity slopes for HY | 0.738 | 0.415 |
| M1 / M2 on point-in-time-classified issuer-days (98%) | + same-day GICS sector (diagnostic) | 0.732 / 0.772 | 0.420 / 0.376 |

M1 is chosen: maturity adds 0.052, HY-specific slopes only 0.007. Adding same-day point-in-time GICS sector effects raises R² by about 0.04 (0.732 → 0.772 on the 98% of issuer-days with a sector), but same-day sector effects absorb contemporaneous sector repricing, and the past-fitted alternative proved unstable (subsection below). Sector is therefore not part of the production model.

**Daily features** (observation-dated, then lagged to decision dates):

| Feature | Definition |
|---|---|
| `residual_p90` (primary) | 90th percentile of issuer residuals |
| `residual_iqr` (primary) | interquartile range of issuer residuals (robust dispersion; MAD and SD also reported) |
| `residual_positive_share` (primary) | share of issuers whose residual exceeds the 90th percentile of all issuer residuals on days strictly before *t* (fixed 0.002 log-point histogram); missing for the first 60 days (from 2006-04-05) |
| `residual_p95`, `mean_positive_residual` | 95th percentile; mean of positive residuals |
| `residual_mean` | diagnostic only: ~0 by construction (median 0.004) |

The IQR ignores a tail smaller than a quarter of issuers, so P90 and the positive share carry the tail.

**Market-implied rating (interpretation only).** Each day's bucket effects are made monotone (weighted isotonic fit), and each bond is mapped to the bucket whose fitted log spread at its own maturity is nearest; issuer = median implied bucket; `notch_gap` = implied − agency bucket (positive = priced worse). Results: median gap 0 every day; 36% of issuers priced ≥1 notch worse and 17% ≥2 notches worse (median day), P90 gap 2. **The mapping is not a usable stress measure**: adjacent buckets are only ~0.1-0.2 log points apart against ~0.4 of residual dispersion, so the ≥1-notch share is dominated by noise, moves against HY stress (corr −0.08 with `hy_spread`; the rating gradient steepens in stress) and peaks in calm late 2013. Reported for interpretation; not a feature.

**Descriptive results (2006-01-09 to 2017-06-01; 1,669,937 issuer-days: HY 653,656, IG 1,016,281).**

- Fit: median daily R² 0.731 (5th percentile 0.613); lowest by year in 2008 (0.644). Max absolute bond residual 5.36 log points, within the screen's range; no duplicate issuer-days; no 2018+ data; every rating date ≤ observation date. `assert_no_lookahead` passes on the real series at 7 cutoffs.
- Levels: `residual_p90` median 0.46 (5th-95th 0.35-0.57, max 0.68 on 2016-02-17); `residual_iqr` 0.42 (0.33-0.52); `residual_positive_share` 0.107 (0.075-0.148, max 0.19 in March-April 2009).
- **Relation to basic HY stress** (decision dates): level correlations with `hy_spread` / `hy_ig_gap`: P90 0.46 / 0.50, IQR 0.51 / 0.53, positive share 0.71 / 0.70. 20-session changes are nearly uncorrelated with HY changes (|r| ≤ 0.05; 97-98% of change variance not explained by the HY changes). In levels, 74% of P90 and 50% of positive-share variance is not explained by HY level and gap.
- **Timing:** residual changes do not lead HY changes (correlation ≤ 0.05 at leads of 10-40 sessions); HY changes lead residual IQR changes by ~20 sessions (0.17). In 2007-09 residual stress peaked in March-April 2009, after the HY peak (2008-12-17); P90's window mean equals its outside mean, while the positive share rose (0.127 vs 0.104).
- **Where it differs from HY stress:** 299 sessions have expanding-z P90 > 1 while expanding-z HY < 0.5, mainly late 2014-2016 (energy) and 2012. In 2015-16 P90 crossed its own expanding 90th percentile on 2015-11-27; the HY-IG gap did so only on 2016-02-10 and the HY level not at all (its history includes 2008). HY-class residual P90 rose from mid-2014 with the oil decline while the HY level was moderate.
- **Who is in the tail (point-in-time GICS, interpretation only; rule fixed in advance: an episode is sector-concentrated if its largest sector's top-decile share is at least twice that sector's share of all issuers).** 2007-09: Financials 48% of the top residual decile vs 16% of issuers (3.0x, concentrated; Financials mean residual 0.35). 2011: Financials 55% vs 18% (3.1x, concentrated; 0.33). 2015-16: Energy 55% vs 15% (3.7x, concentrated; 0.44). 2012-13: Financials 39% vs 17% (2.3x), yet same-day sector effects lower P90 only from 0.50 to 0.47, so that elevation is mostly within sectors. Calm 2013-07..2014-06: broad (largest sector 1.35x). Same-day sector effects (diagnostic) would cut period P90 from 0.46 to 0.36 in 2007-09 and from 0.52 to 0.40 in 2015-16, which is why they are not used.
- **2012-13:** HY residual dispersion rises gradually from late 2011 (HY P90 ~0.40 to ~0.60 by early 2013) while IG falls. It starts before the 2012-02-06 TRACE migration, so it is not a clean data break; uncontrolled callability (HY bonds trading near call prices when rates are low) is a plausible but untested cause.
- **Coverage bias:** within the same TRACE grade, linked bonds trade tighter than unlinked ones (grade H median 238 vs 276 bp in 2007-06, 490 vs 640 in 2010-01, 425 vs 491 in 2016-06). The linked universe leans toward safer, public-equity issuers, so residual tails are probably understated.

**Limitations.** No sector adjustment, by design (decision B), so residual stress mixes issuer-specific and sector-concentrated repricing; no seniority (`debt_type_cd` is empty) or callability (no FISD); yield-to-maturity spreads for callable bonds; ratings end 2017-02-28 so the series ends 2017-06-01 (same as `HYCredit`) and cannot cover the 2018+ holdout with this rating source; issuer-link selection; the threshold's early history (2006-07) is short and calm, so the positive share is relative to a rising bar.

**Distinction from `HYCredit`.** `HYCredit` measures the level and change of the median HY spread. The residual features measure how unevenly spreads are distributed within rating and maturity peers on the same day. They are moderately correlated in levels (0.5-0.7) and nearly orthogonal in changes. Much of what distinguishes them is sector-concentrated stress.

**Sector decision (resolved below: (c) built and tested; recommendation B).** (a) M1, as built: sector-wide blowouts count as residual stress. (b) Same-day sector effects: sector-wide shocks are absorbed; on the subset this removes most of the episodes where residual stress differs from HY stress. (c) Sector premiums estimated from past data only: the closest match to "wider than rating and characteristics normally imply"; needs a point-in-time sector for every GVKEY (pulled from `comp.co_hgic`; see the next subsection).

### Point-in-time sector and a past-fitted sector premium (decision (c), tested, not adopted)

Code: `macro/data/sector.py`, `past_sector_premium` / `add_sector_offset` / spec `M2P` in `macro/features/hy_residual.py`, runner `scripts/hy_credit/residual_sector_research.py`, pull `scripts/hy_credit/wrds_pull_sector.py`. Aggregates: `results/hy_credit/residual_sector/`; issuer-level rows and the sector pull stay in `data/cache/hy_residual/` (git-ignored). The production (M1) outputs are in `results/hy_credit/residual/`.

**Sector source.** Compustat historical GICS, `comp.co_hgic` (one login, 2,120 panel GVKEYs, pre-2018 spells only, `indthru` capped at 2017-12-31): 3,159 spells for 2,065 GVKEYs, no overlaps, 265 issuers change sector during the sample. Each issuer-day gets the two-digit `gsector` of the spell in force that day (`indfrom` <= t <= `indthru`); a later reclassification never reaches back and dates before the first spell stay unclassified (tested). This is genuinely dated, unlike the `comp.company` header used before. Caveat: spells starting 1999-06-30 were assigned when GICS launched, which does not affect a 2006+ sample. Coverage: 98.3% of rated issuer-days (97.9-98.7% every year; HY 98.0%, IG 98.5%). `comp.funda` `sich`/`naicsh` and CRSP `siccd` were pulled as fallbacks and are unused.

**Premium (fixed before the real run).** Each day, a same-day WLS fit (spec `M2`: rating buckets + maturity + sector dummies for sectors with at least 10 issuers; thinner and unclassified issuers form the baseline) gives sector effects, centred to an issuer-weighted mean of zero. The premium for sector s on day t is the mean of those effects over fitted days strictly before t, times n/(n+120) (n = number of earlier days), and 0 until n >= 250 or for unclassified issuers. `M2P` refits rating + maturity each day on log spread minus that premium. The comparison period starts 2007-01-08, the first day on which at least 90% of classified issuers have a usable premium; M1 and M2 are restricted to the same days.

**Results (2007-01-08 to 2017-06-01, 2,598 days, 1,544,942 issuer-days).**

| | M1 no sector | M2P past sector | M2 same-day sector |
|---|---|---|---|
| Median daily R² | 0.727 | 0.742 | 0.766 |
| Median residual IQR / P90 | 0.429 / 0.467 | 0.407 / 0.435 | 0.383 / 0.407 |
| Calm 2013-07..2014-06: cross-sector SD of sector mean residual | 0.098 | 0.094 | 0.011 |
| 2007-09 Financials mean residual | 0.34 | 0.24 | 0.01 |
| 2011 Financials mean residual | 0.33 | 0.10 | 0.01 |
| 2015-16 Energy mean residual | 0.45 | 0.48 | 0.00 |
| Corr. with `hy_spread`: P90 / IQR / positive share | 0.43 / 0.47 / 0.68 | 0.08 / 0.17 / 0.39 | -0.09 / 0.01 / 0.24 |
| P90 mean: 2007-09 / 2011 / 2012-13 / 2015-16 / calm | 0.460 / 0.455 / 0.503 / 0.525 / 0.447 | 0.406 / 0.398 / 0.479 / 0.536 / 0.453 | 0.366 / 0.391 / 0.468 / 0.409 / 0.445 |

- **The past premium does not reliably capture *normal* sector differences.** The financials premium climbs from about 0.03 to 0.25 during 2008-12 and stays near 0.2 through 2017, so the crisis premium becomes "normal". By 2011 most of the financials stress is subtracted (0.33 → 0.10), and in calm 2013-14 financials are over-corrected (−0.11) while energy, whose history was tight, sits at +0.14. Predicting each day's same-day sector effects, the expanding premium beats no adjustment overall (issuer-weighted RMSE 0.115 vs 0.158), but is *worse* than no adjustment in 2013, 2014 and 2015 (0.102 vs 0.095, 0.111 vs 0.098, 0.216 vs 0.197).
- **Unusual repricing is preserved where history was benign.** The 2015-16 energy blowout survives in full (Energy is 58% of the top residual decile, vs 55% under M1 and 25% under same-day effects). The 2007-09 financials stress is partly kept, but not the 2011 one.
- **2012-13** stays elevated in all three variants, including same-day sector effects (P90 0.47-0.50), so it is not a sector effect; within-sector HY dispersion (callability is a plausible but untested cause) remains the open explanation.
- **Distinctness from `HYCredit` is partly an artefact.** M2P's low level correlation with HY spreads (0.08 for P90) comes with the GFC window averaging *below* the calm period (0.406 vs 0.453): slow-moving premium error adds low-frequency noise that is unrelated to HY levels. 20-session changes stay nearly orthogonal to HY changes in every variant (r² ≤ 0.03).
- **Sensitivity (descriptive only, not adopted):** a 504-day rolling premium tracks normal differences better (calm SD 0.061; RMSE 0.084), but absorbs about 40% of the 2015-16 energy repricing (0.48 → 0.29) and most of 2007-09 and 2011. No past-only estimator separates multi-year drift in normal sector premiums from a persistent episode, because both happen on the same time scale.
- **Market-implied rating:** sector adjustment barely changes it. Adjacent-bucket gap / residual IQR rises from 0.468 to 0.484; the ≥1-notch share has the same time-series SD (0.026 vs 0.027) and still moves against HY stress (corr −0.14 vs −0.09). It stays descriptive only.
- Checks: no 2018+ data, no duplicate issuer-days, minimum 175 issuers per fitted day, `assert_no_lookahead` passes on the real M2P series at 6 cutoffs.

**Recommendation: B, retain no-sector M1 as the single preregistered residual add-on.** All three variants are point-in-time and cover the sample. The past-sector premium meets neither goal reliably: it fails to remove normal sector differences in 2013-15 (worse than no adjustment) and erases the 2011 financials stress. Its extra distinctness from HY stress comes with slow estimation drift that turns the GFC into a below-calm period, and its behaviour depends on a window choice that cannot be resolved without looking at outcomes. M1 has no estimated history, is transparent, and its residual stress is openly a mix of idiosyncratic and sector-concentrated repricing; the point-in-time GICS tail composition is reported alongside it for interpretation. **Adopted:** `CHOSEN_SPEC = "M1"`; `HYResidualStress` reads the M1 series; M2/M2P remain in the code for research only and their outputs (`results/hy_credit/residual_sector/`) are labelled as a research comparison.

**To test once the harness exists** (h = 21, same QLIKE / Clark-West protocol; no threshold changes): RV + VIX; + `HYCredit`; + `HYCredit` + `HYResidualStress` (`residual_p90`, `residual_iqr`, `residual_positive_share`); and, once #11 exists, + Baa features. The question is whether residual stress improves forecasts after controlling for broad HY stress.

## Status

- The real-sample validation passes all gates. The development pull (144 months, 2006-01 to 2017-12) and build are complete; QA reports no failures (see Development results).
- Residual-stress extension: production spec M1 (decision B), sector used for interpretation only; development-only and descriptive. No predictive result yet.
- The walk-forward harness (#8, #9, #10) and #11's Baa features are not in the repository on any branch yet. `HYCredit` implements the `Signal` interface and passes `assert_no_lookahead` on synthetic data. No QLIKE or Clark-West result exists, so advancing / not-advancing is undetermined.

## Open questions

- Issuer-link coverage is 60.6% of valid-spread bond-days, with fang_link and CCM inaccessible; 24.3% are conflicts, mostly inside the CRSP route. Asking WRDS/GT for CCM access would add an independent exact link and may raise coverage. Linked issuers skew toward public-equity companies; check whether HY coverage is stable over time.
- Rating coverage after February 2017 (the legacy ratings end). The panel stops at 2017-06-01 unless another point-in-time rating source is added. **The 2018+ holdout cannot be evaluated with this rating source**; a replacement must be chosen before the frozen run.
- Whether TRACE `grade` (H/I) should extend coverage as a robustness classification.
- Bond Returns licensing and construction (`t_spread` units, rating source) before it is used for anything beyond a monthly sanity comparison.
