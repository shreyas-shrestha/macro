"""Residual credit stress for #24: spreads relative to rating and maturity peers.

Development-only, descriptive extension of ``HYCredit`` (see ``research/hy_credit.md``).

Production model (spec ``M1``, decision B), fitted separately on each day's bonds
(cross-sectional, same-day data only):

    log(spread_bp)_b = rating-bucket effect_r(b) + g(log maturity_b) [+ ...] + e_b

- Rating buckets: S&P notches AAA ... B- individually; CCC+ and below pooled (sparse);
  D (default) excluded because yield spreads of defaulted bonds are not meaningful.
- g is a quadratic in x = log(remaining maturity / 5y) (spec ``M1``). ``M3`` lets the
  two maturity slopes differ for HY buckets.
- Sector is NOT part of the production model. Research diagnostics kept here: ``M2``
  (same-day point-in-time GICS sector dummies) and ``M2P`` (log spread minus a sector
  premium fitted on earlier days only: ``past_sector_premium``). Both were examined and
  not adopted (``research/hy_credit.md``); point-in-time sector is used to interpret the
  residual tail, never to residualize the production signal.
- Weighted least squares with weight 1 / (issuer's bond count that day), so every
  issuer has total weight one and multi-bond issuers do not dominate the fit.
- A bucket with fewer than ``MIN_ISSUERS_PER_BUCKET`` issuers that day is dropped (its
  own dummy would fit it exactly); a day with fewer than ``MIN_ISSUERS_PER_DAY`` issuers
  after that is not fitted.

Residual e_b is in log points (0.10 ~ 10% wider than rating/maturity peers that day).
Issuer-day residual = median over the issuer's bonds. Daily features are computed over
issuers (one issuer, one observation). Because each day's fit has a full set of rating
effects, the issuer-weighted mean residual is near zero by construction; it is never
used as a stress measure.

Point in time: the fit for day t uses only bonds traded on t and their ratings in force
on t. ``residual_positive_share`` uses a threshold estimated from residuals of days
strictly before t. ``HYResidualStress`` then applies the same one-session release lag
as ``HYCredit``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from macro.config import HOLDOUT_START
from macro.data.calendar import nyse_sessions
from macro.data.ratings import IG_MAX, RATING_ORDER, normalize_sp_rating
from macro.features.base import Signal
from macro.features.hy_credit import to_decision_dates

CCC_BUCKET = RATING_ORDER["CCC+"]  # 17: CCC+ and below pooled
EXCLUDED_RATINGS = ("D",)
MATURITY_REF_Y = 5.0
MIN_ISSUERS_PER_DAY = 100
MIN_ISSUERS_PER_BUCKET = 3
MIN_ISSUERS_PER_SECTOR = 10  # same-day sector dummy; thinner sectors join the pooled baseline
SECTOR_MIN_DAYS = 250  # a past sector premium is used only after 250 earlier fitted days
SECTOR_SHRINK_DAYS = 120  # premium = past mean * n / (n + 120): thin history shrinks to 0
SECTOR_START_COVERAGE = 0.90  # production series starts when 90% of classified issuers
# have a usable premium
THRESHOLD_QUANTILE = 0.90
THRESHOLD_MIN_DAYS = 60
HIST_EDGES = np.linspace(-4.0, 4.0, 4001)  # 0.002 log-point bins for the expanding quantile
SPECS = {
    "M0": {"maturity": False, "hy_slope": False, "sector": False, "offset": False},
    "M1": {"maturity": True, "hy_slope": False, "sector": False, "offset": False},
    "M2": {"maturity": True, "hy_slope": False, "sector": True, "offset": False},  # same-day
    "M2P": {"maturity": True, "hy_slope": False, "sector": False, "offset": True},  # past sector
    "M3": {"maturity": True, "hy_slope": True, "sector": False, "offset": False},
}
CHOSEN_SPEC = "M1"  # production (decision B); M2P/M2 are research diagnostics only
FEATURES = ("residual_p90", "residual_iqr", "residual_positive_share")

DAILY_CACHE = Path("data/cache/hy_residual/residual_daily.parquet")  # M1 production series


def rating_bucket(rating: pd.Series) -> pd.Series:
    """Notch 1 (AAA) ... 16 (B-), 17 = CCC+ and below; missing for unrated and D."""
    norm = normalize_sp_rating(rating)
    notch = norm.map(RATING_ORDER).astype("Float64")
    notch = notch.where(~norm.isin(EXCLUDED_RATINGS))
    return notch.clip(upper=CCC_BUCKET).astype("Int64")


def prepare_bond_days(frame: pd.DataFrame, sector: pd.Series | None = None) -> pd.DataFrame:
    """Model-ready bond-days: rated (not D), positive spread and maturity, known issuer.

    ``frame`` needs ``trd_exctn_dt``, ``gvkey``, ``rating``, ``spread_bp`` and
    ``remaining_maturity_y``. ``sector``, if given, is either a per-row label aligned with
    ``frame``'s index (point-in-time, e.g. ``macro.data.sector.gics_sector_asof``) or a
    static gvkey -> label map (diagnostic only).
    """
    required = {"trd_exctn_dt", "gvkey", "rating", "spread_bp", "remaining_maturity_y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["trd_exctn_dt"])
    if (dates >= HOLDOUT_START).any():
        raise ValueError("bond-days include holdout dates (2018+)")
    out = pd.DataFrame(
        {
            "trd_exctn_dt": dates.astype("datetime64[ns]"),
            "gvkey": frame["gvkey"].astype("string"),
            "bucket": rating_bucket(frame["rating"]),
            "spread_bp": pd.to_numeric(frame["spread_bp"]),
            "maturity_y": pd.to_numeric(frame["remaining_maturity_y"]),
        },
        index=frame.index,
    )
    keep = (
        out["bucket"].notna()
        & out["gvkey"].notna()
        & out["spread_bp"].gt(0)
        & out["maturity_y"].gt(0)
    )
    out = out.loc[keep].copy()
    out["bucket"] = out["bucket"].astype("int64")
    out["log_spread"] = np.log(out["spread_bp"])
    out["mat_x"] = np.log(out["maturity_y"] / MATURITY_REF_Y)
    out["hy"] = out["bucket"].gt(IG_MAX).astype("int64")
    if sector is not None:
        if sector.index.equals(frame.index):
            out["sector"] = sector.loc[out.index].astype("string")
        else:
            out["sector"] = out["gvkey"].map(sector).astype("string")
    return out.sort_values(["trd_exctn_dt", "gvkey"], kind="stable")


def _isotonic_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators fit, non-decreasing in index order."""
    blocks: list[list[float]] = []  # [mean, weight, length]
    for v, w in zip(values, weights, strict=True):
        blocks.append([float(v), float(w), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, n2 = blocks.pop()
            m1, w1, n1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(n, m) for m, _, n in blocks])


def fit_day(day: pd.DataFrame, spec: str = CHOSEN_SPEC, implied: bool = False) -> dict | None:
    """WLS fit of one day's cross-section. Returns None if the day has too few issuers.

    Result keys: ``index`` (rows used), ``residual``, ``fitted``, ``r2`` (issuer-weighted),
    ``n_issuers``, ``n_bonds``, ``buckets``, ``bucket_effect``, ``beta`` (all
    coefficients: buckets, then maturity, HY slope, sector) and, if ``implied``,
    ``implied_bucket`` per bond (nearest bucket after a monotone fit of bucket effects).
    """
    opts = SPECS[spec]
    issuers_per_bucket = day.groupby("bucket")["gvkey"].nunique()
    good = issuers_per_bucket.index[issuers_per_bucket >= MIN_ISSUERS_PER_BUCKET]
    day = day.loc[day["bucket"].isin(good)]
    n_issuers = day["gvkey"].nunique()
    if n_issuers < MIN_ISSUERS_PER_DAY:
        return None
    buckets = np.sort(day["bucket"].unique())
    bucket_dummies = (day["bucket"].to_numpy()[:, None] == buckets[None, :]).astype(float)
    parts = [bucket_dummies]
    x = day["mat_x"].to_numpy()
    hy = day["hy"].to_numpy()
    if opts["maturity"]:
        parts.append(np.column_stack([x, x**2]))
    if opts["hy_slope"]:
        parts.append(np.column_stack([hy * x, hy * x**2]))
    levels: list[str] = []
    reference = None
    if opts["sector"]:
        sectors = day["sector"]
        counts = day.loc[sectors.notna()].groupby("sector")["gvkey"].nunique()
        usable = sorted(counts.index[counts >= MIN_ISSUERS_PER_SECTOR])
        values = sectors.fillna("").to_numpy()
        pooled = ~np.isin(values, usable)  # unclassified or thin-sector issuers
        if day.loc[pooled, "gvkey"].nunique() >= MIN_ISSUERS_PER_SECTOR:
            levels = usable  # the pooled group is the baseline
        elif len(usable) > 1:
            reference = counts.loc[usable].idxmax()
            levels = [s for s in usable if s != reference]
        if levels:
            parts.append(np.column_stack([values == s for s in levels]).astype(float))
    design = np.hstack(parts)
    y = day["log_spread"].to_numpy()
    offset = day["offset"].to_numpy(float) if opts["offset"] else np.zeros(len(day))
    weight = 1.0 / day.groupby("gvkey")["gvkey"].transform("size").to_numpy()
    root = np.sqrt(weight)
    beta, *_ = np.linalg.lstsq(design * root[:, None], (y - offset) * root, rcond=None)
    fitted = design @ beta + offset
    residual = y - fitted
    ybar = np.average(y, weights=weight)
    r2 = 1.0 - np.sum(weight * residual**2) / np.sum(weight * (y - ybar) ** 2)
    out = {
        "index": day.index,
        "residual": residual,
        "fitted": fitted,
        "r2": float(r2),
        "n_issuers": int(n_issuers),
        "n_bonds": int(len(day)),
        "buckets": buckets,
        "bucket_effect": beta[: len(buckets)],
        "beta": beta,
    }
    if opts["sector"]:
        # Centre so the issuer-weighted mean sector effect that day is zero.
        coef = beta[design.shape[1] - len(levels) :] if levels else np.array([])
        row_effect = parts[-1] @ coef if levels else np.zeros(len(day))
        centre = np.average(row_effect, weights=weight)
        effect = {s: float(coef[i] - centre) for i, s in enumerate(levels)}
        if reference is not None:
            effect[reference] = float(-centre)
        out["sector_effect"] = effect
    if implied:
        out["implied_bucket"] = _implied_buckets(day, beta, buckets, opts, offset)
    return out


def _implied_buckets(
    day: pd.DataFrame, beta: np.ndarray, buckets: np.ndarray, opts, offset: np.ndarray
) -> np.ndarray:
    """Nearest rating bucket (in log spread) for each bond at its own maturity.

    Bucket effects are first made non-decreasing from AAA to CCC (weighted isotonic fit
    on issuer counts), so the mapping is monotone. The maturity terms stay at the bond's
    own maturity; under ``M3`` the HY slope applies to HY candidate buckets. Only defined
    for specs without sector terms.
    """
    k = len(buckets)
    issuers = day.groupby("bucket")["gvkey"].nunique().reindex(buckets).to_numpy(float)
    effect = _isotonic_increasing(beta[:k], issuers)
    x = day["mat_x"].to_numpy()
    rest = np.zeros(len(day))
    j = k
    if opts["maturity"]:
        rest += beta[j] * x + beta[j + 1] * x**2
        j += 2
    hy_shift = np.zeros(len(day))
    if opts["hy_slope"]:
        hy_shift = beta[j] * x + beta[j + 1] * x**2
    base = day["log_spread"].to_numpy() - offset - rest
    candidate = effect[None, :] + np.where(buckets[None, :] > IG_MAX, hy_shift[:, None], 0.0)
    candidate = np.maximum.accumulate(candidate, axis=1)  # monotone per bond under M3
    nearest = np.abs(base[:, None] - candidate).argmin(axis=1)
    return buckets[nearest]


def bond_residuals(
    prepared: pd.DataFrame, spec: str = CHOSEN_SPEC, implied: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every day separately. Returns (bond residual rows, per-day fit statistics)."""
    rows, stats = [], []
    for date, day in prepared.groupby("trd_exctn_dt", sort=True):
        if implied and SPECS[spec]["sector"]:
            raise ValueError("implied buckets are only defined for specs without sector")
        fit = fit_day(day, spec, implied=implied)
        if fit is None:
            stats.append(
                {"trd_exctn_dt": date, "fitted": False, "n_issuers": day["gvkey"].nunique()}
            )
            continue
        part = prepared.loc[fit["index"], ["trd_exctn_dt", "gvkey", "bucket", "spread_bp"]]
        part = part.assign(residual=fit["residual"], fitted=fit["fitted"])
        if implied:
            part["implied_bucket"] = fit["implied_bucket"]
        rows.append(part)
        stat = {
            "trd_exctn_dt": date,
            "fitted": True,
            "r2": fit["r2"],
            "n_issuers": fit["n_issuers"],
            "n_bonds": fit["n_bonds"],
            "n_buckets": len(fit["buckets"]),
        }
        for b, e in zip(fit["buckets"], fit["bucket_effect"], strict=True):
            stat[f"effect_{b}"] = float(e)
        for s, e in fit.get("sector_effect", {}).items():
            stat[f"sector_{s}"] = e
        stats.append(stat)
    if not rows:
        raise ValueError("no day had enough issuers to fit")
    return pd.concat(rows), pd.DataFrame(stats).set_index("trd_exctn_dt")


def sector_effects(stats: pd.DataFrame, dates=None) -> pd.DataFrame:
    """Daily centred same-day sector effects (date x sector) from ``bond_residuals(.., "M2")``.

    ``dates`` (all observation dates) reindexes the result; unfitted days are missing.
    """
    effects = stats.filter(like="sector_").rename(columns=lambda c: c.removeprefix("sector_"))
    if dates is not None:
        effects = effects.reindex(pd.DatetimeIndex(sorted(set(dates)), name=effects.index.name))
    return effects.sort_index()


def past_sector_premium(
    effects: pd.DataFrame,
    min_days: int = SECTOR_MIN_DAYS,
    shrink_days: float = SECTOR_SHRINK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normal sector premium for each date from same-day effects of EARLIER days only.

    premium[s, t] = mean(effect[s, d] for d < t) * n / (n + shrink_days), n = number of
    earlier days with an estimate for s; 0 while n < min_days. Returns (premium, n).
    """
    prior_n = effects.notna().cumsum().shift(1).fillna(0)
    prior_sum = effects.fillna(0).cumsum().shift(1).fillna(0)
    mean = prior_sum / prior_n.where(prior_n > 0)
    premium = (mean * prior_n / (prior_n + shrink_days)).where(prior_n >= min_days, 0.0)
    return premium.fillna(0.0), prior_n


def add_sector_offset(
    prepared: pd.DataFrame, premium: pd.DataFrame, prior_n: pd.DataFrame, min_days=SECTOR_MIN_DAYS
) -> pd.DataFrame:
    """Add ``offset`` (past sector premium, 0 if unclassified or not yet usable) and
    ``premium_usable`` to model-ready bond-days."""
    long = premium.stack().rename("offset").reset_index()
    long.columns = ["trd_exctn_dt", "sector", "offset"]
    usable = prior_n.stack().rename("n").reset_index()
    usable.columns = ["trd_exctn_dt", "sector", "n"]
    long = long.merge(usable, on=["trd_exctn_dt", "sector"])
    long["trd_exctn_dt"] = pd.to_datetime(long["trd_exctn_dt"]).astype("datetime64[ns]")
    long["sector"] = long["sector"].astype("string")
    out = prepared.reset_index().merge(long, on=["trd_exctn_dt", "sector"], how="left")
    out = out.set_index(prepared.index.name or "index")
    out.index.name = prepared.index.name
    out["premium_usable"] = out["n"].ge(min_days).fillna(False).astype(bool)
    out["offset"] = out["offset"].fillna(0.0)
    return out.drop(columns="n")


def production_start(prepared: pd.DataFrame, coverage: float = SECTOR_START_COVERAGE):
    """First date on which at least ``coverage`` of classified issuers have a usable premium."""
    classified = prepared.loc[prepared["sector"].notna()]
    share = classified.groupby("trd_exctn_dt").apply(
        lambda d: d.loc[d["premium_usable"], "gvkey"].nunique() / d["gvkey"].nunique()
    )
    ok = share.index[share >= coverage]
    return pd.Timestamp(ok[0]) if len(ok) else None


def issuer_residuals(bond_res: pd.DataFrame) -> pd.DataFrame:
    """One row per issuer-day: median bond residual, agency bucket, implied bucket gap."""
    keys = ["trd_exctn_dt", "gvkey"]
    grouped = bond_res.groupby(keys, sort=True)
    out = grouped.agg(
        residual=("residual", "median"),
        bucket=("bucket", "first"),
        n_bonds=("residual", "size"),
    )
    if grouped["bucket"].nunique().gt(1).any():
        raise ValueError("an issuer has bonds in two rating buckets on one day")
    if "implied_bucket" in bond_res:
        implied = grouped["implied_bucket"].median()
        out["implied_bucket"] = np.floor(implied + 0.5).astype("int64")
        out["notch_gap"] = out["implied_bucket"] - out["bucket"]
    return out.reset_index()


def expanding_threshold(issuer_res: pd.DataFrame, q: float = THRESHOLD_QUANTILE) -> pd.Series:
    """Per day, the q-quantile of all issuer residuals from days strictly before it.

    Missing until ``THRESHOLD_MIN_DAYS`` earlier days exist. Uses a fixed 0.002
    log-point histogram (values outside [-4, 4] fall in the end bins).
    """
    dates = np.sort(issuer_res["trd_exctn_dt"].unique())
    clipped = issuer_res["residual"].clip(HIST_EDGES[0], HIST_EDGES[-1] - 1e-9)
    by_day = clipped.groupby(issuer_res["trd_exctn_dt"])
    cumulative = np.zeros(len(HIST_EDGES) - 1)
    out = np.full(len(dates), np.nan)
    for i, date in enumerate(dates):
        if i >= THRESHOLD_MIN_DAYS:
            cdf = np.cumsum(cumulative) / cumulative.sum()
            out[i] = HIST_EDGES[1:][np.searchsorted(cdf, q)]
        cumulative += np.histogram(by_day.get_group(date), bins=HIST_EDGES)[0]
    return pd.Series(out, index=pd.DatetimeIndex(dates, name="trd_exctn_dt"), name="threshold")


def daily_residual_features(issuer_res: pd.DataFrame) -> pd.DataFrame:
    """Issuer-weighted daily residual-stress measures, indexed by observation date."""

    def summarize(g: pd.DataFrame) -> pd.Series:
        r = g["residual"].to_numpy()
        q25, q50, q75, q90, q95 = np.quantile(r, [0.25, 0.5, 0.75, 0.9, 0.95])
        positive = r[r > 0]
        row = {
            "n_issuers": len(r),
            "residual_p90": q90,
            "residual_p95": q95,
            "residual_iqr": q75 - q25,
            "residual_mad": float(np.median(np.abs(r - q50))),
            "residual_std": float(np.std(r, ddof=1)),
            "residual_mean": float(r.mean()),
            "residual_median": q50,
            "mean_positive_residual": float(positive.mean()) if len(positive) else np.nan,
        }
        if "notch_gap" in g:
            gap = g["notch_gap"].to_numpy()
            row["notch_gap_ge1_share"] = float((gap >= 1).mean())
            row["notch_gap_ge2_share"] = float((gap >= 2).mean())
            row["notch_gap_median"] = float(np.median(gap))
            row["notch_gap_p90"] = float(np.quantile(gap, 0.9))
        return pd.Series(row)

    daily = issuer_res.groupby("trd_exctn_dt", sort=True)[list(issuer_res.columns)].apply(summarize)
    threshold = expanding_threshold(issuer_res)
    daily["threshold"] = threshold
    above = (
        issuer_res["residual"].to_numpy() > threshold.reindex(issuer_res["trd_exctn_dt"]).to_numpy()
    )
    share = pd.Series(above, index=issuer_res["trd_exctn_dt"]).groupby(level=0).mean()
    daily["residual_positive_share"] = share.where(threshold.notna())
    daily.index = pd.DatetimeIndex(daily.index, name="trd_exctn_dt")
    return daily


def build_residual_features(
    daily: pd.DataFrame, columns=FEATURES, ffill_limit: int = 5, end_date=None
) -> pd.DataFrame:
    """Lag observation-dated daily features to NYSE decision dates (as ``HYCredit``)."""
    decided = to_decision_dates(daily[list(columns)])
    last = (
        decided.index.max()
        if end_date is None
        else max(decided.index.max(), pd.Timestamp(end_date))
    )
    sessions = nyse_sessions(decided.index.min(), last)
    out = decided.reindex(sessions).ffill(limit=ffill_limit)
    out.index.name = "decision_date"
    return out.astype("float64")


class HYResidualStress(Signal):
    """Within-rating TRACE spread stress (#24 extension; development, descriptive).

    Features: ``residual_p90`` (90th percentile of issuer residuals), ``residual_iqr``
    (interquartile range) and ``residual_positive_share`` (share of issuers above the
    expanding historical 90th percentile). Built by ``scripts/hy_credit/residual_research.py``
    into ``DAILY_CACHE``; pass ``daily`` to build from an in-memory frame instead.
    """

    name = "hy_residual_stress"
    source = "WRDS TRACE Enhanced; Compustat S&P ratings; FRED H.15"
    start_date = pd.Timestamp("2006-01-10")
    release_lag = 1

    def __init__(self, daily: pd.DataFrame | None = None):
        self._daily = daily

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        daily = self._daily if self._daily is not None else pd.read_parquet(DAILY_CACHE)
        daily = daily.loc[pd.DatetimeIndex(daily.index) < pd.Timestamp(end_date)]
        features = build_residual_features(daily, end_date=end_date)
        return features.loc[self.start_date : end_date]
