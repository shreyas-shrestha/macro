"""Pure transformations for the TRACE high-yield credit signal (#24).

Input: a clean bond-day panel with one row per date and bond, carrying an issuer key,
``spread_bp``, a point-in-time ``rating_class`` ('IG', 'HY' or missing) and optionally
sector and remaining maturity. See ``macro.data.trace`` and ``macro.data.ratings`` for
how that panel is built. Unrated bonds are excluded from both the HY and IG series.

Pipeline: ``issuer_day_spread`` -> ``aggregate_hy_ig`` (built by ``macro.data.hy_dataset``)
-> ``HYCredit`` signal: ``to_decision_dates`` (release lag 1) -> ``build_hy_features``.
``align_to_sessions`` / ``build_simple_features`` are the earlier session-aligned helpers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from macro.data.calendar import nyse_sessions
from macro.data.ratings import (  # noqa: F401  (re-exported for callers and tests)
    IG_MAX,
    RATING_ORDER,
    normalize_sp_rating,
    rating_class,
)
from macro.features.base import Signal


def add_rating_class(frame: pd.DataFrame, rating_col: str = "splticrm") -> pd.DataFrame:
    """Add ``rating``, ordinal ``rating_ord`` and ``rating_class`` (BBB-/BB+ boundary)."""
    out = frame.copy()
    out["rating"] = normalize_sp_rating(out[rating_col])
    out["rating_ord"] = out["rating"].map(RATING_ORDER)
    out["rating_class"] = rating_class(out["rating"])
    return out


def issuer_day_spread(
    frame: pd.DataFrame,
    issuer_col: str = "issuer_id",
    spread_col: str = "spread_bp",
    date_col: str = "trd_exctn_dt",
) -> pd.DataFrame:
    """Collapse bond-level spreads to one median per issuer and day.

    Issuers with many bonds then count once. Rows with a missing rating class or spread
    are excluded. Raises if one issuer has bonds in both classes on the same day,
    because the rating is meant to be issuer-level.
    Output: date, issuer, ``rating_class``, ``issuer_spread_bp``, ``n_bonds``.
    """
    required = {date_col, issuer_col, "rating_class", spread_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    usable = frame.loc[frame["rating_class"].notna() & frame[spread_col].notna()]
    out = (
        usable.groupby([date_col, issuer_col, "rating_class"], observed=True)[spread_col]
        .agg(issuer_spread_bp="median", n_bonds="size")
        .reset_index()
    )
    clash = out.duplicated([date_col, issuer_col], keep=False)
    if clash.any():
        examples = out.loc[clash, [date_col, issuer_col]].drop_duplicates().head(3)
        raise ValueError(f"Issuers with both IG and HY bonds on one day:\n{examples}")
    return out


def aggregate_hy_ig(issuer_day: pd.DataFrame, date_col: str = "trd_exctn_dt") -> pd.DataFrame:
    """Daily cross-issuer medians: ``hy_spread_bp``, ``ig_spread_bp``, ``hy_ig_gap_bp``.

    Also returns issuer counts ``n_hy_issuers`` and ``n_ig_issuers``.
    """
    grouped = issuer_day.groupby([date_col, "rating_class"], observed=True)["issuer_spread_bp"]
    median = grouped.median().unstack("rating_class")
    count = grouped.size().unstack("rating_class")

    wide = pd.DataFrame(index=median.index)
    for label, name in (("HY", "hy"), ("IG", "ig")):
        wide[f"{name}_spread_bp"] = median[label] if label in median else np.nan
        n = count[label] if label in count else 0
        wide[f"n_{name}_issuers"] = pd.Series(n, index=wide.index).fillna(0).astype("int64")
    wide["hy_ig_gap_bp"] = wide["hy_spread_bp"] - wide["ig_spread_bp"]
    return wide.sort_index()


def align_to_sessions(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Reindex to every NYSE session in the data's range, so a row step is a trading day.

    Sessions with no data become missing rows. Dates that are not NYSE sessions (TRACE
    can print when the bond market is open and NYSE is closed) are dropped, not rolled
    forward. Returns the aligned frame and the dropped dates.
    """
    index = pd.DatetimeIndex(daily.index)
    if index.has_duplicates:
        raise ValueError("daily index has duplicate dates")
    if index.empty:
        return daily.copy(), index
    sessions = nyse_sessions(index.min(), index.max())
    dropped = index[~index.isin(sessions)]
    aligned = daily.reindex(sessions)
    aligned.index.name = daily.index.name
    return aligned, dropped


def expanding_zscore(series: pd.Series, min_periods: int = 60) -> pd.Series:
    """Point-in-time z-score using only observations available through t."""
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def build_simple_features(daily: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Create the initial preregistered TRACE credit feature set.

    Changes are row differences, so ``daily`` must have one row per trading day
    (``align_to_sessions``); a change is missing if either endpoint is missing.
    """
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise ValueError("daily must be indexed by date")
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("daily index must be unique and sorted")
    out = pd.DataFrame(index=daily.index)
    out["hy_level"] = daily["hy_spread_bp"]
    out[f"hy_change_{horizon}d"] = daily["hy_spread_bp"].diff(horizon)
    out["hy_z"] = expanding_zscore(daily["hy_spread_bp"])
    out["hy_ig_gap"] = daily["hy_ig_gap_bp"]
    out[f"hy_ig_gap_change_{horizon}d"] = daily["hy_ig_gap_bp"].diff(horizon)
    return out


# --------------------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------------------

DAILY_CACHE = Path("data/cache/hy_credit/hy_daily.parquet")
FEATURE_COLUMNS = [
    "hy_spread",
    "hy_spread_change_20",
    "hy_spread_expanding_z",
    "hy_ig_gap",
    "hy_ig_gap_change_20",
]


def to_decision_dates(daily: pd.DataFrame, release_lag: int = 1) -> pd.DataFrame:
    """Re-index observation-dated rows to the decision date when they become usable.

    A row observed on day s (any calendar day) enters at the ``release_lag``-th NYSE
    session strictly after s. When several observation days map to one decision date
    (e.g. a bond-market day on which NYSE was closed), the latest observation wins.
    """
    if release_lag < 1:
        raise ValueError("TRACE-based rows need release_lag >= 1")
    obs = pd.DatetimeIndex(daily.index).normalize()
    sessions = nyse_sessions(obs.min(), obs.max() + pd.Timedelta(days=10 + 2 * release_lag))
    positions = sessions.searchsorted(obs, side="right") + (release_lag - 1)
    decision = sessions[positions]
    out = daily.copy()
    out.index = decision
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "decision_date"
    return out


def build_hy_features(
    daily: pd.DataFrame, horizon: int = 20, ffill_limit: int = 5, end_date=None
) -> pd.DataFrame:
    """The five preregistered features on a decision-date (NYSE session) index.

    ``daily`` is observation-dated with ``hy_spread_bp`` and ``hy_ig_gap_bp``. Rows are
    lagged to decision dates, reindexed to every NYSE session, and gaps of at most
    ``ffill_limit`` sessions are filled with the last published value (information
    already available). Changes are over ``horizon`` sessions; the z-score is expanding.
    ``end_date`` extends the session index (so a truncated build has the same rows).
    """
    decided = to_decision_dates(daily[["hy_spread_bp", "hy_ig_gap_bp"]])
    last = (
        decided.index.max()
        if end_date is None
        else max(decided.index.max(), pd.Timestamp(end_date))
    )
    sessions = nyse_sessions(decided.index.min(), last)
    aligned = decided.reindex(sessions).ffill(limit=ffill_limit)
    aligned.index.name = "decision_date"
    out = pd.DataFrame(index=aligned.index)
    out["hy_spread"] = aligned["hy_spread_bp"]
    out["hy_spread_change_20"] = aligned["hy_spread_bp"].diff(horizon)
    out["hy_spread_expanding_z"] = expanding_zscore(aligned["hy_spread_bp"])
    out["hy_ig_gap"] = aligned["hy_ig_gap_bp"]
    out["hy_ig_gap_change_20"] = aligned["hy_ig_gap_bp"].diff(horizon)
    return out.astype("float64")


class HYCredit(Signal):
    """TRACE high-yield credit spread features (#24).

    Inputs: daily issuer-weighted HY and IG median spreads built by
    ``macro.data.hy_dataset`` from WRDS TRACE Enhanced, the TRACE master file, exact
    issuer links, Legacy Compustat S&P ratings and the H.15 Treasury curve, cached at
    ``DAILY_CACHE``. Pass ``daily`` to build from an in-memory frame instead (tests).
    """

    name = "hy_credit"
    source = "WRDS TRACE Enhanced; Compustat S&P ratings; FRED H.15"
    start_date = pd.Timestamp("2006-01-10")
    release_lag = 1

    def __init__(self, daily: pd.DataFrame | None = None):
        self._daily = daily

    def _load(self) -> pd.DataFrame:
        if self._daily is not None:
            return self._daily
        return pd.read_parquet(DAILY_CACHE)

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        daily = self._load()
        daily = daily.loc[pd.DatetimeIndex(daily.index) < pd.Timestamp(end_date)]
        features = build_hy_features(daily, end_date=end_date)
        return features.loc[self.start_date : end_date]
