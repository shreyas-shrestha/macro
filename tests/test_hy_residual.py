"""Synthetic tests for the residual credit-stress extension (#24)."""

import numpy as np
import pandas as pd
import pytest

from macro.features.hy_residual import (
    CCC_BUCKET,
    HIST_EDGES,
    THRESHOLD_MIN_DAYS,
    HYResidualStress,
    add_sector_offset,
    bond_residuals,
    daily_residual_features,
    expanding_threshold,
    fit_day,
    issuer_residuals,
    past_sector_premium,
    prepare_bond_days,
    production_start,
    rating_bucket,
    sector_effects,
)
from macro.features.testing import assert_no_lookahead

BUCKET_RATINGS = ["AA", "A", "BBB", "BB", "B"]  # buckets 3, 6, 9, 12, 15
TRUE_EFFECT = {3: np.log(60), 6: np.log(90), 9: np.log(150), 12: np.log(300), 15: np.log(500)}
TRUE_MAT = (0.10, -0.03)  # log-spread slope and curvature in x = log(m / 5)
SECTORS = ["Energy", "Financials", "Industrials"]
TRUE_SECTOR = {"Energy": 0.20, "Financials": 0.0, "Industrials": -0.10}


def synthetic_bond_days(n_days=3, n_issuers=150, bonds=2, noise=0.0, seed=0, start="2010-01-04"):
    """Bond-days whose log spread = date level + rating + maturity + sector + noise."""
    rng = np.random.default_rng(seed)
    ratings = np.array(BUCKET_RATINGS)[np.arange(n_issuers) % len(BUCKET_RATINGS)]
    sectors = np.array(SECTORS)[(np.arange(n_issuers) // len(BUCKET_RATINGS)) % len(SECTORS)]
    rows = []
    for d, date in enumerate(pd.bdate_range(start, periods=n_days)):
        level = 0.05 * d
        for i in range(n_issuers):
            for _ in range(bonds):
                m = float(rng.uniform(1.5, 25))
                x = np.log(m / 5)
                bucket = int(rating_bucket(pd.Series([ratings[i]])).iloc[0])
                y = level + TRUE_EFFECT[bucket] + TRUE_MAT[0] * x + TRUE_MAT[1] * x**2
                y += TRUE_SECTOR[sectors[i]] + noise * rng.standard_normal()
                rows.append((date, f"G{i:04d}", ratings[i], np.exp(y), m, sectors[i]))
    frame = pd.DataFrame(
        rows,
        columns=["trd_exctn_dt", "gvkey", "rating", "spread_bp", "remaining_maturity_y", "sec"],
    )
    sector = frame.drop_duplicates("gvkey").set_index("gvkey")["sec"]
    return frame, sector


def test_rating_bucket_pools_ccc_and_excludes_default_and_unrated():
    out = rating_bucket(pd.Series(["AAA", "B-", "CCC+", "CC", "C", "D", "NR", None]))
    assert out.tolist()[:5] == [1, 16, CCC_BUCKET, CCC_BUCKET, CCC_BUCKET]
    assert out.iloc[5:].isna().all()


def test_holdout_dates_are_rejected():
    frame, _ = synthetic_bond_days(n_days=1, n_issuers=5, start="2017-12-29")
    frame.loc[0, "trd_exctn_dt"] = pd.Timestamp("2018-01-02")
    with pytest.raises(ValueError, match="holdout"):
        prepare_bond_days(frame)


def test_rating_and_maturity_effects_are_recovered():
    frame, sector = synthetic_bond_days(n_days=1)
    day = prepare_bond_days(frame, sector)
    fit = fit_day(day, "M2")
    effect = dict(zip(fit["buckets"], fit["bucket_effect"], strict=True))
    # Relative rating effects (levels absorb the reference sector).
    for b in TRUE_EFFECT:
        assert effect[b] - effect[3] == pytest.approx(TRUE_EFFECT[b] - TRUE_EFFECT[3], abs=1e-9)
    k = len(fit["buckets"])
    assert fit["beta"][k : k + 2] == pytest.approx(TRUE_MAT, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0)


def test_sector_effects_are_recovered_relative_to_reference():
    frame, sector = synthetic_bond_days(n_days=1)
    fit = fit_day(prepare_bond_days(frame, sector), "M2")
    k = len(fit["buckets"])
    sector_coefs = fit["beta"][k + 2 :]
    # Reference = most-issuer sector (Energy, by construction of the cycle); others relative.
    expected = sorted(TRUE_SECTOR[s] - TRUE_SECTOR["Energy"] for s in ("Financials", "Industrials"))
    assert sorted(sector_coefs) == pytest.approx(expected, abs=1e-9)


def test_rating_only_model_leaves_maturity_in_residual():
    frame, sector = synthetic_bond_days(n_days=1)
    day = prepare_bond_days(frame, sector)
    r2_m0 = fit_day(day, "M0")["r2"]
    r2_m1 = fit_day(day, "M1")["r2"]
    assert r2_m0 < r2_m1 < 1.0  # M1 still misses the sector effect


def test_days_with_too_few_issuers_are_not_fitted():
    frame, _ = synthetic_bond_days(n_days=1, n_issuers=60)
    assert fit_day(prepare_bond_days(frame), "M1") is None


def test_thin_rating_buckets_are_dropped_not_fitted_exactly():
    frame, _ = synthetic_bond_days(n_days=1, noise=0.1)
    lonely = frame.iloc[[0]].assign(gvkey="LONE", rating="CCC")
    fit = fit_day(prepare_bond_days(pd.concat([frame, lonely], ignore_index=True)), "M1")
    assert CCC_BUCKET not in fit["buckets"]


def test_issuer_with_many_bonds_does_not_dominate():
    frame, _ = synthetic_bond_days(n_days=1, noise=0.2, seed=3)
    base = fit_day(prepare_bond_days(frame), "M1")
    target = frame["gvkey"].eq("G0003")  # one BB issuer
    heavy = pd.concat([frame] + [frame.loc[target]] * 40, ignore_index=True)
    fit = fit_day(prepare_bond_days(heavy), "M1")
    # Copies of the same bonds carry total weight one: the fit is unchanged.
    assert fit["beta"] == pytest.approx(base["beta"], abs=1e-9)
    res, _ = bond_residuals(prepare_bond_days(heavy), "M1")
    issuers = issuer_residuals(res)
    assert issuers["gvkey"].is_unique and len(issuers) == 150


def test_issuer_residual_is_median_of_its_bonds():
    bond_res = pd.DataFrame(
        {
            "trd_exctn_dt": pd.Timestamp("2010-01-04"),
            "gvkey": ["A", "A", "A", "B"],
            "bucket": [9, 9, 9, 12],
            "residual": [0.1, 0.5, 2.0, -0.2],
        }
    )
    out = issuer_residuals(bond_res).set_index("gvkey")
    assert out.loc["A", "residual"] == pytest.approx(0.5)
    assert out.loc["A", "n_bonds"] == 3


def _issuer_frame(residuals_by_day):
    rows = [
        (pd.Timestamp(d), f"G{i}", 9, r)
        for d, res in residuals_by_day.items()
        for i, r in enumerate(res)
    ]
    return pd.DataFrame(rows, columns=["trd_exctn_dt", "gvkey", "bucket", "residual"])


def test_daily_quantiles_dispersion_and_positive_mean():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.3, 501)
    daily = daily_residual_features(_issuer_frame({"2010-01-04": r})).iloc[0]
    q25, q50, q75, q90, q95 = np.quantile(r, [0.25, 0.5, 0.75, 0.9, 0.95])
    assert daily["residual_p90"] == pytest.approx(q90)
    assert daily["residual_p95"] == pytest.approx(q95)
    assert daily["residual_iqr"] == pytest.approx(q75 - q25)
    assert daily["residual_mad"] == pytest.approx(np.median(np.abs(r - q50)))
    assert daily["mean_positive_residual"] == pytest.approx(r[r > 0].mean())
    assert daily["n_issuers"] == 501


def test_zero_mean_residual_is_not_mistaken_for_no_stress():
    calm = np.linspace(-0.2, 0.2, 200)
    # Stress: a fifth of issuers blow out, the rest re-centre so the mean stays zero.
    stressed = np.r_[np.full(40, 0.8), np.full(160, -0.2)]
    daily = daily_residual_features(_issuer_frame({"2010-01-04": calm, "2010-01-05": stressed}))
    assert daily["residual_mean"].abs().max() < 1e-12
    assert daily["residual_p90"].iloc[1] > daily["residual_p90"].iloc[0] + 0.5
    assert daily["residual_std"].iloc[1] > daily["residual_std"].iloc[0]
    assert daily["mean_positive_residual"].iloc[1] > daily["mean_positive_residual"].iloc[0]
    # The IQR ignores a tail smaller than 25% of issuers; that is why P90 is reported too.
    assert daily["residual_iqr"].iloc[1] == pytest.approx(0.0)


def test_threshold_uses_only_earlier_days():
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2010-01-04", periods=THRESHOLD_MIN_DAYS + 20)
    data = {d: rng.normal(0, 0.2, 120) for d in days}
    base = expanding_threshold(_issuer_frame(data))
    assert base.iloc[:THRESHOLD_MIN_DAYS].isna().all()
    t = days[THRESHOLD_MIN_DAYS + 5]
    shocked = dict(data)
    for d in days[days >= t]:
        shocked[d] = data[d] + 3.0  # day t and later only
    changed = expanding_threshold(_issuer_frame(shocked))
    pd.testing.assert_series_equal(changed.loc[:t], base.loc[:t])
    assert changed.iloc[-1] > base.iloc[-1]
    prior = np.concatenate([data[d] for d in days[days < t]])
    assert base.loc[t] == pytest.approx(np.quantile(prior, 0.9), abs=2 * np.diff(HIST_EDGES)[0])


def test_positive_share_counts_issuers_above_historical_threshold():
    days = pd.bdate_range("2010-01-04", periods=THRESHOLD_MIN_DAYS + 1)
    data = {d: np.linspace(-1, 1, 101) for d in days[:-1]}
    data[days[-1]] = np.r_[np.full(30, 2.0), np.zeros(70)]
    daily = daily_residual_features(_issuer_frame(data))
    assert daily["residual_positive_share"].iloc[:-1].isna().all()
    assert daily["residual_positive_share"].iloc[-1] == pytest.approx(0.30)


def test_adding_future_days_does_not_change_past_features():
    frame, _ = synthetic_bond_days(n_days=6, noise=0.2, seed=4)
    full_res, _ = bond_residuals(prepare_bond_days(frame), "M1", implied=True)
    full = daily_residual_features(issuer_residuals(full_res))
    cut = pd.Timestamp(frame["trd_exctn_dt"].unique()[3])
    part_res, _ = bond_residuals(
        prepare_bond_days(frame.loc[frame["trd_exctn_dt"] <= cut]), "M1", implied=True
    )
    part = daily_residual_features(issuer_residuals(part_res))
    pd.testing.assert_frame_equal(part, full.loc[:cut])


def test_market_implied_notch_gap():
    frame, _ = synthetic_bond_days(n_days=1, n_issuers=150)
    # Issuer G0001 is rated A (bucket 6) but trades exactly at the BBB (bucket 9) level.
    a = frame["gvkey"].eq("G0001")
    frame.loc[a, "spread_bp"] *= np.exp(TRUE_EFFECT[9] - TRUE_EFFECT[6])
    res, _ = bond_residuals(prepare_bond_days(frame), "M1", implied=True)
    issuers = issuer_residuals(res).set_index("gvkey")
    assert issuers.loc["G0001", "implied_bucket"] == 9
    assert issuers.loc["G0001", "notch_gap"] == 3
    assert issuers.drop(index="G0001")["notch_gap"].eq(0).all()


def _synthetic_daily(n=400, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(pd.bdate_range("2006-01-09", periods=n), name="trd_exctn_dt")
    return pd.DataFrame(
        {
            "residual_p90": 0.3 + 0.01 * rng.standard_normal(n).cumsum(),
            "residual_iqr": 0.25 + 0.01 * rng.standard_normal(n),
            "residual_positive_share": 0.1 + 0.01 * rng.standard_normal(n),
        },
        index=idx,
    )


def test_signal_applies_release_lag_and_has_no_lookahead():
    daily = _synthetic_daily()
    signal = HYResidualStress(daily)
    out = signal.run("2007-06-29")
    first_obs = daily.index[0]
    assert out.index[0] > first_obs
    assert out.loc[out.index[0], "residual_p90"] == pytest.approx(daily["residual_p90"].iloc[0])
    assert_no_lookahead(signal, ["2006-08-31", "2007-01-31"], end_date="2007-06-29")


def test_past_sector_premium_uses_only_earlier_days():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2010-01-04", periods=40)
    effects = pd.DataFrame(rng.normal(0, 0.1, (40, 2)), index=idx, columns=["10", "40"])
    base, _ = past_sector_premium(effects, min_days=5, shrink_days=10)
    t = idx[20]
    shocked = effects.copy()
    shocked.loc[t:] += 1.0
    changed, _ = past_sector_premium(shocked, min_days=5, shrink_days=10)
    pd.testing.assert_frame_equal(changed.loc[:t], base.loc[:t])
    assert (changed.iloc[-1] > base.iloc[-1]).all()


def test_past_sector_premium_waits_for_history_and_shrinks():
    idx = pd.bdate_range("2010-01-04", periods=10)
    effects = pd.DataFrame({"10": 0.3, "40": np.nan}, index=idx)
    premium, n = past_sector_premium(effects, min_days=5, shrink_days=5)
    assert premium["10"].iloc[:5].eq(0).all()  # fewer than 5 earlier days
    assert premium["10"].iloc[5] == pytest.approx(0.3 * 5 / 10)
    assert premium["10"].iloc[9] == pytest.approx(0.3 * 9 / 14)
    assert premium["40"].eq(0).all()  # never estimated: pooled to the market (0)
    assert n["10"].iloc[0] == 0


def _sector_panel(n_days=8, shock=0.0):
    frame, sector = synthetic_bond_days(n_days=n_days, noise=0.02, seed=11)
    last = frame["trd_exctn_dt"].eq(frame["trd_exctn_dt"].max())
    energy = frame["gvkey"].map(sector).eq("Energy")
    frame.loc[last & energy, "spread_bp"] *= np.exp(shock)
    return prepare_bond_days(frame, sector)


def _sector_mean(res, prepared, day, sector="Energy"):
    issuers = issuer_residuals(res)
    issuers = issuers.loc[issuers["trd_exctn_dt"].eq(day)]
    labels = prepared.drop_duplicates("gvkey").set_index("gvkey")["sector"]
    mean = issuers.groupby(issuers["gvkey"].map(labels))["residual"].mean()
    return float(mean[sector])


def test_past_sector_model_removes_normal_premium_but_keeps_a_new_shock():
    prepared = _sector_panel(shock=0.5)
    _, same_day_stats = bond_residuals(prepared, "M2")
    # Only days before the shock define the "normal" premium (min 3 days, no shrinkage).
    premium, n = past_sector_premium(sector_effects(same_day_stats), min_days=3, shrink_days=0)
    with_offset = add_sector_offset(prepared, premium, n, min_days=3)
    days = sorted(prepared["trd_exctn_dt"].unique())
    m1, _ = bond_residuals(prepared, "M1")
    m2, _ = bond_residuals(prepared, "M2")
    m2p, _ = bond_residuals(with_offset, "M2P")
    calm = days[4]
    # Normal premium (Energy +0.2 vs Financials/Industrials) is in M1 but not in M2P.
    assert _sector_mean(m1, prepared, calm) > 0.1
    assert abs(_sector_mean(m2p, prepared, calm)) < 0.02
    # The new Energy repricing on the last day is absorbed by same-day M2, kept by M2P.
    assert abs(_sector_mean(m2, prepared, days[-1])) < 0.02
    assert _sector_mean(m2p, prepared, days[-1]) > 0.25
    assert production_start(with_offset, coverage=0.9) == days[3]


def test_same_day_sector_effects_are_centred():
    prepared = _sector_panel(n_days=1)
    fit = fit_day(prepared, "M2")
    issuers = prepared.drop_duplicates("gvkey")
    centred = issuers["sector"].map(fit["sector_effect"]).mean()
    assert centred == pytest.approx(0.0, abs=1e-9)
    gap = fit["sector_effect"]["Energy"] - fit["sector_effect"]["Industrials"]
    assert gap == pytest.approx(TRUE_SECTOR["Energy"] - TRUE_SECTOR["Industrials"], abs=0.02)
