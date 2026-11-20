"""End-to-end synthetic test of the HY dataset pipeline (no WRDS rows)."""

import numpy as np
import pandas as pd

from macro.data.hy_dataset import (
    Attrition,
    add_master_and_spreads,
    attach_issuers,
    attach_ratings,
    bond_universe_mask,
    clean_trades,
    daily_aggregates,
    link_matches,
    trades_to_bond_days,
)

DAY = pd.Timestamp("2010-01-05")


def _raw():
    rows = []
    seq = 1
    # Bonds: A1, A2 (issuer A, HY), B1 (issuer B, IG), C1 (conflicting links), D1 (convertible).
    for cusip, yields in {
        "AAAAAA101": [9.0, 9.2],
        "AAAAAA119": [10.0],
        "BBBBBB105": [4.0],
        "CCCCCC103": [6.0],
        "DDDDDD109": [5.0],
    }.items():
        for y in yields:
            rows.append(
                {
                    "cusip_id": cusip,
                    "trd_exctn_dt": DAY,
                    "trd_exctn_tm": f"10{seq:04d}",
                    "trd_rpt_dt": DAY,
                    "msg_seq_nb": str(seq),
                    "orig_msg_seq_nb": None,
                    "pr_trd_dt": pd.NaT,
                    "trc_st": "T",
                    "asof_cd": None,
                    "entrd_vol_qt": 100.0,
                    "rptd_pr": 100.0,
                    "yld_pt": y,
                    "yld_sign_cd": "+",
                    "rpt_side_cd": "S",
                    "cntra_mp_id": "C",
                    "dissem_fl": "Y",
                    "wis_fl": "N",
                }
            )
            seq += 1
    raw = pd.DataFrame(rows)
    # A cancelled trade and a non-disseminated inter-dealer buy that must not count.
    cancelled = raw.iloc[[0]].assign(msg_seq_nb="90", yld_pt=50.0)
    cancel = raw.iloc[[0]].assign(msg_seq_nb="91", trc_st="C", orig_msg_seq_nb="90")
    buy = raw.iloc[[0]].assign(
        msg_seq_nb="92", yld_pt=40.0, rpt_side_cd="B", cntra_mp_id="D", dissem_fl="N"
    )
    return pd.concat([raw, cancelled, cancel, buy], ignore_index=True)


def _master():
    base = {
        "sub_prdct_type": "CORP",
        "cnvrb_fl": "N",
        "cpn_type_cd": "FXPV",
        "ind_144a": "N",
        "mtrty_dt": pd.Timestamp("2015-01-05"),
        "stdt": pd.Timestamp("2000-01-01"),
        "enddt": pd.NaT,
    }
    rows = [dict(base, cusip_id=c) for c in ("AAAAAA101", "AAAAAA119", "BBBBBB105", "CCCCCC103")]
    rows.append(dict(base, cusip_id="DDDDDD109", cnvrb_fl="Y"))
    return pd.DataFrame(rows)


def _curve():
    return pd.DataFrame({1.0: [1.0], 10.0: [3.0]}, index=pd.DatetimeIndex([DAY]))


def test_pipeline_end_to_end():
    log = Attrition()
    trades = clean_trades(_raw(), log)
    assert len(trades) == 6  # cancel pair and inter-dealer buy removed
    bond_days = trades_to_bond_days(trades, log)
    spreads = add_master_and_spreads(bond_days, _master(), _curve(), log)
    assert "DDDDDD109" not in set(spreads["cusip_id"])  # convertible excluded
    a1 = spreads.set_index("cusip_id").loc["AAAAAA101"]
    # 5-year remaining maturity: Treasury 1 + (3-1)*4/9; yield is the median of 9.0 and 9.2.
    assert a1["spread_bp"] == np.round(100 * (9.1 - (1 + 2 * (1826 / 365.25 - 1) / 9)), 6) or (
        abs(a1["spread_bp"] - 100 * (9.1 - (1 + 2 * (1826 / 365.25 - 1) / 9))) < 1e-6
    )

    fang = pd.DataFrame(
        {
            "issuer_cusip": ["AAAAAA", "BBBBBB", "CCCCCC"],
            "gvkey": ["001", "002", "003"],
            "ldom": pd.to_datetime(["2009-12-31"] * 3),
            "permno": [1.0, 2.0, 3.0],
        }
    )
    bondcrsp = pd.DataFrame(
        {
            "cusip": ["CCCCCC103"],
            "permno": [30],
            "link_startdt": pd.to_datetime(["2000-01-01"]),
            "link_enddt": pd.to_datetime(["2020-01-01"]),
        }
    )
    ccm = pd.DataFrame(
        {
            "gvkey": ["009"],
            "lpermno": [30],
            "linktype": ["LU"],
            "linkprim": ["P"],
            "linkdt": pd.to_datetime(["1990-01-01"]),
            "linkenddt": [pd.NaT],
        }
    )
    matches = link_matches(spreads, fang=fang, bondcrsp=bondcrsp, ccm=ccm)
    linked = attach_issuers(spreads, matches, log)
    assert set(linked["gvkey"]) == {"001", "002"}  # C1 has conflicting GVKEYs: unlinked

    ratings = pd.DataFrame(
        {
            "gvkey": ["001", "002", "001"],
            "datadate": pd.to_datetime(["2009-12-31", "2009-12-31", "2010-01-31"]),
            "splticrm": ["B+", "A-", "CCC"],  # A's January downgrade must not leak back
        }
    )
    rated = attach_ratings(linked, ratings, log)
    assert set(rated.loc[rated["gvkey"].eq("001"), "rating"]) == {"B+"}

    issuer, daily = daily_aggregates(rated, log, min_issuers=1)
    row = daily.loc[DAY]
    assert row["n_hy_issuers"] == 1 and row["n_ig_issuers"] == 1
    a_spreads = rated.loc[rated["gvkey"].eq("001"), "spread_bp"]
    assert row["hy_spread_bp"] == a_spreads.median()  # issuer A counted once
    stages = log.frame()["stage"].tolist()
    assert stages[0] == "raw_records" and stages[-1] == "daily"


def test_min_issuer_gate_blanks_thin_days():
    log = Attrition()
    rated = pd.DataFrame(
        {
            "trd_exctn_dt": [DAY] * 2,
            "gvkey": ["001", "002"],
            "cusip_id": ["X1", "X2"],
            "rating_class": ["HY", "IG"],
            "spread_bp": [500.0, 100.0],
        }
    )
    _, daily = daily_aggregates(rated, log, min_issuers=2)
    assert daily["hy_spread_bp"].isna().all()
    assert daily["hy_ig_gap_bp"].isna().all()


def test_bond_universe_uses_documented_codes():
    frame = pd.DataFrame(
        {
            "sub_prdct_type": ["CORP", "ELN", "CORP", "CORP", "CORP", None],
            "cnvrb_fl": ["N", "N", "Y", None, "N", "N"],
            "cpn_type_cd": ["FXPV", "FXPV", "FXPV", "FROT", "FXPV", "FXPV"],
            "ind_144a": ["N", "N", "N", "N", "Y", "N"],
        }
    )
    keep, excluded = bond_universe_mask(frame)
    assert keep.tolist() == [True, False, False, False, False, False]
    assert excluded == {"not_corp": 2, "convertible": 1, "not_plain_fixed": 1, "rule_144a": 1}


def test_build_development_runner_on_synthetic_chunks(tmp_path):
    from macro.data.hy_runs import build_development
    from macro.data.treasury import CMT_SERIES

    work = tmp_path / "hy_credit"
    (work / "trace_raw").mkdir(parents=True)
    raw = _raw()
    raw["trd_exctn_dt"] = pd.Timestamp("2010-01-05")
    raw.to_parquet(work / "trace_raw" / "2010-01.parquet", index=False)
    _master().to_parquet(work / "master.parquet", index=False)
    pd.DataFrame(
        {
            "issuer_cusip": ["AAAAAA", "BBBBBB"],
            "gvkey": ["001", "002"],
            "permno": [1.0, 2.0],
            "ldom": pd.to_datetime(["2009-12-31"] * 2),
        }
    ).to_parquet(work / "fang.parquet", index=False)
    pd.DataFrame(
        {
            "gvkey": ["001", "002"],
            "datadate": pd.to_datetime(["2009-12-31"] * 2),
            "splticrm": ["B", "A"],
        }
    ).to_parquet(work / "ratings.parquet", index=False)
    fred = tmp_path / "fred"
    fred.mkdir()
    for series, maturity in CMT_SERIES.items():
        (fred / f"{series}.csv").write_text(
            f"observation_date,{series}\n2010-01-05,{maturity / 10}\n"
        )

    daily = build_development(cache=tmp_path, out_dir=tmp_path / "results")
    assert daily.loc[DAY, "n_hy_issuers"] == 1
    assert (work / "hy_daily.parquet").exists()
    attrition = pd.read_csv(tmp_path / "results" / "attrition.csv")
    assert {"raw_records", "after_issuer_link", "daily"} <= set(attrition["stage"])


def test_crsp_ncusip6_and_comp_cusip6_links():
    bond_days = pd.DataFrame({"cusip_id": ["SUBSID101", "PARENT105"], "trd_exctn_dt": [DAY, DAY]})
    security = pd.DataFrame({"gvkey": ["010"], "cusip": ["PARENT999"]})
    bondcrsp = pd.DataFrame(
        {
            "cusip": ["SUBSID101"],
            "permno": [77],
            "link_startdt": pd.to_datetime(["2000-01-01"]),
            "link_enddt": pd.to_datetime(["2012-01-01"]),
        }
    )
    names = pd.DataFrame(
        {
            "permno": [77, 77],
            "ncusip": ["OLDCUS10", "PARENT10"],
            "namedt": pd.to_datetime(["1990-01-01", "2009-06-01"]),
            "nameenddt": pd.to_datetime(["2009-05-31", "2015-01-01"]),
        }
    )
    m = link_matches(bond_days, bondcrsp=bondcrsp, crsp_names=names, security=security)
    got = set(map(tuple, m[["cusip_id", "gvkey", "method"]].astype(str).values))
    # Subsidiary bond reaches the parent through PERMNO and the name valid on the day.
    assert ("SUBSID101", "010", "crsp_ncusip6") in got
    assert ("PARENT105", "010", "comp_cusip6") in got
    assert len(got) == 2


def test_server_safe_prefilter_keeps_lifecycle_records():
    from macro.data.hy_runs import server_safe_dissem_mask

    raw = pd.DataFrame(
        {
            "trc_st": ["T", "T", "C", "W", "T", "X"],
            "dissem_fl": ["Y", "N", "N", "N", "N", "N"],
            "asof_cd": [None, None, None, None, "R", None],
        }
    )
    assert server_safe_dissem_mask(raw).tolist() == [True, False, True, True, True, True]
