"""Local WRDS extract loaders for the high-yield credit signal (#24).

Raw WRDS data is licensed. Export CSVs from WRDS into ``data/cache/`` and load them
locally; never commit raw extracts to the public repository.

Every loader reads all columns as text first, so identifiers (CUSIP, GVKEY, codes) keep
their leading zeros, then converts the declared date and numeric columns. A non-blank
value that fails to parse raises instead of becoming missing. Missing required columns
raise with their names; missing optional columns are recorded in
``frame.attrs["missing_optional"]``. Rows dated in the holdout (from
``macro.config.HOLDOUT_START``) raise.

Transformations live in ``macro.data.trace`` and ``macro.data.ratings``.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from macro.config import HOLDOUT_START


@dataclass(frozen=True)
class ExtractSchema:
    """Expected columns of one WRDS extract and how to type them."""

    name: str
    required: frozenset[str]
    optional: frozenset[str] = frozenset()
    dates: frozenset[str] = frozenset()
    numerics: frozenset[str] = frozenset()
    holdout_date: str | None = None
    identifiers: frozenset[str] = field(default_factory=frozenset)


TRACE_SCHEMA = ExtractSchema(
    name="TRACE trades",
    required=frozenset({"trd_exctn_dt", "cusip_id", "rptd_pr", "yld_pt"}),
    optional=frozenset(
        {
            # Shared by the BTDS and Enhanced extracts.
            "bond_sym_id",
            "company_symbol",
            "trd_exctn_tm",
            "trc_st",
            "wis_fl",
            "asof_cd",
            "sale_cndtn_cd",
            "sale_cndtn2_cd",
            "spcl_trd_fl",
            "yld_sign_cd",
            "sttl_dt",
            # Disseminated BTDS layout.
            "ascii_rptd_vol_tx",
            "frmt_cd",
            "chng_cd",
            # Enhanced historic layout (lifecycle linking and dedup fields).
            "msg_seq_nb",
            "orig_msg_seq_nb",
            "trd_rpt_dt",
            "trd_rpt_tm",
            "entrd_vol_qt",
            "rpt_side_cd",
            "cntra_mp_id",
            "dissem_fl",
            "trdg_mkt_cd",
            "cmsn_trd",
            "scrty_type_cd",
            # WRDS trace_enhanced.trace_enhanced names (discovered 2026-11-07).
            "sub_prdct",
            "stlmnt_dt",
            "days_to_sttl_ct",
            "buy_cpcty_cd",
            "sell_cpcty_cd",
            "agu_qsr_id",
            "trd_mod_3",
            "trd_mod_4",
            "rptg_party_type",
            "lckd_in_ind",
            "ats_indicator",
            "pr_trd_dt",
            "pull_part",
        }
    ),
    dates=frozenset({"trd_exctn_dt", "trd_rpt_dt", "sttl_dt", "stlmnt_dt", "pr_trd_dt"}),
    numerics=frozenset({"rptd_pr", "yld_pt", "entrd_vol_qt", "msg_seq_nb", "orig_msg_seq_nb"}),
    holdout_date="trd_exctn_dt",
    identifiers=frozenset({"cusip_id"}),
)

MASTER_SCHEMA = ExtractSchema(
    name="TRACE master",
    required=frozenset({"cusip_id", "mtrty_dt"}),
    optional=frozenset(
        {
            "bond_sym_id",
            "company_symbol",
            "issuer_nm",
            "scrty_ds",
            "sub_prdct_type",
            "debt_type_cd",
            "scrty_type_cd",
            "scrty_sbtp_cd",
            "cpn_rt",
            "cpn_type_cd",
            "ind_144a",
            "cnvrb_fl",
            "grade",
            "sym_cd",
            "dissem",
            "stdt",
            "enddt",
        }
    ),
    dates=frozenset({"mtrty_dt", "stdt", "enddt"}),
    numerics=frozenset(),  # cpn_rt is varchar in WRDS; parse explicitly where needed
    identifiers=frozenset({"cusip_id"}),
)

RATINGS_SCHEMA = ExtractSchema(
    name="Compustat ratings",
    required=frozenset({"gvkey", "datadate", "splticrm"}),
    optional=frozenset(
        {"conm", "cusip", "tic", "gsector", "gind", "sic", "naics", "spsdrm", "spsticrm"}
    ),
    dates=frozenset({"datadate"}),
    holdout_date="datadate",
    identifiers=frozenset({"gvkey", "cusip"}),
)

# Backwards-compatible names.
TRACE_REQUIRED = set(TRACE_SCHEMA.required)
MASTER_REQUIRED = set(MASTER_SCHEMA.required)
RATINGS_REQUIRED = set(RATINGS_SCHEMA.required)


class ExtractError(ValueError):
    """A WRDS extract is missing columns, has unparseable values or touches the holdout."""


def validate_columns(frame: pd.DataFrame, schema: ExtractSchema) -> list[str]:
    """Raise naming any missing required column; return the missing optional columns."""
    missing = sorted(schema.required.difference(frame.columns))
    if missing:
        raise ExtractError(f"{schema.name}: missing required columns {missing}")
    return sorted(schema.optional.difference(frame.columns))


def _strict_parse(text: pd.Series, parsed: pd.Series, column: str, schema: ExtractSchema):
    failed = text.notna() & parsed.isna()
    if failed.any():
        examples = text[failed].unique()[:3].tolist()
        raise ExtractError(f"{schema.name}: unparseable {column} values, e.g. {examples}")
    return parsed


def normalize_extract(frame: pd.DataFrame, schema: ExtractSchema) -> pd.DataFrame:
    """Type an all-text frame per ``schema``; see the module docstring for the rules."""
    missing_optional = validate_columns(frame, schema)
    out = frame.copy()
    for column in out.columns:
        text = out[column].astype("string").str.strip()
        text = text.mask(text.eq(""))
        if column in schema.dates:
            out[column] = _strict_parse(text, pd.to_datetime(text, errors="coerce"), column, schema)
        elif column in schema.numerics:
            out[column] = _strict_parse(
                text, pd.to_numeric(text, errors="coerce"), column, schema
            ).astype("float64")
        else:
            out[column] = text
            if column in schema.identifiers:
                out[column] = text.str.upper()
    if schema.holdout_date is not None:
        late = out[schema.holdout_date].ge(HOLDOUT_START).fillna(False)
        if late.any():
            raise ExtractError(
                f"{schema.name}: {int(late.sum())} rows dated on or after "
                f"{HOLDOUT_START.date()} (locked holdout)"
            )
    out.attrs["missing_optional"] = missing_optional
    return out


def read_extract(path: str | Path, schema: ExtractSchema) -> pd.DataFrame:
    """Read a WRDS CSV export as text and normalize it per ``schema``."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return normalize_extract(frame, schema)


def load_trace_trades(path: str | Path) -> pd.DataFrame:
    """Load a WRDS TRACE trade export (BTDS or Enhanced)."""
    return read_extract(path, TRACE_SCHEMA)


def load_trace_master(path: str | Path) -> pd.DataFrame:
    """Load the TRACE corporate-bond master file export."""
    return read_extract(path, MASTER_SCHEMA)


def load_compustat_ratings(path: str | Path) -> pd.DataFrame:
    """Load the Legacy Compustat North America Ratings export."""
    return read_extract(path, RATINGS_SCHEMA)
