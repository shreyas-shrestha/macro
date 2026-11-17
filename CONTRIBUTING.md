# Contributing

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/shreyas-shrestha/macro.git
cd macro
uv sync                         # creates .venv with Python 3.12 and pinned dependencies
cp .env.example .env            # add your FRED API key
uv run pre-commit install       # ruff, ruff-format and nbstripout on every commit
uv run pytest
uv run python data/data_check.py
```

The data check downloads and prints only the development sample (2000 through 2017).

On macOS, if downloads fail with certificate errors, set `SSL_CERT_FILE` in `.env` to the output of `uv run python -m certifi`.

## Workflow

- One issue, one branch, one PR, one review. Branch name: `<issue#>-short-name`, e.g. `11-credit-spreads`.
- The PR description says `Closes #<issue>` and completes the checklist in the template.
- `main` is protected: CI (`ruff` + `pytest`) must pass and one approving review is required.
- Code goes in `macro/`. Notebooks are for exploration only; outputs are stripped on commit and CI rejects notebooks with outputs. Anything a result depends on must be in `macro/`.

## Sample periods

- Development: 2000-01-01 to 2017-12-31 (`macro.config.DEV_START`, `DEV_END`).
- Holdout: 2018-01-01 onward (`HOLDOUT_START`). Locked. Do not load, plot or score it. It is evaluated once, at the end of the semester (#20).

## Timing convention

- Decision date *t* is an NYSE trading day. The row indexed *t* may use only information published by 16:00 ET (market close) on *t*.
- A series observed on day *s* with release lag *L* trading days enters the row at *s + L*, never earlier.
- The forecast made at *t* targets realized variance over trading days *t+1 … t+h*. Day *t*'s return is never part of the target.

Example: BAA10Y for day *s* is published by FRED the next morning, so *L* = 1 and it is first usable at *s + 1*. The VIX close on *t* has *L* = 0 and is usable at *t*, although it is struck at 16:15 ET: a deliberate exception, explained in `data/README.md`.

## Writing a signal

Create `macro/features/<name>.py` with a subclass of `macro.features.base.Signal`:

```python
import pandas as pd

from macro.features.base import Signal


class Credit(Signal):
    name = "credit"
    source = "FRED"
    start_date = pd.Timestamp("2000-01-03")
    release_lag = 1  # worst case across inputs, in trading days

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError
```

- Implement `build` only. The harness calls `run`, which validates the output (index, sorting, NYSE sessions, date range, dtypes).
- Use `macro.data.calendar.nyse_sessions(start_date, end_date)` for dates. `pd.bdate_range` includes exchange holidays and is rejected.
- Fit anything estimated (means, z-scores, regressions) on past data only.
- The interface is version 0 and provisional until #8 and #10 merge. Depend only on `build` and the four attributes.

Every signal has a lookahead test in `tests/test_<name>.py`:

```python
from macro.features.testing import assert_no_lookahead


def test_no_lookahead():
    assert_no_lookahead(Credit(), cutoffs=["2005-06-30", "2010-12-31"])
```

It rebuilds the signal at each cutoff and checks the rows match the full build. Tests must not hit the network; use cached or synthetic data.

## Dependencies

- Add with `uv add <package>` (or `uv add --dev <package>` for tools). Commit `pyproject.toml` and `uv.lock` together.
- On a `uv.lock` conflict, rebase onto `main`, take `main`'s `uv.lock`, re-run `uv lock`, and commit the result. Do not hand-edit the lock file.

## Definition of done

The checklist in `.github/pull_request_template.md`, plus one approving review.
