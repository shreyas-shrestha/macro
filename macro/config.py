"""Constants and environment setup shared across the package."""

import pandas as pd
from dotenv import find_dotenv, load_dotenv

DEV_START = pd.Timestamp("2000-01-01")
DEV_END = pd.Timestamp("2017-12-31")
HOLDOUT_START = pd.Timestamp("2018-01-01")
# Fits start with the VIX history, which keeps the 1987 crash out of every fit.
ESTIMATION_START = pd.Timestamp("1990-01-02")


def load_env() -> bool:
    """Load variables from ``.env`` in the working directory or a parent.

    Existing environment variables are not overridden. Returns True if a file was found.
    """
    return load_dotenv(find_dotenv(usecwd=True))
