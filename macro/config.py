"""Constants and environment setup shared across the package."""

import pandas as pd
from dotenv import find_dotenv, load_dotenv

DEV_START = pd.Timestamp("2000-01-01")
DEV_END = pd.Timestamp("2017-12-31")
HOLDOUT_START = pd.Timestamp("2018-01-01")

DECISION_TIMING = (
    "Decision date t is an NYSE trading day. The row indexed t uses only information "
    "published by 16:00 ET on t. A series observed on day s with release lag L trading "
    "days enters at s + L. The forecast at t targets realized variance over t+1 ... t+h."
)


def load_env() -> bool:
    """Load variables from ``.env`` in the working directory or a parent.

    Existing environment variables are not overridden. Returns True if a file was found.
    """
    return load_dotenv(find_dotenv(usecwd=True))
