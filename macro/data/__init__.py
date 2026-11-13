"""Data loaders and point-in-time alignment.

Read every series through ``macro.data.series.load``: it returns values on NYSE
sessions, each stamped by the session it became usable (see ``pit.align``), and refuses
holdout dates unless asked. One fetcher module per source; raw downloads are cached
under ``data/cache/`` (git-ignored). Release lags and fill rules live in
``series.SERIES`` and are documented in ``data/README.md``.
"""
