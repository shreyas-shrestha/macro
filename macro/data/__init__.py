"""Data loaders, one function per source.

Contract: every returned value is stamped with the date it became knowable (release
date or ALFRED vintage), not only the observation date. Per-series release lags are
applied here. Raw downloads are cached under ``data/cache/`` (git-ignored). Loaders
read credentials via ``macro.config.load_env()``. Filled by #8.
"""
