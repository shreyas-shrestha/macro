"""Macro Team research package: macro signals for S&P 500 volatility.

Subpackages: ``data`` (point-in-time loaders), ``features`` (signals), ``models``
(target and baselines), ``backtest`` (walk-forward evaluation), ``strategy``
(combination and vol-targeting). Shared constants live in ``macro.config``.

Timing convention: decision date t is an NYSE trading day. The row indexed t uses only
information published by 16:00 ET (market close) on t. A series observed on day s with
release lag L trading days enters at s + L, never earlier. The forecast made at t
targets realized variance over trading days t+1 ... t+h; day t's return is never part of
the target. Full statement in ``macro.features.base``.
"""
