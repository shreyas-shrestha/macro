"""Walk-forward evaluation harness.

Contract: refits models walk-forward, scores forecasts with QLIKE and Clark-West tests,
and writes reports to ``results/``. Refuses any date after ``config.DEV_END``
(2017-12-31) unless run with ``--holdout``. Filled by #10.
"""
