"""Forecast target and baseline forecasters.

Contract: defines the target (realized variance over t+1 ... t+h for h in {5, 21, 63})
and the baselines every signal is measured against: RV, HAR-RV, GARCH and VIX. Models
are fit only on data available at the decision date. Filled by #9.
"""
