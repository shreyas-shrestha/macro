"""Signals, one module per signal, each a ``Signal`` subclass (see ``base``).

Contract: output is indexed by decision date and no row uses information published
after that date's market close. Every signal ships with an ``assert_no_lookahead`` test
(see ``testing``). Filled by #11-#17.
"""
