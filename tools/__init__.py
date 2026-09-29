"""
tools -- the algorithms, with no GUI in any of them.

Everything a viewer does to data rather than with widgets: momentum
conversion, cuts and corrections, curve fitting and dispersion, the
Brillouin-zone geometry, processing, de-gridding, spin analysis. Each is
usable from a plain script, which is why they are kept apart from ``ui``.
Needs numpy and scipy only (Python 3.6, numpy 1.15, scipy 1.5 are enough).
"""
