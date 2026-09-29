"""numpy features newer than the lab server's numpy 1.15."""
import numpy as np

_BIG = np.finfo(float).max


def nan_to_num(x, nan=0.0):
    """``np.nan_to_num(x, nan=...)`` (the ``nan=`` keyword is numpy 1.17+):
    NaN -> ``nan``, +inf / -inf -> the largest / smallest float, as numpy
    does. The masks are taken before anything is replaced, so ``nan=-inf``
    gives -inf, as it does in numpy."""
    x = np.array(x, dtype=float, copy=True)
    is_nan, pos, neg = np.isnan(x), np.isposinf(x), np.isneginf(x)
    x[pos] = _BIG
    x[neg] = -_BIG
    x[is_nan] = nan
    return x
