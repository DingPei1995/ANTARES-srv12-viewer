"""
compat -- what the lab server's Python 3.6 does not have.

``dataclasses``  the standard module from Python 3.7 onward; on 3.6 the
                 official backport (``dataclasses36.py``, Apache 2.0, by
                 Eric V. Smith) bundled here, since the server cannot
                 ``pip install``.
``pyfive``       a pure-Python HDF5 *reader* (BSD, Jonathan J. Helmus),
                 standing in for ``h5py``, which the server does not have.
                 Patched here (marked "patched" in the source) for
                 variable-length strings, compact storage, numpy dtypes,
                 and reading a large chunked dataset chunk by chunk as it
                 is sliced; see ``compat/h5.py`` for the h5py-like layer
                 the loaders use. Licence: ``pyfive/LICENSE.txt``.
``numpy_compat`` numpy features newer than numpy 1.15.
"""
