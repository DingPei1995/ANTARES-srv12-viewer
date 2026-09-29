"""
compat/h5.py
============
The part of ``h5py`` this program uses, for reading only, on top of the
bundled pure-Python reader :mod:`compat.pyfive`.

The lab server has no ``h5py``, and cannot install it. Everything the
loaders do with an HDF5 file is *read* it -- walk the groups, read a few
metadata datasets, slice a cube -- and ``pyfive`` does exactly that with
nothing but numpy. Writing is not offered: the program's own saved format is
an ``.npz`` file instead (see :mod:`loader.nxs_file`).

``h5py.File(path, "r")``, ``h5py.Group`` and ``h5py.Dataset`` are the three
names the loaders use, so this module provides those three.

Known limits (pyfive's): chunked datasets written in the HDF5 1.10+ layout
(``libver="latest"``) and filters other than gzip / shuffle / fletcher32
(LZF, LZ4, bitshuffle, blosc ...) are not readable. Files from the ANTARES
DataRecorder and from h5py with default settings are in the readable layout.
"""
import os

from compat.pyfive.high_level import File as _File, Group, Dataset   # noqa: F401

__all__ = ["File", "Group", "Dataset"]


class _Id:
    def __init__(self, handle):
        self._handle = handle

    @property
    def valid(self):
        fh = getattr(self._handle, "_fh", None)
        return fh is not None and not getattr(fh, "closed", True)


class File(_File):
    """``h5py.File`` for reading."""

    def __init__(self, path, mode="r"):
        if mode not in ("r", "rb"):
            raise OSError("HDF5 files can only be read here (no h5py on "
                          "this machine); mode %r refused" % (mode,))
        if not os.path.isfile(path):
            raise OSError("no such file: %s" % path)
        _File.__init__(self, path)

    @property
    def id(self):
        return _Id(self)

    def close(self):
        try:
            _File.close(self)
        except Exception:
            pass
    __del__ = close
