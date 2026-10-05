"""
ui/data.py
==========
The dataset objects every viewer works on -- no GUI in here.

:class:`NxsData` is a dataset read from a file (through the loader
registry), :class:`MemoryData` one computed in this session; both expose the
same attributes (``kind``, ``scan``, ``cut_frame``, ``angle_cube`` ...), so a
computed dataset is displayed, processed and saved exactly like a measured
one. Also the display helpers (smoothing / upsampling for a coarse scan).

Moved here unchanged from the PyQt5 version's ``ui/widgets.py``.
"""
import os
import warnings

import numpy as np

from loader.nxs_file import (load_soleil_nxs, to_kspace_cube, NxsScan,
                             CUBE_KINDS)

def _convolve1d(array, kernel, axis):
    """Separable 1D convolution with edge padding, using only numpy (scipy
    is not a dependency of this app)."""
    pad = len(kernel) // 2
    widths = [(pad, pad) if i == axis else (0, 0) for i in range(array.ndim)]
    padded = np.pad(array, widths, mode="edge")
    out = np.zeros_like(array, dtype=float)
    for offset, weight in enumerate(kernel):
        sl = [slice(None)] * array.ndim
        sl[axis] = slice(offset, offset + array.shape[axis])
        out += weight * padded[tuple(sl)]
    return out


#: Rows of a spatial scan are streamed in bands of at most about this many
#: bytes (as float64), whatever the file's chunking says.
STREAM_BAND_BYTES = 64 * 1024 * 1024

#: A displayed image is resampled up to about this many samples along each
#: axis before it is drawn, which is what turns a coarse scan from a wall of
#: rectangles into a continuous picture.
INTERP_TARGET = 500
#: ... but never by more than this, so a tiny array cannot blow up into a
#: huge texture.
MAX_INTERP_FACTOR = 16


def interp_factor(n, target=INTERP_TARGET):
    """How many display samples to put between each pair of data points.

    Data that is already dense gets 1 (no resampling): there is nothing to
    gain from interpolating a 500-point axis onto 500 points.
    """
    if n < 2:
        return 1
    return int(np.clip(int(np.ceil(target / n)), 1, MAX_INTERP_FACTOR))


def _upsample_axis(array, factor, axis):
    """Linear interpolation along one axis onto ``factor`` times as many
    samples, covering exactly the same extent.

    Sample j of the output sits at index ``-0.5 + (j + 0.5)/factor`` of the
    input, i.e. the output pixels tile the input's outer bounds (each input
    pixel is centred on its axis value and spans half a step either side).
    That keeps the image in exactly the same place on the axes, so only the
    ``scale`` passed to setImage changes.
    """
    if factor <= 1:
        return array
    n = array.shape[axis]
    pos = np.clip(-0.5 + (np.arange(n * factor) + 0.5) / factor, 0, n - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (pos - i0).reshape([-1 if d == axis else 1 for d in range(array.ndim)])
    return np.take(array, i0, axis=axis) * (1 - w) + np.take(array, i1, axis=axis) * w


def upsample2d(array, factors):
    """Bilinear resampling of a 2D array, NaN-aware.

    This is what MATLAB's ``pcolor(...); shading interp`` does for the same
    data, and it is the reason the MATLAB plots look continuous where a
    plain image of the same scan looks like a mosaic: a deflector map is
    typically only a few tens of points across, so each data point covers a
    big block of screen. Blurring alone cannot fix that -- a Gaussian on the
    coarse grid just makes softer *blocks* -- so the array is resampled onto
    a finer grid first.

    Missing points are carried by the same weights as the values, so a NaN
    pulls in its neighbours instead of spreading a hole the size of the
    interpolation kernel.
    """
    fy, fx = (int(max(1, f)) for f in factors)
    if fy <= 1 and fx <= 1:
        return array
    data = np.asarray(array, dtype=float)
    finite = np.isfinite(data)
    if finite.all():
        out = _upsample_axis(_upsample_axis(data, fy, 0), fx, 1)
        return out
    filled = np.where(finite, data, 0.0)
    num = _upsample_axis(_upsample_axis(filled, fy, 0), fx, 1)
    den = _upsample_axis(_upsample_axis(finite.astype(float), fy, 0), fx, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 1e-9, out, np.nan)


def smooth2d(array, sigma=1.0):
    """Gaussian-smooth a 2D array for *display*, to take the staircase edges
    off sparsely-sampled scans.

    NaN-aware: missing points are excluded from both the weighted sum and
    its normalisation, so a gap pulls in its neighbours' values instead of
    poisoning the whole neighbourhood (plain convolution would spread the
    NaN). Pixels with no finite neighbour at all stay NaN.

    This never touches stored data -- callers keep the raw array and smooth
    only what they hand to setImage, so exports stay unsmoothed.
    """
    data = np.asarray(array, dtype=float)
    if sigma <= 0 or data.ndim != 2:
        return data
    radius = max(1, int(np.ceil(2 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    finite = np.isfinite(data)
    filled = np.where(finite, data, 0.0)
    numerator = _convolve1d(_convolve1d(filled, kernel, 0), kernel, 1)
    denominator = _convolve1d(_convolve1d(finite.astype(float), kernel, 0), kernel, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = numerator / denominator
    return np.where(denominator > 0, out, np.nan)


# --------------------------------------------------------------------------
# Data wrapper
# --------------------------------------------------------------------------
def _options_key(options):
    """A hashable identity for a set of load options, for the shared-handle
    registry. None and the default options are the same thing."""
    if options is None:
        return None
    key = (getattr(options, "loader", None),
           tuple(options.permutation) if getattr(options, "permutation", None) else None,
           getattr(options, "axis0_role", "angle") or "angle",
           getattr(options, "axis0_label", "") or "")
    return None if key == (None, None, "angle", "") else key


class NxsData:
    """Loads one data file through the loader registry (:mod:`loader.registry`)
    and exposes the GUI-friendly derived views on top of it. ``self.scan`` is
    the raw :class:`~loader.nxs_file.NxsScan`."""

    #: path -> the single open NxsData for it (see acquire()).
    _open_files: 'dict' = {}

    def __init__(self, nxs_path, entry=None, options=None,
                 progress=None):
        self.path = nxs_path
        self.options = options
        # Through the loader registry rather than straight to the SOLEIL
        # reader: which beamline wrote this, and the axis choices made at
        # load time, are settings now (see loader.registry / ui.loader_dialog).
        # Falling back to the old direct call keeps a file opening even if
        # no loader recognises it but the SOLEIL reader can cope.
        from loader import registry
        try:
            self.scan: NxsScan = registry.load(nxs_path, entry=entry,
                                                  options=options,
                                                  progress=progress)
        except ValueError:
            if options is not None and getattr(options, "loader", None):
                raise
            self.scan = load_soleil_nxs(nxs_path, entry=entry)
        self.kind = self.scan.kind
        self._overview = None
        self._key = None
        self._refs = 1

    @classmethod
    def acquire(cls, nxs_path, entry=None, options=None,
                progress=None):
        """Get the shared NxsData for a file, opening it if needed.

        HDF5 keeps one underlying handle per file per process, so two
        independently opened h5py.File objects for the same path are *not*
        independent -- closing either one invalidates both, and a second
        window on the same scan would kill the first window's lazy reads.
        Callers therefore share one instance and each release it with
        :meth:`close`; the file is only really closed when the last user has
        let go.

        The load options are part of the identity: the same file read with
        its axes in a different order is a different dataset, and handing
        back the cached one would quietly ignore what was asked for.
        """
        key = (os.path.abspath(nxs_path), entry, _options_key(options))
        existing = cls._open_files.get(key)
        if existing is not None and existing.alive():
            existing._refs += 1
            return existing
        if existing is not None:
            # Its file handle died under it. Handing it out again is the
            # "identifier is not of specified type" crash on the first read;
            # drop it and open the file afresh instead.
            cls._open_files.pop(key, None)
            existing._key = None
        data = cls(nxs_path, entry, options=options, progress=progress)
        data._key = key
        cls._open_files[key] = data
        return data

    def retain(self):
        """Take one more reference (released by :meth:`close`) and return
        self. For a holder that did not get this object from
        :meth:`acquire` -- the memory budget handing out what it caches --
        so that every holder owns exactly one reference and closes exactly
        once."""
        self._refs += 1
        return self

    def alive(self):
        """Whether this dataset can still be read: an in-memory one always
        can; a lazy one only while its HDF5 dataset is open."""
        for value in (getattr(self.scan, "value", None),
                      getattr(self.scan, "value4d", None)):
            dset = getattr(value, "_dset", None)
            if dset is not None:
                try:
                    handle = getattr(dset, "file", None)
                    if handle is not None and hasattr(handle, "id") \
                            and not handle.id.valid:
                        return False
                except Exception:                           # noqa: BLE001
                    return False
        return self._refs > 0

    def close(self):
        """Release one reference; the file closes when the last one goes,
        and an array read into memory (:meth:`load_into_memory`) is
        dropped with it."""
        self._refs -= 1
        if self._refs > 0:
            return
        if self._key is not None:
            NxsData._open_files.pop(self._key, None)
            self._key = None
        self.scan.close()
        if getattr(self, "_loaded", False):
            self.scan.value = None
            self._loaded = False
        self._overview = None

    # -- reading the whole dataset ----------------------------------------
    def in_memory(self):
        """Whether slicing this dataset no longer touches the file. A
        spatial scan (spem_4d) is always left in its file: it is far too
        big, and it is read a spectrum at a time."""
        if self.kind == "spem_4d":
            return True
        value = self.scan.value
        return value is None or (isinstance(value, np.ndarray)
                                 and not isinstance(value, np.memmap))

    #: Never read a dataset whole above this, whatever memory is free.
    MAX_IN_MEMORY_BYTES = 4 * 1024 ** 3

    def fits_in_memory(self):
        """Whether reading the whole array is reasonable: at most half of
        the memory the server has available, and at most
        :data:`MAX_IN_MEMORY_BYTES`. A bigger one stays in its file and is
        sliced from there, as before (through the reader's chunk cache)."""
        value = self.scan.value
        size = getattr(value, "nbytes", None)
        if size is None:
            return True
        limit = self.MAX_IN_MEMORY_BYTES
        try:
            from tools import memory as M
            available = M.system_memory().get("available")
            if available:
                limit = min(limit, available // 2)
        except Exception:                                   # noqa: BLE001
            pass
        return int(size) <= limit

    def read_into_memory(self, progress=None):
        """The whole array, read from the file (``progress(done, total)``).
        Safe to run on a worker thread; hand the result to
        :meth:`keep_in_memory` on the GUI thread."""
        value = self.scan.value
        if hasattr(value, "materialise"):
            try:
                return np.asarray(value.materialise(progress=progress))
            except TypeError:                       # no progress argument
                return np.asarray(value.materialise())
        return np.array(value)

    def keep_in_memory(self, array):
        """Use ``array`` (from :meth:`read_into_memory`) instead of the
        file: every slice after this is a numpy slice, and the file is let
        go."""
        if self.in_memory():
            return
        self.scan.value = array
        self._loaded = True
        if self.scan.value4d is None and not self.scan.previews:
            self.scan.close()

    def load_into_memory(self, progress=None):
        """:meth:`read_into_memory` and :meth:`keep_in_memory` in one go."""
        if not self.in_memory():
            self.keep_in_memory(self.read_into_memory(progress))

    # -- spem_4d --------------------------------------------------------
    def _overview_from_preview(self):
        """Spatial overview from one of the file's reduced preview cubes.

        ``data_11`` is the cube already summed over energy and ``data_01``
        over the slit angle, so summing either one's last axis gives the same
        map as summing the whole cube -- at a thousandth of the data volume,
        which is the difference between opening a scan instantly and reading
        a gigabyte off a network share.

        The result is checked against the real cube on a few pixels before
        being trusted; returns None if no preview is present or the check
        fails, and the caller streams the cube instead.
        """
        for name in ("data_11", "data_01"):
            entry = self.scan.previews.get(name)
            if entry is None:
                continue
            dset, perm = entry
            try:
                preview = np.transpose(dset[()], perm)       # (y, x, other)
                overview = preview.sum(axis=2, dtype=np.float64)
            except Exception as exc:
                warnings.warn(f"{name}: could not be read as a preview ({exc})")
                continue

            ny, nx = overview.shape
            samples = [(0, 0), (ny // 2, nx // 2), (ny - 1, nx - 1)]
            ok = True
            for yi, xi in samples:
                true_sum = float(np.asarray(self.scan.value4d[yi, xi], dtype=np.float64).sum())
                claimed = float(overview[yi, xi])
                scale = max(abs(true_sum), abs(claimed), 1.0)
                if abs(true_sum - claimed) / scale > 1e-3:
                    ok = False
                    break
            if ok:
                return overview
            warnings.warn(
                f"{name}: does not reproduce the cube's own sums, so it is not "
                f"a plain projection in this file; falling back to reading the "
                f"cube for the spatial overview.")
        return None

    def _row_bands(self, rows):
        """``rows`` (a range of spatial rows) cut into bands that follow the
        file's chunking, so each compressed chunk is inflated once per
        pass rather than once per row. Bands are kept to about
        :data:`STREAM_BAND_BYTES` so memory stays bounded."""
        cube = self.scan.value4d
        block = int(getattr(cube, "row_block", 1) or 1)
        per_row = max(1, int(np.prod(cube.shape[1:])) * 8)
        block = max(1, min(block, int(STREAM_BAND_BYTES // per_row) or 1))
        rows = list(rows)
        if not rows:
            return
        band = [rows[0]]
        for r in rows[1:]:
            if r == band[-1] + 1 and len(band) < block and \
                    (r // block) == (band[0] // block):
                band.append(r)
            else:
                yield slice(band[0], band[-1] + 1)
                band = [r]
        yield slice(band[0], band[-1] + 1)

    def _overview_by_streaming(self, progress=None):
        """Fallback: sum the cube one band of spatial rows at a time, so
        peak memory stays at one band rather than the whole gigabyte."""
        ny, nx = self.scan.value4d.shape[:2]
        overview = np.empty((ny, nx), dtype=np.float64)
        for band in self._row_bands(range(ny)):
            rows = np.asarray(self.scan.value4d[band], dtype=np.float64)  # (y, x, k, E)
            overview[band] = rows.sum(axis=(2, 3))
            if progress is not None:
                progress(band.stop, ny)
        return overview

    # -- spem_4d --------------------------------------------------------
    def spatial_overview_computed(self, progress=None):
        """(y, x) map, summed over (k, E). Uses the file's reduced preview
        cube when it checks out, and otherwise streams the real cube."""
        assert self.kind == "spem_4d"
        if self._overview is None:
            overview = self._overview_from_preview()
            if overview is None:
                overview = self._overview_by_streaming(progress)
            self._overview = overview
        return self._overview

    @property
    def spatial_overview(self):
        return self.spatial_overview_computed()

    def frame_at(self, yi, xi):
        """(k, E) spectrum at spatial pixel (yi, xi)."""
        assert self.kind == "spem_4d"
        return self.scan.value4d[yi, xi]

    def frame_over_region(self, yslice, xslice):
        """(k, E) spectrum summed over a rectangular spatial region, read one
        spatial row at a time so a large selection never has to fit in
        memory all at once."""
        assert self.kind == "spem_4d"
        start, stop, step = yslice.indices(self.scan.value4d.shape[0])
        total = None
        if step != 1:
            bands = [slice(yi, yi + 1) for yi in range(start, stop, step)]
        else:
            bands = self._row_bands(range(start, stop))
        for band in bands:
            block = np.asarray(self.scan.value4d[band, xslice], dtype=np.float64)
            block = block.sum(axis=(0, 1))
            total = block if total is None else total + block
        return total

    def spatial_over_region(self, kslice, eslice, progress=None):
        """(y, x) map summed over a rectangular (k, E) region -- the inverse
        selection: pick a feature in the spectrum, see where on the sample it
        comes from.

        This one genuinely has to touch every spatial pixel, but only inside
        the selected (k, E) window, and it reads a row at a time.
        """
        assert self.kind == "spem_4d"
        ny, nx = self.scan.value4d.shape[:2]
        out = np.empty((ny, nx), dtype=np.float64)
        for band in self._row_bands(range(ny)):
            block = np.asarray(self.scan.value4d[band, :, kslice, eslice],
                               dtype=np.float64)
            out[band] = block.sum(axis=(2, 3))
            if progress is not None:
                progress(band.stop, ny)
        return out

    # -- spem_1d ----------------------------------------------------------
    @property
    def line_kmap(self):
        """(x, k) map, summed over E, for a real-space *line* scan."""
        assert self.kind == "spem_1d"
        # np.asarray rather than .sum() straight off scan.value: a scan's
        # array may be a LazyCube/LazyArray, which forwards indexing and
        # np.asarray but not ndarray's own methods.
        return np.asarray(self.scan.value).sum(axis=2)

    def frame_at_x(self, xi):
        assert self.kind == "spem_1d"
        return self.scan.value[xi]

    # -- cut ----------------------------------------------------------
    @property
    def cut_frame(self):
        assert self.kind == "cut"
        return self.scan.value

    # -- map (angle/angle/E cube) ---------------------------------------
    @property
    def angle_cube(self):
        """The three axes and the cube, for the Map viewer.

        Named for the angle map it was written for, but a k-map saved to a
        file and read back arrives here too -- same shape, same viewer, axes
        that are momenta and labelled as such. Returns (x, y, E, cube).
        """
        assert self.kind in CUBE_KINDS, self.kind
        return self.scan.x, self.scan.k, self.scan.z, self.scan.value

    def kcube(self, kinetic_energy_eV=None):
        """Optional, NOT used by the GUI by default: converts the
        deflector-angle axis to momentum via loader.nxs_file.to_kspace_cube
        (small-angle free-electron approximation). Returns
        (kx, ky, E, cube). Available for scripted/offline analysis."""
        assert self.kind == "map"
        return to_kspace_cube(self.scan, kinetic_energy_eV=kinetic_energy_eV)


class _MemScan:
    """The subset of :class:`loader.nxs_file.NxsScan` that a computed dataset
    needs to expose.

    A converted k-map, an arbitrary-direction cut and a Fermi-corrected map
    are all computed rather than read: no file, no HDF5 entry, no lazy cube
    behind them. The viewers only ever touch these few attributes, so
    presenting them is enough for a computed dataset to be displayed,
    exported and saved exactly like a measured one.

    Which attributes carry the axes depends on the kind, matching the file
    parser: a 2D ``cut`` uses ``x``/``y``, a 3D ``map``/``k_map`` uses
    ``x``/``k``/``z``.
    """

    def __init__(self, kind, axes, value, labels, info, fourd_info=None):
        from loader.nxs_file import axis_slots

        self.kind = kind
        self.labels = labels
        self.info = info
        self.fourd_info = dict(fourd_info or {})
        self.previews = {}
        self.value = value
        self.value4d = None
        self.x = self.y = self.k = self.z = None

        # Which slots this kind fills comes from loader.nxs_file.AXIS_SLOTS,
        # the one table that says so -- this used to be one of five
        # hand-written copies of it.
        slots = axis_slots(kind, "constructor")
        if not slots:
            slots = ("x", "y")             # unknown 2-D-ish kind: best effort
        if len(axes) != len(slots):
            raise ValueError(
                f"a {kind} takes {len(slots)} axes {slots}, got {len(axes)}")
        for slot, values in zip(slots, axes):
            setattr(self, slot, values)

        if kind == "spem_4d":
            # The cube lives in value4d, where the viewer looks for it, and
            # value mirrors it so the generic paths (saving, the data
            # operations) need no special case.
            self.value4d = value
        elif kind in CUBE_KINDS:
            # A map's second axis is the analyser's, which the viewers reach
            # as either .k or .y depending on which of them is asking.
            self.y = self.k

    def close(self):
        pass


class MemoryData:
    """A dataset computed in this session, quacking like :class:`NxsData`.

    Everything the viewers use -- ``kind``, ``scan``, ``cut_frame`` /
    ``angle_cube``, ``path``, ``close`` -- is here, so a computed dataset
    opens through exactly the same window as the measurement it came from,
    is exported by the same code, and is written by ``save_dataset`` in the
    same format. ``info`` records where it came from and the settings that
    produced it, so a dataset saved now still says how it was made.
    """

    def __init__(self, kind, axes, value, labels, *, source_label,
                 parameters=None, prefix="calc", source_path="",
                 source_info=None, source_motors=None):
        info = dict(source_info or {})
        info["_kind"] = kind
        info["_source"] = source_label
        for key, val in (parameters or {}).items():
            info[f"{prefix}.{key}"] = val

        self.path = source_path
        self.kind = kind
        self.source_label = source_label
        self.parameters = dict(parameters or {})
        self.scan = _MemScan(kind, axes, value, labels, info, source_motors)

    # -- what the viewers ask for -----------------------------------------
    @property
    def cut_frame(self):
        assert self.kind == "cut", self.kind
        return self.scan.value

    # -- spatial scan, for a SPEM computed in this session ------------------
    def spatial_overview_computed(self, progress=None):
        """(y, x) map, summed over (k, E). In memory the whole cube is here
        already, so there is nothing to stream or approximate."""
        assert self.kind == "spem_4d", self.kind
        if getattr(self, "_overview", None) is None:
            self._overview = np.asarray(self.scan.value4d, dtype=np.float64).sum(axis=(2, 3))
        return self._overview

    @property
    def spatial_overview(self):
        return self.spatial_overview_computed()

    def frame_at(self, yi, xi):
        assert self.kind == "spem_4d", self.kind
        return np.asarray(self.scan.value4d[yi, xi], dtype=float)

    def frame_over_region(self, yslice, xslice):
        assert self.kind == "spem_4d", self.kind
        return np.asarray(self.scan.value4d[yslice, xslice],
                          dtype=np.float64).sum(axis=(0, 1))

    def spatial_over_region(self, kslice, eslice, progress=None):
        assert self.kind == "spem_4d", self.kind
        return np.asarray(self.scan.value4d[:, :, kslice, eslice],
                          dtype=np.float64).sum(axis=(2, 3))

    @property
    def line_kmap(self):
        assert self.kind == "spem_1d", self.kind
        return np.asarray(self.scan.value, dtype=float).sum(axis=2)

    def frame_at_x(self, xi):
        assert self.kind == "spem_1d", self.kind
        return np.asarray(self.scan.value[xi], dtype=float)

    @property
    def angle_cube(self):
        """(x, y, E, cube) -- the Map viewer's accessor. Keeps the name for
        a k-map, whose axes are momenta: renaming it would mean forking the
        viewer for no gain, and the axis *labels* say A^-1, which is what
        the user reads."""
        assert self.kind in CUBE_KINDS, self.kind
        return self.scan.x, self.scan.k, self.scan.z, self.scan.value

    def close(self):
        pass

    def retain(self):
        """Nothing to count for an in-memory dataset; see NxsData.retain."""
        return self

    def alive(self):
        return True


def KMapData(kx, ky, energy, cube, *, source_label, parameters,
             source_path="", source_info=None,
             source_motors=None):
    """A Map converted to momentum space, held in memory.

    A thin front for :class:`MemoryData` with the k-map's labels and the
    ``kconv.*`` parameter prefix the conversion round established (the `.m`
    file kept the same record in ``data_k.para``).
    """
    return MemoryData("k_map", (kx, ky, energy), cube,
                      {"x": "kx (\u00c5\u207b\u00b9)", "k": "ky (\u00c5\u207b\u00b9)",
                       "z": "Energy (eV)"},
                      source_label=source_label, parameters=parameters,
                      prefix="kconv", source_path=source_path,
                      source_info=source_info, source_motors=source_motors)


def curve_dataset(kind, x, values, names, *, x_label, value_label, name,
                  step, parameters=None, source_info=None, source_path="",
                  prefix="curve", source=""):
    """A curve table (EDC / MDC / spin EDC) as a :class:`MemoryData`, with
    its channels named and the operation that made it recorded."""
    from tools import curves as C
    from tools import process as P
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    info = P.record_step(dict(source_info or {}),
                         P.Step(step, dict(parameters or {}),
                                source=source or name))
    info.update(C.curve_info(names, value_label))
    return MemoryData(kind, (np.asarray(x, dtype=float),
                             np.arange(values.shape[1], dtype=float)),
                      values, {"x": x_label, "y": "Channel"},
                      source_label=name, parameters=parameters, prefix=prefix,
                      source_path=source_path, source_info=info)


def unit_of(label):
    """The unit in an axis label, e.g. "Angle, deflector (deg)" -> "deg"."""
    if label and "(" in label and label.rstrip().endswith(")"):
        return label[label.rfind("(") + 1:-1]
    return ""


def full_array(data):
    """The dataset's whole array in memory (a lazy one is read)."""
    scan = data.scan
    values = scan.value4d if data.kind == "spem_4d" and scan.value4d is not None \
        else scan.value
    if hasattr(values, "materialise"):
        values = values.materialise()
    return np.asarray(values, dtype=float)
