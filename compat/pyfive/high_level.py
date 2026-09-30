""" High-level classes for reading HDF5 files.  """

from collections import deque
from collections.abc import Mapping, Sequence
import os
import posixpath
import threading
import weakref

import numpy as np

from .core import Reference
from .dataobjects import DataObjects
from .misc_low_level import SuperBlock
from . import chunkcache

#: Every Dataset that keeps a whole decompressed copy (see
#: ``Dataset.__getitem__``), so that "free memory" can drop them all
#: (patched in: ARPES viewer).
_CACHING_DATASETS = weakref.WeakSet()


def clear_dataset_caches():
    """ Drop every whole-dataset copy kept by an open Dataset, and every
    cached chunk. Returns the number of bytes released (patched in). """
    freed = chunkcache.clear()
    for dset in list(_CACHING_DATASETS):
        cache = getattr(dset, '_cache', None)
        if cache is not None:
            freed += int(getattr(cache, 'nbytes', 0) or 0)
            dset._cache = None
    return freed


def dataset_cache_bytes():
    """ Bytes held in whole-dataset copies (patched in). """
    total = 0
    for dset in list(_CACHING_DATASETS):
        cache = getattr(dset, '_cache', None)
        if cache is not None:
            total += int(getattr(cache, 'nbytes', 0) or 0)
    return total


class _NoLock(object):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


_NO_LOCK = _NoLock()


class Group(Mapping):
    """
    An HDF5 Group which may hold attributes, datasets, or other groups.

    Attributes
    ----------
    attrs : dict
        Attributes for this group.
    name : str
        Full path to this group.
    file : File
        File instance where this group resides.
    parent : Group
        Group instance containing this group.

    """

    def __init__(self, name, dataobjects, parent):
        """ initalize. """

        self.parent = parent
        self.file = parent.file
        self.name = name

        # Patched (ARPES viewer): the links are parsed once per group object
        # header and kept on it (see File._dataobjects_at).
        links = getattr(dataobjects, '_links_cache', None)
        if links is None:
            with getattr(parent.file, '_lock', None) or _NO_LOCK:
                links = dataobjects.get_links()
            dataobjects._links_cache = links
        self._links = links
        self._dataobjects = dataobjects
        self._attrs = None  # cached property

    def __repr__(self):
        return '<HDF5 group "%s" (%d members)>' % (self.name, len(self))

    def __len__(self):
        """ Number of links in the group. """
        return len(self._links)

    # Patched (ARPES viewer): h5py answers ``"a/b" in group`` and
    # ``group.get("a/b")`` with False / None whenever the path does not lead
    # to an object, whatever is in the way; Mapping only catches KeyError.
    def __contains__(self, name):
        try:
            self[name]
        except Exception:
            return False
        return True

    def get(self, name, default=None):
        try:
            return self[name]
        except Exception:
            return default

    def _dereference(self, ref):
        """ Deference a Reference object. """
        if not ref:
            raise ValueError('cannot deference null reference')
        obj = self.file._get_object_by_address(ref.address_of_reference)
        if obj is None:
            raise ValueError('reference not found in file')
        return obj

    def __getitem__(self, y):
        """ x.__getitem__(y) <==> x[y] """
        if isinstance(y, Reference):
            return self._dereference(y)

        path = posixpath.normpath(y)
        if path == '.':
            return self
        if path.startswith('/'):
            return self.file[path[1:]]

        if posixpath.dirname(path) != '':
            next_obj, additional_obj = path.split('/', 1)
        else:
            next_obj = path
            additional_obj = '.'

        if next_obj not in self._links:
            raise KeyError('%s not found in group' % (next_obj))

        obj_name = posixpath.join(self.name, next_obj)
        link_target = self._links[next_obj]

        if isinstance(link_target, str):
            try:
                return self.__getitem__(link_target)
            except KeyError:
                return None

        dataobjs = self.file._dataobjects_at(link_target)
        if dataobjs.is_dataset:
            if additional_obj != '.':
                raise KeyError('%s is a dataset, not a group' % (obj_name))
            return Dataset(obj_name, dataobjs, self)
        return Group(obj_name, dataobjs, self)[additional_obj]

    def __iter__(self):
        for k in self._links.keys():
            yield k

    def visit(self, func):
        """
        Recursively visit all names in the group and subgroups.

        func should be a callable with the signature:

            func(name) -> None or return value

        Returning None continues iteration, return anything else stops and
        return that value from the visit method.

        """
        return self.visititems(lambda name, obj: func(name))

    def visititems(self, func):
        """
        Recursively visit all objects in this group and subgroups.

        func should be a callable with the signature:

            func(name, object) -> None or return value

        Returning None continues iteration, return anything else stops and
        return that value from the visit method.

        """
        root_name_length = len(self.name)
        if not self.name.endswith('/'):
            root_name_length += 1
        queue = deque(self.values())
        while queue:
            obj = queue.popleft()
            name = obj.name[root_name_length:]
            ret = func(name, obj)
            if ret is not None:
                return ret
            if isinstance(obj, Group):
                queue.extend(obj.values())
        return None

    @property
    def attrs(self):
        """ attrs attribute. """
        if self._attrs is None:
            with getattr(self.file, '_lock', None) or _NO_LOCK:
                self._attrs = self._dataobjects.get_attributes()
        return self._attrs


class File(Group):
    """
    Open a HDF5 file.

    Note in addition to having file specific methods the File object also
    inherit the full interface of **Group**.

    File is also a context manager and therefore supports the with statement.
    Files opened by the class will be closed after the with block, file-like
    object are not closed.

    Parameters
    ----------
    filename : str or file-like
        Name of file (string or unicode) or file like object which has read
        and seek methods which behaved like a Python file object.

    Attributes
    ----------
    filename : str
        Name of the file on disk, None if not available.
    mode : str
        String indicating that the file is open readonly ("r").
    userblock_size : int
        Size of the user block in bytes (currently always 0).

    """

    def __init__(self, filename):
        """ initalize. """
        self._close = False
        if hasattr(filename, 'read'):
            if not hasattr(filename, 'seek'):
                raise ValueError(
                    'File like object must have a seek method')
            self._fh = filename
            self.filename = getattr(filename, 'name', None)
        else:
            self._fh = open(filename, 'rb')
            self._close = True
            self.filename = filename
        # Patched (ARPES viewer): one lock per file, because the viewer
        # reads the same file from the GUI thread and from worker threads
        # and every read here is a seek followed by a read; and every object
        # header parsed once and kept (by address), because walking to
        # "/a/b/c/d" used to re-parse each of a, b, c from the file on every
        # lookup -- hundreds of metadata lookups per scan.
        self._lock = threading.RLock()
        self._objects = {}
        self._superblock = SuperBlock(self._fh, 0)
        offset = self._superblock.offset_to_dataobjects
        dataobjects = self._dataobjects_at(offset)

        self.file = self
        self.mode = 'r'
        self.userblock_size = 0
        super(File, self).__init__('/', dataobjects, self)

    def __repr__(self):
        return '<HDF5 file "%s" (mode r)>' % (os.path.basename(self.filename))

    def _dataobjects_at(self, offset):
        """ The parsed object header at ``offset``, parsed once (patched). """
        dataobjs = self._objects.get(offset)
        if dataobjs is None:
            with self._lock:
                dataobjs = DataObjects(self._fh, offset)
            self._objects[offset] = dataobjs
        return dataobjs

    def _get_object_by_address(self, obj_addr):
        """ Return the object pointed to by a given address. """
        if self._dataobjects.offset == obj_addr:
            return self
        return self.visititems(
            lambda x, y: y if y._dataobjects.offset == obj_addr else None)

    def close(self):
        """ Close the file. """
        fh = getattr(self, '_fh', None)
        if fh is not None:
            chunkcache.forget_file(fh)
        self._objects = {}
        if self._close and fh is not None:
            fh.close()
    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        self.close()


class Dataset(object):
    """
    A HDF5 Dataset containing an n-dimensional array and meta-data attributes.

    Attributes
    ----------
    shape : tuple
        Dataset dimensions.
    dtype : dtype
        Dataset's type.
    size : int
        Total number of elements in the dataset.
    chunks : tuple or None
        Chunk shape, or NOne is chunked storage not used.
    compression : str or None
        Compression filter used on dataset.  None if compression is not enabled
        for this dataset.
    compression_opts : dict or None
        Options for the compression filter.
    scaleoffset : dict or None
        Setting for the HDF5 scale-offset filter, or None if scale-offset
        compression is not used for this dataset.
    shuffle : bool
        Whether the shuffle filter is applied for this dataset.
    fletcher32 : bool
        Whether the Fletcher32 checksumming is enabled for this dataset.
    fillvalue : float or None
        Value indicating uninitialized portions of the dataset. None is no fill
        values has been defined.
    dim : int
        Number of dimensions.
    dims : None
        Dimension scales.
    attrs : dict
        Attributes for this dataset.
    name : str
        Full path to this dataset.
    file : File
        File instance where this dataset resides.
    parent : Group
        Group instance containing this dataset.

    """

    def __init__(self, name, dataobjects, parent):
        """ initalize. """
        self.parent = parent
        self.file = parent.file
        self.name = name

        self._dataobjects = dataobjects
        self._attrs = None
        self._astype = None

    def __repr__(self):
        info = (os.path.basename(self.name), self.shape, self.dtype)
        return '<HDF5 dataset "%s": shape %s, type "%s">' % info

    def __getitem__(self, args):
        # Patched (ARPES viewer): a chunked dataset is decompressed whole on
        # every read by pyfive, so the result is kept -- the viewers slice
        # the same cube over and over.
        cache = getattr(self, '_cache', None)
        lock = getattr(self.file, '_lock', None) or _NO_LOCK
        if cache is None and self._region_reads():
            region = self._region_of(args)
            if region is not None:
                lo, hi, relative = region
                with lock:
                    data = self._dataobjects.get_region(lo, hi)[relative]
                if self._astype is None:
                    return data
                return data.astype(self._astype)
        if cache is None:
            with lock:
                cache = self._dataobjects.get_data()
            if not isinstance(cache, np.memmap):
                self._cache = cache
                _CACHING_DATASETS.add(self)
        data = cache[args]
        if self._astype is None:
            return data
        return data.astype(self._astype)

    #: Chunked datasets larger than this are read chunk by chunk as they
    #: are indexed rather than decompressed whole (patched in). 16 MB: a
    #: whole copy of anything bigger is kept per open dataset, which is
    #: memory the server may need elsewhere.
    REGION_READ_BYTES = 16 * 1024 * 1024

    def _region_reads(self):
        try:
            if self.chunks is None or isinstance(self._dataobjects.dtype, tuple):
                return False
            return int(np.prod(self.shape)) * self.dtype.itemsize > self.REGION_READ_BYTES
        except Exception:
            return False

    def _region_of(self, args):
        """ ``(lo, hi, relative index)`` for an index of ints and slices,
        or None for anything fancier (read whole instead). """
        if not isinstance(args, tuple):
            args = (args,)
        if any(a is Ellipsis for a in args):
            i = [k for k, a in enumerate(args) if a is Ellipsis][0]
            args = args[:i] + (slice(None),) * (len(self.shape) - len(args) + 1) + args[i + 1:]
        args = args + (slice(None),) * (len(self.shape) - len(args))
        if len(args) != len(self.shape):
            return None
        lo, hi, relative = [], [], []
        for a, n in zip(args, self.shape):
            if isinstance(a, (int, np.integer)):
                a = int(a) + (n if a < 0 else 0)
                if not 0 <= a < n:
                    raise IndexError('index %d out of range' % a)
                lo.append(a)
                hi.append(a + 1)
                relative.append(0)
            elif isinstance(a, slice):
                start, stop, step = a.indices(n)
                if step < 0 or stop <= start:
                    return None
                lo.append(start)
                hi.append(stop)
                relative.append(slice(0, stop - start, step))
            else:
                return None
        return lo, hi, tuple(relative)

    def read_direct(self, array, source_sel=None, dest_sel=None):
        """
        Read from a HDF5 dataset directly into a NumPy array.

        This is equivalent to dset[source_sel] = arr[dset_sel].

        Creation of intermediates is not avoided. This method if provided from
        compatibility with h5py, it is not efficient.

        """
        array[dest_sel] = self[source_sel]

    def astype(self, dtype):
        """
        Return a context manager which returns data as a particular type.

        Conversion is handled by NumPy after reading extracting the data.
        """
        return AstypeContext(self, dtype)

    def len(self):
        """ Return the size of the first axis. """
        return self.shape[0]

    @property
    def shape(self):
        """ shape attribute. """
        return self._dataobjects.shape

    @property
    def maxshape(self):
        """ maxshape attribute. (None for unlimited dimensions) """
        return self._dataobjects.maxshape

    @property
    def ndim(self):
        """ number of dimensions. """
        return len(self.shape)

    @property
    def dtype(self):
        """ dtype attribute (patched: a numpy dtype, as h5py gives). """
        dtype = self._dataobjects.dtype
        if isinstance(dtype, tuple):
            return np.dtype(object)
        return np.dtype(dtype)

    @property
    def value(self):
        """ alias for dataset[()]. """
        DeprecationWarning(
            "dataset.value has been deprecated. Use dataset[()] instead.")
        return self[()]

    @property
    def size(self):
        """ size attribute. """
        return np.prod(self.shape)

    @property
    def chunks(self):
        """ chunks attribute. """
        return self._dataobjects.chunks

    @property
    def compression(self):
        """ compression attribute. """
        return self._dataobjects.compression

    @property
    def compression_opts(self):
        """ compression_opts attribute. """
        return self._dataobjects.compression_opts

    @property
    def scaleoffset(self):
        """ scaleoffset attribute. """
        return None  # TODO support scale-offset filter

    @property
    def shuffle(self):
        """ shuffle attribute. """
        return self._dataobjects.shuffle

    @property
    def fletcher32(self):
        """ fletcher32 attribute. """
        return self._dataobjects.fletcher32

    @property
    def fillvalue(self):
        """ fillvalue attribute. """
        return self._dataobjects.fillvalue

    @property
    def dims(self):
        """ dims attribute. """
        return DimensionManager(self)

    @property
    def attrs(self):
        """ attrs attribute. """
        if self._attrs is None:
            with getattr(self.file, '_lock', None) or _NO_LOCK:
                self._attrs = self._dataobjects.get_attributes()
        return self._attrs


class DimensionManager(Sequence):
    """ Represents a collection of dimensions associated with a dataset. """
    def __init__(self, dset):
        ndim = len(dset.shape)
        dim_list = [[]]*ndim
        if 'DIMENSION_LIST' in dset.attrs:
            dim_list = dset.attrs['DIMENSION_LIST']
        dim_labels = [b'']*ndim
        if 'DIMENSION_LABELS' in dset.attrs:
            dim_labels = dset.attrs['DIMENSION_LABELS']
        self._dims = [
            DimensionProxy(dset.file, label, refs) for
            label, refs in zip(dim_labels, dim_list)]

    def __len__(self):
        return len(self._dims)

    def __getitem__(self, x):
        return self._dims[x]


class DimensionProxy(Sequence):
    """ Represents a HDF5 "dimension". """

    def __init__(self, dset_file, label, refs):
        self.label = label.decode('utf-8')
        self._refs = refs
        self._file = dset_file

    def __len__(self):
        return len(self._refs)

    def __getitem__(self, x):
        return self._file[self._refs[x]]


class AstypeContext(object):
    """
    Context manager which allows changing the type read from a dataset.
    """

    def __init__(self, dset, dtype):
        self._dset = dset
        self._dtype = np.dtype(dtype)

    def __enter__(self):
        self._dset._astype = self._dtype

    def __exit__(self, *args):
        self._dset._astype = None
