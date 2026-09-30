"""
A shared, size-limited cache of decompressed HDF5 chunks (patched in: ARPES
viewer).

Moving the cursor over a spatial (SPEM) scan reads one (k, E) spectrum at a
time out of a cube of a gigabyte or more. Each read touches a handful of
compressed chunks, and every one of them used to be read from disk and
inflated again on every click -- including the chunks the previous click had
just inflated. Keeping the most recently used chunks, up to a fixed number of
bytes, makes the next click on a nearby pixel free.

The limit is a hard one (the least recently used chunks are dropped first),
it is shared by every open file, and it can be changed or emptied from the
Memory panel so the viewer never competes with the other programs running on
the server for more than it was allowed.
"""
from collections import OrderedDict
import threading

#: Default ceiling, in bytes.
DEFAULT_LIMIT_BYTES = 256 * 1024 * 1024

_lock = threading.RLock()
_chunks = OrderedDict()           # (id(fh), address) -> read-only ndarray
_state = {"limit": DEFAULT_LIMIT_BYTES, "bytes": 0, "hits": 0, "misses": 0}


def get(fh, address):
    key = (id(fh), address)
    with _lock:
        chunk = _chunks.get(key)
        if chunk is None:
            _state["misses"] += 1
            return None
        _chunks.move_to_end(key)
        _state["hits"] += 1
        return chunk


def put(fh, address, chunk):
    size = int(chunk.nbytes)
    with _lock:
        if size > _state["limit"] // 4:
            # One chunk bigger than a quarter of the cache would push out
            # everything else for little gain: not kept.
            return
        key = (id(fh), address)
        old = _chunks.pop(key, None)
        if old is not None:
            _state["bytes"] -= int(old.nbytes)
        chunk.setflags(write=False)
        _chunks[key] = chunk
        _state["bytes"] += size
        _evict()


def _evict():
    while _chunks and _state["bytes"] > _state["limit"]:
        _key, old = _chunks.popitem(last=False)
        _state["bytes"] -= int(old.nbytes)


def forget_file(fh):
    """Drop every chunk of a file that is being closed (so that a new file
    object that happens to get the same id never sees them)."""
    ident = id(fh)
    with _lock:
        for key in [k for k in _chunks if k[0] == ident]:
            _state["bytes"] -= int(_chunks.pop(key).nbytes)


def clear():
    """Empty the cache; returns the number of bytes released."""
    with _lock:
        freed = _state["bytes"]
        _chunks.clear()
        _state["bytes"] = 0
        return freed


def set_limit(limit_bytes):
    with _lock:
        _state["limit"] = max(0, int(limit_bytes))
        _evict()


def limit():
    return _state["limit"]


def stats():
    """``{"bytes", "limit", "chunks", "hits", "misses"}``."""
    with _lock:
        return {"bytes": _state["bytes"], "limit": _state["limit"],
                "chunks": len(_chunks), "hits": _state["hits"],
                "misses": _state["misses"]}
