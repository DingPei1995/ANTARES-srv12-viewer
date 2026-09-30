"""
tools/memory.py
===============
How much memory this program and the server are using, and giving memory
back -- without psutil (not installed on the server): everything is read
from ``/proc``, which every Linux has. On a system without ``/proc`` the
functions return ``None`` fields and the GUI says "not available".

Giving memory back has three layers, all used by "Free memory":

1. drop what the program keeps *on spec* -- the cache of decompressed HDF5
   chunks and the whole-dataset copies the bundled HDF5 reader keeps
   (:func:`free_reader_caches`);
2. let Python collect what nothing refers to any more (``gc.collect``);
3. ask the C library to hand freed heap pages back to the system
   (``malloc_trim``). Without this last step a process that once used 10 GB
   can keep showing 10 GB even after every array is gone, because glibc
   keeps freed memory for reuse -- which is memory the other programs on the
   server cannot have.
"""
import ctypes
import gc
import os

__all__ = ["process_memory", "system_memory", "top_processes", "trim",
           "free_reader_caches", "reader_cache_bytes", "fmt_bytes",
           "array_bytes"]


def _read_kv(path):
    """``{"Key": bytes}`` from a ``Key:   123 kB`` file (/proc/meminfo,
    /proc/<pid>/status)."""
    out = {}
    try:
        with open(path, "r") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if not parts:
                    continue
                try:
                    value = int(parts[0])
                except ValueError:
                    continue
                if len(parts) > 1 and parts[1].lower() == "kb":
                    value *= 1024
                out[key.strip()] = value
    except (OSError, IOError):
        pass
    return out


def process_memory(pid="self"):
    """This process's memory, in bytes: ``rss`` (resident now), ``peak``
    (highest resident so far), ``swap`` (pushed out to swap)."""
    status = _read_kv("/proc/%s/status" % pid)
    return {"rss": status.get("VmRSS"), "peak": status.get("VmHWM"),
            "swap": status.get("VmSwap")}


def system_memory():
    """The server's memory, in bytes: ``total``, ``available`` (what can
    still be handed out without swapping), ``used`` (= total - available),
    ``swap_total``, ``swap_used``."""
    info = _read_kv("/proc/meminfo")
    total = info.get("MemTotal")
    available = info.get("MemAvailable")
    if available is None and total is not None:
        # kernels before 3.14: a close approximation
        available = (info.get("MemFree", 0) + info.get("Buffers", 0)
                     + info.get("Cached", 0))
    if total is None:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            total = available = None
    swap_total = info.get("SwapTotal")
    swap_free = info.get("SwapFree")
    return {"total": total, "available": available,
            "used": (total - available) if total is not None and available is not None else None,
            "swap_total": swap_total,
            "swap_used": (swap_total - swap_free) if swap_total is not None and swap_free is not None else None}


def _user_name(uid):
    try:
        import pwd
        return pwd.getpwuid(uid).pw_name
    except Exception:                                       # noqa: BLE001
        return str(uid)


def top_processes(count=10):
    """The processes using the most memory on the server:
    ``[(pid, user, name, rss_bytes), ...]``, largest first. Only what
    ``/proc`` lets any user read (name, owner, resident size)."""
    rows = []
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return rows
    names = {}
    for pid in pids:
        status = {}
        try:
            with open("/proc/%s/status" % pid, "r") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    if key in ("Name", "Uid", "VmRSS"):
                        status[key] = rest.strip()
        except (OSError, IOError):
            continue
        rss = status.get("VmRSS")
        if not rss:
            continue
        try:
            rss_bytes = int(rss.split()[0]) * 1024
            uid = int(status.get("Uid", "0").split()[0])
        except (ValueError, IndexError):
            continue
        if uid not in names:
            names[uid] = _user_name(uid)
        rows.append((int(pid), names[uid], status.get("Name", "?"), rss_bytes))
    rows.sort(key=lambda r: -r[3])
    return rows[:count]


def trim():
    """Collect garbage and return freed heap to the system. Returns True
    when ``malloc_trim`` was available (glibc)."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        return True
    except (OSError, AttributeError):
        return False


def free_reader_caches():
    """Empty the bundled HDF5 reader's caches; bytes released."""
    try:
        from compat.pyfive.high_level import clear_dataset_caches
        return int(clear_dataset_caches())
    except Exception:                                       # noqa: BLE001
        return 0


def reader_cache_bytes():
    """``(chunk cache bytes, chunk cache limit, whole-dataset copies bytes)``."""
    try:
        from compat.pyfive import chunkcache
        from compat.pyfive.high_level import dataset_cache_bytes
        stats = chunkcache.stats()
        return stats["bytes"], stats["limit"], dataset_cache_bytes()
    except Exception:                                       # noqa: BLE001
        return 0, 0, 0


def array_bytes(data):
    """Bytes a dataset (``NxsData`` / ``MemoryData``) holds *in memory*:
    its arrays that are numpy arrays. A lazy array still in its file, or a
    memory map of it, counts as nothing."""
    import numpy as np
    scan = getattr(data, "scan", None)
    if scan is None:
        return 0
    total, seen = 0, set()
    for name in ("value", "value4d"):
        value = getattr(scan, name, None)
        if isinstance(value, np.ndarray) and not isinstance(value, np.memmap) \
                and id(value) not in seen:
            seen.add(id(value))
            base = value.base if isinstance(value.base, np.ndarray) else value
            total += int(base.nbytes)
    overview = getattr(data, "_overview", None)
    if isinstance(overview, np.ndarray):
        total += int(overview.nbytes)
    return total


def fmt_bytes(value):
    if value is None:
        return "n/a"
    value = float(value)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            if unit in ("B", "kB"):
                return "%.0f %s" % (value, unit)
            return "%.2f %s" % (value, unit) if unit != "MB" else "%.0f %s" % (value, unit)
        value /= 1024.0
    return "%.1f TB" % value
