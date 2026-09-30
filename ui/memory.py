"""
ui/memory.py
============
Keeping an eye on memory, so the viewer does not squeeze the other programs
running on the same server.

* :class:`MemoryBar` -- one line at the bottom of the main window: this
  program's memory, the server's free memory as a bar (green / orange /
  red), a **Free memory** button and **Memory...** for the panel. Updated
  every few seconds. When the server runs low it frees the program's caches
  by itself (if allowed) and says so, once.
* :class:`MemoryPanel` -- the details: the program's memory and what is in
  it (the HDF5 reader's caches, the datasets held in memory, the open
  windows), the server's memory and its largest processes, and the controls:
  free caches, move computed datasets out of memory (they are auto-saved,
  and are read back from disk when opened), close windows, and the limits.

What "free" does is in :mod:`tools.memory`; the list of datasets and windows
is the launcher's (``App.memory_datasets`` / ``memory_windows`` /
``unload_datasets`` / ``free_memory``).
"""
import time

import tkinter as tk
from tkinter import ttk

from tools import memory as M
from ui import tkbase as T

#: Defaults of the settings kept in the config file.
DEFAULTS = {
    "MemoryWarnPercent": 10.0,      # warn when the server has less free than this
    "MemoryAutoFree": True,         # ... and free the caches by itself then
    "MemoryProgramLimitGB": 0.0,    # warn when this program uses more (0 = off)
    "ChunkCacheMB": 256,            # HDF5 chunk cache ceiling
}

UPDATE_MS = 3000


def _setting(app, key):
    try:
        return type(DEFAULTS[key])(app.config.get(key, DEFAULTS[key]))
    except (TypeError, ValueError):
        return DEFAULTS[key]


def apply_settings(app):
    """Push the saved settings into the reader (chunk cache ceiling)."""
    from compat.pyfive import chunkcache
    chunkcache.set_limit(max(0, _setting(app, "ChunkCacheMB")) * 1024 * 1024)


def _style(root):
    style = ttk.Style(root)
    for name, colour in (("ok", "#3a9b3a"), ("warn", "#e0a020"), ("low", "#cc3333")):
        try:
            style.configure("%s.Horizontal.TProgressbar" % name, background=colour)
        except tk.TclError:
            pass


def _level(sysmem, warn_percent):
    """"ok" / "warn" / "low" for the server's free memory."""
    total, available = sysmem.get("total"), sysmem.get("available")
    if not total or available is None:
        return "ok", None
    free_pct = 100.0 * available / total
    if free_pct < warn_percent:
        return "low", free_pct
    if free_pct < 2 * warn_percent:
        return "warn", free_pct
    return "ok", free_pct


class MemoryBar(ttk.Frame):
    """The memory line of the main window."""

    def __init__(self, parent, app):
        ttk.Frame.__init__(self, parent)
        self.app = app
        _style(self)
        ttk.Label(self, text="Memory").pack(side="left", padx=(4, 2))
        self.program = ttk.Label(self, text="", width=24)
        self.program.pack(side="left")
        ttk.Label(self, text="server used").pack(side="left", padx=(6, 2))
        self.bar = ttk.Progressbar(self, length=140, maximum=100, mode="determinate",
                                   style="ok.Horizontal.TProgressbar")
        self.bar.pack(side="left")
        self.server = ttk.Label(self, text="", width=30)
        self.server.pack(side="left", padx=4)
        ttk.Button(self, text="Memory...", command=self.open_panel).pack(side="right", padx=2)
        ttk.Button(self, text="Free memory", command=self.free).pack(side="right", padx=2)
        T.Tooltip(self.bar, "How full the server's memory is (all users, all programs). "
                  "Orange / red: little is left -- free memory here, close windows, "
                  "or wait for other jobs.")
        self._warned = False
        self._program_warned = False
        self.panel = None
        self.after(500, self.update_now)

    def open_panel(self):
        if self.panel is not None and self.panel.winfo_exists():
            self.panel.deiconify()
            self.panel.lift()
            self.panel.refresh()
            return self.panel
        self.panel = MemoryPanel(self.winfo_toplevel(), self.app)
        return self.panel

    def free(self):
        report = self.app.free_memory(unload=False)
        self.app.say(report)
        self.update_now(reschedule=False)

    def update_now(self, reschedule=True):
        try:
            self._update()
        except Exception:                                   # noqa: BLE001
            pass
        if reschedule:
            try:
                self.after(UPDATE_MS, self.update_now)
            except tk.TclError:
                pass

    def _update(self):
        proc = M.process_memory()
        sysmem = M.system_memory()
        self.program.configure(text="this program %s" % M.fmt_bytes(proc["rss"]))
        level, free_pct = _level(sysmem, _setting(self.app, "MemoryWarnPercent"))
        if free_pct is None:
            self.server.configure(text="(not available)")
            return
        self.bar.configure(value=100.0 - free_pct, style="%s.Horizontal.TProgressbar" % level)
        self.server.configure(text="%s free of %s" % (M.fmt_bytes(sysmem["available"]),
                                                      M.fmt_bytes(sysmem["total"])),
                              foreground={"ok": "", "warn": "#a06000", "low": "#b00000"}[level])
        if level == "low" and not self._warned:
            self._warned = True
            self._low_memory(sysmem, free_pct)
        elif level == "ok":
            self._warned = False

        limit_gb = _setting(self.app, "MemoryProgramLimitGB")
        if limit_gb > 0 and proc["rss"]:
            if proc["rss"] > limit_gb * 1024 ** 3 and not self._program_warned:
                self._program_warned = True
                report = self.app.free_memory(unload=False)
                T.warning(self.winfo_toplevel(), "Memory",
                          "This program is using %s, more than the %.1f GB set as its "
                          "limit.\n\n%s\n\nIf that is not enough, open Memory... to move "
                          "computed datasets out of memory or close windows."
                          % (M.fmt_bytes(proc["rss"]), limit_gb, report))
            elif proc["rss"] < 0.9 * limit_gb * 1024 ** 3:
                self._program_warned = False

    def _low_memory(self, sysmem, free_pct):
        text = ("The server has only %s free (%.0f%% of %s).\n\nOther programs running "
                "on it may fail if it runs out."
                % (M.fmt_bytes(sysmem["available"]), free_pct, M.fmt_bytes(sysmem["total"])))
        if _setting(self.app, "MemoryAutoFree"):
            text += "\n\nThis program's caches were freed:\n" + self.app.free_memory(unload=False)
        text += ("\n\nOpen Memory... to see which programs use the most, move computed "
                 "datasets out of memory, or close windows.")
        self.app.say("Server memory low: %s free" % M.fmt_bytes(sysmem["available"]))
        T.warning(self.winfo_toplevel(), "Server memory low", text)


class MemoryPanel(tk.Toplevel):
    """The details, and every way of giving memory back."""

    def __init__(self, parent, app):
        tk.Toplevel.__init__(self, parent)
        self.app = app
        self.title("Memory")
        self.geometry("900x760")
        _style(self)

        top = ttk.LabelFrame(self, text="Server")
        top.pack(fill="x", padx=8, pady=(8, 4))
        self.server_bar = ttk.Progressbar(top, length=400, maximum=100,
                                          style="ok.Horizontal.TProgressbar")
        self.server_bar.grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.server_text = ttk.Label(top, text="")
        self.server_text.grid(row=0, column=1, sticky="w", padx=6)
        self.swap_text = ttk.Label(top, text="")
        self.swap_text.grid(row=1, column=0, columnspan=2, sticky="w", padx=4)

        prog = ttk.LabelFrame(self, text="This program")
        prog.pack(fill="x", padx=8, pady=4)
        self.program_text = ttk.Label(prog, text="", justify="left")
        self.program_text.pack(anchor="w", padx=4, pady=2)
        self.cache_text = ttk.Label(prog, text="", justify="left")
        self.cache_text.pack(anchor="w", padx=4, pady=(0, 3))
        row = ttk.Frame(prog)
        row.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(row, text="Free caches", command=lambda: self.free(False)).pack(side="left")
        ttk.Button(row, text="Free everything possible",
                   command=lambda: self.free(True)).pack(side="left", padx=4)
        ttk.Label(row, foreground="#555", text=(
            "Everything = caches + computed datasets not open in a window "
            "(they stay in the list, read back from disk when opened)")).pack(side="left", padx=4)

        panes = ttk.Frame(self)
        panes.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.LabelFrame(panes, text="Computed datasets held in memory")
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.datasets = ttk.Treeview(left, columns=("size", "state"), selectmode="extended",
                                     height=8)
        self.datasets.heading("#0", text="Dataset")
        self.datasets.heading("size", text="Memory")
        self.datasets.heading("state", text="")
        self.datasets.column("#0", width=200)
        self.datasets.column("size", width=80, anchor="e")
        self.datasets.column("state", width=150)
        self.datasets.pack(fill="both", expand=True, padx=3, pady=3)
        ttk.Button(left, text="Move selected out of memory",
                   command=self.unload_selected).pack(anchor="w", padx=3, pady=(0, 4))

        right = ttk.LabelFrame(panes, text="Open viewer windows")
        right.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.windows = ttk.Treeview(right, columns=("size",), selectmode="extended", height=8)
        self.windows.heading("#0", text="Window")
        self.windows.heading("size", text="Its data")
        self.windows.column("#0", width=260)
        self.windows.column("size", width=80, anchor="e")
        self.windows.pack(fill="both", expand=True, padx=3, pady=3)
        ttk.Button(right, text="Close selected windows",
                   command=self.close_selected).pack(anchor="w", padx=3, pady=(0, 4))

        procs = ttk.LabelFrame(self, text="Largest processes on the server")
        procs.pack(fill="both", expand=True, padx=8, pady=4)
        self.procs = ttk.Treeview(procs, columns=("pid", "user", "name", "rss"),
                                  show="headings", height=7)
        for col, text, width, anchor in (("pid", "PID", 70, "e"), ("user", "User", 120, "w"),
                                         ("name", "Program", 260, "w"),
                                         ("rss", "Memory", 100, "e")):
            self.procs.heading(col, text=text)
            self.procs.column(col, width=width, anchor=anchor)
        self.procs.tag_configure("me", foreground="#1f4e8c")
        self.procs.pack(fill="both", expand=True, padx=3, pady=3)

        settings = ttk.LabelFrame(self, text="Settings (kept for next time)")
        settings.pack(fill="x", padx=8, pady=4)
        self.warn_var = tk.StringVar(value=str(_setting(app, "MemoryWarnPercent")))
        self.auto_var = tk.BooleanVar(value=_setting(app, "MemoryAutoFree"))
        self.limit_var = tk.StringVar(value=str(_setting(app, "MemoryProgramLimitGB")))
        self.cache_var = tk.StringVar(value=str(_setting(app, "ChunkCacheMB")))
        ttk.Label(settings, text="Warn when the server has less than").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(settings, textvariable=self.warn_var, width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(settings, text="% free").grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(settings, text="and free this program's caches by itself",
                        variable=self.auto_var).grid(row=0, column=3, sticky="w", padx=8)
        ttk.Label(settings, text="Warn when this program uses more than").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Entry(settings, textvariable=self.limit_var, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(settings, text="GB (0 = never)").grid(row=1, column=2, sticky="w")
        ttk.Label(settings, text="HDF5 chunk cache at most").grid(row=2, column=0, sticky="w", padx=4)
        ttk.Entry(settings, textvariable=self.cache_var, width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(settings, text="MB").grid(row=2, column=2, sticky="w")
        ttk.Label(settings, foreground="#555", text=(
            "(what makes moving around a SPEM scan fast; smaller = less memory, "
            "more re-reading)")).grid(row=2, column=3, sticky="w", padx=8)
        ttk.Button(settings, text="Apply", command=self.apply).grid(row=0, column=4, rowspan=3,
                                                                    padx=8, sticky="e")

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        self.status = ttk.Label(bar, text="", foreground="#1f4e8c")
        self.status.pack(side="left")
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right")
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="right", padx=4)
        self._window_rows = {}
        self._dataset_rows = {}
        self.refresh()
        self._tick()

    # -- display ---------------------------------------------------------------------
    def _tick(self):
        if not self.winfo_exists():
            return
        try:
            self._numbers()
        except Exception:                                   # noqa: BLE001
            pass
        self.after(UPDATE_MS, self._tick)

    def _numbers(self):
        sysmem = M.system_memory()
        level, free_pct = _level(sysmem, _setting(self.app, "MemoryWarnPercent"))
        if free_pct is not None:
            self.server_bar.configure(value=100.0 - free_pct,
                                      style="%s.Horizontal.TProgressbar" % level)
            self.server_text.configure(text="used %s of %s   --   %s free (%.0f%%)" % (
                M.fmt_bytes(sysmem["used"]), M.fmt_bytes(sysmem["total"]),
                M.fmt_bytes(sysmem["available"]), free_pct))
        else:
            self.server_text.configure(text="not available on this system")
        if sysmem.get("swap_total"):
            self.swap_text.configure(text="swap: %s used of %s" % (
                M.fmt_bytes(sysmem["swap_used"]), M.fmt_bytes(sysmem["swap_total"])))
        proc = M.process_memory()
        self.program_text.configure(text="resident now %s    highest so far %s    in swap %s" % (
            M.fmt_bytes(proc["rss"]), M.fmt_bytes(proc["peak"]), M.fmt_bytes(proc["swap"])))
        chunk, limit, whole = M.reader_cache_bytes()
        self.cache_text.configure(text=(
            "HDF5 reader caches: chunks %s (limit %s), whole datasets %s    "
            "files open: %d    viewer windows: %d" % (
                M.fmt_bytes(chunk), M.fmt_bytes(limit), M.fmt_bytes(whole),
                self.app.open_file_count(), len(self.app.viewers))))

    def refresh(self):
        self._numbers()
        self.datasets.delete(*self.datasets.get_children())
        self._dataset_rows = {}
        for row in self.app.memory_datasets():
            state = "open in a window" if row["open"] else (
                "auto-saved" if row["autosaved"] else "not saved anywhere")
            iid = self.datasets.insert("", "end", text=row["name"],
                                       values=(M.fmt_bytes(row["bytes"]), state))
            self._dataset_rows[iid] = row["key"]
        self.windows.delete(*self.windows.get_children())
        self._window_rows = {}
        for row in self.app.memory_windows():
            iid = self.windows.insert("", "end", text=row["title"],
                                      values=(M.fmt_bytes(row["bytes"]),))
            self._window_rows[iid] = row["window"]
        self.procs.delete(*self.procs.get_children())
        import os
        me = os.getpid()
        for pid, user, name, rss in M.top_processes(12):
            self.procs.insert("", "end", values=(pid, user, name + ("  (this program)" if pid == me else ""),
                                                 M.fmt_bytes(rss)),
                              tags=(("me",) if pid == me else ()))

    # -- actions ---------------------------------------------------------------------
    def say(self, text):
        self.status.configure(text=text)
        self.app.say(text)

    def free(self, everything):
        self.say(self.app.free_memory(unload=everything))
        self.refresh()

    def unload_selected(self):
        keys = [self._dataset_rows[i] for i in self.datasets.selection() if i in self._dataset_rows]
        if not keys:
            T.info(self, "Memory", "Select datasets in the list first.")
            return
        count, freed, skipped = self.app.unload_datasets(keys)
        M.trim()
        text = "Moved %d dataset(s) out of memory (%s)." % (count, M.fmt_bytes(freed))
        if skipped:
            T.info(self, "Memory", text + "\n\nKept in memory:\n" + "\n".join(
                "  %s: %s" % s for s in skipped))
        self.say(text)
        self.refresh()

    def close_selected(self):
        windows = [self._window_rows[i] for i in self.windows.selection() if i in self._window_rows]
        if not windows:
            T.info(self, "Memory", "Select windows in the list first.")
            return
        for window in windows:
            try:
                window.close()
            except Exception:                               # noqa: BLE001
                try:
                    window.destroy()
                except tk.TclError:
                    pass
        self.say("Closed %d window(s). %s" % (len(windows), self.app.free_memory(unload=False)))
        self.refresh()

    def apply(self):
        try:
            warn = float(self.warn_var.get())
            limit = float(self.limit_var.get())
            cache = int(float(self.cache_var.get()))
        except ValueError:
            T.warning(self, "Memory", "The settings are numbers.")
            return
        self.app.config["MemoryWarnPercent"] = max(0.0, min(90.0, warn))
        self.app.config["MemoryAutoFree"] = bool(self.auto_var.get())
        self.app.config["MemoryProgramLimitGB"] = max(0.0, limit)
        self.app.config["ChunkCacheMB"] = max(0, cache)
        apply_settings(self.app)
        self.app.save_config()
        self.say("Settings applied (%s)" % time.strftime("%H:%M:%S"))
        self.refresh()
