#!/usr/bin/env python3
"""
ARPES_viewer.py
===============
Launcher for the ARPES viewer (tkinter + matplotlib version, for Python 3.6
with numpy / scipy / matplotlib only -- no PyQt5, pyqtgraph or h5py).

    python ARPES_viewer.py

The main window is a browser: **Load data...** lists the datasets inside the
chosen files (one row per dataset, not per file), a click shows what the
dataset is, and a **double-click** opens it in its own viewer window:

- a real-space scan opens the spatial map with the spectrum at the cursor;
- a cut opens the E-vs-k spectrum with EDC / MDC;
- a map opens on its constant-energy contour, with buttons for its two cuts;
- a curve (EDC, MDC, spin EDC) opens in the curve viewer.

Everything a viewer computes (a k-map, a cut, a processed dataset...) is
added to this list and auto-saved to this session's folder
(``~/.arpes_viewer/sessions``), so that closing -- or losing -- the program
does not lose the work. Right-click a row for what acts on datasets.
"""
import csv
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402
import tkinter as tk                                                  # noqa: E402
from tkinter import ttk, filedialog                                   # noqa: E402

from ui import tkbase as T                                            # noqa: E402
from tools import system                                              # noqa: E402
from tools.dataops import axis_summary, same_format, ARRAY_AXES       # noqa: E402
from loader.nxs_file import CUBE_KINDS, CURVE_KINDS, KIND_LABELS, save_dataset  # noqa: E402
from loader import registry as nxs_loaders                            # noqa: E402
from loader import session as nxs_session                             # noqa: E402
from ui.data import NxsData, MemoryData                               # noqa: E402
from ui import viewers                                                # noqa: E402
from ui.list_actions import availability, kind_of as label_kind       # noqa: E402
from ui import memory as memory_ui                                    # noqa: E402
from tools import memory as memory_tools                              # noqa: E402

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".arpes_viewer", "config.json")

#: Kinds that are not auto-saved: a spatial scan is the measurement itself,
#: gigabytes of it.
NO_AUTOSAVE_KINDS = ("spem_4d", "spem_1d")

#: Which axes each kind shows in the Data information table.
DATA_INFO_AXES = {
    "cut": (("x", "angle"), ("y", "energy")),
    "map": (("x", "angle/k"), ("k", "angle/k"), ("z", "energy")),
    "k_map": (("x", "k"), ("k", "k"), ("z", "energy")),
    "kz_map": (("x", "photon energy"), ("k", "angle/k"), ("z", "energy")),
    "kz_map_k": (("x", "kz"), ("k", "k"), ("z", "energy")),
    "spem_4d": (("x", "spatial"), ("y", "spatial"), ("k", "angle"), ("z", "energy")),
    "spem_1d": (("x", "spatial"), ("y", "angle"), ("z", "energy")),
    "edc": (("x", "energy"), ("y", "channel")),
    "mdc": (("x", "angle/k"), ("y", "channel")),
    "spin_edc": (("x", "energy"), ("y", "channel")),
}


class App:
    """The launcher: the dataset list, the session folder, and the hooks
    the viewers use (``add_dataset``, ``open_data``, ``entries``, ``load``,
    ``names``, ``kind_of``, ``forget_viewer``)."""

    def __init__(self, root):
        self.root = root
        self.items = {}              # key -> record
        self.order = []              # keys in list order
        self.viewers = []
        self.session = nxs_session.SessionStore()
        defaults = {"FilePathLE": os.path.expanduser("~"),
                    "ColorMap": T.DEFAULT_COLORMAP, "FlipColorMap": False}
        defaults.update(memory_ui.DEFAULTS)
        self.config = system.load_config(CONFIG_PATH, defaults=defaults)
        memory_ui.apply_settings(self)
        self.colormap = self.config.get("ColorMap", T.DEFAULT_COLORMAP)
        if self.colormap not in T.COLORMAP_NAMES:
            self.colormap = T.DEFAULT_COLORMAP
        self.flip = bool(self.config.get("FlipColorMap", False))
        self.info_window = None
        self.info_rows = []
        self._counter = 0
        # The dataset of the row last clicked, kept open so that opening it
        # next (the usual double-click) does not read the file a second
        # time. Released when another row is clicked, and by Free memory.
        self._held = None
        self.build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def build(self):
        root = self.root
        root.title("ARPES viewer")
        root.geometry("880x720")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        top = ttk.Frame(root)
        top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="Folder").pack(side="left")
        self.folder_var = tk.StringVar(value=self.config.get("FilePathLE", ""))
        ttk.Entry(top, textvariable=self.folder_var).pack(side="left", fill="x",
                                                          expand=True, padx=4)
        ttk.Button(top, text="Browse...", command=self.browse_folder).pack(side="left")

        row = ttk.Frame(root)
        row.pack(fill="x", padx=6)
        ttk.Button(row, text="Load data...", command=self.open_loader).pack(side="left", padx=2)
        ttk.Button(row, text="Quick add files...", command=self.pick_files).pack(side="left", padx=2)
        ttk.Button(row, text="Clear list", command=self.clear_list).pack(side="left", padx=2)
        ttk.Label(row, text="Default colormap").pack(side="left", padx=(16, 2))
        self.cmap_var = tk.StringVar(value=self.colormap)
        combo = ttk.Combobox(row, textvariable=self.cmap_var, width=14, state="readonly",
                             values=T.COLORMAP_NAMES)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e: self.set_colormap())
        self.flip_var = tk.BooleanVar(value=self.flip)
        ttk.Checkbutton(row, text="Flip", variable=self.flip_var,
                        command=self.set_colormap).pack(side="left")
        self.apply_all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="apply to open windows", variable=self.apply_all_var).pack(side="left")

        mid = ttk.Frame(root)
        mid.pack(fill="both", expand=True, padx=6, pady=4)
        self.tree = ttk.Treeview(mid, columns=("kind", "entry", "where"),
                                 selectmode="extended")
        self.tree.heading("#0", text="Dataset")
        self.tree.heading("kind", text="Kind")
        self.tree.heading("entry", text="Entry")
        self.tree.heading("where", text="File / origin")
        self.tree.column("#0", width=300)
        self.tree.column("kind", width=80)
        self.tree.column("entry", width=110)
        self.tree.column("where", width=300)
        ys = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ys.pack(side="left", fill="y")
        self.tree.bind("<Double-Button-1>", lambda e: self.open_selected())
        self.tree.bind("<Return>", lambda e: self.open_selected())
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.on_select())
        self.tree.bind("<Button-3>", self.show_menu)
        self.tree.bind("<Button-2>", self.show_menu)

        buttons = ttk.Frame(root)
        buttons.pack(fill="x", padx=6)
        for text, command in (("Open", self.open_selected),
                              ("Information...", self.show_information),
                              ("Process...", self.open_processing),
                              ("Data operations...", self.open_data_operations),
                              ("Figure...", self.plot_selected_as_figure),
                              ("Save...", self.save_selected),
                              ("Remove", self.remove_selected)):
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=2, pady=2)

        box = ttk.LabelFrame(root, text="Data information")
        box.pack(fill="x", padx=6, pady=4)
        self.axis_table = ttk.Treeview(box, columns=("label", "min", "max", "num", "step"),
                                       show="headings", height=4)
        for col, width in (("label", 260), ("min", 100), ("max", 100), ("num", 60),
                           ("step", 100)):
            self.axis_table.heading(col, text=col)
            self.axis_table.column(col, width=width)
        self.axis_table.pack(fill="x", padx=3, pady=3)

        self.status = ttk.Label(root, text="", anchor="w", relief="sunken")
        self.status.pack(side="bottom", fill="x")
        self.memory_bar = memory_ui.MemoryBar(root, self)
        self.memory_bar.pack(side="bottom", fill="x", padx=4, pady=(0, 2))

    def say(self, text):
        self.status.configure(text=text)

    # ------------------------------------------------------------------
    # Colormap
    # ------------------------------------------------------------------
    def set_colormap(self):
        self.colormap, self.flip = self.cmap_var.get(), bool(self.flip_var.get())
        if self.apply_all_var.get():
            for window in list(self.viewers):
                try:
                    window.set_colormap(self.colormap, self.flip)
                except tk.TclError:
                    pass

    # ------------------------------------------------------------------
    # Adding datasets
    # ------------------------------------------------------------------
    def browse_folder(self):
        folder = filedialog.askdirectory(parent=self.root,
                                         initialdir=self.folder_var.get() or os.path.expanduser("~"))
        if folder:
            self.folder_var.set(folder)

    def pick_files(self):
        from ui.filebrowser import FilePickerDialog
        start = self.folder_var.get()
        if not os.path.isdir(start):
            start = os.path.expanduser("~")
        dialog = FilePickerDialog(self.root, start)
        paths = dialog.ask()
        if paths:
            self.folder_var.set(os.path.dirname(paths[0]))
            self.add_files(list(paths))

    def open_loader(self):
        from ui.loader_dialog import LoaderDialog
        dialog = LoaderDialog(self.root, self.folder_var.get())
        result = dialog.ask()
        if dialog.last_folder and os.path.isdir(dialog.last_folder):
            self.folder_var.set(dialog.last_folder)
        if not result:
            return None
        paths, options = result
        self.folder_var.set(os.path.dirname(paths[0]))
        self.add_files(paths, options)
        return paths

    def add_files(self, paths, options=None, background=True):
        def work(report):
            scanned = []
            for index, path in enumerate(paths):
                report(index / float(len(paths)), "Reading %s (%d of %d)" % (
                    os.path.basename(path), index + 1, len(paths)))
                scanned.append((path, nxs_loaders.list_entries(path, options)))
            return scanned

        if background and len(paths) > 1:
            if T.run_job(self.root, "Loading data", work,
                         on_done=lambda scanned: self._add_scanned(scanned, options)):
                return
        self._add_scanned(work(lambda *a, **k: None), options)

    def _add_scanned(self, scanned, options=None):
        added = 0
        for path, datasets in scanned:
            name = os.path.basename(path)
            if not datasets:
                datasets = [{"entry": None, "kind": "unknown", "title": None,
                             "start_time": None}]
            for info in datasets:
                source = info.get("path") or path
                key = (os.path.abspath(source), info["entry"])
                if key in self.items:
                    continue
                record = {"memory": None, "backing": None, "exported": True,
                          "options": options}
                record.update(info)
                record["path"] = source
                record["name"] = info.get("name") or name
                self._insert(key, record)
                added += 1
        self.say("Added %d dataset(s); %d in the list" % (added, len(self.items)))

    def _insert(self, key, record):
        self.items[key] = record
        self.order.append(key)
        iid = "r%d" % self._counter
        self._counter += 1
        record["iid"] = iid
        where = (record.get("path") or "") if record.get("memory") is None else \
            ("computed" + (", auto-saved" if record.get("backing") else ""))
        self.tree.insert("", "end", iid=iid, text=record["name"],
                         values=(record["kind"], record.get("entry") or "", where))
        return iid

    def _key_of(self, iid):
        for key, record in self.items.items():
            if record.get("iid") == iid:
                return key
        return None

    def names(self):
        return [record["name"] for record in self.items.values()]

    def unique_name(self, base):
        return T.unique_name(base, self.names())

    def autosave(self, data, name):
        if getattr(data, "kind", None) in NO_AUTOSAVE_KINDS:
            return None
        try:
            return self.session.store_data(data, name)
        except Exception as exc:                            # noqa: BLE001
            traceback.print_exc()
            self.say("Could not auto-save '%s': %s" % (name, exc))
            return None

    def add_dataset(self, data, label=None):
        """List a dataset computed in this session (and auto-save it)."""
        kind = KIND_LABELS.get(getattr(data, "kind", ""), getattr(data, "kind", "unknown"))
        label = self.unique_name(label or getattr(data, "source_label", "computed"))
        if hasattr(data, "source_label"):
            data.source_label = label
        key = ("<computed>", "%s#%d" % (label, self._counter))
        backing = self.autosave(data, label)
        record = {"path": getattr(data, "path", ""), "name": label, "entry": None,
                  "kind": kind, "title": None, "start_time": None, "memory": data,
                  "backing": backing, "exported": False}
        iid = self._insert(key, record)
        self.tree.selection_set(iid)
        self.tree.see(iid)
        self.say("Added %s [%s] to the list%s" % (label, kind,
                                                  " and auto-saved it" if backing else ""))
        return key

    # ------------------------------------------------------------------
    # Selection, reading
    # ------------------------------------------------------------------
    def selected_keys(self):
        return [k for k in (self._key_of(i) for i in self.tree.selection()) if k is not None]

    def label_for(self, key):
        record = self.items.get(key)
        if record is None:
            return "(unknown)"
        if record.get("memory") is None and record.get("entry"):
            siblings = [k for k in self.items if k[0] == key[0]]
            if len(siblings) > 1:
                return "%s \u00b7 %s" % (record["name"], record["entry"])
        return record["name"]

    def entries(self):
        """``(label, key)`` for every row -- for a viewer that needs another
        dataset."""
        return [("%s   [%s]" % (self.label_for(k), self.items[k]["kind"]), k)
                for k in self.order if k in self.items]

    def kind_of(self, key):
        return label_kind(self.items.get(key, {}).get("kind"))

    def load(self, key):
        """The dataset behind a row: in memory, re-read from its auto-saved
        copy, or read from its file."""
        record = self.items[key]
        if record.get("memory") is not None:
            return record["memory"]
        backing = record.get("backing")
        if backing and os.path.isfile(backing):
            return NxsData.acquire(backing, None)
        options = record.get("options")
        loader = None
        try:
            loader = (nxs_loaders.get_loader(options.loader)
                      if options is not None and getattr(options, "loader", None)
                      else nxs_loaders.detect(record["path"]))
        except Exception:                                   # noqa: BLE001
            loader = None
        if loader is not None and nxs_loaders._accepts_progress(loader.load) and not T.busy():
            def work(report):
                def progress(done, total, label):
                    report(done / float(max(1, total)), "%s  (%d of %d)" % (label, done + 1, total))
                return NxsData.acquire(record["path"], record["entry"], options=options,
                                       progress=progress)
            return T.run_blocking(self.root, "Reading %s" % os.path.basename(record["path"]), work)
        return T.with_wait_cursor(self.root, NxsData.acquire, record["path"],
                                  record["entry"], options=options)

    def on_select(self):
        keys = self.selected_keys()
        if len(keys) != 1:
            return
        key = keys[0]
        record = self.items[key]
        if record["kind"] == "unknown":
            self.fill_axes(None)
            self.say("%s: not recognised by any loader" % record["name"])
            return
        try:
            data = self.load(key)
        except Exception as exc:                            # noqa: BLE001
            traceback.print_exc()
            self.fill_axes(None)
            self.say("Failed to load %s: %s" % (self.label_for(key), exc))
            return
        self.fill_axes(data)
        if self.info_window is not None and self.info_window.winfo_exists():
            self.fill_info(key, data)
        self.say("%s: %s -- double-click to open it" % (self.label_for(key), data.kind))
        if record.get("memory") is None:
            self._hold(data)

    def _hold(self, data):
        """Keep ``data`` (one reference to it) until the next row is
        clicked; see ``self._held``."""
        previous, self._held = self._held, data
        if previous is not None:
            try:
                previous.close()
            except Exception:                               # noqa: BLE001
                pass

    def _release_held(self):
        self._hold(None)

    def fill_axes(self, data):
        for item in self.axis_table.get_children():
            self.axis_table.delete(item)
        if data is None:
            return
        labels = getattr(data.scan, "labels", {}) or {}
        for attribute, role in DATA_INFO_AXES.get(data.kind, ()):
            summary = axis_summary(getattr(data.scan, attribute, None))
            if summary is None:
                continue
            lo, hi, num, step = summary
            self.axis_table.insert("", "end", values=(
                labels.get(attribute) or role, "%.6g" % lo, "%.6g" % hi, num, "%.6g" % step))

    # ------------------------------------------------------------------
    # Information window
    # ------------------------------------------------------------------
    def show_information(self):
        keys = self.selected_keys()
        if len(keys) != 1:
            T.info(self.root, "Information", "Select one dataset.")
            return None
        try:
            data = self.load(keys[0])
        except Exception as exc:                            # noqa: BLE001
            T.warning(self.root, "Information", str(exc))
            return None
        if self.info_window is None or not self.info_window.winfo_exists():
            self._build_info_window()
        self.fill_info(keys[0], data)
        if self.items[keys[0]].get("memory") is None:
            data.close()
        self.info_window.deiconify()
        self.info_window.lift()
        return self.info_window

    def _build_info_window(self):
        win = tk.Toplevel(self.root)
        win.title("Dataset information")
        win.geometry("620x640")
        self.info_window = win
        self.info_title = ttk.Label(win, text="", font=("TkDefaultFont", 10, "bold"))
        self.info_title.pack(pady=4)
        row = ttk.Frame(win)
        row.pack(fill="x", padx=6)
        ttk.Label(row, text="Filter").pack(side="left")
        self.filter_var = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.filter_var)
        entry.pack(side="left", fill="x", expand=True, padx=4)
        entry.bind("<KeyRelease>", lambda e: self.apply_filter())
        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=6, pady=4)
        self.info_table = ttk.Treeview(frame, columns=("field", "value"), show="headings")
        self.info_table.heading("field", text="Field")
        self.info_table.heading("value", text="Value")
        self.info_table.column("field", width=230)
        self.info_table.column("value", width=360)
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.info_table.yview)
        self.info_table.configure(yscrollcommand=ys.set)
        self.info_table.pack(side="left", fill="both", expand=True)
        ys.pack(side="left", fill="y")
        ttk.Button(win, text="Export metadata (.csv)", command=self.export_metadata).pack(pady=4)

    def fill_info(self, key, data):
        self.info_key = key
        self.info_title.configure(text="%s  (kind: %s)" % (self.label_for(key), data.kind))
        self.info_rows = ([(k, v) for k, v in data.scan.info.items()]
                          + [("Motor.%s" % k, v) for k, v in data.scan.fourd_info.items()])
        self.apply_filter()

    def apply_filter(self):
        pattern = self.filter_var.get().strip().lower()
        for item in self.info_table.get_children():
            self.info_table.delete(item)
        for k, v in self.info_rows:
            if not pattern or pattern in str(k).lower() or pattern in str(v).lower():
                self.info_table.insert("", "end", values=(str(k), str(v)))

    def export_metadata(self):
        if not self.info_rows:
            return
        record = self.items.get(getattr(self, "info_key", None))
        stem = T.safe_stem(record["name"] if record else "metadata")
        path = T.save_path(self.info_window, "Export metadata as CSV",
                           stem + "_metadata.csv", [("CSV", "*.csv")],
                           initialdir=self.folder_var.get())
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["field", "value"])
                for k, v in self.info_rows:
                    writer.writerow([k, v])
        except OSError as exc:
            T.warning(self.info_window, "Export", "Could not write:\n%s" % exc)
            return
        self.say("Exported %d metadata rows to %s" % (len(self.info_rows), path))

    # ------------------------------------------------------------------
    # Right-click menu
    # ------------------------------------------------------------------
    def show_menu(self, event):
        row = self.tree.identify_row(event.y)
        if row and row not in self.tree.selection():
            self.tree.selection_set(row)
        keys = self.selected_keys()
        rows = [(self.label_for(k), self.items[k]["kind"]) for k in keys]
        menu = tk.Menu(self.root, tearoff=0)
        entries = [
            ("open", "Open", self.open_selected),
            ("slit_cut", "Open slit cut", lambda: self.open_map_cuts("slit")),
            ("deflector_cut", "Open deflector cut", lambda: self.open_map_cuts("deflector")),
            ("info", "Show information", self.show_information),
            ("rename", "Rename...", self.rename_selected),
            None,
            ("figure", "Plot as a figure...", self.plot_selected_as_figure),
            ("stack", "Stack plot (EDC / MDC)...", self.open_stack_plot),
            ("fit", "MDC / EDC fit...", self.open_curve_fit),
            ("view3d", "3D view...", self.open_volume_view),
            ("arithmetic", "Cut arithmetic on the two...", self.open_cut_arithmetic),
            None,
            ("save", "Save dataset(s)...", self.save_selected),
            None,
            ("remove", "Remove from list", self.remove_selected),
            None,
            ("log", "Session log...", self.show_operations_log),
        ]
        for entry in entries:
            if entry is None:
                menu.add_separator()
                continue
            name, text, command = entry
            enabled, reason = availability(name, rows)
            if enabled:
                menu.add_command(label=text, command=command)
            else:
                menu.add_command(label="%s   (%s)" % (text, reason[:60]), state="disabled")
        menu.tk_popup(event.x_root, event.y_root)

    # ------------------------------------------------------------------
    # Opening viewers
    # ------------------------------------------------------------------
    def open_selected(self):
        keys = self.selected_keys()
        if not keys:
            return None
        key = keys[0]
        hidden = next((w for w in self.viewers if getattr(w, "list_key", None) == key
                       and isinstance(w, viewers.ContourWindow)
                       and getattr(w, "hidden", False)), None)
        if hidden is not None:
            hidden.show_map()
            return hidden
        try:
            data = self.load(key)
        except Exception as exc:                            # noqa: BLE001
            traceback.print_exc()
            T.warning(self.root, "Open", "Could not read %s:\n%s" % (self.label_for(key), exc))
            return None
        return self._open(data, self.label_for(key), key)

    def _open(self, data, label, key=None):
        try:
            window = viewers.open_viewer(self, data, label, self.colormap, self.flip)
        except Exception as exc:                            # noqa: BLE001
            traceback.print_exc()
            T.warning(self.root, "Open", "Could not open %s:\n%s" % (label, exc))
            return None
        if window is None:
            T.info(self.root, "Open", "'%s' (%s) has no view." % (label, data.kind))
            return None
        window.list_key = key
        self.viewers.append(window)
        self.say("Opened %s (%d window(s) open)" % (label, len(self.viewers)))
        return window

    def open_data(self, data):
        """Open a dataset a viewer has just computed (listing it first if
        it is not listed yet)."""
        key = next((k for k, r in self.items.items() if r.get("memory") is data), None)
        if key is None:
            key = self.add_dataset(data)
        return self._open(data, self.items[key]["name"], key)

    def open_map_cuts(self, which):
        opened = []
        for key in self.selected_keys():
            contour = next((w for w in self.viewers if getattr(w, "list_key", None) == key
                            and isinstance(w, viewers.ContourWindow)), None)
            if contour is None:
                try:
                    data = self.load(key)
                except Exception as exc:                    # noqa: BLE001
                    T.warning(self.root, "Open a map's cut", str(exc))
                    continue
                contour = self._open(data, self.label_for(key), key)
                if not isinstance(contour, viewers.ContourWindow):
                    continue
                contour.hidden = True
                contour.withdraw()
            opened.append(contour.open_cut(which))
        return opened

    def forget_viewer(self, window):
        if window in self.viewers:
            self.viewers.remove(window)

    # ------------------------------------------------------------------
    # Operations on the selection
    # ------------------------------------------------------------------
    def _load_selected(self, title, kinds=None):
        keys = self.selected_keys()
        if not keys:
            T.info(self.root, title, "Select the dataset(s) first.")
            return []
        out, problems = [], []
        for key in keys:
            try:
                data = self.load(key)
            except Exception as exc:                        # noqa: BLE001
                problems.append("%s: could not be read (%s)" % (self.label_for(key), exc))
                continue
            if kinds is not None and data.kind not in kinds:
                problems.append("%s: %s is not usable here" % (self.label_for(key), data.kind))
                continue
            out.append((self.label_for(key), data))
        if problems:
            T.warning(self.root, title, "\n".join(problems))
        return out

    def open_processing(self):
        datasets = self._load_selected("Process", ("cut",) + CUBE_KINDS)
        if not datasets:
            return None
        kinds = {d.kind for _, d in datasets}
        if len(kinds) > 1 and not kinds <= set(CUBE_KINDS):
            T.warning(self.root, "Process", "Cuts and maps need different panels; "
                      "select one kind at a time.")
            return None
        if kinds == {"cut"}:
            from ui.process import ProcessDialog
            return ProcessDialog(self, datasets, colormap=self.colormap, flip=self.flip)
        from ui.volume import VolumeProcessDialog
        return VolumeProcessDialog(self, datasets, colormap=self.colormap, flip=self.flip)

    def open_data_operations(self):
        from ui.analysis import DataOperationsDialog
        datasets = self._load_selected("Data operations",
                                       tuple(k for k in ARRAY_AXES if k not in CURVE_KINDS))
        if not datasets:
            return None
        shapes = [(d.kind, tuple(np.asarray(getattr(d.scan, a)).size
                                 for a in ARRAY_AXES[d.kind])) for _, d in datasets]
        if not same_format(shapes):
            T.warning(self.root, "Data operations",
                      "The selected datasets are not the same format:\n" + "\n".join(
                          "  %s: %s %s" % (n, k, " x ".join(map(str, s)))
                          for (n, _), (k, s) in zip(datasets, shapes)))
            return None
        return DataOperationsDialog(self, datasets)

    def plot_selected_as_figure(self):
        from ui.figure import FigureWindow, panel_from_dataset
        datasets = self._load_selected("Figure")
        panels, problems = [], []
        for name, data in datasets:
            try:
                panels.append(panel_from_dataset(data, name, self.colormap, self.flip))
            except Exception as exc:                        # noqa: BLE001
                problems.append("%s: %s" % (name, exc))
        if problems:
            T.warning(self.root, "Figure", "\n".join(problems))
        if not panels:
            return None
        return FigureWindow(self.root, panels, "Figure", app=self)

    def open_stack_plot(self):
        from ui.process import StackWindow
        datasets = self._load_selected("Stack plot", ("cut",))
        if not datasets:
            return None
        name, data = datasets[0]
        s = data.scan
        return StackWindow(self.root, np.asarray(s.value, dtype=float), (s.x, s.y),
                           (s.labels.get("x", "x"), s.labels.get("y", "y")), name, app=self)

    def open_curve_fit(self):
        from ui.fit import FitPanel
        from ui.analysis import momentum_cut_reason
        datasets = self._load_selected("MDC / EDC fit", ("cut",))
        if not datasets:
            return None
        name, data = datasets[0]
        reason = momentum_cut_reason(data)
        if reason:
            T.info(self.root, "MDC / EDC fit", reason)
            return None
        return FitPanel(self, data, name, colormap=self.colormap, flip=self.flip)

    def open_volume_view(self):
        from ui.volume import open_volume_view
        datasets = self._load_selected("3D view", CUBE_KINDS)
        if not datasets:
            return None
        name, data = datasets[0]
        return open_volume_view(self.root, self, data, name, self.colormap, self.flip)

    def open_cut_arithmetic(self):
        from ui.cutops import CutArithmeticDialog
        datasets = self._load_selected("Cut arithmetic", ("cut",))
        if len(datasets) != 2:
            if datasets:
                T.info(self.root, "Cut arithmetic", "Select exactly two cuts.")
            return None
        try:
            return CutArithmeticDialog(self, datasets[0], datasets[1],
                                       colormap=self.colormap, flip=self.flip)
        except ValueError as exc:
            T.info(self.root, "Cut arithmetic", str(exc))
            return None

    # ------------------------------------------------------------------
    # Rename, remove, save
    # ------------------------------------------------------------------
    def rename_selected(self):
        keys = self.selected_keys()
        if len(keys) != 1:
            return
        record = self.items[keys[0]]
        name = T.ask_string(self.root, "Rename dataset", "Name:", record["name"])
        if not name or name == record["name"]:
            return
        if name in self.names():
            T.warning(self.root, "Rename", "'%s' is already in the list." % name)
            return
        record["name"] = name
        if record.get("memory") is not None and hasattr(record["memory"], "source_label"):
            record["memory"].source_label = name
        self.tree.item(record["iid"], text=name)

    def remove_selected(self):
        keys = self.selected_keys()
        if not keys:
            return
        computed = [self.items[k]["name"] for k in keys if not self.items[k].get("exported")]
        detail = ""
        if computed:
            detail = ("\n\n%d of them were computed in this session and not saved "
                      "anywhere of your own; their auto-saved copies go too." % len(computed))
        if not T.ask_yes_no(self.root, "Remove from list",
                            "Remove %d dataset(s) from the list?\n\nFiles on disk are "
                            "not deleted.%s" % (len(keys), detail)):
            return
        self._remove_keys(keys)

    def _remove_keys(self, keys):
        self._release_held()
        for key in keys:
            record = self.items.pop(key, None)
            if key in self.order:
                self.order.remove(key)
            if record is None:
                continue
            if record.get("backing"):
                self.session.discard(record["backing"])
            try:
                self.tree.delete(record["iid"])
            except tk.TclError:
                pass
        self.fill_axes(None)
        self.say("Removed %d dataset(s) from the list" % len(keys))

    def clear_list(self):
        keys = list(self.items)
        if not keys:
            self.say("The list is already empty")
            return False
        pending = self.unsaved()
        if pending:
            if not self.offer_to_save("Clear list", pending, "Clear all %d dataset(s)?" % len(keys)):
                return False
        elif not T.ask_yes_no(self.root, "Clear list",
                              "Clear all %d dataset(s) from the list?\n\nFiles on disk "
                              "are not deleted." % len(keys)):
            return False
        self._remove_keys(keys)
        self.say("List cleared")
        return True

    def save_selected(self, keys=None):
        keys = keys or self.selected_keys()
        if not keys:
            return False
        default = self.items[keys[0]]["name"] if len(keys) == 1 else "datasets"
        path = T.save_path(self.root, "Save dataset(s)", T.safe_stem(default) + ".npz",
                           [("ARPES viewer data (numpy)", "*.npz")],
                           initialdir=self.folder_var.get())
        if not path:
            return False
        if not path.lower().endswith(".npz"):
            path += ".npz"
        payload, problems, saved = [], [], []
        for key in keys:
            record = self.items[key]
            try:
                data = self.load(key)
                item = nxs_session.scan_to_dict(data.kind, data.scan, record["name"])
                item["value"] = np.asarray(item["value"])
                payload.append(item)
                saved.append(key)
            except Exception as exc:                        # noqa: BLE001
                problems.append("%s: %s" % (record["name"], exc))
        if not payload:
            T.warning(self.root, "Save", "Nothing could be saved:\n" + "\n".join(problems))
            return False
        try:
            entries = T.with_wait_cursor(self.root, save_dataset, path, payload)
        except Exception as exc:                            # noqa: BLE001
            traceback.print_exc()
            T.warning(self.root, "Save", "Could not write the file:\n%s" % exc)
            return False
        for key, entry in zip(saved, entries):
            record = self.items[key]
            record["exported"] = True
            if record.get("backing"):
                self.session.released(record["backing"], "%s::%s" % (os.path.basename(path), entry))
                record["backing"] = None
            self.tree.set(record["iid"], "where", "saved to %s" % path)
        message = "Saved %d dataset(s) to %s" % (len(payload), os.path.basename(path))
        if problems:
            T.info(self.root, "Save", message + "; skipped:\n" + "\n".join(problems))
        self.say(message)
        return not problems

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    def _in_a_window(self, data):
        for window in self.viewers:
            if getattr(window, "data", None) is data:
                return True
        return False

    def memory_datasets(self):
        """The computed datasets held in memory, for the Memory panel."""
        rows = []
        for key in self.order:
            record = self.items.get(key)
            data = record.get("memory") if record else None
            if data is None:
                continue
            backing = record.get("backing")
            rows.append({"key": key, "name": record["name"],
                         "bytes": memory_tools.array_bytes(data),
                         "autosaved": bool(backing and os.path.isfile(backing)),
                         "open": self._in_a_window(data)})
        return rows

    def memory_windows(self):
        rows = []
        for window in list(self.viewers):
            try:
                title = window.title()
            except tk.TclError:
                continue
            rows.append({"window": window, "title": title,
                         "bytes": memory_tools.array_bytes(getattr(window, "data", None))})
        return rows

    def open_file_count(self):
        from loader.nxs_file import HANDLES
        return HANDLES.open_count()

    def unload_datasets(self, keys=None):
        """Drop the in-memory copy of computed datasets that are auto-saved
        and not in use; the row then reads its auto-saved file when opened.
        Returns ``(count, bytes, [(name, why kept), ...])``."""
        keys = list(keys) if keys is not None else \
            [k for k in self.order if self.items.get(k, {}).get("memory") is not None]
        count, freed, skipped = 0, 0, []
        # A closed window is a cycle of Tk widgets that still points at its
        # dataset until the garbage collector has run; collect first so
        # that it does not count as a user below.
        import gc
        gc.collect()
        for key in keys:
            record = self.items.get(key)
            if record is None or record.get("memory") is None:
                continue
            data = record["memory"]
            backing = record.get("backing")
            if not (backing and os.path.isfile(backing)):
                skipped.append((record["name"], "not auto-saved (save it first)"))
                continue
            if self._in_a_window(data):
                skipped.append((record["name"], "open in a window"))
                continue
            # 3 = this record, the local name, getrefcount's own argument:
            # anything more is some window or tool still using it, and
            # dropping the list's copy would free nothing.
            if sys.getrefcount(data) > 3:
                skipped.append((record["name"], "still used by a tool window"))
                continue
            freed += memory_tools.array_bytes(data)
            record["memory"] = None
            record["path"] = backing
            record["options"] = None
            count += 1
            try:
                self.tree.set(record["iid"], "where", "computed, on disk (auto-saved)")
            except tk.TclError:
                pass
            del data
        return count, freed, skipped

    def free_memory(self, unload=False):
        """Give back what can be given back; a one-line report."""
        before = memory_tools.process_memory()["rss"]
        self._release_held()
        cached = memory_tools.free_reader_caches()
        parts = ["caches %s" % memory_tools.fmt_bytes(cached)]
        if unload:
            count, freed, skipped = self.unload_datasets()
            parts.append("%d dataset(s) moved to disk (%s)" % (count, memory_tools.fmt_bytes(freed)))
            if skipped:
                parts.append("%d kept (in use / not saved)" % len(skipped))
        memory_tools.trim()
        after = memory_tools.process_memory()["rss"]
        text = "Freed: " + ", ".join(parts)
        if before is not None and after is not None:
            text += ". This program: %s -> %s" % (memory_tools.fmt_bytes(before),
                                                   memory_tools.fmt_bytes(after))
        return text

    def save_config(self):
        self.config["FilePathLE"] = self.folder_var.get()
        self.config["ColorMap"] = self.colormap
        self.config["FlipColorMap"] = self.flip
        try:
            system.dump_config(CONFIG_PATH, self.config)
        except Exception:                                   # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Session: log, recovery, closing
    # ------------------------------------------------------------------
    def show_operations_log(self):
        rows = nxs_session.read_log(self.session.root)
        text = "\n".join(" | ".join(row[:6]) for row in rows) or "Nothing logged yet."
        T.TextWindow(self.root, "Operations log -- %s" % self.session.log_path, text)

    def offer_recovery(self):
        leftovers = self.session.leftovers()
        if not leftovers:
            return 0
        total_files = sum(len(e["files"]) for e in leftovers)
        total_bytes = sum(e["bytes"] for e in leftovers)
        newest = leftovers[0]
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(newest["modified"]))
        if not T.ask_yes_no(
                self.root, "Recover unsaved work",
                "%d dataset(s) from %d earlier session(s) were never saved to a file "
                "of your own.\n\nMost recent: %s (%.0f MB)\nTotal: %.0f MB\n\nAdd them "
                "to the list?\n\nThey are in %s and are deleted after %d days."
                % (total_files, len(leftovers), when, newest["bytes"] / 1e6,
                   total_bytes / 1e6, self.session.root, nxs_session.KEEP_DAYS),
                default_no=False):
            return 0
        paths = [os.path.join(e["folder"], f) for e in leftovers for f in e["files"]]
        self.add_files(paths, background=False)
        for record in self.items.values():
            if os.path.dirname(record["path"]).startswith(self.session.root):
                record["exported"] = False
                record["backing"] = record["path"]
        self.session.log("RECOVER", "%d dataset(s)" % len(paths), self.session.root,
                         "listed at startup")
        return len(paths)

    def unsaved(self):
        return [r["name"] for r in self.items.values()
                if (r.get("backing") or r.get("memory") is not None) and not r.get("exported")]

    def offer_to_save(self, title, pending, text):
        shown = "\n".join("  \u2022 %s" % n for n in pending[:8])
        if len(pending) > 8:
            shown += "\n  \u2022 ... and %d more" % (len(pending) - 8)
        from tkinter import messagebox
        answer = messagebox.askyesnocancel(
            title, "%s\n\n%d dataset(s) computed in this session have not been saved "
            "to a file of your own:\n%s\n\nThey are auto-saved to %s (cleared after "
            "%d days).\n\nYes = save them to a file first, No = go on without saving, "
            "Cancel = stop." % (text, len(pending), shown, self.session.folder,
                                nxs_session.KEEP_DAYS), parent=self.root)
        if answer is None:
            return False
        if answer is False:
            return True
        keys = [k for k, r in self.items.items() if r["name"] in pending]
        self.save_selected(keys)
        return not self.unsaved()

    def on_close(self):
        if T.busy() and not T.ask_yes_no(self.root, "Close", "Something is still running. Close anyway?"):
            return
        pending = self.unsaved()
        if pending and not self.offer_to_save("Close ARPES viewer", pending, "Close the program?"):
            return
        self.save_config()
        self._release_held()
        self.session.log("END", "", self.session.folder, "session closed")
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass
    app = App(root)
    app.session.log("START", "", app.session.folder, "session opened")
    try:
        gone = app.session.prune_old()
        if gone:
            app.say("Cleared %d old session folder(s)" % gone)
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
    root.after(200, lambda: _safe(app.offer_recovery))
    root.mainloop()


def _safe(function):
    try:
        function()
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()


if __name__ == "__main__":
    main()
