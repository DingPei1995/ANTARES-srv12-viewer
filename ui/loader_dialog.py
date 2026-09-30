"""
ui/loader_dialog.py
===================
The Load-data window: files, which beamline's reader, and the axis options,
all decided before anything reaches the list. Only *structure* is read here
(entry names, kinds), which is cheap even for a multi-GB scan.

The files are picked in a browser drawn in this window
(:class:`ui.filebrowser.FileBrowser`), not in the Tk file dialog, which on
the server listed folders but not the files in them. What the chosen files
contain is worked out on a worker thread, so the window never freezes on a
long list of files, and each file is only looked at once.
"""
import os
import threading

import tkinter as tk
from tkinter import ttk, filedialog

from loader import registry
from loader.registry import AXIS_ROLES, LoadOptions
from ui import tkbase as T
from ui.filebrowser import FileBrowser, list_data_files

PERMUTATIONS = [
    ("Leave as the file has it", None),
    ("3-D: swap the first two axes (1 <-> 2)", (1, 0, 2)),
    ("3-D: swap the last two axes (2 <-> 3)", (0, 2, 1)),
    ("3-D: swap the outer two axes (1 <-> 3)", (2, 1, 0)),
    ("3-D: rotate forwards (3, 1, 2)", (2, 0, 1)),
    ("3-D: rotate backwards (2, 3, 1)", (1, 2, 0)),
    ("2-D: swap the two axes (transpose)", (1, 0)),
]
AUTO = "Detect automatically"
#: The "Datasets found" list looks into at most this many files.
MAX_LISTED = 200
AS_FILE = "As the file says"


def file_types():
    every, per_loader = [], []
    for loader in registry.loaders():
        per_loader.append((loader.name, " ".join(loader.patterns)))
        every.extend(loader.patterns)
    joined = " ".join(dict.fromkeys(every))
    return [("Data files", joined)] + per_loader + [("All files", "*")]


class LoaderDialog(tk.Toplevel):
    """Modal. ``ask()`` returns ``(paths, LoadOptions)`` or None."""

    def __init__(self, parent, start_folder=""):
        tk.Toplevel.__init__(self, parent)
        self.title("Load data")
        self.transient(parent)
        self.result = None
        self.paths = []
        self.last_folder = start_folder
        self.start_folder = start_folder
        ttk.Label(self, wraplength=620, justify="left", text=(
            "Choose the files, check that the right reader was recognised, and "
            "say what the axes are. Only structure is read here -- the "
            "measurements themselves are read when you open one.")).pack(
            fill="x", padx=8, pady=6)

        self._entry_cache = {}                 # (path, loader) -> entries / error
        self._scan_token = None

        browse = ttk.LabelFrame(self, text="Browse")
        browse.pack(fill="both", expand=True, padx=8)
        self.browser = FileBrowser(browse, start_folder, on_activate=self.add_paths,
                                   height=8)
        self.browser.pack(fill="both", expand=True, padx=3, pady=3)
        row = ttk.Frame(browse)
        row.pack(fill="x", padx=3, pady=(0, 3))
        ttk.Button(row, text="Add selected", command=self.add_selected).pack(side="left")
        ttk.Button(row, text="Add every data file in this folder",
                   command=self.add_folder).pack(side="left", padx=3)
        ttk.Label(row, foreground="#555", text=(
            "double-click a file to add it; Shift / Ctrl + click for several")).pack(side="left", padx=6)

        files = ttk.LabelFrame(self, text="Files to load")
        files.pack(fill="both", expand=False, padx=8, pady=(4, 0))
        row = ttk.Frame(files)
        row.pack(fill="x")
        self.folder_var = tk.StringVar(value=start_folder)
        self.chosen_count = ttk.Label(row, text="none yet")
        self.chosen_count.pack(side="left", padx=3)
        ttk.Button(row, text="Clear", command=lambda: self.set_paths([])).pack(side="right")
        ttk.Button(row, text="Remove selected", command=self.remove_chosen).pack(side="right", padx=3)
        ttk.Button(row, text="System dialog...", command=self.pick_files).pack(side="right")
        self.file_list = tk.Listbox(files, height=4, selectmode="extended")
        self.file_list.pack(fill="both", expand=True, padx=3, pady=3)

        reader = ttk.LabelFrame(self, text="Reader")
        reader.pack(fill="x", padx=8, pady=4)
        self.loader_var = tk.StringVar(value=AUTO)
        combo = ttk.Combobox(reader, textvariable=self.loader_var, state="readonly",
                             width=40, values=[AUTO] + [l.name for l in registry.loaders()])
        combo.pack(anchor="w", padx=3, pady=3)
        combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_entries())
        self.detected = ttk.Label(reader, text="", wraplength=600, justify="left")
        self.detected.pack(fill="x", padx=3)

        found = ttk.LabelFrame(self, text="Datasets found")
        found.pack(fill="both", expand=True, padx=8, pady=4)
        self.entry_list = tk.Listbox(found, height=5)
        self.entry_list.pack(fill="both", expand=True, padx=3, pady=3)

        axes = ttk.LabelFrame(self, text="Axes")
        axes.pack(fill="x", padx=8, pady=4)
        self.perm_var = tk.StringVar(value=PERMUTATIONS[0][0])
        ttk.Label(axes, text="Axis order").grid(row=0, column=0, sticky="w", padx=3)
        ttk.Combobox(axes, textvariable=self.perm_var, state="readonly", width=40,
                     values=[p[0] for p in PERMUTATIONS]).grid(row=0, column=1, sticky="w")
        self.role_names = [AS_FILE] + [
            ("%s (%s)" % (label, unit) if unit else label)
            for key, (label, unit, _c) in AXIS_ROLES.items()]
        self.role_keys = [None] + list(AXIS_ROLES)
        self.role_var = tk.StringVar(value=AS_FILE)
        ttk.Label(axes, text="First axis of a map is").grid(row=1, column=0, sticky="w", padx=3)
        ttk.Combobox(axes, textvariable=self.role_var, state="readonly", width=40,
                     values=self.role_names).grid(row=1, column=1, sticky="w")
        self.label_var = tk.StringVar()
        ttk.Label(axes, text="Label it (optional)").grid(row=2, column=0, sticky="w", padx=3)
        ttk.Entry(axes, textvariable=self.label_var, width=42).grid(row=2, column=1, sticky="w")

        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x", padx=8, pady=6, before=browse)
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side="right", padx=3)
        ttk.Button(bar, text="Add to list", command=self.ok).pack(side="right", padx=3)

    # -- choosing files ------------------------------------------------------
    def add_paths(self, paths):
        known = set(self.paths)
        self.set_paths(self.paths + [p for p in paths if p not in known])

    def add_selected(self):
        paths = self.browser.selected_paths()
        if not paths:
            T.info(self, "Load data", "Select one or more files in the browser first "
                   "(Shift / Ctrl + click for several).")
            return
        self.add_paths(paths)

    def add_folder(self):
        folder = self.browser.folder
        try:
            paths = list_data_files(folder)
        except OSError as exc:
            T.warning(self, "Load data", "Cannot read %s:\n%s" % (folder, exc))
            return
        if not paths:
            T.info(self, "Load data", "No data files in that folder.")
            return
        self.add_paths(paths)

    def remove_chosen(self):
        drop = set(int(i) for i in self.file_list.curselection())
        self.set_paths([p for i, p in enumerate(self.paths) if i not in drop])

    def pick_files(self):
        """The system (Tk) file dialog -- kept as a fallback."""
        start = self.browser.folder or os.path.expanduser("~")
        paths = filedialog.askopenfilenames(parent=self, title="Select data file(s)",
                                            initialdir=start, filetypes=file_types())
        if paths:
            self.add_paths(list(paths))

    def pick_folder(self):
        folder = filedialog.askdirectory(parent=self, initialdir=self.browser.folder,
                                         title="Folder: every recognised file in it")
        if folder:
            self.browser.go(folder)
            self.add_folder()

    def set_paths(self, paths):
        self.paths = list(paths)
        if self.paths:
            self.folder_var.set(os.path.dirname(self.paths[0]))
        self.file_list.delete(0, "end")
        for path in self.paths:
            self.file_list.insert("end", path)
        self.chosen_count.configure(text="%d file(s)" % len(self.paths) if self.paths
                                    else "none yet")
        self.refresh_entries()

    # -- what is in them -----------------------------------------------------
    def _loader_choice(self):
        loader = self.loader_var.get()
        return None if loader == AUTO else loader

    def refresh_entries(self):
        """List the datasets of the chosen files. Files not looked at yet
        are read on a worker thread (what it is reading is shown in the
        list meanwhile); the rest come from what was found before."""
        choice = self._loader_choice()
        todo = [p for p in self.paths[:MAX_LISTED] if (p, choice) not in self._entry_cache]
        if not todo:
            self._show_entries()
            return
        options = self.options()
        cache = self._entry_cache
        self._scan_token = token = object()
        progress = {"text": ""}

        def work():
            for index, path in enumerate(todo):
                if self._scan_token is not token:      # superseded / closed
                    return
                progress["text"] = "Looking into %s (%d of %d)..." % (
                    os.path.basename(path), index + 1, len(todo))
                try:
                    loader = registry.get_loader(choice) if choice else registry.detect(path)
                    entries = registry.list_entries(path, options)
                    cache[(path, choice)] = (loader.name if loader else None, entries, None)
                except Exception as exc:                            # noqa: BLE001
                    cache[(path, choice)] = (None, [], exc)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self.detected.configure(text="")

        def poll():
            if self._scan_token is not token or not self.winfo_exists():
                return
            if thread.is_alive():
                self.entry_list.delete(0, "end")
                self.entry_list.insert("end", progress["text"] or "Looking into the files...")
                self.after(150, poll)
                return
            self._show_entries()
        self.after(50, poll)

    def _show_entries(self):
        if not self.winfo_exists():
            return
        choice = self._loader_choice()
        self.entry_list.delete(0, "end")
        counts = {}
        for path in self.paths[:MAX_LISTED]:
            found = self._entry_cache.get((path, choice))
            name = os.path.basename(path)
            if found is None:
                self.entry_list.insert("end", "%s: not looked at (cancelled)" % name)
                continue
            loader_name, entries, error = found
            counts[loader_name or "not recognised"] = counts.get(loader_name or "not recognised", 0) + 1
            if error is not None:
                self.entry_list.insert("end", "%s: %s" % (name, error))
                continue
            if not entries:
                self.entry_list.insert("end", "%s: nothing recognised" % name)
            for info in entries:
                entry = info.get("entry")
                self.entry_list.insert("end", "%s  \u00b7  %s   [%s]" % (name, entry, info.get("kind"))
                                       if entry else "%s   [%s]" % (name, info.get("kind")))
        if len(self.paths) > MAX_LISTED:
            self.entry_list.insert("end", "... and %d more file(s), not listed here"
                                   % (len(self.paths) - MAX_LISTED))
        self.detected.configure(text=("Detected: " + ", ".join(
            "%d \u00d7 %s" % (n, name) for name, n in sorted(counts.items()))) if counts else "")

    def options(self):
        loader = self.loader_var.get()
        perm = dict(PERMUTATIONS)[self.perm_var.get()]
        role = self.role_keys[self.role_names.index(self.role_var.get())]
        return LoadOptions(loader=None if loader == AUTO else loader,
                           permutation=perm, axis0_role=role,
                           axis0_label=self.label_var.get().strip())

    def ok(self):
        if not self.paths:
            T.info(self, "Load data", "Select one or more files first.")
            return
        self.result = (self.paths, self.options())
        self.last_folder = self.browser.folder
        self.destroy()

    def destroy(self):
        self._scan_token = None                # stops a scan in progress
        tk.Toplevel.destroy(self)

    def ask(self):
        self.grab_set()
        self.master.wait_window(self)
        return self.result
