"""
ui/loader_dialog.py
===================
The Load-data window: files, which beamline's reader, and the axis options,
all decided before anything reaches the list. Only *structure* is read here
(entry names, kinds), which is cheap even for a multi-GB scan.
"""
import os

import tkinter as tk
from tkinter import ttk, filedialog

from loader import registry
from loader.registry import AXIS_ROLES, LoadOptions
from ui import tkbase as T

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
        self.start_folder = start_folder
        ttk.Label(self, wraplength=620, justify="left", text=(
            "Choose the files, check that the right reader was recognised, and "
            "say what the axes are. Only structure is read here -- the "
            "measurements themselves are read when you open one.")).pack(
            fill="x", padx=8, pady=6)

        files = ttk.LabelFrame(self, text="Files")
        files.pack(fill="both", expand=True, padx=8)
        row = ttk.Frame(files)
        row.pack(fill="x")
        self.folder_var = tk.StringVar(value=start_folder)
        ttk.Entry(row, textvariable=self.folder_var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=3)
        ttk.Button(row, text="Select file(s)...", command=self.pick_files).pack(side="left")
        ttk.Button(row, text="Whole folder...", command=self.pick_folder).pack(side="left", padx=3)
        self.file_list = tk.Listbox(files, height=6)
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
        self.entry_list = tk.Listbox(found, height=8)
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
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side="right", padx=3)
        ttk.Button(bar, text="Add to list", command=self.ok).pack(side="right", padx=3)

    def pick_files(self):
        start = self.folder_var.get() or os.path.expanduser("~")
        if not os.path.isdir(start):
            start = os.path.expanduser("~")
        paths = filedialog.askopenfilenames(parent=self, title="Select data file(s)",
                                            initialdir=start, filetypes=file_types())
        if paths:
            self.set_paths(list(paths))

    def pick_folder(self):
        start = self.folder_var.get() or os.path.expanduser("~")
        folder = filedialog.askdirectory(parent=self, initialdir=start,
                                         title="Folder: every recognised file in it")
        if not folder:
            return
        patterns = set()
        for loader in registry.loaders():
            patterns.update(p.lstrip("*").lower() for p in loader.patterns)
        paths = sorted(os.path.join(folder, n) for n in os.listdir(folder)
                       if os.path.isfile(os.path.join(folder, n))
                       and os.path.splitext(n)[1].lower() in patterns)
        if not paths:
            T.info(self, "Load data", "No data files in that folder.")
            return
        self.set_paths(paths)

    def set_paths(self, paths):
        self.paths = list(paths)
        self.folder_var.set(os.path.dirname(self.paths[0]) if self.paths else "")
        self.file_list.delete(0, "end")
        for path in self.paths:
            self.file_list.insert("end", os.path.basename(path))
        counts = {}
        for path in self.paths:
            loader = registry.detect(path)
            name = loader.name if loader else "not recognised"
            counts[name] = counts.get(name, 0) + 1
        self.detected.configure(text="Detected: " + ", ".join(
            "%d × %s" % (n, name) for name, n in sorted(counts.items())))
        self.refresh_entries()

    def options(self):
        loader = self.loader_var.get()
        perm = dict(PERMUTATIONS)[self.perm_var.get()]
        role = self.role_keys[self.role_names.index(self.role_var.get())]
        return LoadOptions(loader=None if loader == AUTO else loader,
                           permutation=perm, axis0_role=role,
                           axis0_label=self.label_var.get().strip())

    def refresh_entries(self):
        self.entry_list.delete(0, "end")
        options = self.options()
        for path in self.paths[:200]:
            try:
                entries = registry.list_entries(path, options)
            except Exception as exc:                            # noqa: BLE001
                self.entry_list.insert("end", "%s: %s" % (os.path.basename(path), exc))
                continue
            if not entries:
                self.entry_list.insert("end", "%s: nothing recognised" % os.path.basename(path))
            for info in entries:
                entry = info.get("entry")
                name = os.path.basename(path)
                self.entry_list.insert("end", "%s  ·  %s   [%s]" % (name, entry, info.get("kind"))
                                       if entry else "%s   [%s]" % (name, info.get("kind")))

    def ok(self):
        if not self.paths:
            T.info(self, "Load data", "Select one or more files first.")
            return
        self.result = (self.paths, self.options())
        self.destroy()

    def ask(self):
        self.grab_set()
        self.master.wait_window(self)
        return self.result
