"""
ui/filebrowser.py
=================
A folder browser drawn inside the program's own windows.

The Tk file dialog on the server (``tk_getOpenFile`` on X11) did not list the
files of a folder -- only the folders -- so a measurement could only be
loaded through *Whole folder...*. This browser lists a folder with
``os.scandir`` and shows every file in it, with its size, date and which
reader will open it, so no system dialog is involved at all:

* type or paste a folder and press Enter, or go *Up* / *Home*, or
  double-click a folder;
* show only the data files the readers recognise (by extension, in any
  letter case), or every file; narrow it with a name filter (``*`` and ``?``
  wildcards, or plain text);
* select one or several files (Shift / Ctrl + click) -- double-click a file
  to take it straight away.
"""
import fnmatch
import os
import time

import tkinter as tk
from tkinter import ttk, filedialog

from loader import registry

DATA_FILES = "Data files"
ALL_FILES = "All files"


def data_extensions():
    """``{".nxs": "SOLEIL ANTARES, ...", ...}`` -- the extensions the
    installed readers claim, lower case, with who claims them."""
    out = {}
    for loader in registry.loaders():
        for pattern in loader.patterns:
            ext = os.path.splitext(pattern)[1].lower()
            if ext:
                names = out.setdefault(ext, [])
                if loader.name not in names:
                    names.append(loader.name)
    return {ext: ", ".join(names) for ext, names in out.items()}


def is_data_file(name, extensions=None):
    extensions = extensions if extensions is not None else data_extensions()
    return os.path.splitext(name)[1].lower() in extensions


def list_data_files(folder):
    """Every recognised data file directly in ``folder``, sorted."""
    extensions = data_extensions()
    out = []
    with os.scandir(folder) as entries:
        for entry in entries:
            try:
                if entry.is_file() and is_data_file(entry.name, extensions):
                    out.append(entry.path)
            except OSError:
                continue
    return sorted(out)


def _size_text(size):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return ("%d %s" % (size, unit)) if unit == "B" else ("%.1f %s" % (size, unit))
        size /= 1024.0
    return ""


class FileBrowser(ttk.Frame):
    """The folder, its sub-folders and files, and a multi-selection.

    ``on_activate(paths)`` is called when a file is double-clicked (or Enter
    is pressed on a selection of files); ``on_folder(folder)`` whenever the
    folder shown changes.
    """

    def __init__(self, parent, folder="", on_activate=None, on_folder=None,
                 height=12):
        ttk.Frame.__init__(self, parent)
        self.on_activate = on_activate
        self.on_folder = on_folder
        self.extensions = data_extensions()
        self.folder = ""
        self._rows = {}                  # iid -> (path, is_dir)
        self._sort = ("name", False)

        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="Folder").pack(side="left")
        self.folder_var = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.folder_var)
        entry.pack(side="left", fill="x", expand=True, padx=3)
        entry.bind("<Return>", lambda e: self.go(self.folder_var.get()))
        ttk.Button(row, text="Go", width=4,
                   command=lambda: self.go(self.folder_var.get())).pack(side="left")
        ttk.Button(row, text="Up", width=4, command=self.up).pack(side="left", padx=2)
        ttk.Button(row, text="Home", width=6,
                   command=lambda: self.go(os.path.expanduser("~"))).pack(side="left")
        ttk.Button(row, text="Refresh", width=8, command=self.refresh).pack(side="left", padx=2)
        ttk.Button(row, text="Choose folder...", command=self.choose_folder).pack(side="left")

        row2 = ttk.Frame(self)
        row2.pack(fill="x", pady=(3, 0))
        ttk.Label(row2, text="Show").pack(side="left")
        shown = "%s (%s)" % (DATA_FILES, " ".join("*" + e for e in sorted(self.extensions)))
        self._data_label = shown
        self.show_var = tk.StringVar(value=shown)
        combo = ttk.Combobox(row2, textvariable=self.show_var, state="readonly", width=38,
                             values=[shown, ALL_FILES])
        combo.pack(side="left", padx=3)
        combo.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        ttk.Label(row2, text="Name filter").pack(side="left", padx=(8, 0))
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(row2, textvariable=self.name_var, width=18)
        name_entry.pack(side="left", padx=3)
        name_entry.bind("<KeyRelease>", lambda e: self.refresh())
        self.count = ttk.Label(row2, text="", foreground="#555")
        self.count.pack(side="left", padx=6)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=3)
        self.tree = ttk.Treeview(body, columns=("size", "modified", "reader"),
                                 selectmode="extended", height=height)
        for col, text, width, anchor in (("#0", "Name", 300, "w"),
                                         ("size", "Size", 80, "e"),
                                         ("modified", "Modified", 130, "w"),
                                         ("reader", "Opened by", 200, "w")):
            key = "name" if col == "#0" else col
            self.tree.heading(col, text=text, command=lambda k=key: self.sort_by(k))
            self.tree.column(col, width=width, anchor=anchor,
                             stretch=(col in ("#0", "reader")))
        ys = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ys.pack(side="left", fill="y")
        self.tree.bind("<Double-Button-1>", self._double_click)
        self.tree.bind("<Return>", lambda e: self._activate_selection())
        self.tree.bind("<BackSpace>", lambda e: self.up())
        self.tree.tag_configure("dir", foreground="#1f4e8c")
        self.tree.tag_configure("other", foreground="#888")

        self.go(folder if folder and os.path.isdir(folder) else os.path.expanduser("~"))

    # -- navigation ----------------------------------------------------------
    def go(self, folder):
        folder = os.path.abspath(os.path.expanduser(str(folder).strip() or "~"))
        if not os.path.isdir(folder):
            self.count.configure(text="not a folder: %s" % folder, foreground="#a33")
            return False
        self.folder = folder
        self.folder_var.set(folder)
        self.refresh()
        if self.on_folder is not None:
            self.on_folder(folder)
        return True

    def up(self):
        parent = os.path.dirname(self.folder.rstrip(os.sep)) or os.sep
        if parent != self.folder:
            self.go(parent)

    def choose_folder(self):
        folder = filedialog.askdirectory(parent=self.winfo_toplevel(),
                                         initialdir=self.folder or os.path.expanduser("~"))
        if folder:
            self.go(folder)

    # -- listing -------------------------------------------------------------
    def _only_data(self):
        return self.show_var.get() != ALL_FILES

    def _name_matches(self, name):
        pattern = self.name_var.get().strip()
        if not pattern:
            return True
        if any(c in pattern for c in "*?["):
            return fnmatch.fnmatch(name.lower(), pattern.lower())
        return pattern.lower() in name.lower()

    def _scan(self):
        dirs, files = [], []
        try:
            entries = list(os.scandir(self.folder))
        except OSError as exc:
            self.count.configure(text="cannot read folder: %s" % exc, foreground="#a33")
            return dirs, files
        for entry in entries:
            name = entry.name
            if name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    dirs.append((name, entry.path, None, None))
                    continue
                if not entry.is_file():
                    continue
                if self._only_data() and not is_data_file(name, self.extensions):
                    continue
                if not self._name_matches(name):
                    continue
                st = entry.stat()
                files.append((name, entry.path, st.st_size, st.st_mtime))
            except OSError:
                continue
        return dirs, files

    def refresh(self):
        if not self.folder:
            return
        selected = set(self.selected_paths())
        dirs, files = self._scan()
        key, reverse = self._sort
        index = {"name": 0, "size": 2, "modified": 3, "reader": 0}[key]
        if key == "reader":
            files.sort(key=lambda r: (self.extensions.get(os.path.splitext(r[0])[1].lower(), "~"),
                                      r[0].lower()), reverse=reverse)
        elif index == 0:
            files.sort(key=lambda r: r[0].lower(), reverse=reverse)
        else:
            files.sort(key=lambda r: r[index] or 0, reverse=reverse)
        dirs.sort(key=lambda r: r[0].lower())

        self.tree.delete(*self.tree.get_children())
        self._rows = {}
        for name, path, _size, _mtime in dirs:
            iid = self.tree.insert("", "end", text=name + "/",
                                   values=("", "", "folder"), tags=("dir",))
            self._rows[iid] = (path, True)
        reselect = []
        for name, path, size, mtime in files:
            reader = self.extensions.get(os.path.splitext(name)[1].lower(), "")
            iid = self.tree.insert("", "end", text=name, values=(
                _size_text(size), time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)),
                reader or "(not a data file)"), tags=(() if reader else ("other",)))
            self._rows[iid] = (path, False)
            if path in selected:
                reselect.append(iid)
        if reselect:
            self.tree.selection_set(reselect)
        self.count.configure(text="%d file(s), %d folder(s)" % (len(files), len(dirs)),
                             foreground="#555")

    def sort_by(self, key):
        current, reverse = self._sort
        self._sort = (key, (not reverse) if current == key else False)
        self.refresh()

    # -- selection -----------------------------------------------------------
    def selected_paths(self):
        """The files (not folders) selected, in list order."""
        return [self._rows[i][0] for i in self.tree.selection()
                if i in self._rows and not self._rows[i][1]]

    def shown_files(self, data_only=True):
        """Every file listed, or only the recognised data files."""
        return [path for path, is_dir in self._rows.values()
                if not is_dir and (not data_only or is_data_file(path, self.extensions))]

    def _double_click(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid or iid not in self._rows:
            return
        path, is_dir = self._rows[iid]
        if is_dir:
            self.go(path)
        elif self.on_activate is not None:
            self.on_activate([path])

    def _activate_selection(self):
        chosen = [self._rows[i] for i in self.tree.selection() if i in self._rows]
        if len(chosen) == 1 and chosen[0][1]:
            self.go(chosen[0][0])
            return
        paths = [p for p, is_dir in chosen if not is_dir]
        if paths and self.on_activate is not None:
            self.on_activate(paths)


class FilePickerDialog(tk.Toplevel):
    """Modal: pick data files with the :class:`FileBrowser`. ``ask()``
    returns the list of paths (empty if cancelled)."""

    def __init__(self, parent, folder="", title="Select data file(s)"):
        tk.Toplevel.__init__(self, parent)
        self.title(title)
        self.transient(parent)
        self.geometry("820x520")
        self.result = []
        self.browser = FileBrowser(self, folder, on_activate=self._take)
        self.browser.pack(fill="both", expand=True, padx=8, pady=6)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(bar, foreground="#555", text=(
            "Double-click a file, or select several (Shift / Ctrl + click).")).pack(side="left")
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side="right", padx=3)
        ttk.Button(bar, text="Every data file in this folder",
                   command=self._take_all).pack(side="right", padx=3)
        ttk.Button(bar, text="Add selected", command=self._take_selected).pack(side="right", padx=3)
        self.bind("<Escape>", lambda e: self.destroy())

    def _take(self, paths):
        self.result = list(paths)
        self.destroy()

    def _take_selected(self):
        paths = self.browser.selected_paths()
        if paths:
            self._take(paths)

    def _take_all(self):
        paths = self.browser.shown_files(data_only=True)
        if paths:
            self._take(paths)

    @property
    def folder(self):
        return self.browser.folder

    def ask(self):
        self.grab_set()
        self.master.wait_window(self)
        return self.result
