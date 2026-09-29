"""
ui/tkbase.py
============
The building blocks every window is made of, in tkinter + matplotlib (the
lab server's Python 3.6 has neither PyQt5 nor pyqtgraph):

* :func:`get_cmap` -- the lab's colormaps (``tools.colormaps``) and
  matplotlib's own, by name.
* :class:`Form` -- a column of typed parameter fields. Every dialog that used
  to read a value off the mouse (a box dragged on an image, points clicked
  along a band) asks for it here instead, as numbers; where clicking still
  makes sense it only *fills in* these fields.
* :func:`run_job` / :func:`run_blocking` -- a long operation on a worker
  thread with a progress window and a working Cancel.
* :class:`ImagePanel` -- a 2-D image with its readout cursor, EDC and MDC,
  integration widths, level/gamma controls, overlays and export.
* small helpers: message boxes, choosing from a list, a text window.

No window-specific logic lives here.
"""
import os
import queue
import threading
import traceback

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure                                  # noqa: E402
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,     # noqa: E402
                                               NavigationToolbar2Tk)
from matplotlib import colors as mcolors                              # noqa: E402
from matplotlib import cm as mcm                                      # noqa: E402

from tools import colormaps                                           # noqa: E402
from ui import tktext                                                 # noqa: E402

tktext.install()

DEFAULT_COLORMAP = "gray"
#: The lab's tables first (as in the Qt version), then matplotlib's.
MPL_COLORMAPS = ("viridis", "inferno", "magma", "plasma", "cividis",
                 "Greys", "binary", "bone", "terrain", "coolwarm",
                 "RdBu_r", "bwr", "seismic", "Spectral_r", "turbo")


def _mpl_has(name):
    try:
        mcm.get_cmap(name)
        return True
    except (ValueError, KeyError):
        return False


COLORMAP_NAMES = list(colormaps.COLORMAP_NAMES) + [
    name for name in MPL_COLORMAPS if _mpl_has(name)]

#: One colour per axis of a cube, in every window: deflector red, slit
#: green, energy blue. A cursor line has the colour of the axis it holds
#: constant, and the EDC / MDC it feeds is drawn in the same colour.
AXIS_COLORS = {"defl": "#d62728", "slit": "#2ca02c", "energy": "#1f77b4"}
EDC_COLOR = "#1f5aa6"
MDC_COLOR = "#c23b22"

PAD = dict(padx=3, pady=2)


# --------------------------------------------------------------------------
# Colormaps
# --------------------------------------------------------------------------
_cmap_cache = {}


def get_cmap(name, flip=False):
    """A matplotlib colormap for ``name`` (lab table or matplotlib name)."""
    key = (name, bool(flip))
    if key in _cmap_cache:
        return _cmap_cache[key]
    try:
        lut = colormaps.get_lut(name, flip=flip)
        cmap = mcolors.ListedColormap(np.asarray(lut, dtype=float) / 255.0,
                                      name=name + ("_r" if flip else ""))
    except KeyError:
        try:
            cmap = mcm.get_cmap(name)
        except ValueError:
            cmap = mcm.get_cmap("gray")
        if flip:
            cmap = cmap.reversed()
    import copy
    cmap = copy.copy(cmap)
    try:
        cmap.set_bad((0.85, 0.85, 0.85, 1.0))
    except Exception:                                      # noqa: BLE001
        pass
    _cmap_cache[key] = cmap
    return cmap


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------
def info(parent, title, text):
    messagebox.showinfo(title, text, parent=parent)


def warning(parent, title, text):
    messagebox.showwarning(title, text, parent=parent)


def error(parent, title, text):
    messagebox.showerror(title, text, parent=parent)


def ask_yes_no(parent, title, text, default_no=True):
    return messagebox.askyesno(title, text, parent=parent,
                               default="no" if default_no else "yes")


def ask_string(parent, title, prompt, initial=""):
    value = simpledialog.askstring(title, prompt, initialvalue=initial,
                                   parent=parent)
    return None if value is None else value.strip()


def ask_float(parent, title, prompt, initial=0.0):
    return simpledialog.askfloat(title, prompt, initialvalue=initial,
                                 parent=parent)


def unique_name(name, taken):
    taken = set(taken or ())
    if name not in taken:
        return name
    n = 2
    while "%s (%d)" % (name, n) in taken:
        n += 1
    return "%s (%d)" % (name, n)


def safe_stem(text):
    """A filesystem-safe basename."""
    text = str(text).replace(" · ", "_")
    stem = os.path.splitext(text)[0] if text.lower().endswith(
        (".nxs", ".npz", ".txt", ".krx", ".h5")) else text
    return "".join(ch if (ch.isalnum() or ch in "._- ") else "_"
                   for ch in stem).strip() or "export"


class TextWindow(tk.Toplevel):
    """A read-only (copyable) text in a window of its own."""

    def __init__(self, parent, title, text, width=100, height=30):
        tk.Toplevel.__init__(self, parent)
        self.title(title)
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True)
        self.text = tk.Text(frame, width=width, height=height, wrap="none",
                            font=("Courier", 10))
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.text.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.set_text(text)
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="Copy all", command=self.copy).pack(side="left", **PAD)
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right", **PAD)

    def set_text(self, text):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.text.configure(state="disabled")

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.text.get("1.0", "end"))


def choose(parent, title, prompt, options, multiple=False, initial=None):
    """Pick one (or several) of ``options`` (strings) from a list. Returns
    the index (or list of indices), or None if cancelled."""
    result = {"value": None}
    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent)
    ttk.Label(win, text=prompt, wraplength=520, justify="left").pack(
        fill="x", padx=8, pady=6)
    frame = ttk.Frame(win)
    frame.pack(fill="both", expand=True, padx=8)
    box = tk.Listbox(frame, width=70, height=min(20, max(4, len(options))),
                     selectmode="extended" if multiple else "browse",
                     exportselection=False)
    sb = ttk.Scrollbar(frame, orient="vertical", command=box.yview)
    box.configure(yscrollcommand=sb.set)
    box.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    for option in options:
        box.insert("end", option)
    if options:
        box.selection_set(initial if initial is not None else 0)

    def ok(*_):
        chosen = list(box.curselection())
        if not chosen:
            return
        result["value"] = chosen if multiple else chosen[0]
        win.destroy()

    box.bind("<Double-Button-1>", ok)
    bar = ttk.Frame(win)
    bar.pack(fill="x", padx=8, pady=6)
    ttk.Button(bar, text="OK", command=ok).pack(side="right", padx=3)
    ttk.Button(bar, text="Cancel", command=win.destroy).pack(side="right", padx=3)
    win.grab_set()
    parent.wait_window(win)
    return result["value"]


def save_path(parent, title, initialfile, filetypes, initialdir=None):
    path = filedialog.asksaveasfilename(
        parent=parent, title=title, initialfile=initialfile,
        initialdir=initialdir or os.path.expanduser("~"), filetypes=filetypes,
        defaultextension=filetypes[0][1].split()[0].lstrip("*"))
    return path or None


# --------------------------------------------------------------------------
# Forms: typed parameters
# --------------------------------------------------------------------------
class FormError(ValueError):
    pass


class Field:
    """One parameter row.

    ``kind``: ``float`` / ``int`` / ``str`` / ``bool`` / ``choice`` /
    ``floats`` (comma- or space-separated numbers) / ``optfloat`` (a number
    or empty for "none") / ``label`` (a line of text, no value).
    """

    def __init__(self, key, label, kind="float", default=None, choices=None,
                 help="", width=14):
        self.key, self.label, self.kind = key, label, kind
        self.default, self.choices, self.help = default, choices, help
        self.width = width


def parse_floats(text):
    text = str(text).replace(",", " ").replace(";", " ")
    return [float(part) for part in text.split()]


class Form(ttk.Frame):
    """A grid of labelled entries built from :class:`Field` specs.

    ``values()`` returns ``{key: value}`` converted to each field's type,
    raising :class:`FormError` naming the field that is wrong. ``on_change``
    (optional) is called after any edit that commits (Enter, a choice, a
    tick), so a preview can follow the parameters.
    """

    def __init__(self, parent, fields, on_change=None, columns=1):
        ttk.Frame.__init__(self, parent)
        self.fields = [f for f in fields if f is not None]
        self.vars = {}
        self.widgets = {}
        self.on_change = on_change
        per_col = int(np.ceil(len(self.fields) / float(max(1, columns))))
        for index, field in enumerate(self.fields):
            col = (index // per_col) * 2 if per_col else 0
            row = index % per_col if per_col else index
            if field.kind == "label":
                lab = ttk.Label(self, text=field.label, wraplength=420,
                                justify="left", foreground="#444")
                lab.grid(row=row, column=col, columnspan=2, sticky="w", **PAD)
                continue
            if field.kind == "bool":
                var = tk.BooleanVar(value=bool(field.default))
                widget = ttk.Checkbutton(self, text=field.label, variable=var,
                                         command=self._changed)
                widget.grid(row=row, column=col, columnspan=2, sticky="w", **PAD)
            else:
                ttk.Label(self, text=field.label).grid(row=row, column=col,
                                                       sticky="w", **PAD)
                if field.kind == "choice":
                    var = tk.StringVar(value=self._show(field.default if field.default
                                                        is not None else field.choices[0]))
                    widget = ttk.Combobox(self, textvariable=var, state="readonly",
                                          values=[self._show(c) for c in field.choices],
                                          width=max(field.width, 10))
                    widget.bind("<<ComboboxSelected>>", self._changed)
                else:
                    var = tk.StringVar(value=self._fmt(field.default, field.kind))
                    widget = ttk.Entry(self, textvariable=var, width=field.width)
                    widget.bind("<Return>", self._changed)
                    widget.bind("<FocusOut>", self._changed)
                widget.grid(row=row, column=col + 1, sticky="w", **PAD)
            if field.help:
                Tooltip(widget, field.help)
            self.vars[field.key] = var
            self.widgets[field.key] = widget

    @staticmethod
    def _show(value):
        """How a choice is displayed (see :mod:`ui.tktext`)."""
        return tktext.ascii_safe(str(value)) if tktext._installed else str(value)

    def _real(self, field, shown):
        """The choice a displayed text stands for."""
        for choice in field.choices or ():
            if self._show(choice) == shown:
                return str(choice)
        return shown

    @staticmethod
    def _fmt(value, kind):
        if value is None:
            return ""
        if kind == "floats":
            return ", ".join("%g" % v for v in value)
        if kind in ("float", "optfloat") and isinstance(value, (float, np.floating)):
            return "%.6g" % value
        return str(value)

    def _changed(self, *_):
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:                                  # noqa: BLE001
                traceback.print_exc()

    def field(self, key):
        for f in self.fields:
            if f.key == key:
                return f
        raise KeyError(key)

    def set(self, key, value):
        f = self.field(key)
        if f.kind == "bool":
            self.vars[key].set(bool(value))
        elif f.kind == "choice":
            self.vars[key].set(self._show(value))
        else:
            self.vars[key].set(self._fmt(value, f.kind))

    def set_choices(self, key, choices, value=None):
        f = self.field(key)
        f.choices = list(choices)
        self.widgets[key].configure(values=[self._show(c) for c in choices])
        if value is not None:
            self.vars[key].set(self._show(value))
        elif self.raw(key) not in [str(c) for c in choices] and choices:
            self.vars[key].set(self._show(choices[0]))

    def enable(self, key, on=True):
        widget = self.widgets.get(key)
        if widget is not None:
            widget.configure(state=("readonly" if isinstance(widget, ttk.Combobox)
                                    else "normal") if on else "disabled")

    def raw(self, key):
        f = self.field(key)
        value = self.vars[key].get()
        return self._real(f, value) if f.kind == "choice" else value

    def value(self, key):
        return self.values(only=(key,))[key]

    def values(self, only=None):
        out = {}
        for f in self.fields:
            if f.kind == "label" or (only is not None and f.key not in only):
                continue
            raw = self.vars[f.key].get()
            try:
                if f.kind == "float":
                    out[f.key] = float(raw)
                elif f.kind == "int":
                    out[f.key] = int(float(raw))
                elif f.kind == "optfloat":
                    out[f.key] = float(raw) if str(raw).strip() else None
                elif f.kind == "floats":
                    out[f.key] = parse_floats(raw)
                elif f.kind == "bool":
                    out[f.key] = bool(raw)
                elif f.kind == "choice":
                    out[f.key] = self._real(f, raw)
                else:
                    out[f.key] = str(raw)
            except (TypeError, ValueError):
                raise FormError("'%s': %r is not a valid %s" % (
                    f.label, raw, {"floats": "list of numbers",
                                   "optfloat": "number (or empty)"}.get(f.kind, f.kind)))
        return out


class Tooltip:
    """A hover tooltip for any widget."""

    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, *_):
        if self.tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry("+%d+%d" % (x, y))
        tk.Label(self.tip, text=self.text, justify="left", wraplength=420,
                 background="#ffffe0", relief="solid", borderwidth=1).pack()

    def hide(self, *_):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class ParamDialog(tk.Toplevel):
    """A modal dialog around a :class:`Form`. ``ask()`` returns the values
    or None when cancelled."""

    def __init__(self, parent, title, fields, text="", ok_text="OK", columns=1):
        tk.Toplevel.__init__(self, parent)
        self.title(title)
        self.transient(parent)
        self.result = None
        if text:
            ttk.Label(self, text=text, wraplength=520, justify="left").pack(
                fill="x", padx=8, pady=6)
        self.form = Form(self, fields, columns=columns)
        self.form.pack(fill="both", expand=True, padx=8, pady=4)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text=ok_text, command=self._ok).pack(side="right", padx=3)
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side="right", padx=3)
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())

    def _ok(self):
        try:
            self.result = self.form.values()
        except FormError as exc:
            warning(self, self.title(), str(exc))
            return
        self.destroy()

    def ask(self):
        self.grab_set()
        self.master.wait_window(self)
        return self.result


def ask_params(parent, title, fields, text="", ok_text="OK", columns=1):
    return ParamDialog(parent, title, fields, text, ok_text, columns).ask()


# --------------------------------------------------------------------------
# Long operations: a worker thread, a progress window, Cancel
# --------------------------------------------------------------------------
class JobCancelled(Exception):
    pass


_BUSY = {"flag": False}


def busy():
    return _BUSY["flag"]


class _Progress(tk.Toplevel):
    def __init__(self, parent, title):
        tk.Toplevel.__init__(self, parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.label = ttk.Label(self, text="Working...", width=60)
        self.label.pack(padx=10, pady=(10, 4))
        self.bar = ttk.Progressbar(self, length=380, mode="determinate",
                                   maximum=1000)
        self.bar.pack(padx=10, pady=4)
        self.cancelled = False
        ttk.Button(self, text="Cancel", command=self.cancel).pack(pady=(4, 10))
        self.protocol("WM_DELETE_WINDOW", self.cancel)

    def cancel(self):
        self.cancelled = True
        self.label.configure(text="Cancelling...")


def run_job(parent, title, work, on_done=None, on_error=None, on_cancel=None,
            delay_ms=400):
    """Run ``work(report)`` on a worker thread.

    ``report(fraction, text)`` updates the progress window, and raises
    :class:`JobCancelled` once Cancel was pressed -- so a long loop that
    reports is also a loop that can be stopped. The window only appears if
    the job takes longer than ``delay_ms``. ``on_done(result)``,
    ``on_error(exc)`` and ``on_cancel()`` run on the GUI thread. One job at a
    time: returns False (and does nothing) while another is running.
    """
    if _BUSY["flag"]:
        return False
    _BUSY["flag"] = True
    messages = queue.Queue()
    state = {"cancel": False, "window": None}

    def report(fraction=None, text=None):
        if state["cancel"]:
            raise JobCancelled()
        messages.put(("progress", fraction, text))

    def worker():
        try:
            result = work(report)
            messages.put(("done", result, None))
        except JobCancelled:
            messages.put(("cancel", None, None))
        except Exception as exc:                               # noqa: BLE001
            traceback.print_exc()
            messages.put(("error", exc, None))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def show_window():
        if state["window"] is None and thread.is_alive():
            state["window"] = _Progress(parent, title)

    parent.after(delay_ms, show_window)

    def poll():
        window = state["window"]
        if window is not None and window.cancelled:
            state["cancel"] = True
        finished = None
        try:
            while True:
                message = messages.get_nowait()
                if message[0] == "progress":
                    if window is not None:
                        if message[1] is not None:
                            window.bar["value"] = int(1000 * max(0.0, min(1.0, message[1])))
                        if message[2]:
                            window.label.configure(text=str(message[2])[:90])
                else:
                    finished = message
        except queue.Empty:
            pass
        if finished is None:
            parent.after(60, poll)
            return
        _BUSY["flag"] = False
        if state["window"] is not None:
            state["window"].destroy()
            state["window"] = None
        kind, value, _ = finished
        if kind == "done" and on_done is not None:
            on_done(value)
        elif kind == "error":
            if on_error is not None:
                on_error(value)
            else:
                error(parent, title, "%s failed:\n%s" % (title, value))
        elif kind == "cancel" and on_cancel is not None:
            on_cancel()

    parent.after(30, poll)
    return True


def run_blocking(parent, title, work, delay_ms=400):
    """:func:`run_job`, waited for: the GUI keeps painting while the caller
    waits. Returns the result; raises the job's exception, or
    :class:`JobCancelled`."""
    outcome = {}
    done = tk.BooleanVar(master=parent, value=False)

    def finish(kind, value=None):
        outcome["kind"], outcome["value"] = kind, value
        done.set(True)

    if not run_job(parent, title, work,
                   on_done=lambda v: finish("done", v),
                   on_error=lambda e: finish("error", e),
                   on_cancel=lambda: finish("cancel"), delay_ms=delay_ms):
        # another job is running: do it here, without a progress window
        return work(lambda *a, **k: None)
    parent.wait_variable(done)
    if outcome["kind"] == "error":
        raise outcome["value"]
    if outcome["kind"] == "cancel":
        raise JobCancelled()
    return outcome["value"]


def with_wait_cursor(widget, function, *args, **kwargs):
    """Run ``function`` with a busy cursor on ``widget``'s toplevel."""
    top = widget.winfo_toplevel()
    try:
        top.configure(cursor="watch")
        top.update_idletasks()
    except tk.TclError:
        pass
    try:
        return function(*args, **kwargs)
    finally:
        try:
            top.configure(cursor="")
        except tk.TclError:
            pass


# --------------------------------------------------------------------------
# A matplotlib figure in a frame
# --------------------------------------------------------------------------
class PlotFrame(ttk.Frame):
    """A matplotlib Figure, its Tk canvas and (optionally) the standard
    navigation toolbar (zoom / pan / home / save)."""

    def __init__(self, parent, figsize=(6, 4.5), toolbar=True, dpi=90):
        ttk.Frame.__init__(self, parent)
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = None
        if toolbar:
            bar = ttk.Frame(self)
            bar.pack(side="top", fill="x")
            self.toolbar = NavigationToolbar2Tk(self.canvas, bar)
            self.toolbar.update()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def draw(self):
        self.canvas.draw_idle()

    def navigating(self):
        """True while the toolbar's zoom or pan mode is on, when a click
        belongs to the toolbar and not to the cursor."""
        return bool(self.toolbar is not None and getattr(self.toolbar, "mode", ""))


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def sorted_axis(axis, values, dim):
    """Make ``axis`` increasing, flipping ``values`` along ``dim`` to match."""
    axis = np.asarray(axis, dtype=float)
    if axis.size > 1 and axis[0] > axis[-1]:
        return axis[::-1], np.flip(values, axis=dim)
    return axis, values


def uniform(axis, rtol=1e-3):
    axis = np.asarray(axis, dtype=float)
    if axis.size < 3:
        return True
    d = np.diff(axis)
    return bool(np.all(np.abs(d - d.mean()) <= rtol * max(abs(d.mean()), 1e-300)))


def edges_of(axis):
    axis = np.asarray(axis, dtype=float)
    if axis.size == 1:
        return np.array([axis[0] - 0.5, axis[0] + 0.5])
    mid = 0.5 * (axis[1:] + axis[:-1])
    return np.concatenate([[axis[0] - (mid[0] - axis[0])], mid,
                           [axis[-1] + (axis[-1] - mid[-1])]])


def nearest_index(axis, value):
    axis = np.asarray(axis, dtype=float)
    return int(np.nanargmin(np.abs(axis - float(value))))


def index_window(axis, centre, half_width):
    """Indices ``lo, hi`` (inclusive) of the samples within ``centre ±
    half_width``; always at least the nearest sample."""
    axis = np.asarray(axis, dtype=float)
    i = nearest_index(axis, centre)
    if half_width <= 0:
        return i, i
    inside = np.where(np.abs(axis - centre) <= half_width + 1e-12)[0]
    if inside.size == 0:
        return i, i
    return int(inside.min()), int(inside.max())


def auto_levels(values, lo_pct=0.5, hi_pct=99.5):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size > 400000:
        finite = finite[::int(finite.size // 400000) + 1]
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if not hi > lo:
        lo, hi = float(finite.min()), float(finite.max())
        if not hi > lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def is_energy_label(label):
    return "eV" in (label or "") or "energy" in (label or "").lower()


# --------------------------------------------------------------------------
# The image panel
# --------------------------------------------------------------------------
class ImagePanel(ttk.Frame):
    """One image with everything around it.

    ``values`` are given in ``(x, y)`` order -- the first index runs across
    the image, the second up it -- together with the two axis vectors.

    * **Cursor** -- click on the image (with the toolbar's zoom / pan off),
      or type the position and press Enter / Go. It reads the value, and when
      ``curves`` is on draws the EDC (along y, at the cursor's x) and the MDC
      (along x, at its y), each summed over its own ``±`` window.
    * **Levels** -- min / max (empty = automatic percentiles), gamma, the
      interpolation of the display and a display-only smoothing.
    * **Box** -- a rectangle, typed as ``x0 x1 y0 y1`` (or taken from the
      current zoom); used by the tools that need a region.
    * **Overlays** -- lines and points drawn by tools (tagged, so a tool can
      clear its own).
    """

    def __init__(self, parent, title="", curves=True, cursor_color="#ffcc00",
                 edc_color=EDC_COLOR, mdc_color=MDC_COLOR, figsize=(6.4, 5.2),
                 on_cursor=None, show_box_row=True, cmap=DEFAULT_COLORMAP,
                 flip=False):
        ttk.Frame.__init__(self, parent)
        self.title = title
        self.curves = curves
        self.cursor_color = cursor_color
        self.edc_color, self.mdc_color = edc_color, mdc_color
        self.on_cursor = on_cursor
        self.on_click_hooks = []
        self.values = None
        self.x = self.y = None
        self.x_label = self.y_label = ""
        self.cursor = None                    # (x, y) or None
        self.box = None                       # (x0, x1, y0, y1) or None
        self.cmap_name, self.flip = cmap, flip
        self._overlays = {}
        self._image = None
        self._keep_view = False

        self.plot = PlotFrame(self, figsize=figsize)
        self.plot.pack(side="top", fill="both", expand=True)
        fig = self.plot.figure
        if curves:
            grid = fig.add_gridspec(2, 2, width_ratios=(4, 1.3),
                                    height_ratios=(1.3, 4), wspace=0.05,
                                    hspace=0.05, left=0.12, right=0.97,
                                    bottom=0.1, top=0.93)
            self.ax = fig.add_subplot(grid[1, 0])
            self.ax_mdc = fig.add_subplot(grid[0, 0], sharex=self.ax)
            self.ax_edc = fig.add_subplot(grid[1, 1], sharey=self.ax)
            self.ax_mdc.tick_params(labelbottom=False, labelsize=7)
            self.ax_edc.tick_params(labelleft=False, labelsize=7)
            self.ax_mdc.set_ylabel("MDC", fontsize=8)
            self.ax_edc.set_xlabel("EDC", fontsize=8)
        else:
            self.ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.1, top=0.93)
            self.ax_mdc = self.ax_edc = None
        if title:
            fig.suptitle(title, fontsize=10)
        self.plot.canvas.mpl_connect("button_press_event", self._on_click)

        self._build_controls(show_box_row)

    # -- controls ------------------------------------------------------------
    def _build_controls(self, show_box_row):
        row1 = ttk.Frame(self)
        row1.pack(side="top", fill="x")
        ttk.Label(row1, text="Levels min").pack(side="left")
        self.vmin_var = tk.StringVar()
        self.vmax_var = tk.StringVar()
        e1 = ttk.Entry(row1, textvariable=self.vmin_var, width=9)
        e1.pack(side="left")
        ttk.Label(row1, text="max").pack(side="left")
        e2 = ttk.Entry(row1, textvariable=self.vmax_var, width=9)
        e2.pack(side="left")
        for e in (e1, e2):
            e.bind("<Return>", lambda ev: self.redraw())
        Tooltip(e1, "Colour scale limits. Empty = automatic (0.5 / 99.5 percentiles). Enter to apply.")
        ttk.Button(row1, text="Auto", width=5,
                   command=self._auto_levels).pack(side="left", padx=2)
        ttk.Label(row1, text="gamma").pack(side="left")
        self.gamma_var = tk.StringVar(value="1")
        e3 = ttk.Entry(row1, textvariable=self.gamma_var, width=5)
        e3.pack(side="left")
        e3.bind("<Return>", lambda ev: self.redraw())
        ttk.Label(row1, text="interp").pack(side="left", padx=(6, 0))
        self.interp_var = tk.StringVar(value="nearest")
        c = ttk.Combobox(row1, textvariable=self.interp_var, width=8,
                         state="readonly",
                         values=["nearest", "bilinear", "bicubic", "gaussian"])
        c.pack(side="left")
        c.bind("<<ComboboxSelected>>", lambda ev: self.redraw())
        ttk.Label(row1, text="smooth σ(px)").pack(side="left", padx=(6, 0))
        self.smooth_var = tk.StringVar(value="0")
        e4 = ttk.Entry(row1, textvariable=self.smooth_var, width=4)
        e4.pack(side="left")
        e4.bind("<Return>", lambda ev: self.redraw())
        Tooltip(e4, "Display-only Gaussian smoothing (pixels). Data and exports are not smoothed.")
        self.grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="grid", variable=self.grid_var,
                        command=self.redraw).pack(side="left", padx=4)
        self.equal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="1:1", variable=self.equal_var,
                        command=self.redraw).pack(side="left")
        self.readout = ttk.Label(row1, text="", foreground="#333")
        self.readout.pack(side="left", padx=6)

        row2 = ttk.Frame(self)
        row2.pack(side="top", fill="x")
        self.cursor_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Cursor", variable=self.cursor_on,
                        command=self.redraw).pack(side="left")
        ttk.Label(row2, text="x").pack(side="left")
        self.cx_var = tk.StringVar()
        self.cy_var = tk.StringVar()
        ex = ttk.Entry(row2, textvariable=self.cx_var, width=9)
        ex.pack(side="left")
        ttk.Label(row2, text="y").pack(side="left")
        ey = ttk.Entry(row2, textvariable=self.cy_var, width=9)
        ey.pack(side="left")
        for e in (ex, ey):
            e.bind("<Return>", lambda ev: self._typed_cursor())
        ttk.Button(row2, text="Go", width=3,
                   command=self._typed_cursor).pack(side="left", padx=2)
        Tooltip(ex, "Click the image (zoom/pan off) or type the position and press Enter.")
        if self.curves:
            ttk.Label(row2, text="EDC ±").pack(side="left", padx=(8, 0))
            self.edc_w = tk.StringVar(value="0")
            ew = ttk.Entry(row2, textvariable=self.edc_w, width=6)
            ew.pack(side="left")
            ttk.Label(row2, text="MDC ±").pack(side="left")
            self.mdc_w = tk.StringVar(value="0")
            mw = ttk.Entry(row2, textvariable=self.mdc_w, width=6)
            mw.pack(side="left")
            for e in (ew, mw):
                e.bind("<Return>", lambda ev: self.redraw())
            Tooltip(ew, "Half-widths (in axis units) the EDC and MDC are summed over. Enter to apply.")
            ttk.Button(row2, text="EDC → list",
                       command=lambda: self._curve_to_list("edc")).pack(side="left", padx=2)
            ttk.Button(row2, text="MDC → list",
                       command=lambda: self._curve_to_list("mdc")).pack(side="left", padx=2)
        self.curve_to_list_hook = None

        self.box_var = tk.StringVar()
        if show_box_row:
            row3 = ttk.Frame(self)
            row3.pack(side="top", fill="x")
            ttk.Label(row3, text="Box x0 x1 y0 y1").pack(side="left")
            eb = ttk.Entry(row3, textvariable=self.box_var, width=34)
            eb.pack(side="left")
            eb.bind("<Return>", lambda ev: self._typed_box())
            ttk.Button(row3, text="Set", width=4,
                       command=self._typed_box).pack(side="left", padx=2)
            ttk.Button(row3, text="From zoom",
                       command=self.box_from_view).pack(side="left", padx=2)
            ttk.Button(row3, text="Clear", width=5,
                       command=lambda: self.set_box(None)).pack(side="left", padx=2)
            Tooltip(eb, "A rectangular region used by some tools (Fermi fit, "
                        "integration, kz edge window...). Type four numbers, or zoom "
                        "the image to the region and press 'From zoom'.")
            self.status = ttk.Label(row3, text="", foreground="#a33")
            self.status.pack(side="left", padx=6)
        else:
            self.status = ttk.Label(row2, text="", foreground="#a33")
            self.status.pack(side="left", padx=6)

    # -- data ------------------------------------------------------------------
    def set_data(self, values, x, y, x_label="", y_label="", keep_view=True,
                 keep_cursor=True):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise ValueError("an image needs a 2-D array, got %s" % (values.shape,))
        x, values = sorted_axis(x, values, 0)
        y, values = sorted_axis(y, values, 1)
        first = self.values is None
        same_extent = (not first and self.x is not None and self.x.size == x.size
                       and self.y.size == y.size
                       and np.allclose([self.x[0], self.x[-1], self.y[0], self.y[-1]],
                                       [x[0], x[-1], y[0], y[-1]]))
        self.values, self.x, self.y = values, x, y
        self.x_label, self.y_label = x_label, y_label
        if self.cursor is None or not keep_cursor or not self._inside(*self.cursor):
            self.cursor = (float(x[len(x) // 2]), float(y[len(y) // 2]))
        self._keep_view = keep_view and same_extent
        self.redraw()

    def _inside(self, cx, cy):
        return (self.x is not None and self.x[0] - 1e-12 <= cx <= self.x[-1] + 1e-12
                and self.y[0] - 1e-12 <= cy <= self.y[-1] + 1e-12)

    def set_colormap(self, name, flip=False):
        self.cmap_name, self.flip = name, flip
        if self._image is not None:
            self._image.set_cmap(get_cmap(name, flip))
            self.plot.draw()

    def levels(self, values):
        lo_text, hi_text = self.vmin_var.get().strip(), self.vmax_var.get().strip()
        auto_lo, auto_hi = auto_levels(values)
        try:
            lo = float(lo_text) if lo_text else auto_lo
            hi = float(hi_text) if hi_text else auto_hi
        except ValueError:
            lo, hi = auto_lo, auto_hi
        if not hi > lo:
            hi = lo + 1e-12 + abs(lo) * 1e-6
        return lo, hi

    def gamma(self):
        try:
            g = float(self.gamma_var.get())
            return g if g > 0 else 1.0
        except ValueError:
            return 1.0

    def _auto_levels(self):
        self.vmin_var.set("")
        self.vmax_var.set("")
        self.redraw()

    def display_array(self):
        values = self.values
        try:
            sigma = float(self.smooth_var.get())
        except ValueError:
            sigma = 0.0
        if sigma > 0:
            from ui.data import smooth2d
            values = smooth2d(values, sigma)
        return values

    def norm(self, values):
        lo, hi = self.levels(values)
        g = self.gamma()
        if abs(g - 1.0) < 1e-9:
            return mcolors.Normalize(vmin=lo, vmax=hi, clip=True)
        return mcolors.PowerNorm(gamma=g, vmin=lo, vmax=hi, clip=True)

    # -- drawing ---------------------------------------------------------------
    def redraw(self):
        if self.values is None:
            return
        ax = self.ax
        keep = self._keep_view and self._image is not None
        xlim, ylim = (ax.get_xlim(), ax.get_ylim()) if keep else (None, None)
        ax.clear()
        values = self.display_array()
        cmap = get_cmap(self.cmap_name, self.flip)
        norm = self.norm(values)
        if uniform(self.x) and uniform(self.y):
            ex, ey = edges_of(self.x), edges_of(self.y)
            self._image = ax.imshow(values.T, origin="lower", aspect="auto",
                                    extent=(ex[0], ex[-1], ey[0], ey[-1]),
                                    cmap=cmap, norm=norm,
                                    interpolation=self.interp_var.get())
        else:
            self._image = ax.pcolormesh(edges_of(self.x), edges_of(self.y),
                                        values.T, cmap=cmap, norm=norm)
        ax.set_xlabel(self.x_label, fontsize=9)
        ax.set_ylabel(self.y_label, fontsize=9)
        ax.tick_params(labelsize=8)
        if self.grid_var.get():
            ax.grid(True, color="w", alpha=0.4, lw=0.5)
        self._draw_cursor_and_curves()
        self._draw_box()
        for tag, items in self._overlays.items():
            for kind, args, kwargs in items:
                getattr(ax, kind)(*args, **kwargs)
        if xlim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            ex, ey = edges_of(self.x), edges_of(self.y)
            ax.set_xlim(ex[0], ex[-1])
            ax.set_ylim(ey[0], ey[-1])
        if self.equal_var.get():
            self._equal_limits()
        self._keep_view = True
        self.plot.draw()

    def _draw_cursor_and_curves(self):
        if self.curves:
            self.ax_edc.clear()
            self.ax_mdc.clear()
            self.ax_mdc.tick_params(labelbottom=False, labelsize=7)
            self.ax_edc.tick_params(labelleft=False, labelsize=7)
        if self.cursor is None:
            return
        cx, cy = self.cursor
        self.cx_var.set("%.5g" % cx)
        self.cy_var.set("%.5g" % cy)
        i = nearest_index(self.x, cx)
        j = nearest_index(self.y, cy)
        value = self.values[i, j]
        self.readout.configure(text="(%.4g, %.4g) = %.4g" % (
            self.x[i], self.y[j], value))
        if not self.cursor_on.get():
            return
        col = self.cursor_color
        self.ax.axvline(cx, color=col, lw=0.8)
        self.ax.axhline(cy, color=col, lw=0.8)
        if not self.curves:
            return
        (edc_x, edc_y, n_e), (mdc_x, mdc_y, n_m) = self.edc(), self.mdc()
        self.ax_edc.plot(edc_y, edc_x, color=self.edc_color, lw=1)
        self.ax_mdc.plot(mdc_x, mdc_y, color=self.mdc_color, lw=1)
        wx, wy = self.half_widths()
        if wx > 0:
            self.ax.axvspan(cx - wx, cx + wx, color=self.edc_color, alpha=0.15)
        if wy > 0:
            self.ax.axhspan(cy - wy, cy + wy, color=self.mdc_color, alpha=0.15)
        self.ax_mdc.set_ylabel("MDC", fontsize=8)
        self.ax_edc.set_xlabel("EDC", fontsize=8)

    def _equal_limits(self):
        """One unit the same length on both axes, by widening whichever
        range is short for the panel's shape (the EDC / MDC panels share
        the axes, so matplotlib's own aspect locking cannot be used)."""
        ax = self.ax
        box = ax.get_position()
        width_in, height_in = self.plot.figure.get_size_inches()
        w, h = box.width * width_in, box.height * height_in
        if w <= 0 or h <= 0:
            return
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dx <= 0 or dy <= 0:
            return
        if dx / dy < w / h:
            half = 0.5 * dy * w / h
            c = 0.5 * (x0 + x1)
            ax.set_xlim(c - half, c + half)
        else:
            half = 0.5 * dx * h / w
            c = 0.5 * (y0 + y1)
            ax.set_ylim(c - half, c + half)

    def half_widths(self):
        def num(var):
            try:
                return max(0.0, float(var.get()))
            except (ValueError, AttributeError):
                return 0.0
        if not self.curves:
            return 0.0, 0.0
        return num(self.edc_w), num(self.mdc_w)

    def edc(self):
        """The curve along y at the cursor's x, summed over ``x ± EDC``:
        ``(y_axis, values, n_summed)``."""
        wx, _ = self.half_widths()
        lo, hi = index_window(self.x, self.cursor[0], wx)
        return self.y, np.nansum(self.values[lo:hi + 1, :], axis=0), hi - lo + 1

    def mdc(self):
        """The curve along x at the cursor's y, summed over ``y ± MDC``."""
        _, wy = self.half_widths()
        lo, hi = index_window(self.y, self.cursor[1], wy)
        return self.x, np.nansum(self.values[:, lo:hi + 1], axis=1), hi - lo + 1

    def _curve_to_list(self, which):
        if self.values is None or self.cursor is None or self.curve_to_list_hook is None:
            return
        wx, wy = self.half_widths()
        if which == "edc":
            axis, values, n = self.edc()
            payload = {"x": axis, "y": values, "x_label": self.y_label,
                       "position": self.cursor[0], "position_label": self.x_label,
                       "half_width": wx, "n_summed": n}
        else:
            axis, values, n = self.mdc()
            payload = {"x": axis, "y": values, "x_label": self.x_label,
                       "position": self.cursor[1], "position_label": self.y_label,
                       "half_width": wy, "n_summed": n}
        self.curve_to_list_hook(payload)

    # -- cursor ------------------------------------------------------------------
    def set_cursor(self, x, y, notify=True, redraw=True):
        if self.values is None:
            return False
        if not self._inside(x, y):
            self.status.configure(text="(%.4g, %.4g) is outside the data" % (x, y))
            return False
        self.status.configure(text="")
        self.cursor = (float(x), float(y))
        if redraw:
            self.redraw()
        if notify and self.on_cursor is not None:
            self.on_cursor(self, self.cursor[0], self.cursor[1])
        return True

    def _typed_cursor(self):
        try:
            x, y = float(self.cx_var.get()), float(self.cy_var.get())
        except ValueError:
            self.status.configure(text="cursor position: two numbers please")
            return
        self.set_cursor(x, y)

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self.plot.navigating():
            return
        for hook in list(self.on_click_hooks):
            if hook(event.xdata, event.ydata, event.button):
                return
        if event.button == 1:
            self.set_cursor(event.xdata, event.ydata)

    # -- box -----------------------------------------------------------------------
    def set_box(self, box, redraw=True):
        if box is not None:
            x0, x1, y0, y1 = [float(v) for v in box]
            box = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            self.box_var.set("%.5g %.5g %.5g %.5g" % box)
        else:
            self.box_var.set("")
        self.box = box
        if redraw:
            self.redraw()

    def _typed_box(self):
        text = self.box_var.get().strip()
        if not text:
            self.set_box(None)
            return
        try:
            values = parse_floats(text)
            if len(values) != 4:
                raise ValueError
        except ValueError:
            self.status.configure(text="box: four numbers x0 x1 y0 y1")
            return
        self.set_box(values)

    def box_from_view(self):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.set_box((x0, x1, y0, y1))

    def _draw_box(self):
        if self.box is None:
            return
        from matplotlib.patches import Rectangle
        x0, x1, y0, y1 = self.box
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    ec="#00e5ff", lw=1.2, ls="--"))

    def box_indices(self):
        """``(i0, i1, j0, j1)`` inclusive index ranges of the box, or of the
        whole image when there is none."""
        if self.box is None:
            return 0, len(self.x) - 1, 0, len(self.y) - 1
        x0, x1, y0, y1 = self.box
        ii = np.where((self.x >= x0) & (self.x <= x1))[0]
        jj = np.where((self.y >= y0) & (self.y <= y1))[0]
        if ii.size == 0 or jj.size == 0:
            raise ValueError("the box holds no data points")
        return int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max())

    # -- overlays ------------------------------------------------------------------
    def add_overlay(self, tag, kind, *args, **kwargs):
        """Draw ``ax.<kind>(*args, **kwargs)`` now and on every redraw."""
        self._overlays.setdefault(tag, []).append((kind, args, kwargs))

    def clear_overlays(self, tag=None):
        if tag is None:
            self._overlays.clear()
        else:
            self._overlays.pop(tag, None)

    # -- export ----------------------------------------------------------------------
    def export_image(self, parent, stem):
        path = save_path(parent, "Export image", stem + ".png",
                         [("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                          ("EPS", "*.eps"), ("TIFF", "*.tif")])
        if not path:
            return None
        self.plot.figure.savefig(path, dpi=300)
        return path

    def export_data(self, parent, stem):
        """The image's numbers as a text matrix: first row the x axis, first
        column the y axis (the layout Igor / Origin / MATLAB read)."""
        if self.values is None:
            return None
        path = save_path(parent, "Export data", stem + ".txt",
                         [("Text matrix", "*.txt"), ("CSV", "*.csv"),
                          ("numpy", "*.npz")])
        if not path:
            return None
        if path.lower().endswith(".npz"):
            np.savez_compressed(path, values=self.values, x=self.x, y=self.y,
                                x_label=self.x_label, y_label=self.y_label)
            return path
        delimiter = "," if path.lower().endswith(".csv") else "\t"
        table = np.full((self.y.size + 1, self.x.size + 1), np.nan)
        table[0, 1:] = self.x
        table[1:, 0] = self.y
        table[1:, 1:] = self.values.T
        header = "first row: %s; first column: %s" % (self.x_label, self.y_label)
        np.savetxt(path, table, delimiter=delimiter, fmt="%.8g", header=header, encoding="utf-8")
        return path


# --------------------------------------------------------------------------
# A slider through one axis of a cube
# --------------------------------------------------------------------------
class SliceControl(ttk.Frame):
    """Which slice of an axis is shown: a slider, the position (**Pos**, in
    axis units; the nearest measured point is taken) and the 1-based index
    (**Ind**, as in the MATLAB tool), ``<`` / ``>`` steps, and an
    integration half-width ``±`` with how the window is combined
    (``mean`` keeps the brightness, ``sum`` gives total counts)."""

    def __init__(self, parent, title, on_change=None, color=None):
        ttk.Frame.__init__(self, parent)
        self.axis = np.zeros(1)
        self.on_change = on_change
        self._index = 0
        top = ttk.Frame(self)
        top.pack(fill="x")
        lab = tk.Label(top, text=title, fg=color or "black")
        lab.pack(side="left", padx=3)
        self.readout = ttk.Label(top, text="")
        self.readout.pack(side="left", padx=6)
        row = ttk.Frame(self)
        row.pack(fill="x")
        # Packed from the right, so the slider gets only what is left and
        # the typed controls are never pushed off a narrow window.
        self.reduce_var = tk.StringVar(value="mean")
        c = ttk.Combobox(row, textvariable=self.reduce_var, width=5,
                         state="readonly", values=["mean", "sum"])
        c.pack(side="right", padx=2)
        c.bind("<<ComboboxSelected>>", lambda ev: self._notify())
        self.width_var = tk.StringVar(value="0")
        e3 = ttk.Entry(row, textvariable=self.width_var, width=7)
        e3.pack(side="right")
        e3.bind("<Return>", lambda ev: self._notify())
        Tooltip(e3, "Integration half-width, in axis units. Enter to apply.")
        ttk.Label(row, text="\u00b1").pack(side="right", padx=(6, 0))
        ttk.Button(row, text=">", width=2, command=lambda: self.step(1)).pack(side="right")
        self.ind_var = tk.StringVar()
        e2 = ttk.Entry(row, textvariable=self.ind_var, width=5)
        e2.pack(side="right")
        e2.bind("<Return>", lambda ev: self._typed_index())
        ttk.Button(row, text="<", width=2, command=lambda: self.step(-1)).pack(side="right")
        self.pos_var = tk.StringVar()
        e = ttk.Entry(row, textvariable=self.pos_var, width=9)
        e.pack(side="right")
        e.bind("<Return>", lambda ev: self._typed_pos())
        ttk.Label(row, text="Pos").pack(side="right")
        self.scale_var = tk.DoubleVar(value=0)
        self.scale = ttk.Scale(row, from_=0, to=1, orient="horizontal",
                               variable=self.scale_var, command=self._slid, length=80)
        self.scale.pack(side="left", fill="x", expand=True, padx=3)
        self.scale.bind("<ButtonRelease-1>", lambda e: self._commit())

    def configure_axis(self, axis, index=None):
        self.axis = np.asarray(axis, dtype=float)
        self.scale.configure(to=max(self.axis.size - 1, 0))
        self.set_index(self.axis.size // 2 if index is None else index, notify=False)

    @property
    def index(self):
        return self._index

    def set_index(self, index, notify=True):
        index = int(np.clip(int(index), 0, max(self.axis.size - 1, 0)))
        self._index = index
        self.scale_var.set(index)
        self.ind_var.set(str(index + 1))
        if self.axis.size:
            self.pos_var.set("%.5g" % self.axis[index])
        lo, hi = self.window()
        extra = (" (%d points)" % (hi - lo + 1)) if hi > lo else ""
        self.readout.configure(text="%.5g%s" % (self.axis[index], extra))
        if notify:
            self._notify()

    def set_position(self, value, notify=True):
        self.set_index(nearest_index(self.axis, value), notify=notify)

    @property
    def position(self):
        return float(self.axis[self._index])

    def step(self, delta):
        self.set_index(self._index + delta)

    def _slid(self, *_):
        index = int(round(self.scale_var.get()))
        if index != self._index:
            self.set_index(index, notify=False)
            # a large cube re-slices slowly: follow the drag only for small
            # ones, and always on release
            self._pending = True
            self.after_idle(self._commit)

    def _commit(self):
        if getattr(self, "_pending", True):
            self._pending = False
            self._notify()

    def _typed_pos(self):
        try:
            self.set_position(float(self.pos_var.get()))
        except ValueError:
            pass

    def _typed_index(self):
        try:
            self.set_index(int(float(self.ind_var.get())) - 1)
        except ValueError:
            pass

    @property
    def half_width(self):
        try:
            return max(0.0, float(self.width_var.get()))
        except ValueError:
            return 0.0

    def window(self):
        if not self.axis.size:
            return 0, 0
        return index_window(self.axis, self.axis[self._index], self.half_width)

    def indices(self):
        lo, hi = self.window()
        return np.arange(lo, hi + 1)

    def combine(self, values, axis):
        values = np.asarray(values, dtype=float)
        if self.reduce_var.get() == "sum":
            return np.nansum(values, axis=axis)
        with np.errstate(invalid="ignore"):
            return np.nanmean(values, axis=axis) if values.shape[axis] > 1 \
                else np.take(values, 0, axis=axis)

    def _notify(self):
        lo, hi = self.window()
        extra = (" (%d points)" % (hi - lo + 1)) if hi > lo else ""
        if self.axis.size:
            self.readout.configure(text="%.5g%s" % (self.axis[self._index], extra))
        if self.on_change is not None:
            self.on_change()
