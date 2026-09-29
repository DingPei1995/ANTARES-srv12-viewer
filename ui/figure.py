"""
ui/figure.py
============
The figure composer: panels laid out for a paper, drawn by matplotlib.

A :class:`Panel` is either an image (a 2-D array on its two axes, with its
colormap, levels and gamma) or a set of curves (with optional ±σ). A
:class:`FigureWindow` lays several out on a grid and exports the page as
PNG / PDF / SVG / EPS / TIFF at a chosen size and resolution:

* rows x columns, figure width and height in mm, font size;
* one shared colour scale for every image (or each its own), a colour bar;
* equal aspect (for kx-ky contours), axis labels only on the outer edges,
  panel letters (a), (b), ...;
* each panel's title, axis ranges, levels, gamma, colormap and axis labels
  can be edited in the panel list;
* more panels can be added from the main list.

:func:`open_slice_series` makes a page of slices through a cube (a row of
constant-energy contours, or of cuts) -- the MATLAB tool's ``massplott_3D``.
"""
import string

import numpy as np
import tkinter as tk
from tkinter import ttk

from ui import tkbase as T
from ui.tkbase import Field
from ui.data import unit_of


class Panel:
    """One panel of a figure."""

    def __init__(self, kind, x_label="", y_label="", title=""):
        self.kind = kind                    # "image" or "curves"
        self.x_label, self.y_label, self.title = x_label, y_label, title
        self.array = self.x = self.y = None
        self.curves = []
        self.cmap, self.flip = T.DEFAULT_COLORMAP, False
        self.levels = None                  # (lo, hi) or None = auto
        self.gamma = 1.0
        self.xlim = self.ylim = None
        self.legend = True

    @classmethod
    def image(cls, array, x, y, x_label="", y_label="", title="", cmap=T.DEFAULT_COLORMAP,
              flip=False, levels=None, gamma=None):
        p = cls("image", x_label, y_label, title)
        array = np.asarray(array, dtype=float)
        x, array = T.sorted_axis(x, array, 0)
        y, array = T.sorted_axis(y, array, 1)
        p.array, p.x, p.y = array, x, y
        p.cmap, p.flip = cmap, flip
        p.levels = tuple(levels) if levels is not None else None
        p.gamma = float(gamma) if gamma else 1.0
        return p

    @classmethod
    def curves(cls, curves, x_label="", y_label="", title=""):
        """``curves``: ``[(x, y, label, colour, sigma or None)]``"""
        p = cls("curves", x_label, y_label, title)
        p.curves = [(np.asarray(c[0], dtype=float), np.asarray(c[1], dtype=float),
                     c[2], c[3], None if len(c) < 5 or c[4] is None
                     else np.asarray(c[4], dtype=float)) for c in curves]
        return p

    def auto_levels(self):
        return T.auto_levels(self.array)


def panel_from_dataset(data, name="", cmap=T.DEFAULT_COLORMAP, flip=False):
    """A cut (or any 2-D dataset) as an image panel; a curve dataset as its
    curves."""
    from loader.nxs_file import CURVE_KINDS
    scan = data.scan
    labels = getattr(scan, "labels", {}) or {}
    if data.kind in CURVE_KINDS:
        from tools import curves as C
        from ui.curves import colour
        values = np.asarray(scan.value, dtype=float)
        values = values[:, None] if values.ndim == 1 else values
        names = C.channel_names(scan.info, values.shape[1], data.kind)
        curves = []
        for index in C.data_channels(names):
            s = C.sigma_of(names, index)
            curves.append((scan.x, values[:, index], names[index], colour(index),
                           values[:, s] if s is not None else None))
        return Panel.curves(curves, labels.get("x", ""), C.value_label(scan.info),
                            title=name)
    value = getattr(scan, "value", None)
    if value is None or np.ndim(value) != 2:
        raise ValueError("only a two-dimensional dataset can be a panel on its own; "
                         "take a slice of a map first")
    return Panel.image(np.asarray(value, dtype=float), scan.x, scan.y,
                       labels.get("x", ""), labels.get("y", ""), title=name,
                       cmap=cmap, flip=flip)


def slice_panels(cube, axes, labels, axis, values, half_width=0.0, titles=True,
                 cmap=T.DEFAULT_COLORMAP, flip=False):
    """Cut a cube ``(x, y, z)`` into image panels along ``axis``."""
    cube = np.asarray(cube, dtype=float)
    axes = [np.asarray(a, dtype=float) for a in axes]
    slicing = axes[axis]
    others = [i for i in range(3) if i != axis]
    unit = unit_of(labels[axis])
    panels = []
    for value in values:
        lo, hi = T.index_window(slicing, value, half_width)
        block = np.take(cube, np.arange(lo, hi + 1), axis=axis)
        with np.errstate(invalid="ignore"):
            image = np.nanmean(block, axis=axis)
        centre = float(slicing[T.nearest_index(slicing, value)])
        title = ("%.4g %s" % (centre, unit)).strip() if titles else ""
        if titles and half_width > 0:
            title += " ±%g" % half_width
        panels.append(Panel.image(image, axes[others[0]], axes[others[1]],
                                  labels[others[0]], labels[others[1]], title=title,
                                  cmap=cmap, flip=flip))
    return panels


def open_slice_series(contour):
    """Ask how to slice the contour's cube, then open the figure."""
    labels = (contour.defl_label, contour.slit_label, contour.energy_label)
    axes = (contour.defl, contour.slit, contour.E)
    names = ["along %s" % l for l in labels]
    v = T.ask_params(contour, "Slice series figure", [
        Field("axis", "Slice", "choice", names[2], names),
        Field("mode", "Positions", "choice", "evenly spaced", ["evenly spaced", "typed values"]),
        Field("n", "How many (evenly spaced)", "int", 6),
        Field("lo", "From (empty = start of axis)", "optfloat", None),
        Field("hi", "To (empty = end of axis)", "optfloat", None),
        Field("values", "Typed values", "str", "", width=30),
        Field("width", "Integrate ±", "float", 0.0),
        Field("cols", "Columns", "int", 3),
        Field("titles", "Title each panel with its position", "bool", True),
        Field("shared", "Same colour scale everywhere", "bool", True),
    ], text="One panel per slice through the cube; the other two axes become "
            "each panel's picture.")
    if v is None:
        return None
    index = names.index(v["axis"])
    axis = np.asarray(axes[index], dtype=float)
    if v["mode"] == "evenly spaced":
        lo = float(axis.min()) if v["lo"] is None else v["lo"]
        hi = float(axis.max()) if v["hi"] is None else v["hi"]
        values = list(np.linspace(lo, hi, max(1, int(v["n"]))))
    else:
        try:
            values = T.parse_floats(v["values"])
        except ValueError:
            values = []
        if not values:
            T.warning(contour, "Slice series", "Type the positions, e.g. 0, -0.1, -0.25")
            return None
    panels = T.with_wait_cursor(contour, slice_panels, contour.full_cube(), axes, labels,
                                index, values, v["width"], v["titles"],
                                contour.colormap, contour.flip)
    window = FigureWindow(contour, panels, "Figure — %s" % contour.filename,
                          app=contour.app, cols=max(1, int(v["cols"])),
                          shared=v["shared"], equal=(index == 2))
    return window


class FigureWindow(tk.Toplevel):
    """A page of panels, with its layout settings and export."""

    def __init__(self, master, panels, title="Figure", app=None, cols=None,
                 shared=True, equal=False):
        tk.Toplevel.__init__(self, master)
        self.title(title)
        self.geometry("1250x860")
        self.app = app
        self.panels = list(panels)
        n = len(self.panels)
        cols = cols or min(n, 3)
        rows = int(np.ceil(n / float(cols)))
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=4, pady=4)
        self.layout = T.Form(left, [
            Field("rows", "Rows", "int", rows),
            Field("cols", "Columns", "int", cols),
            Field("width", "Width (mm)", "float", 85.0 * min(cols, 2)),
            Field("height", "Height (mm)", "float", 70.0 * rows),
            Field("font", "Font size (pt)", "float", 8.0),
            Field("shared", "Shared colour scale", "bool", shared),
            Field("colorbar", "Colour bar", "bool", False),
            Field("equal", "Equal aspect (kx-ky)", "bool", equal),
            Field("outer", "Axis labels on outer edges only", "bool", True),
            Field("letters", "Panel letters (a), (b)...", "bool", n > 1),
            Field("dpi", "Export resolution (dpi)", "int", 300),
        ], on_change=self.refresh)
        self.layout.pack(fill="x")
        ttk.Button(left, text="Redraw", command=self.refresh).pack(fill="x", pady=2)
        ttk.Label(left, text="Panels").pack(anchor="w", pady=(8, 0))
        self.panel_list = tk.Listbox(left, height=8, exportselection=False)
        self.panel_list.pack(fill="x")
        for text, command in (("Edit panel...", self.edit_panel),
                              ("Move up", lambda: self.move(-1)),
                              ("Move down", lambda: self.move(1)),
                              ("Remove panel", self.remove_panel),
                              ("Add a panel from the list...", self.add_from_list),
                              ("Export...", self.export)):
            ttk.Button(left, text=text, command=command).pack(fill="x", pady=1)
        self.plot = T.PlotFrame(self, figsize=(7, 6))
        self.plot.pack(side="left", fill="both", expand=True)
        self.refresh()

    # -- panels ------------------------------------------------------------------------
    def _fill_list(self):
        self.panel_list.delete(0, "end")
        for i, p in enumerate(self.panels):
            self.panel_list.insert("end", "%d. %s [%s]" % (i + 1, p.title or "(untitled)", p.kind))

    def selected(self):
        sel = self.panel_list.curselection()
        return sel[0] if sel else None

    def move(self, delta):
        i = self.selected()
        if i is None:
            return
        j = int(np.clip(i + delta, 0, len(self.panels) - 1))
        self.panels[i], self.panels[j] = self.panels[j], self.panels[i]
        self.refresh()
        self.panel_list.selection_set(j)

    def remove_panel(self):
        i = self.selected()
        if i is not None and len(self.panels) > 1:
            del self.panels[i]
            self.refresh()

    def edit_panel(self):
        i = self.selected()
        if i is None:
            T.info(self, "Edit panel", "Select a panel in the list.")
            return
        p = self.panels[i]
        fields = [Field("title", "Title", "str", p.title, width=36),
                  Field("xl", "x label", "str", p.x_label, width=36),
                  Field("yl", "y label", "str", p.y_label, width=36),
                  Field("xlim", "x range (lo hi, empty = auto)", "str",
                        "" if p.xlim is None else "%g %g" % tuple(p.xlim)),
                  Field("ylim", "y range (lo hi, empty = auto)", "str",
                        "" if p.ylim is None else "%g %g" % tuple(p.ylim))]
        if p.kind == "image":
            fields += [Field("levels", "Levels (lo hi, empty = auto)", "str",
                             "" if p.levels is None else "%g %g" % tuple(p.levels)),
                       Field("gamma", "Gamma", "float", p.gamma),
                       Field("cmap", "Colormap", "choice", p.cmap, T.COLORMAP_NAMES),
                       Field("flip", "Flip colormap", "bool", p.flip)]
        else:
            fields += [Field("legend", "Legend", "bool", p.legend)]
        v = T.ask_params(self, "Edit panel", fields)
        if v is None:
            return

        def pair(text):
            try:
                values = T.parse_floats(text)
            except ValueError:
                return None
            return tuple(values[:2]) if len(values) >= 2 else None
        p.title, p.x_label, p.y_label = v["title"], v["xl"], v["yl"]
        p.xlim, p.ylim = pair(v["xlim"]), pair(v["ylim"])
        if p.kind == "image":
            p.levels = pair(v["levels"])
            p.gamma = v["gamma"] if v["gamma"] > 0 else 1.0
            p.cmap, p.flip = v["cmap"], v["flip"]
        else:
            p.legend = v["legend"]
        self.refresh()

    def add_from_list(self):
        if self.app is None or not self.app.entries():
            T.info(self, "Add a panel", "No dataset list available.")
            return
        entries = self.app.entries()
        index = T.choose(self, "Add a panel", "Dataset (a cut or a curve):",
                         [label for label, _ in entries])
        if index is None:
            return
        label, key = entries[index]
        try:
            data = self.app.load(key)
            panel = panel_from_dataset(data, self.app.label_for(key),
                                       self.app.colormap, self.app.flip)
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Add a panel", str(exc))
            return
        self.panels.append(panel)
        v = self.layout.values()
        if v["rows"] * v["cols"] < len(self.panels):
            self.layout.set("rows", int(np.ceil(len(self.panels) / float(v["cols"]))))
        self.refresh()

    # -- drawing --------------------------------------------------------------------------
    def draw_into(self, fig):
        from matplotlib import colors as mcolors
        try:
            v = self.layout.values()
        except T.FormError as exc:
            T.warning(self, "Figure", str(exc))
            return
        fig.clear()
        rows, cols = max(1, v["rows"]), max(1, v["cols"])
        if rows * cols < len(self.panels):
            rows = int(np.ceil(len(self.panels) / float(cols)))
        font = v["font"]
        images = [p for p in self.panels if p.kind == "image"]
        shared = None
        if v["shared"] and images:
            finite = np.concatenate([p.array[np.isfinite(p.array)].ravel() for p in images])
            shared = T.auto_levels(finite) if finite.size else (0.0, 1.0)
        axes = fig.subplots(rows, cols, squeeze=False)
        mappable = None
        for index, ax in enumerate(axes.ravel()):
            if index >= len(self.panels):
                ax.set_visible(False)
                continue
            p = self.panels[index]
            r, c = divmod(index, cols)
            if p.kind == "image":
                lo, hi = p.levels or shared or p.auto_levels()
                norm = (mcolors.Normalize(lo, hi) if abs(p.gamma - 1) < 1e-9
                        else mcolors.PowerNorm(p.gamma, lo, hi))
                ex, ey = T.edges_of(p.x), T.edges_of(p.y)
                if T.uniform(p.x) and T.uniform(p.y):
                    mappable = ax.imshow(p.array.T, origin="lower", aspect="auto",
                                         extent=(ex[0], ex[-1], ey[0], ey[-1]),
                                         cmap=T.get_cmap(p.cmap, p.flip), norm=norm,
                                         interpolation="nearest")
                else:
                    mappable = ax.pcolormesh(ex, ey, p.array.T, cmap=T.get_cmap(p.cmap, p.flip),
                                             norm=norm)
                if v["equal"]:
                    ax.set_aspect("equal", adjustable="box")
                if v["colorbar"] and not v["shared"]:
                    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
                    cb.ax.tick_params(labelsize=font * 0.8)
            else:
                for x, y, label, colour, sigma in p.curves:
                    ax.plot(x, y, color=colour, lw=1.0, label=label)
                    if sigma is not None:
                        ax.fill_between(x, y - sigma, y + sigma, color=colour, alpha=0.2, lw=0)
                if p.legend and p.curves:
                    ax.legend(fontsize=font * 0.8, frameon=False)
            if p.xlim is not None:
                ax.set_xlim(p.xlim)
            if p.ylim is not None:
                ax.set_ylim(p.ylim)
            outer_x = (not v["outer"]) or r == rows - 1 or index + cols >= len(self.panels)
            outer_y = (not v["outer"]) or c == 0
            ax.set_xlabel(p.x_label if outer_x else "", fontsize=font)
            ax.set_ylabel(p.y_label if outer_y else "", fontsize=font)
            if v["outer"] and not outer_x and p.kind == "image":
                ax.tick_params(labelbottom=False)
            if v["outer"] and not outer_y and p.kind == "image":
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=font * 0.9)
            title = p.title
            if v["letters"]:
                letter = "(%s)" % string.ascii_lowercase[index % 26]
                title = "%s %s" % (letter, title) if title else letter
            if title:
                ax.set_title(title, fontsize=font, loc="left")
        if v["colorbar"] and v["shared"] and mappable is not None:
            cb = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
            cb.ax.tick_params(labelsize=font * 0.8)
        try:
            fig.set_tight_layout(False)
            if not v["colorbar"]:
                fig.tight_layout(pad=0.4)
        except Exception:                                       # noqa: BLE001
            pass

    def refresh(self):
        self._fill_list()
        # The preview fills the window; the page size (mm) applies to Export.
        self.draw_into(self.plot.figure)
        self.plot.draw()

    def export(self):
        path = T.save_path(self, "Export figure", "figure.pdf",
                           [("PDF", "*.pdf"), ("PNG", "*.png"), ("SVG", "*.svg"),
                            ("EPS", "*.eps"), ("TIFF", "*.tif")])
        if not path:
            return None
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        v = self.layout.values()
        fig = Figure(figsize=(v["width"] / 25.4, v["height"] / 25.4))
        FigureCanvasAgg(fig)
        self.draw_into(fig)
        fig.savefig(path, dpi=v["dpi"])
        T.info(self, "Export", "Wrote %s" % path)
        return path
