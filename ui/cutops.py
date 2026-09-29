"""
ui/cutops.py
============
Cut arithmetic: two cuts combined into a third -- linear / circular
dichroism, dividing by a reference, differences, ratios, sums
(:mod:`tools.cutops` does the work).

Shows what the two measurements recorded side by side (rows that differ
are marked), warnings from the recorded polarisations, A and B (scaled, on
A's grid) on one colour scale, and the result -- signed results on a
red-white-blue scale symmetric about zero. **Result to list**,
**Uncertainty to list** (Poisson, when both are raw counts on one grid) and
**All three as a figure**.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk

from tools import cutops
from tools import process as P
from ui import tkbase as T
from ui.tkbase import Field
from ui.data import MemoryData

SUFFIX = {"linear_dichroism": "LD", "circular_dichroism": "CD", "reference": "norm",
          "difference": "diff", "asymmetry": "asym", "ratio": "ratio", "sum": "sum"}


def _fmt(value):
    return "%.6g" % value if isinstance(value, float) else str(value)


class CutArithmeticDialog(tk.Toplevel):
    """``a`` and ``b`` are ``(name, data)`` pairs; A is the first."""

    def __init__(self, app, a, b, master=None, colormap="gray", flip=False,
                 region_source=None):
        self._set_pair(a, b)            # raises ValueError before a window exists
        tk.Toplevel.__init__(self, master or app.root)
        self.app = app
        self.colormap, self.flip = colormap, flip
        self.region_source = region_source
        self.last = None
        self.title("Cut arithmetic")
        self.geometry("1150x950")
        head = ttk.Frame(self)
        head.pack(fill="x", padx=6, pady=4)
        self.names_label = ttk.Label(head, text="", justify="left")
        self.names_label.pack(side="left")
        ttk.Button(head, text="Swap A ↔ B", command=self.swap).pack(side="right")
        self.meta = ttk.Treeview(self, columns=("a", "b"), height=5)
        self.meta.heading("#0", text="")
        self.meta.heading("a", text="A")
        self.meta.heading("b", text="B")
        self.meta.tag_configure("differs", background="#ffe2b8")
        self.meta.pack(fill="x", padx=6)
        self.warn = ttk.Label(self, text="", foreground="#8a4b00", wraplength=1050)
        self.warn.pack(fill="x", padx=6)
        self.preset_keys = list(cutops.PRESETS)
        self.preset_labels = [cutops.PRESETS[k]["label"] for k in self.preset_keys]
        self.op_keys, self.op_labels = list(cutops.OPERATIONS), list(cutops.OPERATIONS.values())
        self.norm_keys, self.norm_labels = list(cutops.NORMALISATIONS), list(cutops.NORMALISATIONS.values())
        self.ref_keys, self.ref_labels = list(cutops.REFERENCE_SHAPES), list(cutops.REFERENCE_SHAPES.values())
        x, y = self.a_axes
        self.form = T.Form(self, [
            Field("preset", "Purpose", "choice", self.preset_labels[0], self.preset_labels),
            Field("operation", "Result", "choice", self.op_labels[0], self.op_labels),
            Field("normalise", "Scale B to A", "choice", self.norm_labels[0], self.norm_labels),
            Field("region", "Region x0 x1 y0 y1", "floats",
                  [float(x.min()), float(x.max()), float(y.min()), float(y.max())], width=30),
            Field("reference", "Divide by", "choice", self.ref_labels[0], self.ref_labels),
            Field("ref_floor", "Reference floor (%)", "float", 5.0),
            Field("int_floor", "Hide where A+B below (%)", "float", 2.0),
        ], on_change=self.refresh, columns=2)
        self.form.pack(fill="x", padx=6)
        self.form.widgets["preset"].bind("<<ComboboxSelected>>", lambda e: self.apply_preset(), add="+")
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6)
        ttk.Button(bar, text="Take the box from A's viewer", command=self.region_from_viewer).pack(side="left")
        self.plot = T.PlotFrame(self, figsize=(11, 4.2))
        self.plot.pack(fill="both", expand=True, padx=6)
        self.axs = [self.plot.figure.add_subplot(1, 3, i + 1) for i in range(3)]
        self.report = tk.Text(self, height=6, width=120, font=("Courier", 9))
        self.report.pack(fill="x", padx=6)
        bar2 = ttk.Frame(self)
        bar2.pack(fill="x", padx=6, pady=4)
        for text, command in (("Result to list", self.export_result),
                              ("Uncertainty to list", self.export_sigma),
                              ("All three as a figure", self.to_figure)):
            ttk.Button(bar2, text=text, command=command).pack(side="left", padx=2)
        ttk.Button(bar2, text="Close", command=self.destroy).pack(side="right")
        self._fill_names()
        self._fill_metadata()
        self.apply_preset()

    # -- the pair ----------------------------------------------------------------------
    def _set_pair(self, a, b):
        (self.name_a, self.a), (self.name_b, self.b) = a, b
        for name, data in (a, b):
            if getattr(data, "kind", None) != "cut":
                raise ValueError("%s is a %s; cut arithmetic works on two cuts. Take a "
                                 "slice of a map first." % (name, getattr(data, "kind", "dataset")))
        sa, sb = self.a.scan, self.b.scan
        self.a_axes = (np.asarray(sa.x, float), np.asarray(sa.y, float))
        self.b_axes = (np.asarray(sb.x, float), np.asarray(sb.y, float))
        self.a_values = np.asarray(self.a.cut_frame, dtype=float)
        self.b_values = np.asarray(self.b.cut_frame, dtype=float)
        self.labels = (sa.labels.get("x", "x"), sa.labels.get("y", "y"))
        self.b_labels = (sb.labels.get("x", "x"), sb.labels.get("y", "y"))

    def swap(self):
        self._set_pair((self.name_b, self.b), (self.name_a, self.a))
        x, y = self.a_axes
        self.form.set("region", [float(x.min()), float(x.max()), float(y.min()), float(y.max())])
        self._fill_names()
        self._fill_metadata()
        self.refresh()

    def _fill_names(self):
        self.names_label.configure(text="A:  %s\nB:  %s" % (self.name_a, self.name_b))

    def _fill_metadata(self):
        for item in self.meta.get_children():
            self.meta.delete(item)
        rows = cutops.metadata_differences(self.a.scan.info or {}, self.b.scan.info or {})
        self.differing = []
        for name, va, vb, differs in rows:
            flagged = differs and name != "Polarisation"
            if flagged:
                self.differing.append(name)
            self.meta.insert("", "end", text=name, values=(
                "—" if va is None else _fmt(va), "—" if vb is None else _fmt(vb)),
                tags=("differs",) if flagged else ())

    # -- settings ---------------------------------------------------------------------
    def apply_preset(self):
        key = self.preset_keys[self.preset_labels.index(self.form.raw("preset"))]
        spec = cutops.PRESETS[key]
        for field, table_keys, table_labels in (("operation", self.op_keys, self.op_labels),
                                                ("normalise", self.norm_keys, self.norm_labels),
                                                ("reference", self.ref_keys, self.ref_labels)):
            source = {"reference": "reference_shape"}.get(field, field)
            if source in spec:
                self.form.set(field, table_labels[table_keys.index(spec[source])])
        self.refresh()

    def region_from_viewer(self):
        box = self.region_source() if self.region_source else None
        if box is None:
            T.info(self, "Cut arithmetic", "A's viewer has no box set (row under its image).")
            return
        self.form.set("region", list(box))
        self.form.set("normalise", cutops.NORMALISATIONS["region"])
        self.refresh()

    def settings(self):
        v = self.form.values()
        region = v["region"]
        if len(region) != 4:
            raise ValueError("the region needs four numbers x0 x1 y0 y1")
        x0, x1, y0, y1 = region
        return dict(operation=self.op_keys[self.op_labels.index(v["operation"])],
                    normalise=self.norm_keys[self.norm_labels.index(v["normalise"])],
                    region=(x0, y0, x1, y1),
                    reference_shape=self.ref_keys[self.ref_labels.index(v["reference"])],
                    reference_floor=v["ref_floor"] / 100.0,
                    intensity_floor=v["int_floor"] / 100.0)

    def preset_key(self):
        return self.preset_keys[self.preset_labels.index(self.form.raw("preset"))]

    # -- computing --------------------------------------------------------------------
    def compute(self):
        return cutops.combine(self.a_values, self.a_axes, self.b_values, self.b_axes,
                              a_labels=self.labels, b_labels=self.b_labels, **self.settings())

    def _image(self, ax, values, lo, hi, cmap, title):
        ax.clear()
        x, y = self.a_axes
        ax.pcolormesh(T.edges_of(np.sort(x)), T.edges_of(np.sort(y)),
                      T.sorted_axis(y, T.sorted_axis(x, values, 0)[1], 1)[1].T,
                      cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(self.labels[0], fontsize=7)
        ax.set_ylabel(self.labels[1], fontsize=7)
        ax.tick_params(labelsize=7)

    def refresh(self):
        if not hasattr(self, "report"):
            return
        try:
            notes = cutops.polarisation_warnings(self.a.scan.info or {}, self.b.scan.info or {},
                                                 self.preset_key())
        except Exception:                                       # noqa: BLE001
            notes = []
        if self.differing:
            notes.append("A and B differ in " + ", ".join(self.differing) + " -- see the table.")
        self.warn.configure(text="\n".join(notes))
        try:
            result = self.compute()
        except (ValueError, ImportError, T.FormError) as exc:
            self.last = None
            self.report.delete("1.0", "end")
            self.report.insert("1.0", str(exc))
            return
        self.last = result
        a, b = self.a_values, result.b_used
        pool = np.concatenate([a[np.isfinite(a)].ravel(), b[np.isfinite(b)].ravel()])
        lo, hi = (np.percentile(pool, [1, 99]) if pool.size else (0.0, 1.0))
        cmap = T.get_cmap(self.colormap, self.flip)
        self._image(self.axs[0], a, lo, hi, cmap, "A: %s" % self.name_a)
        self._image(self.axs[1], b, lo, hi, cmap, "B (scaled, on A's grid)")
        values = result.values
        finite = values[np.isfinite(values)]
        operation = self.settings()["operation"]
        signed = operation in ("difference", "asymmetry")
        if not finite.size:
            rlo, rhi = -1.0, 1.0
        elif signed:
            reach = float(np.percentile(np.abs(finite), 98)) or 1.0
            rlo, rhi = -reach, reach
        else:
            rlo, rhi = (float(v) for v in np.percentile(finite, [2, 98]))
        self._image(self.axs[2], values, rlo, rhi,
                    T.get_cmap("redwhiteblue") if signed else cmap,
                    "%s  [%.4g .. %.4g]" % (cutops.OPERATIONS[operation], rlo, rhi))
        self.plot.draw()
        self.report.delete("1.0", "end")
        self.report.insert("1.0", result.summary() or "")

    # -- leaving --------------------------------------------------------------------
    def _parameters(self, result):
        s = self.settings()
        params = {"a": self.name_a, "b": self.name_b, "preset": self.preset_key(),
                  "operation": s["operation"], "scale_applied_to_b": float(result.scale),
                  "b_resampled": bool(result.resampled), "overlap": float(result.overlap)}
        if s["operation"] == "ratio":
            params.update(reference_shape=s["reference_shape"],
                          reference_floor=s["reference_floor"])
        else:
            params.update(normalise=s["normalise"], intensity_floor=s["intensity_floor"])
            if s["normalise"] == "region":
                params["region"] = list(s["region"])
        if result.median_sigma is not None:
            params["median_sigma"] = float(result.median_sigma)
            params["fraction_beyond_2sigma"] = float(result.significant)
        return params

    def _dataset(self, values, suffix, params):
        info = P.record_step(dict(self.a.scan.info or {}),
                             P.Step("cut_arithmetic", params,
                                    source="%s ∘ %s" % (self.name_a, self.name_b)))
        name = T.unique_name("%s %s" % (self.name_a, suffix),
                             self.app.names() if self.app is not None else [])
        return MemoryData("cut", tuple(self.a_axes), values, dict(self.a.scan.labels),
                          source_label=name, parameters=params, prefix="cutops",
                          source_path=getattr(self.a, "path", ""), source_info=info)

    def _emit(self, data):
        if self.app is not None:
            self.app.add_dataset(data)
        self.report.insert("end", "\nAdded “%s” to the list." % data.source_label)
        return data

    def export_result(self):
        if self.last is None:
            return None
        preset = self.preset_key()
        suffix = SUFFIX.get(preset if preset != "custom" else self.settings()["operation"], "calc")
        return self._emit(self._dataset(self.last.values, suffix, self._parameters(self.last)))

    def export_sigma(self):
        if self.last is None or self.last.sigma is None:
            T.info(self, "Cut arithmetic", "There is no uncertainty for this result (only "
                   "for raw counts on the same grid, and not for a ratio).")
            return None
        params = self._parameters(self.last)
        params["quantity"] = "Poisson standard deviation of the result"
        return self._emit(self._dataset(self.last.sigma, "σ", params))

    def to_figure(self):
        from ui.figure import FigureWindow, Panel
        if self.last is None:
            return None
        x, y = self.a_axes
        operation = self.settings()["operation"]
        signed = operation in ("difference", "asymmetry")
        panels = [Panel.image(self.a_values, x, y, *self.labels, title=self.name_a,
                              cmap=self.colormap, flip=self.flip),
                  Panel.image(self.last.b_used, x, y, *self.labels, title=self.name_b,
                              cmap=self.colormap, flip=self.flip),
                  Panel.image(self.last.values, x, y, *self.labels,
                              title=cutops.OPERATIONS[operation],
                              cmap="redwhiteblue" if signed else self.colormap,
                              flip=False if signed else self.flip)]
        return FigureWindow(self, panels, "Cut arithmetic", app=self.app, cols=3, shared=False)
