"""
ui/process.py
=============
Processing of 2-D datasets (cuts), and the EDC / MDC stack plot.

:class:`ProcessDialog` -- smooth (Gaussian / Savitzky-Golay / moving
average, optional despike), derivatives (-d2/dE2, -d2/dk2, Laplacian,
-d/dE), curvature (2-D, 1-D, gradient enhancement), backgrounds (Shirley,
Tougaard, percentile, polynomial, divide the Fermi cut-off) and mirror
symmetrisation. The parameters are typed; the preview (on a decimated copy)
shows the source and the result side by side; **Apply** runs it on every
selected dataset and lists the results with the operation's suffix.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk

from tools import process as P
from ui import tkbase as T
from ui.tkbase import Field
from ui.data import MemoryData

PREVIEW_POINTS = 250000


def decimate(values, axes, budget=PREVIEW_POINTS):
    values = np.asarray(values, dtype=float)
    if values.size <= budget:
        return values, [np.asarray(a, dtype=float) for a in axes]
    factor = int(np.ceil(np.sqrt(values.size / float(budget))))
    sl = tuple(slice(None, None, factor) for _ in range(values.ndim))
    return values[sl], [np.asarray(a, dtype=float)[::factor] for a in axes]


OPERATIONS = ("Smooth", "Derivative", "Curvature", "Background", "Symmetrise")
SUFFIXES = {"smooth": "_sm", "derivative": "_d2", "curvature": "_cur",
            "gradient": "_grad", "background": "_bg", "fermi": "_fd",
            "despike": "_fix", "symmetry": "_symm"}


class ProcessDialog(tk.Toplevel):
    """Smooth, differentiate, curvature, background, symmetrise a cut."""

    def __init__(self, app, datasets, master=None, colormap="gray", flip=False):
        tk.Toplevel.__init__(self, master or app.root)
        self.app = app
        self.datasets = list(datasets)
        self.colormap, self.flip = colormap, flip
        self.title("Data processing (2D)")
        self.geometry("1150x900")
        name, data = self.datasets[0]
        scan = data.scan
        self.values = np.asarray(scan.value, dtype=float)
        self.axes = [np.asarray(scan.x, dtype=float), np.asarray(scan.y, dtype=float)]
        self.labels = [scan.labels.get("x", "x"), scan.labels.get("y", "y")]
        self.pv, self.pa = decimate(self.values, self.axes)
        names = ", ".join(n for n, _ in self.datasets[:3])
        ttk.Label(self, text="%s   (%d x %d points; %s x %s)" % (
            names, self.values.shape[0], self.values.shape[1], *self.labels)).pack(
            fill="x", padx=6, pady=4)
        top = ttk.Frame(self)
        top.pack(fill="x", padx=6)
        ttk.Label(top, text="Operation").pack(side="left")
        self.op_var = tk.StringVar(value=OPERATIONS[0])
        combo = ttk.Combobox(top, textvariable=self.op_var, values=OPERATIONS,
                             state="readonly", width=14)
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", lambda e: self.show_form())
        self.live = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Live preview", variable=self.live).pack(side="left", padx=8)
        self.form_holder = ttk.Frame(self)
        self.form_holder.pack(fill="x", padx=6)
        self.forms = {}
        sx = abs(self.axes[0][-1] - self.axes[0][0]) or 1.0
        sy = abs(self.axes[1][-1] - self.axes[1][0]) or 1.0
        s0, s1 = self._short(0), self._short(1)
        self.specs = {
            "Smooth": [
                Field("method", "Method", "choice", "gaussian", ["gaussian", "savgol", "box"],
                      help="gaussian: width = sigma; savgol / box: full window"),
                Field("wx", "Width along %s" % self.labels[0], "float", round(sx / 100, 6)),
                Field("wy", "Width along %s" % self.labels[1], "float", round(sy / 100, 6)),
                Field("despike", "Remove spikes first", "bool", False),
                Field("threshold", "  spike threshold (σ)", "float", 6.0),
                Field("scale", "  noise scale", "choice", "poisson", ["poisson", "global", "local"]),
            ],
            "Derivative": [
                Field("quantity", "Quantity", "choice", "-d2/d%s2" % s1,
                      ["-d2/d%s2" % s1, "-d2/d%s2" % s0, "Laplacian", "-d/d%s" % s1]),
                Field("wx", "Pre-smooth along %s" % self.labels[0], "float", round(sx / 60, 6)),
                Field("wy", "Pre-smooth along %s" % self.labels[1], "float", round(sy / 60, 6)),
                Field("window", "Fit window (0 = 7 points)", "float", 0.0),
            ],
            "Curvature": [
                Field("mode", "Mode", "choice", "2d", ["2d", "1d along %s" % s1,
                                                       "1d along %s" % s0, "gradient enhancement"]),
                Field("a0", "a0 (dimensionless)", "float", 1.0),
                Field("ratio", "Anisotropy ratio", "float", 1.0),
                Field("wx", "Pre-smooth along %s" % self.labels[0], "float", round(sx / 60, 6)),
                Field("wy", "Pre-smooth along %s" % self.labels[1], "float", round(sy / 60, 6)),
            ],
            "Background": [
                Field("method", "Method", "choice", "shirley",
                      ["shirley", "tougaard", "percentile", "polynomial",
                       "divide the Fermi cut-off"]),
                Field("dim", "Direction", "choice", "along %s" % self.labels[1],
                      ["along %s" % self.labels[1], "along %s" % self.labels[0]]),
                Field("percentile", "Percentile (%)", "float", 5.0),
                Field("order", "Polynomial order", "int", 2),
                Field("clip", "Hold the result at zero", "bool", False),
                Field("ef", "E_F (eV) [Fermi]", "float", 0.0),
                Field("temp", "Temperature (K) [Fermi]", "float", 30.0),
                Field("res", "Resolution FWHM (eV) [Fermi]", "float", 0.015),
                Field("cut", "Stop above (kT) [Fermi]", "float", 4.0),
            ],
            "Symmetrise": [
                Field("angle", "Mirror line angle (deg)", "float", 0.0),
                Field("cx", "Line passes through x", "float", 0.0),
                Field("cy", "Line passes through y", "float", 0.0),
            ],
        }
        for name_, spec in self.specs.items():
            self.forms[name_] = T.Form(self.form_holder, spec, on_change=self._changed,
                                       columns=2)
        extra = ttk.Frame(self)
        extra.pack(fill="x", padx=6)
        ttk.Button(extra, text="Suggest a0", command=self.suggest_a0).pack(side="left")
        ttk.Button(extra, text="Mirror point: pick on the source", command=self.pick_mirror).pack(side="left", padx=4)
        self.plot = T.PlotFrame(self, figsize=(10, 4.6))
        self.plot.pack(fill="both", expand=True, padx=6)
        self.ax_src = self.plot.figure.add_subplot(121)
        self.ax_res = self.plot.figure.add_subplot(122)
        self.plot.canvas.mpl_connect("button_press_event", self._clicked)
        self._picking = False
        self.note = ttk.Label(self, text="", wraplength=1000, foreground="#225")
        self.note.pack(fill="x", padx=6)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Preview", command=self.refresh).pack(side="left", padx=2)
        ttk.Button(bar, text="Apply", command=self.apply).pack(side="left", padx=2)
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right")
        self.show_form()

    def _short(self, dim):
        return self.labels[dim].split(" (")[0]

    def show_form(self):
        for form in self.forms.values():
            form.pack_forget()
        self.forms[self.op_var.get()].pack(fill="x")
        self.refresh()

    def _changed(self):
        if self.live.get():
            self.refresh()

    # -- the operation ------------------------------------------------------------
    def operation(self):
        name = self.op_var.get()
        v = self.forms[name].values()
        if name == "Smooth":
            kind, widths = v["method"], (v["wx"], v["wy"])
            params = {"method": kind, "width_x": widths[0], "width_y": widths[1]}
            if v["despike"]:
                params.update({"despike": v["threshold"], "despike_scale": v["scale"]})

            def run(values, axes):
                out = values
                if v["despike"]:
                    out, _ = P.despike(out, threshold=v["threshold"], scale=v["scale"])
                if any(w > 0 for w in widths):
                    out = P.SMOOTHERS[kind](out, axes, widths)
                return out, axes
            return "smooth", params, run
        if name == "Derivative":
            choice = self.specs[name][0].choices.index(v["quantity"])
            widths = (v["wx"], v["wy"])
            window = v["window"] or None
            params = {"quantity": ["d2_y", "d2_x", "both", "d1_y"][choice],
                      "smooth_x": widths[0], "smooth_y": widths[1]}
            if window:
                params["window"] = window

            def run(values, axes):
                out = values
                if any(w > 0 for w in widths):
                    out = P.gaussian_smooth(out, axes, widths)
                if choice == 0:
                    out = P.derivative(out, axes, 1, 2, window=window)
                elif choice == 1:
                    out = P.derivative(out, axes, 0, 2, window=window)
                elif choice == 2:
                    out = P.laplacian(out, axes, window=window)
                else:
                    out = P.derivative(out, axes, 1, 1, window=window)
                return out, axes
            return "derivative", params, run
        if name == "Curvature":
            choice = self.specs[name][0].choices.index(v["mode"])
            mode = ["2d", "y", "x"][choice] if choice < 3 else None
            widths = (v["wx"], v["wy"])
            params = ({"mode": mode, "a0": v["a0"], "ratio": v["ratio"],
                       "smooth_x": widths[0], "smooth_y": widths[1]} if mode
                      else {"smooth_x": widths[0], "smooth_y": widths[1]})

            def run(values, axes):
                out = values
                if any(w > 0 for w in widths):
                    out = P.gaussian_smooth(out, axes, widths)
                if mode is None:
                    return P.gradient_enhance(out, axes), axes
                return P.curvature(out, axes, mode=mode, a0=v["a0"], ratio=v["ratio"]), axes
            return ("curvature" if mode else "gradient"), params, run
        if name == "Background":
            dim = 1 if v["dim"].endswith(self.labels[1]) else 0
            if v["method"].startswith("divide"):
                params = {"ef": v["ef"], "temperature": v["temp"], "resolution": v["res"],
                          "cutoff_kt": v["cut"], "axis": dim}

                def run(values, axes):
                    return P.divide_fermi_edge(values, axes, ef=v["ef"], temperature=v["temp"],
                                               resolution=v["res"], dim=dim,
                                               cutoff_kt=v["cut"]), axes
                return "fermi", params, run
            method = v["method"]
            params = {"method": method, "axis": dim, "clip": v["clip"]}
            if method == "percentile":
                params["percentile"] = v["percentile"]
            if method == "polynomial":
                params["order"] = v["order"]

            def run(values, axes):
                out, _ = P.subtract_background(values, axes, method, dim=dim, clip=v["clip"],
                                               percentile=v["percentile"], order=v["order"])
                return out, axes
            return "background", params, run
        centre = (v["cx"], v["cy"])
        params = {"mirror": v["angle"], "centre": list(centre)}

        def run(values, axes):
            result = P.symmetrise(values, axes[0], axes[1], fold=1, centre=centre,
                                  mirror_angles=(v["angle"],))
            self._residual = result.residual
            return result.values, axes
        return "symmetry", params, run

    # -- preview ------------------------------------------------------------------
    def _image(self, ax, values, axes, title):
        ax.clear()
        lo, hi = T.auto_levels(values)
        ex, ey = T.edges_of(axes[0]), T.edges_of(axes[1])
        ax.pcolormesh(ex, ey, np.asarray(values).T, cmap=T.get_cmap(self.colormap, self.flip),
                      vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(self.labels[0], fontsize=8)
        ax.set_ylabel(self.labels[1], fontsize=8)
        ax.tick_params(labelsize=7)

    def refresh(self):
        try:
            key, params, run = self.operation()
            values, axes = run(self.pv, self.pa)
        except Exception as exc:                                # noqa: BLE001
            self.note.configure(text=str(exc))
            return
        self._image(self.ax_src, self.pv, self.pa, "Source")
        self._image(self.ax_res, values, axes, "Result")
        if key == "symmetry":
            v = self.forms["Symmetrise"].values()
            r = np.radians(v["angle"])
            span = max(np.ptp(self.pa[0]), np.ptp(self.pa[1]))
            for ax in (self.ax_src, self.ax_res):
                ax.plot([v["cx"] - span * np.cos(r), v["cx"] + span * np.cos(r)],
                        [v["cy"] - span * np.sin(r), v["cy"] + span * np.sin(r)],
                        "--", color="#00e5ff")
                ax.set_xlim(T.edges_of(self.pa[0])[[0, -1]])
                ax.set_ylim(T.edges_of(self.pa[1])[[0, -1]])
        self.plot.draw()
        pieces = ["%s: %s" % (key, ", ".join("%s=%s" % kv for kv in sorted(params.items())))]
        finite = np.isfinite(values)
        missing = 100.0 * (1.0 - finite.mean()) if finite.size else 0.0
        if missing > 0.05:
            pieces.append("%.1f%% missing" % missing)
        if key == "symmetry" and getattr(self, "_residual", None) is not None:
            pieces.append("the two halves differed by %.1f%%" % (100 * self._residual))
        if self.pv.size < self.values.size:
            pieces.append("preview on a decimated copy")
        self.note.configure(text="  ·  ".join(pieces))

    def suggest_a0(self):
        form = self.forms["Curvature"]
        v = form.values()
        choice = self.specs["Curvature"][0].choices.index(v["mode"])
        mode = ["2d", "y", "x"][choice] if choice < 3 else "2d"
        values = self.pv
        if v["wx"] > 0 or v["wy"] > 0:
            values = P.gaussian_smooth(values, self.pa, (v["wx"], v["wy"]))
        form.set("a0", P.suggest_a0(values, self.pa, mode=mode))
        self.refresh()

    def pick_mirror(self):
        self.op_var.set("Symmetrise")
        self.show_form()
        self._picking = True
        self.note.configure(text="Click a point on the Source image.")

    def _clicked(self, event):
        if not self._picking or event.inaxes is not self.ax_src or event.xdata is None:
            return
        self._picking = False
        form = self.forms["Symmetrise"]
        form.set("cx", event.xdata)
        form.set("cy", event.ydata)
        self.refresh()

    # -- apply ----------------------------------------------------------------------
    def apply(self):
        try:
            key, params, run = self.operation()
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Data processing", str(exc))
            return None
        prepared = [(name, data, np.asarray(data.scan.value, dtype=float),
                     [np.asarray(data.scan.x, dtype=float), np.asarray(data.scan.y, dtype=float)])
                    for name, data in self.datasets]

        def work(report):
            made = []
            for index, (name, data, values, axes) in enumerate(prepared):
                report(index / float(max(len(prepared), 1)), name)
                out, new_axes = run(values, axes)
                info = P.record_step(dict(data.scan.info), P.Step(key, params, source=name))
                made.append(MemoryData("cut", tuple(new_axes), out, dict(data.scan.labels),
                                       source_label="%s%s" % (name, SUFFIXES.get(key, "_proc")),
                                       parameters=params, prefix="proc.%s" % key,
                                       source_path=getattr(data, "path", ""),
                                       source_info=info,
                                       source_motors=dict(data.scan.fourd_info)))
            return made

        def done(made):
            for data in made:
                self.app.add_dataset(data)
            self.note.configure(text="Added %d dataset(s) ending in “%s” to the list."
                                % (len(made), SUFFIXES.get(key, "_proc")))
        T.run_job(self, "Processing", work, on_done=done)


# ==========================================================================
# The EDC / MDC stack plot
# ==========================================================================
class StackWindow(tk.Toplevel):
    """A waterfall of EDCs or MDCs across a 2-D dataset: how many, over
    which range, combined how, normalised how, offset by position or by
    index, coloured along a colormap."""

    def __init__(self, master, values, axes, labels, source_name, app=None):
        tk.Toplevel.__init__(self, master)
        self.app = app if app is not None else getattr(master, "app", None)
        self.values = np.asarray(values, dtype=float)
        self.axes = [np.asarray(a, dtype=float) for a in axes]
        self.labels = list(labels)
        self.source_name = source_name
        self.title("Stack plot — %s" % source_name)
        self.geometry("1150x760")
        side = ttk.Frame(self)
        side.pack(side="left", fill="y", padx=4, pady=4)
        self.form = T.Form(side, [
            Field("direction", "Curves", "choice", "EDCs (along %s)" % labels[1],
                  ["EDCs (along %s)" % labels[1], "MDCs (along %s)" % labels[0]]),
            Field("count", "How many", "int", 12),
            Field("combine", "Combine n neighbours", "int", 1),
            Field("c0", "Curves from (position)", "optfloat", None),
            Field("c1", "Curves to", "optfloat", None),
            Field("l0", "Each curve over: from", "optfloat", None),
            Field("l1", "  to", "optfloat", None),
            Field("normalise", "Normalise each", "choice", "none", ["none", "area", "max"]),
            Field("offset_mode", "Offset", "choice", "by position", ["by position", "by index"]),
            Field("offset", "Offset scale", "float", 1.0),
            Field("shear", "Shear along x", "float", 0.0),
            Field("colour", "Colour", "choice", "colormap by position",
                  ["colormap by position", "single colour"]),
            Field("cmap", "Colormap", "choice", "jet", T.COLORMAP_NAMES),
            Field("width", "Line width", "float", 1.2),
            Field("annotate", "Label each curve", "bool", True),
            Field("fill", "Fill under the curves", "bool", False),
        ], on_change=self.redraw)
        self.form.pack(fill="x")
        ttk.Button(side, text="Redraw", command=self.redraw).pack(fill="x", pady=2)
        ttk.Button(side, text="Curves to the list", command=self.to_list).pack(fill="x", pady=2)
        ttk.Button(side, text="Export the curves (text)...", command=self.export_text).pack(fill="x", pady=2)
        self.note = ttk.Label(side, text="", wraplength=320)
        self.note.pack(fill="x")
        self.plot = T.PlotFrame(self, figsize=(7, 6))
        self.plot.pack(side="left", fill="both", expand=True)
        self.ax = self.plot.figure.add_subplot(111)
        self.redraw()

    def curves(self):
        v = self.form.values()
        edc = v["direction"].startswith("EDC")
        along = self.axes[1] if edc else self.axes[0]
        across = self.axes[0] if edc else self.axes[1]
        lo, hi = 0, across.size - 1
        if v["c0"] is not None:
            lo = T.nearest_index(across, v["c0"])
        if v["c1"] is not None:
            hi = T.nearest_index(across, v["c1"])
        lo, hi = min(lo, hi), max(lo, hi)
        wanted = np.unique(np.linspace(lo, hi, max(1, int(v["count"]))).round().astype(int))
        half = max(0, (int(v["combine"]) - 1) // 2)
        keep = np.ones(along.size, dtype=bool)
        if v["l0"] is not None:
            keep &= along >= v["l0"]
        if v["l1"] is not None:
            keep &= along <= v["l1"]
        if not keep.any():
            keep[:] = True
        along = along[keep]
        out = []
        for i in wanted:
            first, last = max(0, i - half), min(across.size, i + half + 1)
            block = self.values[first:last, :] if edc else self.values[:, first:last]
            line = np.nansum(block, axis=0 if edc else 1)[keep]
            if v["normalise"] == "area":
                total = np.nansum(line) * P.axis_step(along)
                line = line / total if abs(total) > 1e-30 else line
            elif v["normalise"] == "max":
                top = np.nanmax(line)
                line = line / top if abs(top) > 1e-30 else line
            out.append((float(across[i]), line))
        return v, edc, along, out

    def layout_curves(self):
        v, edc, along, curves = self.curves()
        positions = np.array([p for p, _ in curves], dtype=float)
        span = float(positions[-1] - positions[0]) or 1.0
        heights = [np.nanmax(y) - np.nanmin(y) for _, y in curves]
        unit = float(np.nanmedian(heights)) or 1.0
        n = max(len(curves) - 1, 1)
        cmap = T.get_cmap(v["cmap"])
        out = []
        for i, (position, y) in enumerate(curves):
            fraction = ((position - positions[0]) / span if v["offset_mode"] == "by position"
                        else i / float(n))
            shift = fraction * v["offset"] * unit * n
            x = along + fraction * v["shear"] * n
            colour = cmap(fraction) if v["colour"].startswith("colormap") else "#bd4921"
            out.append((position, x, y, shift, colour))
        return v, edc, out

    def redraw(self):
        try:
            v, edc, lines = self.layout_curves()
        except Exception as exc:                                # noqa: BLE001
            self.note.configure(text=str(exc))
            return
        ax = self.ax
        ax.clear()
        for position, x, y, shift, colour in lines:
            ax.plot(x, y + shift, color=colour, lw=v["width"])
            if v["fill"]:
                ax.fill_between(x, shift, y + shift, color=colour, alpha=0.3, lw=0)
            if v["annotate"]:
                ax.text(x[-1], y[-1] + shift, " %.4g" % position, color=colour, fontsize=7,
                        va="center")
        ax.set_xlabel(self.labels[1] if edc else self.labels[0])
        ax.set_ylabel("Intensity (offset)")
        self.plot.draw()
        self.note.configure(text="%d curves" % len(lines))

    def to_list(self):
        from ui.data import curve_dataset
        if self.app is None:
            return
        v, edc, along, curves = self.curves()
        kind = "edc" if edc else "mdc"
        names = ["%s @ %.4g" % (kind.upper(), p) for p, _ in curves]
        values = np.column_stack([y for _, y in curves])
        name = T.unique_name("%s [stack %s]" % (self.source_name, kind.upper()), self.app.names())
        data = curve_dataset(kind, along, values, names,
                             x_label=self.labels[1] if edc else self.labels[0],
                             value_label="Intensity", name=name, step="stack",
                             parameters={"count": len(curves), "combine": v["combine"],
                                         "normalise": v["normalise"]},
                             source=self.source_name)
        self.app.add_dataset(data)
        self.note.configure(text="Added “%s” to the list" % name)

    def export_text(self):
        v, edc, along, curves = self.curves()
        path = T.save_path(self, "Export curves", T.safe_stem(self.source_name) + "_stack.txt",
                           [("Text", "*.txt")])
        if not path:
            return
        table = np.column_stack([along] + [y for _, y in curves])
        header = "\t".join([self.labels[1] if edc else self.labels[0]] +
                           ["%.6g" % p for p, _ in curves])
        np.savetxt(path, table, delimiter="\t", fmt="%.8g", header=header, comments="", encoding="utf-8")
