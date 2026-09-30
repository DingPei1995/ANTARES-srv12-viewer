"""
ui/curves.py
============
The curve viewer -- one-dimensional data (EDC, MDC, spin EDC) -- and the two
panels that open from it: curve fitting and spin analysis.

Every channel is drawn, with error bars where the dataset carries a
``σ <name>`` channel (see :mod:`tools.curves`). Channels can be hidden and
offset into a waterfall. The **Range** (typed ``lo hi``) is what Crop keeps,
what a region normalisation / background is taken from, and the fit window.

**Operations** (Functions menu) each make a new dataset in the list: crop,
bin, normalise, subtract a background, shift the axis, add counting errors.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk

from tools import curves as C
from tools import spin as S
from ui import tkbase as T
from ui.tkbase import Field
from ui.data import curve_dataset

CHANNEL_COLOURS = ("#1f5aa6", "#c23b22", "#2a8a3e", "#8a4fb0", "#d08a00",
                   "#2aa1a8", "#6b6b6b", "#b0306e")


def colour(index):
    return CHANNEL_COLOURS[index % len(CHANNEL_COLOURS)]


class CurveWindow(tk.Toplevel):
    """Every channel of a curve dataset as a curve."""

    def __init__(self, app, data, filename, colormap="gray", flip=False, master=None):
        tk.Toplevel.__init__(self, master or (app.root if app is not None else None))
        self.app, self.data, self.filename = app, data, filename
        self.colormap, self.flip = colormap, flip
        self.kind = data.kind
        scan = data.scan
        self.x = np.asarray(scan.x, dtype=float).reshape(-1)
        values = np.asarray(scan.value, dtype=float)
        self.values = values[:, None] if values.ndim == 1 else values
        if self.values.shape[0] != self.x.size and self.values.shape[1] == self.x.size:
            self.values = self.values.T
        self.info = dict(scan.info or {})
        self.names = C.channel_names(self.info, self.values.shape[1], self.kind)
        self.x_label = (scan.labels or {}).get("x") or (
            "Energy (eV)" if self.kind != "mdc" else "Angle (deg)")
        self.value_label = C.value_label(self.info)
        self.children_windows = []
        self.title("%s  [%s]" % (filename, self.kind))
        self.geometry("1000x700")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self.redraw()

    # -- layout ------------------------------------------------------------------
    def _build(self):
        menubar = tk.Menu(self)
        self.configure(menu=menubar)
        functions = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Functions", menu=functions)
        functions.add_command(label="Curve fit...", command=self.open_fit)
        if self.kind == "spin_edc":
            functions.add_command(label="Spin analysis...", command=self.open_spin_analysis)
        functions.add_separator()
        for key, text in (("crop", "Crop to the Range"), ("bin", "Bin..."),
                          ("normalise", "Normalise..."),
                          ("background", "Subtract a background..."),
                          ("shift", "Shift the axis..."),
                          ("poisson", "Add counting errors (√N)")):
            functions.add_command(label=text, command=lambda k=key: self.ask_operation(k))
        functions.add_separator()
        functions.add_command(label="As a figure...", command=self.to_figure)
        functions.add_command(label="Export as text...", command=self.export_text)

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=4, pady=2)
        self.errors_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Error bars", variable=self.errors_var,
                        command=self.redraw).pack(side="left")
        ttk.Label(bar, text="Offset").pack(side="left", padx=(8, 2))
        self.offset_var = tk.StringVar(value="0")
        e = ttk.Entry(bar, textvariable=self.offset_var, width=8)
        e.pack(side="left")
        e.bind("<Return>", lambda ev: self.redraw())
        ttk.Label(bar, text="Range lo hi").pack(side="left", padx=(12, 2))
        self.range_var = tk.StringVar(value="")
        e2 = ttk.Entry(bar, textvariable=self.range_var, width=20)
        e2.pack(side="left")
        e2.bind("<Return>", lambda ev: self.redraw())
        T.Tooltip(e2, "Two numbers. Used by Crop, region normalisation / "
                      "background, and as the fit window. Empty = off.")
        ttk.Button(bar, text="From zoom", command=self.range_from_zoom).pack(side="left", padx=2)
        self.readout = ttk.Label(bar, text="")
        self.readout.pack(side="left", padx=8)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=4)
        ttk.Label(left, text="Channels").pack(anchor="w")
        self.channel_vars = {}
        for index in C.data_channels(self.names):
            var = tk.BooleanVar(value=True)
            tk.Checkbutton(left, text=self.names[index], variable=var, fg=colour(index),
                           command=self.redraw, anchor="w").pack(fill="x")
            self.channel_vars[index] = var
        self.plot = T.PlotFrame(body, figsize=(7.5, 5))
        self.plot.pack(side="left", fill="both", expand=True)
        self.ax = self.plot.figure.add_subplot(111)
        self.plot.canvas.mpl_connect("motion_notify_event", self._moved)
        self.status = ttk.Label(self, text="", anchor="w", relief="sunken")
        self.status.pack(side="bottom", fill="x")

    def say(self, text):
        self.status.configure(text=text)

    def close(self):
        for window in list(self.children_windows):
            try:
                window.destroy()
            except tk.TclError:
                pass
        if self.app is not None:
            self.app.forget_viewer(self)
        self.destroy()

    def set_colormap(self, name, flip):
        self.colormap, self.flip = name, flip

    # -- range -------------------------------------------------------------------------
    def x_range(self):
        text = self.range_var.get().strip()
        if not text:
            return None
        try:
            values = T.parse_floats(text)
        except ValueError:
            return None
        if len(values) != 2:
            return None
        return (min(values), max(values))

    def set_x_range(self, lo, hi):
        self.range_var.set("%.6g %.6g" % (lo, hi))
        self.redraw()

    def range_from_zoom(self):
        lo, hi = self.ax.get_xlim()
        self.set_x_range(lo, hi)

    def shown_channels(self):
        return [i for i, var in self.channel_vars.items() if var.get()]

    def offset(self):
        try:
            return float(self.offset_var.get())
        except ValueError:
            return 0.0

    def redraw(self):
        ax = self.ax
        xlim = ax.get_xlim() if ax.lines else None
        ax.clear()
        offset = self.offset()
        for order, index in enumerate(self.shown_channels()):
            y = self.values[:, index] + order * offset
            ax.plot(self.x, y, "o-", ms=2.5, lw=1.2, color=colour(index),
                    label=self.names[index])
            s = C.sigma_of(self.names, index)
            if s is not None and self.errors_var.get():
                sigma = self.values[:, s]
                good = np.isfinite(sigma) & np.isfinite(y)
                ax.errorbar(self.x[good], y[good], yerr=sigma[good], fmt="none",
                            ecolor=colour(index), elinewidth=0.8)
        region = self.x_range()
        if region is not None:
            ax.axvspan(region[0], region[1], color="#39c2ff", alpha=0.15)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.value_label)
        ax.grid(True, alpha=0.25)
        if self.shown_channels():
            ax.legend(fontsize=8)
        if xlim is not None:
            ax.set_xlim(xlim)
        self.plot.draw()

    def _moved(self, event):
        if event.inaxes is self.ax and event.xdata is not None:
            self.readout.configure(text="x = %.5g   y = %.5g" % (event.xdata, event.ydata))

    # -- operations --------------------------------------------------------------------
    def existing_names(self):
        return list(self.app.names()) if self.app is not None else []

    def _emit(self, x, values, names, suffix, step, parameters, x_label=None,
              value_label=None, kind=None):
        name = T.unique_name("%s [%s]" % (self.filename, suffix), self.existing_names())
        data = curve_dataset(kind or self.kind, x, values, names,
                             x_label=x_label or self.x_label,
                             value_label=value_label or self.value_label,
                             name=name, step=step, parameters=parameters,
                             source_info=self.info, source=self.filename,
                             source_path=getattr(self.data, "path", ""))
        if self.app is not None:
            self.app.add_dataset(data)
        self.say("Added “%s” to the list" % name)
        return data

    def ask_operation(self, key):
        fields = {
            "crop": None,
            "bin": [Field("factor", "Merge n neighbours", "int", 2),
                    Field("how", "How", "choice", "sum", ["sum", "mean"],
                          help="sum keeps counts as counts")],
            "normalise": [Field("how", "Normalise", "choice", "max", ["max", "area", "region"],
                                help="region = the mean over the Range"),
                          Field("together", "One factor for all channels", "bool", True)],
            "background": [Field("how", "Background", "choice", "constant",
                                 ["constant", "linear", "shirley"],
                                 help="constant / linear are taken from the Range")],
            "shift": [Field("delta", "Subtract from the axis", "float", 0.0),
                      Field("relabel", "Relabel as E − E_F", "bool",
                            T.is_energy_label(self.x_label))],
            "poisson": None,
        }[key]
        params = {}
        if fields is not None:
            params = T.ask_params(self, "Operation", fields)
            if params is None:
                return None
        return self.apply_operation(key, **params)

    def apply_operation(self, key, **p):
        x, values, names = self.x, self.values, list(self.names)
        region = self.x_range()
        try:
            if key == "crop":
                if region is None:
                    raise ValueError("type the Range (lo hi) first")
                x2, v2, n2 = C.crop(x, values, names, *region)
                return self._emit(x2, v2, n2, "crop", "crop", {"range": list(region)})
            if key == "bin":
                factor, how = int(p.get("factor", 2)), p.get("how", "sum")
                x2, v2, n2 = C.rebin(x, values, names, factor, how)
                return self._emit(x2, v2, n2, "bin%d" % factor, "bin",
                                  {"factor": factor, "how": how})
            if key == "normalise":
                how, together = p.get("how", "max"), p.get("together", True)
                if how == "region" and region is None:
                    raise ValueError("type the Range (lo hi) first")
                x2, v2, n2, factors = C.normalise(x, values, names, how, region, together)
                return self._emit(x2, v2, n2, "norm", "normalise",
                                  {"how": how, "together": together,
                                   "region": list(region) if region else "",
                                   "factors": ", ".join("%s=%.6g" % (names[i], f)
                                                        for i, f in factors.items())},
                                  value_label="%s (normalised)" % self.value_label)
            if key == "background":
                how = p.get("how", "constant")
                if how != "shirley" and region is None:
                    raise ValueError("type the Range over the background first")
                x2, v2, n2, _ = C.subtract_background(x, values, names, how, region)
                return self._emit(x2, v2, n2, "bg", "background",
                                  {"how": how, "region": list(region) if region else ""})
            if key == "shift":
                delta = float(p.get("delta", 0.0))
                label = "E − E_F (eV)" if p.get("relabel") else self.x_label
                return self._emit(C.shift_x(x, delta), values, names, "shifted",
                                  "shift_x", {"subtracted": delta}, x_label=label)
            if key == "poisson":
                v2, n2, added = C.with_poisson_sigma(values, names)
                if not added:
                    raise ValueError("every channel either has a σ already or is "
                                     "not raw counts")
                return self._emit(x, v2, n2, "σ", "poisson_sigma",
                                  {"added": ", ".join(added)})
            raise ValueError("unknown operation %r" % key)
        except ValueError as exc:
            T.info(self, "Operation", str(exc))
            return None

    def export_text(self):
        path = T.save_path(self, "Export curves", T.safe_stem(self.filename) + ".txt",
                           [("Text", "*.txt"), ("CSV", "*.csv")])
        if not path:
            return
        delimiter = "," if path.lower().endswith(".csv") else "\t"
        table = np.column_stack([self.x, self.values])
        header = delimiter.join([self.x_label] + list(self.names))
        np.savetxt(path, table, delimiter=delimiter, fmt="%.8g", header=header,
                   comments="", encoding="utf-8")
        self.say("Wrote %s" % path)

    # -- the panels ----------------------------------------------------------------------
    def open_fit(self):
        dialog = CurveFitDialog(self)
        self.children_windows.append(dialog)
        return dialog

    def open_spin_analysis(self):
        try:
            dialog = SpinAnalysisDialog(self)
        except ValueError as exc:
            T.info(self, "Spin analysis", str(exc))
            return None
        self.children_windows.append(dialog)
        return dialog

    def to_figure(self):
        from ui.figure import FigureWindow, Panel
        offset = self.offset()
        curves = []
        for order, index in enumerate(self.shown_channels()):
            s = C.sigma_of(self.names, index)
            curves.append((self.x, self.values[:, index] + order * offset,
                           self.names[index], colour(index),
                           self.values[:, s] if s is not None else None))
        panel = Panel.curves(curves, self.x_label, self.value_label, title=self.filename)
        return FigureWindow(self, [panel], "Figure — %s" % self.filename, app=self.app)


# ==========================================================================
# Curve fit
# ==========================================================================
class CurveFitDialog(tk.Toplevel):
    """Fit one channel: peaks (Lorentzian / Gaussian / Voigt) on a
    background, optionally cut by a Fermi edge; or the Fermi edge itself.

    Peaks are typed as one per line ``shape centre FWHM [hold]`` (e.g.
    ``lorentzian 0.12 0.05``); 'Find peaks' and 'Add by clicking' fill that
    table in.
    """

    def __init__(self, viewer):
        from tools.peaks import BACKGROUNDS
        tk.Toplevel.__init__(self, viewer)
        self.viewer = viewer
        self.title("Curve fit — %s" % viewer.filename)
        self.geometry("980x860")
        self.last = None
        self.edge_found = False
        self.channel_names = [viewer.names[i] for i in C.data_channels(viewer.names)]
        self.channel_index = C.data_channels(viewer.names)
        lo, hi = float(np.nanmin(viewer.x)), float(np.nanmax(viewer.x))
        region = viewer.x_range()
        if region is not None:
            lo, hi = region
        self.form = T.Form(self, [
            Field("channel", "Channel", "choice", self.channel_names[0], self.channel_names),
            Field("model", "Model", "choice", "peaks on a background",
                  ["peaks on a background", "Fermi edge"]),
            Field("lo", "Fit window from", "float", lo),
            Field("hi", "to", "float", hi),
            Field("weighting", "Weighting", "choice", "auto",
                  ["auto", "sigma", "poisson", "none"],
                  help="auto = the channel's own σ if it has one, √N for counts, else uniform"),
            Field("background", "Background (peaks)", "choice", "linear", list(BACKGROUNDS)),
            Field("resolution", "Resolution FWHM (peaks)", "float", 0.0),
            Field("fermi_cut", "× Fermi edge cut-off (peaks)", "bool", False),
            Field("cut_ef", "  cut-off E_F", "float", 0.0),
            Field("cut_t", "  cut-off T (K)", "float", 30.0),
            Field("edge_t", "Temperature (Fermi edge)", "float", 30.0),
            Field("hold_t", "Hold T (Fermi edge)", "bool", True),
        ], on_change=self.draw_data, columns=2)
        self.form.pack(fill="x", padx=6, pady=4)
        peaks = ttk.LabelFrame(self, text="Peaks: shape centre FWHM [hold], one per line")
        peaks.pack(fill="x", padx=6)
        self.peak_text = tk.Text(peaks, height=4, width=60)
        self.peak_text.pack(side="left", fill="x", expand=True, padx=3)
        side = ttk.Frame(peaks)
        side.pack(side="left")
        self.pick_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Add by clicking the plot",
                        variable=self.pick_var).pack(fill="x")
        ttk.Button(side, text="Find peaks", command=self.find_peaks).pack(fill="x")
        ttk.Button(side, text="Clear", command=lambda: self.peak_text.delete("1.0", "end")).pack(fill="x")
        self.plot = T.PlotFrame(self, figsize=(8, 4.2))
        self.plot.pack(fill="both", expand=True, padx=6)
        grid = self.plot.figure.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.05)
        self.ax = self.plot.figure.add_subplot(grid[0])
        self.ax_res = self.plot.figure.add_subplot(grid[1], sharex=self.ax)
        self.plot.canvas.mpl_connect("button_press_event", self._clicked)
        self.report = tk.Text(self, height=9, width=100, font=("Courier", 9))
        self.report.pack(fill="x", padx=6)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=4)
        for text, command in (("Fit", self.run_fit), ("Fit to list", self.export),
                              ("Copy results", self.copy_results),
                              ("List a copy with E_F = 0", self.shift_by_ef)):
            ttk.Button(bar, text=text, command=command).pack(side="left", padx=2)
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right")
        self.draw_data()

    # -- inputs ---------------------------------------------------------------------
    def settings(self):
        return self.form.values()

    def channel(self):
        name = self.form.raw("channel")
        return self.channel_index[self.channel_names.index(name)]

    def weighting(self):
        w = self.form.raw("weighting")
        index = self.channel()
        if w != "auto":
            return w
        if C.sigma_of(self.viewer.names, index) is not None:
            return "sigma"
        if C.looks_like_counts(self.viewer.values[:, index]):
            return "poisson"
        return "none"

    def fit_data(self):
        v = self.settings()
        lo, hi = sorted((v["lo"], v["hi"]))
        x = self.viewer.x
        index = self.channel()
        y = self.viewer.values[:, index]
        keep = (x >= lo) & (x <= hi) & np.isfinite(y)
        s = C.sigma_of(self.viewer.names, index)
        sigma = self.viewer.values[:, s][keep] if s is not None else None
        return x[keep], y[keep], sigma

    def peaks(self):
        from tools.peaks import PEAK_SHAPES
        out = []
        for n, line in enumerate(self.peak_text.get("1.0", "end").splitlines()):
            parts = line.replace(",", " ").split()
            if not parts:
                continue
            shape = parts[0].lower() if parts[0].lower() in PEAK_SHAPES else "lorentzian"
            numbers = parts[1:] if parts[0].lower() in PEAK_SHAPES else parts
            try:
                centre, width = float(numbers[0]), abs(float(numbers[1]))
            except (IndexError, ValueError):
                raise ValueError("peak line %d: need 'shape centre FWHM'" % (n + 1))
            hold = len(numbers) > 2 and numbers[2].lower() in ("hold", "1", "true", "yes")
            out.append({"shape": shape, "centre": centre, "width": width, "hold": hold})
        return out

    def add_peak(self, centre, width=None, shape="lorentzian"):
        from tools.peaks import suggest_seed
        x, y, _ = self.fit_data()
        if width is None and x.size > 3:
            try:
                _, width = suggest_seed(x, y, centre)
            except Exception:                                   # noqa: BLE001
                width = None
        existing = self.peak_text.get("1.0", "end").strip()
        self.peak_text.insert("end", ("\n" if existing else "") + "%s %.6g %.6g" % (
            shape, centre, width or 0.1))

    def _clicked(self, event):
        if not self.pick_var.get() or event.inaxes is not self.ax or event.xdata is None:
            return
        if self.plot.navigating():
            return
        self.add_peak(float(event.xdata))

    def find_peaks(self, max_peaks=6):
        from scipy.signal import find_peaks
        x, y, _ = self.fit_data()
        if x.size < 5:
            return 0
        smooth = np.convolve(y, np.ones(3) / 3.0, mode="same")
        found, props = find_peaks(smooth, prominence=0.05 * float(np.ptp(smooth)))
        order = np.argsort(props["prominences"])[::-1][:max_peaks]
        for index in sorted(found[order]):
            self.add_peak(float(x[index]))
        return len(order)

    # -- fitting --------------------------------------------------------------------
    def run_fit(self):
        from tools.fermi import fit_fermi_edge, initial_guess, steepest_drop
        from tools.kzmap import _is_an_edge
        from tools.peaks import Band, FitSettings, fit_line
        try:
            v = self.settings()
            x, y, sigma = self.fit_data()
            weighting = self.weighting()
            if v["model"] == "Fermi edge":
                fixed = ("temperature",) if v["hold_t"] else ()
                start = initial_guess(x, y, temperature=v["edge_t"])
                start["ef"] = steepest_drop(x, y)
                fit = fit_fermi_edge(x, y, start=start, temperature=v["edge_t"],
                                     fixed=fixed,
                                     weighting="uniform" if weighting == "none" else "poisson")
                step = float(np.median(np.diff(np.sort(x)))) if x.size > 1 else 0.0
                self.edge_found = _is_an_edge(fit, float(x.min()), float(x.max()), 2.0 * step)
                self.last = ("fermi", fit, x, y, sigma)
            else:
                peaks = self.peaks()
                if not peaks:
                    raise ValueError("add at least one peak (type it, click the plot "
                                     "with 'Add by clicking' on, or Find peaks)")
                settings = FitSettings(background=v["background"], resolution=v["resolution"],
                                       weighting="none" if weighting == "none" else "poisson",
                                       fermi=v["fermi_cut"], ef=v["cut_ef"],
                                       temperature=v["cut_t"])
                bands, guesses = [], {}
                for i, peak in enumerate(peaks):
                    name = "P%d" % (i + 1)
                    bands.append(Band(name, shape=peak["shape"], fix_width=peak["hold"]))
                    guesses[name] = (peak["centre"], peak["width"],
                                     float(np.interp(peak["centre"], x, y)))
                fit = fit_line(x, y, bands, settings, guesses=guesses,
                               sigma=sigma if weighting == "sigma" else None)
                self.last = ("peaks", fit, x, y, sigma)
                self.peak_text.delete("1.0", "end")
                self.peak_text.insert("1.0", "\n".join(
                    "%s %.6g %.6g%s" % (p["shape"], fit.centres[i], fit.widths[i],
                                        " hold" if p["hold"] else "")
                    for i, p in enumerate(peaks)))
        except (ValueError, RuntimeError, np.linalg.LinAlgError, T.FormError) as exc:
            self.last = None
            self._report("Could not fit: %s" % exc)
            return None
        self._show_result()
        return self.last[1]

    def rows(self):
        if self.last is None:
            return []
        kind, fit = self.last[0], self.last[1]
        if kind == "fermi":
            from tools.fermi import PARAMETERS
            return [(n, fit.values[n], fit.errors.get(n, float("nan"))) for n in PARAMETERS]
        rows = []
        for i, name in enumerate(fit.names):
            rows += [("%s centre" % name, fit.centres[i], fit.centre_errors[i]),
                     ("%s FWHM" % name, fit.widths[i], fit.width_errors[i]),
                     ("%s height" % name, fit.heights[i], fit.height_errors[i]),
                     ("%s area" % name, fit.areas[i], float("nan"))]
        return rows

    def _curves(self):
        kind, fit, x = self.last[0], self.last[1], self.last[2]
        if kind == "fermi":
            return fit.model(x), []
        return fit.model, [(name, fit.component(i)) for i, name in enumerate(fit.names)]

    def draw_data(self):
        try:
            x, y, sigma = self.fit_data()
        except T.FormError:
            return
        self.ax.clear()
        self.ax_res.clear()
        c = colour(self.channel())
        self.ax.plot(x, y, "o", ms=3, color=c, label="data")
        if sigma is not None:
            self.ax.errorbar(x, y, yerr=sigma, fmt="none", ecolor=c, elinewidth=0.8)
        self.ax.set_ylabel(self.viewer.value_label, fontsize=8)
        self.ax_res.set_xlabel(self.viewer.x_label, fontsize=8)
        self.ax_res.set_ylabel("residual/σ", fontsize=8)
        self.plot.draw()

    def _show_result(self):
        kind, fit, x, y, sigma = self.last
        model, components = self._curves()
        self.draw_data()
        if kind == "fermi":
            order = np.argsort(x)
            self.ax.plot(x[order], np.asarray(model)[order], color="#c23b22", lw=2, label="fit")
            data_x, data_y, model_y = x, y, np.asarray(model)
        else:
            self.ax.plot(fit.x, model, color="#c23b22", lw=2, label="fit")
            for i, (name, comp) in enumerate(components):
                self.ax.plot(fit.x, comp, "--", color=colour(i + 2), lw=1, label=name)
            self.ax.plot(fit.x, fit.background, ":", color="#888", lw=1, label="background")
            data_x, data_y, model_y = fit.x, fit.y, np.asarray(model)
        w = self.weighting()
        if sigma is not None and w == "sigma":
            scale = np.interp(data_x, x, sigma)
        elif w == "none":
            scale = np.full_like(data_y, float(np.std(data_y - model_y)) or 1.0)
        else:
            scale = np.sqrt(np.maximum(np.abs(data_y), 1.0))
        self.ax_res.plot(data_x, (data_y - model_y) / scale, "o", ms=2.5, color="#444")
        self.ax_res.axhline(0, color="#c23b22")
        self.ax.legend(fontsize=7)
        self.plot.draw()
        lines = ["%-18s %14s %12s" % ("parameter", "value", "error")]
        for name, value, error in self.rows():
            lines.append("%-18s %14.6g %12s" % (name, value,
                                                "—" if not np.isfinite(error) else "%.2g" % error))
        if kind == "fermi":
            lines.append(fit.summary())
            if not self.edge_found:
                lines.append("NOTE: no edge found in this window (E_F pinned to the "
                             "end or no significant step) -- not usable to shift the data.")
        else:
            lines.append("reduced chi2 = %.4g, R2 = %.5f%s" % (
                fit.chi2, fit.r_squared, "" if fit.success else "; solver: %s" % fit.message))
            lines.append("Errors are scaled by sqrt(chi2).")
        self._report("\n".join(lines))

    def _report(self, text):
        self.report.delete("1.0", "end")
        self.report.insert("1.0", text)

    def fit_parameters(self):
        v = self.settings()
        kind = self.last[0]
        out = {"model": kind, "channel": v["channel"], "window": [v["lo"], v["hi"]],
               "weighting": self.weighting()}
        if kind == "fermi":
            out["reduced_chi2"] = float(self.last[1].reduced_chi2)
        else:
            out.update(background=v["background"], resolution=v["resolution"],
                       reduced_chi2=float(self.last[1].chi2))
        for name, value, error in self.rows():
            key = name.replace(" ", "_")
            out[key] = float(value)
            if np.isfinite(error):
                out[key + "_err"] = float(error)
        return out

    def export(self):
        if self.last is None:
            T.info(self, "Curve fit", "Fit first.")
            return None
        kind, fit, x, y, sigma = self.last
        model, components = self._curves()
        if kind == "fermi":
            order = np.argsort(x)
            xs, ys, model = x[order], y[order], np.asarray(model)[order]
            sigma = sigma[order] if sigma is not None else None
        else:
            xs, ys = fit.x, fit.y
            if sigma is not None:
                sigma = np.interp(xs, x, sigma)
        channel = self.form.raw("channel")
        columns = [(channel, ys)]
        if sigma is not None:
            columns.append((C.sigma_name(channel), sigma))
        columns += [("fit", model), ("residual", ys - model)]
        columns += [(name, comp) for name, comp in components]
        if kind == "peaks":
            columns.append(("background", fit.background))
        xs, values, names = C.table(xs, columns)
        viewer = self.viewer
        name = T.unique_name("%s [fit]" % viewer.filename, viewer.existing_names())
        data = curve_dataset("mdc" if viewer.kind == "mdc" else "edc", xs, values, names,
                             x_label=viewer.x_label, value_label=viewer.value_label,
                             name=name, step="curve_fit", parameters=self.fit_parameters(),
                             source_info=viewer.info, prefix="fit", source=viewer.filename,
                             source_path=getattr(viewer.data, "path", ""))
        if viewer.app is not None:
            viewer.app.add_dataset(data)
        self.report.insert("end", "\nAdded “%s” to the list." % name)
        return data

    def shift_by_ef(self):
        if self.last is None or self.last[0] != "fermi" or not self.edge_found:
            T.info(self, "Curve fit", "Fit a Fermi edge that was found first.")
            return None
        ef = float(self.last[1].values["ef"])
        return self.viewer.apply_operation("shift", delta=ef, relabel=True)

    def copy_results(self):
        lines = ["parameter\tvalue\terror"]
        lines += ["%s\t%.8g\t%.3g" % r for r in self.rows()]
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))


# ==========================================================================
# Spin analysis
# ==========================================================================
class SpinAnalysisDialog(tk.Toplevel):
    """Polarisation, instrumental asymmetry and spin-resolved spectra from
    the viewer's spin channels. Each channel's axis and sign ("counts as")
    is read from its label and can be overridden."""

    def __init__(self, viewer):
        tk.Toplevel.__init__(self, viewer)
        self.viewer = viewer
        self.title("Spin analysis — %s" % viewer.filename)
        self.geometry("1000x920")
        self.last = None
        counts = viewer.values[:, C.data_channels(viewer.names)]
        if counts.shape[1] < 2:
            raise ValueError("a spin analysis needs at least two channels")
        self.counts = counts
        self.channels = S.channels_from_info(viewer.info, counts.shape[1])

        table = ttk.LabelFrame(self, text="Channels (as recorded; axis and sign can be overridden)")
        table.pack(fill="x", padx=6, pady=4)
        self.axis_vars, self.sign_vars = [], []
        for row, ch in enumerate(self.channels):
            ttk.Label(table, text="C%d %s" % (ch.index, ch.label)).grid(row=row, column=0, sticky="w", padx=3)
            ttk.Label(table, text=ch.setting).grid(row=row, column=1, padx=3)
            ttk.Label(table, text=ch.magnetisation).grid(row=row, column=2, padx=3)
            av = tk.StringVar(value=ch.axis or "-")
            c1 = ttk.Combobox(table, textvariable=av, width=4, state="readonly",
                              values=["-", "X", "Y", "Z"])
            c1.grid(row=row, column=3)
            sv = tk.StringVar(value="+ (up)" if ch.sign >= 0 else "- (down)")
            c2 = ttk.Combobox(table, textvariable=sv, width=9, state="readonly",
                              values=["+ (up)", "- (down)"])
            c2.grid(row=row, column=4)
            for c in (c1, c2):
                c.bind("<<ComboboxSelected>>", lambda e: self.channels_edited())
            self.axis_vars.append(av)
            self.sign_vars.append(sv)
        lo, hi = float(viewer.x.min()), float(viewer.x.max())
        self.form = T.Form(self, [
            Field("axis", "Axis", "choice", "", [""]),
            Field("method", "Method", "choice", "", [""]),
            Field("sherman", "S_eff (Sherman)", "float", S.DEFAULT_SHERMAN),
            Field("bin", "Bin (sum n points)", "int", 1),
            Field("zero", "Zero reference: P = 0 between", "bool", False),
            Field("zlo", "  from", "float", hi - 0.1 * (hi - lo)),
            Field("zhi", "  to", "float", hi),
        ], on_change=self.refresh, columns=2)
        self.form.pack(fill="x", padx=6)
        self.form.widgets["axis"].bind("<<ComboboxSelected>>", lambda e: self.axis_changed(), add="+")
        self.plot = T.PlotFrame(self, figsize=(8, 5.5))
        self.plot.pack(fill="both", expand=True, padx=6)
        grid = self.plot.figure.add_gridspec(3, 1, hspace=0.08)
        self.ax_raw = self.plot.figure.add_subplot(grid[0])
        self.ax_p = self.plot.figure.add_subplot(grid[1], sharex=self.ax_raw)
        self.ax_ud = self.plot.figure.add_subplot(grid[2], sharex=self.ax_raw)
        self.report = tk.Text(self, height=7, width=100, font=("Courier", 9))
        self.report.pack(fill="x", padx=6)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=4)
        for text, command in (("Compute", self.refresh), ("P to list", self.export_polarisation),
                              ("I↑ / I↓ to list", self.export_spin_resolved),
                              ("As a figure", self.to_figure)):
            ttk.Button(bar, text=text, command=command).pack(side="left", padx=2)
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right")
        self.channels_edited()

    def current_channels(self):
        out = []
        for ch, av, sv in zip(self.channels, self.axis_vars, self.sign_vars):
            axis = av.get()
            out.append(S.SpinChannel(ch.index, ch.label, ch.setting, ch.magnetisation,
                                     "" if axis == "-" else axis,
                                     1 if sv.get().startswith("+") else -1))
        return out

    def channels_edited(self):
        axes = S.axes_available(self.current_channels())
        keep = self.form.raw("axis")
        self.form.set_choices("axis", axes or [""], keep if keep in axes else None)
        self.axis_changed()

    def axis_changed(self):
        axis = self.form.raw("axis")
        options, keys = [], []
        channels = self.current_channels()
        if axis:
            report = S.design_report(channels, axis)
            if report["n_plus"] == report["n_minus"] and report["n_plus"]:
                options.append("Cross ratio (all channels)")
                keys.append("cross")
            for setting, _p, _m in S.pairs(channels, axis):
                options.append("One pair: <%s>" % setting)
                keys.append("pair:%s" % setting)
        self._method_keys = dict(zip(options, keys))
        self.form.set_choices("method", options or [""])
        self.refresh()

    def settings(self):
        v = self.form.values()
        return {"axis": v["axis"], "method": self._method_keys.get(v["method"]),
                "sherman": v["sherman"], "bin_factor": max(1, int(v["bin"])),
                "zero_region": (v["zlo"], v["zhi"]) if v["zero"] else None}

    def compute(self):
        s = self.settings()
        if not s["axis"] or not s["method"]:
            raise ValueError("no axis has channels on both sides -- set the channels' "
                             "axis and sign in the table")
        return S.analyse(self.viewer.x, self.counts, self.current_channels(), s["axis"],
                         sherman=s["sherman"], method=s["method"],
                         zero_region=s["zero_region"], bin_factor=s["bin_factor"])

    def refresh(self):
        if not hasattr(self, "report"):
            return
        for ax in (self.ax_raw, self.ax_p, self.ax_ud):
            ax.clear()
        x = self.viewer.x
        for ch in self.channels:
            self.ax_raw.plot(x, self.counts[:, ch.index], color=colour(ch.index), lw=1.2,
                             label="C%d %s" % (ch.index, ch.label))
        self.ax_raw.set_ylabel("Counts", fontsize=8)
        self.ax_raw.legend(fontsize=6)
        try:
            r = self.compute()
        except (ValueError, T.FormError) as exc:
            self.last = None
            self.report.delete("1.0", "end")
            self.report.insert("1.0", str(exc))
            self.plot.draw()
            return
        self.last = r
        self.ax_p.axhline(0, color="#999")
        self.ax_p.errorbar(r.x, r.polarisation, yerr=r.sigma_polarisation, fmt="o-",
                           ms=3, color="#1f5aa6", label="P_%s" % r.axis)
        for i, (setting, (a, _sa)) in enumerate(r.pair_asymmetries.items()):
            self.ax_p.plot(r.x, a / r.sherman, "--", color=colour(i + 3), lw=1,
                           label="<%s> alone" % setting)
        self.ax_p.set_ylabel("P_%s" % r.axis, fontsize=8)
        self.ax_p.legend(fontsize=6)
        for y, s, c, name in ((r.up, r.sigma_up, "#c23b22", "I↑ (+%s)" % r.axis),
                              (r.down, r.sigma_down, "#1f5aa6", "I↓ (−%s)" % r.axis)):
            self.ax_ud.errorbar(r.x, y, yerr=s, fmt="-", color=c, label=name)
        self.ax_ud.set_ylabel("Spin-resolved", fontsize=8)
        self.ax_ud.set_xlabel(self.viewer.x_label, fontsize=8)
        self.ax_ud.legend(fontsize=6)
        self.plot.draw()
        self.report.delete("1.0", "end")
        self.report.insert("1.0", r.summary())

    def _parameters(self, r):
        s = self.settings()
        out = {"axis": r.axis, "method": r.method, "sherman": r.sherman,
               "bin_factor": s["bin_factor"],
               "channels_used": ", ".join("C%d" % i for i in r.channels_used),
               "cancels_transmission": bool(r.design.get("transmission")),
               "cancels_reflectivity": bool(r.design.get("reflectivity"))}
        if s["zero_region"] is not None:
            out["zero_region"] = list(s["zero_region"])
            out["zero_asymmetry"] = float(np.tanh(r.zero_offset / 2))
        if r.instrumental is not None:
            out["instrumental_asymmetry"] = r.instrumental[0]
            out["instrumental_asymmetry_err"] = r.instrumental[1]
        return out

    def _emit(self, columns, suffix, value_label, step):
        viewer, r = self.viewer, self.last
        x, values, names = C.table(r.x, columns)
        name = T.unique_name("%s [%s]" % (viewer.filename, suffix), viewer.existing_names())
        data = curve_dataset("edc", x, values, names, x_label=viewer.x_label,
                             value_label=value_label, name=name, step=step,
                             parameters=self._parameters(r), source_info=viewer.info,
                             prefix="spin", source=viewer.filename,
                             source_path=getattr(viewer.data, "path", ""))
        if viewer.app is not None:
            viewer.app.add_dataset(data)
        self.report.insert("end", "\nAdded “%s” to the list." % name)
        return data

    def export_polarisation(self):
        r = self.last
        if r is None:
            return None
        label = "P_%s" % r.axis
        return self._emit([(label, r.polarisation), (C.sigma_name(label), r.sigma_polarisation)],
                          "P%s" % r.axis, "Spin polarisation %s" % label, "spin_polarisation")

    def export_spin_resolved(self):
        r = self.last
        if r is None:
            return None
        up, down = "I↑ (+%s)" % r.axis, "I↓ (−%s)" % r.axis
        return self._emit([(up, r.up), (down, r.down), (C.sigma_name(up), r.sigma_up),
                           (C.sigma_name(down), r.sigma_down)],
                          "spin %s" % r.axis,
                          "Counts (sum of %d channels)" % len(r.channels_used),
                          "spin_resolved")

    def to_figure(self):
        from ui.figure import FigureWindow, Panel
        r = self.last
        if r is None:
            return None
        xl = self.viewer.x_label
        panels = [Panel.curves([(r.x, r.polarisation, "P_%s" % r.axis, "#1f5aa6",
                                 r.sigma_polarisation)], xl, "P_%s" % r.axis,
                               title="polarisation"),
                  Panel.curves([(r.x, r.up, "I↑ (+%s)" % r.axis, "#c23b22", r.sigma_up),
                                (r.x, r.down, "I↓ (−%s)" % r.axis, "#1f5aa6",
                                 r.sigma_down)], xl, "Counts", title="spin-resolved")]
        return FigureWindow(self, panels, "Spin analysis figure", app=self.viewer.app)
