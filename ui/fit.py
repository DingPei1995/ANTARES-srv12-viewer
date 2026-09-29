"""
ui/fit.py
=========
MDC / EDC fitting of a (k-converted) cut, and what the fitted band says.

1. **Bands and seeds.** Each band needs one or more *seeds*: at a line
   position, the peak's centre, FWHM and height. Seeds are typed in the
   band's seed box (``position centre FWHM [height]`` per line), or placed
   by clicking the line plot with "Seed by clicking" on (the width and
   height are then estimated from the data).
2. **Try this line** fits the line under the position; **Fit the series**
   fits every line (:func:`tools.peaks.fit_series`), with a progress bar.
3. **Dispersion...** -- v_F (straight line) or m* (parabola) with the
   window scan that shows how stable the number is; **Self-energy...** --
   Re / Im Sigma with the Kramers-Kronig check.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk

from tools import peaks as PK
from tools import dispersion as DISP
from ui import tkbase as T
from ui.tkbase import Field
from ui.data import MemoryData

BAND_COLOURS = ("#e8413c", "#2d9bf0", "#20a464", "#f5a623", "#a855c8",
                "#00b2b2", "#d64ea0", "#8a8f00")


def fermi_defaults(data):
    """``(ef, temperature, resolution)`` read off the dataset if it says."""
    info = getattr(getattr(data, "scan", None), "info", {}) or {}
    labels = getattr(getattr(data, "scan", None), "labels", {}) or {}
    ef = temperature = resolution = None
    for key, value in info.items():
        tail = str(key).rsplit(".", 1)[-1].lower()
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if tail == "ef_subtracted":
            ef = 0.0
        elif tail == "ef" and ef is None:
            ef = number
        elif tail == "temperature" and temperature is None:
            temperature = number
        elif tail == "resolution" and resolution is None:
            resolution = number
    y_label = str(labels.get("y", ""))
    if ef is None and ("E_F" in y_label or "E-Ef" in y_label):
        ef = 0.0
    return (0.0 if ef is None else ef, 30.0 if temperature is None else temperature,
            0.0 if resolution is None else resolution)


class FitPanel(tk.Toplevel):
    def __init__(self, app, data, label, master=None, colormap="gray", flip=False):
        tk.Toplevel.__init__(self, master or app.root)
        self.app, self.data, self.label = app, data, label
        scan = data.scan
        self.values = np.asarray(scan.value, dtype=float)
        self.axes = [np.asarray(scan.x, dtype=float), np.asarray(scan.y, dtype=float)]
        self.labels = [scan.labels.get("x", "k"), scan.labels.get("y", "E")]
        self.colormap, self.flip = colormap, flip
        self.bands = []
        self.series = None
        self.title("MDC / EDC fit — %s" % label)
        self.geometry("1350x930")
        ef, temperature, resolution = fermi_defaults(data)

        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=4, pady=4)
        s0, s1 = self._short(0), self._short(1)
        self.form = T.Form(left, [
            Field("direction", "Lines", "choice", "MDCs (along %s)" % s0,
                  ["MDCs (along %s)" % s0, "EDCs (along %s)" % s1]),
            Field("shape", "Peak shape", "choice", "lorentzian", list(PK.PEAK_SHAPES)),
            Field("background", "Background", "choice", "linear", list(PK.BACKGROUNDS)),
            Field("resolution", "Resolution FWHM (eV)", "float", resolution),
            Field("combine", "Combine n lines", "int", 1),
            Field("step", "Fit every nth line", "int", 1),
            Field("lo", "Lines from (empty = all)", "optfloat", None),
            Field("hi", "Lines to", "optfloat", None),
            Field("refine", "Second pass from the first result", "bool", True),
            Field("fermi", "Fermi cut-off (EDCs only)", "bool", False),
            Field("ef", "  E_F (eV)", "float", ef),
            Field("temp", "  Temperature (K)", "float", temperature),
        ], on_change=self._settings_changed)
        self.form.pack(fill="x")
        self.position = T.SliceControl(left, "Line position", on_change=self.refresh_line)
        self.position.pack(fill="x", pady=4)

        bands = ttk.LabelFrame(left, text="Bands")
        bands.pack(fill="x", pady=4)
        self.band_list = tk.Listbox(bands, height=5, exportselection=False)
        self.band_list.pack(fill="x")
        self.band_list.bind("<<ListboxSelect>>", lambda e: self._show_seeds())
        row = ttk.Frame(bands)
        row.pack(fill="x")
        for text, command in (("Add", self.add_band), ("Remove", self.remove_band)):
            ttk.Button(row, text=text, command=command).pack(side="left")
        ttk.Label(row, text="same width as").pack(side="left", padx=(6, 0))
        self.share_var = tk.StringVar()
        e = ttk.Entry(row, textvariable=self.share_var, width=10)
        e.pack(side="left")
        e.bind("<Return>", lambda ev: self._store_seeds())
        ttk.Label(bands, text="Seeds: position centre FWHM [height], one per line").pack(anchor="w")
        self.seed_text = tk.Text(bands, height=6, width=44)
        self.seed_text.pack(fill="x")
        self.seed_text.bind("<FocusOut>", lambda e: self._store_seeds())
        row2 = ttk.Frame(bands)
        row2.pack(fill="x")
        ttk.Button(row2, text="Apply seeds", command=self._store_seeds).pack(side="left")
        self.seed_click = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Seed by clicking the line plot",
                        variable=self.seed_click).pack(side="left", padx=4)

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=4)
        for text, command in (("Try this line", self.try_line),
                              ("Fit the series", self.run_fit),
                              ("Dispersion...", self.open_dispersion),
                              ("Save table...", self.save_table)):
            ttk.Button(buttons, text=text, command=command).pack(fill="x", pady=1)
        self.note = ttk.Label(left, text="", wraplength=380, justify="left", foreground="#225")
        self.note.pack(fill="x")

        self.plot = T.PlotFrame(self, figsize=(8.5, 8))
        self.plot.pack(side="left", fill="both", expand=True)
        grid = self.plot.figure.add_gridspec(3, 1, height_ratios=(2.3, 2, 0.8), hspace=0.35)
        self.ax_img = self.plot.figure.add_subplot(grid[0])
        self.ax_line = self.plot.figure.add_subplot(grid[1])
        self.ax_res = self.plot.figure.add_subplot(grid[2], sharex=self.ax_line)
        self.plot.canvas.mpl_connect("button_press_event", self._clicked)
        self._direction = None
        self.add_band()
        self._settings_changed()

    def _short(self, dim):
        return self.labels[dim].split(" (")[0]

    # -- settings ----------------------------------------------------------------------
    def mdc(self):
        return self.form.raw("direction").startswith("MDC")

    def _settings_changed(self):
        mdc = self.mdc()
        if mdc != self._direction:
            if self._direction is not None:
                for band in self.bands:
                    band.seeds = []
                self.series = None
                self._show_seeds()
            self._direction = mdc
            across = self.axes[1] if mdc else self.axes[0]
            self.position.configure_axis(across)
        try:
            shape = self.form.value("shape")
            for band in self.bands:
                band.shape = shape
        except T.FormError:
            pass
        self.refresh_line()

    def settings(self):
        v = self.form.values()
        return PK.FitSettings(background=v["background"], resolution=v["resolution"],
                              fermi=bool(v["fermi"] and not self.mdc()), ef=v["ef"],
                              temperature=v["temp"])

    # -- bands ----------------------------------------------------------------------------
    def add_band(self):
        taken = {b.name for b in self.bands}
        n = len(self.bands) + 1
        while "Band %d" % n in taken:
            n += 1
        try:
            shape = self.form.value("shape")
        except (T.FormError, AttributeError):
            shape = "lorentzian"
        self.bands.append(PK.Band("Band %d" % n, [], shape=shape, colour=len(self.bands)))
        self._fill_bands(len(self.bands) - 1)

    def remove_band(self):
        i = self.selected_index()
        if i is None or len(self.bands) <= 1:
            return
        name = self.bands[i].name
        for band in self.bands:
            if band.share_width == name:
                band.share_width = ""
        self.bands.pop(i)
        self._fill_bands(0)

    def _fill_bands(self, select=None):
        self.band_list.delete(0, "end")
        for band in self.bands:
            span = band.span()
            self.band_list.insert("end", "%s  (%d seeds%s%s)" % (
                band.name, len(band.seeds),
                "" if span is None else ", %.4g..%.4g" % span,
                ", width = %s" % band.share_width if band.share_width else ""))
            self.band_list.itemconfigure("end", foreground=BAND_COLOURS[band.colour % len(BAND_COLOURS)])
        if select is not None and self.bands:
            self.band_list.selection_clear(0, "end")
            self.band_list.selection_set(select)
        self._show_seeds()

    def selected_index(self):
        sel = self.band_list.curselection()
        return sel[0] if sel else (0 if self.bands else None)

    def selected_band(self):
        i = self.selected_index()
        return None if i is None else self.bands[i]

    def _show_seeds(self):
        band = self.selected_band()
        self.seed_text.delete("1.0", "end")
        if band is None:
            return
        self.share_var.set(band.share_width)
        self.seed_text.insert("1.0", "\n".join(
            "%.6g  %.6g  %.6g  %.6g" % (s.position, s.centre, s.width, s.height)
            for s in band.sorted_seeds()))
        self.refresh_line()

    def _store_seeds(self):
        band = self.selected_band()
        if band is None:
            return
        seeds = []
        for n, line in enumerate(self.seed_text.get("1.0", "end").splitlines()):
            if not line.strip():
                continue
            try:
                numbers = T.parse_floats(line)
                if len(numbers) < 3:
                    raise ValueError
            except ValueError:
                self.note.configure(text="seed line %d: need position centre FWHM [height]" % (n + 1))
                return
            seeds.append(PK.Seed(numbers[0], numbers[1], abs(numbers[2]),
                                 numbers[3] if len(numbers) > 3 else 0.0))
        band.seeds = seeds
        share = self.share_var.get().strip()
        band.share_width = share if share in {b.name for b in self.bands} and share != band.name else ""
        i = self.selected_index()
        self._fill_bands(i)

    # -- the current line -------------------------------------------------------------------
    def current_position(self):
        return self.position.position

    def current_line(self):
        mdc = self.mdc()
        along = self.axes[0] if mdc else self.axes[1]
        across = self.axes[1] if mdc else self.axes[0]
        index = T.nearest_index(across, self.current_position())
        try:
            combine = max(1, int(self.form.value("combine")))
        except T.FormError:
            combine = 1
        half = (combine - 1) // 2
        first, last = max(0, index - half), min(across.size, index + half + 1)
        block = self.values[:, first:last] if mdc else self.values[first:last, :]
        return along, np.nansum(block, axis=1 if mdc else 0)

    def draw_image(self):
        ax = self.ax_img
        ax.clear()
        lo, hi = T.auto_levels(self.values, 1, 99)
        ex, ey = T.edges_of(self.axes[0]), T.edges_of(self.axes[1])
        ax.pcolormesh(ex, ey, self.values.T, cmap=T.get_cmap(self.colormap, self.flip),
                      vmin=lo, vmax=hi)
        ax.set_xlabel(self.labels[0], fontsize=8)
        ax.set_ylabel(self.labels[1], fontsize=8)
        ax.tick_params(labelsize=7)
        mdc = self.mdc()
        p = self.current_position()
        (ax.axhline if mdc else ax.axvline)(p, color="#1f6f8b", lw=1.5)
        for band in self.bands:
            colour = BAND_COLOURS[band.colour % len(BAND_COLOURS)]
            if band.seeds:
                c = [s.centre for s in band.seeds]
                q = [s.position for s in band.seeds]
                ax.plot(c if mdc else q, q if mdc else c, "+", color=colour, ms=10, mew=2)
        if self.series is not None:
            heights = np.asarray(self.series.heights, dtype=float)
            top = float(np.nanmax(heights)) if np.isfinite(heights).any() else 1.0
            for index, name in enumerate(self.series.names):
                band = next((b for b in self.bands if b.name == name), None)
                colour = BAND_COLOURS[(band.colour if band else index) % len(BAND_COLOURS)]
                centres = self.series.centres[:, index]
                good = np.isfinite(centres)
                sizes = 4 + 30 * np.clip(heights[good, index] / top, 0, 1)
                xs = centres[good] if mdc else self.series.positions[good]
                ys = self.series.positions[good] if mdc else centres[good]
                ax.scatter(xs, ys, s=sizes, color=colour, zorder=5)
        ax.set_xlim(ex[0], ex[-1])
        ax.set_ylim(ey[0], ey[-1])

    def refresh_line(self, fit=None):
        if not hasattr(self, "ax_line"):
            return
        self.draw_image()
        along, line = self.current_line()
        ax = self.ax_line
        ax.clear()
        self.ax_res.clear()
        ax.plot(along, line, "o", ms=2.5, color="#888", label="data")
        position = self.current_position()
        try:
            resolution = self.form.value("resolution")
        except T.FormError:
            resolution = 0.0
        for band in [b for b in self.bands if b.seeds]:
            centre, width, height = (float(v[0]) for v in band.guess([position]))
            colour = BAND_COLOURS[band.colour % len(BAND_COLOURS)]
            profile = height * PK.peak_profile(band.shape, along, centre, width, resolution)
            ax.plot(along, profile + float(np.nanmin(line)), "--", color=colour, lw=1,
                    label="%s (guess)" % band.name)
        if fit is not None:
            ax.plot(fit.x, fit.model, color="#bd4921", lw=2, label="fit")
            ax.plot(fit.x, fit.background, ":", color="#999", lw=1, label="background")
            for index, name in enumerate(fit.names):
                band = next((b for b in self.bands if b.name == name), None)
                colour = BAND_COLOURS[(band.colour if band else index) % len(BAND_COLOURS)]
                ax.plot(fit.x, fit.component(index), color=colour, lw=1, label=name)
            self.ax_res.plot(fit.x, fit.residual, "o", ms=2, color="#444")
            self.ax_res.axhline(0, color="#bd4921")
        ax.set_xlabel(self.labels[0] if self.mdc() else self.labels[1], fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
        self.ax_res.tick_params(labelsize=7)
        self.plot.draw()

    def _clicked(self, event):
        if (not self.seed_click.get() or event.inaxes is not self.ax_line
                or event.xdata is None or self.plot.navigating()):
            return
        band = self.selected_band()
        if band is None:
            return
        along, line = self.current_line()
        height, width = PK.suggest_seed(along, line, float(event.xdata))
        position = self.current_position()
        band.seeds = [s for s in band.seeds if abs(s.position - position) > 1e-12]
        band.seeds.append(PK.Seed(position, float(event.xdata), width, height))
        self._fill_bands(self.selected_index())
        self.note.configure(text="%s: seed at %.5g, width %.4g. %d seed(s)." % (
            band.name, event.xdata, width, len(band.seeds)))

    def _tolerance(self):
        across = self.axes[1] if self.mdc() else self.axes[0]
        return abs(float(across[1] - across[0])) if across.size > 1 else 0.0

    def try_line(self):
        self._store_seeds()
        position = self.current_position()
        along, line = self.current_line()
        active = [b for b in self.bands if b.covers(position, self._tolerance())]
        if not active:
            T.info(self, "Fit", "No band is seeded at this line (a band exists between "
                   "its first and last seed).")
            return None
        try:
            fit = PK.fit_line(along, line, active, self.settings(), position=position)
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Fit", str(exc))
            return None
        self.refresh_line(fit)
        return fit

    def run_fit(self):
        self._store_seeds()
        bands = [b for b in self.bands if b.seeds]
        if not bands:
            T.info(self, "Fit", "Seed at least one band first.")
            return None
        try:
            v = self.form.values()
            settings = self.settings()
        except T.FormError as exc:
            T.warning(self, "Fit", str(exc))
            return None
        direction = "mdc" if self.mdc() else "edc"

        def work(report):
            def tick(done, total):
                report(done / float(max(total, 1)), "line %d of %d" % (done, total))
                return True
            return PK.fit_series(self.values, self.axes, bands, settings, direction=direction,
                                 step=max(1, v["step"]), combine=max(1, v["combine"]),
                                 bounds=(v["lo"], v["hi"]), refine=v["refine"], progress=tick)

        def done(series):
            series.x_label, series.y_label = self.labels
            self.series = series
            n = int(np.isfinite(series.centres).any(axis=1).sum())
            chi2 = getattr(series, "chi2", None)
            extra = (", median χ² = %.3g" % np.nanmedian(chi2)) if chi2 is not None else ""
            self.note.configure(text="Fitted %d of %d lines%s. The fitted positions are on "
                                "the picture, sized by amplitude. 'Dispersion...' for v_F and m*."
                                % (n, len(series.positions), extra))
            self.refresh_line()
        T.run_job(self, "Fitting lines", work, on_done=done)

    def open_dispersion(self):
        if self.series is None:
            T.info(self, "Dispersion", "Fit the series first.")
            return None
        return DispersionWindow(self, self.series, self.label, self.app)

    def save_table(self):
        if self.series is None:
            T.info(self, "Save table", "Fit the series first.")
            return None
        path = T.save_path(self, "Save the fitted parameters",
                           T.safe_stem(self.label) + "_fits.csv", [("CSV", "*.csv")])
        if not path:
            return None
        s = self.series
        chi2 = getattr(s, "chi2", np.full(len(s.positions), np.nan))
        header = ["position", "chi2"]
        for name in s.names:
            header += ["%s_centre" % name, "%s_centre_err" % name, "%s_fwhm" % name,
                       "%s_fwhm_err" % name, "%s_height" % name, "%s_area" % name]
        rows = []
        for i, position in enumerate(s.positions):
            row = [position, chi2[i]]
            for j in range(len(s.names)):
                row += [s.centres[i, j], s.centre_errors[i, j], s.widths[i, j],
                        s.width_errors[i, j], s.heights[i, j], s.areas[i, j]]
            rows.append(row)
        np.savetxt(path, np.array(rows, dtype=float), delimiter=",",
                   header=",".join(header), comments="", encoding="utf-8")
        self.note.configure(text="Saved %d rows to %s" % (len(rows), path))
        return path


class DispersionWindow(tk.Toplevel):
    """The fitted band: v_F or m*, and how it depends on the window."""

    def __init__(self, master, series, label, app=None):
        tk.Toplevel.__init__(self, master)
        self.series, self.label, self.app = series, label, app
        self.fit = None
        self.title("Dispersion — %s" % label)
        self.geometry("1150x720")
        self.form = T.Form(self, [
            Field("band", "Band", "choice", series.names[0], list(series.names)),
            Field("quantity", "Quantity", "choice", "v_F (straight line)",
                  ["v_F (straight line)", "m* (parabola)"]),
            Field("centre", "Centre (k of the extremum / E_F)", "float", 0.0),
        ], on_change=self.refresh, columns=3)
        self.form.pack(fill="x", padx=6, pady=4)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6)
        for text, command in (("Guess centre", self.guess_centre), ("Band to list", self.export_band),
                              ("Self-energy...", self.open_self_energy)):
            ttk.Button(bar, text=text, command=command).pack(side="left", padx=2)
        self.plot = T.PlotFrame(self, figsize=(10, 4.6))
        self.plot.pack(fill="both", expand=True, padx=6)
        self.ax_band = self.plot.figure.add_subplot(121)
        self.ax_scan = self.plot.figure.add_subplot(122)
        self.summary = tk.Text(self, height=8, width=110, font=("Courier", 9))
        self.summary.pack(fill="x", padx=6, pady=4)
        self.guess_centre()

    def band_index(self):
        return list(self.series.names).index(self.form.raw("band"))

    def velocity(self):
        return self.form.raw("quantity").startswith("v_F")

    def band_data(self):
        k, energy, error = self.series.band(self.band_index())
        on_k = self.series.error_axis() == "x"
        return k, energy, (error if on_k else None), (None if on_k else error)

    def guess_centre(self):
        k, energy, k_error, energy_error = self.band_data()
        if self.velocity():
            self.form.set("centre", 0.0)
        else:
            try:
                rough = DISP.fit_dispersion(k, energy, k_error=k_error,
                                            energy_error=energy_error, order=2)
                extremum = rough.band_extremum()[0]
                self.form.set("centre", float(extremum) if np.isfinite(extremum)
                              else float(np.mean(k)))
            except Exception:                                   # noqa: BLE001
                self.form.set("centre", float(np.mean(k)) if k.size else 0.0)
        self.refresh()

    def _say(self, text):
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", text)

    def refresh(self):
        from compat.numpy_compat import nan_to_num
        k, energy, k_error, energy_error = self.band_data()
        self.ax_band.clear()
        self.ax_scan.clear()
        if k.size < 3:
            self._say("not enough converged fits for this band")
            self.plot.draw()
            return
        error = k_error if k_error is not None else energy_error
        if k_error is not None:
            self.ax_band.errorbar(k, energy, xerr=nan_to_num(error), fmt="o", ms=3, color="#bd4921")
        else:
            self.ax_band.errorbar(k, energy, yerr=nan_to_num(error), fmt="o", ms=3, color="#bd4921")
        velocity = self.velocity()
        try:
            self.fit = DISP.fit_dispersion(k, energy, k_error=k_error, energy_error=energy_error,
                                           order=1 if velocity else 2)
            grid = np.linspace(k.min(), k.max(), 200)
            self.ax_band.plot(grid, self.fit.value(grid), color="#0d7377", lw=2)
        except Exception as exc:                                # noqa: BLE001
            self._say(str(exc))
            self.plot.draw()
            return
        self.ax_band.set_xlabel(self.series.x_label, fontsize=8)
        self.ax_band.set_ylabel(self.series.y_label, fontsize=8)
        try:
            centre = float(self.form.value("centre"))
        except T.FormError:
            centre = 0.0
        scan = DISP.window_scan(k, energy, k_error=k_error, energy_error=energy_error,
                                quantity="velocity" if velocity else "mass", centre=centre)
        good = np.isfinite(scan.values)
        if good.any():
            self.ax_scan.errorbar(scan.widths[good], scan.values[good],
                                  yerr=nan_to_num(scan.errors[good]), fmt="o-", ms=3,
                                  color="#0d7377")
        self.ax_scan.set_ylabel("dE/dk (eV·Å)" if velocity else "m*/mₑ", fontsize=8)
        self.ax_scan.set_xlabel("window half-width (%s)" % ("eV" if velocity else "Å⁻¹"),
                                fontsize=8)
        lines = [self.fit.summary()]
        plateau = scan.plateau_value()
        if plateau is None:
            lines.append("No stable plateau in the window scan: the number depends on "
                         "the window it is measured over.")
        else:
            value, error, widest = plateau
            first, last = scan.plateau(0.05)
            self.ax_scan.axvspan(scan.widths[first], scan.widths[last], color="#0d7377", alpha=0.15)
            extra = (" = %.3g m/s" % (value * DISP.VELOCITY_FACTOR)) if velocity else ""
            lines.append("Stable out to a window of %.4g%s: %.4g ± %.2g %s%s" % (
                widest, " eV" if velocity else " Å⁻¹", value, error,
                "eV·Å" if velocity else "", extra))
        lines.append("Weighted by the error on %s, the axis the fit measured."
                     % ("k" if k_error is not None else "E"))
        self._say("\n".join(lines))
        self.plot.draw()

    def export_band(self):
        index = self.band_index()
        k, energy, error = self.series.band(index)
        if k.size == 0 or self.app is None:
            return None
        widths = self.series.widths[:, index]
        widths = widths[np.isfinite(self.series.centres[:, index])]
        values = np.column_stack([energy, error, widths[:k.size]])
        data = MemoryData("cut", (k, np.array([0.0, 1.0, 2.0])), values,
                          {"x": self.series.x_label, "y": "energy / error / FWHM"},
                          source_label="%s_%s" % (self.label, self.series.names[index]),
                          parameters={"band": self.series.names[index],
                                      "direction": self.series.direction},
                          prefix="proc.band")
        self.app.add_dataset(data)
        return data

    def open_self_energy(self):
        if self.fit is None:
            return None
        if self.series.direction != "mdc":
            T.info(self, "Self-energy", "The self-energy is read from MDC fits: fit MDCs.")
            return None
        try:
            result = DISP.self_energy(self.series, self.fit, band=self.band_index())
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Self-energy", str(exc))
            return None
        from compat.numpy_compat import nan_to_num
        win = tk.Toplevel(self)
        win.title("Self-energy — %s" % self.label)
        plot = T.PlotFrame(win, figsize=(7, 6))
        plot.pack(fill="both", expand=True)
        ax1 = plot.figure.add_subplot(211)
        ax2 = plot.figure.add_subplot(212, sharex=ax1)
        ax1.errorbar(result.energy, result.real, yerr=nan_to_num(result.real_error), fmt="o",
                     ms=3, color="#bd4921")
        if result.kk_real is not None:
            ax1.plot(result.energy, result.kk_real, "--", color="#0d7377", lw=2)
        ax1.set_ylabel("Re Σ (eV)")
        ax2.errorbar(result.energy, result.imaginary, yerr=nan_to_num(result.imaginary_error),
                     fmt="o", ms=3, color="#0d7377")
        ax2.set_ylabel("Im Σ (eV)")
        ax2.set_xlabel("E − E_F (eV)")
        consistency = result.consistency()
        ttk.Label(win, wraplength=600, text=(
            "Dashed: the Kramers-Kronig transform of Im Σ, shifted to the mean of "
            "Re Σ. Mismatch %.2f (%s)." % (consistency, "consistent" if consistency < 0.3
                                                else "inconsistent"))).pack(fill="x")
        plot.draw()
        return win
