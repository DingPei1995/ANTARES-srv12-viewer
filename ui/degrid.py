"""
ui/degrid.py
============
The De-grid window: remove the detector's grid (MCP pattern or mesh) from a
map (using the map as its own reference) or from a cut (with a ``[grid]``
dataset from a map taken with the same settings, or by notching the cut's
own grid peaks). The algorithm is :mod:`tools.degrid`.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.ndimage import gaussian_filter

from tools import degrid as DG
from tools import process as P
from ui import tkbase as T
from ui.tkbase import Field
from ui.data import MemoryData

INTRO_MAP = ("Finds the detector's grid from the whole map -- it stays on the same "
             "pixels while the photoemission moves from slice to slice -- and divides "
             "it out of every slice, with each slice's own contrast and sub-pixel shift "
             "fitted. The defaults are the best found on real maps.")
INTRO_CUT = ("Removes the detector's grid from this cut. Best: a grid measured on a map "
             "taken with the same detector settings (de-grid that map first with 'Also "
             "list the grid pattern'). Otherwise the cut's own grid peaks are notched out.")
SETTINGS_KEYS = ("lens_mode", "pass_energy_eV", "MBS.lens_mode", "MBS.passenergy",
                 "scienta.Lens Mode", "scienta.Pass Energy", "mbs.Lens Mode",
                 "mbs.Pass Energy")


def settings_match(a, b):
    compared = False
    for key in SETTINGS_KEYS:
        va, vb = (a or {}).get(key), (b or {}).get(key)
        if va is None or vb is None:
            continue
        compared = True
        if str(va).strip() != str(vb).strip():
            return False
    return compared


def parse_box(text):
    try:
        a, e = str(text).split(",")
        a0, a1 = (int(v) for v in a.split(":"))
        e0, e1 = (int(v) for v in e.split(":"))
        return slice(a0, a1), slice(e0, e1)
    except (ValueError, AttributeError):
        return None


def box_text(box):
    return "%d:%d,%d:%d" % (box[0].start, box[0].stop, box[1].start, box[1].stop)


def grid_candidates(app, frame_shape, info):
    """``[(name, pattern, box, matches)]`` for every ``[grid]`` dataset in the
    list with this detector frame."""
    out = []
    if app is None:
        return out
    for label, key in app.entries():
        if "[grid]" not in str(label):
            continue
        try:
            data = app.load(key)
        except Exception:                                       # noqa: BLE001
            continue
        scan = data.scan
        if not scan.info.get("degrid.is_grid"):
            continue
        pattern = np.asarray(scan.value, dtype=float)
        if pattern.shape != tuple(frame_shape):
            continue
        out.append((str(label).split("   [")[0].strip(), pattern,
                    parse_box(scan.info.get("degrid.box")), settings_match(scan.info, info)))
    return out


def second_derivative(image, sigma=1.5):
    smooth = gaussian_filter(np.asarray(image, dtype=float), sigma)
    return -np.gradient(np.gradient(smooth, axis=1), axis=1)


def open_degrid_for(window, data):
    reason = DG.not_pixel_locked(data.scan.info)
    if data.kind not in ("cut", "map", "kz_map"):
        reason = reason or "it is in momentum, resampled off the detector's pixels"
    if reason:
        T.info(window, "De-grid", "This cannot be de-gridded: %s. The grid is only where "
               "the detector put it in data as measured -- de-grid the original, then do "
               "the rest." % reason)
        return None
    candidates = ([] if data.kind != "cut" else
                  grid_candidates(window.app, np.shape(data.cut_frame), data.scan.info))
    return window.adopt(DegridDialog(window, data, window.filename, candidates=candidates,
                                     colormap=window.colormap, flip=window.flip))


class DegridDialog(tk.Toplevel):
    def __init__(self, window, data, name, candidates=(), colormap="gray", flip=False):
        tk.Toplevel.__init__(self, window)
        self.window_, self.data, self.name = window, data, name
        self.app = getattr(window, "app", None)
        self.is_map = data.kind != "cut"
        self.colormap, self.flip = colormap, flip
        self.candidates = list(candidates)
        self.result = None
        self.title("De-grid — %s" % name)
        self.geometry("1200x950")
        scan = data.scan
        if self.is_map:
            self.axes = (np.asarray(scan.x, float), np.asarray(scan.k, float), np.asarray(scan.z, float))
            self.labels = (scan.labels.get("x", "x"), scan.labels.get("k", "angle"),
                           scan.labels.get("z", "energy"))
        else:
            self.axes = (np.asarray(scan.x, float), np.asarray(scan.y, float))
            self.labels = (scan.labels.get("x", "angle"), scan.labels.get("y", "energy"))
        ttk.Label(self, text=INTRO_MAP if self.is_map else INTRO_CUT, wraplength=1100,
                  justify="left").pack(fill="x", padx=6, pady=4)
        d = DG.DEFAULTS
        sources = ["This cut alone (FFT notch)"] + [
            label + ("" if same else "   (settings differ or unknown)")
            for label, _p, _b, same in self.candidates]
        default = 0
        for i, c in enumerate(self.candidates):
            if c[3]:
                default = i + 1
                break
        fields = []
        if not self.is_map:
            fields.append(Field("source", "Grid from", "choice", sources[default], sources))
        fields += [Field("tiles", "Local tiles per side (0 = off)", "int", d["tiles"]),
                   Field("iterations", "Registration passes", "int", d["iterations"]),
                   Field("smooth", "Smoothing σ (px)", "float", d["smooth"]),
                   Field("threshold", "Grid peak threshold", "float", d["seed_threshold"]),
                   Field("derivative", "Show −∂²I/∂E²", "bool", False)]
        if self.is_map:
            fields += [Field("slice", "Preview slice (index)", "int", self.axes[0].size // 2),
                       Field("with_grid", "Also list the grid pattern (for cuts)", "bool", True)]
        self.form = T.Form(self, fields, on_change=self.redraw, columns=3)
        self.form.pack(fill="x", padx=6)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=2)
        ttk.Button(bar, text="Find and remove the grid", command=self.run).pack(side="left")
        ttk.Button(bar, text="Result to list", command=self.export).pack(side="left", padx=4)
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right")
        self.plot = T.PlotFrame(self, figsize=(11, 7))
        self.plot.pack(fill="both", expand=True, padx=6)
        fig = self.plot.figure
        self.ax = [fig.add_subplot(2, 3, i + 1) for i in range(6 if self.is_map else 5)]
        self.report = tk.Text(self, height=6, width=130, font=("Courier", 9))
        self.report.pack(fill="x", padx=6)
        self.report.insert("1.0", "Press “Find and remove the grid”.")
        self.redraw()

    def settings(self):
        v = self.form.values()
        return {"tiles": v["tiles"], "iterations": v["iterations"], "smooth": v["smooth"],
                "seed_threshold": v["threshold"]}

    def cube(self):
        return self.data.angle_cube[3]

    def run(self):
        try:
            s = self.settings()
            if self.is_map:
                cube = np.asarray(self.cube(), dtype=float)

                def work(report):
                    def progress(done, total, label):
                        report(done / float(max(total, 1)), "%s (%d of %d)" % (label, done, total))
                    return DG.degrid_map(cube, settings=s, progress=progress)
                try:
                    result = T.run_blocking(self, "De-gridding the map", work)
                except T.JobCancelled:
                    self._say("Cancelled.")
                    return None
            else:
                frame = np.asarray(self.data.cut_frame, dtype=float)
                v = self.form.values()
                index = self.form.field("source").choices.index(v["source"])
                if index <= 0:
                    result = DG.degrid_cut_notch(frame, settings=s)
                else:
                    _label, pattern, box, _same = self.candidates[index - 1]
                    result = DG.degrid_cut_with_grid(frame, pattern, box=box, settings=s)
        except DG.GridNotFound as exc:
            self.result = None
            self._say("No grid found (%s). Nothing to remove -- or, in a single cut with "
                      "few counts, a grid hidden in the noise: de-grid a map taken with the "
                      "same settings and use its [grid]." % exc)
            return None
        except (ValueError, T.FormError) as exc:
            self.result = None
            self._say(str(exc))
            return None
        self.result = result
        if self.is_map:
            self.form.set("slice", result.preview_index)
        self._say(result.summary())
        self.redraw()
        return result

    def _say(self, text):
        self.report.delete("1.0", "end")
        self.report.insert("1.0", text)

    def images(self):
        if self.is_map:
            try:
                i = int(np.clip(self.form.value("slice"), 0, self.axes[0].size - 1))
            except T.FormError:
                i = self.axes[0].size // 2
            before = np.asarray(self.cube()[i], dtype=float)
            after = np.asarray(self.result.values[i], dtype=float) if self.result is not None else None
            return before, after, (self.axes[1], self.axes[2]), (self.labels[1], self.labels[2])
        before = np.asarray(self.data.cut_frame, dtype=float)
        after = np.asarray(self.result.values, dtype=float) if self.result is not None else None
        return before, after, self.axes, self.labels

    def redraw(self):
        if not hasattr(self, "ax"):
            return
        for ax in self.ax:
            ax.clear()
        before, after, (x, y), (xl, yl) = self.images()
        try:
            deriv = self.form.value("derivative")
        except T.FormError:
            deriv = False
        if deriv:
            before = second_derivative(before)
            after = second_derivative(after) if after is not None else None
        finite = before[np.isfinite(before)]
        lo, hi = (np.percentile(finite, [1, 99.5]) if finite.size else (0, 1))
        if deriv:
            lo = 0.0
        cmap = T.get_cmap(self.colormap, self.flip)
        for ax, image, title in ((self.ax[0], before, "Before"), (self.ax[1], after, "After")):
            ax.set_title(title, fontsize=9)
            if image is None:
                continue
            ax.pcolormesh(T.edges_of(x), T.edges_of(y), image.T, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xlabel(xl, fontsize=7)
            ax.set_ylabel(yl, fontsize=7)
            ax.tick_params(labelsize=7)
        if self.result is not None:
            model = self.result.model
            g = model.pattern()
            c0, c1 = g.shape[0] // 2, g.shape[1] // 2
            patch = g[max(c0 - 80, 0):c0 + 80, max(c1 - 80, 0):c1 + 80]
            reach = float(np.percentile(np.abs(patch), 99)) or 1.0
            self.ax[3].imshow(patch.T, origin="lower", cmap="gray", vmin=-reach, vmax=reach)
            self.ax[3].set_title("The grid found (central 160 x 160 px)", fontsize=9)
            b0, _a, _x, _l = self.images()
            image = b0[model.box]
            ratio = image / np.maximum(gaussian_filter(image, 8.0), 1e-9) - 1
            n0, n1 = ratio.shape
            power = np.abs(np.fft.fft2(ratio * np.outer(np.hanning(n0), np.hanning(n1)))) ** 2
            logp = np.fft.fftshift(np.log10(power + 1e-12))
            plo, phi = np.percentile(logp, [50, 99.9])
            grey = np.clip((logp - plo) / max(phi - plo, 1e-9), 0, 1)
            rgb = np.stack([grey] * 3, axis=-1)
            mask = np.fft.fftshift(model.region)
            rgb[mask] = 0.35 * rgb[mask] + 0.65 * np.array([1.0, 0.1, 0.1])
            f0 = np.fft.fftshift(np.fft.fftfreq(n0))
            f1 = np.fft.fftshift(np.fft.fftfreq(n1))
            self.ax[4].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
                              extent=(f0[0], f0[-1], f1[0], f1[-1]), aspect="auto")
            self.ax[4].set_title("k-space: grid regions in red", fontsize=9)
            if self.is_map:
                fits = self.result.fits
                idx = np.arange(len(fits))
                beta = np.array([f.beta if not f.empty else np.nan for f in fits])
                shift = np.array([f.shift if not f.empty else (np.nan, np.nan) for f in fits])
                ax = self.ax[5]
                ax.plot(idx, beta, color="#1f5aa6", label="contrast β")
                ax.plot(idx, shift[:, 0], color="#c23b22", label="shift, angle (px)")
                ax.plot(idx, shift[:, 1], color="#2a8a3e", label="shift, energy (px)")
                ax.legend(fontsize=6)
                ax.set_xlabel("slice", fontsize=7)
        self.ax[2].set_visible(False)
        self.plot.draw()

    def parameters(self):
        r = self.result
        m = r.model
        params = {"method": r.method}
        params.update(self.settings())
        params.update({"grid_peaks": len(m.peaks), "grid_kspace_fraction": float(m.region.mean()),
                       "grid_rms": m.rms, "box": box_text(m.box),
                       "contrast_before": r.contrast_before, "contrast_after": r.contrast_after})
        fits = [f for f in r.fits if not f.empty]
        if len(fits) > 1:
            betas = np.array([f.beta for f in fits])
            shifts = np.array([f.shift for f in fits])
            params.update(beta_median=float(np.median(betas)), beta_min=float(betas.min()),
                          beta_max=float(betas.max()),
                          shift_std_angle_px=float(shifts[:, 0].std()),
                          shift_std_energy_px=float(shifts[:, 1].std()))
        if not self.is_map:
            v = self.form.values()
            index = self.form.field("source").choices.index(v["source"])
            if index > 0:
                params["grid_source"] = self.candidates[index - 1][0]
        return params

    def export(self):
        if self.result is None:
            T.info(self, "De-grid", "Run it first.")
            return []
        scan = self.data.scan
        params = self.parameters()
        info = P.record_step(dict(scan.info), P.Step("degrid", params, source=self.name))
        names = self.app.names() if self.app is not None else []
        made = []
        motors = dict(getattr(scan, "fourd_info", {}) or {})
        if self.is_map:
            made.append(MemoryData(self.data.kind, self.axes, self.result.values,
                                   dict(scan.labels),
                                   source_label=T.unique_name("%s [degrid]" % self.name, names),
                                   parameters=params, prefix="degrid",
                                   source_path=getattr(self.data, "path", ""), source_info=info,
                                   source_motors=motors))
            if self.form.value("with_grid"):
                grid_info = dict(scan.info)
                grid_info["degrid.is_grid"] = 1
                grid_info["degrid.box"] = box_text(self.result.model.box)
                made.append(MemoryData("cut", (self.axes[1], self.axes[2]),
                                       self.result.model.full(),
                                       {"x": self.labels[1], "y": self.labels[2]},
                                       source_label=T.unique_name("%s [grid]" % self.name, names),
                                       parameters={"is_grid": 1,
                                                   "box": box_text(self.result.model.box),
                                                   "grid_rms": self.result.model.rms},
                                       prefix="degrid", source_path=getattr(self.data, "path", ""),
                                       source_info=grid_info))
        else:
            made.append(MemoryData("cut", self.axes, self.result.values, dict(scan.labels),
                                   source_label=T.unique_name("%s [degrid]" % self.name, names),
                                   parameters=params, prefix="degrid",
                                   source_path=getattr(self.data, "path", ""), source_info=info,
                                   source_motors=motors))
        for data in made:
            if self.app is not None:
                self.app.add_dataset(data)
        self.report.insert("end", "\nAdded " + ", ".join("“%s”" % d.source_label
                                                        for d in made) + " to the list.")
        return made
