"""
ui/kzmap.py
===========
kz map processing: fit the Fermi edge of every spectrum of a photon-energy
scan, shift each to put its own edge at zero, crop to the range they all
cover, and (optionally) normalise each by its total intensity
(:func:`tools.kzmap.process_kz_map`).

The region is typed (angle range = which channels are summed into the EDC,
energy range = what the fit runs over), and is shown as the box on the slit
cut, which opens alongside. The same *index* region is used on every
spectrum.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk

from tools import kzmap
from tools.process import Step, record_step
from ui import tkbase as T
from ui.tkbase import Field
from ui.data import MemoryData

EDGE_HALF_WIDTH_EV = 0.35


class KzMapProcessDialog(tk.Toplevel):
    def __init__(self, contour):
        from ui.analysis import metadata_temperature
        tk.Toplevel.__init__(self, contour)
        self.contour = contour
        self.result = None
        self.title("kz map processing")
        self.geometry("620x720")
        hv, angles, energy, _cube = contour.data.angle_cube
        self.hv = np.asarray(hv, dtype=float)
        self.angles = np.asarray(angles, dtype=float)
        self.energy = np.asarray(energy, dtype=float)
        ttk.Label(self, wraplength=580, justify="left", text=(
            "The box (angle range summed into the EDC, energy window of the fit) "
            "is used, by detector index, on every spectrum. It starts around the "
            "apparent Fermi edge; it is drawn on the slit cut.")).pack(fill="x", padx=6, pady=4)
        self.form = T.Form(self, [
            Field("a0", "Angle from", "float", float(self.angles.min())),
            Field("a1", "Angle to", "float", float(self.angles.max())),
            Field("e0", "Energy from", "float", float(self.energy.min())),
            Field("e1", "Energy to", "float", float(self.energy.max())),
            Field("temperature", "Temperature (K)", "float", metadata_temperature(contour.data)),
            Field("hold", "Hold the temperature", "bool", True),
            Field("normalise", "Normalise each spectrum by its total intensity", "bool", True),
        ], on_change=self.box_changed, columns=2)
        self.form.pack(fill="x", padx=6)
        row = ttk.Frame(self)
        row.pack(fill="x", padx=6)
        ttk.Button(row, text="Around the edge", command=self.select_edge).pack(side="left")
        ttk.Button(row, text="Whole cut", command=self.select_everything).pack(side="left", padx=3)
        ttk.Button(row, text="From the slit cut's box", command=self.from_cut_box).pack(side="left")
        self.region_label = ttk.Label(self, text="", wraplength=580, justify="left")
        self.region_label.pack(fill="x", padx=6, pady=4)
        self.plot = T.PlotFrame(self, figsize=(5.5, 3.2), toolbar=False)
        self.plot.pack(fill="both", expand=True, padx=6)
        self.ax = self.plot.figure.add_subplot(111)
        self.plot.figure.subplots_adjust(left=0.16, bottom=0.18)
        self.note = tk.Text(self, height=7, width=80, font=("Courier", 9))
        self.note.pack(fill="x", padx=6)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Fit and assemble", command=self.run).pack(side="left")
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right")
        contour.open_cut("slit")
        self.select_edge()

    def cut_panel(self):
        window = self.contour.cut_windows.get("slit")
        return None if window is None else window.panel

    def select_edge(self):
        lo, hi = float(self.energy.min()), float(self.energy.max())
        try:
            from tools.fermi import initial_guess
            edc = np.nansum(self.contour.full_cube(), axis=(0, 1))
            centre = float(initial_guess(self.energy, edc,
                                         float(self.form.value("temperature")))["ef"])
        except Exception:                                       # noqa: BLE001
            centre = 0.5 * (lo + hi)
        low, high = max(lo, centre - EDGE_HALF_WIDTH_EV), min(hi, centre + EDGE_HALF_WIDTH_EV)
        if self.energy.size > 1 and high - low < 8 * abs(self.energy[1] - self.energy[0]):
            low, high = lo, hi
        self.form.set("a0", float(self.angles.min()))
        self.form.set("a1", float(self.angles.max()))
        self.form.set("e0", low)
        self.form.set("e1", high)
        self.box_changed()

    def select_everything(self):
        for key, value in (("a0", self.angles.min()), ("a1", self.angles.max()),
                           ("e0", self.energy.min()), ("e1", self.energy.max())):
            self.form.set(key, float(value))
        self.box_changed()

    def from_cut_box(self):
        panel = self.cut_panel()
        if panel is None or panel.box is None:
            T.info(self, "kz map processing", "Set a box on the slit cut first.")
            return
        x0, x1, y0, y1 = panel.box
        for key, value in (("a0", x0), ("a1", x1), ("e0", y0), ("e1", y1)):
            self.form.set(key, value)
        self.box_changed()

    def index_region(self):
        v = self.form.values()
        a = sorted((T.nearest_index(self.angles, v["a0"]), T.nearest_index(self.angles, v["a1"])))
        e = sorted((T.nearest_index(self.energy, v["e0"]), T.nearest_index(self.energy, v["e1"])))
        return (a[0], a[1]), (e[0], e[1])

    def box_changed(self):
        try:
            v = self.form.values()
            (a0, a1), (e0, e1) = self.index_region()
        except T.FormError as exc:
            self.region_label.configure(text=str(exc))
            return
        panel = self.cut_panel()
        if panel is not None:
            panel.set_box((v["a0"], v["a1"], v["e0"], v["e1"]))
        self.region_label.configure(text=(
            "Angle channels %d-%d (%.3f to %.3f), %d summed into the EDC.\nFitting over "
            "energy %.4f to %.4f eV (%d points)%s." % (
                a0, a1, self.angles[a0], self.angles[a1], a1 - a0 + 1,
                self.energy[e0], self.energy[e1], e1 - e0 + 1,
                "" if e1 - e0 + 1 >= 8 else " -- too few: a fit needs at least 8")))

    def run(self):
        try:
            region = self.index_region()
            v = self.form.values()
        except T.FormError as exc:
            T.warning(self, "kz map processing", str(exc))
            return None
        if region[1][1] - region[1][0] + 1 < 8:
            T.warning(self, "kz map processing", "The energy window needs at least 8 points.")
            return None
        cube = self.contour.full_cube()
        energy = self.energy
        fixed = ("temperature",) if v["hold"] else ()

        def work(report):
            def tick(done, total):
                report(done / float(max(1, total)),
                       "Fitting the Fermi edge, spectrum %d of %d" % (done + 1, total))
            return kzmap.process_kz_map(cube, energy, region, temperature=v["temperature"],
                                        fixed=fixed, normalise=v["normalise"], progress=tick)

        T.run_job(self, "kz map processing", work,
                  on_done=lambda result: self.finish(result, region, v),
                  on_error=lambda exc: self._say(str(exc)))
        return True

    def _say(self, text):
        self.note.delete("1.0", "end")
        self.note.insert("1.0", text)

    def finish(self, result, region, v):
        self.result = result
        ok = result.ok
        self.ax.clear()
        self.ax.plot(self.hv[ok], result.ef[ok], "o-", ms=3, color="#1f77b4", label="fitted")
        self.ax.plot(self.hv[~ok], result.ef[~ok], "x", ms=8, color="#d62728",
                     label="interpolated")
        self.ax.set_xlabel(self.contour.data.scan.labels.get("x", "hv (eV)"), fontsize=8)
        self.ax.set_ylabel("E_F (eV)", fontsize=8)
        self.ax.legend(fontsize=7)
        self.plot.draw()
        data = self.contour.data
        scan = data.scan
        labels = dict(scan.labels or {})
        labels["z"] = "E - E_F (eV)"
        parameters = {"angle_index_from": int(region[0][0]), "angle_index_to": int(region[0][1]),
                      "energy_index_from": int(region[1][0]), "energy_index_to": int(region[1][1]),
                      "temperature_K": float(v["temperature"]), "temperature_held": bool(v["hold"]),
                      "normalised_by_total": bool(result.normalised),
                      "fermi_level_spread_eV": float(result.spread),
                      "energy_trimmed_eV": float(result.trimmed),
                      "spectra_fitted": int(result.ok.sum()),
                      "spectra_interpolated": int((~result.ok).sum())}
        info = record_step(dict(scan.info), Step("kz_align", parameters,
                                                 source=self.contour.filename))
        info["kz.fermi_level_eV"] = np.asarray(result.ef, dtype=float)
        info["kz.fermi_level_fitted"] = np.asarray(result.ok, dtype=bool)
        dataset = MemoryData(data.kind, (self.hv, self.angles, result.energy), result.cube,
                             labels, source_label=self.contour.unique("%s [E-Ef]" % self.contour.filename),
                             parameters=parameters, prefix="kz_align",
                             source_path=getattr(data, "path", ""), source_info=info,
                             source_motors=dict(scan.fourd_info))
        self.contour.emit(dataset)
        self._say(result.summary() + "\nAdded to the list as “%s”." % dataset.source_label)
