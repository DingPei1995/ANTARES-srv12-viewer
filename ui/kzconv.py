"""
ui/kzconv.py
============
kz map -> momentum: convert a photon-energy scan to ``(k_z, k_par, E)``
(:func:`tools.kzconv.to_kz_cube`), choosing the inner potential V0.

* the preview reconverts one energy slice with the typed settings;
* the zone boundaries (from the lattice and the chosen surface normal) are
  drawn over it;
* **Scan V0** finds where the k_z period equals the lattice's;
* **Measure a k_z period** -- two k_z positions (typed, or clicked on the
  preview) of the same feature one zone apart; the matching lattice planes
  are listed (:mod:`tools.cleavage`).
"""
import numpy as np
import tkinter as tk
from tkinter import ttk

from tools import cleavage, kzconv
from tools.lattice import LatticeParams, validate_lattice_parameters
from tools.process import Step, record_step
from ui import tkbase as T
from ui.tkbase import Field
from ui.data import MemoryData

FLATNESS_WARNING_EV = 0.03


class KzConversionDialog(tk.Toplevel):
    def __init__(self, contour):
        tk.Toplevel.__init__(self, contour)
        self.contour = contour
        self.title("kz map -> momentum")
        self.geometry("1250x880")
        hv, angle, energy, _cube = contour.data.angle_cube
        self.hv = np.asarray(hv, dtype=float)
        self.angle = np.asarray(angle, dtype=float)
        self.energy = np.asarray(energy, dtype=float)
        self.normals = []
        self.banner = ttk.Label(self, text="", wraplength=1150, foreground="#8a4b00")
        self.banner.pack(fill="x", padx=6, pady=4)
        self._flatness()
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=6)
        e_default = float(np.clip(0.0, self.energy.min(), self.energy.max()))
        self.form = T.Form(left, [
            Field("V0", "Inner potential V0 (eV)", "float", 12.0),
            Field("mstar", "Effective mass (m*/me)", "float", 1.0),
            Field("phi", "Work function (eV)", "float", self._work_function()),
            Field("offset", "Angle offset (deg)", "float", 0.0),
            Field("theta", "Sample theta (deg)", "float", self._theta()),
            Field("n_kz", "k_z points", "int", 256),
            Field("n_kpar", "k_par points", "int", 256),
            Field("preview", "Preview at energy (eV)", "float", e_default),
            Field("sg", "Space group (zone lines)", "int", 194),
            Field("a", "Lattice a (Å)", "float", 3.2),
            Field("c", "Lattice c (Å)", "float", 6.0),
            Field("normal", "Surface normal / period", "choice", "", [""]),
            Field("zones", "Draw the zone boundaries", "bool", True),
        ], on_change=self.refresh)
        self.form.pack(fill="x")
        for key in ("sg", "a", "c"):
            self.form.widgets[key].bind("<Return>", lambda e: self.lattice_changed(), add="+")
        ttk.Button(left, text="Update lattice", command=self.lattice_changed).pack(fill="x", pady=2)
        per = ttk.LabelFrame(left, text="Measure a k_z period")
        per.pack(fill="x", pady=4)
        self.period_form = T.Form(per, [
            Field("k1", "First k_z", "optfloat", None),
            Field("k2", "Same feature, one zone along", "optfloat", None),
        ])
        self.period_form.pack(fill="x")
        prow = ttk.Frame(per)
        prow.pack(fill="x")
        ttk.Button(prow, text="Pick two on the preview", command=self.pick).pack(side="left")
        ttk.Button(prow, text="Match planes", command=self.match_period).pack(side="left", padx=3)
        bar = ttk.Frame(left)
        bar.pack(fill="x", pady=4)
        ttk.Button(bar, text="Scan V0", command=self.scan_v0).pack(fill="x")
        ttk.Button(bar, text="Convert the whole cube", command=self.convert).pack(fill="x", pady=2)
        ttk.Button(bar, text="Close", command=self.destroy).pack(fill="x")
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        self.plot = T.PlotFrame(right, figsize=(7.5, 5.5))
        self.plot.pack(fill="both", expand=True)
        grid = self.plot.figure.add_gridspec(1, 2, width_ratios=(2.2, 1), wspace=0.3)
        self.ax = self.plot.figure.add_subplot(grid[0])
        self.ax_scan = self.plot.figure.add_subplot(grid[1])
        self.plot.canvas.mpl_connect("button_press_event", self._clicked)
        self.report = tk.Text(right, height=9, width=100, font=("Courier", 9))
        self.report.pack(fill="x")
        self._picking = []
        self._scan = None
        self.lattice_changed()

    # -- what the file knows ------------------------------------------------------
    def _work_function(self):
        info = self.contour.data.scan.info or {}
        for key in ("cassiopee.work_function_eV", "work_function_eV", "analyser_work_function_eV"):
            value = info.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                return float(value)
            if isinstance(value, (list, np.ndarray)) and np.size(value):
                return float(np.nanmean(np.asarray(value, dtype=float)))
        return 4.5

    def _theta(self):
        value = (self.contour.data.scan.info or {}).get("cassiopee.sample_theta_deg")
        return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else 0.0

    def _flatness(self):
        try:
            spread, _p, _a = kzconv.edge_flatness(self.contour.full_cube(), self.angle, self.energy)
        except Exception:                                       # noqa: BLE001
            return
        if spread > FLATNESS_WARNING_EV:
            self.banner.configure(text=(
                "The Fermi edge still bends by %.3f eV across the analyser angle. "
                "Straighten it first (slit cut -> FS correction), or that bend becomes a "
                "bend in k_z that looks like dispersion. Converting anyway is allowed." % spread))
        else:
            self.banner.configure(text="The Fermi edge is flat across the analyser angle to "
                                       "%.3f eV. Good to convert." % spread)

    # -- settings ---------------------------------------------------------------------
    def settings(self):
        v = self.form.values()
        return dict(inner_potential=v["V0"], work_function=v["phi"], effective_mass=v["mstar"],
                    angle_offset=v["offset"], theta_position=v["theta"])

    def lattice(self):
        v = self.form.values()
        return LatticeParams(a=v["a"], b=v["a"], c=v["c"], space_group=int(v["sg"]))

    def lattice_changed(self):
        self.complaints = []
        self.normals = []
        try:
            params = self.lattice()
            self.complaints = list(validate_lattice_parameters(params))
            for hkl, length, _d in cleavage.reciprocal_lengths(params, max_index=2)[:12]:
                self.normals.append(("(%s)   period %.4f Å⁻¹" % (
                    " ".join(str(int(v)) for v in hkl), length), length))
        except Exception as exc:                                # noqa: BLE001
            self.complaints = ["lattice not usable: %s" % exc]
        names = [n for n, _ in self.normals] or ["(none)"]
        self.form.set_choices("normal", names)
        self.refresh()

    def surface_period(self):
        name = self.form.raw("normal")
        for n, length in self.normals:
            if n == name:
                return float(length)
        try:
            return 2.0 * np.pi / float(self.form.value("c"))
        except T.FormError:
            return float("nan")

    # -- preview -----------------------------------------------------------------------
    def preview_slice(self):
        v = self.form.values()
        index = T.nearest_index(self.energy, v["preview"])
        one = np.asarray([float(self.energy[index])])
        # the one energy plane, not the whole cube as float
        plane = np.asarray(self.contour._cube()[:, :, index], dtype=float)
        kz_axis, kpar_axis, _e, out = kzconv.to_kz_cube(
            self.hv, self.angle, one, plane[:, :, None],
            n_kz=int(v["n_kz"]), n_kpar=int(v["n_kpar"]), **self.settings())
        return kz_axis, kpar_axis, out[:, :, 0]

    def refresh(self):
        if not hasattr(self, "ax"):
            return
        try:
            kz, kpar, plane = self.preview_slice()
        except Exception as exc:                                # noqa: BLE001
            self._say(str(exc))
            return
        ax = self.ax
        ax.clear()
        lo, hi = T.auto_levels(plane)
        ax.pcolormesh(T.edges_of(kz), T.edges_of(kpar), plane.T,
                      cmap=T.get_cmap(self.contour.colormap, self.contour.flip), vmin=lo, vmax=hi)
        ax.set_xlabel("k_z (Å⁻¹)")
        ax.set_ylabel("k_par (Å⁻¹)")
        ax.set_aspect("equal", adjustable="box")
        period = self.surface_period()
        if self.form.value("zones") and np.isfinite(period) and period > 0:
            for n in range(int(np.floor(kz[0] / period)), int(np.ceil(kz[-1] / period)) + 1):
                if kz[0] <= n * period <= kz[-1]:
                    ax.axvline(n * period, color="#e8e8e8", ls="--", lw=1)
        try:
            v = self.period_form.values()
            for key in ("k1", "k2"):
                if v[key] is not None:
                    ax.axvline(v[key], color="#ff7f0e", lw=1.5)
        except T.FormError:
            pass
        self.plot.draw()
        coverage = 100.0 * float(np.isfinite(plane).mean())
        lines = ["k_z %.3f to %.3f A^-1 (%.2f zones of %.4f A^-1)" % (
            kz[0], kz[-1], (kz[-1] - kz[0]) / period if period else float("nan"), period),
                 "k_par %.3f to %.3f A^-1" % (kpar[0], kpar[-1]),
                 "%.0f%% of the grid is covered by the measurement" % coverage]
        lines += self.complaints
        self._say("\n".join(lines))

    def _say(self, text):
        self.report.delete("1.0", "end")
        self.report.insert("1.0", text)

    # -- period ----------------------------------------------------------------------------
    def pick(self):
        self._picking = [None]
        self._say("Click the same feature twice on the preview, one zone apart.")

    def _clicked(self, event):
        if not self._picking or event.inaxes is not self.ax or event.xdata is None:
            return
        if self.plot.navigating():
            return
        if self._picking == [None]:
            self._picking = [event.xdata]
            self.period_form.set("k1", event.xdata)
            self.period_form.set("k2", None)
            return
        self.period_form.set("k2", event.xdata)
        self._picking = []
        self.match_period()

    def match_period(self):
        try:
            v = self.period_form.values()
        except T.FormError as exc:
            self._say(str(exc))
            return
        if v["k1"] is None or v["k2"] is None:
            self._say("Give both k_z positions.")
            return
        distance = abs(v["k2"] - v["k1"])
        if distance <= 0:
            self._say("Those two points are at the same k_z.")
            return
        lines = ["Picked k_z separation %.4f A^-1 (real-space repeat %.3f A)" % (
            distance, 2 * np.pi / distance)]
        try:
            found = cleavage.candidates(distance, self.lattice(), tolerance=0.15)
        except Exception as exc:                                # noqa: BLE001
            lines.append(str(exc))
            found = []
        if not found:
            lines.append("No lattice plane matches that within 15%: check the lattice "
                         "constants, or the two points are not one zone apart.")
        else:
            lines.append("Planes that match within 15%:")
            lines += ["  " + c.describe() for c in found[:6]]
        self.refresh()
        self._say("\n".join(lines))

    # -- V0 scan ---------------------------------------------------------------------------
    def scan_v0(self):
        cube = self.contour.full_cube()
        period = self.surface_period()
        settings = self.settings()
        settings.pop("inner_potential")
        binding = float(self.form.value("preview"))

        def work(report):
            def tick(done, total):
                report(done / float(max(1, total)), "Trying inner potentials (%d of %d)" % (done + 1, total))
            return kzconv.scan_inner_potential(self.hv, self.angle, self.energy, cube,
                                               spacing=2.0 * np.pi / period,
                                               binding_energy=binding, progress=tick, **settings)
        T.run_job(self, "Scanning the inner potential", work, on_done=self.show_scan,
                  on_error=lambda exc: self._say(str(exc)))

    def show_scan(self, result):
        ax = self.ax_scan
        ax.clear()
        good = np.isfinite(result.periods)
        ax.plot(result.inner_potentials[good], result.periods[good], color="#1f77b4")
        ax.axhline(result.target, color="#d62728", ls="--")
        ax.set_xlabel("V0 (eV)", fontsize=8)
        ax.set_ylabel("k_z period (A^-1)", fontsize=8)
        self.plot.draw()
        if result.best is None:
            self._say("The measured period never equals the lattice's within the scanned "
                      "range: check the lattice, the surface normal or widen the scan.")
            return
        self._say("The k_z period matches %.4f A^-1 at V0 = %.1f eV.\nSensitivity %.5f A^-1 "
                  "per eV, so measuring the period to 2%% pins V0 only to about +/-%.1f eV "
                  "(the scan's photon-energy range, not the fit)." % (
                      result.target, result.best, result.sensitivity, result.uncertainty()))
        self.form.set("V0", float(result.best))
        self.refresh()

    # -- converting ----------------------------------------------------------------------
    def convert(self):
        try:
            v = self.form.values()
            settings = self.settings()
        except T.FormError as exc:
            T.warning(self, "kz -> momentum", str(exc))
            return
        cube = self.contour.full_cube()
        hv, angle, energy = self.hv, self.angle, self.energy

        def work(report):
            def tick(done, total):
                report(done / float(max(1, total)), "Converting energy slice %d of %d" % (done + 1, total))
            return kzconv.to_kz_cube(hv, angle, energy, cube, n_kz=int(v["n_kz"]),
                                     n_kpar=int(v["n_kpar"]), progress=tick, **settings)
        T.run_job(self, "Converting to momentum", work,
                  on_done=lambda converted: self.finish(converted, v, settings),
                  on_error=lambda exc: self._say(str(exc)))

    def finish(self, converted, v, settings):
        kz_axis, kpar_axis, energy, out = converted
        data = self.contour.data
        scan = data.scan
        parameters = dict(settings)
        parameters.update({"n_kz": int(v["n_kz"]), "n_kpar": int(v["n_kpar"]),
                           "space_group": int(v["sg"]), "lattice_a": float(v["a"]),
                           "lattice_c": float(v["c"]),
                           "surface_period_invA": float(self.surface_period())})
        info = record_step(dict(scan.info), Step("kz_to_k", parameters, source=self.contour.filename))
        info["kz.inner_potential_eV"] = float(settings["inner_potential"])
        name = self.contour.unique("%s [kz]" % self.contour.filename)
        result = MemoryData("kz_map_k", (kz_axis, kpar_axis, energy), out,
                            {"x": "k_z (Å⁻¹)", "k": "k_par (Å⁻¹)",
                             "z": scan.labels.get("z", "E - E_F (eV)")},
                            source_label=name, parameters=parameters, prefix="kz_to_k",
                            source_path=getattr(data, "path", ""), source_info=info,
                            source_motors=dict(scan.fourd_info))
        self.contour.emit(result)
        coverage = 100.0 * float(np.isfinite(out).mean())
        self._say("Converted to (k_z, k_par, E) at V0 = %.1f eV.\nk_z %.3f to %.3f, k_par %.3f "
                  "to %.3f A^-1, %.0f%% covered.\nAdded to the list as “%s”." % (
                      settings["inner_potential"], kz_axis[0], kz_axis[-1], kpar_axis[0],
                      kpar_axis[-1], coverage, name))
