"""
ui/analysis.py
==============
The analysis dialogs a viewer opens from its Functions menu:

* :class:`KConversionDialog` -- a map from angle to momentum;
* :class:`CutKConversionDialog` -- one cut from angle to momentum;
* :class:`ArbitraryCutDialog` -- a cut along a path through a contour;
* :class:`FSCorrectionDialog` -- straighten a curved feature (Fermi edge);
* :class:`FermiFitDialog` -- fit E_F, offset / divide it out, per channel;
* :class:`DataOperationsDialog` -- truncate / self-normalise / compress;
* :class:`BrillouinZoneDialog` -- overlay a (moire) Brillouin zone.

In the PyQt5 version several of these read their input off the mouse
(points clicked on a contour, a box dragged on a cut). Here every such input
is a **typed field**; the "Pick by clicking" buttons only fill those fields
from clicks on the image, so the numbers can always be checked and corrected
before anything is computed.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk

from loader.nxs_file import CUBE_KINDS, energy_slot
from tools.dataops import (truncate, self_normalize, compress, ARRAY_AXES,
                           CONSTRUCTOR_AXES)
from ui import tkbase as T
from ui.tkbase import Field
from ui.data import MemoryData, KMapData, unit_of, full_array

MOMENTUM_MARKS = ("Å⁻¹", "A^-1", "1/Å", "Å-1", "AA^-1")


# --------------------------------------------------------------------------
# Reasons, from the metadata
# --------------------------------------------------------------------------
def momentum_cut_reason(data):
    """Empty if this dataset can be fitted by the MDC/EDC fit, else why."""
    if getattr(data, "kind", "") != "cut":
        return "The fit panel works on a 2-D cut. Take a slice of a map first."
    labels = getattr(getattr(data, "scan", None), "labels", {}) or {}
    x_label = str(labels.get("x", ""))
    info = getattr(getattr(data, "scan", None), "info", {}) or {}
    converted = any(str(key).startswith("kcut.") for key in info)
    if converted or any(mark in x_label for mark in MOMENTUM_MARKS):
        return ""
    return ("This cut's first axis is “%s”, which is not a momentum. "
            "Fit only k-converted cuts (v_F in eV·Å, m* ...): use "
            "“Cut k conversion” first, or slice a converted k-map."
            % (x_label or "unnamed"))


def cut_deflector_angle(data, default=0.0):
    info = dict(getattr(data.scan, "info", {}) or {})
    for key in ("DeflectorAngle_deg", "kcut.deflector_deg"):
        if key in info:
            try:
                return float(np.asarray(info[key]).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                pass
    return float(default)


def cut_k_conversion_blocked(data):
    """Why this cut cannot be converted from angle to momentum, or None."""
    info = dict(getattr(data.scan, "info", {}) or {})
    if any(str(key).startswith("kcut.") for key in info):
        return "this cut is already in momentum"
    if any(str(key).startswith("arbcut.") for key in info):
        return ("this cut was taken along a path through a map, not at one "
                "deflector position — convert the map instead, then cut it")
    panel = str(info.get("slice.panel", ""))
    if panel and "slit" not in panel.lower() and "cut" not in panel.lower():
        return "this is a slice through a map (%s), not a slit cut" % panel
    label = str(getattr(data.scan, "labels", {}).get("x", ""))
    if "Å" in label:
        return "this cut's angle axis is already a momentum"
    return None


def sample_rotations(data):
    motors = dict(getattr(data.scan, "fourd_info", {}) or {})
    return {name: float(motors[name]) for name in ("SRn", "SRz")
            if name in motors and isinstance(motors[name], (int, float, np.floating))}


def metadata_temperature(data, default=30.0):
    info = dict(getattr(data.scan, "info", {}) or {})
    info.update(getattr(data.scan, "fourd_info", {}) or {})
    for key, value in info.items():
        name = str(key).lower()
        if "temp" not in name or "setpoint" in name:
            continue
        try:
            number = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if 1.0 <= number <= 1000.0:
            return number
    return float(default)


def reference_frame(data):
    """(angle, energy, frame (angle, E), angle label, energy label): a cut
    as it is, a cube summed over its first (deflector) axis."""
    scan = data.scan
    if data.kind == "cut":
        return (np.asarray(scan.x, dtype=float), np.asarray(scan.y, dtype=float),
                np.asarray(scan.value, dtype=float),
                scan.labels.get("x", "angle"), scan.labels.get("y", "E"))
    if data.kind in CUBE_KINDS:
        cube = np.asarray(scan.value, dtype=float)
        return (np.asarray(scan.k, dtype=float), np.asarray(scan.z, dtype=float),
                np.nansum(cube, axis=0),
                scan.labels.get("k", "angle"), scan.labels.get("z", "E"))
    raise ValueError("%s has no angle-vs-energy frame to fit" % data.kind)


# --------------------------------------------------------------------------
# A non-modal tool window with a form
# --------------------------------------------------------------------------
class ToolDialog(tk.Toplevel):
    """A modeless dialog belonging to a viewer: a form, a note, buttons."""

    def __init__(self, window, title, intro=""):
        tk.Toplevel.__init__(self, window)
        self.window_ = window
        self.title(title)
        self.protocol("WM_DELETE_WINDOW", self.close)
        if intro:
            ttk.Label(self, text=intro, wraplength=560, justify="left").pack(
                fill="x", padx=8, pady=6)
        self.content = ttk.Frame(self)
        self.content.pack(fill="both", expand=True, padx=8)
        self.note = ttk.Label(self, text="", wraplength=560, justify="left",
                              foreground="#225")
        self.note.pack(fill="x", padx=8, pady=4)
        self.buttons = ttk.Frame(self)
        self.buttons.pack(fill="x", padx=8, pady=6)
        ttk.Button(self.buttons, text="Close", command=self.close).pack(side="right", padx=3)
        self._click_hook = None

    def button(self, text, command, side="left"):
        b = ttk.Button(self.buttons, text=text, command=command)
        b.pack(side=side, padx=3)
        return b

    def say(self, text):
        try:
            self.note.configure(text=text)
        except tk.TclError:
            pass

    def values(self, form):
        try:
            return form.values()
        except T.FormError as exc:
            T.warning(self, self.title(), str(exc))
            return None

    # -- clicking on a panel fills fields ------------------------------------------
    def start_picking(self, panel, callback):
        self.stop_picking()

        def hook(x, y, button):
            if button == 1:
                callback(x, y)
                return True
            return False
        self._click_hook = (panel, hook)
        panel.on_click_hooks.append(hook)

    def stop_picking(self):
        if self._click_hook is not None:
            panel, hook = self._click_hook
            if hook in panel.on_click_hooks:
                panel.on_click_hooks.remove(hook)
            self._click_hook = None

    def close(self):
        self.stop_picking()
        try:
            self.destroy()
        except tk.TclError:
            pass


class PointsBox(ttk.Frame):
    """Points typed as ``x y`` per line (or ``x, y``)."""

    def __init__(self, parent, height=6, on_change=None):
        ttk.Frame.__init__(self, parent)
        self.text = tk.Text(self, width=34, height=height)
        self.text.pack(side="left", fill="both", expand=True)
        self.on_change = on_change
        self.text.bind("<KeyRelease>", lambda e: on_change() if on_change else None)

    def points(self):
        out = []
        for line in self.text.get("1.0", "end").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            numbers = T.parse_floats(line)
            if len(numbers) < 2:
                raise ValueError("each line needs two numbers: %r" % line)
            out.append((numbers[0], numbers[1]))
        return out

    def set_points(self, points):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n".join("%.6g  %.6g" % (x, y) for x, y in points))
        if self.on_change:
            self.on_change()

    def add(self, x, y):
        existing = self.text.get("1.0", "end").strip()
        self.text.insert("end", ("\n" if existing else "") + "%.6g  %.6g" % (x, y))
        if self.on_change:
            self.on_change()


# --------------------------------------------------------------------------
# Map k conversion
# --------------------------------------------------------------------------
class KConversionDialog(ToolDialog):
    """Angle map -> (kx, ky, E). The origin (theta, phi offsets) starts at
    the contour's cursor ("Read cursor" copies it again)."""

    def __init__(self, contour):
        ToolDialog.__init__(
            self, contour, "Map k conversion",
            "Theta / phi offsets are the angles of the point that becomes "
            "k = (0, 0) (initially the contour's cursor; move the cursor and "
            "press 'Read cursor', or type them). The sample rotation can be "
            "typed, or computed from two points along a direction that "
            "should end up vertical.")
        self.contour = contour
        n_e = len(contour.E)
        step = abs(float(contour.E[1] - contour.E[0])) * 1000 if n_e > 1 else 10.0
        theta, phi = contour.readout_position()
        self.form = T.Form(self.content, [
            Field("name", "Name", "str", contour.unique("%s [k]" % contour.filename), width=36),
            Field("theta", "Theta offset (deg)", "float", theta,
                  help="Deflector angle of the k-space origin."),
            Field("phi", "Phi offset (deg)", "float", phi,
                  help="Slit angle of the k-space origin."),
            Field("azimuth", "Sample rotation (deg)", "float", 0.0),
            Field("eoff", "Energy offset (eV)", "float", 0.0,
                  help="Added to the kinetic energy before converting."),
            Field("mode", "Output grid", "choice", "by element number",
                  ["by element number", "by resolution"]),
            Field("nkx", "kx points", "int", 100),
            Field("nky", "ky points", "int", 100),
            Field("ne", "E points", "int", n_e),
            Field("dkx", "kx step (Å⁻¹)", "float", 0.01),
            Field("dky", "ky step (Å⁻¹)", "float", 0.01),
            Field("de", "E step (meV)", "float", round(max(step, 0.001), 3)),
        ], on_change=self.update_note, columns=2)
        self.form.pack(fill="x")
        rot = ttk.LabelFrame(self.content, text="Rotation from two points (x y per line)")
        rot.pack(fill="x", pady=4)
        self.rot_points = PointsBox(rot, height=2)
        self.rot_points.pack(side="left", fill="x", expand=True, padx=4)
        side = ttk.Frame(rot)
        side.pack(side="left")
        ttk.Button(side, text="Pick by clicking", command=self.pick_rotation).pack(fill="x")
        ttk.Button(side, text="Compute rotation", command=self.compute_rotation).pack(fill="x")
        self.button("Read cursor", self.read_cursor)
        self.button("Convert", self.convert)
        self.update_note()

    def read_cursor(self):
        theta, phi = self.contour.readout_position()
        self.form.set("theta", theta)
        self.form.set("phi", phi)
        self.update_note()

    def pick_rotation(self):
        self.rot_points.set_points([])
        self.say("Click one or two points on the contour along the direction "
                 "that should end up vertical (one point: the line from the origin).")

        def add(x, y):
            pts = self.rot_points.points()
            if len(pts) >= 2:
                pts = pts[-1:]
            pts.append((x, y))
            self.rot_points.set_points(pts)
            if len(pts) == 2:
                self.stop_picking()
                self.compute_rotation()
        self.start_picking(self.contour.panel, add)

    def compute_rotation(self):
        from tools.kspace import points_to_azimuth
        try:
            pts = self.rot_points.points()
            v = self.form.values()
        except (ValueError, T.FormError) as exc:
            self.say(str(exc))
            return
        if len(pts) == 1:
            pts = [(v["theta"], v["phi"]), pts[0]]
        if len(pts) != 2:
            self.say("Give one or two points.")
            return
        try:
            rotation = points_to_azimuth(pts)
        except ValueError as exc:
            self.say(str(exc))
            return
        self.form.set("azimuth", rotation)
        panel = self.contour.panel
        panel.clear_overlays("krot")
        panel.add_overlay("krot", "plot", [p[0] for p in pts], [p[1] for p in pts],
                          "o-", color="#00e5ff")
        panel.redraw()
        self.say("Rotation %.3f° stands that line vertical (along ky)." % rotation)

    def _extent(self, v):
        from tools.kspace import k_extent
        c = self.contour
        return k_extent(c.defl, c.slit, c.E, theta_offset_deg=v["theta"],
                        phi_offset_deg=v["phi"], azimuth_deg=v["azimuth"],
                        energy_offset_eV=v["eoff"])

    def grid_counts(self, v):
        if v["mode"] == "by element number":
            return int(v["nkx"]), int(v["nky"]), int(v["ne"])
        kx_lo, kx_hi, ky_lo, ky_hi = self._extent(v)
        span = abs(float(self.contour.E[-1] - self.contour.E[0])) * 1000.0
        return (max(int(round((kx_hi - kx_lo) / v["dkx"])) + 1, 2),
                max(int(round((ky_hi - ky_lo) / v["dky"])) + 1, 2),
                max(int(round(span / v["de"])) + 1, 2))

    def update_note(self):
        try:
            v = self.form.values()
            n = self.grid_counts(v)
            kx_lo, kx_hi, ky_lo, ky_hi = self._extent(v)
        except Exception as exc:                                # noqa: BLE001
            self.say(str(exc))
            return
        self.say("kx %.3g .. %.3g, ky %.3g .. %.3g Å⁻¹ → %d x %d x %d points"
                 % (kx_lo, kx_hi, ky_lo, ky_hi, n[0], n[1], n[2]))

    def settings(self):
        v = self.values(self.form)
        if v is None:
            return None, None
        n = self.grid_counts(v)
        return ({"theta_offset_deg": v["theta"], "phi_offset_deg": v["phi"],
                 "azimuth_deg": v["azimuth"], "energy_offset_eV": v["eoff"],
                 "n_kx": n[0], "n_ky": n[1], "n_energy": n[2]},
                v["name"].strip() or "%s [k]" % self.contour.filename)

    def convert(self):
        from tools.kspace import convert_map
        try:
            settings, name = self.settings()
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "k conversion", str(exc))
            return
        if settings is None:
            return
        c = self.contour
        cube = c.full_cube()

        def work(report):
            report(None, "Converting to k-space...")
            return convert_map(c.defl, c.slit, c.E, cube, **settings)

        def done(result):
            kx, ky, energy, kcube = result
            info = {k: v for k, v in c.data.scan.info.items() if not k.startswith("kconv.")}
            kdata = KMapData(kx, ky, energy, kcube, source_label=name,
                             parameters=settings,
                             source_path=getattr(c.data, "path", ""),
                             source_info=info,
                             source_motors=dict(c.data.scan.fourd_info))
            c.emit(kdata)
            finite = float(np.isfinite(kcube).mean() * 100.0)
            c.say("Converted to k-space: %dx%dx%d, %.0f%% inside the light cone "
                  "-- added to the list" % (kcube.shape + (finite,)))
            c.panel.clear_overlays("krot")
            self.close()

        T.run_job(self, "k conversion", work, on_done=done,
                  on_error=lambda exc: T.warning(self, "k conversion",
                                                 "Conversion failed:\n%s" % exc))


# --------------------------------------------------------------------------
# Cut k conversion
# --------------------------------------------------------------------------
class CutKConversionDialog(ToolDialog):
    """One cut from angle to momentum. Needs where Gamma is (from a
    converted map in the list, or typed) and the cut's deflector angle."""

    def __init__(self, window):
        ToolDialog.__init__(
            self, window, "Cut k conversion",
            "A cut is a line in (kx, ky) that generally misses Γ, so the "
            "conversion needs Γ's angles: inherit them from a converted "
            "map ('From a converted map...'), or type them.")
        scan = window.data.scan
        self.slit = np.asarray(scan.x, dtype=float)
        self.energy = np.asarray(scan.y, dtype=float)
        recorded = cut_deflector_angle(window.data, default=float("nan"))
        n_e = self.energy.size
        step = abs(float(self.energy[1] - self.energy[0])) * 1000 if n_e > 1 else 10.0
        self.form = T.Form(self.content, [
            Field("name", "Name", "str", window.unique("%s [k]" % window.filename), width=36),
            Field("gdefl", "Γ deflector angle (deg)", "float", 0.0,
                  help="The Map conversion's theta offset."),
            Field("gslit", "Γ slit angle (deg)", "float", 0.0,
                  help="The Map conversion's phi offset."),
            Field("azimuth", "Sample rotation (deg)", "float", 0.0),
            Field("defl", "This cut's deflector angle (deg)", "float",
                  0.0 if not np.isfinite(recorded) else recorded,
                  help="Read from the file when it records it."),
            Field("eoff", "Energy offset (eV)", "float", 0.0),
            Field("mode", "Output grid", "choice", "by element number",
                  ["by element number", "by resolution"]),
            Field("nk", "k points", "int", max(self.slit.size, 2)),
            Field("ne", "E points", "int", n_e),
            Field("dk", "k step (Å⁻¹)", "float", 0.01),
            Field("de", "E step (meV)", "float", round(max(step, 0.001), 3)),
            Field("radial", "Measure from Γ itself (distance from Γ)", "bool", False),
            Field("trim", "Trim to the k range every energy covers", "bool", False),
        ], on_change=self.update_note)
        self.form.pack(fill="x")
        self.source_note = ttk.Label(self.content, text=(
            "Deflector angle read from the file." if np.isfinite(recorded)
            else "This file records no deflector angle — type it."),
            wraplength=540)
        self.source_note.pack(fill="x")
        self.button("From a converted map...", self.from_converted_map)
        self.button("Convert", self.convert)
        self.update_note()

    def geometry_(self, v):
        return {"deflector_deg": v["defl"], "gamma_deflector_deg": v["gdefl"],
                "gamma_slit_deg": v["gslit"], "azimuth_deg": v["azimuth"],
                "energy_offset_eV": v["eoff"]}

    def grid_counts(self, v):
        from tools.cutk import cut_extent
        if v["mode"] == "by element number":
            return int(v["nk"]), int(v["ne"])
        lo, hi = cut_extent(self.slit, self.energy, radial=v["radial"],
                            trim=v["trim"], **self.geometry_(v))
        span = abs(float(self.energy[-1] - self.energy[0])) * 1000.0
        return (max(int(round((hi - lo) / v["dk"])) + 1, 2),
                max(int(round(span / v["de"])) + 1, 2))

    def update_note(self):
        from tools.cutk import cut_extent, cut_momenta
        try:
            v = self.form.values()
            geometry = self.geometry_(v)
            lo, hi = cut_extent(self.slit, self.energy, radial=v["radial"],
                                trim=v["trim"], **geometry)
            n_k, n_e = self.grid_counts(v)
            _k_par, k_perp = cut_momenta(self.slit, self.energy, **geometry)
        except Exception as exc:                                # noqa: BLE001
            self.say(str(exc))
            return
        perp = np.abs(k_perp)
        text = "k from %+.4g to %+.4g Å⁻¹, %d x %d points. " % (lo, hi, n_k, n_e)
        if perp.max() < 1e-6:
            text += "The cut runs through Γ."
        else:
            spread = float(perp.max() - perp.min())
            text += ("The cut runs %.4g Å⁻¹ from Γ, varying by "
                     "%.4g along it (the conversion follows the exact arc)."
                     % (perp.mean(), spread))
            if spread > 0.05:
                text += " ⚠ A visibly curved path through the zone."
        self.say(text)

    def from_converted_map(self):
        app = self.window_.app
        if app is None:
            T.info(self, "Cut k conversion", "No dataset list is available here.")
            return None
        candidates = [(label, key) for label, key in app.entries()
                      if app.kind_of(key) == "k_map"]
        if not candidates:
            T.info(self, "Cut k conversion",
                   "No converted map is in the list yet. Convert a map of this "
                   "sample first, or type Γ's angles.")
            return None
        index = T.choose(self, "Inherit Γ from a map", "Converted map:",
                         [label for label, _ in candidates])
        if index is None:
            return None
        label, key = candidates[index]
        try:
            data = app.load(key)
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Cut k conversion", "That map could not be read:\n%s" % exc)
            return None
        info = dict(getattr(data.scan, "info", {}) or {})
        try:
            theta = float(info["kconv.theta_offset_deg"])
            phi = float(info["kconv.phi_offset_deg"])
        except (KeyError, TypeError, ValueError):
            T.warning(self, "Cut k conversion",
                      "That dataset does not record a k conversion.")
            return None
        azimuth = float(info.get("kconv.azimuth_deg", 0.0))
        self.form.set("gdefl", theta)
        self.form.set("gslit", phi)
        self.form.set("azimuth", azimuth)
        mine, theirs = sample_rotations(self.window_.data), sample_rotations(data)
        shared = sorted(set(mine) & set(theirs))
        drift = {n: mine[n] - theirs[n] for n in shared if abs(mine[n] - theirs[n]) > 0.01}
        note = "Γ taken from %s." % label
        if drift:
            note += (" ⚠ The manipulator moved between the two: " +
                     ", ".join("%s by %+.3f°" % (n, d) for n, d in drift.items()))
        self.source_note.configure(text=note)
        self.update_note()
        return theta, phi, azimuth

    def convert(self):
        from tools.cutk import convert_cut, K_LABEL, K_RADIAL_LABEL
        v = self.values(self.form)
        if v is None:
            return None
        try:
            n_k, n_e = self.grid_counts(v)
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Cut k conversion", str(exc))
            return None
        settings = dict(self.geometry_(v))
        settings.update({"n_k": n_k, "n_energy": n_e, "radial": v["radial"],
                         "trim": v["trim"]})
        w = self.window_
        scan = w.data.scan
        try:
            k_axis, energy, values, report = T.with_wait_cursor(
                self, convert_cut, scan.x, scan.y, w.data.cut_frame, **settings)
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Cut k conversion", "Conversion failed:\n%s" % exc)
            return None
        parameters = dict(settings)
        parameters.update({key: report[key] for key in
                           ("k_perp", "k_perp_min", "k_perp_max", "bow",
                            "energy_drift", "k_step", "measured_fraction")
                           if key in report})
        info = {k: val for k, val in scan.info.items() if not k.startswith("kcut.")}
        result = MemoryData(
            "cut", (np.asarray(k_axis, dtype=float), np.asarray(energy, dtype=float)),
            np.asarray(values, dtype=float),
            {"x": K_RADIAL_LABEL if settings["radial"] else K_LABEL,
             "y": scan.labels.get("y", "Energy (eV)")},
            source_label=v["name"].strip() or "%s [k]" % w.filename,
            parameters=parameters, prefix="kcut",
            source_path=getattr(w.data, "path", ""), source_info=info,
            source_motors=dict(scan.fourd_info))
        w.emit(result)
        w.say("Converted to k: %dx%d, %.0f%% measured -- added to the list" % (
            values.shape[0], values.shape[1], report.get("measured_fraction", 1) * 100))
        self.close()
        return result


# --------------------------------------------------------------------------
# Arbitrary cut
# --------------------------------------------------------------------------
class ArbitraryCutDialog(ToolDialog):
    """A path of 2-6 points across the contour; the cut follows it."""

    def __init__(self, contour):
        ToolDialog.__init__(
            self, contour, "Arbitrary cut",
            "Type the path's points (x y per line, in the contour's units), "
            "or press 'Pick by clicking' and click them on the contour. The "
            "cut runs through them in order.")
        self.contour = contour
        self.points_box = PointsBox(self.content, height=6, on_change=self.redraw)
        self.points_box.pack(fill="x")
        self.form = T.Form(self.content, [
            Field("name", "Name", "str", contour.unique("%s [arb cut]" % contour.filename),
                  width=36),
            Field("separate", "One dataset per segment", "bool", False),
        ])
        self.form.pack(fill="x", pady=4)
        self.button("Pick by clicking", self.pick)
        self.button("Clear", lambda: self.points_box.set_points([]))
        self.button("Plot cut", self.run)

    def pick(self):
        self.say("Click the contour to add points (up to six); 'Plot cut' when done.")
        self.start_picking(self.contour.panel, self.points_box.add)

    def redraw(self):
        panel = self.contour.panel
        panel.clear_overlays("arbcut")
        try:
            pts = self.points_box.points()
        except ValueError as exc:
            self.say(str(exc))
            return
        if pts:
            panel.add_overlay("arbcut", "plot", [p[0] for p in pts], [p[1] for p in pts],
                              "o-", color="#00e5ff", lw=1.5)
        panel.redraw()
        if len(pts) >= 2:
            xs, ys = np.array(pts).T
            self.say("%d points, %.4g %s long." % (len(pts), float(np.sum(np.hypot(
                np.diff(xs), np.diff(ys)))), unit_of(self.contour.defl_label)))

    def close(self):
        try:
            self.contour.panel.clear_overlays("arbcut")
            self.contour.panel.redraw()
        except tk.TclError:
            pass
        ToolDialog.close(self)

    def run(self):
        from tools.analysis import arbitrary_cut
        try:
            points = self.points_box.points()
        except ValueError as exc:
            T.warning(self, "Arbitrary cut", str(exc))
            return None
        if len(points) < 2:
            T.warning(self, "Arbitrary cut", "Give at least two points.")
            return None
        if len(points) > 6:
            points = points[:6]
        v = self.values(self.form)
        if v is None:
            return None
        c = self.contour
        name = v["name"].strip() or "%s [arb cut]" % c.filename
        unit = unit_of(c.defl_label)
        distance_label = "Distance along path (%s)" % unit if unit else "Distance along path"
        try:
            distance, energy, values, joints = T.with_wait_cursor(
                self, arbitrary_cut, c.defl, c.slit, c.E, c.full_cube(), points)
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Arbitrary cut", "The cut failed:\n%s" % exc)
            return None
        info = {k: val for k, val in c.data.scan.info.items() if not k.startswith("arbcut.")}

        def make(label, d, vals, parameters):
            return MemoryData("cut", (np.asarray(d, dtype=float), np.asarray(energy, dtype=float)),
                              np.asarray(vals, dtype=float),
                              {"x": distance_label, "y": c.energy_label},
                              source_label=label, parameters=parameters, prefix="arbcut",
                              source_path=getattr(c.data, "path", ""), source_info=info,
                              source_motors=dict(c.data.scan.fourd_info))

        created = []
        if v["separate"]:
            edges = list(joints) + [float(distance[-1])]
            start = 0.0
            for i, end in enumerate(edges):
                mask = (distance >= start - 1e-9) & (distance <= end + 1e-9)
                created.append(make("%s %d" % (name, i + 1),
                                    distance[mask] - distance[mask][0], values[mask],
                                    {"points": [list(points[i]), list(points[i + 1])],
                                     "segment": i + 1}))
                start = end
        else:
            created.append(make(name, distance, values,
                                {"points": [list(p) for p in points],
                                 "joints": list(map(float, joints))}))
        for data in created:
            c.emit(data)
        c.say("Arbitrary cut: %d points -- added %d dataset(s) to the list"
              % (len(points), len(created)))
        if created:
            c.open_computed(created[0])
        return created


# --------------------------------------------------------------------------
# Fermi-surface correction
# --------------------------------------------------------------------------
class FSCorrectionDialog(ToolDialog):
    """Points along a feature that should be flat (x = angle, y = energy),
    a polynomial through them, and every angle column shifted in energy to
    straighten it. The points can also be measured on a reference (gold)."""

    def __init__(self, window):
        ToolDialog.__init__(
            self, window, "Fermi-surface correction",
            "Points along the feature that should be flat (a Fermi edge): "
            "type them (angle energy per line), click them on the image "
            "('Pick by clicking'), or fit a gold reference channel by channel "
            "('From a reference...'). The fitted curve is drawn on the image.")
        if window.fs_correction_target() is None:
            self.after(10, self.close)
            return
        self.points_box = PointsBox(self.content, height=8, on_change=self.refresh)
        self.points_box.pack(fill="x")
        self.form = T.Form(self.content, [
            Field("order", "Fit order", "int", 2,
                  help="Polynomial order; 2 (a parabola) is the MATLAB tool's."),
            Field("name", "Name", "str", window.unique("%s [FS corr]" % window.filename),
                  width=36),
        ], on_change=self.refresh)
        self.form.pack(fill="x", pady=4)
        self.button("Pick by clicking", self.pick)
        self.button("From a reference...", self.from_reference)
        self.button("Clear", lambda: self.points_box.set_points([]))
        self.button("Correct", self.correct)
        self.refresh()

    def panel(self):
        return self.window_.image_panels()[0]

    def pick(self):
        self.say("Click along the feature on the image.")
        self.start_picking(self.panel(), self.points_box.add)

    def coefficients(self):
        from tools.analysis import fit_feature
        order = int(self.form.value("order"))
        pts = self.points_box.points()
        if len(pts) < order + 1:
            return None, order, pts
        return fit_feature(pts, order=order), order, pts

    def refresh(self):
        panel = self.panel()
        panel.clear_overlays("fscorr")
        try:
            coeffs, order, pts = self.coefficients()
        except (ValueError, T.FormError) as exc:
            self.say(str(exc))
            return
        if pts:
            panel.add_overlay("fscorr", "plot", [p[0] for p in pts], [p[1] for p in pts],
                              "o", color="#00e5ff", ms=4)
        if coeffs is None:
            self.say("%d point(s); an order-%d fit needs %d." % (len(pts), order, order + 1))
        else:
            axis = np.asarray(self.window_.fs_angle_axis(), dtype=float)
            xs = np.linspace(float(axis.min()), float(axis.max()), 200)
            panel.add_overlay("fscorr", "plot", xs, np.polyval(coeffs, xs), "--",
                              color="#ff00aa", lw=1.2)
            fitted = np.polyval(coeffs, axis)
            self.say("%d points fitted; the feature spans %.4g and will be flattened."
                     % (len(pts), float(np.max(fitted) - np.min(fitted))))
        panel.redraw()

    def close(self):
        try:
            self.panel().clear_overlays("fscorr")
            self.panel().redraw()
        except (tk.TclError, IndexError, AttributeError):
            pass
        ToolDialog.close(self)

    def from_reference(self):
        app = self.window_.app
        if app is None or not app.entries():
            T.info(self, "Reference", "Load the reference measurement into the list first.")
            return None
        entries = app.entries()
        index = T.choose(self, "Fermi surface from a reference",
                         "Reference (a gold cut or map; a map is summed over "
                         "the deflector):", [label for label, _ in entries])
        if index is None:
            return None
        try:
            data = app.load(entries[index][1])
            angles, energy, frame, _a, _e = reference_frame(data)
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Reference", str(exc))
            return None
        p = T.ask_params(self, "Reference fit", [
            Field("temperature", "Temperature (K)", "float", metadata_temperature(data)),
            Field("half_width", "Neighbours (± channels)", "int", 3),
            Field("step", "Fit every Nth channel", "int", max(1, int(np.ceil(angles.size / 120.)))),
        ])
        if p is None:
            return None
        result = run_channel_fit(self, angles, energy, frame, temperature=p["temperature"],
                                 half_width=p["half_width"], step=p["step"])
        if result is None:
            return None
        angles, ef, ok = result
        if not ok.any():
            T.warning(self, "Reference", "No channel of that reference could be fitted.")
            return None
        self.points_box.set_points(list(zip(angles[ok], ef[ok])))
        self.say("%d channels fitted; E_F varies by %.3g meV across the detector."
                 % (int(ok.sum()), float(np.nanmax(ef[ok]) - np.nanmin(ef[ok])) * 1000))
        return result

    def correct(self):
        from tools.analysis import fs_correction
        try:
            coeffs, order, pts = self.coefficients()
        except (ValueError, T.FormError) as exc:
            T.warning(self, "FS correction", str(exc))
            return None
        if coeffs is None:
            T.warning(self, "FS correction",
                      "An order-%d fit needs at least %d points." % (order, order + 1))
            return None
        w = self.window_
        target = w.fs_correction_target()
        values, angle_axis, energy_axis, angle_dim, energy_dim, kind, axes_of = target
        try:
            corrected, new_energy = T.with_wait_cursor(
                self, fs_correction, values, angle_axis, energy_axis, coeffs,
                angle_dim=angle_dim, energy_dim=energy_dim)
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "FS correction", "The correction failed:\n%s" % exc)
            return None
        scan = w.data.scan
        info = {k: v for k, v in scan.info.items() if not k.startswith("fscorr.")}
        data = MemoryData(kind, axes_of(new_energy), corrected, dict(scan.labels),
                          source_label=self.form.value("name").strip() or
                          "%s [FS corr]" % w.filename,
                          parameters={"order": order,
                                      "coefficients": list(map(float, coeffs)),
                                      "points": [list(map(float, p)) for p in pts],
                                      "angle_axis": w.fs_angle_label()},
                          prefix="fscorr", source_path=getattr(w.data, "path", ""),
                          source_info=info, source_motors=dict(scan.fourd_info))
        w.emit(data, open_window=True)
        w.say("Fermi-surface correction applied (order %d) -- added to the list" % order)
        self.close()
        return data


def run_channel_fit(parent, angles, energy, frame, *, temperature,
                    fixed=("temperature",), window=(None, None), half_width=0,
                    step=1, start=None):
    """Fit every (Nth) channel's Fermi edge, with a progress window and
    Cancel (keeps what was fitted so far)."""
    from tools.fermi import fit_channels
    state = {"cancelled": False}

    def work(report):
        def progress(done, total):
            try:
                report(done / float(max(total, 1)), "channel %d of %d" % (done, total))
            except T.JobCancelled:
                state["cancelled"] = True
                return False
            return True
        return fit_channels(angles, energy, frame, half_width=half_width, step=step,
                            temperature=temperature, fixed=fixed, window=window,
                            start=start, progress=progress)

    try:
        ef, _err, ok = T.run_blocking(parent, "Fitting the edge channel by channel", work)
    except T.JobCancelled:
        return None
    except Exception as exc:                                    # noqa: BLE001
        T.warning(parent, "Fermi level", "Channel fitting failed:\n%s" % exc)
        return None
    return np.asarray(angles, dtype=float), ef, ok


# --------------------------------------------------------------------------
# Fermi level
# --------------------------------------------------------------------------
class FermiFitDialog(ToolDialog):
    """Fit the Fermi edge of the EDC summed over an angle range, within an
    energy window. Both ranges are typed (or taken from the image's box)."""

    def __init__(self, window):
        from tools.fermi import PARAMETERS, PARAMETER_LABELS
        self.params = PARAMETERS
        angles, energy, frame, a_label, e_label = reference_frame(window.data)
        ToolDialog.__init__(
            self, window, "Fermi level fitting",
            "The EDC is the data summed over the angle range; the fit runs "
            "over the energy window. Type them, or set a box on the image "
            "and press 'From the box'. Temperature is held by default (on one "
            "edge it cannot be separated from the resolution).")
        self.angles, self.energy, self.frame = angles, energy, frame
        self.a_label, self.e_label = a_label, e_label
        self.fit_result = None
        self.region = T.Form(self.content, [
            Field("a0", "%s from" % a_label, "float", float(np.min(angles))),
            Field("a1", "to", "float", float(np.max(angles))),
            Field("e0", "Fit window %s from" % e_label, "float", float(np.min(energy))),
            Field("e1", "to", "float", float(np.max(energy))),
        ], on_change=self.refresh_plot, columns=2)
        self.region.pack(fill="x")
        grid = ttk.LabelFrame(self.content, text="Parameters (value, hold, uncertainty)")
        grid.pack(fill="x", pady=4)
        self.value_vars, self.hold_vars, self.err_labels = {}, {}, {}
        for row, name in enumerate(PARAMETERS):
            ttk.Label(grid, text=PARAMETER_LABELS[name]).grid(row=row, column=0, sticky="w", padx=3)
            var = tk.StringVar(value="0")
            ttk.Entry(grid, textvariable=var, width=14).grid(row=row, column=1)
            hold = tk.BooleanVar(value=(name == "temperature"))
            ttk.Checkbutton(grid, text="hold", variable=hold).grid(row=row, column=2)
            lab = ttk.Label(grid, text="", width=16)
            lab.grid(row=row, column=3)
            self.value_vars[name], self.hold_vars[name], self.err_labels[name] = var, hold, lab
        self.value_vars["temperature"].set("%.4g" % metadata_temperature(window.data))
        self.plot = T.PlotFrame(self.content, figsize=(5.5, 3.0), toolbar=False)
        self.plot.pack(fill="both", expand=True)
        self.ax = self.plot.figure.add_subplot(111)
        self.plot.figure.subplots_adjust(left=0.14, bottom=0.18, right=0.97, top=0.95)
        adv = ttk.LabelFrame(self.content, text="Advanced")
        adv.pack(fill="x", pady=4)
        self.adv = T.Form(adv, [
            Field("cutoff", "Divide out: cut off above (kT)", "float", 4.0),
            Field("nb", "Per channel: neighbours (±)", "int", 2),
            Field("step", "Per channel: every Nth channel", "int", 1),
        ], columns=1)
        self.adv.pack(side="left")
        ab = ttk.Frame(adv)
        ab.pack(side="left", padx=6)
        ttk.Button(ab, text="Divide the Fermi cut-off out", command=self.divide_out).pack(fill="x")
        ttk.Button(ab, text="Fit E_F channel by channel", command=self.fit_by_channel).pack(fill="x")
        self.button("From the box", self.from_box)
        self.button("Estimate", self.estimate)
        self.button("Fit", self.run_fit)
        self.button("Offset energy axis", self.offset_energy_axis)
        self.estimate()

    def from_box(self):
        panel = self.window_.image_panels()[0]
        if panel.box is None:
            T.info(self, "Fermi level", "Set a box on the image first (row under it).")
            return
        x0, x1, y0, y1 = panel.box
        if self.window_.data.kind == "cut":
            for key, value in (("a0", x0), ("a1", x1), ("e0", y0), ("e1", y1)):
                self.region.set(key, value)
        else:
            self.region.set("e0", y0)
            self.region.set("e1", y1)
        self.refresh_plot()

    def edc(self):
        r = self.region.values()
        lo, hi = sorted((r["a0"], r["a1"]))
        inside = (self.angles >= lo) & (self.angles <= hi)
        if not inside.any():
            inside = np.zeros_like(self.angles, dtype=bool)
            inside[int(np.argmin(np.abs(self.angles - lo)))] = True
        return self.energy, np.nansum(self.frame[inside], axis=0)

    def window_range(self):
        r = self.region.values()
        return tuple(sorted((r["e0"], r["e1"])))

    def parameters(self):
        out = {}
        for name in self.params:
            try:
                out[name] = float(self.value_vars[name].get())
            except ValueError:
                raise T.FormError("%s: not a number" % name)
        return out

    def held(self):
        return tuple(n for n in self.params if self.hold_vars[n].get())

    def _show(self, values, errors=None):
        for name in self.params:
            self.value_vars[name].set("%.6g" % float(values[name]))
            e = (errors or {}).get(name)
            self.err_labels[name].configure(
                text="" if e is None or not np.isfinite(e) else
                ("held" if name in self.held() and e == 0 else "± %.4g" % e))

    def refresh_plot(self, model=None):
        try:
            energy, intensity = self.edc()
            lo, hi = self.window_range()
        except T.FormError as exc:
            self.say(str(exc))
            return
        inside = (energy >= lo) & (energy <= hi)
        if inside.sum() < 2:
            inside = np.ones_like(energy, dtype=bool)
        e, i = energy[inside], intensity[inside]
        self.ax.clear()
        self.ax.plot(e, i, color="#1f6f8b", lw=1.5, label="EDC")
        if self.fit_result is not None:
            self.ax.plot(e, self.fit_result.model(e), color="#bd4921", lw=1.5, label="fit")
            self.ax.axvline(self.fit_result.values["ef"], color="#2f7a3f", ls="--")
        elif model is not None:
            self.ax.plot(e, model(e), color="#bd4921", lw=1, ls=":")
        self.ax.set_xlabel(self.e_label, fontsize=8)
        self.ax.tick_params(labelsize=7)
        self.plot.draw()

    def estimate(self):
        from tools.fermi import initial_guess, fermi_edge_model
        try:
            energy, intensity = self.edc()
            lo, hi = self.window_range()
            temperature = float(self.value_vars["temperature"].get())
        except (T.FormError, ValueError) as exc:
            self.say(str(exc))
            return None
        inside = (energy >= lo) & (energy <= hi)
        try:
            guess = initial_guess(energy[inside], intensity[inside], temperature=temperature)
        except ValueError as exc:
            self.say(str(exc))
            return None
        guess["temperature"] = temperature
        self.fit_result = None
        self._show(guess)
        self.refresh_plot(model=lambda e: fermi_edge_model(e, **guess))
        self.say("Starting values estimated from the data. Press Fit.")
        return guess

    def run_fit(self):
        from tools.fermi import fit_fermi_edge
        try:
            energy, intensity = self.edc()
            start = self.parameters()
            result = fit_fermi_edge(energy, intensity, start=start, fixed=self.held(),
                                    temperature=start["temperature"],
                                    window=self.window_range())
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Fermi level", "The fit failed:\n%s" % exc)
            return None
        self.fit_result = result
        self._show(result.values, result.errors)
        self.refresh_plot()
        self.say(result.summary())
        panel = self.window_.image_panels()[0]
        panel.clear_overlays("ef")
        panel.add_overlay("ef", "axhline", result.values["ef"], color="#2f7a3f",
                          ls="--", lw=1)
        panel.redraw()
        return result

    def fit_info(self):
        from tools.fermi import PARAMETERS
        if self.fit_result is None:
            return {}
        r = self.fit_result
        info = {name: float(r.values[name]) for name in PARAMETERS}
        info.update({"%s_err" % n: float(r.errors.get(n, float("nan"))) for n in PARAMETERS})
        info["held"] = ", ".join(r.fixed) or "(nothing)"
        info["reduced_chi2"] = float(r.reduced_chi2)
        info["combined_width_eV"] = float(r.combined_width)
        info["window_eV"] = "%.6g..%.6g" % tuple(r.window)
        pair = r.correlation.get(("temperature", "resolution"))
        if pair is not None:
            info["corr_T_resolution"] = float(pair)
        return info

    def offset_energy_axis(self):
        if self.fit_result is None:
            T.info(self, "Fermi level", "Fit first.")
            return None
        return offset_energy_axis(self.window_, self.fit_result.values["ef"],
                                  self.fit_info())

    def divide_out(self):
        if self.fit_result is None:
            T.info(self, "Fermi level", "Fit first: the division uses the fitted edge.")
            return None
        try:
            cutoff = float(self.adv.value("cutoff"))
        except T.FormError as exc:
            T.warning(self, "Fermi level", str(exc))
            return None
        return divide_fermi_cutoff(self.window_, self.fit_result, cutoff, self.fit_info())

    def fit_by_channel(self):
        try:
            a = self.adv.values()
            start = self.parameters()
        except T.FormError as exc:
            T.warning(self, "Fermi level", str(exc))
            return None
        result = run_channel_fit(self, self.angles, self.energy, self.frame,
                                 temperature=start["temperature"], fixed=self.held(),
                                 window=self.window_range(), half_width=a["nb"],
                                 step=a["step"], start=start)
        if result is None:
            return None
        angles, ef, ok = result
        if not ok.any():
            self.say("No channel could be fitted.")
            return None
        panel = self.window_.image_panels()[0]
        panel.clear_overlays("efch")
        order = np.argsort(angles[ok])
        panel.add_overlay("efch", "plot", angles[ok][order], ef[ok][order], "o-",
                          color="#ff00aa", ms=2, lw=1)
        panel.redraw()
        self.say("%d of %d channels fitted; E_F varies by %.3g meV across the "
                 "detector (drawn over the image) -- use FS correction to flatten it."
                 % (int(ok.sum()), angles.size,
                    float(np.nanmax(ef[ok]) - np.nanmin(ef[ok])) * 1000))
        return result


def offset_energy_axis(window, ef, info=None):
    """List a copy of the window's dataset with E - E_F on its energy axis."""
    data = window.data
    scan = data.scan
    names = CONSTRUCTOR_AXES.get(data.kind)
    energy_name = energy_slot(data.kind)
    if names is None or energy_name is None:
        T.warning(window, "Offset energy axis", "%s has no energy axis to shift." % data.kind)
        return None
    values = full_array(data)
    axes = []
    for name in names:
        axis = np.asarray(getattr(scan, name), dtype=float)
        axes.append(axis - float(ef) if name == energy_name else axis)
    labels = dict(scan.labels)
    labels[energy_name] = "E - E_F (%s)" % (unit_of(labels.get(energy_name, "")) or "eV")
    parameters = dict(info or {})
    parameters["ef_subtracted"] = float(ef)
    result = MemoryData(data.kind, tuple(axes), values, labels,
                        source_label=window.unique("%s [E-Ef]" % window.filename),
                        parameters=parameters, prefix="fitEF",
                        source_path=getattr(data, "path", ""),
                        source_info=dict(scan.info), source_motors=dict(scan.fourd_info))
    window.emit(result)
    window.say("Energy axis shifted by %.6g eV -- added to the list" % ef)
    return result


def divide_fermi_cutoff(window, fit, cutoff_kt, info=None):
    from tools.fermi import divide_fermi
    data = window.data
    scan = data.scan
    names = CONSTRUCTOR_AXES.get(data.kind)
    energy_name = energy_slot(data.kind)
    if names is None or energy_name is None:
        return None
    values = full_array(data)
    energy_dim = ARRAY_AXES[data.kind].index(energy_name)
    energy = np.asarray(getattr(scan, energy_name), dtype=float)
    divided = divide_fermi(energy, values, ef=fit.values["ef"],
                           temperature=fit.values["temperature"],
                           resolution=fit.values["resolution"], energy_dim=energy_dim,
                           cutoff_kt=float(cutoff_kt),
                           background=(fit.values["bkg0"], fit.values["bkg1"]))
    parameters = dict(info or {})
    parameters["cutoff_kT"] = float(cutoff_kt)
    result = MemoryData(data.kind, tuple(np.asarray(getattr(scan, n), dtype=float)
                                         for n in names),
                        divided, dict(scan.labels),
                        source_label=window.unique("%s [dFD]" % window.filename),
                        parameters=parameters, prefix="dFD",
                        source_path=getattr(data, "path", ""),
                        source_info=dict(scan.info), source_motors=dict(scan.fourd_info))
    window.emit(result)
    window.say("Fermi cut-off divided out above %g kT -- added to the list" % cutoff_kt)
    return result


# --------------------------------------------------------------------------
# Data operations on the selected datasets
# --------------------------------------------------------------------------
class DataOperationsDialog(tk.Toplevel):
    """Truncate, self-normalise or compress one or more datasets of the same
    format; each result is listed with the MATLAB suffix (``_tk``,
    ``_s_nor``, ``_comb``)."""

    def __init__(self, app, datasets, master=None):
        tk.Toplevel.__init__(self, master or app.root)
        self.app = app
        self.datasets = list(datasets)
        self.title("Data operations")
        kind = self.datasets[0][1].kind
        self.kind = kind
        self.axis_names = ARRAY_AXES.get(kind, ())
        scan = self.datasets[0][1].scan
        self.axes = [np.asarray(getattr(scan, n), dtype=float) for n in self.axis_names]
        self.axis_labels = [scan.labels.get(n, n) for n in self.axis_names]
        shape = " x ".join(str(a.size) for a in self.axes)
        ttk.Label(self, text="%d dataset(s) of kind %s, %s points: %s" % (
            len(self.datasets), kind, shape,
            ", ".join(n for n, _ in self.datasets[:4])), wraplength=560).pack(
            fill="x", padx=8, pady=6)
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=8)
        captions = ["%s [%.4g .. %.4g, %d pts]" % (lab, a.min(), a.max(), a.size)
                    for lab, a in zip(self.axis_labels, self.axes)]
        page = ttk.Frame(self.tabs)
        fields = []
        for i, cap in enumerate(captions):
            fields.append(Field("lo%d" % i, cap + " from", "optfloat", None))
            fields.append(Field("hi%d" % i, "to", "optfloat", None))
        fields.append(Field("by", "Bounds are", "choice", "value", ["value", "index (1-based)"]))
        self.trunc = T.Form(page, fields, columns=1)
        self.trunc.pack(fill="x")
        self.tabs.add(page, text="Truncate")
        page = ttk.Frame(self.tabs)
        fields = [Field("along%d" % i, "along " + lab.split(" (")[0], "bool", False)
                  for i, lab in enumerate(self.axis_labels)]
        fields += [Field("w0", "Window from (empty = all)", "optfloat", None),
                   Field("w1", "Window to", "optfloat", None),
                   Field("by", "Window is", "choice", "value", ["value", "index"]),
                   Field("peak", "Normalise to the peak instead of the sum", "bool", False)]
        self.norm = T.Form(page, fields)
        self.norm.pack(fill="x")
        self.tabs.add(page, text="Self-normalise")
        page = ttk.Frame(self.tabs)
        self.comp = T.Form(page, [Field("f%d" % i, cap, "int", 1)
                                  for i, cap in enumerate(captions)])
        self.comp.pack(fill="x")
        self.tabs.add(page, text="Compress (sum blocks)")
        self.note = ttk.Label(self, text="", wraplength=560)
        self.note.pack(fill="x", padx=8)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right", padx=3)
        ttk.Button(bar, text="Apply", command=self.apply).pack(side="right", padx=3)

    def _settings(self):
        tab = self.tabs.index(self.tabs.select())
        n = len(self.axes)
        if tab == 0:
            v = self.trunc.values()
            bounds = [(v["lo%d" % i], v["hi%d" % i]) for i in range(n)]
            if all(lo is None and hi is None for lo, hi in bounds):
                raise ValueError("Set at least one bound to truncate.")
            by_index = v["by"].startswith("index")
            return ("_tk", "trunc", {"bounds": [list(b) for b in bounds],
                                     "by": "index" if by_index else "value"},
                    lambda values, axes: truncate(values, axes, bounds, by_index=by_index))
        if tab == 1:
            v = self.norm.values()
            dims = [i for i in range(n) if v["along%d" % i]]
            if not dims:
                raise ValueError("Tick the axis (or two axes) to normalise along.")
            if len(dims) > 2:
                raise ValueError("Normalise along one axis (lines) or two (planes).")
            window = (v["w0"], v["w1"])
            by_index = v["by"] == "index"
            return ("_s_nor", "snorm",
                    {"along": [self.axis_names[d] for d in dims], "window": list(window),
                     "by": v["by"], "to_peak": v["peak"]},
                    lambda values, axes: (self_normalize(values, axes, dims, window=window,
                                                         by_index=by_index,
                                                         to_peak=v["peak"]), axes))
        v = self.comp.values()
        factors = [max(1, int(v["f%d" % i])) for i in range(n)]
        if all(f == 1 for f in factors):
            raise ValueError("Set a factor above 1 on at least one axis.")
        return ("_comb", "compress", {"factors": factors},
                lambda values, axes: compress(values, axes, factors))

    def apply(self):
        try:
            suffix, prefix, settings, operation = self._settings()
        except (ValueError, T.FormError) as exc:
            T.warning(self, "Data operations", str(exc))
            return None
        created = []
        for name, data in self.datasets:
            try:
                values = full_array(data)
                axes = [np.asarray(getattr(data.scan, n), dtype=float) for n in self.axis_names]
                new_values, new_axes = T.with_wait_cursor(self, operation, values, axes)
            except Exception as exc:                            # noqa: BLE001
                T.warning(self, "Data operations", "%s: %s" % (name, exc))
                return None
            by_name = dict(zip(self.axis_names, new_axes))
            order = CONSTRUCTOR_AXES.get(data.kind, self.axis_names)
            created.append(MemoryData(
                data.kind, tuple(by_name[n] for n in order), new_values,
                dict(data.scan.labels), source_label="%s%s" % (name, suffix),
                parameters=settings, prefix=prefix, source_path=getattr(data, "path", ""),
                source_info=dict(data.scan.info), source_motors=dict(data.scan.fourd_info)))
        for data in created:
            self.app.add_dataset(data)
        self.note.configure(text="Created %d dataset(s) ending in “%s”." % (
            len(created), suffix))
        return created


# --------------------------------------------------------------------------
# Brillouin zone overlay
# --------------------------------------------------------------------------
class BrillouinZoneDialog(ToolDialog):
    """Overlay a Brillouin zone -- of a 3-D crystal cut by a plane, of a 2-D
    layer, or two layers' moire zone -- on a k-space contour."""

    PREFIX = "bz."

    def __init__(self, contour):
        ToolDialog.__init__(
            self, contour, "Brillouin zone",
            "3D: space group first (it fixes which cell parameters are free), "
            "then the cut plane (hkl normal against the conventional "
            "reciprocal cell, offset in Å⁻¹ from Γ). 2D: "
            "the layer lattice; tick moire for two layers. Nothing is drawn "
            "until Plot.")
        self.contour = contour
        info = contour.data.scan.info

        def saved(key, default):
            return info.get(self.PREFIX + key, default)
        self.form = T.Form(self.content, [
            Field("mode", "Mode", "choice", "3D crystal" if saved("mode", "3d") == "3d"
                  else "2D layer", ["3D crystal", "2D layer"]),
            Field("zone", "Zone", "choice", saved("zone", "conventional"),
                  ["conventional", "irreducible"]),
            Field("sg", "Space group no. (3D)", "int", int(saved("space_group", 1))),
            Field("a", "a (Å)", "float", float(saved("a", 3.0))),
            Field("b", "b (Å)", "float", float(saved("b", 3.0))),
            Field("c", "c (Å)", "float", float(saved("c", 3.0))),
            Field("alpha", "alpha (deg)", "float", float(saved("alpha", 90.0))),
            Field("beta", "beta (deg)", "float", float(saved("beta", 90.0))),
            Field("gamma", "gamma (deg)", "float", float(saved("gamma", 90.0))),
            Field("hkl", "Plane normal h k l (3D)", "floats",
                  [float(saved("h", 0)), float(saved("k", 0)), float(saved("l", 1))]),
            Field("offset", "Offset along normal (Å⁻¹)", "float",
                  float(saved("offset", 0.0))),
            Field("a2", "Layer a (Å) (2D)", "float", float(saved("a2", 3.0))),
            Field("b2", "Layer b (Å) (2D)", "float", float(saved("b2", 3.0))),
            Field("gamma2", "Layer gamma (deg) (2D)", "float", float(saved("gamma2", 120.0))),
            Field("azimuth", "Azimuth (deg)", "float", float(saved("azimuth_deg", 0.0)),
                  help="Rotates the drawn zone (counter-clockwise) about k = 0."),
            Field("tile", "Repeat across the field of view", "bool",
                  bool(int(saved("tile", 1)))),
            Field("moire", "Moire of two layers (2D only)", "bool",
                  bool(int(saved("moire", 0)))),
            Field("same", "Bottom layer = same lattice, twisted", "bool",
                  bool(int(saved("moire_same", 1)))),
            Field("a2b", "Bottom a (Å)", "float", float(saved("a2b", 3.0))),
            Field("b2b", "Bottom b (Å)", "float", float(saved("b2b", 3.0))),
            Field("gamma2b", "Bottom gamma (deg)", "float", float(saved("gamma2b", 120.0))),
            Field("twist", "Twist (deg)", "float", float(saved("twist_deg", 0.0))),
        ], on_change=self.refresh_info, columns=2)
        self.form.pack(fill="x")
        self.info_label = ttk.Label(self.content, text="", wraplength=560, justify="left")
        self.info_label.pack(fill="x", pady=4)
        self.button("Apply space-group constraints", self.apply_constraints)
        self.button("3D preview...", self.preview)
        self.button("Clear overlay", self.clear)
        self.button("Plot", self.plot)
        self.refresh_info()

    def v(self):
        return self.form.values()

    def apply_constraints(self):
        from tools.lattice import free_parameters
        v = self.v()
        c = free_parameters(int(v["sg"]))
        for target, source in c.mirrors.items():
            self.form.set(target, v[source])
        for name, value in c.fixed_angles.items():
            self.form.set(name, float(value))
        self.refresh_info()

    def refresh_info(self):
        from tools.spacegroups import spacegroup_info
        from tools.lattice import free_parameters
        try:
            v = self.v()
            if v["mode"] == "3D crystal":
                number = int(v["sg"])
                sg = spacegroup_info(number)
                text = ("%s (#%d) — %s, Bravais %s, point group %s\n%s\n%s"
                        % (sg.symbol, sg.number, sg.crystal_system, sg.bravais_symbol,
                           sg.point_group, free_parameters(number).note,
                           self.cut_description()))
            else:
                text = "2D layer lattice" + (" + moire" if v["moire"] else "")
        except Exception as exc:                                # noqa: BLE001
            text = str(exc)
        self.info_label.configure(text=text)

    def lattice_params(self):
        from tools.lattice import LatticeParams
        v = self.v()
        return LatticeParams(v["a"], v["b"], v["c"], v["alpha"], v["beta"], v["gamma"],
                             space_group=int(v["sg"]))

    def zone_3d(self):
        from tools.lattice import (primitive_vectors, conventional_vectors,
                                   reciprocal_vectors, point_group_operations)
        from tools.bz3d import wigner_seitz_cell, irreducible_wedge
        params = self.lattice_params()
        b = reciprocal_vectors(primitive_vectors(params))
        b_conv = reciprocal_vectors(conventional_vectors(params))
        zone = wigner_seitz_cell(b)
        if self.v()["zone"] == "irreducible":
            zone = irreducible_wedge(zone, point_group_operations(params))
        return zone, b, b_conv

    def cut_geometry(self):
        from tools.bz3d import plane_cut
        v = self.v()
        if len(v["hkl"]) != 3:
            raise ValueError("the plane normal needs three numbers h k l")
        zone, b, b_conv = self.zone_3d()
        h, k, l = v["hkl"]
        normal = h * b_conv[0] + k * b_conv[1] + l * b_conv[2]
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            raise ValueError("the cut plane normal (h, k, l) cannot be (0, 0, 0)")
        normal = normal / length
        point = normal * float(v["offset"])
        u, vv = plane_cut(zone.faces, normal, point)
        return zone, b, b_conv, normal, point, u, vv

    def cut_description(self):
        try:
            _zone, _b, _bc, _n, _p, u, _v = self.cut_geometry()
        except Exception as exc:                                # noqa: BLE001
            return str(exc)
        return ("the plane misses the zone" if u is None
                else "the cut is a %d-sided polygon" % (len(u) - 1))

    @staticmethod
    def rotate(x, y, deg):
        r = np.radians(deg)
        c, s = np.cos(r), np.sin(r)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return x * c - y * s, x * s + y * c

    def view_window(self):
        kx, ky = self.contour.defl, self.contour.slit
        half = max(float(np.hypot(np.max(np.abs(kx)), np.max(np.abs(ky)))), 1e-6)
        return (-half, half), (-half, half)

    def zone_2d(self, a, b_len, gamma_deg, twist=0.0):
        from tools.bz2d import (reciprocal_vectors_2d, wigner_seitz_cell_2d, tile_2d,
                                lattice_point_group_2d, irreducible_cell_2d)
        v = self.v()
        a1 = np.array([a, 0.0])
        a2 = np.array([b_len * np.cos(np.radians(gamma_deg)),
                       b_len * np.sin(np.radians(gamma_deg))])
        if twist:
            r = np.radians(twist)
            rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
            a1, a2 = rot.dot(a1), rot.dot(a2)
        g1, g2 = reciprocal_vectors_2d(a1, a2)
        polygon = wigner_seitz_cell_2d(g1, g2)
        return polygon, g1, g2, v

    def drawn(self, polygon, g1, g2):
        from tools.bz2d import tile_2d, lattice_point_group_2d, irreducible_cell_2d
        v = self.v()
        if v["zone"] == "irreducible":
            polygon = irreducible_cell_2d(polygon, lattice_point_group_2d(g1, g2))
        if v["tile"]:
            xr, yr = self.view_window()
            tiles = tile_2d(polygon, g1, g2, xr, yr)
        else:
            tiles = [polygon]
        return [self.rotate(t[:, 0], t[:, 1], v["azimuth"]) for t in tiles]

    def draw_layers(self, layers):
        panel = self.contour.panel
        panel.clear_overlays("bz")
        for polygons, color in layers:
            for x, y in polygons:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                panel.add_overlay("bz", "plot", x, y, "-", color=color, lw=1.2)
        panel.redraw()

    def plot(self):
        try:
            v = self.v()
            if v["mode"] == "3D crystal":
                message = self._plot_3d(v)
            elif v["moire"]:
                message = self._plot_moire(v)
            else:
                polygon, g1, g2, _ = self.zone_2d(v["a2"], v["b2"], v["gamma2"])
                polys = self.drawn(polygon, g1, g2)
                self.draw_layers([(polys, "#39c2ff")])
                message = "%s zone: %d polygon(s) drawn." % (v["zone"], len(polys))
        except Exception as exc:                                # noqa: BLE001
            self.clear()
            self.say("Could not plot: %s" % exc)
            return None
        self.say(message)
        self.save_settings(v)
        return message

    def _plot_3d(self, v):
        from tools.bz3d import tile_and_cut
        zone, b, _bc, normal, point, u, vv = self.cut_geometry()
        if v["tile"]:
            ur, vr = self.view_window()
            cuts = tile_and_cut(zone, b, normal, point, ur, vr)
        else:
            cuts = [] if u is None else [(u, vv)]
        if not cuts:
            raise ValueError("the cut plane does not meet the zone -- reduce the "
                             "offset, or check the (h k l) normal")
        polys = [self.rotate(cu, cv, v["azimuth"]) for cu, cv in cuts]
        self.draw_layers([(polys, "#39c2ff")])
        return "%s zone: %d polygon(s) drawn on the contour." % (v["zone"], len(polys))

    def _plot_moire(self, v):
        from tools.bz2d import reciprocal_vectors_2d, wigner_seitz_cell_2d
        from tools.moire import moire_bz
        top, g1t, g2t, _ = self.zone_2d(v["a2"], v["b2"], v["gamma2"])
        if v["same"]:
            ab, bb, gb = v["a2"], v["b2"], v["gamma2"]
        else:
            ab, bb, gb = v["a2b"], v["b2b"], v["gamma2b"]
        bot, g1b, g2b, _ = self.zone_2d(ab, bb, gb, twist=v["twist"])
        gm1, gm2, moire_polygon = moire_bz(g1t, g2t, g1b, g2b)
        am1, _am2 = reciprocal_vectors_2d(gm1, gm2)
        layers = [(self.drawn(top, g1t, g2t), "#39c2ff"),
                  (self.drawn(bot, g1b, g2b), "#ff9c39"),
                  (self.drawn(moire_polygon, gm1, gm2), "#ff3ce0")]
        self.draw_layers(layers)
        return ("moire real-space period ≈ %.4g Å (top blue, bottom "
                "orange, moire magenta)" % float(np.linalg.norm(am1)))

    def clear(self):
        self.contour.panel.clear_overlays("bz")
        self.contour.panel.redraw()

    def save_settings(self, v):
        info = self.contour.data.scan.info
        h, k, l = (list(v["hkl"]) + [0, 0, 1])[:3]
        values = {"mode": "3d" if v["mode"] == "3D crystal" else "2d",
                  "zone": v["zone"], "space_group": int(v["sg"]), "a": v["a"],
                  "b": v["b"], "c": v["c"], "alpha": v["alpha"], "beta": v["beta"],
                  "gamma": v["gamma"], "h": h, "k": k, "l": l, "offset": v["offset"],
                  "a2": v["a2"], "b2": v["b2"], "gamma2": v["gamma2"],
                  "azimuth_deg": v["azimuth"], "tile": int(v["tile"]),
                  "moire": int(v["moire"]), "moire_same": int(v["same"]),
                  "a2b": v["a2b"], "b2b": v["b2b"], "gamma2b": v["gamma2b"],
                  "twist_deg": v["twist"]}
        for key, value in values.items():
            info[self.PREFIX + key] = value

    def preview(self):
        """The zone and the cut plane in 3-D (matplotlib's own 3-D axes:
        drag to turn)."""
        try:
            zone, _b, _bc, normal, point, u, vv = self.cut_geometry()
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "3D preview", str(exc))
            return None
        from mpl_toolkits.mplot3d import Axes3D                  # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from tools.bz3d import plane_basis
        win = tk.Toplevel(self)
        win.title("Brillouin zone and cut plane")
        plot = T.PlotFrame(win, figsize=(6, 6))
        plot.pack(fill="both", expand=True)
        ax = plot.figure.add_subplot(111, projection="3d")
        for face in zone.faces:
            f = np.asarray(face)
            ax.plot(np.append(f[:, 0], f[0, 0]), np.append(f[:, 1], f[0, 1]),
                    np.append(f[:, 2], f[0, 2]), color="#39c2ff", lw=0.8)
        if u is not None:
            e1, e2 = plane_basis(normal)[:2]
            pts = [point + a * e1 + b * e2 for a, b in zip(u, vv)]
            ax.add_collection3d(Poly3DCollection([pts], alpha=0.35, facecolor="#ff9c39"))
        extent = float(np.max(np.abs(zone.vertices)))
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-extent, extent)
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        ax.set_zlabel("kz")
        plot.draw()
        return win
