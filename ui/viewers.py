"""
ui/viewers.py
=============
The per-dataset viewer windows (tkinter + matplotlib).

- :class:`CutWindow` (``cut``): the E-vs-k spectrum with EDC / MDC.
- :class:`ContourWindow` (``map``, ``k_map``, ``kz_map``, ``kz_map_k``): the
  constant-energy contour; **Deflector cut** / **Slit cut** open the two
  orthogonal cuts (:class:`MapCutWindow`). The three share one cursor point
  in the cube: moving it in any of them re-slices the others.
- :class:`SpatialScanWindow` (``spem_4d`` / ``spem_1d``): the real-space map
  and the spectrum at the cursor, with region integration both ways.
- :class:`FrameWindow`: a frozen image popped out of any of them.
- Curves (EDC, MDC, spin EDC) open in :class:`ui.curves.CurveWindow`.

Every viewer has a menu bar: **Functions** (what this window can do, by
section), **Slice** (save the slice on screen to the list, pop it out) and
**Export** (image / data of the panel).

``app`` is the launcher (:mod:`ARPES_viewer`); a viewer uses it to list
what it computes (``app.add_dataset``), to open a new viewer
(``app.open_data``), and to reach the other listed datasets
(``app.entries`` / ``app.load``). A viewer also works with ``app=None``.
"""
import os

import numpy as np
import tkinter as tk
from tkinter import ttk

from loader.nxs_file import CUBE_KINDS, CURVE_KINDS, MOMENTUM_KINDS
from ui import tkbase as T
from ui.data import MemoryData, curve_dataset, unit_of


class ViewerWindow(tk.Toplevel):
    """Scaffolding shared by every viewer: title, menus, colormap row,
    status line, list hooks, and closing (with the windows it owns)."""

    def __init__(self, app, data, filename, colormap=T.DEFAULT_COLORMAP,
                 flip=False, master=None):
        tk.Toplevel.__init__(self, master or (app.root if app is not None else None))
        self.app = app
        self.data = data
        self.filename = filename
        self.colormap, self.flip = colormap, flip
        self.children_windows = []
        self.title("%s  [%s]" % (filename, data.kind))
        self.geometry("1000x820")
        self.protocol("WM_DELETE_WINDOW", self.close)

        self.menubar = tk.Menu(self)
        self.configure(menu=self.menubar)
        self.functions_menu = tk.Menu(self.menubar, tearoff=0)
        self.slice_menu = tk.Menu(self.menubar, tearoff=0)
        self.export_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Functions", menu=self.functions_menu)
        self.menubar.add_cascade(label="Slice", menu=self.slice_menu)
        self.menubar.add_cascade(label="Export", menu=self.export_menu)
        self._sections = []
        self.slice_menu.add_command(label="Save slice to the main list...",
                                    command=self.save_slice_to_list)
        self.slice_menu.add_command(label="Open slice in a new window",
                                    command=self.pop_out_slice)
        self.slice_menu.add_command(label="Slice as a figure...",
                                    command=self.slice_to_figure)
        self.export_menu.add_command(label="Export image (PNG/PDF/SVG)...",
                                     command=lambda: self.export("image"))
        self.export_menu.add_command(label="Export data (text matrix / npz)...",
                                     command=lambda: self.export("data"))

        self.top = ttk.Frame(self)
        self.top.pack(side="top", fill="x")
        self.body = ttk.Frame(self)
        self.body.pack(side="top", fill="both", expand=True)
        self.status = ttk.Label(self, text="", anchor="w", relief="sunken")
        self.status.pack(side="bottom", fill="x")

    # -- the top row ------------------------------------------------------------
    def build_colormap_row(self, extra=()):
        for text, command in extra:
            ttk.Button(self.top, text=text, command=command).pack(side="left", padx=3, pady=2)
        ttk.Label(self.top, text="Colormap").pack(side="left", padx=(10, 2))
        self.cmap_var = tk.StringVar(value=self.colormap)
        combo = ttk.Combobox(self.top, textvariable=self.cmap_var, width=14,
                             state="readonly", values=T.COLORMAP_NAMES)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e: self._cmap_changed())
        self.flip_var = tk.BooleanVar(value=self.flip)
        ttk.Checkbutton(self.top, text="Flip", variable=self.flip_var,
                        command=self._cmap_changed).pack(side="left", padx=3)

    def _cmap_changed(self):
        self.set_colormap(self.cmap_var.get(), self.flip_var.get())

    def set_colormap(self, name, flip):
        self.colormap, self.flip = name, bool(flip)
        if hasattr(self, "cmap_var"):
            self.cmap_var.set(name)
            self.flip_var.set(bool(flip))
        for panel in self.image_panels():
            panel.set_colormap(name, flip)

    def add_function(self, label, command, section=None, enabled=True, reason=""):
        """One entry of the Functions menu; a disabled one says why when
        chosen."""
        if section and section not in self._sections:
            if self._sections:
                self.functions_menu.add_separator()
            self.functions_menu.add_command(label="— %s —" % section,
                                            state="disabled")
            self._sections.append(section)
        if enabled:
            self.functions_menu.add_command(label=label, command=command)
        else:
            self.functions_menu.add_command(
                label=label + "  (not available)",
                command=lambda: T.info(self, label, "Not available here: " + reason))

    def say(self, text):
        try:
            self.status.configure(text=text)
        except tk.TclError:
            pass

    # -- for subclasses ---------------------------------------------------------
    def image_panels(self):
        return ()

    def exportable_panels(self):
        """``(suffix, array (x, y), x_axis, y_axis, x_label, y_label, panel)``"""
        return []

    def slice_label(self):
        return ""

    # -- list hooks -------------------------------------------------------------------
    def existing_names(self):
        return list(self.app.names()) if self.app is not None else []

    def unique(self, name):
        return T.unique_name(name, self.existing_names())

    def emit(self, data, open_window=False):
        """A dataset computed here: to the list (and auto-saved)."""
        if self.app is not None:
            self.app.add_dataset(data)
        if open_window:
            self.open_computed(data)
        return data

    def open_computed(self, data):
        if self.app is not None:
            return self.app.open_data(data)
        window = open_viewer(None, data, getattr(data, "source_label", "computed"),
                             self.colormap, self.flip, master=self)
        if window is not None:
            self.children_windows.append(window)
        return window

    def adopt(self, window):
        """A dialog or window belonging to this viewer: closed with it."""
        self.children_windows.append(window)
        return window

    # -- the slice on screen -----------------------------------------------------------
    def _choose_panel(self, title):
        panels = self.exportable_panels()
        if not panels:
            T.warning(self, title, "Nothing displayed yet.")
            return None
        if len(panels) == 1:
            return panels[0]
        index = T.choose(self, title, "Which panel?", [p[0] for p in panels])
        return None if index is None else panels[index]

    def save_slice_to_list(self):
        entry = self._choose_panel("Save slice")
        if entry is None:
            return None
        suffix, array, xaxis, yaxis, xlabel, ylabel = entry[:6]
        label = self.slice_label()
        default = self.unique("%s [%s]%s" % (self.filename, suffix,
                                            (" " + label) if label else ""))
        name = T.ask_string(self, "Save slice", "Name for the slice:", default)
        if not name:
            return None
        data = MemoryData(
            "cut", (np.asarray(xaxis, dtype=float), np.asarray(yaxis, dtype=float)),
            np.array(array, dtype=float, copy=True), {"x": xlabel, "y": ylabel},
            source_label=name,
            parameters={"source": self.filename, "panel": suffix,
                        "position": label or "n/a"},
            prefix="slice", source_path=getattr(self.data, "path", ""),
            source_info=dict(getattr(self.data.scan, "info", {})))
        self.emit(data)
        self.say("Saved slice as %s" % name)
        return data

    def pop_out_slice(self):
        entry = self._choose_panel("Pop out")
        if entry is None:
            return None
        suffix, array, xaxis, yaxis, xlabel, ylabel = entry[:6]
        label = self.slice_label()
        window = FrameWindow(self, "%s · %s%s" % (self.filename, suffix,
                                                       (" · " + label) if label else ""),
                             array, xaxis, yaxis, xlabel, ylabel,
                             self.colormap, self.flip)
        return window

    def slice_to_figure(self):
        entry = self._choose_panel("Figure")
        if entry is None:
            return None
        from ui.figure import FigureWindow, Panel
        suffix, array, xaxis, yaxis, xlabel, ylabel = entry[:6]
        panel_widget = entry[6] if len(entry) > 6 else None
        levels = gamma = None
        if panel_widget is not None and panel_widget.values is not None:
            levels = panel_widget.levels(panel_widget.values)
            gamma = panel_widget.gamma()
        panel = Panel.image(array, xaxis, yaxis, xlabel, ylabel,
                            title=("%s %s" % (suffix, self.slice_label())).strip(),
                            cmap=self.colormap, flip=self.flip, levels=levels,
                            gamma=gamma)
        return FigureWindow(self, [panel], "Figure — %s" % self.filename,
                            app=self.app)

    def export(self, what):
        entry = self._choose_panel("Export")
        if entry is None or len(entry) < 7 or entry[6] is None:
            return None
        panel = entry[6]
        stem = T.safe_stem("%s_%s" % (self.filename, entry[0]))
        path = (panel.export_image(self, stem) if what == "image"
                else panel.export_data(self, stem))
        if path:
            self.say("Wrote %s" % path)
        return path

    # -- EDC / MDC to the list ---------------------------------------------------------
    def wire_curves(self, panel):
        panel.curve_to_list_hook = self.curve_to_list

    def curve_to_list(self, payload):
        kind = "edc" if T.is_energy_label(payload["x_label"]) else "mdc"
        word = kind.upper()
        width = payload["half_width"]
        suffix = "%s @ %.4g" % (word, payload["position"]) + (
            " ±%g" % width if width > 0 else "")
        label = self.slice_label()
        name = self.unique("%s [%s]%s" % (self.filename, suffix,
                                          (" " + label) if label else ""))
        parameters = {"curve": word, "position": payload["position"],
                      "position_axis": payload.get("position_label") or "position",
                      "half_width": width, "points_summed": payload["n_summed"],
                      "slice": label or ""}
        data = curve_dataset(
            kind, payload["x"], np.asarray(payload["y"], dtype=float)[:, None],
            [word], x_label=payload["x_label"],
            value_label=("Intensity (sum of %d)" % payload["n_summed"]
                         if payload["n_summed"] > 1 else "Intensity"),
            name=name, step="take_" + kind, parameters=parameters,
            source_info=dict(getattr(self.data.scan, "info", {}) or {}),
            source=self.filename, source_path=getattr(self.data, "path", ""))
        self.emit(data)
        self.say("Added “%s” to the list" % name)
        return data

    # -- closing -------------------------------------------------------------------------
    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for window in list(self.children_windows):
            try:
                if window.winfo_exists() and not isinstance(window, FrameWindow):
                    if hasattr(window, "close"):
                        window.close()
                    else:
                        window.destroy()
            except tk.TclError:
                pass
        try:
            self.data.close()
        except Exception:                                  # noqa: BLE001
            pass
        if self.app is not None:
            self.app.forget_viewer(self)
        try:
            self.destroy()
        except tk.TclError:
            pass
        self.drop_arrays()

    def drop_arrays(self):
        """Let go of the arrays this window holds (its dataset, the slices
        on screen, any copy it made), so a closed window that something
        still points at -- a pending callback, a reference cycle -- does
        not keep them in memory."""
        for panel in self.image_panels():
            try:
                panel.release()
            except Exception:                              # noqa: BLE001
                pass
        for name in ("_full", "cube", "_map"):
            if getattr(self, name, None) is not None:
                setattr(self, name, None)

    # -- shared analysis entry points (implemented in ui.analysis) ------------------------
    def open_fermi_fit(self):
        from ui.analysis import FermiFitDialog
        try:
            return self.adopt(FermiFitDialog(self))
        except ValueError as exc:
            T.warning(self, "Fermi level", str(exc))
            return None

    def open_fs_correction(self):
        from ui.analysis import FSCorrectionDialog
        return self.adopt(FSCorrectionDialog(self))


# --------------------------------------------------------------------------
# A frozen image
# --------------------------------------------------------------------------
class FrameWindow(ViewerWindow):
    """A copy of one slice, in a window of its own, to compare side by side."""

    def __init__(self, owner, title, array, x, y, x_label, y_label,
                 colormap=T.DEFAULT_COLORMAP, flip=False):
        data = MemoryData("cut", (np.asarray(x, dtype=float), np.asarray(y, dtype=float)),
                          np.array(array, dtype=float, copy=True),
                          {"x": x_label, "y": y_label}, source_label=title,
                          source_info=dict(getattr(owner.data.scan, "info", {}) or {}))
        ViewerWindow.__init__(self, owner.app, data, title, colormap, flip,
                              master=owner)
        self.title(title)
        self.geometry("900x760")
        self.build_colormap_row()
        self.panel = T.ImagePanel(self.body, title, curves=True, cmap=colormap,
                                  flip=flip)
        self.panel.pack(fill="both", expand=True)
        self.wire_curves(self.panel)
        self.panel.set_data(data.scan.value, data.scan.x, data.scan.y,
                            x_label, y_label)

    def image_panels(self):
        return (self.panel,)

    def exportable_panels(self):
        s = self.data.scan
        return [("Slice", s.value, s.x, s.y, s.labels["x"], s.labels["y"], self.panel)]


# --------------------------------------------------------------------------
# A single cut
# --------------------------------------------------------------------------
class CutWindow(ViewerWindow):
    """One E-vs-k (or E-vs-angle, or E-vs-distance) spectrum."""

    def __init__(self, app, data, filename, colormap=T.DEFAULT_COLORMAP,
                 flip=False, master=None):
        ViewerWindow.__init__(self, app, data, filename, colormap, flip, master)
        scan = data.scan
        from ui.analysis import (cut_k_conversion_blocked, momentum_cut_reason)
        self.add_function("Fermi level...", self.open_fermi_fit, section="Analysis")
        reason = momentum_cut_reason(data)
        self.add_function("MDC / EDC fit...", self.open_mdc_edc_fit,
                          enabled=not reason, reason=reason)
        self.add_function("Stack plot (EDC / MDC)...", self.open_stack,
                          section="Analysis")
        self.add_function("FS correction...", self.open_fs_correction,
                          section="Data operations")
        blocked = cut_k_conversion_blocked(data)
        self.add_function("Cut k conversion...", self.open_cut_k_conversion,
                          enabled=not blocked, reason=blocked or "")
        self.add_function("Cut arithmetic (this cut = A)...", self.open_cut_arithmetic)
        self.add_function("De-grid...", self.open_degrid)
        self.add_function("Process (smooth, derivative, curvature...)...",
                          self.open_process)
        self.build_colormap_row()

        title = ("%s vs %s" % (scan.labels.get("y", "E"), scan.labels.get("x", "k"))
                 if scan.labels.get("x") else "E vs k (single cut)")
        self.panel = T.ImagePanel(self.body, title, curves=True, cmap=colormap,
                                  flip=flip)
        self.panel.pack(fill="both", expand=True)
        self.wire_curves(self.panel)
        self.panel.set_data(np.asarray(data.cut_frame, dtype=float), scan.x, scan.y,
                            scan.labels.get("x", "x"), scan.labels.get("y", "E"))
        joints = scan.info.get("arbcut.joints")
        if joints is not None:
            for position in np.atleast_1d(joints):
                self.panel.add_overlay("joints", "axvline", float(position),
                                       color="#2f7a3f", ls="--", lw=1.5)
            self.panel.redraw()
        self.say("Click the image (toolbar zoom/pan off) or type x / y to move "
                 "the cursor; EDC / MDC follow it. Functions has the tools.")

    def image_panels(self):
        return (self.panel,)

    def exportable_panels(self):
        s = self.data.scan
        return [("Cut", self.panel.values, self.panel.x, self.panel.y,
                 s.labels.get("x", "k"), s.labels.get("y", "E"), self.panel)]

    # FS correction: the cut itself
    def fs_angle_axis(self):
        return self.data.scan.x

    def fs_angle_label(self):
        return self.data.scan.labels.get("x", "angle")

    def fs_correction_target(self):
        scan = self.data.scan
        return (np.asarray(self.data.cut_frame, dtype=float), scan.x, scan.y,
                0, 1, "cut",
                lambda new_energy: (np.asarray(scan.x, dtype=float), new_energy))

    def open_cut_k_conversion(self):
        from ui.analysis import CutKConversionDialog
        return self.adopt(CutKConversionDialog(self))

    def open_mdc_edc_fit(self):
        from ui.fit import FitPanel
        return self.adopt(FitPanel(self.app, self.data, self.filename, master=self,
                                   colormap=self.colormap, flip=self.flip))

    def open_stack(self):
        from ui.process import StackWindow
        scan = self.data.scan
        return self.adopt(StackWindow(self, np.asarray(self.data.cut_frame, dtype=float),
                                      (scan.x, scan.y),
                                      (scan.labels.get("x", "x"), scan.labels.get("y", "y")),
                                      self.filename))

    def open_cut_arithmetic(self):
        from ui.cutops import CutArithmeticDialog
        if self.app is None:
            T.info(self, "Cut arithmetic", "No dataset list to choose B from.")
            return None
        entries = [(label, key) for label, key in self.app.entries()
                   if self.app.kind_of(key) == "cut" and key != getattr(self, "list_key", None)]
        if not entries:
            T.info(self, "Cut arithmetic", "There is no other cut in the list.")
            return None
        index = T.choose(self, "Cut arithmetic: choose B",
                         "A is “%s”. Choose the other cut (B):" % self.filename,
                         [label for label, _ in entries])
        if index is None:
            return None
        label, key = entries[index]
        try:
            other = self.app.load(key)
        except Exception as exc:                           # noqa: BLE001
            T.warning(self, "Cut arithmetic", "Could not read it:\n%s" % exc)
            return None
        try:
            dialog = CutArithmeticDialog(self.app, (self.filename, self.data),
                                         (label, other), master=self,
                                         colormap=self.colormap, flip=self.flip,
                                         region_source=lambda: self.panel.box)
        except ValueError as exc:
            T.info(self, "Cut arithmetic", str(exc))
            return None
        return self.adopt(dialog)

    def open_degrid(self):
        from ui.degrid import open_degrid_for
        return open_degrid_for(self, self.data)

    def open_process(self):
        from ui.process import ProcessDialog
        return self.adopt(ProcessDialog(self.app, [(self.filename, self.data)],
                                        master=self, colormap=self.colormap,
                                        flip=self.flip))


# --------------------------------------------------------------------------
# A cube: the contour, and its two cuts
# --------------------------------------------------------------------------
class ContourWindow(ViewerWindow):
    """The constant-energy contour of a map (or k-map / kz map).

    The contour, its **slit cut** and its **deflector cut** share one point
    in the cube (deflector, slit, energy): the contour's cursor is (deflector,
    slit) and its slider the energy; the slit cut's cursor is (slit, energy)
    and its slider the deflector; the deflector cut the other way round.
    Moving any of them moves the point everywhere and re-slices the others.
    """

    def __init__(self, app, data, filename, colormap=T.DEFAULT_COLORMAP,
                 flip=False, master=None):
        ViewerWindow.__init__(self, app, data, filename, colormap, flip, master)
        self.defl, self.slit, self.E, self.cube = data.angle_cube
        self.defl = np.asarray(self.defl, dtype=float)
        self.slit = np.asarray(self.slit, dtype=float)
        self.E = np.asarray(self.E, dtype=float)
        labels = data.scan.labels
        self.defl_label = labels.get("x", "angle (deflector)")
        self.slit_label = labels.get("k", "angle (along slit)")
        self.energy_label = labels.get("z", "Energy (eV)")
        self.cut_windows = {}
        self._full = None
        kind = data.kind

        self.add_function("Arbitrary cut...", self.open_arbitrary_cut,
                          section="Data operations")
        if kind not in MOMENTUM_KINDS and kind != "kz_map":
            self.add_function("Map k conversion...", self.open_k_conversion)
        if kind == "kz_map":
            self.add_function("kz map processing...", self.open_kz_processing)
            self.add_function("kz -> momentum...", self.open_kz_conversion)
        if kind in ("map", "kz_map"):
            self.add_function("De-grid map...", self.open_degrid)
        self.add_function("Process the cube (3-D)...", self.open_volume_process)
        self.add_function("3D view...", self.open_volume_view, section="Visualization")
        if kind == "k_map":
            self.add_function("Brillouin zone...", self.open_brillouin_zone)
        self.add_function("Slice series figure...", self.open_slice_figure)

        self.build_colormap_row((("Deflector cut", lambda: self.open_cut("deflector")),
                                 ("Slit cut", lambda: self.open_cut("slit"))))
        self.e_control = T.SliceControl(self.body, "Energy: %s" % self.energy_label,
                                        on_change=self._energy_moved,
                                        color=T.AXIS_COLORS["energy"])
        self.e_control.pack(fill="x")
        self.panel = T.ImagePanel(
            self.body, "Constant-E contour: %s vs %s" % (self.defl_label, self.slit_label),
            curves=True, cmap=colormap, flip=flip, on_cursor=self._contour_cursor,
            edc_color=T.AXIS_COLORS["defl"], mdc_color=T.AXIS_COLORS["slit"])
        self.panel.pack(fill="both", expand=True)
        if kind in MOMENTUM_KINDS or unit_of(self.defl_label) == unit_of(self.slit_label):
            self.panel.equal_var.set(True)      # kx-ky (or angle-angle): same scale
        self.wire_curves(self.panel)
        self.e_control.configure_axis(self.E)
        self.point = [float(self.defl[len(self.defl) // 2]),
                      float(self.slit[len(self.slit) // 2]),
                      float(self.E[self.e_control.index])]
        self.refresh_contour()
        self.say("Deflector cut / Slit cut open the orthogonal cuts; the cursor "
                 "is shared. Click (zoom/pan off) or type to move it.")

    def image_panels(self):
        return (self.panel,)

    def full_cube(self):
        """The whole cube in memory (a lazy one is read once)."""
        if self._full is None:
            self._full = T.with_wait_cursor(self, lambda: np.asarray(self.cube, dtype=float))
        return self._full

    def _cube(self):
        return self._full if self._full is not None else self.cube

    # -- slicing ----------------------------------------------------------------
    def refresh_contour(self):
        idxs = self.e_control.indices()
        frame = self.e_control.combine(np.asarray(self._cube()[:, :, idxs[0]:idxs[-1] + 1]), 2)
        self.panel.cursor = (self.point[0], self.point[1])
        self.panel.set_data(frame, self.defl, self.slit, self.defl_label,
                            self.slit_label)

    def slice_label(self):
        i = self.e_control.index
        return "%.4g eV (ind %d)" % (self.E[i], i + 1)

    def exportable_panels(self):
        return [("ConstE_contour", self.panel.values, self.panel.x, self.panel.y,
                 self.defl_label, self.slit_label, self.panel)]

    # -- the shared point ----------------------------------------------------------
    def set_point(self, defl=None, slit=None, energy=None, source=None):
        if defl is not None:
            self.point[0] = float(defl)
        if slit is not None:
            self.point[1] = float(slit)
        if energy is not None:
            self.point[2] = float(energy)
        if source is not self:
            if energy is not None:
                self.e_control.set_position(self.point[2], notify=False)
                self.refresh_contour()
            else:
                self.panel.set_cursor(self.point[0], self.point[1], notify=False)
        for which, window in list(self.cut_windows.items()):
            if window is not source:
                window.follow_point(self.point)

    def _contour_cursor(self, panel, x, y):
        self.set_point(defl=x, slit=y, source=self)

    def _energy_moved(self):
        self.point[2] = float(self.E[self.e_control.index])
        self.refresh_contour()
        for window in self.cut_windows.values():
            window.follow_point(self.point)

    # -- the cuts ----------------------------------------------------------------------
    def open_cut(self, which):
        window = self.cut_windows.get(which)
        if window is not None and window.winfo_exists():
            window.lift()
            return window
        window = MapCutWindow(self, which)
        self.cut_windows[which] = window
        return window

    def cut_closed(self, which):
        self.cut_windows.pop(which, None)
        if not self.cut_windows and getattr(self, "hidden", False):
            self.close()

    def show_map(self):
        self.hidden = False
        self.deiconify()
        self.lift()

    def close(self):
        for window in list(self.cut_windows.values()):
            try:
                window.close()
            except tk.TclError:
                pass
        self.cut_windows.clear()
        ViewerWindow.close(self)

    # -- tools ---------------------------------------------------------------------------
    def first_axis_is_an_angle(self):
        from loader import registry
        return registry.role_is_angle(self.data.scan)

    def readout_position(self):
        return self.point[0], self.point[1]

    def open_k_conversion(self):
        from ui.analysis import KConversionDialog
        if not self.first_axis_is_an_angle():
            role = self.data.scan.info.get("axis0.role", "not an angle")
            T.info(self, "k conversion",
                   "This map's first axis was loaded as “%s”, not an "
                   "emission angle, so converting it to momentum would not mean "
                   "anything.\n\nIf that is wrong, re-open the file from the "
                   "loader and set the first axis to an angle." % role)
            return None
        return self.adopt(KConversionDialog(self))

    def open_arbitrary_cut(self):
        from ui.analysis import ArbitraryCutDialog
        return self.adopt(ArbitraryCutDialog(self))

    def open_kz_processing(self):
        from ui.kzmap import KzMapProcessDialog
        return self.adopt(KzMapProcessDialog(self))

    def open_kz_conversion(self):
        from ui.kzconv import KzConversionDialog
        return self.adopt(KzConversionDialog(self))

    def open_degrid(self):
        from ui.degrid import open_degrid_for
        return open_degrid_for(self, self.data)

    def open_brillouin_zone(self):
        from ui.analysis import BrillouinZoneDialog
        return self.adopt(BrillouinZoneDialog(self))

    def open_volume_process(self):
        from ui.volume import VolumeProcessDialog
        return self.adopt(VolumeProcessDialog(self.app, [(self.filename, self.data)],
                                              master=self, colormap=self.colormap,
                                              flip=self.flip))

    def open_volume_view(self):
        from ui.volume import open_volume_view
        return open_volume_view(self, self.app, self.data, self.filename,
                                self.colormap, self.flip)

    def open_slice_figure(self):
        from ui.figure import open_slice_series
        return open_slice_series(self)


class MapCutWindow(ViewerWindow):
    """One orthogonal cut through a map: ``which`` = ``"slit"`` (slit angle
    vs energy, at / integrated around the cursor's deflector angle) or
    ``"deflector"`` (deflector angle vs energy, around its slit angle)."""

    def __init__(self, contour, which):
        ViewerWindow.__init__(self, contour.app, contour.data, contour.filename,
                              contour.colormap, contour.flip, master=contour.app.root
                              if contour.app is not None else contour)
        self.contour = contour
        self.which = which
        if which == "deflector":
            self.x_axis, self.x_label = contour.defl, contour.defl_label
            self.sum_axis, sum_label = contour.slit, contour.slit_label
            colour_x, colour_s = T.AXIS_COLORS["defl"], T.AXIS_COLORS["slit"]
        else:
            self.x_axis, self.x_label = contour.slit, contour.slit_label
            self.sum_axis, sum_label = contour.defl, contour.defl_label
            colour_x, colour_s = T.AXIS_COLORS["slit"], T.AXIS_COLORS["defl"]
        title = "%s vs %s" % (self.x_label, contour.energy_label)
        self.title("%s  [%s]" % (contour.filename, title))
        self.geometry("900x800")
        if which == "slit":
            self.add_function("FS correction (whole map)...", self.open_fs_correction,
                              section="Data operations")
            if contour.data.kind in ("map", "kz_map"):
                self.add_function("De-grid map...", contour.open_degrid)
        self.add_function("Show the map (constant-E contour)", contour.show_map,
                          section="Visualization")
        self.add_function("Stack plot (EDC / MDC)...", self.open_stack)
        self.build_colormap_row()
        self.control = T.SliceControl(self.body, "At / integrate over: %s" % sum_label,
                                      on_change=self._slider_moved, color=colour_s)
        self.control.pack(fill="x")
        self.panel = T.ImagePanel(self.body, title, curves=True, cmap=contour.colormap,
                                  flip=contour.flip, on_cursor=self._cursor_moved,
                                  edc_color=colour_x, mdc_color=T.AXIS_COLORS["energy"])
        self.panel.pack(fill="both", expand=True)
        self.wire_curves(self.panel)
        across = contour.point[1] if which == "deflector" else contour.point[0]
        self.control.configure_axis(self.sum_axis,
                                    index=T.nearest_index(self.sum_axis, across))
        self.refresh_cut()

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for window in list(self.children_windows):
            try:
                window.destroy()
            except tk.TclError:
                pass
        # the data belongs to the contour
        self.contour.cut_closed(self.which)
        try:
            self.destroy()
        except tk.TclError:
            pass
        self.drop_arrays()

    def image_panels(self):
        return (self.panel,)

    def refresh_cut(self):
        idxs = self.control.indices()
        cube = self.contour._cube()
        if self.which == "deflector":
            frame = self.control.combine(np.asarray(cube[:, idxs[0]:idxs[-1] + 1, :]), 1)
        else:
            frame = self.control.combine(np.asarray(cube[idxs[0]:idxs[-1] + 1, :, :]), 0)
        p = self.contour.point
        along = p[0] if self.which == "deflector" else p[1]
        self.panel.cursor = (along, p[2])
        self.panel.set_data(frame, self.x_axis, self.contour.E, self.x_label,
                            self.contour.energy_label)

    def follow_point(self, point):
        across = point[1] if self.which == "deflector" else point[0]
        along = point[0] if self.which == "deflector" else point[1]
        if T.nearest_index(self.sum_axis, across) != self.control.index:
            self.control.set_position(across, notify=False)
            self.panel.cursor = (along, point[2])
            self.refresh_cut()
        else:
            self.panel.set_cursor(along, point[2], notify=False)

    def _slider_moved(self):
        value = float(self.sum_axis[self.control.index])
        self.refresh_cut()
        if self.which == "deflector":
            self.contour.set_point(slit=value, source=self)
        else:
            self.contour.set_point(defl=value, source=self)

    def _cursor_moved(self, panel, x, y):
        if self.which == "deflector":
            self.contour.set_point(defl=x, energy=y, source=self)
        else:
            self.contour.set_point(slit=x, energy=y, source=self)

    def slice_label(self):
        i = self.control.index
        return "%.4g (ind %d)" % (self.sum_axis[i], i + 1)

    def exportable_panels(self):
        suffix = "Deflector_vs_E" if self.which == "deflector" else "Slit_vs_E"
        return [(suffix, self.panel.values, self.panel.x, self.panel.y,
                 self.x_label, self.contour.energy_label, self.panel)]

    def open_stack(self):
        from ui.process import StackWindow
        return self.adopt(StackWindow(self, self.panel.values,
                                      (self.panel.x, self.panel.y),
                                      (self.x_label, self.contour.energy_label),
                                      "%s %s" % (self.filename, self.slice_label())))

    # FS correction: the whole cube, fitted on this slit cut
    def fs_angle_axis(self):
        return self.x_axis

    def fs_angle_label(self):
        return self.x_label

    def fs_correction_target(self):
        if self.which != "slit":
            return None
        c = self.contour
        return (c.full_cube(), c.slit, c.E, 1, 2, c.data.kind,
                lambda new_energy: (np.asarray(c.defl, dtype=float),
                                    np.asarray(c.slit, dtype=float), new_energy))


# --------------------------------------------------------------------------
# Real-space scans
# --------------------------------------------------------------------------
class SpatialScanWindow(ViewerWindow):
    """The real-space map beside the spectrum at the cursor.

    Click the map (or type) to see the spectrum at that pixel. **Integrate
    box -> spectrum** sums the spectrum over the map's box; **Integrate box
    -> map** shows the map summed over the spectrum's (k, E) box -- pick a
    feature, see where on the sample it comes from.

    The map opens the way the file's metadata says the beamline draws it
    (``Spatial.swap_xy``, ``Spatial.invert_x`` / ``Spatial.invert_y``, see
    ``loader.nxs_file._spatial_labels``): an ANTARES coarse scan with ST
    across, large to small, and SZ down the side increasing downwards; a
    fine scan with PIX across, large to small, and PIY decreasing
    downwards. **Swap X/Y** and the **Reverse** boxes change that; each
    axis keeps its own direction when the two are swapped.
    """

    def __init__(self, app, data, filename, colormap=T.DEFAULT_COLORMAP,
                 flip=False, master=None):
        ViewerWindow.__init__(self, app, data, filename, colormap, flip, master)
        self.geometry("1500x820")
        scan = data.scan
        self.labels = scan.labels
        info = getattr(scan, "info", {}) or {}
        self.frame_label = ""
        extra = []
        if data.kind == "spem_4d":
            extra = [("Integrate map box → spectrum", self.integrate_spatial_box),
                     ("Integrate spectrum box → map", self.integrate_frame_box),
                     ("Whole map", self.reset_map)]
        self.build_colormap_row(extra)

        # The starting orientation, from the metadata; then the user's.
        self.swap_var = tk.BooleanVar(value=bool(info.get("Spatial.swap_xy", False))
                                      and data.kind == "spem_4d")
        self.rev_x_var = tk.BooleanVar(value=bool(info.get("Spatial.invert_x", False)))
        self.rev_y_var = tk.BooleanVar(value=bool(info.get("Spatial.invert_y", False)))
        orient = ttk.Frame(self.top)
        orient.pack(side="left", padx=(12, 0))
        if data.kind == "spem_4d":
            ttk.Checkbutton(orient, text="Swap X/Y", variable=self.swap_var,
                            command=self._orientation_changed).pack(side="left")
        ttk.Checkbutton(orient, text="Reverse %s" % self.axis_name("x"),
                        variable=self.rev_x_var,
                        command=self._orientation_changed).pack(side="left", padx=2)
        if data.kind == "spem_4d":
            ttk.Checkbutton(orient, text="Reverse %s" % self.axis_name("y"),
                            variable=self.rev_y_var,
                            command=self._orientation_changed).pack(side="left", padx=2)

        panes = ttk.Frame(self.body)
        panes.pack(fill="both", expand=True)
        self.spatial = T.ImagePanel(panes, "Spatial map", curves=False,
                                    cmap=colormap, flip=flip,
                                    on_cursor=self._pixel_moved, figsize=(6, 5.2))
        self.spatial.pack(side="left", fill="both", expand=True)
        self.frame = T.ImagePanel(panes, "E vs k at cursor / selection", curves=True,
                                  cmap=colormap, flip=flip, figsize=(6.4, 5.2))
        self.frame.pack(side="left", fill="both", expand=True)
        self.wire_curves(self.frame)

        if data.kind == "spem_4d":
            overview = T.with_wait_cursor(self, data.spatial_overview_computed)
            self._map = np.asarray(overview)          # (y, x)
            self._show_map()
            self._show_frame_at(len(scan.y) // 2, len(scan.x) // 2)
        else:  # spem_1d: (x, angle) map summed over energy
            self._map = np.asarray(data.line_kmap)    # (x, k)
            self._show_map()
            self._show_frame_at_x(len(scan.x) // 2)

    def image_panels(self):
        return (self.spatial, self.frame)

    def slice_label(self):
        return self.frame_label

    # -- orientation -----------------------------------------------------------
    def axis_name(self, slot):
        """The stage name of a spatial axis ("ST", "PIX"...), from its title."""
        label = self.labels.get(slot, slot) or slot
        return label.split("(")[0].strip() or slot

    def swapped(self):
        return bool(self.swap_var.get()) and self.data.kind == "spem_4d"

    def _show_map(self, keep_view=False):
        """Put ``self._map`` on the spatial panel, the way the orientation
        controls say."""
        scan = self.data.scan
        if self.data.kind == "spem_4d":
            xl, yl = self.labels.get("x", "x"), self.labels.get("y", "y")
            if self.swapped():              # data y across, data x up the side
                self.spatial.set_orientation(self.rev_y_var.get(), self.rev_x_var.get(),
                                             redraw=False)
                self.spatial.set_data(self._map, scan.y, scan.x, yl, xl,
                                      keep_view=keep_view)
            else:
                self.spatial.set_orientation(self.rev_x_var.get(), self.rev_y_var.get(),
                                             redraw=False)
                self.spatial.set_data(self._map.T, scan.x, scan.y, xl, yl,
                                      keep_view=keep_view)
        else:
            self.spatial.set_orientation(self.rev_x_var.get(), False, redraw=False)
            self.spatial.set_data(self._map, scan.x, scan.y, self.labels.get("x", "x"),
                                  self.labels.get("y", "angle"), keep_view=keep_view)

    def _orientation_changed(self):
        self.spatial.set_box(None, redraw=False)
        self._show_map(keep_view=False)

    def _data_box_indices(self):
        """The spatial box as ``(ix0, ix1, iy0, iy1)`` in the data's own
        (x, y) order, whichever way round the map is shown."""
        i0, i1, j0, j1 = self.spatial.box_indices()
        return (j0, j1, i0, i1) if self.swapped() else (i0, i1, j0, j1)

    # -- the spectrum at the cursor ---------------------------------------------
    def _pixel_moved(self, panel, x, y):
        scan = self.data.scan
        if self.data.kind == "spem_4d":
            if self.swapped():
                x, y = y, x
            self._show_frame_at(T.nearest_index(scan.y, y), T.nearest_index(scan.x, x))
        else:
            self._show_frame_at_x(T.nearest_index(scan.x, x))

    def _show_frame_at(self, row, col):
        scan = self.data.scan
        frame = np.asarray(self.data.frame_at(row, col), dtype=float)
        self.frame.set_data(frame, scan.k, scan.z, self.labels.get("k", "k"),
                            self.labels.get("z", "E"))
        self.frame_label = "%s=%.4g, %s=%.4g" % (self.axis_name("x"), scan.x[col],
                                                 self.axis_name("y"), scan.y[row])
        self.say("pixel (%s)" % self.frame_label)

    def _show_frame_at_x(self, xi):
        scan = self.data.scan
        frame = np.asarray(self.data.frame_at_x(xi), dtype=float)
        self.frame.set_data(frame, scan.y, scan.z, self.labels.get("y", "k"),
                            self.labels.get("z", "E"))
        self.frame_label = "%s=%.4g" % (self.axis_name("x"), scan.x[xi])
        self.say(self.frame_label)

    def integrate_spatial_box(self):
        if self.spatial.box is None:
            T.info(self, "Integrate", "Set a box on the spatial map first "
                   "(type x0 x1 y0 y1 under it, or zoom and press 'From zoom').")
            return
        scan = self.data.scan
        ix0, ix1, iy0, iy1 = self._data_box_indices()
        frame = T.with_wait_cursor(self, self.data.frame_over_region,
                                   slice(iy0, iy1 + 1), slice(ix0, ix1 + 1))
        self.frame.set_data(frame, scan.k, scan.z, self.labels.get("k", "k"),
                            self.labels.get("z", "E"))
        self.frame_label = "%s[%.4g..%.4g], %s[%.4g..%.4g] integrated" % (
            self.axis_name("x"), scan.x[ix0], scan.x[ix1],
            self.axis_name("y"), scan.y[iy0], scan.y[iy1])
        self.say("Spectrum integrated over " + self.frame_label)

    def integrate_frame_box(self):
        if self.frame.box is None:
            T.info(self, "Integrate", "Set a box on the spectrum first.")
            return
        ik0, ik1, ie0, ie1 = self.frame.box_indices()

        def work(report):
            return self.data.spatial_over_region(
                slice(ik0, ik1 + 1), slice(ie0, ie1 + 1),
                progress=lambda done, total: report(done / float(total), "row %d of %d" % (done, total)))

        try:
            smap = T.run_blocking(self, "Integrating the cube", work)
        except T.JobCancelled:
            return
        self._map = np.asarray(smap)
        self._show_map(keep_view=True)
        self.say("Map integrated over the spectrum's box")

    def reset_map(self):
        self._map = np.asarray(self.data.spatial_overview_computed())
        self._show_map(keep_view=True)

    def exportable_panels(self):
        scan = self.data.scan
        out = []
        if self.frame.values is not None:
            xl = self.labels.get("k" if self.data.kind == "spem_4d" else "y", "k")
            out.append(("EvsK", self.frame.values, self.frame.x, self.frame.y, xl,
                        self.labels.get("z", "E"), self.frame))
        if self.spatial.values is not None:
            out.append(("SpatialMap", self.spatial.values, self.spatial.x,
                        self.spatial.y, self.spatial.x_label, self.spatial.y_label,
                        self.spatial))
        return out


def ensure_in_memory(parent, data, filename=""):
    """Read a lazily opened dataset into memory (see ``NxsData.in_memory``:
    a spatial scan stays in its file)."""
    in_memory = getattr(data, "in_memory", None)
    if in_memory is None or in_memory():
        return
    if not data.fits_in_memory():
        # too big to hold: sliced from the file, as before
        return
    if parent is None:
        data.load_into_memory()
        return

    def work(report):
        return data.read_into_memory(
            progress=lambda done, total: report(done / float(max(1, total)),
                                                "%d of %d" % (done, total)))

    array = T.run_blocking(parent, "Reading %s into memory" % (filename or "the data"),
                           work)
    data.keep_in_memory(array)


def open_viewer(app, data, filename, colormap=T.DEFAULT_COLORMAP, flip=False,
                master=None):
    """The right viewer for a dataset, or None for a kind with nothing to
    show. Anything but a spatial scan is read into memory first (once,
    with a progress window), so that moving a slider or the cursor slices
    a numpy array instead of reading the file again. Raises
    :class:`ui.tkbase.JobCancelled` if that read was cancelled."""
    ensure_in_memory(master or (app.root if app is not None else None), data, filename)
    if data.kind in ("spem_4d", "spem_1d"):
        return SpatialScanWindow(app, data, filename, colormap, flip, master)
    if data.kind == "cut":
        return CutWindow(app, data, filename, colormap, flip, master)
    if data.kind in CUBE_KINDS:
        return ContourWindow(app, data, filename, colormap, flip, master)
    if data.kind in CURVE_KINDS:
        from ui.curves import CurveWindow
        return CurveWindow(app, data, filename, colormap, flip, master)
    return None
