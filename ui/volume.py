"""
ui/volume.py
============
Cubes: processing over a whole cube, and the 3-D view.

:class:`VolumeProcessDialog` -- the 2-D operations applied plane by plane
(on the planes you choose: constant energy, or constant angle), or a
rotational / mirror symmetrisation of every constant-energy plane.

:func:`open_volume_view` -- choose the region and the sampling, then a
:class:`VolumeWindow` with three views of the cube: orthogonal **slices**, a
**notched cube**, and **projections** (maximum intensity, sum, alpha
composite, isosurface). The geometry (faces, camera, painter's ordering,
projections) is :mod:`tools.volume`; this file only draws it with
matplotlib. The camera is set by typing azimuth / elevation (no mouse
orbiting, which the Qt version had).
"""
import numpy as np
import tkinter as tk
from tkinter import ttk

from tools import process as P
from tools import volume as V
from ui import tkbase as T
from ui.tkbase import Field
from ui.data import MemoryData
from ui.process import decimate

PLANE_OPERATIONS = ["Gaussian smooth", "Savitzky-Golay smooth", "Remove spikes",
                    "Second derivative (first axis)", "Second derivative (second axis)",
                    "Curvature 2D", "Curvature 1D (first axis)",
                    "Curvature 1D (second axis)", "Normalise each line"]
SUFFIXES = {"smooth": "_sm", "derivative": "_d2", "curvature": "_cur",
            "normalise": "_nor", "symmetry": "_symm", "despike": "_fix"}


class VolumeProcessDialog(tk.Toplevel):
    def __init__(self, app, datasets, master=None, colormap="gray", flip=False):
        tk.Toplevel.__init__(self, master or app.root)
        self.app = app
        self.datasets = list(datasets)
        self.colormap, self.flip = colormap, flip
        self.title("Data processing (3D)")
        self.geometry("1150x900")
        name, data = self.datasets[0]
        x, k, z, cube = data.angle_cube
        self.values = T.with_wait_cursor(self if master is None else master,
                                         lambda: np.asarray(cube, dtype=float))
        self.axes = [np.asarray(x, float), np.asarray(k, float), np.asarray(z, float)]
        scan = data.scan
        self.labels = [scan.labels.get("x", "x"), scan.labels.get("k", "y"),
                       scan.labels.get("z", "z")]
        ttk.Label(self, text="%s   (%s points; %s)" % (
            name, " x ".join(str(a.size) for a in self.axes), " x ".join(self.labels))).pack(
            fill="x", padx=6, pady=4)
        shorts = [l.split(" (")[0] for l in self.labels]
        self.plane_names = []
        for dim in range(3):
            others = [shorts[d] for d in range(3) if d != dim]
            self.plane_names.append("%s-%s planes (at each %s)" % (others[0], others[1], shorts[dim]))
        top = ttk.Frame(self)
        top.pack(fill="x", padx=6)
        ttk.Label(top, text="Mode").pack(side="left")
        self.mode = tk.StringVar(value="Plane-wise")
        c = ttk.Combobox(top, textvariable=self.mode, values=["Plane-wise", "Symmetrise"],
                         state="readonly", width=12)
        c.pack(side="left", padx=4)
        c.bind("<<ComboboxSelected>>", lambda e: self.show_form())
        self.holder = ttk.Frame(self)
        self.holder.pack(fill="x", padx=6)
        self.plane_form = T.Form(self.holder, [
            Field("planes", "Planes", "choice", self.plane_names[2], self.plane_names),
            Field("op", "Operation", "choice", PLANE_OPERATIONS[0], PLANE_OPERATIONS),
            Field("wu", "Width, first axis of the plane", "float", 0.0),
            Field("wv", "Width, second axis of the plane", "float", 0.0),
            Field("a0", "a0 (curvature)", "float", 1.0),
            Field("norm", "Normalise to", "choice", "area", ["area", "max", "mean"]),
        ], on_change=self.refresh, columns=2)
        self.sym_form = T.Form(self.holder, [
            Field("fold", "Rotation order n (n-fold)", "int", 6),
            Field("mirrors", "Mirror lines (deg, e.g. 0, 30)", "str", ""),
            Field("inversion", "Also k → −k", "bool", False),
            Field("cx", "Centre x", "float", 0.0),
            Field("cy", "Centre y", "float", 0.0),
            Field("output", "Output", "choice", "symmetrised", ["symmetrised", "coverage map"]),
        ], on_change=self.refresh, columns=2)
        row = ttk.Frame(self)
        row.pack(fill="x", padx=6)
        self.preview_ctl = T.SliceControl(row, "Preview plane", on_change=self.refresh)
        self.preview_ctl.pack(fill="x")
        self.plot = T.PlotFrame(self, figsize=(10, 4.6))
        self.plot.pack(fill="both", expand=True, padx=6)
        self.ax_src = self.plot.figure.add_subplot(121)
        self.ax_res = self.plot.figure.add_subplot(122)
        self.note = ttk.Label(self, text="", wraplength=1000, foreground="#225")
        self.note.pack(fill="x", padx=6)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Preview", command=self.refresh).pack(side="left", padx=2)
        ttk.Button(bar, text="Apply to the whole cube", command=self.apply).pack(side="left", padx=2)
        ttk.Button(bar, text="Open the 3D view", command=self.open_view).pack(side="left", padx=2)
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right")
        self._dim = None
        self.show_form()

    def show_form(self):
        self.plane_form.pack_forget()
        self.sym_form.pack_forget()
        (self.plane_form if self.mode.get() == "Plane-wise" else self.sym_form).pack(fill="x")
        self.refresh()

    def plane_dim(self):
        if self.mode.get() != "Plane-wise":
            return 2
        return self.plane_names.index(self.plane_form.raw("planes"))

    def plane_function(self):
        v = self.plane_form.values()
        dim = self.plane_names.index(v["planes"])
        axes = [self.axes[d] for d in range(3) if d != dim]
        choice = PLANE_OPERATIONS.index(v["op"])
        widths, a0, mode = (v["wu"], v["wv"]), v["a0"], v["norm"]

        def run(plane):
            if choice == 0:
                return P.gaussian_smooth(plane, axes, widths)
            if choice == 1:
                return P.savgol_smooth(plane, axes, widths)
            if choice == 2:
                return P.despike(plane)[0]
            if choice == 3:
                return P.derivative(plane, axes, 0, 2)
            if choice == 4:
                return P.derivative(plane, axes, 1, 2)
            if choice == 5:
                return P.curvature(plane, axes, mode="2d", a0=a0)
            if choice == 6:
                return P.curvature(plane, axes, mode="x", a0=a0)
            if choice == 7:
                return P.curvature(plane, axes, mode="y", a0=a0)
            return P.normalise(plane, axes, 0, mode=mode)
        params = {"planes": self.labels[dim], "operation": v["op"]}
        if choice in (0, 1):
            params.update({"width_u": widths[0], "width_v": widths[1]})
        if choice in (5, 6, 7):
            params["a0"] = a0
        if choice == 8:
            params["mode"] = mode
        key = {0: "smooth", 1: "smooth", 2: "despike", 3: "derivative", 4: "derivative",
               5: "curvature", 6: "curvature", 7: "curvature", 8: "normalise"}[choice]
        return key, params, dim, run

    def symmetry_settings(self):
        v = self.sym_form.values()
        mirrors = tuple(T.parse_floats(v["mirrors"])) if v["mirrors"].strip() else ()
        return {"fold": max(1, int(v["fold"])), "centre": (v["cx"], v["cy"]),
                "mirrors": mirrors, "inversion": v["inversion"],
                "coverage": v["output"] == "coverage map"}

    def refresh(self):
        try:
            dim = self.plane_dim()
        except (ValueError, T.FormError):
            return
        if dim != self._dim:
            self._dim = dim
            self.preview_ctl.configure_axis(self.axes[dim])
        index = self.preview_ctl.index
        plane = np.take(self.values, index, axis=dim)
        axes = [self.axes[d] for d in range(3) if d != dim]
        labels = [self.labels[d] for d in range(3) if d != dim]
        small, small_axes = decimate(plane, axes)
        try:
            if self.mode.get() == "Plane-wise":
                key, params, _, run = self.plane_function()
                result = run(small)
            else:
                s = self.symmetry_settings()
                key, params = "symmetry", s
                r = P.symmetrise(small, small_axes[0], small_axes[1], fold=s["fold"],
                                 centre=s["centre"], mirror_angles=s["mirrors"],
                                 inversion=s["inversion"])
                result = r.coverage if s["coverage"] else r.values
        except Exception as exc:                                # noqa: BLE001
            self.note.configure(text=str(exc))
            return
        for ax, values, title in ((self.ax_src, small, "Source"), (self.ax_res, result, "Result")):
            ax.clear()
            lo, hi = T.auto_levels(values)
            ax.pcolormesh(T.edges_of(small_axes[0]), T.edges_of(small_axes[1]),
                          np.asarray(values).T, cmap=T.get_cmap(self.colormap, self.flip),
                          vmin=lo, vmax=hi)
            ax.set_title("%s  (%s = %.4g)" % (title, self.labels[dim].split(" (")[0],
                                              self.axes[dim][index]), fontsize=9)
            ax.set_xlabel(labels[0], fontsize=8)
            ax.set_ylabel(labels[1], fontsize=8)
        self.plot.draw()
        self.note.configure(text="%s: %s" % (key, ", ".join("%s=%s" % kv for kv in sorted(params.items()))))

    def apply(self):
        try:
            plane_wise = self.mode.get() == "Plane-wise"
            if plane_wise:
                key, params, dim, run = self.plane_function()
                sym = None
            else:
                key, dim, run = "symmetry", None, None
                sym = self.symmetry_settings()
                params = {"fold": sym["fold"], "centre": list(sym["centre"]),
                          "inversion": sym["inversion"],
                          "output": "coverage map" if sym["coverage"] else "symmetrised"}
                if sym["mirrors"]:
                    params["mirrors"] = list(sym["mirrors"])
            prepared = []
            for name, data in self.datasets:
                x, k, z, cube = data.angle_cube
                prepared.append((name, data, np.asarray(cube, dtype=float),
                                 [np.asarray(x, float), np.asarray(k, float), np.asarray(z, float)]))
        except Exception as exc:                                # noqa: BLE001
            T.warning(self, "Data processing (3D)", str(exc))
            return None

        def work(report):
            created = []
            for index, (name, data, values, axes) in enumerate(prepared):
                base, span = index / float(len(prepared)), 1.0 / len(prepared)

                def tick(done, total, _b=base, _s=span, _n=name):
                    report(_b + _s * (done / float(max(total, 1))), _n)
                    return True
                if plane_wise:
                    out = V.apply_plane_wise(values, run, axis=dim, progress=tick)
                else:
                    out, coverage, _c = V.symmetrise_volume(
                        values, axes[0], axes[1], axes[2], fold=sym["fold"],
                        centre=sym["centre"], mirror_angles=sym["mirrors"],
                        inversion=sym["inversion"], progress=tick)
                    if sym["coverage"]:
                        out = coverage
                info = P.record_step(dict(data.scan.info), P.Step(key, params, source=name))
                created.append(MemoryData(data.kind, tuple(axes), out, dict(data.scan.labels),
                                          source_label="%s%s" % (name, SUFFIXES.get(key, "_proc")),
                                          parameters=params, prefix="proc.%s" % key,
                                          source_path=getattr(data, "path", ""),
                                          source_info=info,
                                          source_motors=dict(data.scan.fourd_info)))
            return created

        def done(created):
            for data in created:
                self.app.add_dataset(data)
            self.note.configure(text="Added %d dataset(s) to the list." % len(created))
        T.run_job(self, "Processing the cube", work, on_done=done)

    def open_view(self):
        name, data = self.datasets[0]
        return open_volume_view(self, self.app, data, name, self.colormap, self.flip)


# ==========================================================================
# The 3-D view
# ==========================================================================
def open_volume_view(master, app, data, name, colormap="gray", flip=False):
    """Ask for the region and sampling, then open the 3-D view."""
    x, k, z, cube = data.angle_cube
    axes = [np.asarray(a, dtype=float) for a in (x, k, z)]
    labels = [data.scan.labels.get(s, s) for s in ("x", "k", "z")]
    fields = []
    for d, (axis, label) in enumerate(zip(axes, labels)):
        fields += [Field("lo%d" % d, "%s from" % label, "float", float(axis.min())),
                   Field("hi%d" % d, "to", "float", float(axis.max())),
                   Field("n%d" % d, "points (binned mean)", "int", int(min(axis.size, 120)))]
    v = T.ask_params(master, "What to show in 3D", fields, columns=1,
                     text="The views resample the whole cube for every frame, so the "
                          "region and the sampling decide how fast they are.")
    if v is None:
        return None
    values = T.with_wait_cursor(master, lambda: np.asarray(cube, dtype=float))
    try:
        reduced, new_axes = V.reduce_volume(values, axes,
                                            ranges=[(v["lo%d" % d], v["hi%d" % d]) for d in range(3)],
                                            targets=[max(2, v["n%d" % d]) for d in range(3)])
    except Exception as exc:                                    # noqa: BLE001
        T.warning(master, "3D view", str(exc))
        return None
    return VolumeWindow(master, reduced, new_axes, labels, name, colormap, flip)


class VolumeWindow(tk.Toplevel):
    def __init__(self, master, values, axes, labels, source_name, colormap="gray", flip=False):
        tk.Toplevel.__init__(self, master)
        self.values = np.asarray(values, dtype=float)
        self.axes = [np.asarray(a, dtype=float) for a in axes]
        self.labels = list(labels)
        self.title("3D view — %s" % source_name)
        self.geometry("1250x820")
        self.auto = T.auto_levels(self.values, 1, 99.5)
        side = ttk.Frame(self)
        side.pack(side="left", fill="y", padx=4, pady=4)
        shorts = [l.split(" (")[0] for l in self.labels]
        mid = [float(a[a.size // 2]) for a in self.axes]
        notch = [float(a[int(a.size * 0.6)]) for a in self.axes]
        self.form = T.Form(side, [
            Field("view", "View", "choice", "Slices",
                  ["Slices", "Notched cube", "Max intensity", "Sum along the view",
                   "Alpha composite", "Isosurface"]),
            Field("s0", "Slice at %s (empty = off)" % shorts[0], "optfloat", mid[0]),
            Field("s1", "Slice at %s" % shorts[1], "optfloat", mid[1]),
            Field("s2", "Slice at %s" % shorts[2], "optfloat", mid[2]),
            Field("n0", "Notch at %s" % shorts[0], "float", notch[0]),
            Field("n1", "Notch at %s" % shorts[1], "float", notch[1]),
            Field("n2", "Notch at %s" % shorts[2], "float", notch[2]),
            Field("corner", "Notch corner", "choice", "+ + +",
                  ["+ + +", "- + +", "+ - +", "- - +", "+ + -", "- + -", "+ - -", "- - -"]),
            Field("samples", "Projection samples", "int", 120),
            Field("iso", "Isosurface level (0..1 of levels)", "float", 0.5),
            Field("alpha", "Alpha opacity", "float", 0.06),
            Field("azimuth", "Azimuth (deg)", "float", 45.0),
            Field("elevation", "Elevation (deg)", "float", 25.0),
            Field("zaspect", "Height of the energy axis", "float", 0.75),
            Field("lo", "Levels min", "float", self.auto[0]),
            Field("hi", "Levels max", "float", self.auto[1]),
            Field("gamma", "Gamma", "float", 1.0),
            Field("cmap", "Colormap", "choice", colormap, T.COLORMAP_NAMES),
            Field("flip", "Flip colormap", "bool", flip),
            Field("edges", "Outline the faces", "bool", True),
        ], on_change=self.rebuild)
        self.form.pack(fill="x")
        ttk.Button(side, text="Redraw", command=self.rebuild).pack(fill="x", pady=2)
        ttk.Button(side, text="Save the view...", command=self.save_view).pack(fill="x", pady=2)
        self.note = ttk.Label(side, text="", wraplength=300)
        self.note.pack(fill="x")
        self.plot = T.PlotFrame(self, figsize=(7.5, 7))
        self.plot.pack(side="left", fill="both", expand=True)
        self.ax = self.plot.figure.add_subplot(111)
        self.rebuild()

    def camera(self, v):
        return V.Camera(azimuth=v["azimuth"], elevation=v["elevation"],
                        aspect=(1.0, 1.0, v["zaspect"]))

    def draw_into(self, ax):
        from matplotlib import colors as mcolors
        from matplotlib.transforms import Affine2D
        v = self.form.values()
        ax.clear()
        ax.set_axis_off()
        cam = self.camera(v)
        bounds = V._bounds(self.axes)
        cmap = T.get_cmap(v["cmap"], v["flip"])
        lo, hi = (v["lo"], v["hi"]) if v["hi"] > v["lo"] else self.auto
        norm = (mcolors.Normalize(lo, hi, clip=True) if abs(v["gamma"] - 1) < 1e-9
                else mcolors.PowerNorm(v["gamma"], lo, hi, clip=True))
        view = v["view"]
        if view in ("Slices", "Notched cube"):
            if view == "Slices":
                positions = [v["s0"], v["s1"], v["s2"]]
                faces = V.slice_faces(self.values, self.axes, positions, density=200)
                visible = V.visible(faces, cam, bounds, cull=False)
            else:
                signs = tuple(1 if c == "+" else -1 for c in v["corner"].split())
                faces = V.notched_box(self.values, self.axes, [v["n0"], v["n1"], v["n2"]],
                                      corner=signs, density=200)
                visible = V.visible(faces, cam, bounds, cull=True)
            edges = dict(zip((id(f) for f in visible), V.outline_edges(visible)))
            light = np.array([0.35, 0.25, 0.90])
            light /= np.linalg.norm(light)
            allpts = []
            for order, (face, screen) in enumerate(V.sort_faces(visible, cam, bounds)):
                screen = np.asarray(screen, dtype=float)
                allpts.append(screen)
                nu, nv = face.image.shape
                rgba = cmap(norm(np.ma.masked_invalid(face.image)))
                shade = 1.0
                if face.outward:
                    shade = 0.78 + 0.22 * float(abs(np.dot(face.normal(), light)))
                rgba[..., :3] *= shade
                p0, p1, p3 = screen[0], screen[1], screen[3]
                affine = Affine2D.from_values((p1[0] - p0[0]) / nu, (p1[1] - p0[1]) / nu,
                                              (p3[0] - p0[0]) / nv, (p3[1] - p0[1]) / nv,
                                              p0[0], p0[1])
                ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower",
                          extent=(0, nu, 0, nv), interpolation="nearest",
                          transform=affine + ax.transData, zorder=2 * order)
                if v["edges"]:
                    for i, draw in enumerate(edges[id(face)]):
                        if draw:
                            a, b = screen[i], screen[(i + 1) % 4]
                            ax.plot([a[0], b[0]], [a[1], b[1]], color="k", lw=0.6,
                                    alpha=0.4, zorder=2 * order + 1)
            self._axes_tripod(ax, cam, bounds, v)
            corners = np.array([[bounds[0][i], bounds[1][j], bounds[2][k]]
                                for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            screen, _ = cam.project(corners, bounds)
            pad = 0.1 * float(np.ptp(screen))
            ax.set_xlim(screen[:, 0].min() - pad, screen[:, 0].max() + pad)
            ax.set_ylim(screen[:, 1].min() - pad, screen[:, 1].max() + pad)
            ax.set_aspect("equal")
            self.note.configure(text="%d face(s) drawn" % len(visible))
            return
        cube = V.view_volume(self.values, self.axes, cam, samples=max(20, v["samples"]),
                             bounds=bounds)
        if view == "Isosurface":
            level = lo + v["iso"] * (hi - lo)
            depth, shade = V.isosurface_depth(cube, level)
            ax.imshow(shade.T, origin="lower", cmap=cmap, vmin=0, vmax=1)
            self.note.configure(text="Isosurface at %.4g: %.1f%% of the view hits it"
                                % (level, 100 * float(np.isfinite(depth).mean())))
        else:
            kind = {"Max intensity": "mip", "Sum along the view": "sum",
                    "Alpha composite": "alpha"}[view]
            image = V.project_volume(cube, mode=kind, levels=(lo, hi), alpha=v["alpha"])
            if kind == "mip":
                ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm)
            else:
                ax.imshow(image.T, origin="lower", cmap=cmap,
                          vmin=float(np.nanmin(image)), vmax=float(np.nanmax(image)) or 1.0)
            self.note.configure(text="%s over %d samples along the view" % (view, v["samples"]))
        ax.set_aspect("equal")

    def _axes_tripod(self, ax, cam, bounds, v):
        origin = np.array([[bounds[0][0], bounds[1][0], bounds[2][0]]])
        o, _ = cam.project(origin, bounds)
        for d, colour in zip(range(3), ("#d62728", "#2ca02c", "#1f77b4")):
            end = origin.copy()
            end[0, d] = bounds[d][1]
            e, _ = cam.project(end, bounds)
            ax.annotate("", xy=e[0], xytext=o[0],
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.2), zorder=10000)
            ax.text(e[0][0], e[0][1], " " + self.labels[d].split(" (")[0], color=colour,
                    fontsize=8, zorder=10001)

    def rebuild(self):
        try:
            T.with_wait_cursor(self, self.draw_into, self.ax)
        except Exception as exc:                                # noqa: BLE001
            self.note.configure(text=str(exc))
        self.plot.draw()

    def save_view(self):
        path = T.save_path(self, "Save the view", "view3d.png",
                           [("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            self.plot.figure.savefig(path, dpi=300)
