# ARPES viewer (srv12 version: Python 3.6, tkinter + matplotlib)

A program for ARPES data: browse the measurements in a folder, open them in
their own windows, convert to momentum, cut, fit, process, and lay the
results out as a figure. It reads **SOLEIL ANTARES** (NeXus/HDF5 `.nxs`),
**SOLEIL CASSIOPEE** (Scienta SES text, single spectra and whole folders),
**CASSIOPEE spin (MBS `.krx` / `.txt`)**, and its own saved format.

This is the version for the lab server (`com-antares@srv12`), which has
**Python 3.6.13, numpy 1.15.4, scipy 1.5.4, matplotlib 3.3.4** and no
PyQt5, pyqtgraph or h5py. Everything runs on what is installed there:

```
python ARPES_viewer.py
```

Needs: Python 3.6+, numpy, scipy, matplotlib, and **tkinter** (part of
Python; check with `python -c "import tkinter"`) and an X display (log in
with `ssh -X` / `ssh -Y`, or use VNC / NoMachine).

> 中文简要说明：本版本为实验室服务器（Python 3.6.13，无 PyQt5/pyqtgraph/h5py）
> 改写。界面改为 tkinter + matplotlib；HDF5 读取改用自带的纯 Python 读取器
> （`compat/pyfive`），不需要 h5py；程序自己的保存格式改为 `.npz`（旧版保存的
> `.nxs` 仍可读取）。原来需要鼠标拖动获取的参数（光标、选区框、点、范围、3D
> 视角等）全部改为可手动输入的数值；图上点击只是帮你填入这些数值。
>
> 本次更新：(1) Loader 窗口内置文件浏览器，直接列出文件夹里的文件（不再依赖
> 服务器上只显示文件夹的 Tk 文件对话框），可多选、按名称过滤；(2) SPEM 读取
> 提速：HDF5 对象头只解析一次、解压后的数据块进入有上限的缓存、点击光标只重画
> 光标线；(3) 主窗口底部显示本程序和服务器的内存占用，**Free memory** 释放缓存，
> **Memory...** 面板可查看服务器上占内存最多的进程、把计算结果移出内存、关闭窗口，
> 并设置低内存警告与自动释放。

---

## What changed from the PyQt5 version

| PyQt5 version | this version |
|---|---|
| PyQt5 + pyqtgraph windows | tkinter windows, plots drawn by matplotlib (with its zoom / pan / save toolbar) |
| h5py | `compat/pyfive` -- a bundled pure-Python HDF5 **reader** (gzip / shuffle / fletcher32; chunked data is read chunk by chunk as it is sliced) |
| saved datasets: `.nxs` written with h5py | saved datasets: **`.npz`** (numpy's compressed archive + a JSON description). `.nxs` files saved by the old version are still read. |
| drag the readout cursor | click the image (toolbar zoom/pan off) **or type x / y** and press Enter |
| drag a selection box | type **x0 x1 y0 y1** under the image, or zoom to the region and press *From zoom* |
| click points on a contour (arbitrary cut, FS correction, rotation, BZ direction, kz period, peak seeds) | points are **typed** (one `x y` per line); *Pick by clicking* only fills those lines in |
| draggable Range in the curve viewer | typed `lo hi` (or *From zoom*) |
| dragging the kz-map edge box on the slit cut | typed angle / energy range (or *From the slit cut's box*) |
| turning the 3-D view with the mouse | typed azimuth / elevation |
| picking cut B by clicking it in the main list | chosen from a list |
| Qt figure composer (text notes, arrows, scale bars, insets, second axes) | a matplotlib figure page: grid, shared colour scale, colour bar, equal aspect, outer labels only, panel letters, per-panel title / ranges / levels / colormap, export PDF / PNG / SVG / EPS / TIFF |
| Qt per-panel export dialog (image at chosen pixel size, axes-only EPS) | **Export** menu: the panel as PNG / PDF / SVG / EPS / TIFF (300 dpi), the data as a text matrix / CSV / npz |

The analysis itself (`tools/`) and the file readers (`loader/`) are the same
code as before, only made Python-3.6 compatible, so every number comes out
the same.

---

## Getting data in

**Load data...** opens the Loader window:

- **Browse** -- the folder's files, listed by the program itself (the Tk
  file dialog on the server showed the folders but not the files in them).
  Type or paste a folder and press Enter, or use *Up* / *Home* /
  double-click a folder. *Show* switches between the recognised data files
  (by extension, any letter case) and all files; *Name filter* narrows the
  list (plain text or `*` / `?` wildcards). Double-click a file, or select
  several (Shift / Ctrl + click) and press *Add selected*, or *Add every data
  file in this folder*. The system dialog is still there
  (*System dialog...*) as a fallback.
- **Files to load** -- what will be added; *Remove selected* / *Clear*.
- **Reader** -- detection is automatic and says what it found; override it
  when a file from an unfamiliar layout is misread. Files this program saved
  are always recognised.
- **Datasets found** -- what is inside, before anything is read. A `.nxs`
  file often holds several measurements; each gets its own row.
- **Axis order** -- reorder the array's dimensions, once, at the door.
- **First axis of a map is** -- angle (default), photon energy,
  temperature, gate voltage ... Only an angle axis is offered a k conversion.

**Quick add files...** skips the options (same browser, in a small window).

What is inside the chosen files is worked out on a worker thread, so the
Loader never freezes on a long list, and each file is only looked at once.

A numbered CASSIOPEE folder (`<name>_1_ROI1_.txt`, `<name>_2_ROI1_.txt`, ...)
is listed once, as the assembled cube: a `map` when the polar angle was
stepped, a `kz_map` when the monochromator was.

## The main window

One row per *dataset*. Click a row for its axes (the *Data information*
table), **double-click** (or *Open*) to open it. **Information...** shows all
the metadata (with a filter and CSV export).

Right-click a row (or a selection) for: Open, Open slit cut / deflector cut
(of a map, without its contour), Show information, Rename, Plot as a figure,
Stack plot, MDC / EDC fit, 3D view, Cut arithmetic on the two, Save, Remove,
Session log. An entry that does not fit the selection is greyed, with the
reason beside it. The buttons under the list: Open, Information, **Process**
(2-D for cuts, 3-D for cubes), **Data operations** (truncate / self-normalise
/ compress several datasets of the same format), Figure, Save, Remove.

The default colormap (with *apply to open windows*) is set in the top row;
each viewer also has its own.

## Data kinds

| kind | shape | what it is |
|---|---|---|
| `cut` | (angle, E) | one spectrum |
| `map` | (angle, angle/k, E) | a deflector or polar-angle scan |
| `k_map` | (kx, ky, E) | a map converted to momentum |
| `kz_map` | (hv, angle/k, E) | a photon-energy scan |
| `kz_map_k` | (k_z, k_par, E) | that scan converted to momentum |
| `spem_1d` / `spem_4d` | (x, k, E) / (x, y, k, E) | real-space scans |
| `edc` / `mdc` / `spin_edc` | (axis, channel) | curves (with optional `σ <name>` channels) |

## The viewer windows

Every viewer has a menu bar -- **Functions** (the tools, by section),
**Slice** (save the slice on screen to the list, open it in a new window, as
a figure) and **Export** -- and a row with its colormap. Every image panel
has, under it:

- **Levels min / max** (empty = automatic), **gamma**, the display
  interpolation, a display-only smoothing, grid, **1:1** (equal scale);
- the **Cursor** (x, y typed, or click the image) with **EDC ±** and
  **MDC ±** integration half-widths, and **EDC → list** / **MDC → list**;
- the **Box** `x0 x1 y0 y1` (typed or *From zoom*), used by the tools that
  need a region.

The toolbar above each plot is matplotlib's: *home*, *back / forward*,
*pan*, *zoom*, *save*. While zoom or pan is on, clicks belong to it, not to
the cursor.

- **A cut** opens as the E-vs-k spectrum with its EDC and MDC.
- **A cube** opens on its constant-energy contour, with an energy slider
  (Pos / Ind / ± / mean-sum). **Deflector cut** and **Slit cut** open the
  orthogonal cuts. The three share one point in the cube -- deflector red,
  slit green, energy blue -- so moving the cursor or slider in any of them
  re-slices the others.
- **A spatial scan** opens the real-space map and the spectrum at the
  cursor; *Integrate map box → spectrum* and *Integrate spectrum box → map*
  sum over a box on either.
- **A curve** opens in the curve viewer.

## Analysis (Functions menus)

| viewer | functions |
|---|---|
| map / kz map / k-map | Arbitrary cut, Map k conversion, kz map processing, kz → momentum, De-grid map, Process the cube (3-D), 3D view, Brillouin zone (k-maps), Slice series figure |
| cut | Fermi level, MDC / EDC fit (k-converted cuts), Stack plot, FS correction, Cut k conversion, Cut arithmetic, De-grid, Process |
| slit cut of a map | FS correction (applied to the whole map), De-grid map, Stack plot, Show the map |
| curve | Curve fit, Spin analysis (spin EDCs), Crop / Bin / Normalise / Subtract a background / Shift the axis / Add counting errors, As a figure, Export as text |

- **Map k conversion** -- theta / phi offsets (initially the contour's
  cursor; *Read cursor* copies it again), sample rotation (typed, or from
  two points), energy offset, output grid by number of points or by step.
- **Cut k conversion** -- Γ's angles inherited from a converted map in the
  list or typed; the cut's deflector angle from the file.
- **Fermi level** -- EDC summed over an angle range, fitted over an energy
  window (typed, or *From the box*); temperature held by default; then
  *Offset energy axis* (E − E_F), *Divide the Fermi cut-off out*, or fit E_F
  channel by channel.
- **FS correction** -- points along the feature that should be flat
  (typed, clicked, or measured on a gold reference channel by channel),
  a polynomial through them, every column shifted.
- **MDC / EDC fit** -- bands with seeds (`position centre FWHM [height]`
  per line, or click the line plot), *Try this line*, *Fit the series*,
  then **Dispersion** (v_F, m*, with the window scan) and **Self-energy**
  (Re / Im Σ with the Kramers-Kronig check).
- **kz map processing** -- per-spectrum Fermi level from a typed box,
  alignment, crop, normalisation; E_F vs hv plotted.
- **kz → momentum** -- V0 (with *Scan V0*), zone boundaries, the two-point
  k_z period with matching lattice planes.
- **Cut arithmetic** -- LD, CD, reference division, A − B, A / B,
  (A − B)/(A + B), A + B; scaling B to A; Poisson uncertainty.
- **De-grid** -- the detector grid removed from a map (the map is its own
  reference) or a cut (with a `[grid]` from a map, or a notch filter).
- **Brillouin zone** -- 3-D crystal (space group, cell, cut plane) or 2-D
  layer, conventional or irreducible, tiled, moiré of two layers, a 3-D
  preview of the cut plane.
- **Process** -- smooth, derivatives, curvature, backgrounds,
  symmetrisation (2-D); the same plane by plane, or rotational symmetrisation
  (3-D); before / after preview.
- **3D view** -- orthogonal slices, notched cube, maximum-intensity / sum /
  alpha-composite projections, isosurface.

## Nothing is lost: autosave, the log, and recovery

Every computed dataset is written to this run's session folder
(`~/.arpes_viewer/sessions/`) the moment it exists (spatial scans excepted).
Session folders are deleted after three days, or as soon as the dataset has
been saved to a file of your own. `operations.log` there records every
store / save / discard. On startup, datasets left by an earlier session are
offered back; closing with unsaved computed data asks first.

**Save...** writes the selected datasets to one `.npz` file, which this
program opens again (all rows come back, with their metadata).

## Memory

The line at the bottom of the main window shows this program's memory and
how full the **server's** memory is (all users, all programs; green / orange
/ red), updated every few seconds.

- **Free memory** gives back what the program keeps only for speed: the
  HDF5 reader's caches (decompressed chunks, small datasets read whole) and
  the file behind the row last clicked; then Python's garbage collector and
  `malloc_trim`, so the freed memory really goes back to the server.
- **Memory...** opens the panel: the program's memory now / at its peak /
  in swap and what its caches hold; the computed datasets held in memory
  (**Move selected out of memory** -- they are auto-saved, stay in the list,
  and are read back from disk when opened); the open viewer windows (**Close
  selected windows**); the largest processes on the server; and the
  settings, kept for next time:
  - warn when the server has less than *N* % free (default 10 %), and then
    free the program's caches by itself;
  - warn when this program uses more than *N* GB (0 = never);
  - the ceiling of the HDF5 chunk cache (default 256 MB).

**Free everything possible** does both: the caches, and every computed
dataset that is auto-saved and not open in a window.

## Speed of SPEM (spatial) scans

A spatial scan stays in its file and is read as you move around it. Three
things make that fast with the bundled pure-Python HDF5 reader:

- each HDF5 object header is parsed once per file (reading the metadata of
  a scan used to re-parse every group on every lookup);
- decompressed chunks are kept in a shared cache with a hard ceiling (the
  setting above), so the spectrum of a nearby pixel costs nothing; region
  sums and the overview stream the cube in bands that follow the file's
  chunking, so each chunk is inflated once;
- a click on the spatial map only redraws the cursor lines (the map itself
  is not re-rendered), and a new spectrum of the same size replaces the
  pixels of the image instead of rebuilding the plot.

The row last clicked in the main list is kept open, so double-clicking it
does not read the file a second time.

## Long operations

Conversions, fits over many lines, de-gridding and multi-file loading run
on a worker thread with a progress window and a working Cancel.

## Adding a beamline

Copy `loader/soleil.py` (HDF5) or `loader/cassiopee.py` (text), implement
`can_open`, `list_entries` and `load` (returning an `NxsScan` with the axis
slots its `kind` names in `loader/nxs_file.AXIS_SLOTS`), and add the module to
`_install_default_loaders()` in `loader/registry.py`.

## Layout of the code

```
ARPES_viewer.py   the launcher: the dataset list, the session folder
loader/           reading data in, and keeping it          (no GUI)
tools/            the algorithms                           (no GUI)
ui/               the windows (tkinter + matplotlib)
compat/           what Python 3.6 / the server lacks: the dataclasses
                  backport, the pure-Python HDF5 reader (pyfive, patched;
                  its chunk cache is compat/pyfive/chunkcache.py),
                  numpy helpers
```

| File | Role |
|---|---|
| `loader/nxs_file.py` | The SOLEIL parser, `NxsScan`, the axis-slot table, lazy arrays, the `.npz` save format. |
| `loader/registry.py` | Which reader opens a file; load-time axis order and axis role. |
| `loader/soleil.py`, `cassiopee.py`, `cassiopee_spin.py`, `native.py` | The readers. |
| `loader/session.py` | Autosave folder, memory budget, operations log. |
| `tools/memory.py` | Memory of the program and of the server (from `/proc`), the largest processes, freeing (`gc`, `malloc_trim`, the reader's caches). |
| `tools/*.py` | k conversion (`kspace`, `cutk`, `kzconv`), Fermi edge (`fermi`), peaks and dispersion (`peaks`, `dispersion`), processing (`process`, `volume`), `curves`, `spin`, `degrid`, `cutops`, `kzmap`, Brillouin zones (`lattice`, `spacegroups`, `bz2d`, `bz3d`, `moire`, `cleavage`), `dataops`, `colormaps`, `system`. |
| `ui/tkbase.py` | Forms, the worker thread + progress window, the image panel, the slice slider, colormaps. |
| `ui/filebrowser.py` | The folder browser of the Loader and of *Quick add files...*. |
| `ui/memory.py` | The memory line of the main window and the Memory panel. |
| `ui/tktext.py` | Draws non-Latin-1 labels (Å⁻¹, σ, Γ, →) as ASCII on Tk installations without Unicode fonts (set `ARPES_UNICODE=1` to turn off). |
| `ui/data.py` | The dataset objects the viewers use (`NxsData`, `MemoryData`). |
| `ui/viewers.py` | Cut, contour + cuts, spatial scan, pop-out windows. |
| `ui/analysis.py` | k conversion, arbitrary cut, FS correction, Fermi level, data operations, Brillouin zone. |
| `ui/curves.py`, `fit.py`, `process.py`, `volume.py`, `figure.py`, `cutops.py`, `degrid.py`, `kzmap.py`, `kzconv.py`, `loader_dialog.py`, `list_actions.py` | The other windows. |

## Assumptions worth knowing about

**Axis order in SOLEIL `.nxs` files.** MATLAB reports HDF5 dimensions in
reverse order compared to Python, so `load_soleil_nxs.m`'s `permute` cannot
be copied index for index. `loader.nxs_file.align_and_transpose` matches each
array axis to a physical axis **by length** against the calibration arrays;
it warns if two axes have the same length. To check a new file:

```python
from loader.nxs_file import inspect_nxs
inspect_nxs("your_file.nxs")
```

**HDF5 files the bundled reader cannot open.** Files written in the HDF5
1.10+ format with new-style chunk indexes (`libver="latest"`), or compressed
with filters other than gzip (LZF, LZ4, bitshuffle, Blosc), give a clear
error. ANTARES DataRecorder files and h5py files with default settings are
readable.

**k conversion.** For a `map`, the analyser's slit axis and the deflector
angle are converted with the free-electron formula
`k = 0.5123 sqrt(E_kin) sin(angle)`; see `tools/kspace.py`.

**CASSIOPEE energy axes** are straightened to the line through them (the
text files print a fixed number of significant figures).
