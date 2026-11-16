#!/usr/bin/env python3
"""
make_map.py - turn a map_*.csv from mag_logger.py (2-sensor gradiometer) into magnetic field maps.

Needs:  numpy, matplotlib, scipy
        (Ubuntu: sudo apt install python3-numpy python3-matplotlib python3-scipy)

WHAT IS PLOTTED  (--channel)
  grad     gradient = bottom sensor - top sensor (default). Earth's field cancels, the magnet stays.
  bottom   bottom sensor only (like the old single-sensor maps)
  top      top sensor only
  all      all three
  Single-sensor files (mag_logger.py --single, or old files) work too: one plain field map.

BACKGROUND  (--background)
  auto     default. grad: subtract the zero reading taken in mag_logger.py, or point 1 if you
           skipped zeroing. bottom/top: nothing subtracted.
  null     subtract the zero reading (each point uses the latest zero taken before it)
  first    subtract point 1
  median   subtract the median of all points
  none     subtract nothing

USAGE
  python3 make_map.py                          # newest logs/map_*.csv, gradient maps
  python3 make_map.py logs/map_20261112_152821.csv
  python3 make_map.py --channel all            # gradient + bottom + top maps
  python3 make_map.py --baseline 8             # sensor spacing in cm -> gradient in uT/cm
                                               # (taken from the CSV if you gave it to the logger)
  python3 make_map.py --baseline 0             # show the plain difference in uT instead
  python3 make_map.py --magnet 10 10           # where the magnet was (default: grid centre)
  python3 make_map.py --no-show                # only save the images

Saves next to the CSV:
  map_XXXX_overview.png         bottom vs top vs gradient, side by side
  map_XXXX_<channel>.png        three analysis maps (strength, z component, in-plane direction)
  map_XXXX_<channel>_sheet.png  your measurements drawn on the printed grid sheet
  map_XXXX_<channel>_sheet.pdf  same, true scale A4 - print at 100% to lay over the paper sheet
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.interpolate import griddata

LIMIT_XY, LIMIT_Z = 1300.0, 2500.0         # uT, BNO055 magnetometer range per axis
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d9d9d6"
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ2 = LinearSegmentedColormap.from_list("seq_orange", [
    "#fdf3ee", "#fbd3c0", "#f6a985", "#eb6834", "#c4501f", "#8f3512"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", [
    "#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#a32a2a"])

CHANNEL_NAMES = {"grad": "Gradient (bottom - top sensor)",
                 "bottom": "Bottom sensor", "top": "Top sensor",
                 "single": "Magnetic field"}


# ----------------------------------------------------------------- data
def load(path):
    """Read a map CSV. Works for gradiometer files and old single-sensor files."""
    pts, nulls, current_null = {}, [], None
    baseline, bottom_addr = None, None
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        grad = "dbx" in (rd.fieldnames or [])
        for r in rd:
            try:
                if grad:
                    bot = np.array([float(r["bx_bot"]), float(r["by_bot"]), float(r["bz_bot"])])
                    top = np.array([float(r["bx_top"]), float(r["by_top"]), float(r["bz_top"])])
                    rec = {"bottom": bot, "top": top, "grad": bot - top}
                    if (r.get("baseline_cm") or "").strip():
                        baseline = float(r["baseline_cm"])
                    bottom_addr = r.get("bottom_addr") or bottom_addr
                    if (r.get("kind") or "point").strip() == "null":
                        current_null = rec
                        nulls.append(rec)
                        continue
                else:
                    rec = {"single": np.array([float(r["bx"]), float(r["by"]), float(r["bz"])])}
                key = (float(r["x"]), float(r["y"]))
            except (KeyError, ValueError, TypeError):
                continue
            rec["null"] = current_null
            pts[key] = rec                           # later row = redo, wins
    if not pts:
        sys.exit(f"No usable points in {path}")
    keys = list(pts)
    chans = ["grad", "bottom", "top"] if grad else ["single"]
    D = {"x": np.array([k[0] for k in keys]), "y": np.array([k[1] for k in keys]),
         "grad_file": grad, "channels": chans, "n_nulls": len(nulls),
         "baseline": baseline, "bottom_addr": bottom_addr,
         "v": {ch: np.array([pts[k][ch] for k in keys]) for ch in chans}, "null": None}
    if nulls:                                         # points before the first zero use the first zero
        D["null"] = {ch: np.array([(pts[k]["null"] or nulls[0])[ch] for k in keys]) for ch in chans}
    return D


def near_limit(v):
    """True where a sensor axis is at (or very near) the BNO055 range limit."""
    return ((np.abs(v[:, 0]) > 0.95 * LIMIT_XY) | (np.abs(v[:, 1]) > 0.95 * LIMIT_XY)
            | (np.abs(v[:, 2]) > 0.95 * LIMIT_Z))


def resolve_background(mode, ch, D):
    if mode == "auto":
        mode = ("null" if D["null"] is not None else "first") if ch == "grad" else "none"
    if mode == "null" and D["null"] is None:
        print("   (no zero reading in this file -> subtracting point 1 instead)")
        mode = "first"
    return mode


def prepare(D, ch, bg_mode, baseline):
    """Background-subtract (and for the gradient, divide by the baseline) one channel."""
    v = D["v"][ch].astype(float)
    mode = resolve_background(bg_mode, ch, D)
    if mode == "null":
        ref = D["null"][ch]
        desc = "zero reading subtracted"
        if D["n_nulls"] == 1:
            desc += f": {np.round(ref[0], 1)} uT"
    elif mode == "first":
        ref = v[0]
        desc = f"point 1 subtracted: {np.round(ref, 1)} uT"
    elif mode == "median":
        ref = np.median(v, axis=0)
        desc = f"median subtracted: {np.round(ref, 1)} uT"
    else:
        ref, desc = None, "no background subtracted"
    if ref is not None:
        v = v - ref

    unit, floor = "uT", 1.0
    if ch == "grad":
        comp, sym = "ΔB", "|ΔB|"
        if baseline:
            v = v / baseline
            unit, floor = "uT/cm", 1.0 / baseline
            comp, sym = "ΔB/d", "|ΔB|/d"
            desc += f", divided by baseline d = {baseline:g} cm"
    else:
        comp, sym = "B", ("|B|" if ref is None else "|B-B0|")     # B0 = subtracted background

    if ch == "grad":
        over = near_limit(D["v"]["bottom"]) | near_limit(D["v"]["top"])
    else:
        over = near_limit(D["v"][ch])
    return {"ch": ch, "name": CHANNEL_NAMES[ch], "v": v, "s": np.linalg.norm(v, axis=1),
            "over": over, "comp": comp, "sym": sym, "unit": unit, "floor": floor,
            "desc": desc, "mode": mode}


def point_numbers(x, y):
    """Number of each point in the logger's snake order (same as on the printed sheet)."""
    xs, ys = np.unique(x), np.unique(y)
    nums = []
    for xi, yi in zip(x, y):
        i, j = int(np.where(xs == xi)[0][0]), int(np.where(ys == yi)[0][0])
        nums.append(j * len(xs) + (i if j % 2 == 0 else len(xs) - 1 - i) + 1)
    return nums


def interp(x, y, v, gx, gy, method, log=False):
    """Interpolate between measured points. log=True works on log10(v) so a single
    strong spike doesn't create rings/overshoot around it."""
    vv = np.log10(np.clip(v, 1e-3, None)) if log else v
    try:
        out = griddata((x, y), vv, (gx, gy), method=method)
    except Exception:
        out = griddata((x, y), vv, (gx, gy), method="linear")
    return 10 ** out if log else out


def magnitude_norm(values, floor=1.0):
    """Log colour scale when values span a big range (typical near a magnet).
    floor = noise level; weaker values are drawn with the lowest colour."""
    v = values[np.isfinite(values)]
    lo, hi = max(np.nanmin(v), floor), np.nanmax(v)
    if hi / lo > 20:
        return LogNorm(vmin=lo, vmax=hi), True
    return Normalize(vmin=np.nanmin(v), vmax=hi), False


def fmtv(v):
    """Short number for labels: 523, 12, 3.4, 0.27"""
    a = abs(v)
    return f"{v:.0f}" if a >= 10 else (f"{v:.1f}" if a >= 1 else f"{v:.2f}")


def draw_magnet(ax, mx, my, size):
    for sgn in (1, -1):
        ax.plot([mx - size, mx + size], [my - size * sgn, my + size * sgn], color="#e34948",
                lw=3, solid_capstyle="round", alpha=0.9, zorder=5)


# ----------------------------------------------------------- overview figure
def overview_figure(x, y, Pb, Pt, Pg, units, method, magnet, out):
    """Bottom vs top vs gradient strength. Bottom and top share one colour scale."""
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200))
    ext = (x.min(), x.max(), y.min(), y.max())
    shared, shared_log = magnitude_norm(np.r_[Pb["s"], Pt["s"]], Pb["floor"])
    gnorm, glog = magnitude_norm(Pg["s"], Pg["floor"])
    msize = 0.025 * max(x.max() - x.min(), y.max() - y.min())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    what = {"null": "the zero reading", "first": "point 1", "median": "the median of all points",
            "none": "nothing"}[Pg["mode"]]
    fig.suptitle(f"Gradiometer overview   -   subtracted from each map: {what}", fontsize=12, color=INK)
    panels = ((Pb, shared, shared_log, SEQ, "BOTTOM sensor"),
              (Pt, shared, shared_log, SEQ, "TOP sensor"),
              (Pg, gnorm, glog, SEQ2, "GRADIENT = bottom - top"))
    for ax, (P, norm, is_log, cmap, title) in zip(axes, panels):
        g = interp(x, y, P["s"], gx, gy, method, log=is_log)
        im = ax.imshow(g, origin="lower", extent=ext, cmap=cmap, norm=norm, aspect="equal")
        ax.plot(x, y, "o", ms=3.5, mfc=INK, mec="white", mew=0.7)
        draw_magnet(ax, magnet[0], magnet[1], msize)
        ax.set_title(f"{title}: {P['sym']} ({P['unit']}{', log scale' if is_log else ''})",
                     color=INK, fontsize=11)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(P["unit"], color=INK2)
        ax.set_xlabel(f"x ({units})", color=INK2)
        ax.set_ylabel(f"y ({units})", color=INK2)
        ax.tick_params(colors=INK2)
    fig.text(0.5, 0.02, "Bottom and top use the same colours: the top sensor sees much less of the magnet. "
             "Their difference (right) removes Earth's field and anything far away.",
             ha="center", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=150)
    return fig


# ----------------------------------------------------------- analysis figure
def analysis_figure(x, y, P, units, method, out):
    v, b, unit, comp = P["v"], P["s"], P["unit"], P["comp"]
    bx, by, bz = v[:, 0], v[:, 1], v[:, 2]
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200))
    norm, is_log = magnitude_norm(b, P["floor"])
    gB = interp(x, y, b, gx, gy, method, log=is_log)
    gBz = interp(x, y, bz, gx, gy, method)
    bxy = np.hypot(bx, by)
    gBxy = interp(x, y, bxy, gx, gy, method, log=is_log)
    ext = (x.min(), x.max(), y.min(), y.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{P['name']}   -   {P['desc']}", fontsize=12, color=INK)

    ax = axes[0]
    im = ax.imshow(gB, origin="lower", extent=ext, cmap=SEQ, norm=norm, aspect="equal")
    ax.contour(gx, gy, gB, levels=8, colors="white", linewidths=0.6, alpha=0.8,
               norm=norm if is_log else None)
    ax.plot(x, y, "o", ms=4, mfc=INK, mec="white", mew=0.8)
    ax.set_title(f"Strength {P['sym']} ({unit}{', log colour scale' if is_log else ''})", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1]
    lim = max(np.nanmax(np.abs(gBz)), 1e-6)
    im = ax.imshow(gBz, origin="lower", extent=ext, cmap=DIV,
                   norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="equal")
    ax.plot(x, y, "o", ms=4, mfc=INK, mec="white", mew=0.8)
    ax.set_title(f"{comp}z: up (red) / down (blue) through the sheet, {unit}", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[2]
    n2, _ = magnitude_norm(bxy, P["floor"])
    im = ax.imshow(gBxy, origin="lower", extent=ext, cmap=SEQ2, norm=n2, aspect="equal")
    ux, uy = bx / np.maximum(bxy, 1e-9), by / np.maximum(bxy, 1e-9)
    ax.quiver(x, y, ux, uy, color=INK, angles="xy", pivot="middle", width=0.006, scale=12)
    ax.set_title(f"In-plane {comp}: arrows = direction, colour = strength ({unit})", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.85)

    for ax in axes:
        ax.set_xlabel(f"x ({units})", color=INK2)
        ax.set_ylabel(f"y ({units})", color=INK2)
        ax.tick_params(colors=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


# --------------------------------------------------------------- sheet figure
def sheet_figure(x, y, P, magnet, method, csv_name, out_base):
    """Draw the data on the same A4 layout as grid_sheet_A4.pdf (true scale when it fits)."""
    v, b, sym, unit = P["v"], P["s"], P["sym"], P["unit"]
    bx, by = v[:, 0], v[:, 1]
    ref_first = P["mode"] == "first"
    W, H = 21.0, 29.7
    OX, OY = 0.5, 3.9
    gw, gh = x.max() - x.min(), y.max() - y.min()
    s = min(1.0, 20.0 / max(gw, gh, 1e-9))            # cm on paper per unit of x/y
    true_scale = s == 1.0
    P_ = lambda px, py: (OX + (px - x.min()) * s, OY + (py - y.min()) * s)

    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    # heatmap of the field inside the grid square
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), 300), np.linspace(y.min(), y.max(), 300))
    norm, is_log = magnitude_norm(b, P["floor"])
    gB = interp(x, y, b, gx, gy, method, log=is_log)
    (x0, y0), (x1, y1) = P_(x.min(), y.min()), P_(x.max(), y.max())
    im = ax.imshow(gB, origin="lower", extent=(x0, x1, y0, y1), cmap=SEQ, norm=norm,
                   interpolation="bilinear", zorder=1)
    cs = ax.contour(OX + (gx - x.min()) * s, OY + (gy - y.min()) * s, gB, levels=6,
                    colors="white", linewidths=0.7, alpha=0.85, zorder=2,
                    norm=norm if is_log else None)
    ax.clabel(cs, fontsize=6.5, fmt=fmtv, colors="white")

    # grid lines through the measured rows/cols
    for xv in np.unique(x):
        ax.plot([P_(xv, 0)[0]] * 2, [y0, y1], color="#333", lw=0.9, alpha=0.55, zorder=3)
    for yv in np.unique(y):
        ax.plot([x0, x1], [P_(0, yv)[1]] * 2, color="#333", lw=0.9, alpha=0.55, zorder=3)

    # route arrows (faint, so the data stays the focus)
    nums = point_numbers(x, y)
    order = np.argsort(nums)
    for a, c in zip(order, order[1:]):
        ax.add_patch(FancyArrowPatch(P_(x[a], y[a]), P_(x[c], y[c]), arrowstyle="-|>",
                                     mutation_scale=12, color="white", lw=1.6, alpha=0.6,
                                     shrinkA=16, shrinkB=16, zorder=4))

    # magnet X
    mx, my = P_(*magnet)
    draw_magnet(ax, mx, my, 0.7)

    # each measured point: dot, number, field direction arrow, value
    bxy = np.hypot(bx, by)
    for i in range(len(x)):
        px, py = P_(x[i], y[i])
        right = x[i] < x.max()
        top = y[i] < y.max()
        if bxy[i] > 1e-6:                               # in-plane direction, fixed length
            L = 1.0
            ax.add_patch(FancyArrowPatch((px, py), (px + L * bx[i] / bxy[i], py + L * by[i] / bxy[i]),
                                         arrowstyle="-|>", mutation_scale=13, color=INK, lw=1.8,
                                         shrinkA=0, shrinkB=0, zorder=6))
        over = bool(P["over"][i])
        ax.add_patch(Circle((px, py), 0.2, fc=INK, ec="white", lw=1.4, zorder=7))
        if over:
            ax.add_patch(Circle((px, py), 0.45, fc="none", ec="#fab219", lw=2.2, zorder=7))
        sx, sy = (1 if right else -1), (1 if top else -1)
        ha, va = ("left" if right else "right"), ("bottom" if top else "top")
        ax.text(px + 0.3 * sx, py + 0.3 * sy, f"{nums[i]}", fontsize=9.5, fontweight="bold",
                ha=ha, va=va, zorder=8,
                bbox=dict(boxstyle="circle,pad=0.15", fc="#fff4c2", ec="#b8860b", lw=0.7))
        label = "ref 0" if (ref_first and i == 0) else fmtv(b[i]) + (" (!)" if over else "")
        ax.text(px + 0.95 * sx, py + 0.33 * sy, label, fontsize=7.8, color=INK, ha=ha, va=va, zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    # axis numbers
    for xv in np.unique(x):
        ax.text(P_(xv, 0)[0], y0 - 0.55, f"{xv:g}", fontsize=8, ha="center", va="top", color=INK2)
    ax.text(x1, y0 - 1.1, "x (cm) ->", fontsize=8.5, ha="right", va="top", color=INK2)

    # header
    title = ("Measured magnetic field on your grid sheet" if P["ch"] == "single"
             else f"{P['name']} on your grid sheet")
    ax.text(W / 2, H - 0.7, title, fontsize=14, fontweight="bold", ha="center", va="top", color=INK)
    ax.text(W / 2, H - 1.45, f"{csv_name}   |   {P['desc']}", fontsize=8.5, ha="center", va="top",
            color=INK2)
    notes = (f"Colour = {sym} in {unit} ({'log scale' if is_log else 'linear'}); white lines join equal values.\n"
             f"White box next to each point number = measured {sym} in {unit}"
             + (" ( ref = the subtracted point 1 )" if ref_first else "")
             + ". Black arrow = direction along the paper\n"
             "(assumes the chips' X axis pointed right and Y pointed up the page).  "
             "Yellow ring + (!) = a sensor at its limit, not reliable.\n"
             + ("Printed at 100% this page lines up with grid_sheet_A4.pdf." if true_scale
                else f"NOT to scale: grid shrunk to fit the page ({s:.2f} cm per cm)."))
    ax.text(1.0, H - 2.15, notes, fontsize=7.8, ha="left", va="top", color=INK, linespacing=1.5)

    # colour bar along the bottom
    cax = fig.add_axes([0.06, 0.055, 0.6, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{sym} ({unit})", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    ax.text(15.2, 1.45, "X = magnet position\n(yellow) = point number", fontsize=8, color=INK2, va="center")

    fig.savefig(out_base + ".png", dpi=150)
    fig.savefig(out_base + ".pdf")
    return fig


# ---------------------------------------------------------------- line plot
def line_plot(pos, label, P, out):
    order = np.argsort(pos)
    v, comp = P["v"], P["comp"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for vals, name, col in ((v[:, 0], f"{comp}x", "#2a78d6"), (v[:, 1], f"{comp}y", "#eb6834"),
                            (v[:, 2], f"{comp}z", "#1baf7a"), (P["s"], P["sym"], INK)):
        ax.plot(pos[order], vals[order], "o-", label=name, color=col, lw=2 if name == P["sym"] else 1.5)
    ax.set_xlabel(label); ax.set_ylabel(f"field ({P['unit']})")
    ax.set_title(f"{P['name']} profile   -   {P['desc']}", fontsize=10)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150)
    return fig


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Make magnetic field / gradient maps from mag_logger.py output.")
    ap.add_argument("csv", nargs="?", help="map_*.csv file (default: newest in ./logs)")
    ap.add_argument("--channel", choices=["grad", "bottom", "top", "all"], default="grad",
                    help="what to map (default: grad = bottom - top)")
    ap.add_argument("--background", choices=["auto", "null", "first", "median", "none"], default="auto",
                    help="subtract a background field vector before plotting")
    ap.add_argument("--baseline", type=float, default=None,
                    help="distance between the two sensors in cm (0 = show plain difference in uT)")
    ap.add_argument("--method", choices=["cubic", "linear", "nearest"], default="cubic",
                    help="interpolation between measured points")
    ap.add_argument("--magnet", type=float, nargs=2, metavar=("X", "Y"),
                    help="magnet position for the sheet view (default: grid centre)")
    ap.add_argument("--units", default="cm", help="units of x and y (labels only)")
    ap.add_argument("--no-show", action="store_true", help="only save the images")
    args = ap.parse_args()

    path = args.csv
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))
        files = sorted(glob.glob(os.path.join(here, "logs", "map_*.csv")), key=os.path.getmtime)
        if not files:
            sys.exit("No logs/map_*.csv found. Run mag_logger.py first or give a file name.")
        path = files[-1]
    print(f"Using {path}")

    D = load(path)
    x, y = D["x"], D["y"]
    if D["grad_file"]:
        chans = D["channels"] if args.channel == "all" else [args.channel]
        baseline = args.baseline if args.baseline is not None else D["baseline"]
        baseline = baseline if baseline and baseline > 0 else None
        print(f"{len(x)} points, {D['n_nulls']} zero reading(s), bottom = sensor {D['bottom_addr'] or 'A'}"
              + (f", baseline {baseline:g} cm" if baseline else ""))
    else:
        chans, baseline = ["single"], None
        print(f"{len(x)} points (single-sensor file)")

    base = os.path.splitext(path)[0]
    is_line = len(np.unique(y)) == 1 or len(np.unique(x)) == 1 or len(x) < 4
    magnet = args.magnet or ((x.min() + x.max()) / 2, (y.min() + y.max()) / 2)
    saved = []

    if D["grad_file"] and not is_line:
        bg = resolve_background(args.background, "grad", D)
        Pg = prepare(D, "grad", bg, baseline)
        Pb, Pt = prepare(D, "bottom", bg, None), prepare(D, "top", bg, None)
        overview_figure(x, y, Pb, Pt, Pg, args.units, args.method, magnet, base + "_overview.png")
        saved.append(base + "_overview.png")

    for ch in chans:
        P = prepare(D, ch, args.background, baseline)
        suffix = "" if ch == "single" else "_" + ch
        print(f"[{P['name']}] {P['desc']}")
        print(f"    {P['sym']}  min={P['s'].min():.2f}  max={P['s'].max():.2f}  mean={P['s'].mean():.2f} {P['unit']}")
        n_over = int(P["over"].sum())
        if n_over:
            print(f"    WARNING: {n_over} point(s) with a sensor at its limit (~{LIMIT_XY:.0f} uT x/y,"
                  f" {LIMIT_Z:.0f} uT z) - move the magnet further away.")
        if is_line:
            pos, label = (x, f"x ({args.units})") if len(np.unique(y)) == 1 else (y, f"y ({args.units})")
            line_plot(pos, label, P, base + suffix + ".png")
            saved.append(base + suffix + ".png")
        else:
            analysis_figure(x, y, P, args.units, args.method, base + suffix + ".png")
            sheet_figure(x, y, P, magnet, args.method, os.path.basename(path), base + suffix + "_sheet")
            saved += [base + suffix + ".png", base + suffix + "_sheet.png", base + suffix + "_sheet.pdf"]

    print("Saved:\n  " + "\n  ".join(saved))
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
