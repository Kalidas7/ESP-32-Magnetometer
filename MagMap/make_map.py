#!/usr/bin/env python3
"""
make_map.py - turn a map_*.csv from mag_logger.py into magnetic field maps.

Needs:  numpy, matplotlib, scipy
        (Ubuntu: sudo apt install python3-numpy python3-matplotlib python3-scipy)

USAGE
  python3 make_map.py                          # newest logs/map_*.csv
  python3 make_map.py logs/map_20261109_152821.csv
  python3 make_map.py --background first       # subtract point 1 (Earth's field + sensor offset)
  python3 make_map.py --background median      # subtract the median of all points
  python3 make_map.py --magnet 10 10           # where the magnet was (default: grid centre)
  python3 make_map.py --no-show                # only save the images

Saves next to the CSV:
  map_XXXX.png         three analysis maps (strength, Bz, in-plane direction)
  map_XXXX_sheet.png   your measurements drawn on the printed grid sheet
  map_XXXX_sheet.pdf   same, true scale A4 - print at 100% to lay over the paper sheet
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.interpolate import griddata

SENSOR_LIMIT = 1300.0                      # uT, BNO055 x/y range (z is ~2500)
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d9d9d6"
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ2 = LinearSegmentedColormap.from_list("seq_orange", [
    "#fdf3ee", "#fbd3c0", "#f6a985", "#eb6834", "#c4501f", "#8f3512"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", [
    "#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#a32a2a"])


# ----------------------------------------------------------------- data
def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                key = (float(r["x"]), float(r["y"]))
                rows[key] = (float(r["bx"]), float(r["by"]), float(r["bz"]))  # later row = redo, wins
            except (KeyError, ValueError):
                continue
    if not rows:
        sys.exit(f"No usable points in {path}")
    xy = np.array(list(rows.keys()))
    b = np.array(list(rows.values()))
    return xy[:, 0], xy[:, 1], b[:, 0], b[:, 1], b[:, 2]


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


def magnitude_norm(values):
    """Log colour scale when values span a big range (typical near a magnet)."""
    v = values[np.isfinite(values)]
    lo, hi = max(np.nanmin(v), 1.0), np.nanmax(v)
    if hi / lo > 20:
        return LogNorm(vmin=lo, vmax=hi), True
    return Normalize(vmin=np.nanmin(v), vmax=hi), False


# ----------------------------------------------------------- analysis figure
def analysis_figure(x, y, bx, by, bz, b, sym, title_extra, units, method, out):
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200))
    norm, is_log = magnitude_norm(b)
    gB = interp(x, y, b, gx, gy, method, log=is_log)
    gBz = interp(x, y, bz, gx, gy, method)
    bxy = np.hypot(bx, by)
    gBxy = interp(x, y, bxy, gx, gy, method, log=is_log)
    ext = (x.min(), x.max(), y.min(), y.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Magnetic field map" + title_extra, fontsize=13, color=INK)

    ax = axes[0]
    im = ax.imshow(gB, origin="lower", extent=ext, cmap=SEQ, norm=norm, aspect="equal")
    ax.contour(gx, gy, gB, levels=8, colors="white", linewidths=0.6, alpha=0.8,
               norm=norm if is_log else None)
    ax.plot(x, y, "o", ms=4, mfc=INK, mec="white", mew=0.8)
    ax.set_title(f"Total field strength {sym} (uT{', log colour scale' if is_log else ''})", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1]
    lim = max(np.nanmax(np.abs(gBz)), 1e-6)
    im = ax.imshow(gBz, origin="lower", extent=ext, cmap=DIV,
                   norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="equal")
    ax.plot(x, y, "o", ms=4, mfc=INK, mec="white", mew=0.8)
    ax.set_title("Bz: field up (red) / down (blue) through the sheet, uT", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[2]
    n2, _ = magnitude_norm(bxy)
    im = ax.imshow(gBxy, origin="lower", extent=ext, cmap=SEQ2, norm=n2, aspect="equal")
    ux, uy = bx / np.maximum(bxy, 1e-9), by / np.maximum(bxy, 1e-9)
    ax.quiver(x, y, ux, uy, color=INK, angles="xy", pivot="middle", width=0.006, scale=12)
    ax.set_title("In-plane field: arrows = direction, colour = strength (uT)", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.85)

    for ax in axes:
        ax.set_xlabel(f"x ({units})", color=INK2)
        ax.set_ylabel(f"y ({units})", color=INK2)
        ax.tick_params(colors=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


# --------------------------------------------------------------- sheet figure
def sheet_figure(x, y, bx, by, b, b_raw, sym, subtitle, magnet, method, csv_name, out_base,
                 args_ref_first=False):
    """Draw the data on the same A4 layout as grid_sheet_A4.pdf (true scale when it fits)."""
    W, H = 21.0, 29.7
    OX, OY = 0.5, 3.9
    gw, gh = x.max() - x.min(), y.max() - y.min()
    s = min(1.0, 20.0 / max(gw, gh, 1e-9))            # cm on paper per unit of x/y
    true_scale = s == 1.0
    P = lambda px, py: (OX + (px - x.min()) * s, OY + (py - y.min()) * s)

    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    # heatmap of the field inside the grid square
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), 300), np.linspace(y.min(), y.max(), 300))
    norm, is_log = magnitude_norm(b)
    gB = interp(x, y, b, gx, gy, method, log=is_log)
    (x0, y0), (x1, y1) = P(x.min(), y.min()), P(x.max(), y.max())
    im = ax.imshow(gB, origin="lower", extent=(x0, x1, y0, y1), cmap=SEQ, norm=norm,
                   interpolation="bilinear", zorder=1)
    cs = ax.contour(OX + (gx - x.min()) * s, OY + (gy - y.min()) * s, gB, levels=6,
                    colors="white", linewidths=0.7, alpha=0.85, zorder=2,
                    norm=norm if is_log else None)
    ax.clabel(cs, fontsize=6.5, fmt=lambda v: f"{v:.0f}", colors="white")

    # grid lines (1 unit fine, measured rows/cols bold)
    for v in np.unique(x):
        ax.plot([P(v, 0)[0]] * 2, [y0, y1], color="#333", lw=0.9, alpha=0.55, zorder=3)
    for v in np.unique(y):
        ax.plot([x0, x1], [P(0, v)[1]] * 2, color="#333", lw=0.9, alpha=0.55, zorder=3)

    # route arrows (faint, so the data stays the focus)
    nums = point_numbers(x, y)
    order = np.argsort(nums)
    for a, c in zip(order, order[1:]):
        ax.add_patch(FancyArrowPatch(P(x[a], y[a]), P(x[c], y[c]), arrowstyle="-|>",
                                     mutation_scale=12, color="white", lw=1.6, alpha=0.6,
                                     shrinkA=16, shrinkB=16, zorder=4))

    # magnet X
    mx, my = P(*magnet)
    for sgn in (1, -1):
        ax.plot([mx - 0.7, mx + 0.7], [my - 0.7 * sgn, my + 0.7 * sgn], color="#e34948",
                lw=3.5, solid_capstyle="round", alpha=0.9, zorder=5)

    # each measured point: dot, number, field direction arrow, value
    bxy = np.hypot(bx, by)
    for i in range(len(x)):
        px, py = P(x[i], y[i])
        right = x[i] < x.max()
        top = y[i] < y.max()
        if bxy[i] > 1e-6:                               # in-plane direction, fixed length
            L = 1.0
            ax.add_patch(FancyArrowPatch((px, py), (px + L * bx[i] / bxy[i], py + L * by[i] / bxy[i]),
                                         arrowstyle="-|>", mutation_scale=13, color=INK, lw=1.8,
                                         shrinkA=0, shrinkB=0, zorder=6))
        over = b_raw[i] > SENSOR_LIMIT
        ax.add_patch(Circle((px, py), 0.2, fc=INK, ec="white", lw=1.4, zorder=7))
        if over:
            ax.add_patch(Circle((px, py), 0.45, fc="none", ec="#fab219", lw=2.2, zorder=7))
        sx, sy = (1 if right else -1), (1 if top else -1)
        ha, va = ("left" if right else "right"), ("bottom" if top else "top")
        ax.text(px + 0.3 * sx, py + 0.3 * sy, f"{nums[i]}", fontsize=9.5, fontweight="bold",
                ha=ha, va=va, zorder=8,
                bbox=dict(boxstyle="circle,pad=0.15", fc="#fff4c2", ec="#b8860b", lw=0.7))
        label = "ref 0" if (args_ref_first and i == 0) else f"{b[i]:.0f}" + (" (!)" if over else "")
        ax.text(px + 0.95 * sx, py + 0.33 * sy, label, fontsize=7.8, color=INK, ha=ha, va=va, zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    # axis numbers
    for v in np.unique(x):
        ax.text(P(v, 0)[0], y0 - 0.55, f"{v:g}", fontsize=8, ha="center", va="top", color=INK2)
    ax.text(x1, y0 - 1.1, "x (cm) ->", fontsize=8.5, ha="right", va="top", color=INK2)

    # header
    ax.text(W / 2, H - 0.7, "Measured magnetic field on your grid sheet", fontsize=14,
            fontweight="bold", ha="center", va="top", color=INK)
    ax.text(W / 2, H - 1.45, f"{csv_name}   |   {subtitle}", fontsize=8.5, ha="center", va="top", color=INK2)
    notes = (f"Colour = {sym} ({'log scale' if is_log else 'linear'}); white lines join equal strength (uT).\n"
             "White box next to each point number = measured value in uT ( ref = the subtracted point 1 ). Black arrow = field direction along the paper\n"
             "(assumes the chip's X axis pointed right and Y pointed up the page).  "
             "Yellow ring + (!) = above the sensor limit, not reliable.\n"
             + ("Printed at 100% this page lines up with grid_sheet_A4.pdf." if true_scale
                else f"NOT to scale: grid shrunk to fit the page ({s:.2f} cm per cm)."))
    ax.text(1.0, H - 2.15, notes, fontsize=7.8, ha="left", va="top", color=INK, linespacing=1.5)

    # colour bar along the bottom
    cax = fig.add_axes([0.06, 0.055, 0.6, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{sym} (uT)", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    ax.text(15.2, 1.45, "X = magnet position\n(yellow) = point number", fontsize=8, color=INK2, va="center")

    fig.savefig(out_base + ".png", dpi=150)
    fig.savefig(out_base + ".pdf")
    return fig


# ---------------------------------------------------------------------- main
def line_plot(pos, label, bx, by, bz, b, title, out):
    order = np.argsort(pos)
    fig, ax = plt.subplots(figsize=(9, 5))
    for v, name, col in ((bx, "Bx", "#2a78d6"), (by, "By", "#eb6834"), (bz, "Bz", "#1baf7a"), (b, "|B|", INK)):
        ax.plot(pos[order], v[order], "o-", label=name, color=col, lw=2 if name == "|B|" else 1.5)
    ax.set_xlabel(label); ax.set_ylabel("field (uT)"); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150)
    return fig


def main():
    ap = argparse.ArgumentParser(description="Make magnetic field maps from mag_logger.py output.")
    ap.add_argument("csv", nargs="?", help="map_*.csv file (default: newest in ./logs)")
    ap.add_argument("--background", choices=["none", "first", "median"], default="none",
                    help="subtract a background field vector before plotting")
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

    x, y, bx, by, bz = load(path)
    b_raw = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    title_extra, subtitle = "", "no background subtracted"
    if args.background != "none":
        ref = (np.array([bx[0], by[0], bz[0]]) if args.background == "first"
               else np.median(np.c_[bx, by, bz], axis=0))
        bx, by, bz = bx - ref[0], by - ref[1], bz - ref[2]
        subtitle = f"background ({args.background} point) subtracted: {np.round(ref, 1)} uT"
        title_extra = "  -  " + subtitle
        print(f"Subtracted background vector {ref.round(2)} uT")
    b = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    sym = "|dB|" if args.background != "none" else "|B|"
    base = os.path.splitext(path)[0]
    print(f"{len(x)} points   {sym} min={b.min():.2f}  max={b.max():.2f}  mean={b.mean():.2f} uT")
    over = int((b_raw > SENSOR_LIMIT).sum())
    if over:
        print(f"WARNING: {over} point(s) above the sensor limit (~{SENSOR_LIMIT:.0f} uT) - move the magnet further away.")

    if len(np.unique(y)) == 1 or len(np.unique(x)) == 1 or len(x) < 4:
        pos, label = (x, f"x ({args.units})") if len(np.unique(y)) == 1 else (y, f"y ({args.units})")
        line_plot(pos, label, bx, by, bz, b, "Magnetic field profile" + title_extra, base + ".png")
        print(f"Saved {base}.png")
    else:
        analysis_figure(x, y, bx, by, bz, b, sym, title_extra, args.units, args.method, base + ".png")
        magnet = args.magnet or ((x.min() + x.max()) / 2, (y.min() + y.max()) / 2)
        sheet_figure(x, y, bx, by, b, b_raw, sym, subtitle, magnet, args.method,
                     os.path.basename(path), base + "_sheet", args.background == "first")
        print(f"Saved {base}.png\nSaved {base}_sheet.png  and  {base}_sheet.pdf (true-scale A4)")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
