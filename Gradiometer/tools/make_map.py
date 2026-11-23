#!/usr/bin/env python3
"""
make_map.py - turn a grid scan into magnetic maps and find the magnet.

    python3 make_map.py magnet.csv --nx 9 --ny 9 --spacing 5 --background background.csv --cal calibration.json

HOW THE SCAN MUST BE DONE
  - A grid of nx x ny points, --spacing cm apart. Point 1 is a corner. Go along the first row
    (the x direction) point by point, then start the next row from the SAME side (y grows by one step).
  - The stick lies along the rows (x) and always points the same way. Never turn it around.
    Put the stick's centre (the middle between the two chips) on each grid point.
  - At each point: press BOOT, hold still while the LED is on, move when it goes off.

WHAT IT DOES
  1. Each point: averages the readings 0.5 s to 5.5 s after the press. 5 s also cancels B's 5-second pulse.
  2. Applies calibration.json if given. Without it, raw values are used, which is fine for a map
     as long as the stick never turns.
  3. Subtracts the background scan (same grid, no magnet). Without one, subtracts the median of the
     edge points.
  4. Draws maps of A - B (X, Y, Z and size) and of sensor A alone.
  5. Fits one small magnet (dipole): position, depth below the sensors, strength and direction.
     It fits twice, on A and B separately and on A - B only. If both agree, trust it more.
Saves <name>_map.png, <name>_points.csv and <name>_fit.json.
Needs: numpy, matplotlib
"""
import argparse
import csv
import json
import sys

import numpy as np


# ------------------------------------------------------------------ reading the scan
def load(path):
    d = np.genfromtxt(path, delimiter=",", skip_header=1, comments="#")
    if d.ndim != 2 or d.shape[1] != 8 or len(d) < 30:
        sys.exit(f"{path}: expected rows of time_ms,point,Ax,Ay,Az,Bx,By,Bz")
    d = d[10:]
    for c in (slice(2, 5), slice(5, 8)):
        d[np.all(d[:, c] == 0, axis=1), c] = np.nan
    return d[:, 0] / 1000.0, d[:, 1].astype(int), d[:, 2:5], d[:, 5:8]


def apply_cal(A, B, cal):
    A = (A - np.array(cal["A"]["offset"])) @ np.array(cal["A"]["matrix"]).T
    B = (B - np.array(cal["B"]["offset"])) @ np.array(cal["B"]["matrix"]).T
    B = B @ np.array(cal["B_to_A"]["matrix"]).T + np.array(cal["B_to_A"]["offset"])
    return A, B


def point_means(path, npts, settle, avg, cal):
    """Average A and B over each point's hold window. Returns a list (None = missing) and notes."""
    t, pt, A, B = load(path)
    if cal is not None:
        A, B = apply_cal(A, B, cal)
    out, notes = [], []
    for k in range(1, npts + 1):
        idx = np.flatnonzero(pt == k)
        if len(idx) == 0:
            out.append(None)
            notes.append(f"point {k}: missing")
            continue
        t0 = t[idx[0]]
        w = idx[(t[idx] >= t0 + settle) & (t[idx] <= t0 + settle + avg)]
        w = w[~(np.isnan(A[w]).any(axis=1) | np.isnan(B[w]).any(axis=1))]
        if len(w) < 5:
            out.append(None)
            notes.append(f"point {k}: too few readings (moved on too early?)")
            continue
        if t[idx[-1]] - t0 < settle + 0.9 * avg:
            notes.append(f"point {k}: next press came early, averaged only {t[idx[-1]] - t0 - settle:.1f} s")
        D = A[w] - B[w]
        out.append({"a": A[w].mean(axis=0), "b": B[w].mean(axis=0),
                    "sd": D.std(axis=0, ddof=1), "n": len(w)})
    extra = int(pt.max()) - npts
    if extra > 0:
        notes.append(f"{extra} more point(s) were pressed than the grid has - check --nx/--ny")
    return out, notes


# ------------------------------------------------------------------ geometry
ROT = {"+x": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "-x": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
       "+y": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "-y": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]}


def dipole_G(P, S):
    """Field (uT) at points P (N,3, metres) from a unit moment (1 A m^2) along x, y, z at S: (N,3,3)."""
    r = P - S
    r2 = np.sum(r * r, axis=1)
    k = 0.1 / r2 ** 2.5
    G = 3 * r[:, :, None] * r[:, None, :] - r2[:, None, None] * np.eye(3)[None]
    return G * k[:, None, None]


def fit_dipole(PA, PB, a, b, mode, box):
    """mode 'both': fit A and B readings (each with its own constant offset).
       mode 'diff': fit A - B only (with one constant offset).
       Returns dict with source position (m), moment (A m^2), rms misfit (uT)."""
    n = len(PA)

    def solve(S):
        GA, GB = dipole_G(PA, S), dipole_G(PB, S)
        if mode == "both":
            I = np.eye(3)[None].repeat(n, axis=0)
            Z = np.zeros_like(I)
            top = np.concatenate([GA, I, Z], axis=2).reshape(3 * n, 9)
            bot = np.concatenate([GB, Z, I], axis=2).reshape(3 * n, 9)
            X, y = np.vstack([top, bot]), np.concatenate([a.ravel(), b.ravel()])
        else:
            I = np.eye(3)[None].repeat(n, axis=0)
            X = np.concatenate([GA - GB, I], axis=2).reshape(3 * n, 6)
            y = (a - b).ravel()
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        return float(np.sqrt(np.mean(res ** 2))), coef

    (x0, x1), (y0, y1), zmax, step = box
    best = (np.inf, None, None)
    for sx in np.arange(x0, x1 + 1e-9, step):
        for sy in np.arange(y0, y1 + 1e-9, step):
            for sz in np.arange(0.02, zmax + 1e-9, 0.02):
                S = np.array([sx, sy, -sz])
                r, c = solve(S)
                if r < best[0]:
                    best = (r, S, c)
    rms, S, coef = best
    st = step / 2
    while st > 0.0005:                                   # refine: small steps in each direction
        improved = False
        for k in range(3):
            for sg in (-1, 1):
                T = S.copy()
                T[k] += sg * st
                if T[2] > -0.01:
                    continue
                r, c = solve(T)
                if r < rms:
                    rms, S, coef, improved = r, T, c, True
        if not improved:
            st /= 2
    return {"pos": S, "moment": coef[:3], "rms": rms}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Make magnetic maps from a grid scan and find the magnet.")
    ap.add_argument("csv", help="the scan with the magnet")
    ap.add_argument("--nx", type=int, required=True, help="points per row")
    ap.add_argument("--ny", type=int, required=True, help="number of rows")
    ap.add_argument("--spacing", type=float, required=True, help="grid spacing in cm")
    ap.add_argument("--background", help="the same grid scanned without the magnet")
    ap.add_argument("--cal", help="calibration.json from calibrate.py")
    ap.add_argument("--settle", type=float, default=0.5, help="seconds to skip after each press")
    ap.add_argument("--avg", type=float, default=5.0, help="seconds to average at each point")
    ap.add_argument("--baseline", type=float, default=16.0, help="distance between the two chips in cm")
    ap.add_argument("--a-end", choices=["left", "right"], default="left",
                    help="which end of the stick sensor A is on (looking at the grid with x to the right)")
    ap.add_argument("--x-arrow", choices=list(ROT), default="+x",
                    help="which way the X arrows printed on the boards point on the grid")
    ap.add_argument("--b-height", type=float, default=0.0, help="how much higher B's chip sits than A's, cm")
    ap.add_argument("--max-depth", type=float, default=60.0, help="deepest depth the fit searches, cm")
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    npts = args.nx * args.ny
    cal = None
    if args.cal:
        with open(args.cal) as f:
            cal = json.load(f)
    s = args.spacing / 100.0
    R = np.array(ROT[args.x_arrow], float)

    pts, notes = point_means(args.csv, npts, args.settle, args.avg, cal)
    if args.background:
        bg, bnotes = point_means(args.background, npts, args.settle, args.avg, cal)
        notes += ["background " + n for n in bnotes]
    good = [i for i, p in enumerate(pts) if p is not None]
    if len(good) < 6:
        sys.exit("Fewer than 6 usable points. Check --nx/--ny and that you pressed BOOT at each point.")

    A = np.full((npts, 3), np.nan)
    B = np.full((npts, 3), np.nan)
    SD = np.full((npts, 3), np.nan)
    for i in good:
        A[i], B[i], SD[i] = pts[i]["a"], pts[i]["b"], pts[i]["sd"]

    # ---- subtract the reference
    if args.background:
        bgA = np.array([p["a"] if p is not None else [np.nan] * 3 for p in bg])
        bgB = np.array([p["b"] if p is not None else [np.nan] * 3 for p in bg])
        for X in (bgA, bgB):                              # missing background points: use the median
            X[np.isnan(X).any(axis=1)] = np.nanmedian(X, axis=0)
        A, B = A - bgA, B - bgB
        ref_note = "background scan subtracted point by point"
    else:
        col = np.arange(npts) % args.nx
        row = np.arange(npts) // args.nx
        edge = (col == 0) | (col == args.nx - 1) | (row == 0) | (row == args.ny - 1)
        A = A - np.nanmedian(A[edge], axis=0)
        B = B - np.nanmedian(B[edge], axis=0)
        ref_note = "no background scan: median of the edge points subtracted"

    # ---- into grid directions (x along rows, y across rows, z up)
    A, B, SD = A @ R.T, B @ R.T, np.abs(SD @ R.T)
    D = A - B
    col = np.arange(npts) % args.nx
    row = np.arange(npts) // args.nx
    xs, ys = col * s, row * s

    print(f"\n{args.csv}: grid {args.nx} x {args.ny}, spacing {args.spacing:g} cm, "
          f"{len(good)} of {npts} points usable")
    print(f"calibration: {'applied (' + args.cal + ')' if cal else 'none (raw values)'}; {ref_note}")
    for n in notes:
        print("  note:", n)
    med_sd = np.nanmedian(SD, axis=0)
    bad = [i + 1 for i in good if np.any(SD[i] > 4 * med_sd + 0.2)]
    if bad:
        print(f"  note: points {bad} jittered a lot while holding - moved or bumped? Consider redoing them.")
    dmag = np.linalg.norm(D, axis=1)
    peak = int(np.nanargmax(dmag))
    print(f"\nStrongest A - B: {dmag[peak]:.2f} uT at point {peak + 1} "
          f"(x {xs[peak] * 100:.0f} cm, y {ys[peak] * 100:.0f} cm)")
    print(f"Jitter while holding (median SD of A - B): x {med_sd[0]:.2f}, y {med_sd[1]:.2f}, z {med_sd[2]:.2f} uT")

    # ---- save the per-point table
    stem = args.csv.rsplit(".", 1)[0]
    with open(stem + "_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point", "x_cm", "y_cm", "dX_uT", "dY_uT", "dZ_uT", "size_A-B_uT",
                    "A_x_uT", "A_y_uT", "A_z_uT", "B_x_uT", "B_y_uT", "B_z_uT"])
        for i in range(npts):
            w.writerow([i + 1, round(xs[i] * 100, 2), round(ys[i] * 100, 2)] +
                       [round(v, 3) if np.isfinite(v) else "" for v in list(D[i]) + [dmag[i]] + list(A[i]) + list(B[i])])

    # ---- fit
    fits = {}
    if not args.no_fit:
        half = args.baseline / 200.0
        sign = -1 if args.a_end == "left" else 1
        PA = np.column_stack([xs + sign * half, ys, np.zeros(npts)])
        PB = np.column_stack([xs - sign * half, ys, np.full(npts, args.b_height / 100.0)])
        ok = np.array([i in good for i in range(npts)])
        margin = max(2 * s, 0.10)
        box = ((-margin, xs.max() + margin), (-margin, ys.max() + margin), args.max_depth / 100.0, s / 2)
        print("\nFitting one small magnet (this takes a few seconds)...")
        for mode in ("both", "diff"):
            fits[mode] = fit_dipole(PA[ok], PB[ok], A[ok], B[ok], mode, box)
        print(f"\n{'fit on':<18}|{'x cm':>7} |{'y cm':>7} |{'depth cm':>9} |{'strength A m2':>14} |{'tilt deg':>9} |{'misfit uT':>10}")
        print("-" * 18 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 10 + "+" + "-" * 15 + "+" + "-" * 10 + "+" + "-" * 11)
        for mode, label in (("both", "A and B separately"), ("diff", "A - B only")):
            f_ = fits[mode]
            m = f_["moment"]
            tilt = np.degrees(np.arccos(abs(m[2]) / max(np.linalg.norm(m), 1e-12)))
            f_["tilt"] = tilt
            print(f"{label:<18}|{f_['pos'][0] * 100:>7.1f} |{f_['pos'][1] * 100:>7.1f} |{-f_['pos'][2] * 100:>9.1f} |"
                  f"{np.linalg.norm(m):>14.3f} |{tilt:>9.0f} |{f_['rms']:>10.2f}")
        fb, fd = fits["both"], fits["diff"]
        gap = np.linalg.norm(fb["pos"] - fd["pos"]) * 100
        print("depth = below the sensor chips (subtract the sheet thickness for depth below the surface).")
        print("tilt = 0 means the magnet stands upright, 90 means it lies flat.")
        if dmag[peak] < 5 * max(fd["rms"], 0.05):
            print("-> The signal is weak compared with the misfit: the fit is unreliable. Use a shallower magnet or a finer grid.")
        elif not (-0.5 * s <= fb["pos"][0] <= xs.max() + 0.5 * s and -0.5 * s <= fb["pos"][1] <= ys.max() + 0.5 * s):
            print("-> The magnet seems to be outside the scanned area. Extend the grid on that side.")
        elif gap > max(3.0, 0.15 * (-fb["pos"][2] * 100)):
            print(f"-> The two fits disagree by {gap:.1f} cm: treat the result as rough.")
        else:
            print(f"-> The two fits agree within {gap:.1f} cm. Compare them with your ruler.")
        with open(stem + "_fit.json", "w") as f:
            json.dump({mode: {"x_cm": round(fv["pos"][0] * 100, 2), "y_cm": round(fv["pos"][1] * 100, 2),
                              "depth_below_sensors_cm": round(-fv["pos"][2] * 100, 2),
                              "moment_Am2": [round(v, 4) for v in fv["moment"]],
                              "strength_Am2": round(float(np.linalg.norm(fv["moment"])), 4),
                              "tilt_deg": round(float(fv["tilt"]), 1), "misfit_uT": round(fv["rms"], 3)}
                       for mode, fv in fits.items()}, f, indent=2)

    print(f"\nSaved: {stem}_points.csv" + (f", {stem}_fit.json" if fits else ""))

    # ---- pictures
    try:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is missing, so no maps.")
        return
    ext = [-s * 50, (args.nx - 0.5) * s * 100, -s * 50, (args.ny - 0.5) * s * 100]
    grid = lambda v: v.reshape(args.ny, args.nx)
    panels = [(D[:, 2], "A - B, Z (up), drawn at the stick centre", True), (D[:, 0], "A - B, X (along rows)", True),
              (D[:, 1], "A - B, Y (across rows)", True), (dmag, "size of A - B", False),
              (A[:, 2], "sensor A alone, Z (up), drawn where A was", True)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    shiftA = (-1 if args.a_end == "left" else 1) * args.baseline / 2     # cm: where A sits vs the centre
    for ax, (v, title, signed) in zip(axs.ravel(), panels):
        vm = np.nanmax(np.abs(v)) or 1
        dx = shiftA if title.startswith("sensor A") else 0.0
        e = [ext[0] + dx, ext[1] + dx, ext[2], ext[3]]
        im = ax.imshow(grid(v), origin="lower", extent=e, cmap="RdBu_r" if signed else "viridis",
                       vmin=-vm if signed else 0, vmax=vm, interpolation="nearest")
        ax.plot(xs * 100 + dx, ys * 100, ".", color="#898781", ms=2)
        if fits:
            for mode, mk in (("both", "o"), ("diff", "s")):
                p = fits[mode]["pos"]
                ax.plot(p[0] * 100, p[1] * 100, mk, mfc="none", mec="k", mew=2, ms=12)
        ax.set_title(title + "  (uT)", fontsize=10)
        ax.set_xlabel("x cm")
        ax.set_ylabel("y cm")
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axs[1, 2]
    ax.axis("off")
    lines = [f"Grid {args.nx} x {args.ny}, {args.spacing:g} cm", f"{len(good)} of {npts} points usable",
             "calibrated" if cal else "raw values (no calibration)", ref_note, ""]
    if fits:
        for mode, label in (("both", "fit on A and B (circle)"), ("diff", "fit on A - B (square)")):
            f_ = fits[mode]
            lines += [label + ":",
                      f"  x {f_['pos'][0] * 100:.1f} cm, y {f_['pos'][1] * 100:.1f} cm",
                      f"  depth {-f_['pos'][2] * 100:.1f} cm below the chips",
                      f"  strength {np.linalg.norm(f_['moment']):.3f} A m2, tilt {f_['tilt']:.0f} deg",
                      f"  misfit {f_['rms']:.2f} uT", ""]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10, transform=ax.transAxes)
    fig.suptitle(f"{args.csv}: magnetic maps from the grid scan", fontsize=12)
    fig.tight_layout()
    fig.savefig(stem + "_map.png", dpi=110)
    print(f"Saved: {stem}_map.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
