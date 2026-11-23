#!/usr/bin/env python3
"""
calibrate.py - work out the corrections for both magnetometers from ONE turning recording.

    python calibrate.py calib.csv --field 41.8

--field is your city's Earth-field strength in uT (e.g. Delhi 48.8, Mumbai 43.6, Bengaluru 41.8).

What it does
  1. Each sensor on its own (ellipsoid fit). Turned in every direction, a perfect sensor's readings
     lie on a round ball around zero. Real readings lie on a shifted, squashed, tilted "egg".
     The fit finds each sensor's own offset (the shift) and reshape matrix (the squash and tilt):
         corrected = matrix @ (raw - offset)
  2. Match B to A. After step 1 both are nearly right; this finds the last small twist and gain
     difference between the two boards, using only the slow parts of the recording:
         B_final = match_matrix @ B_corrected + match_offset
  3. Prints a before/after table and a pass/fail table, and saves:
         calibration.json          the numbers, to apply to later recordings
         calibration_report.png    pictures of the raw eggs and the corrected balls

Needs:  pip install numpy matplotlib
"""
import argparse
import datetime
import json
import sys

import numpy as np

# ------------------------------------------------------------------ pass / fail limits (uT, degrees)
SPREAD_GOOD, SPREAD_OK = 1.0, 2.0      # how much |corrected| still varies while turning (std)
MATCH_GOOD, MATCH_OK = 1.0, 2.0        # A - B left after matching (rms)
COVER_GOOD, COVER_OK = 0.70, 0.50      # share of directions the stick was pointed in
ANGLE_WARN = 10.0                      # boards more than this far apart -> axis settings probably wrong
SLOW_UT_PER_S = 15.0                   # matching uses only parts where the field changes slower than this


class FitError(Exception):
    pass


# ------------------------------------------------------------------ data
def load(path):
    d = np.genfromtxt(path, delimiter=",", skip_header=1, comments="#")
    if d.ndim != 2 or d.shape[1] != 8 or len(d) < 30:
        sys.exit(f"{path}: expected columns time_ms,point,Ax,Ay,Az,Bx,By,Bz")
    d = d[10:]                                      # skip the first 0.5 s (sensors still starting)
    for c in (slice(2, 5), slice(5, 8)):            # all-zero = sensor had no value yet
        d[np.all(d[:, c] == 0, axis=1), c] = np.nan
    return d[:, 0], d[:, 2:5], d[:, 5:8]


# ------------------------------------------------------------------ step 1: ellipsoid fit
def _quadric_fit(P):
    """Fit the general quadric through the points; return centre c and shape A with
    (p - c)^T A (p - c) = 1 on the egg's surface."""
    m = P.mean(axis=0)
    s = np.abs(P - m).max() or 1.0                 # work in scaled units for accuracy
    q = (P - m) / s
    x, y, z = q.T
    D = np.column_stack([x * x, y * y, z * z, 2 * y * z, 2 * x * z, 2 * x * y,
                         2 * x, 2 * y, 2 * z, np.ones_like(x)])
    v = np.linalg.svd(D, full_matrices=False)[2][-1]
    M = np.array([[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]])
    u, d = v[6:9], v[9]
    ev = np.linalg.eigvalsh(M)
    if np.all(ev < 0):
        M, u, d = -M, -u, -d
    elif not np.all(ev > 0):
        raise FitError("the readings do not form an egg shape - turn the stick in more directions")
    cq = -np.linalg.solve(M, u)
    k = cq @ M @ cq - d
    if k <= 0:
        raise FitError("fit failed - turn the stick in more directions")
    return m + s * cq, (M / k) / s ** 2


def fit_sensor(P_all, field):
    """Return dict with offset, matrix (corrected = matrix @ (raw - offset)), and fit facts.
    'good' marks the rows that are neither missing nor spikes."""
    valid = ~np.isnan(P_all).any(axis=1)
    P = P_all[valid]
    if len(P) < 200:
        raise FitError(f"only {len(P)} readings - record for longer")
    c, A = _quadric_fit(P)
    w, V = np.linalg.eigh(A)
    W = field * (V * np.sqrt(w)) @ V.T
    # drop spikes (readings far off the egg) and fit once more
    r = np.linalg.norm((P - c) @ W.T, axis=1) - field
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r - np.median(r)) < max(5 * mad, 2.0)
    if keep.sum() < len(P) and keep.sum() >= 200:
        c, A = _quadric_fit(P[keep])
        w, V = np.linalg.eigh(A)
        W = field * (V * np.sqrt(w)) @ V.T
    r = np.linalg.norm((P - c) @ W.T, axis=1) - field
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r - np.median(r)) < max(5 * mad, 2.0)
    good = np.zeros(len(P_all), bool)
    good[np.flatnonzero(valid)[keep]] = True
    semi = 1 / np.sqrt(w)                          # the egg's three half-axes, raw uT
    return {"offset": c, "matrix": W, "semi_axes": np.sort(semi)[::-1], "good": good,
            "egg_ratio": semi.max() / semi.min(), "dropped": int(len(P) - keep.sum())}


def correct(P, cal):
    return (P - cal["offset"]) @ cal["matrix"].T


def coverage(U):
    """Share of 32 evenly spread directions the stick pointed in (at least 5 readings each)."""
    U = U[~np.isnan(U).any(axis=1)]
    n = U / np.linalg.norm(U, axis=1, keepdims=True)
    k = 32
    i = np.arange(k) + 0.5
    zz = 1 - 2 * i / k
    rr = np.sqrt(1 - zz * zz)
    ph = i * np.pi * (3 - np.sqrt(5))
    dirs = np.column_stack([rr * np.cos(ph), rr * np.sin(ph), zz])
    counts = np.bincount(np.argmax(n @ dirs.T, axis=1), minlength=k)
    return float((counts >= 5).mean())


# ------------------------------------------------------------------ step 2: match B to A
def fit_match(t, a, b, good):
    ok = good & ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    t, a, b = t[ok], a[ok], b[ok]
    rate = np.full(len(a), np.inf)                 # how fast the field changes, uT per second
    dt = (t[2:] - t[:-2]) / 1000.0
    rate[1:-1] = np.linalg.norm(a[2:] - a[:-2], axis=1) / np.where(dt > 0, dt, np.inf)
    slow = rate < SLOW_UT_PER_S
    if slow.sum() < 300:                           # turned fast everywhere: use the slower half
        slow = rate <= np.percentile(rate[1:-1], 50)
    for _ in range(2):                             # fit, drop odd readings, fit again
        X = np.column_stack([b[slow], np.ones(slow.sum())])
        beta = np.linalg.lstsq(X, a[slow], rcond=None)[0]
        M, o = beta[:3].T, beta[3]
        res = a - (b @ M.T + o)
        rn = np.linalg.norm(res, axis=1)
        med = np.median(rn[slow])
        mad = 1.4826 * np.median(np.abs(rn[slow] - med))
        slow = slow & (rn < med + max(5 * mad, 1.0))
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return {"matrix": M, "offset": o, "angle": angle, "gains": S,
            "rms": float(np.sqrt(np.mean(np.sum(res[slow] ** 2, axis=1)))),
            "used": int(slow.sum()), "total": int(len(a)), "t": t, "res": res, "slow": slow}


# ------------------------------------------------------------------ printing helpers
def rms_norm(v):
    v = v[~np.isnan(v).any(axis=1)]
    return float(np.sqrt(np.mean(np.sum(v ** 2, axis=1))))


def size_stats(P, good=None):
    if good is not None:
        P = P[good]
    s = np.linalg.norm(P[~np.isnan(P).any(axis=1)], axis=1)
    return s.min(), s.max(), s.mean(), s.std()


def grade(value, good, ok, higher_is_better=False):
    if higher_is_better:
        return "PASS" if value >= good else "MARGINAL" if value >= ok else "FAIL"
    return "PASS" if value <= good else "MARGINAL" if value <= ok else "FAIL"


def line(widths):
    print("+".join("-" * w for w in widths))


def main():
    ap = argparse.ArgumentParser(description="Calibrate both magnetometers from one turning recording.")
    ap.add_argument("csv", help="recording made while turning the stick in every direction")
    ap.add_argument("--field", type=float, default=None, help="your city's Earth-field strength in uT")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--no-show", action="store_true", help="only save the picture")
    args = ap.parse_args()

    t, A_raw, B_raw = load(args.csv)
    print(f"\n{args.csv}: {len(t)} readings over {(t[-1] - t[0]) / 1000:.0f} s")

    field = args.field
    if field is None:
        field = 45.0
        print("No --field given: using 45 uT. Pass your city's value for true uT numbers.")

    # ---------------- step 1
    cals = {}
    for name, P in (("A", A_raw), ("B", B_raw)):
        try:
            cals[name] = fit_sensor(P, field)
        except FitError as e:
            sys.exit(f"\nSensor {name}: {e}.")
    A_cal, B_cal = correct(A_raw, cals["A"]), correct(B_raw, cals["B"])

    # ---------------- step 2
    good = cals["A"]["good"] & cals["B"]["good"]
    mt = fit_match(t, A_cal, B_cal, good)
    B_fin = B_cal @ mt["matrix"].T + mt["offset"]

    # ---------------- results
    cov = min(coverage(A_cal), coverage(B_cal))
    rawA, rawB = size_stats(A_raw, cals["A"]["good"]), size_stats(B_raw, cals["B"]["good"])
    calA, calB = size_stats(A_cal, cals["A"]["good"]), size_stats(B_cal, cals["B"]["good"])
    finB = size_stats(B_fin, cals["B"]["good"])

    print("\nWHAT EACH SENSOR'S OWN FIX FOUND")
    w = [8, 26, 22, 12, 9]
    print(f"{'sensor':<8}|{'offset X, Y, Z (uT)':>25} |{'egg half-axes (uT)':>21} |{'egg ratio':>11} |{'spikes':>8}")
    line(w)
    for n in ("A", "B"):
        c = cals[n]
        off = ", ".join(f"{v:6.1f}" for v in c["offset"])
        sa = ", ".join(f"{v:5.1f}" for v in c["semi_axes"])
        print(f"{n:<8}|{off:>25} |{sa:>21} |{c['egg_ratio']:>11.3f} |{c['dropped']:>8d}")
    print("egg ratio 1.000 = perfectly round; 1.05 = 5% squashed. Spikes = odd readings left out.")

    print("\nBEFORE AND AFTER (uT)")
    w = [30, 15, 20, 17]
    print(f"{'':<30}|{'raw':>14} |{'own fix':>19} |{'+ matching':>16}")
    line(w)
    print(f"{'|A| range while turning':<30}|{rawA[0]:>6.1f} to {rawA[1]:<5.1f}|{calA[0]:>10.1f} to {calA[1]:<6.1f}|{'(A unchanged)':>16}")
    print(f"{'|B| range while turning':<30}|{rawB[0]:>6.1f} to {rawB[1]:<5.1f}|{calB[0]:>10.1f} to {calB[1]:<6.1f}|{finB[0]:>7.1f} to {finB[1]:<5.1f}")
    print(f"{'|A| spread (std)':<30}|{rawA[3]:>14.2f} |{calA[3]:>19.2f} |{'':>16}")
    print(f"{'|B| spread (std)':<30}|{rawB[3]:>14.2f} |{calB[3]:>19.2f} |{finB[3]:>16.2f}")
    print(f"{'A - B (rms)':<30}|{rms_norm((A_raw - B_raw)[good]):>14.2f} |{rms_norm((A_cal - B_cal)[good]):>19.2f} |{mt['rms']:>16.2f}")
    print(f"target size = {field:g} uT. Matching used {mt['used']} of {mt['total']} readings (the slow, normal ones).")

    print("\nMATCH B TO A")
    names = "XYZ"
    for i in range(3):
        row = "  ".join(f"{mt['matrix'][i, j]:+7.4f}*B{names[j]}" for j in range(3))
        print(f"  A{names[i]} = {row}  {mt['offset'][i]:+6.2f}")
    print(f"  boards {mt['angle']:.1f} degrees from square, gains {', '.join(f'{g:.3f}' for g in mt['gains'])}")

    checks = [
        ("Directions covered", f"{cov * 100:.0f}%", grade(cov, COVER_GOOD, COVER_OK, True), "70% or more"),
        ("|A| spread after own fix", f"{calA[3]:.2f} uT", grade(calA[3], SPREAD_GOOD, SPREAD_OK), "1 uT or less"),
        ("|B| spread after own fix", f"{calB[3]:.2f} uT", grade(calB[3], SPREAD_GOOD, SPREAD_OK), "1 uT or less"),
        ("A - B after matching", f"{mt['rms']:.2f} uT", grade(mt["rms"], MATCH_GOOD, MATCH_OK), "1 uT or less"),
        ("Boards from square", f"{mt['angle']:.1f} deg", "PASS" if mt["angle"] < ANGLE_WARN else "CHECK", "under 10 deg"),
    ]
    print("\nCHECKS")
    w = [27, 12, 10, 14]
    print(f"{'check':<27}|{'result':>11} |{'grade':>9} |{'want':>13}")
    line(w)
    for c in checks:
        print(f"{c[0]:<27}|{c[1]:>11} |{c[2]:>9} |{c[3]:>13}")

    grades = [c[2] for c in checks]
    print()
    if all(g == "PASS" for g in grades):
        print("RESULT: PASS - the corrections work. Move on to the slide test and the 2D map.")
    else:
        print("RESULT: NOT YET - what to try:")
        if checks[0][2] != "PASS":
            print(" - Turn the stick in more directions: point each end up, down and to every side, and roll it.")
        if checks[1][2] != "PASS" or checks[2][2] != "PASS":
            print(" - Move away from steel (tables, chairs, laptops, phones). Keep the USB cable taped still.")
            print(" - Check for loose wires or anything on the board that moves while turning.")
        if checks[3][2] != "PASS" and checks[1][2] == "PASS" and checks[2][2] == "PASS":
            print(" - Each sensor is fine alone, so turn more SLOWLY (the two sensors are read a few ms apart),")
            print("   and check that nothing on the board flexes or moves.")
        if checks[4][2] != "PASS":
            print(" - The boards look far from square: run the turn test and fix AXES_B in the sketch.")

    # ---------------- save
    out = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": args.csv, "field_uT": field,
        "A": {"offset": cals["A"]["offset"].tolist(), "matrix": cals["A"]["matrix"].tolist()},
        "B": {"offset": cals["B"]["offset"].tolist(), "matrix": cals["B"]["matrix"].tolist()},
        "B_to_A": {"matrix": mt["matrix"].tolist(), "offset": mt["offset"].tolist()},
        "how_to_apply": "A = A.matrix @ (A_raw - A.offset);  "
                        "B = B_to_A.matrix @ (B.matrix @ (B_raw - B.offset)) + B_to_A.offset",
        "results": {c[0]: {"value": c[1], "grade": c[2]} for c in checks},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.out}")

    # ---------------- pictures
    try:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is missing, so no pictures. Install it with:  pip install matplotlib")
        return

    fig = plt.figure(figsize=(14, 9))
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    sx, sy, sz = field * np.cos(u) * np.sin(v), field * np.sin(u) * np.sin(v), field * np.cos(v)
    step = max(1, len(t) // 1500)
    panels = [(1, A_raw, "A raw (shifted, squashed egg)"), (2, A_cal, "A after its own fix"),
              (4, B_raw, "B raw (shifted, squashed egg)"), (5, B_fin, "B after its own fix + matching")]
    for pos, P, title in panels:
        ax = fig.add_subplot(2, 3, pos, projection="3d")
        Q = P[::step]
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=3, c="#2a78d6" if pos < 4 else "#eb6834")
        ax.plot_wireframe(sx, sy, sz, color="#898781", lw=0.3, alpha=0.5)
        lim = max(np.nanmax(np.abs(P)), field) * 1.1
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X uT"); ax.set_ylabel("Y uT"); ax.set_zlabel("Z uT")
    ts = (t - t[0]) / 1000
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(ts, np.linalg.norm(A_cal, axis=1), color="#2a78d6", lw=1, label="|A| corrected")
    ax.plot(ts, np.linalg.norm(B_fin, axis=1), color="#eb6834", lw=1, ls="--", label="|B| corrected")
    ax.axhline(field, color="#898781", lw=0.8, ls=":")
    ax.set_ylim(field - 5, field + 5)
    ax.set_title("Size while turning: should stay flat at your field", fontsize=10)
    ax.set_xlabel("time s"); ax.set_ylabel("uT"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 3, 6)
    resn = np.linalg.norm(mt["res"], axis=1)
    ax.plot(mt["t"] / 1000 - t[0] / 1000, resn, color="#898781", lw=0.8, label="all readings")
    ax.plot(mt["t"][mt["slow"]] / 1000 - t[0] / 1000, resn[mt["slow"]], ".", ms=2, color="#1baf7a",
            label="slow parts (used)")
    ax.axhline(1.0, color="#e34948", lw=0.8, ls=":", label="1 uT goal")
    ax.set_ylim(0, max(3, np.percentile(resn, 99) * 1.1))
    ax.set_title("A - B left after matching", fontsize=10)
    ax.set_xlabel("time s"); ax.set_ylabel("uT"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("calibration_report.png", dpi=110)
    print("Saved: calibration_report.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
