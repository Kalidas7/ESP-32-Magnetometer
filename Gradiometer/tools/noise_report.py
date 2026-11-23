#!/usr/bin/env python3
"""
noise_report.py - noise and standard deviation of a STILL recording (nothing moving).

    python3 noise_report.py warmup.csv
    python3 noise_report.py quiet.csv --skip 30      # ignore the first 30 s

Prints:
  1. mean, standard deviation, min, max and peak-to-peak for every axis of A, B and A - B
  2. standard deviation after averaging (how much holding still at each point helps)
  3. drift (slow creep) per 10 minutes
  4. a check for a slow repeating pulse (like B's 5-second wave)
Saves:
  <name>_noise.csv   the numbers, for a spreadsheet or your report
  <name>_noise.png   histograms and the averaging curve

Noise = standard deviation of the readings while nothing moves.
Needs:  numpy, matplotlib
"""
import argparse
import csv
import sys

import numpy as np

AVG_SECONDS = [0.05, 0.5, 1, 2, 5, 10]


def load(path, skip):
    d = np.genfromtxt(path, delimiter=",", skip_header=1, comments="#")
    if d.ndim != 2 or d.shape[1] != 8 or len(d) < 100:
        sys.exit(f"{path}: expected at least 100 rows of time_ms,point,Ax,Ay,Az,Bx,By,Bz")
    d = d[10:]                                          # first 0.5 s: sensors still starting
    for c in (slice(2, 5), slice(5, 8)):                # all-zero = no value yet
        d[np.all(d[:, c] == 0, axis=1), c] = np.nan
    t = (d[:, 0] - d[0, 0]) / 1000.0
    keep = t >= skip
    d, t = d[keep], t[keep] - t[keep][0]
    ok = ~np.isnan(d[:, 2:8]).any(axis=1)
    return t[ok], d[ok, 2:5], d[ok, 5:8]


def block_std(x, n):
    m = len(x) // n
    if m < 5:
        return np.nan
    return x[:m * n].reshape(m, n).mean(axis=1).std(ddof=1)


def slow_pulse(x, fs):
    """Strongest rhythm between 0.1 and 1 Hz: (period in s, its size as rms uT)."""
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    band = (f >= 0.1) & (f <= 1.0)
    if band.sum() < 3:
        return np.nan, np.nan
    f0 = f[band][np.argmax(np.abs(X[band]))]
    Y = X.copy()
    Y[~((f >= 0.1) & (f <= 0.5))] = 0
    return 1 / f0, np.fft.irfft(Y, len(x)).std()


def main():
    ap = argparse.ArgumentParser(description="Noise and standard deviation of a still recording.")
    ap.add_argument("csv")
    ap.add_argument("--skip", type=float, default=0, help="seconds to ignore at the start")
    ap.add_argument("--no-show", action="store_true", help="only save the picture")
    args = ap.parse_args()

    t, A, B = load(args.csv, args.skip)
    fs = len(t) / t[-1]
    D = A - B
    chans = {"A x": A[:, 0], "A y": A[:, 1], "A z": A[:, 2], "|A|": np.linalg.norm(A, axis=1),
             "B x": B[:, 0], "B y": B[:, 1], "B z": B[:, 2], "|B|": np.linalg.norm(B, axis=1),
             "A-B x": D[:, 0], "A-B y": D[:, 1], "A-B z": D[:, 2]}
    new_b = np.r_[True, np.any(np.diff(B, axis=0) != 0, axis=1)].sum() / t[-1]
    steps = np.diff(np.unique(B[:, 0]))
    bstep = np.median(steps[steps > 0]) if (steps > 0).any() else np.nan

    print(f"\n{args.csv}: {len(t)} readings over {t[-1] / 60:.1f} min ({fs:.1f} per second)")
    print(f"A = MPU9250 (each reading = average over 50 ms). B = BNO055: {new_b:.1f} new values per second,"
          f" steps of {bstep:.3f} uT.")
    print("Noise = standard deviation (SD) of the readings while nothing moves. Units: uT.\n")

    rows = []
    print("1. EACH AXIS (single readings)")
    print(f"{'axis':<7}|{'mean':>9} |{'SD (noise)':>11} |{'min':>9} |{'max':>9} |{'peak-peak':>10}")
    print("-" * 7 + "+" + "-" * 10 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10)
    for n, x in chans.items():
        sd = x.std(ddof=1)
        print(f"{n:<7}|{x.mean():>9.2f} |{sd:>11.3f} |{x.min():>9.2f} |{x.max():>9.2f} |{np.ptp(x):>10.2f}")
        rows.append({"axis": n, "mean_uT": round(x.mean(), 3), "SD_uT": round(sd, 4),
                     "min_uT": round(x.min(), 3), "max_uT": round(x.max(), 3), "peak_peak_uT": round(np.ptp(x), 3)})

    print("\n2. SD AFTER AVERAGING (hold still this long at each point)")
    hdr = "".join(f"{(str(s) + ' s'):>9}" for s in AVG_SECONDS)
    print(f"{'axis':<7}|{hdr}")
    print("-" * 7 + "+" + "-" * (9 * len(AVG_SECONDS)))
    for r, (n, x) in zip(rows, chans.items()):
        if n.startswith("|"):
            continue
        vals = [block_std(x, max(1, int(round(s * fs)))) for s in AVG_SECONDS]
        for s, v in zip(AVG_SECONDS, vals):
            r[f"SD_avg_{s}s_uT"] = round(float(v), 4)
        print(f"{n:<7}|" + "".join(f"{v:>9.3f}" for v in vals))

    print("\n3. DRIFT (slow creep, straight-line trend)")
    for r, (n, x) in zip(rows, chans.items()):
        slope = np.polyfit(t, x, 1)[0] * 600
        r["drift_uT_per_10min"] = round(slope, 3)
    print("  " + "   ".join(f"{n}: {r['drift_uT_per_10min']:+.2f}" for r, n in zip(rows, chans) if not n.startswith("|")))
    print("  (uT per 10 minutes)")

    print("\n4. SLOW REPEATING PULSE CHECK (0.1 to 1 Hz)")
    print(f"{'axis':<7}|{'period s':>10} |{'size uT':>9} |")
    print("-" * 7 + "+" + "-" * 11 + "+" + "-" * 10 + "+")
    pulse = {}
    for n in ("A x", "A y", "A z", "B x", "B y", "B z"):
        p, s = slow_pulse(chans[n], fs)
        pulse[n] = s
        print(f"{n:<7}|{p:>10.1f} |{s:>9.3f} |")
    worst = max(("B x", "B y", "B z"), key=lambda k: pulse[k])
    a_ref = np.mean([pulse["A x"], pulse["A y"], pulse["A z"]])
    if pulse[worst] > max(0.15, 2.5 * a_ref):
        print(f"  -> B has a slow repeating pulse (biggest on {worst}, {pulse[worst]:.2f} uT). A does not.")
        print("     Likely mains interference (chargers, PC power supply, power strips). Try moving away.")
    else:
        print("  -> no slow pulse worth worrying about.")

    out = args.csv.rsplit(".", 1)[0] + "_noise.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + [k for k in rows[1] if k not in rows[0]])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out}")

    try:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is missing, so no picture.")
        return
    fig = plt.figure(figsize=(12, 8))
    for k, ax_name in enumerate("xyz"):
        ax = fig.add_subplot(2, 3, k + 1)
        for x, col, lab in ((chans[f"A {ax_name}"], "#2a78d6", "A"), (chans[f"B {ax_name}"], "#eb6834", "B")):
            dev = x - x.mean()
            ax.hist(dev, bins=np.arange(-3.0, 3.01, 0.375), histtype="step", lw=1.6, color=col,
                    label=f"{lab} {ax_name}: SD {x.std(ddof=1):.2f} uT")
        ax.set_title(f"{ax_name.upper()} axis: spread of readings around the mean", fontsize=10)
        ax.set_xlabel("reading minus mean, uT (bins 0.375 uT = B's step size)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 1, 2)
    secs = np.array([0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15])
    for n, col in (("A-B x", "#2a78d6"), ("A-B y", "#1baf7a"), ("A-B z", "#eb6834")):
        ax.loglog(secs, [block_std(chans[n], max(1, int(round(s * fs)))) for s in secs], "o-", color=col, label=n)
    sd0 = chans["A-B x"].std(ddof=1)
    ax.loglog(secs, sd0 / np.sqrt(secs / secs[0]), "k:", lw=1, label="purely random noise would follow this")
    ax.set_title("SD of A - B after averaging: how long to hold still at each map point", fontsize=10)
    ax.set_xlabel("averaging time, s")
    ax.set_ylabel("SD, uT")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = args.csv.rsplit(".", 1)[0] + "_noise.png"
    fig.savefig(png, dpi=110)
    print(f"Saved: {png}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
