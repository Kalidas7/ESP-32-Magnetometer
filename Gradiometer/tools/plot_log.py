#!/usr/bin/env python3
"""
plot_log.py - look at a raw recording: warm-up, noise, and the turn test.

    python plot_log.py warmup.csv            # noise table for the last 60 s
    python plot_log.py turn.csv --last 20    # noise table for the last 20 s

Plots each axis of A and B over time. Each line has its own average removed, so A and B sit on
top of each other and you can see whether they rise and fall together (turn test).
The bottom panel shows the raw size |A| and |B| (creeping = warm-up drift).

Prints a table of mean, noise and drift for every axis. Saves the plot next to the CSV as .png.
Needs:  pip install numpy matplotlib
"""
import argparse
import sys

import numpy as np

COLS = ["Ax", "Ay", "Az", "Bx", "By", "Bz"]


def load(path):
    d = np.genfromtxt(path, delimiter=",", skip_header=1, comments="#")
    if d.ndim != 2 or d.shape[1] != 8 or len(d) < 30:
        sys.exit(f"{path}: expected at least 30 rows of time_ms,point,Ax,Ay,Az,Bx,By,Bz")
    d = d[10:]                                      # skip the first 0.5 s (sensors still starting)
    for c in (slice(2, 5), slice(5, 8)):            # all-zero = sensor had no value yet
        d[np.all(d[:, c] == 0, axis=1), c] = np.nan
    return (d[:, 0] - d[0, 0]) / 1000.0, d[:, 1], d[:, 2:5], d[:, 5:8]


def main():
    ap = argparse.ArgumentParser(description="Plot a raw recording and print noise and drift.")
    ap.add_argument("csv")
    ap.add_argument("--last", type=float, default=60, help="seconds at the end used for the noise table")
    ap.add_argument("--no-show", action="store_true", help="only save the picture")
    args = ap.parse_args()

    t, pt, A, B = load(args.csv)
    raw = np.column_stack([A, B])
    mags = np.column_stack([np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)])

    # ---- table: mean, noise (end of the file), drift (first minute vs last minute)
    end = t >= t[-1] - args.last
    first = t <= min(60, t[-1] / 2)
    last = t >= t[-1] - min(60, t[-1] / 2)
    names = COLS + ["|A|", "|B|"]
    allv = np.column_stack([raw, mags])
    print(f"\n{args.csv}: {len(t)} rows, {t[-1]:.0f} s, about {len(t) / max(t[-1], 1e-9):.1f} per second")
    print(f"noise = typical jitter over the last {args.last:g} s (keep the stick still then); odd jumps are counted as spikes.")
    print(f"drift = average over the last minute minus average over the first minute.\n")
    print(f"{'axis':<6}|{'mean uT':>10} |{'noise uT':>10} |{'spikes':>8} |{'drift uT':>10}")
    print("-" * 6 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 9 + "+" + "-" * 11)
    for i, n in enumerate(names):
        v = allv[:, i]
        ok = ~np.isnan(v)
        if ok.sum() < 3:
            print(f"{n:<6}|{'--':>10} |{'--':>10} |{'--':>8} |{'--':>10}")
            continue
        e = v[end][~np.isnan(v[end])]
        med = np.median(e)
        noise = 1.4826 * np.median(np.abs(e - med))       # typical jitter, not fooled by spikes
        spikes = int(np.sum(np.abs(e - med) > max(6 * noise, 0.5)))
        mean = np.mean(e[np.abs(e - med) <= max(6 * noise, 0.5)])
        drift = np.nanmedian(v[last]) - np.nanmedian(v[first])
        print(f"{n:<6}|{mean:>10.2f} |{noise:>10.3f} |{spikes:>8d} |{drift:>10.2f}")
    nan_a = np.isnan(A).any(axis=1).sum()
    nan_b = np.isnan(B).any(axis=1).sum()
    if nan_a or nan_b:
        print(f"\nMissing readings: A {nan_a}, B {nan_b} rows.")

    # ---- plot
    try:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is missing. Install it with:  pip install matplotlib")

    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    changes = t[1:][np.diff(pt) != 0]
    for k, axis in enumerate("XYZ"):
        a = A[:, k] - np.nanmean(A[:, k])
        b = B[:, k] - np.nanmean(B[:, k])
        ax[k].plot(t, a, color="#2a78d6", lw=1.2, label=f"A {axis} (average removed)")
        ax[k].plot(t, b, color="#eb6834", lw=1.2, ls="--", label=f"B {axis} (average removed)")
        ax[k].set_ylabel(f"{axis}  uT")
        ax[k].legend(loc="upper right", fontsize=8)
        ax[k].grid(alpha=0.3)
    ax[3].plot(t, mags[:, 0], color="#2a78d6", lw=1.2, label="|A| raw")
    ax[3].plot(t, mags[:, 1], color="#eb6834", lw=1.2, ls="--", label="|B| raw")
    ax[3].set_ylabel("size  uT")
    ax[3].set_xlabel("time  s")
    ax[3].legend(loc="upper right", fontsize=8)
    ax[3].grid(alpha=0.3)
    for c in changes:
        for a_ in ax:
            a_.axvline(c, color="#898781", lw=0.6, ls=":")
    fig.suptitle(f"{args.csv}  -  turn test: A and B should rise and fall together on every axis")
    fig.tight_layout()
    png = args.csv.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=120)
    print(f"\nPicture saved: {png}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
