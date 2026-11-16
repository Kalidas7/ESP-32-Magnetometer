#!/usr/bin/env python3
"""
mag_logger.py - receive data from the 2-sensor magnetometer GRADIOMETER (ESP32 over WiFi)
and save it to CSV files on this computer.

Needs only plain Python 3 (no extra packages).

THE IDEA
  Two magnetometers are fixed one above the other on the probe:
     sensor A = BNO055                         -> BOTTOM sensor
     sensor B = second BNO055 or MPU-9250      -> TOP sensor      (--swap if reversed)
  gradient  dB = B_bottom - B_top
  Earth's field is the same at both sensors, so it cancels. A nearby magnet / piece of iron
  is much stronger at the bottom sensor, so it remains.

  Two different chips (BNO055 + MPU-9250) have different axes, sensitivity and offsets.
  --calibrate learns how to line sensor B up with sensor A (you turn the probe in all
  directions for 30 s, away from magnets). It is saved and used automatically after that.
  Then you ZERO the probe once per session, at a spot with no magnet nearby.

USAGE (in this order the first time)
  python3 mag_logger.py --watch
        Live values: check that both sensors work.
  python3 mag_logger.py --calibrate
        Line sensor B up with sensor A (needed for BNO055 + MPU-9250, good for two BNO055s).
  python3 mag_logger.py --watch --baseline 8
        Press Enter (or the IO0/BOOT button) to zero, then turn the probe around:
        |dB| should stay within a few uT.
  python3 mag_logger.py --grid 0 20 0 20 2 --baseline 8
        Guided mapping: x from 0 to 20, y from 0 to 20, every 2 (e.g. cm).
        First it asks you to zero, then tells you where to put the probe;
        press Enter or the IO0 button to measure each spot.
  python3 mag_logger.py --baseline 8
        Free mapping: type the position "x y" and press Enter at each spot.

  --baseline  distance between the two sensor chips in cm (optional, adds uT/cm values)
  --swap      use this if sensor B is the LOWER one
  --no-cal    ignore the saved calibration
  --single    map with ONE sensor only (sensor A, or B with --swap), e.g. when the MPU board
              has no magnetometer. No zeroing/calibration needed; make_map.py handles the file.

Keys while mapping:  Enter / IO0 button = measure   z = zero again
                     s = skip   r = redo previous   q = quit

Files go in ./logs/ (next to this script):
  raw_<date_time>.csv  every single sample, raw values of both sensors
  map_<date_time>.csv  one averaged row per position, plus "null" rows from zeroing
                       (top sensor already lined up with the bottom one) -> make_map.py
  cross_cal.json       the --calibrate result
"""
import argparse
import csv
import json
import math
import os
import queue
import shutil
import socket
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime

IN_FIELDS = ["seq", "esp_ms", "mode", "ok",
             "xa", "ya", "za", "xb", "yb", "zb",
             "cala", "calb", "heading", "roll", "pitch"]
INT_FIELDS = ("seq", "esp_ms", "mode", "ok", "cala", "calb")
MODE_NAMES = {2: "MAGONLY (fixed offsets - good for a gradiometer)",
              12: "NDOF (auto-calibrating - the difference may jump)",
              11: "NDOF_FMC_OFF", 9: "COMPASS", 10: "M4G", 7: "AMG"}

MAP_FIELDS = ["kind", "x", "y",
              "bx_bot", "by_bot", "bz_bot", "b_bot",
              "bx_top", "by_top", "bz_top", "b_top",
              "dbx", "dby", "dbz", "db", "dbx_std", "dby_std", "dbz_std",
              "n", "cal_bot", "cal_top", "baseline_cm", "bottom_addr", "pc_time"]


# ----------------------------------------------------------------- vectors
def vsub(a, b):
    return tuple(p - q for p, q in zip(a, b))


def vnorm(a):
    return math.sqrt(sum(c * c for c in a))


def vmean(vs):
    return tuple(statistics.fmean(v[i] for v in vs) for i in range(3))


def vstd(vs):
    return tuple(statistics.pstdev([v[i] for v in vs]) if len(vs) > 1 else 0.0 for i in range(3))


def fvec(v, nd=2):
    return "(" + ", ".join(f"{c:+.{nd}f}" for c in v) + ")"


# ------------------------------------------------------- cross-calibration
def solve(A, b):
    """Solve A x = b (small square system) by Gaussian elimination."""
    n = len(b)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-9:
            raise ValueError("not enough variety in the data (probe not turned enough)")
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col:
                f = M[r][col] / M[col][col]
                for k in range(col, n + 1):
                    M[r][k] -= f * M[col][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_cross_cal(A, B):
    """Least squares fit  A = M.B + c  (M 3x3, c 3). Lines sensor B up with sensor A:
    M takes care of different axes, tilt between the boards and sensitivity, c of offsets."""
    X = [list(b) + [1.0] for b in B]
    XtX = [[sum(r[i] * r[j] for r in X) for j in range(4)] for i in range(4)]
    M, c = [], []
    for k in range(3):
        Xty = [sum(r[i] * a[k] for r, a in zip(X, A)) for i in range(4)]
        sol = solve(XtX, Xty)
        M.append(sol[:3])
        c.append(sol[3])
    cal = {"M": M, "c": c}
    res = [vnorm(vsub(a, apply_cal(cal, b))) for a, b in zip(A, B)]
    cal["rms"] = math.sqrt(statistics.fmean(r * r for r in res))
    return cal


def apply_cal(cal, b):
    """Sensor B reading -> the same axes/scale as sensor A."""
    if not cal:
        return tuple(b)
    M, c = cal["M"], cal["c"]
    return tuple(M[k][0] * b[0] + M[k][1] * b[1] + M[k][2] * b[2] + c[k] for k in range(3))


def still_samples(samples, max_rate):
    """Keep samples taken while the probe turned slowly (field changing < max_rate uT/s).
    The two chips don't measure at exactly the same moment, so fast turning makes them
    disagree; slow samples give a much better calibration."""
    keep = []
    for p, s, n in zip(samples, samples[1:], samples[2:]):
        dt = (n["esp_ms"] - p["esp_ms"]) / 1000.0
        if not 0 < dt < 0.6:
            continue
        rate = max(vnorm(vsub(n["a"], p["a"])), vnorm(vsub(n["b"], p["b"]))) / dt
        if rate <= max_rate:
            keep.append(s)
    return keep


def coverage(A):
    """Smallest per-axis range of sensor A, as a fraction of the field strength (1.4 = good)."""
    rng = [max(v[i] for v in A) - min(v[i] for v in A) for i in range(3)]
    return min(rng) / max(statistics.fmean(vnorm(v) for v in A), 1e-6), rng


def robust_fit(A, B):
    """Fit, drop samples that disagree far more than the rest (glitches), fit again."""
    cal = fit_cross_cal(A, B)
    for _ in range(2):
        res = [vnorm(vsub(a, apply_cal(cal, b))) for a, b in zip(A, B)]
        lim = max(3 * statistics.median(res), 1.0)
        keep = [i for i, r in enumerate(res) if r <= lim]
        if len(keep) == len(A) or len(keep) < 50:
            break
        A, B = [A[i] for i in keep], [B[i] for i in keep]
        cal = fit_cross_cal(A, B)
    cal["n_used"] = len(A)
    return cal


def load_cal(path):
    try:
        with open(path) as f:
            cal = json.load(f)
        if len(cal["M"]) == 3 and len(cal["c"]) == 3:
            return cal
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


# ------------------------------------------------------------------ packets
def parse_packet(data, swap=False, cal=None):
    """'GRAD,seq,ms,mode,ok,xa,ya,za,xb,yb,zb,ca,cb,h,r,p' -> dict, or None."""
    try:
        parts = data.decode("ascii").strip().split(",")
    except UnicodeDecodeError:
        return None
    if len(parts) != len(IN_FIELDS) + 1 or parts[0] != "GRAD":
        return None
    try:
        vals = [float(p) for p in parts[1:]]
    except ValueError:
        return None
    s = dict(zip(IN_FIELDS, vals))
    for k in INT_FIELDS:
        s[k] = int(s[k])
    a, b = (s["xa"], s["ya"], s["za"]), (s["xb"], s["yb"], s["zb"])
    s["a"], s["b"] = a, b                              # raw values
    b_al = apply_cal(cal, b)                           # B lined up with A
    ok_a, ok_b = bool(s["ok"] & 1), bool(s["ok"] & 2)
    if swap:
        s["bot"], s["top"], s["cal_bot"], s["cal_top"] = b_al, a, s["calb"], s["cala"]
    else:
        s["bot"], s["top"], s["cal_bot"], s["cal_top"] = a, b_al, s["cala"], s["calb"]
    s["both"] = ok_a and ok_b
    s["ok_bot"] = ok_b if swap else ok_a
    s["b_bot"], s["b_top"] = vnorm(s["bot"]), vnorm(s["top"])
    s["d"] = vsub(s["bot"], s["top"])                  # gradient: bottom - top
    s["fusion"] = s["mode"] >= 8
    return s


class Receiver(threading.Thread):
    """Background thread: receives UDP packets, writes the raw CSV, keeps a buffer."""

    def __init__(self, port, raw_path, swap, cal):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", port))
        self.sock.settimeout(0.5)
        self.port = port
        self._last_hello = 0.0
        self.swap = swap
        self.cal = cal
        self.cond = threading.Condition()
        self.buf = deque(maxlen=5000)
        self.count = 0
        self.lost = 0
        self.last_seq = None
        self.esp_ip = None
        self.button_presses = 0
        self.null = None                  # zero reading of dB (bottom - top), set by zeroing
        self.info = {"A": "?", "B": "?"}  # sensor names sent by the ESP32
        self._last_btn = None
        self._last_note, self._last_note_t = None, 0.0
        self.running = True
        self.raw_f = open(raw_path, "w", newline="")
        self.raw_w = csv.writer(self.raw_f)
        self.raw_w.writerow(["pc_time", "iso_time", "esp_ip"] + IN_FIELDS +
                            ["bA", "bB", "dBx_bot_minus_top", "dBy_bot_minus_top",
                             "dBz_bot_minus_top", "dB", "cross_cal_used"])

    def note(self, text):
        """Show a message from/about the ESP32 (not more than every 15 s if repeated)."""
        now = time.time()
        if text != self._last_note or now - self._last_note_t > 15:
            self._last_note, self._last_note_t = text, now
            print(f"\n[ESP32] {text}", flush=True)

    def run(self):
        last_flush = time.time()
        while self.running:
            try:
                data, addr = self.sock.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                break
            if data.startswith(b"BTN,"):          # IO0 button on the ESP32 was pressed
                btn_id = data[4:].strip()
                if btn_id != self._last_btn:     # the ESP sends 3 copies; count once
                    self._last_btn = btn_id
                    with self.cond:
                        self.button_presses += 1
                        self.cond.notify_all()
                continue
            if data.startswith(b"INFO,"):
                info = dict(p.split("=", 1) for p in data[5:].decode("ascii", "replace").strip().split(",")
                            if "=" in p)
                if info.get("A") != self.info.get("A") or info.get("B") != self.info.get("B"):
                    self.info = {"A": info.get("A", "?"), "B": info.get("B", "?")}
                    print(f"\n[ESP32] sensor A = {self.info['A']},  sensor B = {self.info['B']}", flush=True)
                continue
            if data.startswith(b"ERR,"):
                self.note(data[4:].decode("ascii", "replace").strip())
                continue
            if data.startswith(b"MAG,"):
                self.note("Getting data from the OLD single-sensor sketch. "
                          "Upload the new 2-sensor MagSensor_ESP32.ino to the ESP32.")
                continue
            s = parse_packet(data, self.swap, self.cal)
            if s is None:
                continue
            t = time.time()
            s["pc_time"] = t
            if self.last_seq is not None and s["seq"] > self.last_seq + 1:
                self.lost += s["seq"] - self.last_seq - 1
            self.last_seq = s["seq"]
            self.raw_w.writerow([f"{t:.3f}", datetime.fromtimestamp(t).isoformat(timespec="milliseconds"),
                                 addr[0]] + [s[k] for k in IN_FIELDS] +
                                [f"{vnorm(s['a']):.2f}", f"{vnorm(s['b']):.2f}"] +
                                [f"{c:.2f}" for c in s["d"]] + [f"{vnorm(s['d']):.2f}",
                                                                1 if self.cal else 0])
            if t - self._last_hello > 2.0:            # tell the ESP32 where we are, so it
                self._last_hello = t                 # sends straight here (fewer lost packets)
                try:
                    self.sock.sendto(b"HELLO", (addr[0], self.port))
                except OSError:
                    pass
            with self.cond:
                self.esp_ip = addr[0]
                self.buf.append(s)
                self.count += 1
                self.cond.notify_all()
            if t - last_flush > 1.0:
                self.raw_f.flush()
                last_flush = t

    def latest(self):
        with self.cond:
            return self.buf[-1] if self.buf else None

    def since(self, t0):
        """All samples received after time t0."""
        with self.cond:
            return [s for s in self.buf if s["pc_time"] >= t0]

    def collect(self, n, timeout):
        """Wait for the next n NEW samples (or until timeout) and return them."""
        with self.cond:
            start = self.count
            deadline = time.time() + timeout
            while self.count - start < n:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self.cond.wait(remaining)
            got = min(self.count - start, len(self.buf))
            return list(self.buf)[-got:] if got > 0 else []

    def missing_text(self, ok):
        miss = [f"{k} ({self.info[k]})" for k, bit in (("A", 1), ("B", 2)) if not ok & bit]
        return "sensor " + " and ".join(miss)

    def stop(self):
        self.running = False
        self.join(timeout=2)
        self.sock.close()
        self.raw_f.close()


# ----------------------------------------------------------------- keyboard
class KeyboardLines(threading.Thread):
    """Reads typed lines in the background so we can also react to the ESP32 button."""

    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()

    def run(self):
        for line in sys.stdin:
            self.q.put(line)
        self.q.put(None)                         # end of input


def ask(prompt, rx, keys):
    """Show prompt; return typed text, or "" when Enter or the ESP32 button is pressed."""
    print(prompt, end="", flush=True)
    start = rx.button_presses
    while True:
        try:
            line = keys.q.get(timeout=0.05)
        except queue.Empty:
            if rx.button_presses != start:
                print("[button]")
                return ""
            continue
        if line is None:
            raise EOFError
        return line.strip().lower()


# ------------------------------------------------------------------ display
def chip(name):
    """'BNO055@0x28' -> 'BNO055', 'MPU-9250(AK8963)@0x68' -> 'MPU-9250(AK8963)'"""
    return name.split("@")[0]


def wait_for_esp(rx, args):
    print("Waiting for data from the ESP32 ", end="", flush=True)
    t0 = time.time()
    hinted = False
    while rx.latest() is None:
        time.sleep(0.5)
        print(".", end="", flush=True)
        if not hinted and time.time() - t0 > 10:
            hinted = True
            print("\n  Nothing yet. Check: ESP32 powered and on the SAME WiFi as this laptop,"
                  "\n  firewall allows Python (Ubuntu: sudo ufw allow 4210/udp),"
                  "\n  port matches the sketch (4210). Still waiting ", end="", flush=True)
    t1 = time.time()
    while rx.info["A"] == "?" and time.time() - t1 < 6:     # sensor names come every 5 s
        time.sleep(0.2)
    s = rx.latest()
    print(f"\nReceiving from ESP32 at {rx.esp_ip}")
    print(f"BNO055 mode: {MODE_NAMES.get(s['mode'], s['mode'])}")
    bot, top = ("B", "A") if args.swap else ("A", "B")
    if not args.single:
        print(f"Bottom = sensor {bot} ({rx.info[bot]}),  top = sensor {top} ({rx.info[top]})"
              + ("   (swapped with --swap)" if args.swap else "   (use --swap if it's the other way round)"))
    if args.single:
        print(f"SINGLE-SENSOR mode: using sensor {bot} ({rx.info[bot]}) only - no zeroing or"
              " calibration needed.")
    elif rx.cal:
        print(f"Cross-calibration ON (sensor B lined up with A, fit error {rx.cal['rms']:.2f} uT)")
        old = rx.cal.get("sensors")
        if old and old != rx.info and "?" not in rx.info.values():
            print(f"!! The calibration was made with A={old.get('A')}, B={old.get('B')}. "
                  "Run --calibrate again if you changed sensors.")
    elif not args.calibrate:
        different = chip(rx.info["A"]) != chip(rx.info["B"]) and "?" not in rx.info.values()
        print("!! No cross-calibration yet." +
              (" Your two sensors are DIFFERENT types: run  python3 mag_logger.py --calibrate  first,"
               "\n   otherwise the difference between them is not meaningful." if different else
               " Running  --calibrate  once makes the difference more accurate."))
    if not args.single and not s["both"]:
        print(f"!! WARNING: {rx.missing_text(s['ok'])} is not answering. Points can't be"
              " measured until both work.")
        if s["ok_bot"]:
            print("   (If your MPU board has no magnetometer, you can still map with one sensor:"
                  " add --single)")
    print()


def fit(line):
    """Cut a status line to the terminal width so '\\r' updates stay on one line."""
    w = max(20, shutil.get_terminal_size((100, 20)).columns - 1)
    return line[:w].ljust(w)


def status_line(s, rx, baseline, single=False):
    if single:
        if not s["ok_bot"]:
            return f"!! the sensor is not answering - check its wires   lost={rx.lost}"
        b = s["bot"]
        return (f"|B|={s['b_bot']:8.2f} uT   Bx={b[0]:+8.2f}  By={b[1]:+8.2f}  Bz={b[2]:+8.2f}"
                f"   lost={rx.lost}")
    if not s["both"]:
        return f"!! {rx.missing_text(s['ok'])} not answering - check its wires   lost={rx.lost}"
    d = vsub(s["d"], rx.null) if rx.null else s["d"]
    db = vnorm(d)
    g = f" ({db / baseline:.2f} uT/cm)" if baseline else ""
    z = "" if rx.null else " NOT ZEROED"
    cal = f"  cal {s['cal_bot']}/{s['cal_top']}" if s["fusion"] else ""
    return (f"|dB|={db:7.2f} uT{g}{z}  dB=({d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f})  "
            f"bottom|B|={s['b_bot']:.1f} top|B|={s['b_top']:.1f}{cal}  lost={rx.lost}")


def live(rx, single=False):
    s = rx.latest()
    if not s:
        return ""
    if single:
        return f"(now |B|={s['b_bot']:.1f} uT)" if s["ok_bot"] else "(!! sensor missing)"
    if not s["both"]:
        return f"(!! {rx.missing_text(s['ok'])} missing)"
    d = vsub(s["d"], rx.null) if rx.null else s["d"]
    return f"(now |dB|={vnorm(d):.1f} uT{'' if rx.null else ', not zeroed'})"


# ----------------------------------------------------------------- measuring
def measure(rx, n, settle, timeout, single=False):
    """Average n samples of both sensors. Returns (result, None) or (None, error text)."""
    time.sleep(settle)                    # let your hand / the probe stop moving
    samples = rx.collect(n, timeout)
    if not samples:
        return None, "No data received - is the ESP32 still connected?"
    if single:                            # one sensor only (the bottom one)
        good = [s for s in samples if s["ok_bot"]]
        if len(good) < max(2, len(samples) // 2):
            return None, "The sensor is not answering (see the ESP32 message)."
        bot = [s["bot"] for s in good]
        std = vstd(bot)
        return {"bot": vmean(bot), "std": std, "noise": max(std), "n": len(good),
                "cal_bot": good[-1]["cal_bot"], "heading": good[-1]["heading"]}, None
    good = [s for s in samples if s["both"]]
    if len(good) < max(2, len(samples) // 2):
        return None, f"{rx.missing_text(samples[-1]['ok'])} is not answering (see the ESP32 message)."
    bot, top, d = [s["bot"] for s in good], [s["top"] for s in good], [s["d"] for s in good]
    d_std = vstd(d)
    return {"bot": vmean(bot), "top": vmean(top), "d": vmean(d), "d_std": d_std,
            "noise": max(max(vstd(bot)), max(vstd(top)), max(d_std)),
            "n": len(good), "cal_bot": good[-1]["cal_bot"], "cal_top": good[-1]["cal_top"]}, None


def grid_points(xmin, xmax, ymin, ymax, step):
    """Snake order (left->right, then right->left on the next row) so you walk less."""
    def frange(a, b, st):
        out, k = [], 0
        while a + k * st <= b + 1e-9:
            out.append(round(a + k * st, 6))
            k += 1
        return out
    xs, ys = frange(xmin, xmax, step), frange(ymin, ymax, step)
    pts = []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else xs[::-1]
        pts += [(x, y) for x in row]
    return pts


def calibrate(rx, args, cal_path):
    """Learn how to line sensor B up with sensor A, from many probe orientations."""
    keys = KeyboardLines()
    keys.start()
    print("CROSS-CALIBRATION - lines sensor B up with sensor A.")
    print("  1. Take the probe at least 1 m away from the magnet, steel tables, laptops, phones.")
    print(f"  2. After Enter, SLOWLY turn the probe in ALL directions for {args.cal_seconds:g} s:")
    print("     tilt it, roll it, point it up/down, turn it upside down - like drawing a big '8'.")
    print("     Go slowly: one full turn should take about 10 s (fast turning spoils the result).")
    print("     Keep both boards fixed on the stick (that's how they'll be used).\n")
    ask("Press Enter (or the IO0 button) to start > ", rx, keys)
    t0 = time.time()
    while time.time() - t0 < args.cal_seconds:
        time.sleep(1.0)
        got = [s for s in rx.since(t0) if s["both"]]
        if got:
            a = [s["a"] for s in got]
            rng = [max(v[i] for v in a) - min(v[i] for v in a) for i in range(3)]
            target = 1.4 * statistics.fmean(vnorm(v) for v in a)
            cover = min(100, int(100 * min(rng) / max(target, 1)))
            print(f"\r  {time.time() - t0:4.0f} s   {len(got)} samples   turned enough: {cover:3d} %"
                  "   (keep turning, aim for 100 %)   ", end="", flush=True)
    print()
    got = [s for s in rx.since(t0) if s["both"]]
    if len(got) < 100:
        print(f"!! Only {len(got)} good samples - are both sensors working? Not saved.")
        return
    got.sort(key=lambda s: s["esp_ms"])
    cov, rng = coverage([s["a"] for s in got])
    if cov < 0.7:
        print(f"!! The probe was not turned enough in all directions (ranges {fvec(rng, 0)} uT). "
              "Try again and turn it more, including upside down. Not saved.")
        return
    use, used_rate = got, None
    for max_rate in (25, 40, 60, 100):                 # uT per second
        sel = still_samples(got, max_rate)
        if len(sel) >= 150 and coverage([s["a"] for s in sel])[0] >= 0.7:
            use, used_rate = sel, max_rate
            break
    A, B = [s["a"] for s in use], [s["b"] for s in use]
    try:
        cal = robust_fit(A, B)
    except ValueError as e:
        print(f"!! {e}. Not saved.")
        return
    fast = len(got) - len(use)
    print(f"Used {cal['n_used']} of {len(got)} samples"
          + (f" ({fast} left out: taken while turning faster than {used_rate} uT/s)" if used_rate else
             "  (!! almost everything was turned fast - next time turn more slowly)")
          + (f", {len(use) - cal['n_used']} glitches left out" if len(use) > cal["n_used"] else ""))
    rms = cal["rms"]
    quality = ("very good" if rms < 1.5 else "good" if rms < 3 else "usable" if rms < 6 else
               "POOR - steel/magnets nearby (go outdoors, away from cars and buildings), a board or\n"
               "   cable moved on the stick, or it was turned too fast")
    print("\nResult: sensor B lined up with A as   B_lined_up = M x B + c")
    for k in range(3):
        print(f"   [{cal['M'][k][0]:+7.3f} {cal['M'][k][1]:+7.3f} {cal['M'][k][2]:+7.3f}]   c={cal['c'][k]:+8.2f} uT")
    print(f"Remaining error {rms:.2f} uT  ->  {quality}")
    cal.update({"n": len(got), "time": datetime.now().isoformat(timespec="seconds"),
                "sensors": dict(rx.info)})
    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"Saved {cal_path}  - used automatically from now on.")
    print("Next:  python3 mag_logger.py --watch   (zero it, then turn the probe: |dB| should stay small)")


SINGLE_FIELDS = ["x", "y", "bx", "by", "bz", "b", "bx_std", "by_std", "bz_std",
                 "n", "cal_mag", "heading", "pc_time"]


def mapping(rx, map_path, args):
    f = open(map_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(SINGLE_FIELDS if args.single else MAP_FIELDS)
    timeout = args.samples / 5 + 5
    keys = KeyboardLines()
    keys.start()
    single = args.single

    def write_row(kind, x, y, r):
        f3 = lambda v: f"{v:.3f}"
        w.writerow([kind, x, y,
                    *map(f3, r["bot"]), f3(vnorm(r["bot"])),
                    *map(f3, r["top"]), f3(vnorm(r["top"])),
                    *map(f3, r["d"]), f3(vnorm(r["d"])), *map(f3, r["d_std"]),
                    r["n"], r["cal_bot"], r["cal_top"],
                    args.baseline if args.baseline else "",
                    "B" if args.swap else "A", f"{time.time():.3f}"])
        f.flush()

    def do_null():
        r, err = measure(rx, args.samples, args.settle, timeout)
        if r is None:
            print(f"   !! {err} Not zeroed.")
            return False
        write_row("null", "", "", r)
        rx.null = r["d"]
        print(f"\a   zeroed. Offset between the two sensors {fvec(r['d'])} uT "
              f"(|{vnorm(r['d']):.1f}| uT) will be removed.")
        return True

    def do_point(x, y):
        r, err = measure(rx, args.samples, args.settle, timeout, single)
        if r is None:
            print(f"   !! {err} Point not saved.")
            return False
        if single:
            b = r["bot"]
            w.writerow([f"{x:g}", f"{y:g}", *(f"{v:.3f}" for v in b), f"{vnorm(b):.3f}",
                        *(f"{v:.3f}" for v in r["std"]), r["n"], r["cal_bot"],
                        f"{r['heading']:.1f}", f"{time.time():.3f}"])
            f.flush()
            print(f"\a   saved x={x:g} y={y:g}   Bx={b[0]:.2f} By={b[1]:.2f} Bz={b[2]:.2f}"
                  f"  |B|={vnorm(b):.2f} uT  ({r['n']} samples)"
                  + ("   <- noisy, probe moved? press r to redo" if r["noise"] > args.noise_warn else ""))
            return True
        write_row("point", f"{x:g}", f"{y:g}", r)
        d = vsub(r["d"], rx.null) if rx.null else r["d"]
        db = vnorm(d)
        g = f" = {db / args.baseline:.2f} uT/cm" if args.baseline else ""
        noisy = r["noise"] > args.noise_warn
        print(f"\a   saved x={x:g} y={y:g}   |dB|={db:.2f} uT{g}"
              f"{'' if rx.null else ' (not zeroed)'}   bottom |B|={vnorm(r['bot']):.1f}"
              f"  top |B|={vnorm(r['top']):.1f}  ({r['n']} samples)"
              + ("   <- noisy, probe moved? press r to redo" if noisy else ""))
        return True

    try:
        if args.grid:
            pts = grid_points(*args.grid)
            print(f"Grid mapping: {len(pts)} points. Keep the probe upright and pointing the SAME way"
                  " at every point.")
            print("Enter or the IO0 button on the ESP32 = measure" + ("" if single else "   z = zero again")
                  + "   s = skip   r = redo previous   q = quit\n")
            if not single:
                print("STEP 1 - ZERO: hold the probe the way you'll hold it for mapping, but where there is")
                print("no magnet nearby (or take the magnet away for a moment).")
                cmd = ask(f"   Enter / button = zero now   s = skip {live(rx)} > ", rx, keys)
                if cmd == "q":
                    return
                if cmd != "s":
                    do_null()
                print("\nSTEP 2 - MAP (put the magnet back if you removed it)")
            i = 0
            while i < len(pts):
                x, y = pts[i]
                cmd = ask(f"[{i + 1}/{len(pts)}] put probe at x={x:g}, y={y:g} {live(rx, single)} > ", rx, keys)
                if cmd == "q":
                    break
                if cmd == "z" and not single:
                    do_null()
                    continue
                if cmd == "s":
                    i += 1
                    continue
                if cmd == "r":
                    if i > 0:
                        i -= 1
                        print(f"   going back to x={pts[i][0]:g}, y={pts[i][1]:g} (new value replaces old)")
                    continue
                if cmd:
                    print("   Enter = measure,  z = zero,  s = skip,  r = redo previous,  q = quit")
                    continue
                if do_point(x, y):
                    i += 1
            print("\nGrid finished." if i >= len(pts) else "\nStopped.")
        else:
            print("Free mapping. Put the probe somewhere, type its position as  x y  and press Enter.")
            print("Keep the probe upright and pointing the SAME way at every point.")
            print("q = quit" if single else
                  "z = zero the gradiometer (do this FIRST, away from the magnet)   q = quit")
            print()
            while True:
                cmd = ask(f"position x y  ({'q' if single else 'z / q'}) {live(rx, single)} > ", rx, keys)
                if cmd == "q":
                    break
                if cmd == "z" and not single:
                    do_null()
                    continue
                try:
                    x, y = [float(v) for v in cmd.replace(",", " ").split()]
                except ValueError:
                    print("   type two numbers, e.g.  10 25")
                    continue
                do_point(x, y)
    finally:
        f.close()


def watch(rx, args):
    keys = KeyboardLines()
    keys.start()
    print("Live values (Ctrl+C to stop). Everything is also saved to the raw log.")
    if not args.single:
        print("Press Enter (or the IO0 button) to ZERO the gradiometer where you are now.")
        print("Check: after zeroing, slowly turn the probe around. If |dB| stays within a few uT,")
        print("the two sensors are well lined up.")
    print()
    btn = rx.button_presses
    while True:
        zero = False
        try:
            zero = keys.q.get_nowait() is not None        # None = keyboard closed, ignore
        except queue.Empty:
            pass
        if rx.button_presses != btn:
            btn = rx.button_presses
            zero = True
        if zero and not args.single:
            print()
            r, err = measure(rx, args.samples, 0.0, args.samples / 5 + 5)
            if r:
                rx.null = r["d"]
                print(f"   zeroed. Offset between the sensors {fvec(r['d'])} uT")
            else:
                print(f"   !! {err}")
        s = rx.latest()
        if s:
            print("\r" + fit(status_line(s, rx, args.baseline, args.single)), end="", flush=True)
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser(description="Log the 2-sensor magnetometer gradiometer sent by the ESP32.")
    ap.add_argument("--port", type=int, default=4210, help="UDP port (same as in the sketch)")
    ap.add_argument("--watch", action="store_true", help="only show live values (and log raw data)")
    ap.add_argument("--calibrate", action="store_true", help="line sensor B up with sensor A (once)")
    ap.add_argument("--cal-seconds", type=float, default=45, help="how long --calibrate records")
    ap.add_argument("--no-cal", action="store_true", help="ignore the saved cross-calibration")
    ap.add_argument("--grid", type=float, nargs=5, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "STEP"),
                    help="guided grid mapping")
    ap.add_argument("--baseline", type=float, default=None,
                    help="distance between the two sensor chips in cm (adds uT/cm)")
    ap.add_argument("--swap", action="store_true", help="sensor B is the BOTTOM one")
    ap.add_argument("--single", action="store_true",
                    help="use ONE sensor only (A, or B with --swap): plain field map, no gradient")
    ap.add_argument("--samples", type=int, default=20, help="samples averaged per point (20 = 1 s)")
    ap.add_argument("--settle", type=float, default=0.5, help="seconds to wait after Enter before measuring")
    ap.add_argument("--noise-warn", type=float, default=2.0, help="warn if std dev above this (uT)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    args = ap.parse_args()
    if args.baseline is not None and args.baseline <= 0:
        args.baseline = None
    if os.name == "nt":
        os.system("")                      # lets Windows terminals handle the live display

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(args.outdir, f"raw_{stamp}.csv")
    map_path = os.path.join(args.outdir, f"map_{stamp}.csv")
    cal_path = os.path.join(args.outdir, "cross_cal.json")
    cal = None if (args.no_cal or args.calibrate) else load_cal(cal_path)

    try:
        rx = Receiver(args.port, raw_path, args.swap, cal)
    except OSError as e:
        sys.exit(f"Could not open UDP port {args.port}: {e}\n(Is another mag_logger already running?)")
    rx.start()
    print(f"Raw log: {raw_path}")

    try:
        wait_for_esp(rx, args)
        if args.calibrate:
            calibrate(rx, args, cal_path)
        elif args.watch:
            watch(rx, args)
        else:
            print(f"Map file: {map_path}")
            mapping(rx, map_path, args)
            print(f"\nNow make the map:  python3 make_map.py \"{map_path}\"")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        rx.stop()
        print(f"Raw data saved to {raw_path}  (lost packets: {rx.lost})")


if __name__ == "__main__":
    main()
