#!/usr/bin/env python3
"""
mag_logger.py - receive BNO055 magnetometer data from the ESP32 over WiFi
and save it to CSV files on this computer.

Needs only plain Python 3 (no extra packages).

USAGE
  python mag_logger.py --watch
        Show live values. Use this first to check everything works.

  python mag_logger.py --grid 0 50 0 30 5
        Guided mapping: x from 0 to 50, y from 0 to 30, every 5 (e.g. cm).
        It tells you where to put the sensor; press Enter to measure each spot.

  python mag_logger.py
        Free mapping: type the position "x y" and press Enter at each spot.

Files go in ./logs/ (next to this script):
  raw_<date_time>.csv  every single sample received (continuous log)
  map_<date_time>.csv  one averaged row per position  -> use with make_map.py
"""
import argparse
import csv
import math
import os
import socket
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime

IN_FIELDS = ["seq", "esp_ms", "mx", "my", "mz", "heading", "roll", "pitch",
             "cal_sys", "cal_gyr", "cal_acc", "cal_mag"]


def parse_packet(data):
    """'MAG,seq,ms,mx,my,mz,h,r,p,cs,cg,ca,cm' -> dict, or None if not ours."""
    try:
        parts = data.decode("ascii").strip().split(",")
    except UnicodeDecodeError:
        return None
    if len(parts) != 13 or parts[0] != "MAG":
        return None
    try:
        vals = [float(p) for p in parts[1:]]
    except ValueError:
        return None
    s = dict(zip(IN_FIELDS, vals))
    for k in ("seq", "esp_ms", "cal_sys", "cal_gyr", "cal_acc", "cal_mag"):
        s[k] = int(s[k])
    s["b"] = math.sqrt(s["mx"] ** 2 + s["my"] ** 2 + s["mz"] ** 2)
    return s


class Receiver(threading.Thread):
    """Background thread: receives UDP packets, writes the raw CSV, keeps a buffer."""

    def __init__(self, port, raw_path):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", port))
        self.sock.settimeout(0.5)
        self.cond = threading.Condition()
        self.buf = deque(maxlen=5000)
        self.count = 0
        self.lost = 0
        self.last_seq = None
        self.esp_ip = None
        self.button_presses = 0
        self._last_btn = None
        self.running = True
        self.raw_f = open(raw_path, "w", newline="")
        self.raw_w = csv.writer(self.raw_f)
        self.raw_w.writerow(["pc_time", "iso_time", "esp_ip"] + IN_FIELDS + ["b_total"])

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
            s = parse_packet(data)
            if s is None:
                continue
            t = time.time()
            s["pc_time"] = t
            if self.last_seq is not None and s["seq"] > self.last_seq + 1:
                self.lost += s["seq"] - self.last_seq - 1
            self.last_seq = s["seq"]
            self.raw_w.writerow([f"{t:.3f}", datetime.fromtimestamp(t).isoformat(timespec="milliseconds"),
                                 addr[0]] + [s[k] for k in IN_FIELDS] + [f"{s['b']:.2f}"])
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

    def stop(self):
        self.running = False
        self.join(timeout=2)
        self.sock.close()
        self.raw_f.close()


class KeyboardLines(threading.Thread):
    """Reads typed lines in the background so we can also react to the ESP32 button."""

    def __init__(self):
        super().__init__(daemon=True)
        self.q = __import__("queue").Queue()

    def run(self):
        for line in sys.stdin:
            self.q.put(line)
        self.q.put(None)                         # end of input


def ask(prompt, rx, keys):
    """Show prompt; return typed text, or "" when Enter or the ESP32 button is pressed."""
    import queue
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


def wait_for_esp(rx, timeout=None):
    print("Waiting for data from the ESP32 ", end="", flush=True)
    t0 = time.time()
    hinted = False
    while rx.latest() is None:
        time.sleep(0.5)
        print(".", end="", flush=True)
        if not hinted and time.time() - t0 > 10:
            hinted = True
            print("\n  Nothing yet. Check: ESP32 powered and on the SAME WiFi as this laptop,"
                  "\n  firewall allows Python (Windows: allow on Private networks),"
                  "\n  port matches the sketch (4210). Still waiting ", end="", flush=True)
        if timeout and time.time() - t0 > timeout:
            return False
    print(f"\nReceiving from ESP32 at {rx.esp_ip}\n")
    return True


def fmt(s):
    return (f"Bx={s['mx']:8.2f}  By={s['my']:8.2f}  Bz={s['mz']:8.2f}  "
            f"|B|={s['b']:8.2f} uT   heading={s['heading']:6.1f}  magcal={s['cal_mag']}/3")


def watch(rx):
    print("Live values (Ctrl+C to stop). Everything is also saved to the raw log.\n")
    while True:
        s = rx.latest()
        if s:
            print("\r" + fmt(s) + f"   lost={rx.lost}   ", end="", flush=True)
        time.sleep(0.1)


def measure(rx, n, settle, timeout):
    time.sleep(settle)                    # let your hand / the sensor stop moving
    samples = rx.collect(n, timeout)
    if not samples:
        return None
    r = {}
    for k in ("mx", "my", "mz"):
        vals = [s[k] for s in samples]
        r[k] = statistics.fmean(vals)
        r[k + "_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    r["b"] = math.sqrt(r["mx"] ** 2 + r["my"] ** 2 + r["mz"] ** 2)
    r["n"] = len(samples)
    r["cal_mag"] = samples[-1]["cal_mag"]
    r["heading"] = samples[-1]["heading"]
    return r


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


def mapping(rx, map_path, args):
    f = open(map_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["x", "y", "bx", "by", "bz", "b", "bx_std", "by_std", "bz_std",
                "n", "cal_mag", "heading", "pc_time"])

    def do_point(x, y):
        r = measure(rx, args.samples, args.settle, timeout=args.samples / 5 + 5)
        if r is None:
            print("   !! No data received - is the ESP32 still connected? Point not saved.")
            return False
        w.writerow([x, y, f"{r['mx']:.3f}", f"{r['my']:.3f}", f"{r['mz']:.3f}", f"{r['b']:.3f}",
                    f"{r['mx_std']:.3f}", f"{r['my_std']:.3f}", f"{r['mz_std']:.3f}",
                    r["n"], r["cal_mag"], f"{r['heading']:.1f}", f"{time.time():.3f}"])
        f.flush()
        noisy = max(r["mx_std"], r["my_std"], r["mz_std"]) > args.noise_warn
        print(f"\a   saved  x={x} y={y}  Bx={r['mx']:.2f} By={r['my']:.2f} Bz={r['mz']:.2f}"
              f"  |B|={r['b']:.2f} uT  ({r['n']} samples, magcal {r['cal_mag']}/3)"
              + ("   <- noisy, sensor moved? press r to redo" if noisy else ""))
        return True

    try:
        if args.grid:
            pts = grid_points(*args.grid)
            print(f"Grid mapping: {len(pts)} points. Keep the sensor pointing the SAME way at every point.")
            print("Enter or the IO0 button on the ESP32 = measure   s = skip   r = redo previous   q = quit\n")
            keys = KeyboardLines()
            keys.start()
            i = 0
            while i < len(pts):
                x, y = pts[i]
                s = rx.latest()
                now = f"(now |B|={s['b']:.1f} uT)" if s else ""
                cmd = ask(f"[{i + 1}/{len(pts)}] put sensor at x={x}, y={y} {now} > ", rx, keys)
                if cmd == "q":
                    break
                if cmd == "s":
                    i += 1
                    continue
                if cmd == "r":
                    if i > 0:
                        i -= 1
                        print(f"   going back to x={pts[i][0]}, y={pts[i][1]} (new value replaces old)")
                    continue
                if do_point(x, y):
                    i += 1
            print("\nGrid finished." if i >= len(pts) else "\nStopped.")
        else:
            print("Free mapping. Put the sensor somewhere, type its position as  x y  and press Enter.")
            print("Keep the sensor pointing the SAME way at every point. Type q to quit.\n")
            while True:
                cmd = input("position x y > ").strip().lower()
                if cmd == "q":
                    break
                try:
                    x, y = [float(v) for v in cmd.replace(",", " ").split()]
                except ValueError:
                    print("   type two numbers, e.g.  10 25")
                    continue
                do_point(x, y)
    finally:
        f.close()


def main():
    ap = argparse.ArgumentParser(description="Log BNO055 magnetometer data sent by the ESP32 over WiFi.")
    ap.add_argument("--port", type=int, default=4210, help="UDP port (same as in the sketch)")
    ap.add_argument("--watch", action="store_true", help="only show live values (and log raw data)")
    ap.add_argument("--grid", type=float, nargs=5, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "STEP"),
                    help="guided grid mapping")
    ap.add_argument("--samples", type=int, default=20, help="samples averaged per point (20 = 1 s)")
    ap.add_argument("--settle", type=float, default=0.5, help="seconds to wait after Enter before measuring")
    ap.add_argument("--noise-warn", type=float, default=2.0, help="warn if std dev above this (uT)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(args.outdir, f"raw_{stamp}.csv")
    map_path = os.path.join(args.outdir, f"map_{stamp}.csv")

    try:
        rx = Receiver(args.port, raw_path)
    except OSError as e:
        sys.exit(f"Could not open UDP port {args.port}: {e}\n(Is another mag_logger already running?)")
    rx.start()
    print(f"Raw log: {raw_path}")

    try:
        wait_for_esp(rx)
        if args.watch:
            watch(rx)
        else:
            print(f"Map file: {map_path}")
            mapping(rx, map_path, args)
            print(f"\nNow make the map:  python make_map.py \"{map_path}\"")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        rx.stop()
        print(f"Raw data saved to {raw_path}  (lost packets: {rx.lost})")


if __name__ == "__main__":
    main()
