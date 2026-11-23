#!/usr/bin/env python3
"""
log_serial.py - save the ESP32 raw logger's data to a CSV file on the laptop.

    python3 log_serial.py /dev/ttyUSB0 warmup.csv --minutes 15     (USB, Ubuntu)
    python log_serial.py COM5 calib.csv --minutes 2                 (USB, Windows)
    python log_serial.py wifi scan.csv --minutes 2                  (WiFi: finds the board by itself)
    python3 log_serial.py 172.20.10.3 scan.csv                      (WiFi, using the IP the board printed)

It switches the sketch to CSV mode, saves every data line, shows a live progress line,
and stops after --minutes (or when you press Ctrl+C). At the end it switches the sketch
back to table mode.

For USB: close the Arduino Serial Monitor first: only one program can use the port at a time.
For WiFi: the laptop must be on the same WiFi network as the board.
Needs:  pip install pyserial   (only for USB)
"""
import argparse
import socket
import sys
import time

TCP_PORT = 3333


class TcpLink:
    """WiFi connection to the board, used like a serial port (readline / write)."""

    def __init__(self, host, port, sock=None):
        self.host, self.port, self.buf = host, port, b""
        self.s = sock or socket.create_connection((host, port), timeout=5)
        self.s.settimeout(0.5)

    def readline(self):
        while b"\n" not in self.buf:
            try:
                chunk = self.s.recv(4096)
            except socket.timeout:
                return b""
            if not chunk:
                raise ConnectionError("the board closed the WiFi connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line + b"\n"

    def write(self, data):
        self.s.sendall(data)

    def reset_input_buffer(self):
        self.buf = b""
        self.s.settimeout(0.05)
        try:
            while self.s.recv(4096):
                pass
        except (socket.timeout, OSError):
            pass
        self.s.settimeout(0.5)

    def close(self):
        self.s.close()


def local_ipv4s():
    """This laptop's own addresses on WiFi / LAN."""
    ips = set()
    for target in ("8.8.8.8", "172.20.10.1"):
        try:
            u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            u.connect((target, 80))               # sends nothing; just picks the network's address
            ips.add(u.getsockname()[0])
            u.close()
        except OSError:
            pass
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith(("127.", "169.254.", "0.")))


def find_board(port=TCP_PORT):
    """Find the board on the laptop's WiFi: try gradiometer.local, then knock on every address."""
    from concurrent.futures import ThreadPoolExecutor
    print("Looking for the board on WiFi...")
    try:
        return "gradiometer.local", socket.create_connection(("gradiometer.local", port), timeout=2)
    except OSError:
        pass
    mine = local_ipv4s()
    hosts = []
    for ip in mine:
        base = ip.rsplit(".", 1)[0]
        hosts += [f"{base}.{i}" for i in range(1, 255) if f"{base}.{i}" != ip and f"{base}.{i}" not in hosts]

    def knock(h):
        try:
            return h, socket.create_connection((h, port), timeout=0.6)
        except OSError:
            return h, None

    found = None
    with ThreadPoolExecutor(64) as ex:
        for h, sk in ex.map(knock, hosts):
            if sk is not None and found is None:
                found = (h, sk)
            elif sk is not None:
                sk.close()
    if found is None:
        sys.exit("Board not found on WiFi. Check:\n"
                 f"  - this laptop is on the board's WiFi (laptop addresses: {', '.join(mine) or 'none'})\n"
                 "  - the hotspot was on BEFORE the board started (press EN on the board, wait 15 s)\n"
                 "  - iPhone: Personal Hotspot ON and Maximize Compatibility ON")
    return found


def is_usb(port):
    return port.startswith("/dev/") or port.upper().startswith("COM")

HEADER = "time_ms,point,Ax,Ay,Az,Bx,By,Bz"


def open_port(port, baud):
    """Open the port, trying not to reset the ESP32 (some boards reset anyway)."""
    try:
        import serial
    except ImportError:
        sys.exit("pyserial is missing. Install it with:  pip install pyserial")
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, baud, 0.5
    try:
        ser.dtr = False
        ser.rts = False
        ser.open()
    except (OSError, serial.SerialException):
        if ser.is_open:
            ser.close()
        ser = serial.Serial(port, baud, timeout=0.5)
    return ser


def parse_row(line):
    """Return the 8 values of a data line, or None if it is not a data line."""
    parts = line.split(",")
    if len(parts) != 8:
        return None
    try:
        int(parts[0]), int(parts[1])
        [float(p) for p in parts[2:]]          # 'nan' is allowed
    except ValueError:
        return None
    return parts


def main():
    ap = argparse.ArgumentParser(description="Save the ESP32 raw logger's data to a CSV file.")
    ap.add_argument("port", help="USB port (COM5, /dev/ttyUSB0), or 'wifi' to find the board, or its WiFi address (172.20.10.3)")
    ap.add_argument("outfile", help="CSV file to write, e.g. calib.csv")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this many minutes (0 = until Ctrl+C)")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    wifi = not is_usb(args.port)
    host, _, p = args.port.partition(":")
    tcp_port = int(p) if p else TCP_PORT
    if wifi and host.lower() == "wifi":
        host, sock = find_board(tcp_port)
        ser = TcpLink(host, tcp_port, sock)
        print(f"Found the board at {host} over WiFi.")
        time.sleep(0.5)
    elif wifi:
        try:
            ser = TcpLink(host, tcp_port)
        except OSError as e:
            sys.exit(f"Could not reach {host}:{tcp_port} over WiFi ({e}).\n"
                     "Check the board printed 'WiFi | connected', the laptop is on the same WiFi, "
                     "or use the IP address the board printed instead of the name.")
        print(f"Connected over WiFi to {host}:{tcp_port}.")
        time.sleep(0.5)
    else:
        ser = open_port(args.port, args.baud)
        print(f"Opened {args.port}. Waiting 3 s in case the ESP32 restarted...")
        time.sleep(3)
    ser.reset_input_buffer()
    ser.write(b"c\n")                            # CSV mode

    rows, skipped, t0, last_show = 0, 0, time.time(), 0.0
    last_ask, got_bytes, shown_other, warned = t0, False, 0, False
    garbled, garble_warned = 0, False
    stop_at = t0 + args.minutes * 60 if args.minutes > 0 else None
    print(f"Saving to {args.outfile}. " + (f"Stops after {args.minutes:g} min." if stop_at else "Press Ctrl+C to stop."))

    try:
        with open(args.outfile, "w", newline="") as f:
            f.write(HEADER + "\n")
            while stop_at is None or time.time() < stop_at:
                try:
                    raw = ser.readline()
                except (ConnectionError, OSError) as e:
                    if not wifi:
                        raise
                    print(f"\n  WiFi connection lost ({e}). Reconnecting...")
                    while stop_at is None or time.time() < stop_at:
                        time.sleep(2)
                        try:
                            ser = TcpLink(host, tcp_port)
                            ser.write(b"c\n")
                            print("  Reconnected.")
                            break
                        except OSError:
                            print("  ...still trying", end="", flush=True)
                    continue
                now = time.time()
                got_bytes = got_bytes or bool(raw)
                if rows == 0 and now - last_ask > 2:     # no data yet: ask for CSV mode again
                    if not wifi:
                        ser.baudrate = args.baud         # put the speed back in case another program changed it
                    ser.write(b"c\n")
                    last_ask = now
                    if not got_bytes and now - t0 > 6 and not warned:
                        warned = True
                        print("\n  Nothing at all is arriving. Check:")
                        print("   - the Arduino IDE is fully closed (it can hold the port even with the monitor tab closed)")
                        print("   - the port name is right:  ls /dev/ttyUSB*")
                        print("   - press the EN (reset) button on the ESP32 once")
                line = raw.decode("ascii", errors="replace").strip()
                if not line:
                    continue
                if not wifi and sum(ch == "\ufffd" or (ord(ch) < 32 and ch != "\t") for ch in line) > len(line) // 4:
                    garbled += 1
                    if garbled >= 3 and not garble_warned:
                        garble_warned = True
                        print("\n  The text from the board is garbled: another program is changing the USB port's speed.")
                        print("  On Ubuntu this is usually ModemManager. Press Ctrl+C, then run:")
                        print("     sudo systemctl stop ModemManager")
                        print("     sudo fuser -v " + args.port + "      (should print nothing)")
                        print("  unplug and replug the board, wait 5 s, and start this again.")
                    continue
                if line.startswith("#"):         # messages from the sketch, e.g. point changes
                    print("\n  ESP32:", line[1:].strip())
                    continue
                parts = parse_row(line)
                if parts is None:
                    skipped += 1                 # table lines, header, half lines
                    if rows == 0 and shown_other < 5:
                        shown_other += 1
                        print(f"  (received, not data yet: {line[:70]})")
                    continue
                f.write(",".join(parts) + "\n")
                rows += 1
                if now - last_show >= 1:
                    last_show = now
                    f.flush()                    # so the file on disk keeps up
                    a = "  ".join(f"{float(v):7.2f}" for v in parts[2:5])
                    b = "  ".join(f"{float(v):7.2f}" for v in parts[5:8])
                    print(f"\r{now - t0:6.0f} s  rows {rows:6d}  point {parts[1]:>3}  A {a}   B {b}   ",
                          end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ser.write(b"t\n")                    # back to table mode
            ser.close()
        except Exception:
            pass

    secs = time.time() - t0
    print(f"\n\nDone: {rows} rows in {secs:.0f} s (about {rows / max(secs, 1):.1f} per second) saved to {args.outfile}")
    if rows == 0:
        print("No data arrived. Check the port name, close the Serial Monitor, and check the baud rate (115200).")


if __name__ == "__main__":
    main()
