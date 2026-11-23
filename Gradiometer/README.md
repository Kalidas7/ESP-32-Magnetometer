# Gradiometer: MPU9250 + BNO055 on one ESP32

Two magnetometers on one stick, read by one ESP32. The ESP32 only reads and sends raw numbers
(over USB, and over WiFi if available). A PC does everything else: noise tests, calibration,
maps and finding a magnet's position and depth.

```
Gradiometer/
  README.md
  Gradiometer_RawLogger/
    Gradiometer_RawLogger.ino      ESP32 sketch (open this in the Arduino IDE)
    secrets.h                      WiFi name and password
    secrets_example.h              template for secrets.h
  tools/
    log_serial.py                  saves the ESP32's data to a CSV file (USB or WiFi)
    plot_log.py                    plots a recording, prints noise and drift
    noise_report.py                noise (standard deviation) of a still recording
    calibrate.py                   works out both sensors' corrections
    make_map.py                    turns a grid scan into maps and finds the magnet
```

## 1. Get the code (on any PC)

    git clone https://github.com/Kalidas7/ESP-32-Magnetometer.git

or on GitHub: Code > Download ZIP, then unzip. Everything below happens inside the `Gradiometer` folder.

## 2. Put the program on the ESP32 (once, Linux or Windows)

The ESP32 keeps the program after this, even when unplugged. You only redo it when the sketch changes.

1. Install Arduino IDE 2 from arduino.cc/en/software.
2. ESP32 boards: File > Preferences > "Additional boards manager URLs", paste
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`, click OK.
   Tools > Board > Boards Manager, search `esp32`, install **esp32 by Espressif Systems**.
3. Libraries (Tools > Manage Libraries): **MPU9250** by hideakitai, and **Adafruit BNO055** by Adafruit
   (click **Install all** for its dependencies).
4. WiFi: the hotspot name and password are in `Gradiometer_RawLogger/secrets.h`. Edit it for another network.
   Delete `secrets.h` for USB only.
5. Open `Gradiometer_RawLogger/Gradiometer_RawLogger.ino`. Tools > Board > esp32 > **NodeMCU-32S**,
   Tools > Port > your port. Click Upload.
6. Serial Monitor at **115200** baud, **Newline**, press EN: both sensors must say OK, and the WiFi line says
   `connected ... laptop uses 172.20.10.x` or `USB only`.

Port problems:
- Linux "Permission denied": `sudo usermod -a -G dialout $USER`, then log out and back in.
  Port appears and disappears: `sudo apt remove brltty`.
- Windows, no port: install the USB driver for the board's USB chip (CP210x or CH340).

## 3. Record and analyse on any PC (for example a Windows laptop)

The PC that records does not need the Arduino IDE. It needs Python:

- Windows: install Python 3 from python.org (tick "Add python.exe to PATH"), then in Command Prompt:
  `pip install pyserial numpy matplotlib`
- Ubuntu: `sudo apt install python3-serial python3-numpy python3-matplotlib`
  (if your terminal shows `(base)`, type `conda deactivate` first)

Connect the PC to the ESP32 by **USB** (port `COM5` on Windows, `/dev/ttyUSB0` on Linux) or by **WiFi**
(the PC joined to the same hotspot; write `wifi` instead of a port and the logger finds the board itself).
Open a terminal in the `tools` folder and run the commands below.
For USB, close the Arduino Serial Monitor first.

| Job | Windows | Linux |
|---|---|---|
| Record over USB | `python log_serial.py COM5 quiet.csv --minutes 2` | `python3 log_serial.py /dev/ttyUSB0 quiet.csv --minutes 2` |
| Record over WiFi | `python log_serial.py wifi scan.csv` | `python3 log_serial.py wifi scan.csv` |
| Plot a recording | `python plot_log.py quiet.csv` | `python3 plot_log.py quiet.csv` |
| Noise report | `python noise_report.py quiet.csv` | `python3 noise_report.py quiet.csv` |
| Calibrate | `python calibrate.py calib.csv --field 41.8` | `python3 calibrate.py calib.csv --field 41.8` |
| Map | `python make_map.py magnet.csv --nx 9 --ny 9 --spacing 5 --background background.csv` | same with `python3` |

Without `--minutes`, recording runs until Ctrl+C.
`--field` = your city's Earth-field strength in uT (Delhi 48.8, Kolkata 46.5, Mumbai 43.6,
Hyderabad 43.4, Chennai 42.1, Bengaluru 41.8, Thiruvananthapuram 40.9).

## 4. Calibration

Record 2 minutes while turning the stick slowly through every direction, away from steel, then run
`calibrate.py`. Pass marks:

| Check | Want |
|---|---|
| Directions covered | 70% or more |
| \|A\| and \|B\| spread after own fix | 1 uT or less |
| A - B after matching | 1 uT or less |
| Boards from square | under 10 degrees |

`calibration.json` holds the numbers. Add `--cal calibration.json` to `make_map.py` to use them.

## 5. Grid scan for a map

1. Draw a grid of dots on a flat cardboard or plywood sheet, for example 9 x 9 dots 5 cm apart.
   Put a ruler or wooden strip along each row as a guide so the stick cannot turn.
2. Mark the centre of the stick (half way between the two chips). Lay the stick along the rows,
   always pointing the same way, printed X arrows along the rows. Never turn it around.
3. Start recording without a time limit. Press EN (or type `r`) so points start at 1.
4. Point 1 is the first corner. At each dot: put the stick centre on it, press BOOT, hold still while the
   blue LED is on (6 s), move when it goes off. Go along row 1, then start row 2 from the same side.
5. Scan the whole grid once WITHOUT the magnet (`background.csv`), then once WITH the magnet under the sheet
   at a measured position and depth (`magnet.csv`). Write the magnet's position and depth down.
6. Run `make_map.py` (table above). If sensor A is at the right-hand end of the stick, add `--a-end right`.

## 6. WiFi notes

- iPhone hotspot: Settings > Personal Hotspot > Allow Others to Join ON and **Maximize Compatibility ON**
  (the ESP32 only works on 2.4 GHz). Keep that screen open while the ESP32 starts.
- The board tries for 10 s at start-up. If it can't connect, WiFi is switched off and USB is used.
  USB always works as well.
- `wifi` as the port makes the logger search the hotspot for the board. You can also give the address
  the board prints at start-up (for example `172.20.10.3`) or `gradiometer.local`.
- The board only looks for WiFi when it starts: turn the hotspot on first, then press EN.
- WiFi draws current in bursts that the sensors can feel. Measure the noise with WiFi on before trusting maps.

## 7. Test: where does B's 5-second pulse come from?

Three 2-minute recordings on the laptop. Keep everything still; keep the phone 1 m from the board.

| # | File | Setup | Command (Windows) |
|---|---|---|---|
| 1 | `near_usb.csv` | Board on the laptop's USB, next to the desktop PC and power strips. Hotspot OFF, press EN. | `python log_serial.py COM5 near_usb.csv --minutes 2` |
| 2 | `near_wifi.csv` | Same place. Hotspot ON, laptop joined to it, press EN, wait 15 s. | `python log_serial.py wifi near_wifi.csv --minutes 2` |
| 3 | `far_wifi.csv` | Board on a power bank, 2 m or more from the PC, monitor, power strips and the laptop's charger. Laptop on battery. | `python log_serial.py wifi far_wifi.csv --minutes 2` |

Then `python noise_report.py near_usb.csv` (and the other two). Section 4 says whether B's pulse is there.

- 2 worse than 1: the WiFi radio adds noise.
- Pulse gone in 3: the PC or mains wiring causes it.
- Pulse still there in 3: it comes from inside the BNO055.
