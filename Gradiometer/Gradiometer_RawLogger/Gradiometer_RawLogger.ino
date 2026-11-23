/*
  Gradiometer_RawLogger.ino  -  step 1 of the plan: read both magnetometers, print raw numbers.

      A = MPU9250 magnetometer   (hideakitai "MPU9250" library)
      B = BNO055 magnetometer    (Adafruit "Adafruit BNO055" library), in MAGONLY mode

  This sketch does NOT calibrate or match anything. It only reads and prints, in uT.
  Calibration, matching, maps and fitting are done later on the laptop, from saved data.

  WIRING (both sensors on the same I2C bus)
      VCC/VIN -> 3V3,  GND -> GND,  SDA -> GPIO21,  SCL -> GPIO22
      MPU9250 at 0x68 (AD0 low), BNO055 at 0x28 or 0x29 (found automatically)

  OUTPUT
      Table mode (default): one averaged line every 0.5 s, easy to read in the Serial Monitor.
      CSV mode:             20 lines per second, for saving to a file on the laptop.
                            Every line that is not data starts with '#'.

  COMMANDS (type the letter in the Serial Monitor, then press Enter)
      t  table mode
      c  CSV mode
      s  noise test: keep the stick still for 10 s, then a table of mean and noise is printed
      n  next point (same as pressing the BOOT button)
      r  reset the point number to 0

  GRID SCAN (for maps)
      Put the stick on a grid point, press BOOT: the LED turns ON = hold still.
      After HOLD_MS the LED turns OFF = move to the next point and press again.
      Presses while the LED is on are ignored, so a double press cannot skip a point.
      h  show this list

  Serial Monitor: 115200 baud, line ending "Newline".

  WIFI (optional)
      Put your WiFi name and password in secrets.h (copy secrets_example.h, same folder).
      At start-up the board tries WiFi for 10 s. If it connects, the laptop can read the data over WiFi:
          python3 log_serial.py gradiometer.local data.csv      (or use the IP address it prints)
      If it does not connect, WiFi is switched off and only USB is used.
      USB always works as well, so nothing is lost either way.
      WiFi draws current in bursts, which the sensors can feel: run a noise test with WiFi on before trusting it.
*/
#include <Wire.h>
#include <math.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include "MPU9250.h"
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

// =============================================================== settings
const bool     USE_A         = true;     // false = ignore the MPU9250 (to test the BNO055 alone)
const bool     USE_B         = true;     // false = ignore the BNO055  (to test the MPU9250 alone)
const uint8_t  MPU_ADDR      = 0x68;     // 0x69 if the MPU9250's AD0 pin is high

// Which raw axis becomes the X, Y, Z that is printed.
// +1 = raw X, +2 = raw Y, +3 = raw Z, a minus sign flips that axis.
// MPU9250: its magnetometer chip sits rotated inside: X and Y are swapped and Z is flipped,
// compared with the arrows printed on the board. {+2, +1, -3} turns it back into the printed arrows.
const int8_t   AXES_A[3]     = { +2, +1, -3 };
// BNO055: the chip's own axes. The turn test tells you whether this needs changing.
const int8_t   AXES_B[3]     = { +1, +2, +3 };

const uint32_t SAMPLE_MS     = 50;       // one reading every 50 ms = 20 per second
const int      TABLE_AVG     = 10;       // table mode: average 10 readings -> one line every 0.5 s
const int      HEADER_EVERY  = 20;       // repeat the column names every 20 table lines
const uint32_t STATS_MS      = 10000;    // 's' noise test length
const bool     BNO_SAFE_READ = true;     // read the BNO055 one byte at a time (more reliable on some ESP32 boards)
const int      BUTTON_PIN    = 0;        // BOOT button: press = next point
const int      LED_PIN       = 2;        // on-board LED of the NodeMCU-32S: on = hold still
const uint32_t HOLD_MS       = 6000;     // how long to hold still at each grid point
const int      SDA_PIN       = 21;
const int      SCL_PIN       = 22;

// ---- WiFi. The name and password live in secrets.h (same folder). Without secrets.h: USB only.
#if __has_include("secrets.h")
#include "secrets.h"
#endif
#ifndef SECRET_WIFI_SSID
#define SECRET_WIFI_SSID     ""
#define SECRET_WIFI_PASSWORD ""
#endif
const char*    WIFI_SSID     = SECRET_WIFI_SSID;      // ' and ’ in the name both work
const char*    WIFI_PASSWORD = SECRET_WIFI_PASSWORD;
const char*    HOSTNAME      = "gradiometer";   // the laptop can use gradiometer.local
const uint16_t TCP_PORT      = 3333;
const uint32_t WIFI_WAIT_MS  = 10000;    // give up on WiFi after this and use USB only
const bool     WIFI_LOW_POWER = true;    // weaker radio = smaller current bursts near the sensors

// =============================================================== globals
MPU9250 mpu;
Adafruit_BNO055 bno28(55, 0x28, &Wire);
Adafruit_BNO055 bno29(55, 0x29, &Wire);
Adafruit_BNO055* bno = nullptr;
uint8_t bnoAddr = 0;

bool haveA = false, haveB = false;
bool csvMode = false;

WiFiServer server(TCP_PORT);
WiFiClient client;
bool wifiOn = false;

// Everything printed goes to USB and, when a laptop is connected over WiFi, to WiFi as well.
class DualOut : public Print {
 public:
  size_t write(uint8_t c) override { return write(&c, 1); }
  size_t write(const uint8_t* buf, size_t n) override {
    Serial.write(buf, n);
    if (wifiOn && client.connected()) client.write(buf, n);
    return n;
  }
};
DualOut out;

float    mpuSum[3] = { 0, 0, 0 };        // MPU readings collected since the last row
int      mpuN = 0;
uint32_t lastSample = 0;
uint32_t bErrors = 0;

int   point = 0;                         // point number, changed by the button or 'n'
bool  buttonWasDown = false;
uint32_t lastPress = 0;
bool     holding = false;                // true while the LED says "hold still"
uint32_t holdUntil = 0;

float tabSum[8];                         // table averaging: Ax Ay Az |A| Bx By Bz |B|
int   tabN[2] = { 0, 0 };                // readings counted for A and for B
int   tabRows = 0;
int   tabLines = 0;

bool     statsOn = false;                // 's' noise test
uint32_t statsStart = 0;
double   stSum[8], stSq[8];
int      stN[2];

const char* CH_NAME[8] = { "A x", "A y", "A z", "|A|", "B x", "B y", "B z", "|B|" };

// =============================================================== small helpers
void prefix() {                          // in CSV mode every non-data line starts with '#'
  if (csvMode) out.print("# ");
}

void dashes(int n) {
  for (int i = 0; i < n; i++) out.print('-');
}

float mag3(const float v[3]) {
  return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// out[i] = raw axis chosen by map[i], flipped if map[i] is negative
void remap(const float raw[3], const int8_t map[3], float out[3]) {
  for (int i = 0; i < 3; i++) {
    int k = abs(map[i]) - 1;
    if (k < 0 || k > 2) k = i;
    out[i] = (map[i] < 0) ? -raw[k] : raw[k];
  }
}

// 8-character column: number with 2 decimals, or "--" when there is no reading
void cell(char* out, float v) {
  if (isnan(v)) snprintf(out, 9, "%8s", "--");
  else          snprintf(out, 9, "%8.2f", v);
}

// =============================================================== BNO055 reading
bool bnoReadReg(uint8_t reg, uint8_t& val) {
  Wire.beginTransmission(bnoAddr);
  Wire.write(reg);
  if (Wire.endTransmission() != 0) return false;
  if (Wire.requestFrom(bnoAddr, (uint8_t)1) != 1) return false;
  val = Wire.read();
  return true;
}

// Magnetometer data: registers 0x0E..0x13 (X, Y, Z, low byte first). 16 counts = 1 uT.
bool readB(float b[3]) {
  float raw[3];
  if (BNO_SAFE_READ) {
    for (int ax = 0; ax < 3; ax++) {
      uint8_t hi1 = 0, lo = 0, hi2 = 0;
      int tries = 0;
      do {                                              // high, low, high again:
        if (!bnoReadReg(0x0F + 2 * ax, hi1)) return false;   // if the high byte changed
        if (!bnoReadReg(0x0E + 2 * ax, lo))  return false;   // in between, a new value
        if (!bnoReadReg(0x0F + 2 * ax, hi2)) return false;   // arrived: read again
      } while (hi1 != hi2 && ++tries < 3);
      raw[ax] = (int16_t)((hi2 << 8) | lo) / 16.0f;
    }
  } else {
    imu::Vector<3> v = bno->getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
    raw[0] = v.x();
    raw[1] = v.y();
    raw[2] = v.z();
  }
  remap(raw, AXES_B, b);
  return true;
}

// =============================================================== printing
void printHelp() {
  prefix(); out.println("Commands (letter + Enter):  t = table   c = CSV   s = noise test (10 s)");
  prefix(); out.println("                            n = next point   r = point to 0   h = help");
  prefix(); out.println("BOOT button = next point; LED on = hold still, LED off = move on.");
}

void printTableHeader() {
  out.println();
  if (bErrors) out.printf("(BNO055 read errors so far: %lu)\n", (unsigned long)bErrors);
  out.printf("%8s |%6s |%8s%8s%8s%8s |%8s%8s%8s%8s\n",
                "time s", "point", "A x", "A y", "A z", "|A|", "B x", "B y", "B z", "|B|");
  dashes(9); out.print('+'); dashes(7); out.print('+'); dashes(33); out.print('+'); dashes(32);
  out.println();
}

void printCsvHeader() {
  out.println("time_ms,point,Ax,Ay,Az,Bx,By,Bz");
}

void printTableLine(uint32_t now) {
  float v[8];
  for (int i = 0; i < 4; i++) v[i] = tabN[0] ? tabSum[i] / tabN[0] : NAN;
  for (int i = 4; i < 8; i++) v[i] = tabN[1] ? tabSum[i] / tabN[1] : NAN;
  char c[8][9];
  for (int i = 0; i < 8; i++) cell(c[i], v[i]);
  out.printf("%8.1f |%6d |%s%s%s%s |%s%s%s%s\n", now / 1000.0f, point,
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
  if (++tabLines % HEADER_EVERY == 0) printTableHeader();
}

void clearTable() {
  for (int i = 0; i < 8; i++) tabSum[i] = 0;
  tabN[0] = tabN[1] = 0;
  tabRows = 0;
}

// =============================================================== noise test ('s')
void startStats() {
  for (int i = 0; i < 8; i++) stSum[i] = stSq[i] = 0;
  stN[0] = stN[1] = 0;
  statsOn = true;
  statsStart = millis();
  prefix(); out.printf("Noise test: keep the stick completely still for %lu s...\n",
                          (unsigned long)(STATS_MS / 1000));
}

void addStats(bool okA, const float a[3], bool okB, const float b[3]) {
  float v[8] = { a[0], a[1], a[2], mag3(a), b[0], b[1], b[2], mag3(b) };
  for (int s = 0; s < 2; s++) {
    if (s == 0 ? !okA : !okB) continue;
    stN[s]++;
    for (int i = 4 * s; i < 4 * s + 4; i++) { stSum[i] += v[i]; stSq[i] += (double)v[i] * v[i]; }
  }
}

void printStats() {
  statsOn = false;
  out.println();
  prefix(); out.printf("Noise test result (%d readings of A, %d of B)\n", stN[0], stN[1]);
  prefix(); out.printf("%-7s|%11s |%13s\n", "axis", "mean uT", "noise uT");
  prefix(); dashes(7); out.print('+'); dashes(12); out.print('+'); dashes(14); out.println();
  for (int i = 0; i < 8; i++) {
    int n = stN[i / 4];
    prefix();
    if (n < 2) { out.printf("%-7s|%11s |%13s\n", CH_NAME[i], "--", "--"); continue; }
    double mean = stSum[i] / n;
    double var = (stSq[i] - n * mean * mean) / (n - 1);
    out.printf("%-7s|%11.2f |%13.3f\n", CH_NAME[i], mean, sqrt(var > 0 ? var : 0));
  }
  prefix(); out.println("noise = how much the reading jitters while nothing moves (standard deviation).");
  if (!csvMode) printTableHeader();
}

// =============================================================== one row of data
void handleRow(uint32_t now, bool okA, const float a[3], bool okB, const float b[3]) {
  if (statsOn) {
    addStats(okA, a, okB, b);
    if (now - statsStart >= STATS_MS) printStats();
  }

  if (csvMode) {
    out.printf("%lu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", (unsigned long)now, point,
                  a[0], a[1], a[2], b[0], b[1], b[2]);           // missing values print as nan
    return;
  }

  if (okA) { for (int i = 0; i < 3; i++) tabSum[i] += a[i];     tabSum[3] += mag3(a); tabN[0]++; }
  if (okB) { for (int i = 0; i < 3; i++) tabSum[4 + i] += b[i]; tabSum[7] += mag3(b); tabN[1]++; }
  if (++tabRows >= TABLE_AVG) {
    if (!statsOn) printTableLine(now);
    clearTable();
  }
}

// =============================================================== commands and button
void nextPoint() {
  if (holding) {                                    // still holding the last point
    prefix(); out.println("(still holding - press ignored)");
    return;
  }
  point++;
  holding = true;
  holdUntil = millis() + HOLD_MS;
  digitalWrite(LED_PIN, HIGH);
  prefix(); out.printf(">>> point %d at %.2f s - hold still (LED on)\n", point, millis() / 1000.0f);
}

void checkHold() {
  if (holding && (int32_t)(millis() - holdUntil) >= 0) {
    holding = false;
    digitalWrite(LED_PIN, LOW);
    prefix(); out.printf(">>> point %d done - move to the next point\n", point);
  }
}

void handleChar(char c) {
    switch (c) {
      case 't':
        csvMode = false; clearTable();
        out.println("\n>>> Table mode");
        printTableHeader();
        break;
      case 'c':
        csvMode = true;
        out.println("# CSV mode. Columns: time in ms, point number, A and B in uT. Type t for the table.");
        printCsvHeader();
        break;
      case 's': startStats(); break;
      case 'n': nextPoint(); break;
      case 'r':
        point = 0;
        holding = false;
        digitalWrite(LED_PIN, LOW);
        prefix(); out.println(">>> point number reset to 0");
        break;
      case 'h': printHelp(); break;
      default: break;                               // Enter, spaces, anything else
    }
}

void handleCommands() {
  while (Serial.available()) handleChar(tolower(Serial.read()));
  while (wifiOn && client.connected() && client.available()) handleChar(tolower(client.read()));
}

// =============================================================== WiFi
// iPhones usually put a curly apostrophe (’) in the hotspot name. Treat ’, ‘ and ' as the same.
String plainName(String s) {
  s.replace("\xE2\x80\x99", "'");
  s.replace("\xE2\x80\x98", "'");
  return s;
}

void startWiFi() {
  if (strlen(WIFI_SSID) == 0) {                     // no WiFi name: radio off, USB only
    WiFi.mode(WIFI_OFF);
    return;
  }
  WiFi.setHostname(HOSTNAME);
  WiFi.mode(WIFI_STA);
  String ssid = WIFI_SSID;                          // look for the network to get its exact name
  bool seen = false;
  int n = WiFi.scanNetworks();
  for (int i = 0; i < n; i++) {
    if (plainName(WiFi.SSID(i)) == plainName(ssid)) { ssid = WiFi.SSID(i); seen = true; break; }
  }
  WiFi.scanDelete();
  if (!seen)
    out.printf("WiFi \"%s\" not seen. Hotspot on? On iPhone turn on Maximize Compatibility (2.4 GHz).\n", WIFI_SSID);
  out.printf("Connecting to WiFi \"%s\"", ssid.c_str());
  WiFi.begin(ssid.c_str(), WIFI_PASSWORD);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < WIFI_WAIT_MS) {
    delay(250);
    out.print('.');
  }
  out.println();
  if (WiFi.status() == WL_CONNECTED) {
    WiFi.setSleep(false);                           // steady current instead of on/off bursts
    if (WIFI_LOW_POWER) WiFi.setTxPower(WIFI_POWER_8_5dBm);
    server.begin();
    MDNS.begin(HOSTNAME);
    wifiOn = true;
  } else {
    WiFi.disconnect(true);                          // give up: switch the radio off
    WiFi.mode(WIFI_OFF);
  }
}

void checkClient() {
  if (!wifiOn) return;
  if (client.connected()) return;
  WiFiClient c = server.accept();
  if (c) {
    client = c;
    client.setNoDelay(true);
    prefix(); out.println("WiFi laptop connected. Type c for CSV, t for the table.");
  }
}

void checkButton() {
  bool down = digitalRead(BUTTON_PIN) == LOW;
  if (down && !buttonWasDown && millis() - lastPress > 300) {
    lastPress = millis();
    nextPoint();
  }
  buttonWasDown = down;
}

// =============================================================== setup
void scanI2C() {
  out.print("I2C devices found:");
  int n = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) { out.printf(" 0x%02X", addr); n++; }
  }
  out.println(n ? "" : " none - check the wiring");
  out.println("Expected: 0x68 (MPU9250) and 0x28 or 0x29 (BNO055).");
}

void setup() {
  Serial.begin(115200);
  delay(500);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  delay(800);                                       // the BNO055 needs ~650 ms after power-up

  out.println("\n=== Gradiometer raw logger: A = MPU9250, B = BNO055, all values raw in uT ===");
  scanI2C();

  if (USE_A) {
    haveA = mpu.setup(MPU_ADDR);
    if (haveA) mpu.ahrs(false);                     // orientation maths not needed
  }
  if (USE_B) {
    if (bno28.begin(OPERATION_MODE_MAGONLY))      { bno = &bno28; bnoAddr = 0x28; }
    else if (bno29.begin(OPERATION_MODE_MAGONLY)) { bno = &bno29; bnoAddr = 0x29; }
    if (bno) {
      delay(100);
      bno->setExtCrystalUse(true);                  // keeps MAGONLY mode
      haveB = true;
    }
  }
  Wire.setClock(100000);                            // keep the bus slow for the BNO055
  startWiFi();

  out.println();
  out.printf("%-7s|%-9s|%-9s|%s\n", "sensor", " chip", " address", " status");
  dashes(7); out.print('+'); dashes(9); out.print('+'); dashes(9); out.print('+'); dashes(28); out.println();
  out.printf("%-7s| %-8s| 0x%02X    | %s\n", "A", "MPU9250", MPU_ADDR,
                !USE_A ? "switched off" : haveA ? "OK" : "NOT FOUND");
  if (haveB)
    out.printf("%-7s| %-8s| 0x%02X    | %s\n", "B", "BNO055", bnoAddr, "OK (MAGONLY mode)");
  else
    out.printf("%-7s| %-8s| %-8s| %s\n", "B", "BNO055", "--", !USE_B ? "switched off" : "NOT FOUND");
  if (wifiOn)
    out.printf("%-7s| connected to \"%s\": laptop uses %s or %s.local, port %u\n", "WiFi",
               WiFi.SSID().c_str(), WiFi.localIP().toString().c_str(), HOSTNAME, TCP_PORT);
  else
    out.printf("%-7s| %s\n", "WiFi", strlen(WIFI_SSID) ? "not connected - USB only" : "off (no WiFi name set) - USB only");
  out.println("Raw values: the two sensors will NOT agree until they are calibrated on the laptop.");
  out.println();

  printHelp();
  clearTable();
  printTableHeader();
  lastSample = millis();
}

// =============================================================== loop
void loop() {
  checkClient();
  handleCommands();
  checkButton();
  checkHold();

  if (haveA && mpu.update()) {                      // the MPU magnetometer makes 100 values/s
    float raw[3] = { mpu.getMagX() / 10.0f,         // library gives milligauss: 10 mG = 1 uT
                     mpu.getMagY() / 10.0f,
                     mpu.getMagZ() / 10.0f };
    float a[3];
    remap(raw, AXES_A, a);
    for (int i = 0; i < 3; i++) mpuSum[i] += a[i];
    mpuN++;
  }

  uint32_t now = millis();
  if (now - lastSample < SAMPLE_MS) return;
  lastSample += SAMPLE_MS;
  if (now - lastSample > 5 * SAMPLE_MS) lastSample = now;   // fell far behind: catch up

  float a[3] = { NAN, NAN, NAN }, b[3] = { NAN, NAN, NAN };
  bool okA = false, okB = false;
  if (haveA && mpuN > 0) {                          // average of the MPU values in these 50 ms
    for (int i = 0; i < 3; i++) a[i] = mpuSum[i] / mpuN;
    okA = true;
  }
  mpuSum[0] = mpuSum[1] = mpuSum[2] = 0;
  mpuN = 0;
  if (haveB) {
    okB = readB(b);
    if (!okB) { bErrors++; b[0] = b[1] = b[2] = NAN; }
  }
  handleRow(now, okA, a, okB, b);
}
