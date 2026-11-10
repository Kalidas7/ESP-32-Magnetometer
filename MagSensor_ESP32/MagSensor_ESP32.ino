/*
  MagSensor_ESP32.ino
  ------------------------------------------------------------
  NodeMCU-32S (ESP-32S module) + GY-BNO055
  Reads the BNO055 magnetometer and streams it over WiFi (UDP)
  to your laptop, where mag_logger.py saves it to CSV files.

  WIRING (I2C)
    GY-BNO055 VIN  -> ESP32 3V3
    GY-BNO055 GND  -> ESP32 GND
    GY-BNO055 SDA  -> ESP32 GPIO21  (pin labelled D21 / 21)
    GY-BNO055 SCL  -> ESP32 GPIO22  (pin labelled D22 / 22)
    (leave PS0, PS1, ADR, INT, RST unconnected)

  ARDUINO IDE SETUP
    1. File > Preferences > Additional boards manager URLs:
       https://espressif.github.io/arduino-esp32/package_esp32_index.json
    2. Tools > Board > Boards Manager > install "esp32" by Espressif
    3. Tools > Board > esp32 > "NodeMCU-32S"  (or "ESP32 Dev Module")
    4. Tools > Manage Libraries > install "Adafruit BNO055"
       (accept "Adafruit Unified Sensor" and "Adafruit BusIO" when asked)

  Packet format sent to the laptop (one line per sample):
    MAG,seq,esp_ms,mx,my,mz,heading,roll,pitch,cal_sys,cal_gyr,cal_acc,cal_mag
    mx,my,mz are in microtesla (uT).
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

// ======================= USER SETTINGS =======================

// 0 = join your existing WiFi (home router or phone hotspot)
// 1 = ESP32 creates its own WiFi network "MagSensor"; connect the laptop to it
//     (useful outdoors / in a lab with no router)
#define USE_AP_MODE 0

// Used when USE_AP_MODE = 0  (must be a 2.4 GHz network, ESP32 can't do 5 GHz)
// Used when USE_AP_MODE = 0  (must be a 2.4 GHz network, ESP32 can't do 5 GHz)
const char* WIFI_SSID     = "Kalidas’s iPhone";
const char* WIFI_PASSWORD = "hello1234";

// Used when USE_AP_MODE = 1
const char* AP_SSID     = "MagSensor";
const char* AP_PASSWORD = "magsensor123";      // at least 8 characters

// Where to send data. "" = broadcast to every device on the network (easiest).
// If your network blocks broadcast, put your laptop's IP here, e.g. "192.168.1.42".
const char* LAPTOP_IP = "";
const uint16_t UDP_PORT = 4210;                // must match mag_logger.py --port

// OPERATION_MODE_NDOF    : full fusion. Magnetometer is auto-calibrated, you also
//                          get heading/roll/pitch. Good default.
// OPERATION_MODE_MAGONLY : magnetometer only, no auto-recalibration. Use this if
//                          your values drift when you work close to strong magnets.
#define SENSOR_MODE OPERATION_MODE_NDOF

const uint16_t SAMPLE_RATE_HZ = 20;            // BNO055 mag updates at ~20 Hz
const bool USE_EXT_CRYSTAL = true;             // set false if readings look frozen/odd
const int  I2C_SDA = 21;
const int  I2C_SCL = 22;
const int  LED_PIN = 2;                        // on-board blue LED on NodeMCU-32S
const char* OTA_HOSTNAME = "magsensor";        // for wireless code upload
// =============================================================

Adafruit_BNO055 bno28(55, 0x28, &Wire);        // GY-BNO055 boards use 0x28 or 0x29,
Adafruit_BNO055 bno29(55, 0x29, &Wire);        // we try both automatically
Adafruit_BNO055* bno = nullptr;

WiFiUDP udp;
IPAddress destIP;
uint32_t seq = 0;

IPAddress broadcastFor(IPAddress ip, IPAddress mask) {
  IPAddress b;
  for (int i = 0; i < 4; i++) b[i] = ip[i] | (~mask[i] & 0xFF);
  return b;
}

void blink(int times, int ms) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH); delay(ms);
    digitalWrite(LED_PIN, LOW);  delay(ms);
  }
}

void startSensor() {
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);                       // 100 kHz is most reliable with BNO055

  Serial.println("Looking for BNO055...");
  if (bno28.begin(SENSOR_MODE)) {
    bno = &bno28; Serial.println("BNO055 found at 0x28");
  } else if (bno29.begin(SENSOR_MODE)) {
    bno = &bno29; Serial.println("BNO055 found at 0x29");
  } else {
    Serial.println("ERROR: BNO055 not found. Check wiring: SDA->21, SCL->22, VIN->3V3, GND->GND");
    while (true) blink(1, 100);                // fast blink forever = sensor problem
  }
  delay(100);
  bno->setExtCrystalUse(USE_EXT_CRYSTAL);
}

void startWiFi() {
#if USE_AP_MODE
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  delay(200);
  Serial.printf("WiFi network created: \"%s\"  password: \"%s\"\n", AP_SSID, AP_PASSWORD);
  Serial.print("ESP32 IP: "); Serial.println(WiFi.softAPIP());
  destIP = IPAddress(192, 168, 4, 255);        // AP subnet broadcast
#else
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(OTA_HOSTNAME);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to \"%s\"", WIFI_SSID);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    blink(1, 250);
    Serial.print(".");
    if (millis() - t0 > 30000) {
      Serial.println("\nCould not connect (wrong name/password? 5 GHz network?). Restarting...");
      ESP.restart();
    }
  }
  Serial.print("\nConnected. ESP32 IP: "); Serial.println(WiFi.localIP());
  destIP = broadcastFor(WiFi.localIP(), WiFi.subnetMask());
#endif
  WiFi.setSleep(false);                        // fewer dropped packets

  if (strlen(LAPTOP_IP) > 0) destIP.fromString(LAPTOP_IP);
  Serial.print("Sending data to "); Serial.print(destIP);
  Serial.printf(" port %u\n", UDP_PORT);
  udp.begin(UDP_PORT);
  blink(3, 80);                                // 3 quick blinks = WiFi OK
}

void startOTA() {
  ArduinoOTA.setHostname(OTA_HOSTNAME);
  ArduinoOTA.onStart([]() { Serial.println("OTA update starting..."); });
  ArduinoOTA.onEnd([]()   { Serial.println("\nOTA done, rebooting"); });
  ArduinoOTA.onError([](ota_error_t e) { Serial.printf("OTA error %u\n", e); });
  ArduinoOTA.begin();
  Serial.printf("Wireless upload ready: Tools > Port > \"%s\" (network port)\n", OTA_HOSTNAME);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  delay(500);
  Serial.println("\n=== MagSensor ESP32 + BNO055 ===");
  startSensor();
  startWiFi();
  startOTA();
}

void loop() {
  ArduinoOTA.handle();

  static uint32_t lastSample = 0;
  uint32_t now = millis();
  if (now - lastSample < 1000UL / SAMPLE_RATE_HZ) return;
  lastSample = now;

  imu::Vector<3> m = bno->getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER); // uT
  imu::Vector<3> e = bno->getVector(Adafruit_BNO055::VECTOR_EULER);        // degrees
  uint8_t cSys = 0, cGyr = 0, cAcc = 0, cMag = 0;
  bno->getCalibration(&cSys, &cGyr, &cAcc, &cMag);

  char buf[180];
  int n = snprintf(buf, sizeof(buf),
                   "MAG,%lu,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%u,%u,%u,%u",
                   (unsigned long)seq++, (unsigned long)now,
                   m.x(), m.y(), m.z(), e.x(), e.y(), e.z(),
                   cSys, cGyr, cAcc, cMag);

  udp.beginPacket(destIP, UDP_PORT);
  udp.write((const uint8_t*)buf, n);
  udp.endPacket();

  Serial.println(buf);                         // also visible in Serial Monitor (USB)

  if (seq % 20 == 0) digitalWrite(LED_PIN, !digitalRead(LED_PIN)); // slow heartbeat
}
