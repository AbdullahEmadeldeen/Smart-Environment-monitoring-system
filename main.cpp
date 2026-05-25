// ═══════════════════════════════════════════════════════════════════════════════
//  Smart Adaptive Environment Monitor v3.0
//  Board : ESP32-S3 N16R8 UNO
//
//  ── Sensor Inputs
//  ──────────────────────────────────────────────────────────── GPIO 12 → DHT22
//  DATA
//             • VCC → 3.3V
//             • GND → GND
//             • DATA → GPIO12
//             • 10kΩ pull-up resistor between DATA and 3.3V
//
//  GPIO 4  → MQ-135 Analog Output (AOUT)
//             • VCC → 5V
//             • GND → GND
//             • AOUT → GPIO4
//             • Voltage divider required for ESP32 safety
//               (10kΩ + 10kΩ to reduce 5V → 2.5V max)
//
//  GPIO 8  → LDR Module Digital Output (DO)
//             • VCC → 3.3V
//             • GND → GND
//             • DO  → GPIO8
//             • HIGH = Bright
//             • LOW  = Dark
//
//  ── Actuator Outputs
//  ───────────────────────────────────────────────────────── GPIO 5  → Passive
//  Buzzer HYLD1205A
//             • Controlled using ESP32 LEDC PWM
//
//  GPIO 7  → Relay Module Input
//             • ACTIVE LOW relay
//             • LOW  = Fan ON
//             • HIGH = Fan OFF
//
//  GPIO 9  → SG90 Servo Signal
//             • Orange/Yellow signal wire
//             • Powered externally if possible
//
//  GPIO 18 → RGB LED Green Pin  (330Ω resistor)
//  GPIO 17 → RGB LED Blue Pin   (330Ω resistor)
//  GPIO 20 → RGB LED Red Pin    (330Ω resistor)
//
//  GPIO 2  → ESP32 Onboard Status LED
//
//  ── System States
//  ──────────────────────────────────────────────────────────── NORMAL:
//      • Green LED ON
//      • Fan OFF
//      • Buzzer OFF
//      • Servo 0°
//
//  WARNING:
//      • Blue LED ON
//      • Fan ON continuously
//      • Buzzer blinking
//      • Servo 90°
//
//  DANGEROUS:
//      • Red LED ON
//      • Fan ON continuously
//      • Buzzer continuous
//      • Servo 180°
//
//  ── MQTT Data Flow
//  ─────────────────────────────────────────────────────────── Publish Topic :
//  "esp32/smart_env/sensors" Subscribe Topic : "esp32/smart_env/commands"
//
//  ── Required Libraries
//  ─────────────────────────────────────────────────────── • ESP32Servo    by
//  Kevin Harrington • DHT sensor    by Adafruit • ArduinoJson   by Benoit
//  Blanchon • PubSubClient  by Nick O'Leary
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <ArduinoJson.h>
#include <DHT.h>
#include <ESP32Servo.h> // ⚠ Must use ESP32Servo, NOT standard Servo.h
#include <PubSubClient.h>
#include <WiFi.h>

// ─── WiFi — fill in before flashing ──────────────────────────────────────────
const char *WIFI_SSID = "Etisalat 4G iModem-655D";
const char *WIFI_PASSWORD = "90623777";

// ─── MQTT broker
// ──────────────────────────────────────────────────────────────
const char *MQTT_BROKER = "broker.hivemq.com";
const int MQTT_PORT = 1883;
const char *MQTT_CLIENT_ID = "ESP32_SmartEnv_001";
const char *TOPIC_DATA = "esp32/smart_env/sensors";      // ESP32 → laptop
const char *TOPIC_COMMANDS = "esp32/smart_env/commands"; // laptop AI → ESP32

// ─── Pin definitions
// ──────────────────────────────────────────────────────────
#define DHT_PIN 12
#define DHT_TYPE DHT22
#define MQ135_PIN 4  // Analog input (ADC)
#define LDR_PIN 8    // Digital input (DO from LDR module)
#define BUZZER_PIN 5 // PWM output (LEDC)
#define RELAY_PIN 7  // Active LOW  (LOW = fan ON)
#define SERVO_PIN 9  // SG90 PWM signal
#define RGB_GREEN 18 // RGB Green  (330 Ω in series)
#define RGB_BLUE 17  // RGB Blue   (330 Ω in series)
#define RGB_RED 20   // RGB Red    (330 Ω in series)
#define STATUS_LED 2 // Onboard LED

// ─── LEDC buzzer resolution
// ───────────────────────────────────────────────────
#define BUZZ_BITS 8 // 8-bit resolution

// ─── Gas alarm threshold (ppm)
// ────────────────────────────────────────────────
#define GAS_ALARM_RAW 1000

// ─── Timing constants
// ─────────────────────────────────────────────────────────
const unsigned long PUBLISH_INTERVAL = 2000;   // sensor publish period (ms)
const unsigned long RECONNECT_INTERVAL = 5000; // MQTT retry (ms)
const unsigned long WIFI_RETRY_MS = 10000;     // WiFi watchdog (ms)

// ─── Objects
// ──────────────────────────────────────────────────────────────────
DHT dht(DHT_PIN, DHT_TYPE);
Servo servo;
WiFiClient espClient;
PubSubClient mqtt(espClient);

// ─── Timing vars
// ──────────────────────────────────────────────────────────────
unsigned long lastPublish = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long lastWifiCheck = 0;
unsigned long lastAIMessage = 0;

// ─── Buzzer non-blocking state
// ────────────────────────────────────────────────
bool buzzHigh = false;
uint8_t buzzPattern = 0; // 0=off  1=warning (slow)  2=critical (fast)

// ─── AI command state (updated by MQTT callback)
// ──────────────────────────────
struct AIState {
  float fan_speed = 0.0f;   // 0–100 %
  float alarm_level = 0.0f; // 0–100
  float vent_angle = 0.0f;  // 0–180 °
  float risk_level = 0.0f;  // 0–100
  char rbf_label[16] = "normal";
  char som_state[16] = "normal";
  bool received = false;
} ai;

bool warningBuzzOn = false;
unsigned long lastWarningBuzz = 0;

// ─── Forward declarations
// ─────────────────────────────────────────────────────
void connectWifi();
bool mqttReconnect();
void onMqttMessage(char *, byte *, unsigned int);
void publishSensors();
void applyActuators();
void updateBuzzer();
void setRGB(bool r, bool g, bool b);
void printBanner(float temp, float hum, float gas, bool alarm, int light,
                 unsigned long uptime);

// ═══════════════════════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(200);

  // ── Input pins ────────────────────────────────────────────────────────────
  pinMode(LDR_PIN, INPUT);

  // ── Output pins ───────────────────────────────────────────────────────────
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(RGB_RED, OUTPUT);
  pinMode(RGB_GREEN, OUTPUT);
  pinMode(RGB_BLUE, OUTPUT);
  pinMode(STATUS_LED, OUTPUT);

  // Safe initial states
  digitalWrite(RELAY_PIN, LOW); // Relay OFF  (active LOW)
  digitalWrite(STATUS_LED, LOW);
  setRGB(false, false, false);

  // ── Buzzer via LEDC (ESP32 Arduino core v3.x API) ─────────────────────────
  ledcAttach(BUZZER_PIN, 1000, BUZZ_BITS);
  ledcWriteTone(BUZZER_PIN, 0);
  ledcWrite(BUZZER_PIN, 0); // silent

  // ── Servo ─────────────────────────────────────────────────────────────────
  servo.attach(SERVO_PIN, 500, 2400); // 500–2400 µs pulse range for SG90
  servo.write(0);                     // closed position

  // ── DHT22 ─────────────────────────────────────────────────────────────────
  dht.begin();
  lastAIMessage = millis();

  // ── Banner ────────────────────────────────────────────────────────────────
  Serial.println(F("\n╔══════════════════════════════════════════════╗"));
  Serial.println(F("║  Smart Adaptive Environment Monitor  v3.0    ║"));
  Serial.println(F("║  ESP32-S3 N16R8  |  8 Components  |  AI      ║"));
  Serial.println(F("╠══════════════════════════════════════════════╣"));
  Serial.printf("║  Sensor topic  : %-28s║\n", TOPIC_DATA);
  Serial.printf("║  Command topic : %-28s║\n", TOPIC_COMMANDS);
  Serial.println(F("╚══════════════════════════════════════════════╝\n"));

  // ── WiFi + MQTT ───────────────────────────────────────────────────────────
  connectWifi();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(onMqttMessage);
  mqtt.setBufferSize(512);

  // Startup: flash green = system normal
  setRGB(false, true, false);
  delay(600);
  setRGB(false, false, false);

  Serial.println(F("[✓] Setup complete — waiting for AI commands…\n"));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  LOOP
// ═══════════════════════════════════════════════════════════════════════════════
void loop() {
  unsigned long now = millis();

  // ── WiFi watchdog ─────────────────────────────────────────────────────────
  if (now - lastWifiCheck > WIFI_RETRY_MS) {
    lastWifiCheck = now;
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println(F("[!] WiFi lost — reconnecting..."));
      connectWifi();
    }
  }

  // ── MQTT keep-alive / reconnect ───────────────────────────────────────────
  if (!mqtt.connected()) {
    if (now - lastReconnectAttempt > RECONNECT_INTERVAL) {
      lastReconnectAttempt = now;
      if (mqttReconnect())
        lastReconnectAttempt = 0;
    }
  } else {
    mqtt.loop(); // process incoming command messages
  }

  // ── Non-blocking actuator / buzzer patterns ───────────────────────────────
  updateBuzzer();
  // ── Periodic sensor publish ───────────────────────────────────────────────
  if (now - lastPublish > PUBLISH_INTERVAL && mqtt.connected()) {
    lastPublish = now;
    publishSensors();
  }
  if (millis() - lastAIMessage > 15000 && ai.received) {

    ai.received = false;

    strcpy(ai.som_state, "normal");

    applyActuators();

    Serial.println("[SAFETY] AI timeout -> NORMAL mode");
  }
  yield();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  WiFi helpers
// ═══════════════════════════════════════════════════════════════════════════════
void connectWifi() {
  if (WiFi.status() == WL_CONNECTED)
    return;
  Serial.printf("[WiFi] Connecting to \"%s\" ", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 40) {
    delay(500);
    Serial.print('.');
    tries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[✓] WiFi  IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println(F("\n[!] WiFi failed — will retry"));
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MQTT helpers
// ═══════════════════════════════════════════════════════════════════════════════
bool mqttReconnect() {
  Serial.print(F("[MQTT] Connecting... "));
  if (mqtt.connect(MQTT_CLIENT_ID)) {
    Serial.println(F("connected!"));
    mqtt.subscribe(TOPIC_COMMANDS);
    Serial.printf("[MQTT] Subscribed to commands topic: %s\n", TOPIC_COMMANDS);
    return true;
  }
  Serial.printf("failed (rc=%d)\n", mqtt.state());
  return false;
}

// ─── Incoming AI command handler
// ──────────────────────────────────────────────
void onMqttMessage(char *topic, byte *payload, unsigned int len) {
  if (strcmp(topic, TOPIC_COMMANDS) != 0)
    return;

  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, payload, len);
  if (err) {
    Serial.printf("[!] JSON parse error: %s\n", err.c_str());
    return;
  }

  // Update AI state
  ai.fan_speed = doc["fan_speed"] | 0.0f;
  ai.alarm_level = doc["alarm_level"] | 0.0f;
  ai.vent_angle = doc["vent_angle"] | 0.0f;
  ai.risk_level = doc["risk_level"] | 0.0f;
  strlcpy(ai.rbf_label, doc["rbf_label"] | "normal", sizeof(ai.rbf_label));
  if (doc.containsKey("som_state")) {
    strlcpy(ai.som_state, doc["som_state"], sizeof(ai.som_state));
  } else {
    strlcpy(ai.som_state, ai.rbf_label, sizeof(ai.som_state));
  }
  for (size_t i = 0; i < sizeof(ai.som_state); ++i) {
    ai.som_state[i] = tolower(ai.som_state[i]);
    if (ai.som_state[i] == '\0')
      break;
  }
  ai.received = true;
  lastAIMessage = millis();

  Serial.println(F("\n══════════ AI Commands Received ══════════"));
  Serial.printf("  RBF label : %s\n", ai.rbf_label);
  Serial.printf("  SOM state : %s\n", ai.som_state);
  Serial.printf("  Risk      : %.1f/100\n", ai.risk_level);
  Serial.printf("  Fan speed : %.1f %%\n", ai.fan_speed);
  Serial.printf("  Alarm     : %.1f\n", ai.alarm_level);
  Serial.printf("  Vent angle: %.1f °\n", ai.vent_angle);

  // Apply all actuators immediately upon receiving commands
  applyActuators();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Actuator control
//  All outputs driven by ai.som_state  (normal / warning / dangerous)
// ═══════════════════════════════════════════════════════════════════════════════
void applyActuators() {
  bool isNormal = (strcmp(ai.som_state, "normal") == 0);
  bool isWarning = (strcmp(ai.som_state, "warning") == 0);
  bool isDangerous = (strcmp(ai.som_state, "dangerous") == 0);

  int angle = isDangerous ? 180 : (isWarning ? 90 : 0);
  ai.vent_angle = angle;

  if (isDangerous) {
    // ── DANGEROUS: solid red LED, fan 100%, buzzer continuous ──────────
    ai.fan_speed = 100.0f;
    ai.alarm_level = 100.0f;
    ai.risk_level = 100.0f;
    setRGB(true, false, false);    // solid red
    digitalWrite(RELAY_PIN, HIGH); // fan ON full power
    buzzPattern = 2;               // continuous alarm tone
  } else if (isWarning) {

    ai.fan_speed = 100.0f;
    ai.alarm_level = 50.0f;
    ai.risk_level = 60.0f;

    // Blue LED
    setRGB(false, false, true);

    // Fan ON continuously
    digitalWrite(RELAY_PIN, HIGH);

    // Blinking buzzer
    buzzPattern = 1;

    warningBuzzOn = false;
    buzzHigh = false;
    lastWarningBuzz = millis();

  } else {
    // ── NORMAL: green LED, fan OFF, buzzer silent, servo 0° ───────────
    ai.fan_speed = 0.0f;
    ai.alarm_level = 0.0f;
    ai.risk_level = 10.0f;
    setRGB(false, true, false);   // green LED
    digitalWrite(RELAY_PIN, LOW); // fan OFF
    buzzPattern = 0;
    warningBuzzOn = false;
    buzzHigh = false;
  }

  servo.write(angle);

  const char *stateName =
      isDangerous ? "DANGEROUS" : (isWarning ? "WARNING" : "NORMAL");
  const char *fanStatus = (isDangerous || isWarning) ? "ON" : "OFF";
  const char *buzzText =
      isDangerous ? "CONTINUOUS" : (isWarning ? "BLINKING" : "SILENT");

  Serial.println(F("─────────── Actuators Applied ────────────"));
  Serial.printf("  SOM state : %s\n", stateName);
  Serial.printf("  RGB LED   : %s\n",
                isDangerous ? "RED (solid)" : (isWarning ? "BLUE" : "GREEN"));
  Serial.printf("  Servo     : %d °\n", angle);
  Serial.printf("  Fan power : %s\n", fanStatus);
  Serial.printf("  Buzzer    : %s\n\n", buzzText);
}

// ─── Non-blocking buzzer state machine ───────────────────────────────────────
//  Called every loop() iteration — never blocks
void updateBuzzer() {
  unsigned long now = millis();

  if (buzzPattern == 0) {

    ledcWriteTone(BUZZER_PIN, 0);
    ledcWrite(BUZZER_PIN, 0);

    buzzHigh = false;
    warningBuzzOn = false;

    return;
  }

  if (buzzPattern == 2) {
    if (!buzzHigh) {
      ledcWriteTone(BUZZER_PIN, 2500);
      ledcWrite(BUZZER_PIN, 180);
      buzzHigh = true;
    }
    return;
  }

  if (buzzPattern == 1) {

    if (now - lastWarningBuzz >= 500UL) {

      lastWarningBuzz = now;

      warningBuzzOn = !warningBuzzOn;

      if (warningBuzzOn) {

        ledcWriteTone(BUZZER_PIN, 1500);
        ledcWrite(BUZZER_PIN, 180);

      } else {

        ledcWriteTone(BUZZER_PIN, 0);
        ledcWrite(BUZZER_PIN, 0);
      }
    }

    return;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Sensor reading & MQTT publish
// ═══════════════════════════════════════════════════════════════════════════════
void publishSensors() {
  // ── DHT22 ─────────────────────────────────────────────────────────────────
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println(F("[!] DHT22 read failed — skipping cycle"));
    return;
  }

  // ── MQ-135 analog → gas ppm ───────────────────────────────────────────────
  //  Map ADC (0–4095) → extended range (50–1000 ppm) for higher
  //  gas sensitivity — small ADC changes produce larger ppm swings.
  int mq135Raw = analogRead(MQ135_PIN);
  float gas_level = mq135Raw;
  bool gasAlarm = (gas_level >= GAS_ALARM_RAW);

  // ── LDR digital output ────────────────────────────────────────────────────
  int light = !digitalRead(LDR_PIN); // 1 = lit, 0 = dark

  // ── Print to Serial monitor
  printBanner(temperature, humidity, gas_level, gasAlarm, light, millis());

  // ── Build JSON payload ────────────────────────────────────────────────────
  StaticJsonDocument<256> doc;
  doc["temperature"] = round(temperature * 100.0f) / 100.0f;
  doc["humidity"] = round(humidity * 100.0f) / 100.0f;
  doc["gas_level"] = gas_level;
  doc["gas_alarm"] = gasAlarm;
  doc["light"] = light;
  doc["timestamp"] = millis();

  char json[320];
  serializeJson(doc, json, sizeof(json));

  if (mqtt.publish(TOPIC_DATA, json)) {
    // Quick status LED blink = publish OK
    digitalWrite(STATUS_LED, HIGH);
    delay(40);
    digitalWrite(STATUS_LED, LOW);
  } else {
    Serial.println(F("[!] MQTT publish failed — check connection"));
  }
}

// ─── RGB LED helper (common cathode) ─────────────────────────────────────────
void setRGB(bool r, bool g, bool b) {
  digitalWrite(RGB_RED, r ? HIGH : LOW);
  digitalWrite(RGB_GREEN, g ? HIGH : LOW);
  digitalWrite(RGB_BLUE, b ? HIGH : LOW);
}

// ─── Serial status banner
// ─────────────────────────────────────────────────────
void printBanner(float temp, float hum, float gas, bool alarm, int light,
                 unsigned long uptime) {
  const char *stateStr = (strcmp(ai.som_state, "dangerous") == 0) ? "DANGEROUS"
                         : (strcmp(ai.som_state, "warning") == 0) ? "WARNING"
                                                                  : "NORMAL";

  Serial.println(F("╔══════════════════════════════════════════════╗"));
  Serial.println(F("║    Smart Environment Monitor — Sensor Read   ║"));
  Serial.println(F("╠══════════════════════════════════════════════╣"));
  Serial.printf("║  Temperature  : %6.2f °C                   ║\n", temp);
  Serial.printf("║  Humidity     : %6.2f %%                    ║\n", hum);
  Serial.printf("║  Gas (MQ135)  : %6.0f LEVEL  %-9s      ║\n", gas,
                alarm ? "⚠ ALARM" : "OK");
  Serial.printf("║  Light        : %-3s                         ║\n",
                light ? "LIT" : "DARK");
  Serial.println(F("╠══════════════════════════════════════════════╣"));
  Serial.printf("║  AI State     : %-10s  Risk: %5.1f/100  ║\n", stateStr,
                ai.risk_level);
  Serial.printf("║  Fan/Relay    : %-4s   Servo: %3d°          ║\n",
                (strcmp(ai.som_state, "normal") == 0) ? "OFF" : "ON",
                (int)constrain(ai.vent_angle, 0, 180));
  Serial.println(F("╚══════════════════════════════════════════════╝"));
  Serial.printf("  ✈  Published → %s\n\n", TOPIC_DATA);
}