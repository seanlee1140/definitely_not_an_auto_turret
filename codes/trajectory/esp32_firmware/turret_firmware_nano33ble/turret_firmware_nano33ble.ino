/*
 * turret_firmware_nano33ble.ino
 * ==============================
 * Same firmware as turret_firmware.ino but for Arduino Nano 33 BLE.
 *
 * Changes from ESP32 version:
 *   - #include <Servo.h>  instead of <ESP32Servo.h>  (built-in, no install)
 *   - Different pin numbers (see below)
 *   - Everything else identical — same serial protocol, same Python scripts
 *
 * Serial protocol (115200 baud, newline-terminated):
 *   PAN_TO <angle_deg>   — absolute angle from home
 *   PAN <steps>          — relative half-steps (+CW / -CCW)
 *   TILT <angle_deg>     — servo 0–180
 *   SPEED <us_per_step>  — stepper delay (default 1500 µs)
 *   THROTTLE <us>        — ESC throttle 1000 (stop) to 2000 (full)
 *   FIRE                 — spin up ESC to fire speed, trigger relay, spin down
 *   HOME                 — return to 0° pan, 90° tilt, ESC to idle
 *   STATUS               — report positions + throttle
 *   STOP                 — de-energise stepper coils
 *
 * Wiring (ULN2003A chip):
 *   1B (pin 1)  ← D2
 *   2B (pin 2)  ← D3
 *   3B (pin 3)  ← D4
 *   4B (pin 4)  ← D5
 *   E  (pin 8)  ← GND
 *   COM(pin 9)  ← 24V supply
 *   Servo signal → D6
 *   Fire relay   → D7  (active HIGH, set -1 to disable)
 *
 * Stepper: Sanyo 1.8°/step unipolar (5-wire)
 *   Half-step mode: 400 steps = 360°
 *
 * ⚠ Nano 33 BLE is 3.3V logic — ULN2003 inputs are fine at 3.3V.
 *   Do NOT connect 5V signals to Nano 33 BLE pins (not 5V tolerant).
 */

#include <Servo.h>

// ── Pin assignments ────────────────────────────────────────────────────────
const int STEP_IN1  = 5;
const int STEP_IN2  = 4;
const int STEP_IN3  = 3;
const int STEP_IN4  = 2;
const int SERVO_PIN = 6;
const int FIRE_PIN  = 7;   // relay/solenoid — set -1 to disable
const int ESC_PIN   = 12;  // ESC signal wire (white/yellow)

// ── Stepper config ─────────────────────────────────────────────────────────
// Sanyo 1.8°/step, half-step mode: 400 steps = 360°
const int   STEPS_PER_REV = 400;
const float STEPS_PER_DEG = (float)STEPS_PER_REV / 360.0f;

// Half-step sequence (8 phases)
const uint8_t HALF_STEP[8][4] = {
  {1, 0, 0, 0},
  {1, 1, 0, 0},
  {0, 1, 0, 0},
  {0, 1, 1, 0},
  {0, 0, 1, 0},
  {0, 0, 1, 1},
  {0, 0, 0, 1},
  {1, 0, 0, 1}
};

int  stepIndex    = 0;
long currentSteps = 0;
int  stepDelayUs  = 1500;

// ── Servo ──────────────────────────────────────────────────────────────────
Servo  tiltServo;
float  currentTiltDeg = 90.0f;

// ── ESC ────────────────────────────────────────────────────────────────────
Servo escMotor;
const int ESC_MIN_US   = 1000;  // zero throttle / stop
const int ESC_MAX_US   = 2000;  // full throttle
const int ESC_FIRE_US  = 1600;  // throttle used during FIRE command — tune this
int       currentEscUs = ESC_MIN_US;

// ── Serial buffer ──────────────────────────────────────────────────────────
String cmdBuf = "";

// ═══════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);

  pinMode(STEP_IN1, OUTPUT);
  pinMode(STEP_IN2, OUTPUT);
  pinMode(STEP_IN3, OUTPUT);
  pinMode(STEP_IN4, OUTPUT);
  stepperOff();

  tiltServo.attach(SERVO_PIN);
  tiltServo.write((int)currentTiltDeg);
  delay(300);

  if (FIRE_PIN >= 0) {
    pinMode(FIRE_PIN, OUTPUT);
    digitalWrite(FIRE_PIN, LOW);
  }

  // ── ESC arming sequence ──────────────────────────────────────────────────
  // Must send 1000 µs (zero throttle) and wait before the ESC accepts commands
  escMotor.attach(ESC_PIN, ESC_MIN_US, ESC_MAX_US);
  escMotor.writeMicroseconds(ESC_MIN_US);
  delay(3000);  // ESC beeps here — wait it out

  Serial.println("TURRET_READY");
}

// ═══════════════════════════════════════════════════════════════════════════
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      processCommand(cmdBuf);
      cmdBuf = "";
    } else if (c != '\r') {
      cmdBuf += c;
      if (cmdBuf.length() > 64) cmdBuf = "";
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd.startsWith("PAN_TO ")) {
    float targetDeg   = cmd.substring(7).toFloat();
    long  targetSteps = (long)(targetDeg * STEPS_PER_DEG);
    long  delta       = targetSteps - currentSteps;
    moveStepper(delta);
    Serial.print("PAN_DONE ");
    Serial.println(currentSteps);

  } else if (cmd.startsWith("PAN ")) {
    long steps = cmd.substring(4).toInt();
    moveStepper(steps);
    Serial.print("PAN_DONE ");
    Serial.println(currentSteps);

  } else if (cmd.startsWith("TILT ")) {
    float angle = cmd.substring(5).toFloat();
    angle = constrain(angle, 0.0f, 180.0f);
    currentTiltDeg = angle;
    tiltServo.write((int)angle);
    delay(20);
    Serial.println("TILT_DONE");

  } else if (cmd.startsWith("SPEED ")) {
    int spd = cmd.substring(6).toInt();
    stepDelayUs = constrain(spd, 300, 15000);
    Serial.println("OK");

  } else if (cmd.startsWith("THROTTLE ")) {
    int us = cmd.substring(9).toInt();
    us = constrain(us, ESC_MIN_US, ESC_MAX_US);
    currentEscUs = us;
    escMotor.writeMicroseconds(us);
    Serial.print("THROTTLE_OK ");
    Serial.println(us);

  } else if (cmd == "FIRE") {
    // Spin up ESC → trigger relay → spin down
    escMotor.writeMicroseconds(ESC_FIRE_US);
    currentEscUs = ESC_FIRE_US;
    delay(600);  // let motor reach speed
    if (FIRE_PIN >= 0) {
      digitalWrite(FIRE_PIN, HIGH);
      delay(100);
      digitalWrite(FIRE_PIN, LOW);
    }
    delay(200);
    escMotor.writeMicroseconds(ESC_MIN_US);
    currentEscUs = ESC_MIN_US;
    Serial.println("FIRED");

  } else if (cmd == "HOME") {
    moveStepper(-currentSteps);
    currentTiltDeg = 90.0f;
    tiltServo.write(90);
    escMotor.writeMicroseconds(ESC_MIN_US);
    currentEscUs = ESC_MIN_US;
    delay(300);
    Serial.println("HOME_DONE");

  } else if (cmd == "STATUS") {
    float panDeg = (float)currentSteps / STEPS_PER_DEG;
    Serial.print("STATUS pan_steps=");
    Serial.print(currentSteps);
    Serial.print(" pan_deg=");
    Serial.print(panDeg, 2);
    Serial.print(" tilt_deg=");
    Serial.print(currentTiltDeg, 1);
    Serial.print(" esc_us=");
    Serial.println(currentEscUs);

  } else if (cmd == "STOP") {
    stepperOff();
    Serial.println("OK");

  } else {
    Serial.print("ERROR unknown: ");
    Serial.println(cmd);
  }
}

// ── Stepper helpers ─────────────────────────────────────────────────────────
void moveStepper(long steps) {
  if (steps == 0) return;
  int  dir      = (steps > 0) ? 1 : -1;
  long absSteps = abs(steps);

  for (long i = 0; i < absSteps; i++) {
    stepIndex = (stepIndex + dir + 8) % 8;
    applyStep(stepIndex);
    delayMicroseconds(stepDelayUs);
  }
  currentSteps += steps;
  stepperOff();
}

void applyStep(int idx) {
  digitalWrite(STEP_IN1, HALF_STEP[idx][0]);
  digitalWrite(STEP_IN2, HALF_STEP[idx][1]);
  digitalWrite(STEP_IN3, HALF_STEP[idx][2]);
  digitalWrite(STEP_IN4, HALF_STEP[idx][3]);
}

void stepperOff() {
  digitalWrite(STEP_IN1, LOW);
  digitalWrite(STEP_IN2, LOW);
  digitalWrite(STEP_IN3, LOW);
  digitalWrite(STEP_IN4, LOW);
}
