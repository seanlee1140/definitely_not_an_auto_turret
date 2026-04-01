/*
 * turret_firmware.ino
 * ====================
 * ESP32 firmware for 2-axis turret:
 *   Pan  (Z-axis) : Unipolar stepper via ULN2003 driver
 *   Tilt          : Standard servo, 0–180°
 *
 * Serial protocol (115200 baud, newline-terminated commands):
 *
 *   PAN_TO <angle_deg>   — rotate stepper to absolute angle from home
 *   PAN <steps>          — relative move by N half-steps (+CW / -CCW)
 *   TILT <angle_deg>     — set servo angle 0–180
 *   SPEED <us_per_step>  — set stepper delay (default 1500 µs/step)
 *   FIRE                 — trigger firing mechanism (wire up to your relay/solenoid)
 *   HOME                 — return stepper to 0° and servo to 90°
 *   STATUS               — report current positions
 *   STOP                 — de-energise stepper coils
 *
 * Wiring (ULN2003 breakout → ESP32):
 *   IN1 → GPIO 19
 *   IN2 → GPIO 18
 *   IN3 → GPIO 5
 *   IN4 → GPIO 17
 *   Servo signal → GPIO 4
 *   Fire relay/solenoid → GPIO 2  (active HIGH)
 *
 * Stepper: 28BYJ-48 (or any unipolar 4-phase stepper)
 *   Half-step mode: 2048 steps per revolution
 *   Full step:      1024 steps per revolution (change STEPS_PER_REV below)
 *
 * Dependencies:
 *   ESP32Servo library  (install via Arduino Library Manager)
 */

#include <ESP32Servo.h>

// ── Pin assignments ────────────────────────────────────────────────────────
const int STEP_IN1  = 19;
const int STEP_IN2  = 18;
const int STEP_IN3  = 5;
const int STEP_IN4  = 17;
const int SERVO_PIN = 4;
const int FIRE_PIN  = 2;   // set -1 to disable

// ── Stepper config ─────────────────────────────────────────────────────────
// 28BYJ-48 in half-step mode: 2048 steps = 360°
const int   STEPS_PER_REV  = 2048;
const float STEPS_PER_DEG  = (float)STEPS_PER_REV / 360.0f;

// Half-step sequence (8 phases, smoother than full-step)
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

int  stepIndex   = 0;
long currentSteps = 0;        // cumulative half-step count from home
int  stepDelayUs  = 1500;     // µs per half-step (lower = faster, min ~500)

// ── Servo ──────────────────────────────────────────────────────────────────
Servo  tiltServo;
float  currentTiltDeg = 90.0f;

// ── Serial command buffer ──────────────────────────────────────────────────
String cmdBuf = "";

// ═══════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);

  // Stepper
  pinMode(STEP_IN1, OUTPUT);
  pinMode(STEP_IN2, OUTPUT);
  pinMode(STEP_IN3, OUTPUT);
  pinMode(STEP_IN4, OUTPUT);
  stepperOff();   // de-energise — don't hold current at boot

  // Servo (500–2500 µs pulse range covers full 0–180° on most servos)
  tiltServo.attach(SERVO_PIN, 500, 2500);
  tiltServo.write((int)currentTiltDeg);
  delay(300);

  // Fire pin
  if (FIRE_PIN >= 0) {
    pinMode(FIRE_PIN, OUTPUT);
    digitalWrite(FIRE_PIN, LOW);
  }

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
      if (cmdBuf.length() > 64) cmdBuf = "";  // guard against garbage
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  // ── PAN_TO <angle_deg> ────────────────────────────────────────────────
  if (cmd.startsWith("PAN_TO ")) {
    float targetDeg   = cmd.substring(7).toFloat();
    long  targetSteps = (long)(targetDeg * STEPS_PER_DEG);
    long  delta       = targetSteps - currentSteps;
    moveStepper(delta);
    Serial.print("PAN_DONE ");
    Serial.println(currentSteps);

  // ── PAN <steps> ───────────────────────────────────────────────────────
  } else if (cmd.startsWith("PAN ")) {
    long steps = cmd.substring(4).toInt();
    moveStepper(steps);
    Serial.print("PAN_DONE ");
    Serial.println(currentSteps);

  // ── TILT <angle_deg> ─────────────────────────────────────────────────
  } else if (cmd.startsWith("TILT ")) {
    float angle = cmd.substring(5).toFloat();
    angle = constrain(angle, 0.0f, 180.0f);
    currentTiltDeg = angle;
    tiltServo.write((int)angle);
    delay(20);   // give servo time to start moving
    Serial.println("TILT_DONE");

  // ── SPEED <us_per_step> ───────────────────────────────────────────────
  } else if (cmd.startsWith("SPEED ")) {
    int spd    = cmd.substring(6).toInt();
    stepDelayUs = constrain(spd, 300, 15000);
    Serial.println("OK");

  // ── FIRE ──────────────────────────────────────────────────────────────
  } else if (cmd == "FIRE") {
    if (FIRE_PIN >= 0) {
      digitalWrite(FIRE_PIN, HIGH);
      delay(100);
      digitalWrite(FIRE_PIN, LOW);
    }
    Serial.println("FIRED");

  // ── HOME ──────────────────────────────────────────────────────────────
  } else if (cmd == "HOME") {
    moveStepper(-currentSteps);   // back to step 0
    currentTiltDeg = 90.0f;
    tiltServo.write(90);
    delay(300);
    Serial.println("HOME_DONE");

  // ── STATUS ────────────────────────────────────────────────────────────
  } else if (cmd == "STATUS") {
    float panDeg = (float)currentSteps / STEPS_PER_DEG;
    Serial.print("STATUS pan_steps=");
    Serial.print(currentSteps);
    Serial.print(" pan_deg=");
    Serial.print(panDeg, 2);
    Serial.print(" tilt_deg=");
    Serial.println(currentTiltDeg, 1);

  // ── STOP ──────────────────────────────────────────────────────────────
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
  stepperOff();   // de-energise coils to prevent heating
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
