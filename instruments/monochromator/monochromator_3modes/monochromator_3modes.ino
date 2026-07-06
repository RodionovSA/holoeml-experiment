// =====================================================================
// monochromator_3modes.ino
//
// Firmware for the Arduino-driven monochromator stepper.
// Controls a single stepper via AccelStepper (DRIVER mode), homes
// against a photodiode flag, and moves to a requested wavelength on
// command. Designed to be driven over serial by the Python wrapper in
// src/monochromator/mono_class.py.
//
// Wiring:
//   D8  -> stepper DIR
//   D9  -> stepper STEP
//   D3  -> aux LED              (configured as OUTPUT; not currently driven)
//   D5  -> home-flag LED        (configured as OUTPUT; not currently driven)
//   A0  -> aux photodiode       (configured as INPUT; not read by this sketch)
//   A5  -> home-flag photodiode (used for homing; >threshold == blocked)
//
// Serial protocol (9600 baud, '\n' terminated, ASCII):
//
//   in : home
//   out: INFO: homing
//        ADC: <value>
//        OK: homed
//
//   in : mode <0|1|2>           (0=VIS, 1=IR, 2=SWITCH)
//   out: OK: mode <VIS|IR|SWITCH>
//        ERR: invalid_mode
//
//   in : wavelength <float_nm>
//   out: INFO: moving <nm>
//        OK: wavelength <nm>
//        ERR: out_of_range <min_nm> <max_nm>
//        ERR: no_mode               (mode not selected yet)
//        OK: stopped                (after `stop` during a move)
//
//   in : jog <signed_steps>          (relative raw move; no calibration/bounds)
//   in : move <signed_steps>         (absolute raw move to a step from home 0)
//   out: INFO: moving_steps <target>
//        OK: moved <pos>
//        ERR: invalid_move
//        OK: stopped                (after `stop` during a move)
//
//   in : stop                       (interrupts an in-progress move)
//   out: OK: stopped
//
//   in : status
//   out: INFO: mode=<VIS|IR|SWITCH|none> homed=<0|1> pos=<steps>
//
// Calibration is a linear fit per grating range:
//   steps = (wavelength_nm + offset) / slope
// Recalibration requires reflashing.
// =====================================================================

#include <AccelStepper.h>

// ---- Pin assignments ------------------------------------------------
constexpr uint8_t STEPPER_DIR_PIN     = 8;
constexpr uint8_t STEPPER_STEP_PIN    = 9;
constexpr uint8_t AUX_PHOTODIODE_PIN  = A0;
constexpr uint8_t HOME_PHOTODIODE_PIN = A5;
constexpr uint8_t AUX_LED_PIN         = 3;
constexpr uint8_t HOME_LED_PIN        = 5;

// ---- Tuning constants -----------------------------------------------
constexpr long  SERIAL_BAUD          = 9600;
constexpr float STEPPER_MAX_SPEED    = 500.0f;
constexpr float STEPPER_ACCELERATION = 50.0f;
constexpr int   PHOTODIODE_THRESHOLD = 300;     // ADC counts; >threshold == flag blocked
constexpr float HOMING_SEEK_SPEED    = 50.0f;   // steps/sec; slow for edge accuracy
constexpr long  HOMING_BACKOFF_STEPS = 50;      // forward margin past unblocked edge

// ---- Wavelength -> step calibration ---------------------------------
struct GratingCalibration {
  float minNm;
  float maxNm;
  float offset;
  float slope;
};

constexpr GratingCalibration VIS_CAL        = { 350.0f, 1000.0f, 374.0828f, 1.1164f };
constexpr GratingCalibration IR_CAL         = { 587.0f, 2000.0f, 4715.4390f, 0.5099f };
constexpr float              SWITCH_MIN_NM       = 350.0f;
constexpr float              SWITCH_MAX_NM       = 2000.0f;
constexpr float              SWITCH_CROSSOVER_NM = 650.0f;
// SWITCH mode uses a slightly different VIS fit below the crossover and IR_CAL above.
constexpr GratingCalibration SWITCH_VIS_CAL = { SWITCH_MIN_NM, SWITCH_CROSSOVER_NM, 389.2407f, 1.1127f };

// ---- State ----------------------------------------------------------
enum class GratingMode : uint8_t { VIS = 0, IR = 1, SWITCH = 2, NONE = 255 };

AccelStepper stepper(AccelStepper::DRIVER, STEPPER_STEP_PIN, STEPPER_DIR_PIN);
GratingMode  currentMode = GratingMode::NONE;
bool         motorHomed  = false;

// ---- Forward declarations -------------------------------------------
void homeMotor();
void handleModeCommand(const String& input);
void handleWavelengthCommand(const String& input);
void handleManualMoveCommand(const String& input, bool relative);
bool runToTargetWithStop();
void reportStatus();

// =====================================================================
// Setup
// =====================================================================
void setup() {
  Serial.begin(SERIAL_BAUD);

  stepper.setMaxSpeed(STEPPER_MAX_SPEED);
  stepper.setAcceleration(STEPPER_ACCELERATION);

  pinMode(AUX_PHOTODIODE_PIN,  INPUT);
  pinMode(HOME_PHOTODIODE_PIN, INPUT);
  pinMode(AUX_LED_PIN,         OUTPUT);
  pinMode(HOME_LED_PIN,        OUTPUT);

  Serial.println("INFO: monochromator initialized; send `home` to start");
}

// =====================================================================
// Main loop: dispatch one serial command per iteration
// =====================================================================
void loop() {
  if (Serial.available() <= 0) return;

  String input = Serial.readStringUntil('\n');
  input.trim();
  if (input.length() == 0) return;

  if (input == "home") {
    homeMotor();
  } else if (input.startsWith("mode")) {
    handleModeCommand(input);
  } else if (input.startsWith("wavelength")) {
    handleWavelengthCommand(input);
  } else if (input.startsWith("jog")) {
    handleManualMoveCommand(input, /*relative=*/true);
  } else if (input.startsWith("move")) {
    handleManualMoveCommand(input, /*relative=*/false);
  } else if (input == "status") {
    reportStatus();
  } else if (input == "stop") {
    // No move in progress to interrupt; acknowledge anyway so the
    // wrapper has consistent semantics for `stop`.
    Serial.println("OK: stopped");
  } else {
    Serial.print("ERR: unknown_command ");
    Serial.println(input);
  }
}

// =====================================================================
// Helpers
// =====================================================================

const char* modeName(GratingMode m) {
  switch (m) {
    case GratingMode::VIS:    return "VIS";
    case GratingMode::IR:     return "IR";
    case GratingMode::SWITCH: return "SWITCH";
    default:                  return "none";
  }
}

// Drive the stepper at constant slow speed in `direction` (+1 / -1)
// until the home photodiode's blocked state matches `targetBlocked`.
// Uses runSpeed() (no acceleration profile) so edge detection is at a
// known low velocity for repeatability.
void seekHomeEdge(int direction, bool targetBlocked) {
  stepper.setSpeed(direction * HOMING_SEEK_SPEED);
  while (true) {
    bool blocked = analogRead(HOME_PHOTODIODE_PIN) > PHOTODIODE_THRESHOLD;
    if (blocked == targetBlocked) break;
    stepper.runSpeed();
  }
}

// =====================================================================
// Command handlers
// =====================================================================

// Home against the photodiode flag, leaving the stepper at position 0
// just inside the blocked region. If the flag is already blocked at
// entry, first move forward off the flag (plus HOMING_BACKOFF_STEPS)
// so the final approach is always from the unblocked side.
void homeMotor() {
  Serial.println("INFO: homing");

  if (analogRead(HOME_PHOTODIODE_PIN) > PHOTODIODE_THRESHOLD) {
    seekHomeEdge(+1, /*targetBlocked=*/false);
    long startPos = stepper.currentPosition();
    stepper.setSpeed(+HOMING_SEEK_SPEED);
    while (stepper.currentPosition() - startPos < HOMING_BACKOFF_STEPS) {
      stepper.runSpeed();
    }
  }
  seekHomeEdge(-1, /*targetBlocked=*/true);

  Serial.print("ADC: ");
  Serial.println(analogRead(HOME_PHOTODIODE_PIN));
  stepper.setCurrentPosition(0);
  motorHomed = true;
  Serial.println("OK: homed");
}

// `mode <0|1|2>` — set the active grating mode.
void handleModeCommand(const String& input) {
  int sp = input.indexOf(' ');
  if (sp < 0) {
    Serial.println("ERR: invalid_mode");
    return;
  }
  int value = input.substring(sp + 1).toInt();
  switch (value) {
    case 0: currentMode = GratingMode::VIS;    break;
    case 1: currentMode = GratingMode::IR;     break;
    case 2: currentMode = GratingMode::SWITCH; break;
    default:
      Serial.println("ERR: invalid_mode");
      return;
  }
  Serial.print("OK: mode ");
  Serial.println(modeName(currentMode));
}

// `wavelength <nm>` — move the stepper to the wavelength's target step.
// Allows interruption with `stop` mid-move.
void handleWavelengthCommand(const String& input) {
  if (currentMode == GratingMode::NONE) {
    Serial.println("ERR: no_mode");
    return;
  }
  int sp = input.indexOf(' ');
  if (sp < 0) {
    Serial.println("ERR: invalid_wavelength");
    return;
  }
  float wavelength = input.substring(sp + 1).toFloat();

  GratingCalibration cal;
  float boundsMin, boundsMax;
  switch (currentMode) {
    case GratingMode::VIS:
      cal = VIS_CAL;
      boundsMin = VIS_CAL.minNm;
      boundsMax = VIS_CAL.maxNm;
      break;
    case GratingMode::IR:
      cal = IR_CAL;
      boundsMin = IR_CAL.minNm;
      boundsMax = IR_CAL.maxNm;
      break;
    case GratingMode::SWITCH:
      boundsMin = SWITCH_MIN_NM;
      boundsMax = SWITCH_MAX_NM;
      cal = (wavelength < SWITCH_CROSSOVER_NM) ? SWITCH_VIS_CAL : IR_CAL;
      break;
    default:
      Serial.println("ERR: no_mode");
      return;
  }

  if (wavelength < boundsMin || wavelength > boundsMax) {
    Serial.print("ERR: out_of_range ");
    Serial.print(boundsMin);
    Serial.print(' ');
    Serial.println(boundsMax);
    return;
  }

  long targetSteps = (long)((wavelength + cal.offset) / cal.slope);

  Serial.print("INFO: moving ");
  Serial.println(wavelength);

  stepper.moveTo(targetSteps);
  if (runToTargetWithStop()) return;

  Serial.print("OK: wavelength ");
  Serial.println(wavelength);
}

// Run to the currently-set target (via stepper.moveTo/move), allowing
// `stop` to interrupt mid-move. Returns true if interrupted (in which
// case "OK: stopped" has already been printed); false if it ran to
// completion (caller prints its own OK on that path).
bool runToTargetWithStop() {
  while (stepper.distanceToGo() != 0) {
    stepper.run();

    if (Serial.available() > 0) {
      String maybeStop = Serial.readStringUntil('\n');
      maybeStop.trim();
      if (maybeStop == "stop") {
        stepper.stop();
        while (stepper.isRunning()) stepper.run();   // smooth deceleration
        Serial.println("OK: stopped");
        return true;
      }
      // Any other command received mid-move is dropped.
    }
  }
  return false;
}

// `jog <signed_steps>`  — relative raw move by a signed step delta.
// `move <signed_steps>` — absolute raw move to a step position (from home 0).
// No calibration, no wavelength bounds check: the point of this command
// is to reach steps (e.g. the zero diffraction order) outside the
// wavelength-calibrated range. Does not require a grating mode.
void handleManualMoveCommand(const String& input, bool relative) {
  int sp = input.indexOf(' ');
  if (sp < 0) {
    Serial.println("ERR: invalid_move");
    return;
  }
  long steps = input.substring(sp + 1).toInt();   // toInt() handles leading '-'

  if (relative) {
    stepper.move(steps);
  } else {
    stepper.moveTo(steps);
  }
  Serial.print("INFO: moving_steps ");
  Serial.println(stepper.targetPosition());

  if (runToTargetWithStop()) return;

  Serial.print("OK: moved ");
  Serial.println(stepper.currentPosition());
}

void reportStatus() {
  Serial.print("INFO: mode=");
  Serial.print(modeName(currentMode));
  Serial.print(" homed=");
  Serial.print(motorHomed ? 1 : 0);
  Serial.print(" pos=");
  Serial.println(stepper.currentPosition());
}
