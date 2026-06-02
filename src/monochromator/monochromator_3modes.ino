///////////////////////////////////////////////////////
//Library variable definitions 
///////////////////////////////////////////////////////

#include <AccelStepper.h>

const int dirPin = 8;
const int stepPin = 9;
const int photodiodeS1Pin = A0;
const int photodiodeS2Pin = A5;
const int ledS1Pin = 3;
const int ledS2Pin = 5;

int currentStep = 0; // Track the current step position

#define motorInterfaceType 1

// Creates an instance
AccelStepper myStepper(motorInterfaceType, stepPin, dirPin);

bool emergencyStop = false; // Variable to track emergency stop state
int gratingMode = 0; // 0: VIS, 1: IR, 2: Both with switch at 650 nm

///////////////////////////////////////////////////////
//Setup. 
//This is executed when the Arduino is initialized with the initialize_arduino() method.
///////////////////////////////////////////////////////

void setup() {
  // Initialize Serial Communication --> 9600 bits are transmitted each second
  Serial.begin(9600);

  // Stepper motor setup
  myStepper.setMaxSpeed(500);
  myStepper.setAcceleration(50);
      
  // Set pin modes
  pinMode(dirPin, OUTPUT);
  pinMode(stepPin, OUTPUT);
  pinMode(photodiodeS1Pin, INPUT);
  pinMode(photodiodeS2Pin, INPUT);
  pinMode(ledS1Pin, OUTPUT);
  pinMode(ledS2Pin, OUTPUT); 


  // Ask for the homing command
  Serial.println("Your Monochromator is now initialized and on.");
  Serial.println("Before starting operation please first home the motor.");
}

//////////////////////////////////////////////////////
// Loop function
//After initizalization the loop() function is executed constantly and it waits 
// commands so it can take actions
///////////////////////////////////////////////////////


void loop() {
  //here we check constantly if there is some information waiting in the serial communication
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n'); // Read the input from Serial Monitor
    input.trim(); // Trim any leading/trailing whitespace

    // Handle command and update state logic
    if (input == "home") {
      Serial.println("Homing command received.");
      Serial.flush();  // Ensure the immediate response is sent out
      homeStepperMotor(); //this is the homing function, defined below the loop 
    } else if (input == "mode_selected") { 
    //mode_slected just takes the value of the prefered mode for work
    while (true) {
            String modeInput = Serial.readStringUntil('\n');
            modeInput.trim();
            int modeSelection = modeInput.toInt();
            if (modeSelection == 0 || modeSelection == 1 || modeSelection == 2) {
                gratingMode = modeSelection;
                Serial.print("Grating mode selected. You are now operating with the ");
                if (gratingMode == 0) {
                    Serial.println("VIS Grating");
                } else if (gratingMode == 1) {
                    Serial.println("IR Grating");
                } else if (gratingMode == 2) {
                    Serial.println("Switch Mode");
                }
                break;  // Exit the loop once a valid mode is received
            } else {
                Serial.println("Invalid selection. Please choose an existing operating mode:");
            }
      }
    } else if (input.startsWith("wavelength_selected")) {
      //receiving of command to move to a specific wavelength along with 
      //the number of the specific wavelength
      float targetWavelength = input.substring(20).toFloat(); // Extract wavelength value
      //Serial.println("command received."); //<-- command for debugging if needed
      moveToWavelength(targetWavelength); // this is move to function, defined after the loop
    }
  }
}


//////////////////////////////////////////////////////
// Additionally defined functions, called in the loop
///////////////////////////////////////////////////////

void homeStepperMotor() {
    // Check if the photodiode S2 is already blocked (value > 300)
    if (analogRead(photodiodeS2Pin) > 300) {
        // Move forward until S2 is unblocked (value <= 300)
        while (analogRead(photodiodeS2Pin) > 300) {
            myStepper.move(100);
            myStepper.run();
        }
        // Move a little more to ensure proper unblocking
        myStepper.move(50);
        while (myStepper.distanceToGo() != 0) {
            myStepper.run();
        }
        // Now move backward until S2 is blocked (value < 300)
        while (analogRead(photodiodeS2Pin) < 300 && !emergencyStop) {
            myStepper.move(-1000000);
            myStepper.run();
        }
    } else {
        // Move backward until S2 is blocked (value < 300)
        while (analogRead(photodiodeS2Pin) < 300 && !emergencyStop) {
            myStepper.move(-1000000);
            myStepper.run();
        }
    }
    // Print the value of photodiode S2 for debugging
    Serial.print("Photodiode S2: "); //this value should be around 300 for proper work
    Serial.println(analogRead(photodiodeS2Pin));

    // Reset the stepper position to zero (home)
    myStepper.setCurrentPosition(0);
    
    Serial.println("Homed to initial position.");
    Serial.println("You can now select the grating you wish to operate with.");
}

void moveToWavelength(float targetWavelength) {
  int targetSteps;
  bool validInput = true;

  // Check if the input wavelength is within the valid range for the selected mode
  if (gratingMode == 0) { // VIS Grating only
    if (targetWavelength < 350 || targetWavelength > 1000) {
      Serial.println("Invalid wavelength for VIS Grating. Please enter a wavelength between 350 and 1000 nm:");
      validInput = false;
    } else {
      Serial.print("Moving to wavelength: ");
      Serial.println(targetWavelength);
      targetSteps = (targetWavelength + 377.2128) / 1.1164;
    }
  } else if (gratingMode == 1) { // IR Grating only
    if (targetWavelength < 587 || targetWavelength > 2000) {
      Serial.println("Invalid wavelength for IR Grating. Please enter a wavelength between 587 and 2000 nm:");
      validInput = false;
    } else {
      targetSteps = (targetWavelength + 4715.4390) / 0.5099;
    }
  } else if (gratingMode == 2) { // Switching between gratings mode
    if (targetWavelength < 350 || targetWavelength > 2000) {
      Serial.println("Invalid wavelength for switching mode. Please enter a wavelength between 350 and 2000 nm:");
      validInput = false;
    } else {
      if (targetWavelength < 650) {
        targetSteps = (targetWavelength + 389.2407) / 1.1127; // linear approximation VIS range
      } else {
        targetSteps = (targetWavelength + 4715.4390) / 0.5099; // linear approximation IR range
      }
    }
  }

  if (validInput) {
    myStepper.moveTo(targetSteps);
    //Serial.println("Moving to steps"); //<-- command for debugging if needed
    while (myStepper.distanceToGo() != 0) {
      myStepper.run(); // Continuously run the stepper

      // Check for an emergency stop command
      if (Serial.available() > 0) {
        String stopCommand = Serial.readStringUntil('\n');
        stopCommand.trim();
        if (stopCommand == "stop") {
          Serial.println("Emergency stop initiated.");
          myStepper.stop();
          Serial.println("You can now enter a new wavelength or choose another grating to work with.");
          return; // Exit the function early
        }
      }
    }

    // Only print this if the stepper has successfully finished its move
    Serial.print("Target wavelength reached. Current wavelength is: ");
    Serial.println(targetWavelength);
    Serial.println("You can now enter a new wavelength or choose another grating to work with.");
  }
}










