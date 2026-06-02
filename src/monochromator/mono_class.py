import serial
import serial.tools.list_ports
import time
import sys
import msvcrt
import os

#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # Reopen stdout in unbuffered mode

class MonochromatorControl:
    #Create an instance of MonochromatorControl 
    # without immediately needing to connect to the Arduino or the COM.
    def __init__(self, port=None, baud_rate=9600, timeout=1):
        self.ser = None
        self.arduino_initialized = False  # Flag to track Arduino initialization
        self.motor_homed = False #Flag to track the initial homing of the motor
        self.grating_selected = False #Flag to track if grating mode is selected
        
        if not port:
            print("First select a COM port.")
            ports = self.list_available_ports()
            if ports:
                print("Available COM ports:")
                print(f"{ports}")
                port=input("Enter the number of the COM port to connect to: ")
            else:
                print("No COM ports available. Exiting...")
                return
        self.connect_to_monochromator(port, baud_rate, timeout)
    
    @staticmethod
    def list_available_ports():
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]


    #Connecting to the COM port, without communicating with the Arduino
    def connect_to_monochromator(self, selected_port=None, baud_rate=9600, timeout=1):
        while not self.ser:
            if not selected_port:
                print("First select a COM port.")
                ports = self.list_available_ports()
                if ports:
                    print(f"Available COM ports: {ports}")
                    selected_port = input("Enter the COM port to connect to (e.g., COM3): ").strip()
                else:
                    print("No COM ports available. Exiting...")
                    return
            try:
                self.ser = serial.Serial(selected_port, baud_rate, timeout=timeout)
                time.sleep(0.1)  # Give some time for the connection to establish
                print(f"Connected to {selected_port}, you can now initialize and start your Monochromator.")
            except serial.SerialException:
                print(f"Could not open {selected_port}. Please try again.")
                selected_port = None  # Reset selected_port to prompt the user again

    #Assumes the COM port is connected and prepeares the Arduino to receive commands
    def initialize_arduino(self):
        if not self.ser:
            print("COM port is not selected.")
            return
        
        time.sleep(0.1)  # Allow some time for Arduino to initialize and send data

        while self.ser.in_waiting > 0:  # Check if there are more lines to read
            line = self.ser.readline().decode().strip()
            print(line)  
        self.arduino_initialized = True
    
    def home_motor(self):
        if not self.ser:
            print("COM port is not selected.")
            return
        if not self.arduino_initialized:
            print("The Arduino is off, please press START.")
            return

        self.ser.write(b'home\n')

        while True:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode().strip()
                print(line)
                if "You can now select the grating you wish to operate with." in line: 
                    break

        self.motor_homed = True  # Update the motor homed status
    
    def select_grating_mode(self, mode=None):
        if not self.ser:
            print("COM port is not selected.")
            return
        if not self.arduino_initialized:
            print("The Arduino is off, please press START.")
            return
        if not self.motor_homed:
            print("Please home the motor first to start operation.")
            return

        valid_modes = ["VIS Grating", "IR Grating", "Switch Mode"]

        if mode is None:
            while True:
                if not self.grating_selected:
                    mode = input("Please select a grating mode (VIS Grating/IR Grating/Switch mode): ").strip()
                elif self.grating_selected:
                    # Ask the user if they want to change the grating mode
                    confirm = input("A grating mode is already selected. Are you sure you want to change it? (y/n): ").strip().lower()
                    if confirm != 'y':
                        print("Grating mode change cancelled.")
                        return  # Exit the method if the user does not want to change the mode
                    mode = input("Please select a new grating mode (VIS Grating/IR Grating/Switch mode): ").strip()

                if mode in valid_modes:
                    break
                else:
                    print("Invalid grating mode selected. Please try again.")
        
        if mode not in valid_modes:
            raise ValueError(f'Invalid grating mode selected. Must be in {valid_modes}')

        # Send the initial mode selection command
        self.ser.write(b'mode_selected\n')

        # Send the grating mode selection based on user input
        if mode == "VIS Grating":
            self.ser.write(b'0\n')
        elif mode == "IR Grating":
            self.ser.write(b'1\n')
        elif mode == "Switch Mode":
            self.ser.write(b'2\n')

        # Wait for confirmation from the Arduino
        while True:
            if self.ser.in_waiting > 0:  # Check if there are more lines to read
                line = self.ser.readline().decode().strip()
                print(line)
                if "Grating mode selected." in line:
                    break

        self.grating_selected = True

        
    def set_wavelength(self, wavelength):
        if not self.ser:
            print("COM port is not selected.")
            return
        if not self.arduino_initialized:
            print("The Arduino is off, please press START.")
            return
        if not self.motor_homed:
            print("Please home the motor first to start operation.")
            return
        if not self.grating_selected:
            print("Please select a preferred grating to work with and then move to the desired wavelength.")
            return 
        
        # Prompt the user to enter the target wavelength
        #wavelength = input("Enter the target wavelength: ").strip()

        # Send the command along with the wavelength to the Arduino
        command = f'wavelength_selected {wavelength}\n'
        self.ser.write(command.encode())

        while True:
            # Check for user input
            if msvcrt.kbhit():
                emergency = msvcrt.getch().decode('utf-8').lower()
                if emergency == 'e':
                    self.ser.write(b'stop\n')
                    break

            # Check for Arduino responses
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode().strip()
                print(line)
                if "You can now enter a new wavelength or choose another grating to work with." in line or "Emergency stop initiated." in line:
                    break

    def disconnect(self):
        if self.ser:
            try:
                self.ser.close()
                print("Disconnected from Arduino.")
            except Exception as e:
                print(f"Error disconnecting: {e}")
            finally:
                self.ser = None
        else:
            print("No serial connection to disconnect.")