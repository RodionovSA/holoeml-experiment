from monochromator.mono_class import MonochromatorControl

def main():
    # Initialize MonochromatorControl
    mono = MonochromatorControl()

    # Ask the user if they want to continue after creating the instance
    if input("Do you want to continue with initializing Arduino? (yes/no): ").strip().lower() != 'yes':
        print("Exiting the program.")
        return
    
    # Initialize Arduino
    mono.initialize_arduino()

    # Ask the user if they want to continue after initializing the Arduino
    if input("Do you want to continue with homing the motor? (yes/no): ").strip().lower() != 'yes':
        print("Exiting the program.")
        return

    # Home the motore
    
    mono.home_motor()

    # Ask the user if they want to continue after homing the motor
    if input("Do you want to continue with selecting the grating mode? (yes/no): ").strip().lower() != 'yes':
        print("Exiting the program.")
        return

    mono.select_grating_mode()

    # Ask the user if they want to continue after selecting the grating mode
    if input("Do you want to continue with setting the wavelength? (yes/no): ").strip().lower() != 'yes':
        print("Exiting the program.")
        return

    mono.set_wavelength()

    print("Program completed.")

if __name__ == "__main__":
    main()

