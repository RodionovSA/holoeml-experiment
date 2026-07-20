# Amplitude measurement protocol

This is the protocol for polarization-resolved hyperspectral measurements with the HoloEML setup. The idea behind the measurement is to record an image for each wavelength and polarization state. By default, the wavelength range is 400-1000 nm with a 5 nm step, and there are two polarization states: linear polarization along the x or y axis. Each run records three frame types: sample, reference, and black. The final transmission is computed as T(wvl, pol, x, y) = (Sample - Black) / (Reference - Black).

**Important:** For these measurements, the reference arm must be covered so that only one arm's signal reaches the camera.

**Important:** All measurements must be performed in absolute darkness.

## Instruments
- Monochromator: Newport TLS260-300X with a 300 W Xenon lamp and a 300 um fixed slit (5 nm bandwidth)
- Camera: Thorlabs LP126 MU
- Filter wheel: part of the monochromator
- Polarizer: Thorlabs DGL10 on a Thorlabs K10CR2
- Focus: Thorlabs brushed motor with a K-cube

## 1. Configuration file
First, open `instruments/config/config.yaml` and check the serial number for all devices, as well as the port and address for the monochromator and filter wheel — this is the shared equipment config used by every protocol and script. Measurement-specific parameters (wavelength sweep, brightness calibration, measurement settings) live separately in `amplitude/config/config.yaml`.

## 2. Connection
Connect the monochromator's Arduino board, filter wheel, camera, z-translation stage, and polarizer to your PC via USB. Run `sh ./amp_testconn.sh` to verify that everything is connected properly.

## 3. Warming up
According to our analysis, the monochromator's lamp requires 2 hours of warm-up time to stabilize its output power. Before starting any measurements (including brightness calibration), the lamp must be allowed to warm up for 2 hours. Additionally, the camera requires some warm-up time, so it is good practice to connect it and let it warm up for the same duration as the lamp.

## 4. Brightness calibration
By default, exposure settings files are located in `amplitude/config`, with separate files for x and y polarization. If any change to the setup could affect illumination, or if new illumination conditions are introduced, a new brightness calibration must be run for both x and y polarization.

To do so, first check the brightness calibration section of `amplitude/config/config.yaml`. The most important parameters are `calib_target_brightness`, `calib_priority`, and `calib_max_exposure_ms`.

1. Define the target brightness to be reached without a sample present. This is defined as `calib_target_brightness * 4096`. Through experimentation, we found that 0.5 is the optimal value.
2. Define the optimization priority: exposure time or gain. Exposure time is the preferred mode.
3. Define the maximum exposure time. Based on our experience with the Thorlabs camera, it is best not to exceed 1000 ms.

After changing the config, it is recommended to reset the gain values in both exposure settings files to 0, to avoid converging on a high-gain/low-exposure optimum. Finally, make sure there is no sample on the stage and that the beam reaches the second objective without obstruction, then run `sh ./amp_run.sh --calibration`. This can take anywhere from 20 minutes to a few hours, depending on the target and exposure times. Once complete, verify that `exposure_settings_x` and `exposure_settings_y` have been updated.

## 5. Measurement order
It is highly recommended to record sample measurements first, since they require focusing and sample alignment, which take time. Once that is done, reference and black measurements can be run together. The same reference and black measurements can also be reused for multiple sample measurements, as long as the monochromator and camera have not been powered off in between.

## 6. Sample measurements
Place your sample on the stage, then run `sh ./focus_run.sh` to open the live focusing app (live camera view with exposure/gain sliders, focus-motor `move_to`/`move_by` controls, a sharpness readout, zoom, and an alignment crosshair). Make sure the illumination wavelength is greater than 490 nm and that you can clearly see the field of view. Use the manual K-cube controllers to adjust focus and positioning so that your sample is within the camera's field of view and roughly in focus, then use the manual rotation stage to adjust the sample's rotational position.

In the focusing app, use the `move_by` control with small steps to fine-tune the focus, watching the sharpness readout. Important: verify that the motor moves precisely and returns to the same position each time (manual K-cube control can interfere with Python control). Once everything is set, close the focusing app (this releases the camera and focus motor) and run `sh ./amp_run.sh -m sample`.

## 7. Reference and black measurements
It is recommended to run reference and black measurements together. To do so, remove your sample and run `sh ./amp_run.sh -m reference black`.
