import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import time
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE

def create_camera_sdk():
    try:
        # if on Windows, use the provided setup script to add the DLLs folder to the PATH
        try:
            from .windows_setup import configure_path  # when imported as a package (e.g. from a notebook)
        except ImportError:
            from windows_setup import configure_path   # when run directly as a script
        configure_path()
    except ImportError:
        pass

    sdk = TLCameraSDK()

    return sdk

class Camera_CS126:
    """ Class to control Thorlabs cameras - CS126 and LP126"""
    def __init__(self, 
                 sdk: TLCameraSDK, 
                 serial_number: str, 
                 cam_type='MU'):
        """ """
        self.sdk = sdk
        self.serial_number = serial_number
        self.cam_type = cam_type 
        self.bit_depth = np.uint16
        self.out_bit_depth = np.float32

    def Init_device(self):
        available_cameras = self.sdk.discover_available_cameras()
        if len(available_cameras) < 1:
            print("No cameras detected")
            exit()
        if self.serial_number not in available_cameras:
            print('No cameras with the serial number')
            exit()
        else:
            self.Camera = self.sdk.open_camera(self.serial_number)

        print('Camera ' +  str(self.serial_number) +  ' is connected')

    def Shutdown_device(self):
        if self.Camera.is_armed:
            self.Camera.disarm()

        self.Camera.dispose()
        print('Camera ' +  str(self.serial_number) +  ' is disconnected')

    def Set_settings(self, 
                     exposure_time, 
                     gain, 
                     black_level, 
                     bit_depth=np.uint8, 
                     out_bit_depth = np.float32,
                     frame_rate_control_value=None):
        self.Camera.exposure_time_us = exposure_time
        self.Camera.gain = gain
        self.Camera.black_level = black_level
        self.bit_depth = bit_depth
        self.out_bit_depth = out_bit_depth
        if frame_rate_control_value != None:
            self.Camera.frame_rate_control_value = frame_rate_control_value
            self.Camera.is_frame_rate_control_enabled = True

    def Set_exposute_time_us(self, exposure_time):
        self.Camera.exposure_time_us = exposure_time

    def Arm(self):
        self.Camera.arm(2)
        self.Camera.issue_software_trigger()

    def Disarm(self):
        self.Camera.disarm()

    def Get_image(self, Num_frames_to_average=1, Num_frames_to_drop=0, delay=0):
        if self.cam_type == 'MU':
            av_image_array = np.zeros([Num_frames_to_average, self.Camera.image_height_pixels, self.Camera.image_width_pixels, 1], 
                                      dtype=self.bit_depth)
        else:
            av_image_array = np.zeros([Num_frames_to_average, self.Camera.image_height_pixels, self.Camera.image_width_pixels, 3], 
                                      dtype=self.bit_depth)
            
        #Drop images
        for i in range(Num_frames_to_drop):
            frame = None
            while frame is None:
                frame = self.Camera.get_pending_frame_or_null()
            time.sleep(delay)
        
        #Capture images
        for i in range(Num_frames_to_average):
            frame = None
            while frame is None:
                frame = self.Camera.get_pending_frame_or_null()

            if frame is not None:
                frame.image_buffer
                image_buffer_copy = np.copy(frame.image_buffer)
                numpy_shaped_image = image_buffer_copy.reshape(self.Camera.image_height_pixels, self.Camera.image_width_pixels)
                if self.cam_type == 'MU':
                    nd_image_array = np.full((self.Camera.image_height_pixels, self.Camera.image_width_pixels, 1), 0, dtype=self.bit_depth)
                    nd_image_array[:,:,0] = numpy_shaped_image
                else:
                    nd_image_array = np.full((self.Camera.image_height_pixels, self.Camera.image_width_pixels, 3), 0, dtype=self.bit_depth)
                    nd_image_array[:,:,0] = numpy_shaped_image
                    nd_image_array[:,:,1] = numpy_shaped_image
                    nd_image_array[:,:,2] = numpy_shaped_image
            else:
                print('No_frames')
            av_image_array[i, :, :, :] = nd_image_array

        return av_image_array.mean(axis=0, dtype=self.out_bit_depth)

#Test
if __name__=="__main__":
    serial_number = '35595'
    camerasdk = create_camera_sdk()
    Camera = Camera_CS126(camerasdk, serial_number)
    Camera.Init_device()
    Camera.Set_settings(exposure_time=10000, gain=0, black_level=5, bit_depth=np.uint16)

    Camera.Arm()
    time.sleep(0.1)
    image = Camera.Get_image(Num_frames_to_average=5)

    #Continuous_capturing_with_plt(Camera, zoom_value=0.4)

    Camera.Disarm()
    Camera.Shutdown_device()

    np.save('test_image.npy', image)
