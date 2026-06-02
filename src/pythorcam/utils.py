import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def zoom(img, scaling_factor):
    return cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)

#Get video from a camera
def Continuous_capturing_with_plt(Camera, zoom_value):
    try:
        plt.ion()
        fig, ax = plt.subplots()

        Image = Camera.Get_image()
        img = ax.imshow(zoom(Image, 0.5), cmap='grey')

        while True:
            
            Image = Camera.Get_image()
                
            img.set_data(zoom(Image, zoom_value))
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("loop terminated")
        
    finally:
        plt.ioff()
        plt.show()

def Continuous_capturing_with_frame_plt(Camera, zoom_value, top_line, left_line):
    pixel_size = 3.45

    Structure_horizontal_size = 1135
    Structure_vertical_size = 753

    Structure_horizontal_zoom = 10 * Structure_horizontal_size #10x objective
    Structure_vertical_zoom = 10 * Structure_vertical_size

    Num_pixels_structure_horizontal = int(Structure_horizontal_zoom / pixel_size)
    Num_pixels_structure_vertical = int(Structure_vertical_zoom / pixel_size)

    try:
        plt.ion()
        fig, ax = plt.subplots()

        Image = Camera.Get_image()
        img = ax.imshow(zoom(Image, zoom_value), cmap='grey')
        plt.axhline(y=int(zoom_value*top_line), color='r', linestyle='-')
        plt.axvline(x=int(zoom_value*left_line), color='r', linestyle='-')
        plt.axhline(y=int(zoom_value*top_line) + int(zoom_value*Num_pixels_structure_vertical), color='r', linestyle='-')
        plt.axvline(x=int(zoom_value*left_line) + int(zoom_value*Num_pixels_structure_horizontal), color='r', linestyle='-')

        while True:
            
            Image = Camera.Get_image()
                
            img.set_data(zoom(Image, zoom_value))
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("loop terminated")
        
    finally:
        plt.ioff()
        plt.show()
        

#Autofocus functions (works only for CS126MU)
def calculate_focus_measure(image):
    #image = cv2.medianBlur(image.astype(np.uint8), int(9))
    return cv2.Laplacian(image, cv2.CV_64F).var()

def autofocus(Camera_trans, Camera_ref, K_cube, Start_Position, End_Position, Step_size, velocity=0.1, 
              Num_frames_to_average = 1, Num_frames_to_drop = 5, delay=0):
    # ####Crop region####
    # pixel_size = 3.45

    # Structure_horizontal_size = 1135
    # Structure_vertical_size = 753

    # Structure_horizontal_zoom = 10 * Structure_horizontal_size #10x objective
    # Structure_vertical_zoom = 10 * Structure_vertical_size

    # Num_pixels_structure_horizontal = int(Structure_horizontal_zoom / pixel_size)
    # Num_pixels_structure_vertical = int(Structure_vertical_zoom / pixel_size)

    # Left = left_boundary
    # Right = left_boundary + Num_pixels_structure_horizontal
    # Top = top_boundary
    # Bottom = top_boundary + Num_pixels_structure_vertical
    
    #######################
    best_focus = -1
    best_position = Start_Position

    K_cube.Move_to_position(float(Start_Position), float(0.1))
    time.sleep(0.5)

    initial_position = Start_Position
    steps = int((End_Position - Start_Position)/Step_size)
    positions = np.linspace(Start_Position, End_Position, steps)

    Focus_values = -np.ones([steps])
    Images = np.zeros([steps, Camera_trans.Camera.image_height_pixels, Camera_trans.Camera.image_width_pixels])
    for step in range(steps):
        current_focus = -1

        Image_trans = Camera_trans.Get_image(Num_frames_to_average = int(Num_frames_to_average), 
                                 Num_frames_to_drop = int(Num_frames_to_drop), delay=delay)
        
        Image_ref = Camera_ref.Get_image(Num_frames_to_average = int(Num_frames_to_average), 
                                 Num_frames_to_drop = int(Num_frames_to_drop), delay=delay)
            
        current_focus = calculate_focus_measure(Image_trans/Image_ref)*1000
        Focus_values[step] = current_focus
        Images[step, :, :] = Image_trans[:,:,0]

        print('current_focus: ', current_focus)

        if (current_focus > best_focus) and (step != 0):
            best_focus = current_focus
            best_position = K_cube.Get_position()

        K_cube.Move_to_position(float(Start_Position + (step + 1)*Step_size), float(velocity))
        time.sleep(0.1)

    Focus = np.array([positions, Focus_values])
    K_cube.Move_to_position(float(best_position), float(0.05))
    time.sleep(0.5)

    return best_position, Focus, Images

#Autoexposure
def autoexposure(Camera, initital_exposure_time, target_brightness, tolerance, increment, max_number_of_steps, 
                 Num_frames_to_average = 1, Num_frames_to_drop = 5, delay=0):
    # ####Crop region####
    # pixel_size = 3.45

    # Structure_horizontal_size = 1135
    # Structure_vertical_size = 753

    # Structure_horizontal_zoom = 10 * Structure_horizontal_size #10x objective
    # Structure_vertical_zoom = 10 * Structure_vertical_size

    # Num_pixels_structure_horizontal = int(Structure_horizontal_zoom / pixel_size)
    # Num_pixels_structure_vertical = int(Structure_vertical_zoom / pixel_size)

    # Left = left_boundary
    # Right = left_boundary + Num_pixels_structure_horizontal
    # Top = top_boundary
    # Bottom = top_boundary + Num_pixels_structure_vertical
    
    ###################################################
    Camera.Set_exposute_time_us(initital_exposure_time)

    current_exposure_time = initital_exposure_time

    for i in range(max_number_of_steps):
        Image = Camera.Get_image(Num_frames_to_average=int(Num_frames_to_average), 
                                 Num_frames_to_drop=int(Num_frames_to_drop), delay=delay)

        mean_brightness = Image.mean()
    
        if abs(mean_brightness - target_brightness) < tolerance:
            print('Target brightness reached')
            print('Value: ', mean_brightness)
            break
         
        if mean_brightness < target_brightness:
            current_exposure_time = int(min(current_exposure_time*(1 + increment), 14700924))
        else:
            current_exposure_time = int(max(current_exposure_time*(1 - increment), 28))
        
        #print(i)
        Camera.Set_exposute_time_us(current_exposure_time)
        time.sleep(0.5)

    return current_exposure_time