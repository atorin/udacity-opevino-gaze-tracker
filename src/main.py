import os
import sys
import cv2
import numpy as np
import time

from input_feeder import InputFeeder
from parser import get_args
from face_detection import FaceDetection
from landmarks import Landmarks
from headpose import HeadPose
from gaze_estimator import GazeEstimator
from mouse_controller import MouseController

MODELS = {
    'fd': "Face detection",
    'lm': "Landmark detection",
    'hp': "Head pose estimation",
    'ge': "Gaze estimation"
}

def load_models(p):
    """
    Load the OpenVINO models in a dictionary to handle them more easily

    Input: `p`, a dictionary with the models' paths
    """
    # Get the device ('CPU' will be selected if None)
    models = {}
    models['fd'] = FaceDetection()
    models['lm'] = Landmarks()
    models['hp'] = HeadPose()
    models['ge'] = GazeEstimator()

    # Load all the files with the relative device
    for label in ['fd','lm','hp','ge']:
        start = time.time()
        models[label].load_model(p[f'mod_{label}'], device=p[f'device_{label}'])    
        print(f'Model: {MODELS[label]} --- Loading time: {1000*(time.time()-start):.1f} ms')

    return models

def crop_eyes(image, upper_corner, eyes):
    '''
    Return the coordinates of the eyes relative to
    the upper left corner of the face crop.
    '''
    e1, e2 = eyes
    Y,X = upper_corner

    # print(f"Eyes coords: {e1}, {e2}")

    # the size of half the square surrounding the eye
    s = 30
    eye1_x = (e1[0]-s+X, e1[0]+s+X)
    eye1_y = (e1[1]-s+Y, e1[1]+s+Y)
    eye2_x = (e2[0]-s+X, e2[0]+s+X)
    eye2_y = (e2[1]-s+Y, e2[1]+s+Y)

    e1_slx = slice(eye1_x[0], eye1_x[1])
    e1_sly = slice(eye1_y[0], eye1_y[1])
    e2_slx = slice(eye2_x[0], eye2_x[1])
    e2_sly = slice(eye2_y[0], eye2_y[1])

    eye_crop_1 = image[e1_slx, e1_sly]
    eye_crop_2 = image[e2_slx, e2_sly]

    return eye_crop_1, eye_crop_2

def get_position(x, y):
    SPEED = 10
    X = int(x * SPEED * 640 + 640)
    Y = int(y * SPEED * 400 + 400)
    return X, Y

def plot_position(image, x, y):
    x = int(x)
    y = int(y)
    image = cv2.circle(image, (x,y), radius=5, color=(255,255,255), thickness=-1)
    return image

def plot_gaze(image, gaze, eye1, eye2):
    '''
    Plot arrow from eye to gaze.

    Test the length of the gaze vector with different values
    of the expansion coefficient.
    '''
    EXP=100
    end_1 = int(eye1[0]+EXP*gaze[0]), int(eye1[1]-EXP*gaze[1])
    end_2 = int(eye2[0]+EXP*gaze[0]), int(eye2[1]-EXP*gaze[1])
    image = cv2.line(image, eye1, end_1, (255,255,255), thickness=2)
    image = cv2.line(image, eye2, end_2, (255,255,255), thickness=2)

    # print(f"From {eye1} to {end_1}")
    return image

def main_loop(image, models):
    # find and crop the face
    face, upper_corner, image = models['fd'].infer_and_crop(image)

    # check if there is at least one face
    if np.all(face.shape):
        # find landmarks
        face, eye1, eye2 = models['lm'].infer_and_get_eyes(face)
        
        # find eye boxes
        eb1, eb2 = crop_eyes(image, upper_corner, (eye1, eye2))
        # image = draw_eyes(image, eb1, eb2)

        # find head pose vector
        angles = models['hp'].infer_and_plot_vecs(face)

        gaze = models['ge'].infer_gaze(eb1, eb2, angles).flatten()

        face = plot_gaze(face, gaze, eye1, eye2)

        # print(f"Gaze array: {gaze}")

        x, y = get_position(gaze[0], gaze[1])

        pos = gaze[0], gaze[1]
    else:
        print("===ERROR!\nNo one is in front of the camera. Closing program!\n===")
        sys.exit(0)

    return image, pos

def main():
    # Load parameters
    params = get_args()

    mouse_prec = params['mouse_prec']
    mouse_speed = params['mouse_speed']
    mouse = MouseController(mouse_prec, mouse_speed)
    models = load_models(params)

    # Load input feed
    input_type = params['input_type']
    if input_type=='cam':
        input_file = None
    else:
        input_file = params['input_file_path']

    feed=InputFeeder(input_type=input_type, input_file=input_file)
    feed.load_data()
    for batch in feed.next_batch():
        if batch is not None:
            image, pos = main_loop(batch, models)
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            mouse.move(pos[0], pos[1])
            # break
        else:
            break
    feed.close()

if __name__ == "__main__":
    main()