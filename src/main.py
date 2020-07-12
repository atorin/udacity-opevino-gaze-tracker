import os
import cv2

from input_feeder import InputFeeder
from face_detection import FaceDetection
from landmarks import Landmarks
from headpose import HeadPose
from gaze_estimator import GazeEstimator
import numpy as np
from mouse_controller import MouseController

def load_models():
    models = {}
    models['fd'] = FaceDetection()
    models['fd'].load_model('../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml',
        device="CPU"    
    )

    models['lm'] = Landmarks()
    models['lm'].load_model('../models/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml',
        device="CPU"
    )

    models['hp'] = HeadPose()
    models['hp'].load_model('../models/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml',
        device="CPU"
    )

    models['gaze'] = GazeEstimator()
    models['gaze'].load_model('../models/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml',
        device="CPU"
    )
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

    print(f"From {eye1} to {end_1}")
    return image

def main_loop(image, models):
    # find and crop the face
    face, upper_corner, image = models['fd'].infer_and_crop(image)

    if face is not None:
        # find landmarks
        face, eye1, eye2 = models['lm'].infer_and_get_eyes(face)
        
        # find eye boxes
        eb1, eb2 = crop_eyes(image, upper_corner, (eye1, eye2))
        # image = draw_eyes(image, eb1, eb2)

        # find head pose vector
        angles = models['hp'].infer_and_plot_vecs(face)

        gaze = models['gaze'].infer_gaze(eb1, eb2, angles).flatten()

        face = plot_gaze(face, gaze, eye1, eye2)

        # print(f"Gaze array: {gaze}")

        x, y = get_position(gaze[0], gaze[1])

    pos = gaze[0], gaze[1]

    return image, pos

def main():
    mouse = MouseController('low', 'slow')
    models = load_models()
    # feed=InputFeeder(input_type='video', input_file='../bin/demo.mp4')
    feed=InputFeeder(input_type='cam')
    feed.load_data()
    for batch in feed.next_batch():
        if batch is not None:
            image, pos = main_loop(batch, models)
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            mouse.move(pos[0], pos[1])
        else:
            break
    feed.close()

if __name__ == "__main__":
    main()