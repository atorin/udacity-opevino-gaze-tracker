import os
import cv2

from input_feeder import InputFeeder
from face_detection import FaceDetection


def load_models():
    models = {}
    models['fd'] = FaceDetection()
    models['fd'].load_model('../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml',
        device="CPU"    
    )

    return models

def main_loop(image, models):
    # find and crop the face
    face, image = models['fd'].infer_and_crop(image)

    return face 

def main():
    models = load_models()
    feed=InputFeeder(input_type='video', input_file='../bin/demo.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        if batch is not None:
            cv2.imshow('frame', main_loop(batch, models))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    feed.close()

if __name__ == "__main__":
    main()