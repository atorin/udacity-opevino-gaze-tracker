import os

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
    proc_image = models['fd'].preprocess_input(image)
    output = models['fd'].infer(proc_image)
    print(output)

def main():
    models = load_models()
    feed=InputFeeder(input_type='video', input_file='../bin/demo.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        main_loop(batch, models)
    feed.close()

if __name__ == "__main__":
    main()