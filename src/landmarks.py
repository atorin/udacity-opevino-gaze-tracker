'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from model import OpenvinoModel
from openvino.inference_engine import IENetwork, IECore
import numpy as np

# x and y side of input image
SIDE = 48

def get_image_shape(image):
    H,W,_ = image.shape
    return H,W

def get_coords(face, eyes):
    '''
    Get the actual coordinates of a face bounding box
    and the inference score
    '''
    H,W = get_image_shape(face)

    x1, y1, x2, y2 = eyes

    x1 = int(W * x1)
    x2 = int(W * x2)
    y1 = int(H * y1)
    y2 = int(H * y2)

    return (x1,y1), (x2,y2)

def draw_eyes(face, eyes):
    '''
    Draw boxes around a person in the image.

    '''
    e1,e2 = get_coords(face, eyes)

    # print(f"Eyes coords: {e1}, {e2}")

    # colour = (255,0,0)
    # face = cv2.circle(face, e1, radius=5, color=colour, thickness=-1)
    # face = cv2.circle(face, e2, radius=5, color=colour, thickness=-1)

    # the size of half the square surrounding the eye
    s = 30
    eye1 = ((e1[0]-s, e1[1]-s), (e1[0]+s, e1[1]+s))
    eye2 = ((e2[0]-s, e2[1]-s), (e2[0]+s, e2[1]+s))

    face = cv2.rectangle(face, eye1[0], eye1[1], (0,0,255), 2)
    face = cv2.rectangle(face, eye2[0], eye2[1], (0,0,255), 2)

    return face

class Landmarks(OpenvinoModel):
    '''
    Class for the Face Detection Model.

    Inherit all the main methods from the parent class.
    Only define specific methods here. 
    '''

    def infer(self, image):
        '''
        Perform sync inference
        '''
        proc_image = self.preprocess_input(image)
        return self.exec_network.infer({self.input_blob: proc_image})[self.output_blob]

    def preprocess_input(self, orig_image):
        '''
        Given an input image, height and width:
        - Resize to width and height
        - Transpose the final "channel" dimension to be first
        - Reshape the image to add a "batch" of 1 at the start 
        '''
        image = np.copy(orig_image)
        height = SIDE
        width = SIDE

        # print(f"Image size: {image.shape}")
        try:
            image = cv2.resize(image, (width, height), cv2.INTER_CUBIC)
        except TypeError:
            return None
        image = image.transpose((2,0,1))
        image = image.reshape(1, 3, height, width)

        return image

    def preprocess_output(self, outputs):
        '''
        Extract the face with maximum score
        '''
        # reshape output array to 1d
        arr = outputs.flatten()
        eyes = arr[:4]
        return eyes

    def infer_and_get_eyes(self, image):
        '''
        Perform inference on the image and crop the face
        '''
        # perform inference
        outputs = self.infer(image)
        eyes = self.preprocess_output(outputs)
        image = draw_eyes(image, eyes)
        eye1, eye2 = get_coords(image, eyes)
        # face_crop = crop_face(image, face)

        return image, eye1, eye2