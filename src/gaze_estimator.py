'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from model import OpenvinoModel
from openvino.inference_engine import IENetwork, IECore
import numpy as np

# x and y side of input image
SIDE = 60

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

def draw_vecs(face, angles):
    '''
    Draw boxes around a person in the image.

    '''
    return face

class GazeEstimator(OpenvinoModel):
    '''
    Class for the Gaze Estimation Model.

    Inherit all the main methods from the parent class.
    Only define specific methods here. 
    '''

    def infer(self, eye1, eye2, angles):
        '''
        Perform sync inference

        Inputs (from documentation)
            Blob in the format [BxCxHxW] where:
            B - batch size
            C - number of channels
            H - image height
            W - image width
            with the name left_eye_image and the shape [1x3x60x60].

            Blob in the format [BxCxHxW] where:
            B - batch size
            C - number of channels
            H - image height
            W - image width
            with the name right_eye_image and the shape [1x3x60x60].

            Blob in the format [BxC] where:
            B - batch size
            C - number of channels
            with the name head_pose_angles and the shape [1x3].
        '''


        print(f"Real Eye 1: {eye1.shape}")
        proc_eye1 = self.preprocess_input(eye1)
        print(f"Eye 1: {proc_eye1.shape}")
        print(f"Real Eye 2: {eye2.shape}")
        proc_eye2 = self.preprocess_input(eye2)
        print(f"Eye 2: {proc_eye2.shape}")
        # build input blob
        input_blob = {
            'left_eye_image' : proc_eye1,
            'right_eye_image' : proc_eye2,
            'head_pose_angles' : angles
        }

        # Return the complete dictionary
        # Use the full input blob as input
        return self.exec_network.infer(input_blob)

    def preprocess_input(self, orig_image):
        '''
        Given an input image, height and width:
        - Resize to width and height
        - Transpose the final "channel" dimension to be first
        - Reshape the image to add a "batch" of 1 at the start 
        '''

        image = np.copy(orig_image)
        image = image.transpose((2,0,1))
        image = image.reshape(1, 3, SIDE, SIDE)

        return image

    def preprocess_output(self, outputs):
        '''
        Extract the angles
        '''
        # reshape output array to 1d
        gaze = outputs['gaze_vector']
        return gaze

    def infer_gaze(self, eye1, eye2, angles):
        '''
        Perform inference on the image and crop the face
        '''
        try:
            # perform inference
            outputs = self.infer(eye1, eye2, angles)
            gaze = self.preprocess_output(outputs)
            # image = draw_vecs(image, angles)

            return gaze 
        except:
            print("Error")
            return []