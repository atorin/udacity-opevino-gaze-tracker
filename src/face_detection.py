'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from model import OpenvinoModel
from openvino.inference_engine import IENetwork, IECore
import numpy as np

def get_image_shape(image):
    H,W,_ = image.shape
    return H,W

def get_coords_score(face, dims):
    '''
    Get the actual coordinates of a face bounding box
    and the inference score
    '''
    H,W = dims

    _,_ , score, x1, y1, x2, y2 = face

    x1 = int(W * x1)
    x2 = int(W * x2)
    y1 = int(H * y1)
    y2 = int(H * y2)

    return (x1,y1), (x2,y2), score

def draw_box(image, face, thres):
    '''
    Draw boxes around a person in the image.

    '''
    H,W = get_image_shape(image)

    p1,p2,score = get_coords_score(face, (H,W))

    if score>=thres:
        colour = (0,255,0)

        image = cv2.rectangle(image, p1, p2, colour, 2)

    return image

def crop_face(image, face):
    H,W = get_image_shape(image)

    (x1,y1),(x2,y2),_ = get_coords_score(face, (H,W))

    slx = slice(x1,x2)
    sly = slice(y1,y2)

    return image[sly,slx]

class FaceDetection(OpenvinoModel):
    '''
    Class for the Face Detection Model.

    Inherit all the main methods from the parent class.
    Only define specific methods here. 
    '''
    
    def preprocess_input(self, orig_image):
        '''
        Given an input image, height and width:
        - Resize to width and height
        - Transpose the final "channel" dimension to be first
        - Reshape the image to add a "batch" of 1 at the start 
        '''
        image = np.copy(orig_image)
        height = 384
        width = 672
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
        # grab content of the blob
        content = outputs[0][0]
        face = content[0]
        print(face)
        if face[1]==0:
            # no faces found
            return []
        for f in content[1:]:
            if f[1]==0:
                # exactly one face was found
                return face
            if f[2]>face[2]:
                # if new score is bigger than previous, keep new
                face = f
        return face

    def infer_and_crop(self, image):
        '''
        Perform inference on the image and crop the face
        '''
        # perform inference
        proc_image = self.preprocess_input(image)
        outputs = self.infer(proc_image)
        face = self.preprocess_output(outputs)
        image = draw_box(image, face, 0.6)
        face_crop = crop_face(image, face)

        return face_crop, image