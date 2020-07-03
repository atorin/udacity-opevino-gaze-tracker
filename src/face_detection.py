'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from model import OpenvinoModel
from openvino.inference_engine import IENetwork, IECore

class FaceDetection(OpenvinoModel):
    '''
    Class for the Face Detection Model.

    Inherit all the main methods from the parent class.
    Only define specific methods here. 
    '''
    
    def preprocess_input(self, image):
        '''
        Given an input image, height and width:
        - Resize to width and height
        - Transpose the final "channel" dimension to be first
        - Reshape the image to add a "batch" of 1 at the start 
        '''
        # image = np.copy(input_image)
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
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
