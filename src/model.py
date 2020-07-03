'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
from openvino.inference_engine import IENetwork, IECore

class OpenvinoModel:
    '''
    Parent class for all the models.

    Use to DRY the code.
    '''
    def __init__(self):
        ### Initialize any class variables desired ###
        self.ie = None
        self.input_blob = None
        self.network = None
        self.exec_network = None
        self.request_handle = None
        self.request_status = None

    def load_model(self, model_path, device="CPU"):
        # Initialize the ie
        self.ie = IECore()

        ### Load the model ###
        model_xml = model_path
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Read the IR as a IENetwork
        self.network = self.ie.read_network(model=model_xml, weights=model_bin)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        ### Return the loaded inference ie ###
        # Load the IENetwork into the ie
        self.exec_network = self.ie.load_network(network=self.network, device_name=device, num_requests=0)

        return self.exec_network

    def infer(self, image):
        '''
        Perform sync inference
        '''
        return self.exec_network.infer({self.input_blob: image})[self.output_blob]

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
