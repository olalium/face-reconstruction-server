import keras
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.utils.generic_utils import CustomObjectScope
from keras import backend as K
import tensorflow as tf

class MobilenetPosPredictor():
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1  # this *1.1 is some what of amystery..
        self.model = None

        # set tensorflow session GPU usage
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        sess = tf.Session(config=config)
        set_session(sess)

    def restore(self, model_path):
        with CustomObjectScope(
                {'relu6': relu6}):  # ,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D
            self.model = keras.models.load_model(model_path)

    def predict(self, image):
        x = image[np.newaxis, :, :, :]
        pos = self.model.predict(x=x)
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        raise NotImplementedError

def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)