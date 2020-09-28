import os

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale, estimate_transform, warp

from networks import MobilenetPosPredictor
from image_processor import ImageProcessor

TRIANGLES_PATH = 'Data/uv-data/triangles.txt'
FACE_IND_PATH = 'Data/uv-data/face_ind.txt'
EXTRA_FACE_IND_PATH = 'Data/uv-data/extra_bfm_ind.txt'
BFM_KPT_IND = 'Data/uv-data/bfm_kpt_ind.txt'
MODEL_PATH = 'Data/net-data/trained_fg_then_real.h5'

class Predictor(object):
    def __init__(self):
        super(Predictor, self).__init__()
        self.triangles = np.loadtxt(TRIANGLES_PATH).astype(np.int32)
        self.face_ind = np.loadtxt(FACE_IND_PATH).astype(np.int32)
        self.extra_face_ind = np.loadtxt(EXTRA_FACE_IND_PATH).astype(np.int32)
        self.bfm_kpt_ind = np.loadtxt(BFM_KPT_IND).astype(np.int32)
        self.pos_predictor = self.generate_restored_model()
        self.image_processor = ImageProcessor()
    
    def generate_restored_model(self):
        pos_predictor = MobilenetPosPredictor(256, 256)
        mobilenet_pos_predictor = os.path.join('', MODEL_PATH)  # Data/net-data/keras_mobilenet_prn_20_epochs_097.h5')
        if not os.path.isfile(mobilenet_pos_predictor):
            print("please download trained model first.")
            return null
        #TODO validate
        pos_predictor.restore(mobilenet_pos_predictor)
        return pos_predictor

    def generate_and_save_obj_from_pos(self, pos, image, obj_name):
        all_vertices = np.reshape(pos, [256 ** 2, -1])
        vertices = all_vertices[self.face_ind, :]

        save_vertices = vertices.copy()
        save_vertices[:, 1] = 256 - 1 - save_vertices[:, 1]

        [h, w, _] = image.shape
        vertices[:, 0] = np.minimum(np.maximum(vertices[:, 0], 0), w - 1)  # x
        vertices[:, 1] = np.minimum(np.maximum(vertices[:, 1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        
        colors = image[ind[:, 1], ind[:, 0], :]  # n x 3

        write_obj_with_colors(obj_name, save_vertices, self.triangles, colors)

    def predict_face_from_images(self, image1, image2, app):
        app.logger.info("im in boyys")
        cropped_image1, crop_tform1 = self.image_processor.get_cropped_image(image1)
        app.logger.info("im in boyys1")
        cropped_image2, crop_tform2 = self.image_processor.get_cropped_image(image2)
        app.logger.info("im in boyys2")
        concatenated_images = self.image_processor.concat_images(cropped_image1, cropped_image2)
        app.logger.info("im in boyys3")
        
        cropped_pos = self.pos_predictor.predict(concatenated_images)
        app.logger.info("im in boyys4")
        pos = self.image_processor.uncrop_pos(cropped_pos, crop_tform1)
        app.logger.info("im in boyys5")

        self.generate_and_save_obj_from_pos(pos, image1, 'temp.obj')
        app.logger.info("im in boyys6")



        



        
        


            

