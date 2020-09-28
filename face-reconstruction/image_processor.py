import dlib
import numpy as np
from skimage.transform import rescale, estimate_transform, warp

FACE_DETECTOR_PATH = 'Data/net-data/mmod_human_face_detector.dat'
SHAPE_PREDICTOR_PATH = 'Data/net-data/shape_predictor_68_face_landmarks.dat'

class ImageProcessor(object):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.face_detector = dlib.cnn_face_detection_model_v1(FACE_DETECTOR_PATH)
        self.shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    def uncrop_pos(self, cropped_pos, cropping_tform):
        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2, :].copy() / cropping_tform.params[0, 0]
        cropped_vertices[2, :] = 1
        vertices = np.dot(np.linalg.inv(cropping_tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2, :], z))
        pos = np.reshape(vertices.T, [256, 256, 3])
        return pos

    def clean_image(self, image):
        image = image[:,:,:3]
        if image.shape != (256, 256, 3):
            max_size = max(image.shape[0], image.shape[1])
            if max_size > 1000:
                image = rescale(image, 1000. / max_size)
                image = (image * 255).astype(np.uint8)
            image = np.around(image, decimals=1).astype(np.uint8)
        return image
    
    def get_cropping_transform(self, image):
        detected_faces = self.face_detector(image, 1)
        if len(detected_faces) == 0:
            print('warning: no detected face')
            return None

        d = detected_faces[0].rect  ## only use the first detected face (assume that each input image only contains one face)
        left = d.left()
        right = d.right()
        top = d.top()
        bottom = d.bottom()
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
        size = int(old_size * 1.58)

        shape = self.shape_predictor(image, d)
        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, 255], [255, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        return tform

    def crop_image(self, image, cropping_tform):
        float_img = image / 256.0 / 1.1
        if not cropping_tform:
            return float_img
        else:
            return warp(float_img, cropping_tform.inverse, output_shape=(256, 256))
            
    def get_cropped_image(self, image):
        image = self.clean_image(image)
        crop_tform = self.get_cropping_transform(image)
        if crop_tform is None:
            return image, None
        cropped_image = self.crop_image(image, crop_tform)
        return cropped_image, crop_tform
    
    def concat_images(self, image1, image2):
        concat_images = np.concatenate((image1, image2), axis=2)
        return concat_images
  