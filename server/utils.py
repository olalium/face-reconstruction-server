import base64
import sys
import uuid

from skimage.transform import resize, rescale
from skimage.io import imread, imsave
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SHAPE = (512, 512, 3)

def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def filename_is_valid(filename):
    if filename == '':
        return False
    if not allowed_file(filename):
        return False
    return True

def validate_request(request):
    try:
        files = request.files
    except:
        return False, 'files not in request'

    if 'image1' not in files or 'image2' not in files:
        return False, 'image(s) not in request'

    files = [files['image1'], files['image2']]
    images = []
    for file in files:
        if not filename_is_valid(file.filename):
            return False, 'filename is not valid'
        image = imread(file)
        if len(image.shape) != 3:
            return False, 'wrong image shape'
        if image.shape[2] != 3:
            return False, 'wrong number of image channels'
        images.append(image)

    return True, images

def generate_queue_item(images):
    image0 = resize_image(images[0])
    image1 = resize_image(images[1])

    encoded_image0 = base64_encode_image(image0)
    encoded_image1 = base64_encode_image(image1)

    k = str(uuid.uuid4())
    d = {
        "id": k,
        "images": [
            encoded_image0,
            encoded_image1
        ]
    }
    return k, d

def resize_image(image):
    image_resized = np.zeros(IMAGE_SHAPE, np.uint8)

    max_size = max(image.shape[0], image.shape[1])
    image_rescaled = rescale(image, 512. / max_size, multichannel=True, anti_aliasing=True)
    image_rescaled = (image_rescaled * 255).astype(np.uint8)
    image_rescaled = np.around(image_rescaled, decimals=1).astype(np.uint8)
    image_resized[:image_rescaled.shape[0],:image_rescaled.shape[1]] = image_rescaled[:,:]
    return image_resized