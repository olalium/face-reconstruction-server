import os

import re
import base64
import logging
import sys
import json
import numpy as np

IMAGE_SHAPE = (512, 512, 3)
POS_SHAPE = (256, 256, 3)
UUID_PATTERN = re.compile(r'^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$', re.IGNORECASE)

# TODO add json validation schema
def load_and_decode_data(data, db):
    try:
        json_data = json.loads(data)
        id = json_data["id"]
        assert UUID_PATTERN.match(id)
        images = json_data["images"]
        assert len(images) == 2

        decoded_images = []
        for image in images:
            decoded_image = base64_decode_image(image)
            decoded_images.append(decoded_image)
        return decoded_images, id
    except:
        if 'images' not in locals():
            pass
        elif 'decoded_images' not in locals():
            db.set(id, "error_json")
        else:
            db.set(id, "error_decode")
        return False, ''

def base64_decode_image(a):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")
    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=np.uint8)
    a = a.reshape(IMAGE_SHAPE)
    # return the decoded image
    return a

def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1  # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:

        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0],
                                               colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)