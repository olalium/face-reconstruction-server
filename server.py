import os
from flask import Flask, request, redirect, url_for, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from skimage.io import imread, imsave
import redis
import sys
import uuid
import json
sys.path.append('face-reconstruction')
from utils import base64_encode_image, base64_decode_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 # Max size of content is 2MB (1MB per image)

db = redis.StrictRedis(host="redis", port=6379, db=0)

UPLOAD_FOLDER = '/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def filename_is_valid(filename):
    if filename == '':
        return False
    if not allowed_file(filename):
        return False
    return True

def evaluate_files(files):
    if 'image1' not in files or 'image2' not in files:
        return False, jsonify('image not in request')
    
    file1 = files['image1']
    file2 = files['image2']

    if not filename_is_valid(file1.filename) or not filename_is_valid(file2.filename):
        return False, jsonify('filename is not valid')
    return True, [file1, file2]

@app.route('/api/predict/result/<id>', methods = ['GET'] )
@cross_origin()
def get_results(id):
    k = id
    app.logger.info(k)
    output = db.get(k)
    print(output)
    if output is None:
        return jsonify('invalid key')
    return output

@app.route('/api/predict', methods = ['POST'] )
@cross_origin(allow_headers=['Content-Type'])
def predict_face():
    evaluation = evaluate_files(request.files)
    if not evaluation[0]:
        return jsonify(evaluation[1]), 400
    
    image1 = imread(evaluation[1][0])
    image2 = imread(evaluation[1][1])

    k = str(uuid.uuid4())
    d = {
        "id": k,
        "images": [
            base64_encode_image(image1),
            base64_encode_image(image2)
        ]
    }
    db.rpush("image_queue", json.dumps(d))

    return jsonify('job added to queue', k), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')