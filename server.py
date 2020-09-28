import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import sys
sys.path.append('face-reconstruction')
from predictor import Predictor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 # Max size of content is 2MB (1MB per image)

predictor = Predictor()

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

@app.route('/api/predict', methods = ['POST'] )
@cross_origin(allow_headers=['Content-Type'])
def predict_face():
    if 'image1' not in request.files:
        app.logger.info("image1 not ok")
    if 'image2' not in request.files:
        app.logger.info("image2 not ok")
    image1 = request.files['image1']
    image2 = request.files['image2']
    if not filename_is_valid(image1.filename) or not filename_is_valid(image2.filename):
        app.logger.info("one filename is invalid")
    filename_image1 = secure_filename(image1.filename)
    filename_image2 = secure_filename(image2.filename)
    app.logger.info("going in boyys")
    predictor.predict_face_from_images(image1, image2, app)
    app.logger.info(os.listdir())
    
    return jsonify(filename_image1), 200
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')