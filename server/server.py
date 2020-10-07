import os
from flask import Flask, request, redirect, url_for, jsonify, send_from_directory
#from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import redis
import sys
import json
import re
from utils import generate_queue_item, validate_request

UUID_PATTERN = re.compile(r'^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$', re.IGNORECASE)
FOLDER_PATH = 'objs/'
FILE_ENDING = '.obj'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # Max size of content is 10MB (5MB per image)

db = redis.StrictRedis(host="redis", port=6379, charset="utf-8", db=0, decode_responses=True)

@app.route('/api/predict/status/<id>', methods = ['GET'] )
#@cross_origin()
def get_status(id):
    if not UUID_PATTERN.match(id):
        return jsonify('invalid key'), 400
    
    output = db.get(id)
    app.logger.info(str(output))
    if output is None:
        return jsonify('invalid key'), 400

    return jsonify(status=output), 200

@app.route('/api/predict/result/<id>', methods= ['GET'])
#@cross_origin(origin='*', headers=['Content- Type','Authorization'])
def get_result(id):
    if not UUID_PATTERN.match(id):
        return jsonify('invalid key'), 400
    
    output = db.get(id)
    app.logger.info(str(output))
    if output is None:
        return jsonify('invalid key'), 400
    
    if output == 'success':
        return send_from_directory(FOLDER_PATH, 
                                    id + FILE_ENDING, 
                                    as_attachment = True, 
                                    attachment_filename = id + FILE_ENDING)
    return jsonify('invalid key'), 400

@app.route('/api/predict/add', methods = ['POST'] )
#@cross_origin(allow_headers=['Content-Type'])
def predict_face():
    request_is_valid, validation_response = validate_request(request)
    if not request_is_valid:
        return jsonify(status=validation_response), 400
    
    try:
        k, d = generate_queue_item(validation_response, app)
        db.rpush("image_queue", json.dumps(d))
        db.set(k, "queued")
    except:
        return jsonify(status='could not add data to queue'), 500
    
    return jsonify(status='job added to queue', id=k), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')