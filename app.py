from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({
        'message': "test"
    })

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return {'error': 'No image file provided'}, 400

    file = request.files['image']
    file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return {'message': 'Image uploaded successfully'}, 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)