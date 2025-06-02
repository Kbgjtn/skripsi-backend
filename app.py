from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from PIL import Image

app = Flask(__name__)
CORS(app)

# === Load InceptionV4 model ===
CLASS_NAMES = ["Algal Spot", "Brown Blight", "Gray Blight", "Healthy", "Helopeltis", "Red Spot", "Red Rust", "Red Spider Infested", "White Spot"]
IMAGE_SIZE = 299
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
num_classes = 9
model = timm.create_model('inception_v4', pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('InceptionV4.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    img = Image.open(image_file).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).squeeze() 
        topk = torch.topk(probs, k=3)
        indices = topk.indices.tolist()
        confidences = topk.values.tolist()

    predictions = [
        {"class": CLASS_NAMES[idx], "confidence": round(conf * 100, 2)}
        for idx, conf in zip(indices, confidences)
    ]

    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(debug=True, port=5000)