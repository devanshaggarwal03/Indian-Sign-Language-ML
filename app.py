from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import math

app = Flask(__name__)
model = load_model("Model/keras_model.h5")
labels = ["A", "B", "C", "D", "E", "F", "G"]
imgSize = 300

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
    image = np.array(image)

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    h, w, _ = image.shape
    aspectRatio = h / w

    try:
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(image, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(image, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        inputSize = model.input_shape[1:3]
        imgInput = cv2.resize(imgWhite, inputSize)
        imgInput = imgInput / 255.0
        imgInput = np.expand_dims(imgInput, axis=0)

        if len(model.input_shape) == 2:
            imgInput = imgInput.reshape(1, -1)

        prediction = model.predict(imgInput)
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]

        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
