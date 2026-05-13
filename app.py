from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model once when server starts
model = load_model("cat_dog_classifier.h5")

def prepare_image(img):
    """Resize and preprocess image for prediction"""
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize
    return img_array

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if image came from file upload or camera (base64)
        if "file" in request.files:
            file = request.files["file"]
            img = Image.open(file.stream).convert("RGB")
        elif request.json and "image" in request.json:
            # base64 image from camera
            img_data = request.json["image"].split(",")[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            return jsonify({"error": "No image provided"}), 400

        img_array = prepare_image(img)
        prediction = model.predict(img_array)

        # If model has single output neuron (sigmoid)
        if prediction.shape[1] == 1:
            confidence = float(prediction[0][0])
            if confidence > 0.5:
                label = "Dog"
                conf_percent = round(confidence * 100, 2)
            else:
                label = "Cat"
                conf_percent = round((1 - confidence) * 100, 2)
        # If model has 2 output neurons (softmax)
        else:
            class_idx = np.argmax(prediction[0])
            labels = ["Cat", "Dog"]
            label = labels[class_idx]
            conf_percent = round(float(prediction[0][class_idx]) * 100, 2)

        return jsonify({
            "label": label,
            "confidence": conf_percent
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
