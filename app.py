from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import base64

app = Flask(__name__)

session = ort.InferenceSession("mask_detector.onnx")
input_name = session.get_inputs()[0].name


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    img_b64 = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_b64)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    face = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Ensure valid face crop
    if face is None or face.size == 0:
        return jsonify({"result": "No Face", "confidence": 0})

    # Preprocess for ONNX
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    preds = session.run(None, {input_name: face})[0][0]
    mask, withoutMask = preds

    result = "Mask" if mask > withoutMask else "No Mask"
    conf = round(float(max(mask, withoutMask)) * 100, 2)

    return jsonify({"result": result, "confidence": conf})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
