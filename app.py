from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import base64

app = Flask(__name__)

# ----- ONNX MODEL -----
session = ort.InferenceSession("mask_detector.onnx")
input_name = session.get_inputs()[0].name

# ----- FACE DETECTOR (mediapipe replacement using OpenCV) -----
prototxt = "face_detector/deploy.prototxt"
weights = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxt, weights)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    (h, w) = frame.shape[:2]

    # Face detection (OpenCV DNN)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    best_face = None
    max_conf = 0

    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > max_conf:
            max_conf = conf
            best_face = detections[0, 0, i]

    if best_face is None or max_conf < 0.5:
        return jsonify({"result": "No Face", "confidence": 0})

    box = best_face[3:7] * np.array([w, h, w, h])
    startX, startY, endX, endY = box.astype(int)

    face = frame[startY:endY, startX:endX]
    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    preds = session.run(None, {input_name: face})[0][0]
    mask, without = preds

    result = "Mask" if mask > without else "No Mask"
    conf = round(float(max(mask, without)) * 100, 2)

    return jsonify({
        "result": result,
        "confidence": conf
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
