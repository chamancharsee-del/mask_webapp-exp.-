from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import base64

app = Flask(__name__)

# -----------------------------
# LOAD ONNX MODEL
# -----------------------------
session = ort.InferenceSession("mask_detector.onnx")
input_name = session.get_inputs()[0].name

# -----------------------------
# LOAD FACE DETECTOR (OpenCV)
# -----------------------------
prototxt = "face_detector/deploy.prototxt"
weights = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxt, weights)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img_b64 = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_b64)

    # Decode image
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    (h, w) = frame.shape[:2]

    # -----------------------------
    # FACE DETECTION (OpenCV DNN)
    # -----------------------------
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    best_conf = 0
    best_face = None

    # Pick strongest detection
    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > best_conf:
            best_conf = conf
            best_face = detections[0, 0, i]

    if best_face is None or best_conf < 0.5:
        return jsonify({"result": "No Face", "confidence": 0})

    # Extract bounding box
    box = best_face[3:7] * np.array([w, h, w, h])
    startX, startY, endX, endY = box.astype(int)

    # Fix bounds
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(w - 1, endX)
    endY = min(h - 1, endY)

    # -----------------------------
    # CROP FACE SAFELY
    # -----------------------------
    face = frame[startY:endY, startX:endX]

    # Ensure valid crop
    if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
        return jsonify({"result": "No Face", "confidence": 0})

    # -----------------------------
    # PREPROCESS FOR ONNX
    # -----------------------------
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)          # IMPORTANT FIX
    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    # -----------------------------
    # RUN MODEL
    # -----------------------------
    preds = session.run(None, {input_name: face})[0][0]
    mask, withoutMask = preds

    result = "Mask" if mask > withoutMask else "No Mask"
    confidence = round(float(max(mask, withoutMask)) * 100, 2)

    # -----------------------------
    # RETURN RESULT
    # -----------------------------
    return jsonify({
        "result": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
