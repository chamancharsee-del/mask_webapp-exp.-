from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import base64

app = Flask(__name__)

# Load ONNX model
session = ort.InferenceSession("mask_detector.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Load OpenCV face detector
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Decode base64 → image
    img_b64 = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_b64)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"result": "No Face", "confidence": 0})

    (h, w) = frame.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    max_conf = 0
    final_label = "No Face"

    # Loop through detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            face = frame[startY:endY, startX:endX]

            if face is None or face.size == 0:
                continue

            # ONNX preprocessing
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_norm = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_norm, axis=0)

            preds = session.run(None, {input_name: face_input})[0][0]
            mask, withoutMask = preds

            label = "Mask" if mask > withoutMask else "No Mask"
            conf = float(max(mask, withoutMask))

            if conf > max_conf:
                max_conf = conf
                final_label = label

    if max_conf == 0:
        return jsonify({"result": "No Face", "confidence": 0})

    return jsonify({
        "result": final_label,
        "confidence": round(max_conf * 100, 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
