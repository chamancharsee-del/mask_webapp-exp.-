const video = document.getElementById("video");
const statusText = document.getElementById("status");
const statusBox = document.getElementById("status-box");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        statusText.innerHTML = "Camera Active";
    })
    .catch(err => {
        statusText.innerHTML = "Camera Blocked";
        console.error(err);
    });


// Setup Mediapipe Face Detector
let faceDetector = new FaceDetection.FaceDetection({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
    }
});

faceDetector.setOptions({
    model: 'short',
    minDetectionConfidence: 0.5
});

faceDetector.onResults(gotFaces);

let camera = new Camera(video, {
    onFrame: async () => {
        await faceDetector.send({ image: video });
    },
    width: 640,
    height: 480
});
camera.start();


let currentFace = null;
let lastStatus = "";

function gotFaces(results) {
    if (results.detections.length === 0) {
        currentFace = null;
        return;
    }

    currentFace = results.detections[0];
}


// Send crop to backend every 500ms
setInterval(() => {
    if (!currentFace) {
        statusText.innerHTML = "No Face Detected";
        statusBox.style.background = "#444";
        return;
    }

    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    let ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    let dataURL = canvas.toDataURL("image/jpeg");

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL })
    })
    .then(r => r.json())
    .then(data => {
        let result = data.result;
        let conf = data.confidence;

        statusText.innerHTML = `${result} (${conf}%)`;

        if (result !== lastStatus) {
            statusBox.style.background =
                result === "Mask" ? "#2ecc71" : "#e74c3c";
            lastStatus = result;
        }
    });

}, 500);
