const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusBox = document.getElementById("status-box");

// Start camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        statusBox.innerHTML = "Camera Active";
    })
    .catch(err => {
        statusBox.innerHTML = "Camera Blocked";
        console.error(err);
    });

const faceDetector = new FaceDetection.FaceDetection({
    locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
});
faceDetector.setOptions({ model: "short", minDetectionConfidence: 0.5 });

async function loop() {
    await faceDetector.send({ image: video });
    requestAnimationFrame(loop);
}

faceDetector.onResults(async (res) => {
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!res.detections || res.detections.length === 0) {
        statusBox.innerHTML = "No Face";
        statusBox.style.background = "#113344";
        return;
    }

    let b = res.detections[0].boundingBox;
    let vw = video.videoWidth;
    let vh = video.videoHeight;

    let x = b.xCenter * vw - (b.width * vw) / 2;
    let y = b.yCenter * vh - (b.height * vh) / 2;
    let w = b.width * vw;
    let h = b.height * vh;

    // Draw HUD box
    const neon = "#00eaff";
    ctx.lineWidth = 4;
    ctx.strokeStyle = neon;
    ctx.shadowBlur = 20;
    ctx.shadowColor = neon;
    ctx.strokeRect(x, y, w, h);

    // Crop EXACT face area from video
    let tempCanvas = document.createElement("canvas");
    tempCanvas.width = w;
    tempCanvas.height = h;
    tempCanvas.getContext("2d").drawImage(video, x, y, w, h, 0, 0, w, h);

    let faceImage = tempCanvas.toDataURL("image/jpeg");

    // Send ONLY the cropped face to backend
    let resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: faceImage })
    });

    let data = await resp.json();

    statusBox.innerHTML = `${data.result} (${data.confidence}%)`;
    statusBox.style.background =
        data.result === "Mask" ? "#0f9" : "#f33";
});

video.addEventListener("loadeddata", loop);
