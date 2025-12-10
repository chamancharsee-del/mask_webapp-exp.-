const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusBox = document.getElementById("status-box");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(s => {
        video.srcObject = s;
        statusBox.innerHTML = "Camera Active";
    })
    .catch(err => {
        statusBox.innerHTML = "Camera Blocked";
        console.error(err);
    });

// Mediapipe detector
const faceDetector = new FaceDetection.FaceDetection({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
});
faceDetector.setOptions({ model: "short", minDetectionConfidence: 0.5 });

async function beginHUD() {
    video.addEventListener("loadeddata", () => loopHUD());
}

async function loopHUD() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    await faceDetector.send({ image: video });
    requestAnimationFrame(loopHUD);
}

faceDetector.onResults(async (res) => {
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!res.detections || res.detections.length === 0) {
        statusBox.innerHTML = "No Face Detected";
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

    // Crop face for prediction
    let tmp = document.createElement("canvas");
    tmp.width = w;
    tmp.height = h;
    let tctx = tmp.getContext("2d");
    tctx.drawImage(video, x, y, w, h, 0, 0, w, h);

    let faceData = tmp.toDataURL("image/jpeg");

    let resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: faceData })
    });

    let data = await resp.json();
    let result = data.result;
    let conf = data.confidence;

    // HUD Colors
    const neon = result === "Mask" ? "#00ffaa" : "#ff3355";

    // Draw futuristic glowing box
    ctx.strokeStyle = neon;
    ctx.lineWidth = 4;
    ctx.shadowBlur = 22;
    ctx.shadowColor = neon;
    ctx.strokeRect(x, y, w, h);

    // Label
    ctx.font = "22px Segoe UI";
    ctx.fillStyle = neon;
    ctx.shadowBlur = 0;
    ctx.fillText(`${result} (${conf}%)`, x + 8, y - 10);

    // Update status pill
    statusBox.innerHTML = `${result} (${conf}%)`;
    statusBox.style.background = neon + "33";
});

beginHUD();
