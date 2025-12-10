const video = document.getElementById("video");
const statusText = document.getElementById("status");
const statusBox = document.getElementById("status-box");
const hudCanvas = document.getElementById("hudCanvas");
const hudCtx = hudCanvas.getContext("2d");

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

let lastStatus = "";

// Main loop
setInterval(() => {
    if (video.videoWidth === 0) return;

    // Resize HUD canvas
    hudCanvas.width = video.videoWidth;
    hudCanvas.height = video.videoHeight;

    let temp = document.createElement("canvas");
    temp.width = video.videoWidth;
    temp.height = video.videoHeight;
    let ctx = temp.getContext("2d");

    ctx.drawImage(video, 0, 0);

    let dataURL = temp.toDataURL("image/jpeg");

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL })
    })
    .then(r => r.json())
    .then(data => {
        let result = data.result;
        let confidence = data.confidence;

        statusText.innerHTML = `${result} (${confidence}%)`;

        // Update color
        statusBox.style.background = 
            result === "Mask" ? "#2ecc71" : "#e74c3c";

        // HUD FRAME
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);

        hudCtx.strokeStyle = result === "Mask" ? "#00ff9d" : "#ff0040";
        hudCtx.lineWidth = 4;
        hudCtx.shadowBlur = 15;
        hudCtx.shadowColor = hudCtx.strokeStyle;

        // Draw Iron-Man style frame
        let w = hudCanvas.width * 0.6;
        let h = hudCanvas.height * 0.65;
        let x = (hudCanvas.width - w) / 2;
        let y = (hudCanvas.height - h) / 2;

        hudCtx.strokeRect(x, y, w, h);

        // Confidence text
        hudCtx.font = "28px Orbitron";
        hudCtx.fillStyle = "#00eaff";
        hudCtx.fillText(`${confidence}%`, x + w/2 - 40, y - 15);

        lastStatus = result;
    });

}, 500);
