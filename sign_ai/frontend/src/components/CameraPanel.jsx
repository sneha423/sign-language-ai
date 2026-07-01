import { useEffect, useRef, useState } from "react";

export default function CameraPanel({ onCapture, loading }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;

          videoRef.current.onloadedmetadata = async () => {
            try {
              await videoRef.current.play();
              setCameraOn(true);
              setCameraError("");
            } catch (err) {
              setCameraOn(false);
              setCameraError(`Video play failed: ${err.message}`);
              console.error("Video play error:", err);
            }
          };
        }
      } catch (err) {
        setCameraOn(false);
        setCameraError(`${err.name}: ${err.message}`);
        console.error("Camera access error:", err);
      }
    };

    startCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) {
      setCameraError("Camera elements not ready");
      return;
    }

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setCameraError("Camera stream not ready yet");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (!blob) {
        setCameraError("Failed to capture image from camera");
        return;
      }

      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      onCapture(file);
    }, "image/jpeg");
  };

  return (
    <section className="panel">
      <h2>Live Camera</h2>
      <p>Capture a frame from your webcam and predict the sign.</p>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="camera-view"
      />

      <canvas ref={canvasRef} style={{ display: "none" }} />

      {cameraError && (
        <p style={{ color: "red", marginTop: "10px" }}>{cameraError}</p>
      )}

      <button
        className="action-btn"
        onClick={captureImage}
        disabled={!cameraOn || loading}
      >
        {loading ? "Processing..." : "Capture & Predict"}
      </button>
    </section>
  );
}