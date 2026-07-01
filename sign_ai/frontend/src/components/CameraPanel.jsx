import { useEffect, useRef, useState } from "react";

export default function CameraPanel({ onCapture, loading }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraOn, setCameraOn] = useState(false);

  useEffect(() => {
    let stream;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setCameraOn(true);
        }
      } catch (err) {
        setCameraOn(false);
      }
    };

    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const captureImage = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      onCapture(file);
    }, "image/jpeg");
  };

  return (
    <section className="panel">
      <h2>Live Camera</h2>
      <p>Capture a frame from your webcam and predict the sign.</p>
      <video ref={videoRef} autoPlay playsInline className="camera-view" />
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <button className="action-btn" onClick={captureImage} disabled={!cameraOn || loading}>
        {loading ? "Processing..." : "Capture & Predict"}
      </button>
    </section>
  );
}