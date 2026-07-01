import { useEffect, useRef, useState } from "react";

export default function CameraPanel({ onCapture, loading }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [startingCamera, setStartingCamera] = useState(false);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
  };

  const startCamera = async () => {
    setCameraError("");
    setStartingCamera(true);

    try {
      stopCamera();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
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
          } finally {
            setStartingCamera(false);
          }
        };
      } else {
        setStartingCamera(false);
      }
    } catch (err) {
      setCameraOn(false);
      setStartingCamera(false);

      if (err.name === "AbortError") {
        setCameraError(
          "Camera start timed out. Close other apps using the camera and try again."
        );
      } else if (err.name === "NotAllowedError") {
        setCameraError(
          "Camera permission denied. Please allow camera access in the browser."
        );
      } else if (err.name === "NotFoundError") {
        setCameraError("No camera device found.");
      } else if (err.name === "NotReadableError") {
        setCameraError(
          "Camera is busy or unavailable. Close other apps using the camera."
        );
      } else {
        setCameraError(`${err.name}: ${err.message}`);
      }

      console.error("Camera access error:", err);
    }
  };

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

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setCameraError("Failed to capture image from camera");
          return;
        }

        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        onCapture(file);
      },
      "image/jpeg",
      0.95
    );
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

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

      {!cameraOn ? (
        <button
          className="action-btn"
          onClick={startCamera}
          disabled={startingCamera}
        >
          {startingCamera ? "Starting Camera..." : "Start Camera"}
        </button>
      ) : (
        <>
          <button
            className="action-btn"
            onClick={captureImage}
            disabled={loading}
          >
            {loading ? "Processing..." : "Capture & Predict"}
          </button>

          <button
            className="action-btn"
            onClick={stopCamera}
            style={{ marginTop: "10px" }}
          >
            Stop Camera
          </button>
        </>
      )}
    </section>
  );
}