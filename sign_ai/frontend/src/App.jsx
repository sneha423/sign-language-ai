import { useEffect, useState } from "react";
import Header from "./components/Header";
import UploadPanel from "./components/UploadPanel";
import CameraPanel from "./components/CameraPanel";
import ResultCard from "./components/ResultCard";
import StatusBar from "./components/StatusBar";

const API_BASE = import.meta.env.VITE_API_BASE_URL;

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState("Checking...");
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => res.json())
      .then(() => setApiStatus("Connected"))
      .catch(() => setApiStatus("Offline"));
  }, []);

  const sendToApi = async (file) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);

      if (data.success) {
        setHistory((prev) => [
          {
            prediction: data.prediction,
            confidence: data.confidence,
            time: new Date().toLocaleTimeString(),
          },
          ...prev.slice(0, 4),
        ]);
      }
    } catch (error) {
      setResult({ success: false, error: "API request failed" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <Header />
      <StatusBar apiStatus={apiStatus} />

      <main className="dashboard-grid">
        <div className="panel-column">
          <UploadPanel onUpload={sendToApi} loading={loading} />
          <CameraPanel onCapture={sendToApi} loading={loading} />
        </div>

        <div className="panel-column">
          <ResultCard result={result} loading={loading} history={history} />
        </div>
      </main>
    </div>
  );
}