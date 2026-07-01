export default function StatusBar({ apiStatus }) {
  return (
    <div className="status-bar">
      <span className={`status-dot ${apiStatus === "Connected" ? "online" : "offline"}`}></span>
      <span>Backend: {apiStatus}</span>
    </div>
  );
}