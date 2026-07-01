import { useState } from "react";

export default function UploadPanel({ onUpload, loading }) {
  const [fileName, setFileName] = useState("");

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
      onUpload(file);
    }
  };

  return (
    <section className="panel">
      <h2>Image Upload</h2>
      <p>Select a hand-sign image from your system.</p>
      <label className="upload-box">
        <input type="file" accept="image/*" onChange={handleChange} hidden />
        <span>{fileName || (loading ? "Uploading..." : "Choose image")}</span>
      </label>
    </section>
  );
}