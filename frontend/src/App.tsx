import { useState } from "react";
import Viewer3D from "./components/Viewer3D";
import SplatViewer from "./components/SplatViewer";
import MeasureTool from "./components/MeasureTool";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [pointcloudUrl, setPointcloudUrl] = useState<string | null>(null);
  const [status, setStatus] = useState("Upload an image to begin.");

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setStatus("Reconstructing...");
    const formData = new FormData();
    formData.append("file", file);

    const resp = await fetch(`${API_URL}/api/v1/reconstruct`, {
      method: "POST",
      body: formData,
    });
    const data = await resp.json();
    setJobId(data.job_id);
    setPointcloudUrl(`${API_URL}${data.pointcloud_url}`);
    setStatus(`Reconstructed ${data.num_points} points. job_id=${data.job_id}`);
  }

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1.5rem" }}>
      <h1>DepthCraft</h1>
      <p>{status}</p>
      <input type="file" accept="image/*" onChange={handleUpload} />

      {pointcloudUrl && (
        <div style={{ marginTop: "1rem" }}>
          <Viewer3D pointcloudUrl={pointcloudUrl} />
          <MeasureTool apiUrl={API_URL} jobId={jobId} />
        </div>
      )}

      {/* Swap in <SplatViewer /> once a job has produced a .splat export */}
      <SplatViewer splatUrl={null} />
    </div>
  );
}
