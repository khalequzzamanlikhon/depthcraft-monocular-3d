import { useState } from "react";

interface MeasureToolProps {
  apiUrl: string;
  jobId: string | null;
}

/** Click-to-measure UI: collects two clicks (as normalized image-space
 * coordinates from the depth/reconstruction source image) and calls
 * POST /api/v1/measure. Wire the click coordinates up to your actual
 * <img> or canvas click handler for the source photo. */
export default function MeasureTool({ apiUrl, jobId }: MeasureToolProps) {
  const [pointA, setPointA] = useState({ x: 100, y: 100 });
  const [pointB, setPointB] = useState({ x: 300, y: 300 });
  const [result, setResult] = useState<string>("");

  async function measure() {
    if (!jobId) return;
    const resp = await fetch(`${apiUrl}/api/v1/measure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, point_a: pointA, point_b: pointB }),
    });
    if (!resp.ok) {
      setResult(`Error: ${resp.status} ${await resp.text()}`);
      return;
    }
    const data = await resp.json();
    setResult(
      `Distance: ${data.distance_m.toFixed(2)} m ± ${data.uncertainty_1sigma_m.toFixed(2)} m ` +
        `(95% CI ± ${data.uncertainty_95ci_m.toFixed(2)} m)`
    );
  }

  return (
    <div style={{ marginTop: "1rem" }}>
      <h3>Measure</h3>
      <label>
        Point A (px):
        <input
          type="number"
          value={pointA.x}
          onChange={(e) => setPointA({ ...pointA, x: Number(e.target.value) })}
        />
        <input
          type="number"
          value={pointA.y}
          onChange={(e) => setPointA({ ...pointA, y: Number(e.target.value) })}
        />
      </label>
      <label style={{ marginLeft: "1rem" }}>
        Point B (px):
        <input
          type="number"
          value={pointB.x}
          onChange={(e) => setPointB({ ...pointB, x: Number(e.target.value) })}
        />
        <input
          type="number"
          value={pointB.y}
          onChange={(e) => setPointB({ ...pointB, y: Number(e.target.value) })}
        />
      </label>
      <button onClick={measure} style={{ marginLeft: "1rem" }}>
        Measure
      </button>
      <p>{result}</p>
    </div>
  );
}
