interface SplatViewerProps {
  splatUrl: string | null;
}

/** Placeholder wiring for gsplat.js. Once a .splat export exists (see
 * scripts/export_splat.py), install `@mkkellogg/gaussian-splats-3d` (or
 * antimatter15/splat's viewer) and mount its canvas here, e.g.:
 *
 *   import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
 *   const viewer = new GaussianSplats3D.Viewer({ rootElement: containerRef.current });
 *   viewer.addSplatScene(splatUrl).then(() => viewer.start());
 */
export default function SplatViewer({ splatUrl }: SplatViewerProps) {
  if (!splatUrl) {
    return (
      <div style={{ padding: "1rem", color: "#888" }}>
        No Gaussian Splat scene yet — train one via scripts/train_3dgs.py, export
        with scripts/export_splat.py, then pass its URL here.
      </div>
    );
  }
  return <div id="splat-viewer-root" style={{ width: "100%", height: 480 }} />;
}
