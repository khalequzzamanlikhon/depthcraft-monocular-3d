import { Canvas } from "@react-three/fiber";
import { OrbitControls, Points, PointMaterial } from "@react-three/drei";
import { useEffect, useState } from "react";
import * as THREE from "three";

interface Viewer3DProps {
  pointcloudUrl: string;
}

/** Minimal PLY point-cloud viewer. For textured meshes, swap Points for
 * <primitive object={gltf.scene} /> via useGLTF once a .glb is available. */
export default function Viewer3D({ pointcloudUrl }: Viewer3DProps) {
  const [positions, setPositions] = useState<Float32Array | null>(null);

  useEffect(() => {
    let cancelled = false;
    import("three/examples/jsm/loaders/PLYLoader.js").then(({ PLYLoader }) => {
      const loader = new PLYLoader();
      loader.load(pointcloudUrl, (geometry) => {
        if (cancelled) return;
        geometry.computeBoundingSphere();
        setPositions(geometry.attributes.position.array as Float32Array);
      });
    });
    return () => {
      cancelled = true;
    };
  }, [pointcloudUrl]);

  return (
    <div style={{ width: "100%", height: 480, background: "#111" }}>
      <Canvas camera={{ position: [0, 0, 3] }}>
        <OrbitControls />
        <ambientLight intensity={0.8} />
        {positions && (
          <Points positions={positions}>
            <PointMaterial size={0.01} color={new THREE.Color("white")} sizeAttenuation />
          </Points>
        )}
      </Canvas>
    </div>
  );
}
