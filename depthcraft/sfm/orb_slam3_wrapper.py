"""Real-time SLAM fast path (ORB-SLAM3) for live webcam demos.

ORB-SLAM3 is a C++ library; the common integration pattern is via its
Python bindings (pyorbslam3-style wheels are unofficial/community-built) or
by shelling out to a compiled binary that dumps a trajectory + sparse map.
This wrapper defines the interface DepthCraft expects and a subprocess-based
implementation, since a pure-pip install is not standardized.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


class ORBSlam3Wrapper:
    def __init__(self, orb_slam3_binary: str = "orb_slam3_mono", vocab_path: str = "", config_path: str = ""):
        self.binary = orb_slam3_binary
        self.vocab_path = vocab_path
        self.config_path = config_path

    def is_available(self) -> bool:
        from shutil import which

        return which(self.binary) is not None

    def run(self, video_path: str | Path, output_dir: str | Path) -> dict:
        """Runs the ORB-SLAM3 binary and parses its output trajectory (TUM format).

        Expects the binary to accept: <binary> <vocab> <config> <video> <output_dir>
        and write `CameraTrajectory.txt` (TUM format: t x y z qx qy qz qw) plus
        `sparse_map.ply`. Adjust to match your local ORB-SLAM3 build's CLI.
        """
        if not self.is_available():
            raise RuntimeError(
                f"'{self.binary}' not found on PATH. Build ORB-SLAM3 and expose its "
                "monocular example binary, or use colmap_wrapper.py for offline SfM."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [self.binary, self.vocab_path, self.config_path, str(video_path), str(output_dir)]
        log.info(f"Running ORB-SLAM3: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        traj_path = output_dir / "CameraTrajectory.txt"
        poses = _parse_tum_trajectory(traj_path) if traj_path.exists() else []
        return {"poses": poses, "sparse_map": str(output_dir / "sparse_map.ply")}


def _parse_tum_trajectory(path: Path) -> list[dict]:
    poses = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t, x, y, z, qx, qy, qz, qw = map(float, line.split())
        poses.append({"t": t, "translation": [x, y, z], "quaternion": [qx, qy, qz, qw]})
    return poses
