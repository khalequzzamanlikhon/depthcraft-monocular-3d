"""TSDF volumetric fusion via Open3D's ScalableTSDFVolume (Phase 3 step 4).

Fully real implementation -- this is exactly what the spec says: don't
write TSDF from scratch, use Open3D's battle-tested integrator.
"""
from __future__ import annotations

import numpy as np

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


class TSDFFuser:
    def __init__(self, voxel_length: float = 0.01, sdf_trunc: float = 0.05):
        import open3d as o3d

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self._o3d = o3d

    def integrate_frame(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        pose_w2c: np.ndarray,
        depth_scale: float = 1.0,
        depth_trunc: float = 6.0,
    ) -> None:
        """Integrate a single RGB-D frame given its metric depth and world-to-camera pose."""
        o3d = self._o3d
        h, w = depth.shape
        color_img = o3d.geometry.Image(np.ascontiguousarray(color.astype(np.uint8)))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1.0 / depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        intr = o3d.camera.PinholeCameraIntrinsic(
            w, h, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        )
        self.volume.integrate(rgbd, intr, pose_w2c)

    def integrate_sequence(self, frames: list[dict], intrinsics: np.ndarray) -> None:
        """frames: list of {'color': HxWx3 uint8, 'depth': HxW float32 (m), 'pose_w2c': 4x4}"""
        for i, f in enumerate(frames):
            self.integrate_frame(f["color"], f["depth"], intrinsics, f["pose_w2c"])
            if (i + 1) % 10 == 0:
                log.info(f"Integrated {i + 1}/{len(frames)} frames into TSDF volume")

    def extract_pointcloud(self):
        return self.volume.extract_point_cloud()

    def extract_mesh(self):
        return self.volume.extract_triangle_mesh()
