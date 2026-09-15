"""Interactive local Open3D viewer with click-to-measure (Phase 5 step 1)."""
from __future__ import annotations

import numpy as np

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


class Open3DViewer:
    """Thin wrapper around Open3D's VisualizerWithVertexSelection for point-picking.

    Usage pattern (must run with a display / not headless):
        viewer = Open3DViewer()
        viewer.load_geometry(pcd_or_mesh)
        picked = viewer.run_and_get_picks()  # blocks until window closed
        # picked is a list of {index, coord} from Open3D's picker
    """

    def __init__(self, window_name: str = "DepthCraft Viewer"):
        self.window_name = window_name
        self._geometry = None

    def load_geometry(self, geometry) -> None:
        self._geometry = geometry

    def run_and_get_picks(self) -> list[dict]:
        import open3d as o3d

        if self._geometry is None:
            raise ValueError("No geometry loaded. Call load_geometry() first.")

        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window(window_name=self.window_name)
        vis.add_geometry(self._geometry)
        log.info("Shift+click two points to select, then close the window to measure.")
        vis.run()
        picks = vis.get_picked_points()
        vis.destroy_window()
        return [{"index": p.index, "coord": np.array(p.coord)} for p in picks]

    @staticmethod
    def render_offscreen(geometry, output_png: str, width: int = 1280, height: int = 720):
        """Headless snapshot -- useful in CI/servers without a display."""
        import open3d as o3d

        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        renderer.scene.add_geometry("geom", geometry, mat)
        renderer.scene.camera.look_at([0, 0, 0], [0, 0, -3], [0, -1, 0])
        img = renderer.render_to_image()
        o3d.io.write_image(output_png, img)
        log.info(f"Rendered offscreen snapshot to {output_png}")
