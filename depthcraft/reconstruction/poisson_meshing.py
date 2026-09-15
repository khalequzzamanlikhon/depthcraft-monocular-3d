"""Poisson surface reconstruction + cleanup (Phase 4 Path A step 3)."""
from __future__ import annotations

import numpy as np

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


def poisson_reconstruct(pcd, depth: int = 10, density_quantile_cutoff: float = 0.01):
    """Run Poisson reconstruction and strip low-density (unreliable) vertices.

    depth: octree depth, 8-12 per spec (higher = more detail, more noise-sensitive).
    """
    import open3d as o3d

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)

    cutoff = np.quantile(densities, density_quantile_cutoff)
    vertices_to_remove = densities < cutoff
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh = remove_disconnected_components(mesh)
    return mesh


def remove_disconnected_components(mesh, min_triangle_fraction: float = 0.02):
    """Drop small disconnected mesh islands (noise blobs) below a size fraction."""
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    min_triangles = max(1, int(min_triangle_fraction * len(mesh.triangles)))
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_triangles
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh


def export_mesh(mesh, path: str) -> None:
    import open3d as o3d

    o3d.io.write_triangle_mesh(path, mesh)
    log.info(f"Exported mesh to {path}")
