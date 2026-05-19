"""
Point-to-point ICP for computing sequence poses when no ground-truth
odometry is available (icp_required: True in config).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def point_to_point_icp(
    src: np.ndarray,
    dst: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    max_correspondence_dist: Optional[float] = None,
) -> np.ndarray:
    """
    Point-to-point ICP. Returns T such that T @ src ≈ dst.

    Args:
        src: (N, 3) source point cloud.
        dst: (M, 3) destination point cloud.

    Returns:
        T: (4, 4) rigid transformation.
    """
    from scipy.spatial import cKDTree

    T = np.eye(4, dtype=np.float64)
    src_h = np.ones((len(src), 4), dtype=np.float64)
    src_h[:, :3] = src.astype(np.float64)

    dst_tree = cKDTree(dst.astype(np.float64))

    for _ in range(max_iterations):
        src_current = (T @ src_h.T).T[:, :3]
        dists, indices = dst_tree.query(src_current, k=1, workers=-1)

        threshold = max_correspondence_dist or (3.0 * np.median(dists))
        inliers = dists < threshold
        if inliers.sum() < 6:
            break

        src_in = src_current[inliers]
        dst_in = dst[indices[inliers]].astype(np.float64)

        src_c = src_in.mean(axis=0)
        dst_c = dst_in.mean(axis=0)
        H = (src_in - src_c).T @ (dst_in - dst_c)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = dst_c - R @ src_c

        delta_T = np.eye(4, dtype=np.float64)
        delta_T[:3, :3] = R
        delta_T[:3, 3] = t
        T = delta_T @ T

        angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        if np.linalg.norm(t) + abs(angle) < tolerance:
            break

    return T


def compute_sequence_poses(
    scans: List[np.ndarray],
    max_iterations: int = 50,
    voxel_size: Optional[float] = 0.5,
) -> List[np.ndarray]:
    """
    Compute cumulative world-frame poses for a scan sequence via pairwise ICP.

    Args:
        scans:       List of (N, 3) point arrays (xyz only).
        voxel_size:  If set, down-sample each cloud before ICP.

    Returns:
        poses: List of (4, 4) matrices; poses[0] = identity.
    """
    poses = [np.eye(4, dtype=np.float64)]

    for i in range(1, len(scans)):
        src = scans[i]
        dst = scans[i - 1]

        if voxel_size is not None:
            src = _voxel_downsample(src, voxel_size)
            dst = _voxel_downsample(dst, voxel_size)

        T_rel = point_to_point_icp(src, dst, max_iterations=max_iterations)
        poses.append(poses[-1] @ T_rel)

    return poses


def _voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    keys = np.floor(xyz / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return xyz[unique_idx]
