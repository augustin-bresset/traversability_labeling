from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class TraversabilityLabeler:
    """
    Labels point cloud points as traversable based on the robot trajectory.

    A point is traversable if the robot's footprint passed over it
    (within height_range) at any step of the trajectory range [traj_lo, traj_hi).

    Key rule: a point observed at scan time k should only be labeled traversable
    by poses AFTER k (i.e. traj_lo >= k+1).  Using past poses would cause dynamic
    objects (people, vehicles) that happen to be at a former robot position to be
    incorrectly labeled as traversable.

    Optional forward accumulation (forward_accum=True):
    When building the accumulated cloud from past scans, only keep points that
    were observed *ahead* of the robot at their respective scan time (positive
    projection onto the direction of motion at that scan).  This prevents points
    from people or objects following behind the vehicle from ever entering the
    accumulated cloud, complementing the future-only labeling constraint.
    The filtering is applied externally (in the accumulation loop) via the
    `forward_mask` helper; the labeling methods themselves are unchanged.
    """

    def __init__(
        self,
        robot_shape: str = "square",
        robot_size: float = 1.0,
        height_min: float = -0.5,
        height_max: float = 0.3,
        trajectory_window: int = 100,
        forward_accum: bool = False,
    ):
        if robot_shape not in ("square", "round"):
            raise ValueError(f"Unknown robot shape '{robot_shape}'. Use 'square' or 'round'.")
        self.robot_shape = robot_shape
        self.half_size = robot_size / 2.0
        self.height_min = height_min
        self.height_max = height_max
        self.trajectory_window = trajectory_window
        self.forward_accum = forward_accum

    # ------------------------------------------------------------------
    # Forward-accumulation helper
    # ------------------------------------------------------------------

    def forward_mask(
        self,
        xyz: np.ndarray,
        poses: List[np.ndarray],
        scan_idx: int,
    ) -> np.ndarray:
        """
        Boolean mask selecting points that were *ahead* of the robot at scan_idx.

        xyz must be in scan_idx's local frame (robot at the origin).
        "Ahead" means a positive projection onto the direction of motion,
        estimated from the adjacent pose (next if available, previous otherwise).

        Returns an all-True mask if the direction cannot be determined (e.g.
        single-scan sequence), so the caller never silently drops all points.
        """
        T_scan_world = np.linalg.inv(poses[scan_idx])
        if scan_idx + 1 < len(poses):
            other = (T_scan_world @ poses[scan_idx + 1])[:3, 3]
        elif scan_idx > 0:
            # Use reversed previous pose as a proxy for forward
            other = -(T_scan_world @ poses[scan_idx - 1])[:3, 3]
        else:
            return np.ones(len(xyz), dtype=bool)

        fwd = other[:2]
        norm = np.linalg.norm(fwd)
        if norm < 1e-6:
            return np.ones(len(xyz), dtype=bool)
        fwd = fwd / norm

        return (xyz[:, :2] @ fwd) > 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _in_footprint(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        if self.robot_shape == "round":
            return dx ** 2 + dy ** 2 < self.half_size ** 2
        return (np.abs(dx) < self.half_size) & (np.abs(dy) < self.half_size)

    def _label_range(
        self,
        xyz: np.ndarray,
        poses: List[np.ndarray],
        reference_idx: int,
        lo: int,
        hi: int,
    ) -> np.ndarray:
        """Core labeling: check trajectory poses in [lo, hi) against xyz."""
        labels = np.zeros(len(xyz), dtype=np.uint8)

        height_mask = (xyz[:, 2] >= self.height_min) & (xyz[:, 2] <= self.height_max)
        candidates = np.where(height_mask)[0]
        if len(candidates) == 0 or lo >= hi:
            return labels

        T_scan_world = np.linalg.inv(poses[reference_idx])
        xyz_c = xyz[candidates]

        for k in range(lo, hi):
            T_scan_k = T_scan_world @ poses[k]
            traj_xy = T_scan_k[:2, 3]
            dx = xyz_c[:, 0] - traj_xy[0]
            dy = xyz_c[:, 1] - traj_xy[1]
            labels[candidates[self._in_footprint(dx, dy)]] = 1

        return labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_scan(
        self,
        xyz: np.ndarray,
        poses: List[np.ndarray],
        current_idx: int,
        traj_lo: Optional[int] = None,
        traj_hi: Optional[int] = None,
    ) -> np.ndarray:
        """
        Label a single scan's points.

        Args:
            xyz:         (N, 3) scan points in current_idx's local frame.
            poses:       List of 4×4 world-frame poses.
            current_idx: Index of the current scan.
            traj_lo:     First trajectory pose to check (default: current_idx + 1).
            traj_hi:     One-past-last pose to check (default: current_idx + window + 1).

        Returns:
            labels: (N,) uint8 - 1 = traversable, 0 = not traversable.

        Note:
            The default traj_lo is current_idx + 1 (strictly future poses only).
            This prevents dynamic objects that happen to be at a former robot
            position from being labeled as traversable.
        """
        lo = traj_lo if traj_lo is not None else current_idx + 1
        hi = traj_hi if traj_hi is not None else min(len(poses), current_idx + self.trajectory_window + 1)
        return self._label_range(xyz, poses, current_idx, lo, hi)

    def label_accumulated(
        self,
        xyz_acc: np.ndarray,
        scan_origins: np.ndarray,
        poses: List[np.ndarray],
        current_idx: int,
    ) -> np.ndarray:
        """
        Label an accumulated point cloud where points come from multiple scans.

        Each point i came from scan scan_origins[i].  It is labeled traversable
        only by poses AFTER its origin scan, i.e. poses in
        [scan_origins[i] + 1, current_idx + window + 1).

        This correctly:
          - Labels terrain from past scans that the robot subsequently drove over.
          - Avoids labeling people/objects at former robot positions as traversable.

        When forward_accum=True the caller is expected to have already filtered
        xyz_acc via forward_mask before calling this method, so that only
        forward-observed points from each past scan are present.

        Args:
            xyz_acc:      (N, 3) accumulated points, all in current_idx's frame.
            scan_origins: (N,) int array - origin scan index for each point.
            poses:        List of 4×4 world-frame poses.
            current_idx:  Reference scan (defines the coordinate frame).

        Returns:
            labels: (N,) uint8.
        """
        labels = np.zeros(len(xyz_acc), dtype=np.uint8)
        hi_global = min(len(poses), current_idx + self.trajectory_window + 1)

        for origin_k in np.unique(scan_origins):
            mask = scan_origins == origin_k
            lo = int(origin_k) + 1          # only poses AFTER the point was observed
            hi = hi_global
            if lo >= hi:
                continue
            labels[mask] = self._label_range(
                xyz_acc[mask], poses, current_idx, lo, hi
            )

        return labels
