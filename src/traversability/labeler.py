from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _label_core(xyz, poses_arr, T_inv, lo, hi, half_size, height_min, height_max, is_round):
        N = xyz.shape[0]
        labels = np.zeros(N, dtype=np.uint8)
        active = np.empty(N, dtype=np.int64)
        n_active = 0
        for i in range(N):
            if xyz[i, 2] >= height_min and xyz[i, 2] <= height_max:
                active[n_active] = i
                n_active += 1
        for k in range(lo, hi):
            if n_active == 0:
                break
            # Compute only the 6 scalars needed from T = T_inv @ poses_arr[k]
            # (rows 0 and 1, columns 0, 1, 3) — avoids scipy dependency for np.dot
            P = poses_arr[k]
            tx  = T_inv[0,0]*P[0,3] + T_inv[0,1]*P[1,3] + T_inv[0,2]*P[2,3] + T_inv[0,3]*P[3,3]
            ty  = T_inv[1,0]*P[0,3] + T_inv[1,1]*P[1,3] + T_inv[1,2]*P[2,3] + T_inv[1,3]*P[3,3]
            r00 = T_inv[0,0]*P[0,0] + T_inv[0,1]*P[1,0] + T_inv[0,2]*P[2,0] + T_inv[0,3]*P[3,0]
            r10 = T_inv[1,0]*P[0,0] + T_inv[1,1]*P[1,0] + T_inv[1,2]*P[2,0] + T_inv[1,3]*P[3,0]
            r01 = T_inv[0,0]*P[0,1] + T_inv[0,1]*P[1,1] + T_inv[0,2]*P[2,1] + T_inv[0,3]*P[3,1]
            r11 = T_inv[1,0]*P[0,1] + T_inv[1,1]*P[1,1] + T_inv[1,2]*P[2,1] + T_inv[1,3]*P[3,1]
            new_n = 0
            for j in range(n_active):
                idx = active[j]
                dx = xyz[idx, 0] - tx
                dy = xyz[idx, 1] - ty
                dxr = dx * r00 + dy * r10
                dyr = dx * r01 + dy * r11
                if is_round:
                    hit = dxr * dxr + dyr * dyr < half_size * half_size
                else:
                    hit = dxr > -half_size and dxr < half_size and dyr > -half_size and dyr < half_size
                if hit:
                    labels[idx] = 1
                else:
                    active[new_n] = idx
                    new_n += 1
            n_active = new_n
        return labels

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False


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
        lidar_range: Optional[float] = None,
    ):
        if robot_shape not in ("square", "round"):
            raise ValueError(f"Unknown robot shape '{robot_shape}'. Use 'square' or 'round'.")
        self.robot_shape = robot_shape
        self.half_size = robot_size / 2.0
        self.height_min = height_min
        self.height_max = height_max
        self.trajectory_window = trajectory_window
        self.forward_accum = forward_accum
        self.lidar_range = lidar_range

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

    def _get_poses_arr(self, poses: List[np.ndarray]) -> np.ndarray:
        """Return poses stacked as (N, 4, 4) float64, cached by list identity."""
        key = id(poses)
        if not hasattr(self, '_poses_cache') or self._poses_cache[0] != key:
            self._poses_cache = (key, np.stack(poses).astype(np.float64))
        return self._poses_cache[1]

    def _label_range(
        self,
        xyz: np.ndarray,
        poses: List[np.ndarray],
        reference_idx: int,
        lo: int,
        hi: int,
    ) -> np.ndarray:
        """Core labeling: check trajectory poses in [lo, hi) against xyz.

        Uses Numba JIT when available; falls back to a pure-NumPy loop with a
        shrinking candidate set (early exit once all height-filtered points are
        covered).
        """
        labels = np.zeros(len(xyz), dtype=np.uint8)

        height_mask = (xyz[:, 2] >= self.height_min) & (xyz[:, 2] <= self.height_max)
        if not height_mask.any() or lo >= hi:
            return labels

        T_inv = np.linalg.inv(poses[reference_idx])

        if _HAS_NUMBA:
            return _label_core(
                np.asarray(xyz, dtype=np.float64),
                self._get_poses_arr(poses),
                T_inv,
                lo, hi,
                float(self.half_size),
                float(self.height_min), float(self.height_max),
                self.robot_shape == "round",
            )

        # NumPy fallback with shrinking candidate set
        unlabeled = np.where(height_mask)[0]
        for k in range(lo, hi):
            if len(unlabeled) == 0:
                break
            T       = T_inv @ poses[k]
            traj_xy = T[:2, 3]
            R_2d    = T[:2, :2]
            delta       = xyz[unlabeled, :2] - traj_xy
            delta_robot = delta @ R_2d
            hit = self._in_footprint(delta_robot[:, 0], delta_robot[:, 1])
            labels[unlabeled[hit]] = 1
            unlabeled = unlabeled[~hit]

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
            poses:       List of 4x4 world-frame poses.
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

    def label_scan_by_range(
        self,
        xyz: np.ndarray,
        poses: List[np.ndarray],
        current_idx: int,
        lidar_range: Optional[float] = None,
    ) -> np.ndarray:
        """
        Label a scan using a distance-based trajectory window.

        Advances the robot along its future poses and stops as soon as the
        robot is more than lidar_range metres (straight-line) from its position
        at current_idx — or at the end of the sequence.  Falls back to
        self.lidar_range if the argument is None.

        This is more principled than the pose-count window because it ties the
        horizon directly to the physical sensing range of the LiDAR: points
        further away than the sensor can see were not present in the scan and
        should not be labeled from it.

        Args:
            xyz:          (N, 3) scan points in current_idx's local frame.
            poses:        List of 4x4 world-frame poses.
            current_idx:  Index of the current scan.
            lidar_range:  Max distance (m) from the scan origin.  Defaults to
                          self.lidar_range; if both are None, all future poses
                          are used.

        Returns:
            labels: (N,) uint8 - 1 = traversable, 0 = not traversable.
        """
        effective_range = lidar_range if lidar_range is not None else self.lidar_range
        lo = current_idx + 1
        ref_pos = poses[current_idx][:3, 3]

        if effective_range is not None:
            hi = lo
            for j in range(lo, len(poses)):
                if np.linalg.norm(poses[j][:3, 3] - ref_pos) > effective_range:
                    break
                hi = j + 1
        else:
            hi = len(poses)

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
            poses:        List of 4x4 world-frame poses.
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
