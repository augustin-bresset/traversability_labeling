"""
Generic KITTI-format LiDAR sequence reader, with a RELLIS-3D layout wrapper.
"""

from __future__ import annotations

from pathlib import Path
from glob import glob
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from src.robot import Robot


class KittiSequence:
    """
    One LiDAR sequence in KITTI binary format.

    Args:
        cloud_dir:   Directory containing *.bin files (XYZI float32, 4 floats/point).
        poses_file:  Path to a KITTI-format poses.txt (12 values/line → 3x4 matrix).
                     If None or the file does not exist, poses are unavailable.
        max_rad:     Range filter applied when loading each scan.
    """

    def __init__(
        self,
        cloud_dir: str,
        poses_file: Optional[str] = None,
        max_rad: float = 50.0,
        robot: Optional["Robot"] = None,
        target_indices: Optional[set] = None,
    ):
        self.cloud_dir = Path(cloud_dir)
        self.max_rad   = max_rad
        self.robot     = robot
        self.target_indices = target_indices  # None → label all frames

        self.cloud_files: List[str] = sorted(glob(str(self.cloud_dir / "*.bin")))
        if not self.cloud_files:
            raise FileNotFoundError(f"No .bin files found in {self.cloud_dir}")

        poses_path = Path(poses_file) if poses_file else None
        self.poses: Optional[List[np.ndarray]] = (
            self._load_kitti_poses(poses_path)
            if poses_path is not None and poses_path.exists()
            else None
        )

    @staticmethod
    def _load_kitti_poses(path: Path) -> List[np.ndarray]:
        """Read KITTI-format poses: 12 floats per line → 4x4 matrix."""
        poses = []
        with open(path) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
        return poses

    def __len__(self) -> int:
        return len(self.cloud_files)

    def get_scan(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (xyz, intensity): shapes (N,3) and (N,), dtype float32."""
        raw = np.fromfile(self.cloud_files[idx], dtype=np.float32).reshape(-1, 4)
        xyz, intensity = raw[:, :3], raw[:, 3]
        mask = np.linalg.norm(xyz, axis=1) < self.max_rad
        if self.robot is not None:
            mask &= ~self.robot.self_hit_mask(xyz)
        return xyz[mask], intensity[mask]

    def get_pose(self, idx: int) -> Optional[np.ndarray]:
        if self.poses is None or idx >= len(self.poses):
            return None
        return self.poses[idx]

    def has_poses(self) -> bool:
        return self.poses is not None and len(self.poses) == len(self)

    @property
    def name(self) -> str:
        return self.cloud_dir.parent.name
