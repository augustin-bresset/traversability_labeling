"""
RELLIS-3D dataset loader for traversability labeling.

Expected dataset structure:
    <root>/
    ├── 00000/
    │   ├── os1_cloud_node_kitti_bin/   # LiDAR scans (XYZI float32)
    │   │   └── frame*.bin
    │   └── poses.txt                   # Optional: KITTI-style poses (12 values/line)
    ├── 00001/ ...
    ├── pt_train.lst                    # Optional: frame-level split files
    ├── pt_val.lst
    └── pt_test.lst
"""

from __future__ import annotations

import os
from pathlib import Path
from glob import glob
from typing import List, Optional, Tuple

import numpy as np


SPLIT_SEQUENCES = {
    "train": ["00000", "00001", "00002", "00003"],
    "val":   ["00004"],
    "test":  ["00004"],
}

CLOUD_SUBDIR = "os1_cloud_node_kitti_bin"


class Rellis3DSequence:
    """
    One RELLIS-3D sequence: ordered access to scans and poses.
    """

    def __init__(self, seq_dir: str, max_rad: float = 50.0):
        self.seq_dir = Path(seq_dir)
        self.max_rad = max_rad

        cloud_glob = str(self.seq_dir / CLOUD_SUBDIR / "*.bin")
        self.cloud_files: List[str] = sorted(glob(cloud_glob))

        if not self.cloud_files:
            raise FileNotFoundError(f"No .bin files found at {cloud_glob}")

        poses_file = self.seq_dir / "poses.txt"
        self.poses: Optional[List[np.ndarray]] = (
            self._load_kitti_poses(poses_file) if poses_file.exists() else None
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_kitti_poses(path: Path) -> List[np.ndarray]:
        """Read KITTI-format poses: 12 floats per line -> 4×4 matrix."""
        poses = []
        with open(path) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
        return poses

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cloud_files)

    def get_scan(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (xyz, intensity): shapes (N,3) and (N,), dtype float32."""
        raw = np.fromfile(self.cloud_files[idx], dtype=np.float32).reshape(-1, 4)
        xyz = raw[:, :3]
        intensity = raw[:, 3]
        mask = np.linalg.norm(xyz, axis=1) < self.max_rad
        return xyz[mask], intensity[mask]

    def get_pose(self, idx: int) -> Optional[np.ndarray]:
        """Return 4×4 world-frame pose or None if poses not available."""
        if self.poses is None or idx >= len(self.poses):
            return None
        return self.poses[idx]

    @property
    def name(self) -> str:
        return self.seq_dir.name

    def has_poses(self) -> bool:
        return self.poses is not None and len(self.poses) == len(self)


class Rellis3D:
    """
    RELLIS-3D dataset: iterates over sequences for traversability labeling.

    The dataset root may or may not contain a "Rellis-3D" sub-directory;
    both layouts are supported.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        max_rad: float = 50.0,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_rad = max_rad

        seq_names = SPLIT_SEQUENCES.get(split)
        if seq_names is None:
            raise ValueError(f"Unknown split '{split}'. Choose from {list(SPLIT_SEQUENCES)}")

        self.sequences: List[Rellis3DSequence] = []
        for name in seq_names:
            seq_path = self._find_seq(name)
            if seq_path is not None:
                self.sequences.append(Rellis3DSequence(str(seq_path), max_rad))

        if not self.sequences:
            raise FileNotFoundError(
                f"No RELLIS-3D sequences found in '{root_dir}' for split '{split}'.\n"
                f"Expected sequences: {seq_names}"
            )

    def _find_seq(self, name: str) -> Optional[Path]:
        for candidate in (
            self.root_dir / "Rellis-3D" / name,
            self.root_dir / name,
        ):
            if (candidate / CLOUD_SUBDIR).exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)

    def __getitem__(self, idx: int) -> Rellis3DSequence:
        return self.sequences[idx]
