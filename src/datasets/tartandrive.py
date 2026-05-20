"""
TartanDrive dataset loader.

Point clouds — separate .npy files per scan:
  <seq_dir>/<lidar_subdir>/XXXXXX.npy           → (N, 3) float32  XYZ (sensor frame)
  <seq_dir>/<lidar_subdir>/XXXXXX_intensity.npy → (N,)   uint8    intensity

Odometry — matched to lidar timestamps:
  <seq_dir>/<odom_subdir>/odometry.npy    → (M, 13) float64
      columns 0-2 : x, y, z  (position)
      columns 3-6 : qx, qy, qz, qw  (unit quaternion)
      columns 7-12: velocities (unused)
  <seq_dir>/<odom_subdir>/timestamps.txt  → M lines, one Unix timestamp per line

For each lidar scan the nearest-timestamp odometry entry is used.
Supported odom sources: "super_odom" (local frame), "gps_odom" (UTM/global frame).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from src.robot import Robot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (qx, qy, qz, qw) → 3×3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def load_vehicle_lidar_tf(
    tf_file: str,
    tf_key: str,
) -> np.ndarray:
    """
    Load a 4×4 T_vehicle_lidar matrix from a static-transforms .npy dict.

    Args:
        tf_file: Path to the .npy file (saved as a dict with allow_pickle=True).
        tf_key:  Key in the dict, e.g. "vehicle__to__livox_frame".

    Returns:
        T_vehicle_lidar: (4,4) float64 matrix.
    """
    tfs = np.load(tf_file, allow_pickle=True).item()
    if tf_key not in tfs:
        available = list(tfs.keys())
        raise KeyError(f"TF key '{tf_key}' not found. Available: {available}")
    return tfs[tf_key].astype(np.float64)


def _load_odom_poses(
    odom_dir: Path,
    lidar_timestamps: np.ndarray,
    T_vehicle_lidar: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Load odometry and return one 4×4 SE3 pose per lidar scan,
    matched by nearest timestamp.

    If T_vehicle_lidar is provided (transform from lidar frame to vehicle frame),
    each pose is composed as  T_world_vehicle @ T_vehicle_lidar  so that the
    returned poses express the lidar origin in world coordinates.
    """
    odom_ts = np.loadtxt(odom_dir / "timestamps.txt")
    odom    = np.load(odom_dir / "odometry.npy")   # (M, 13)

    # Nearest-neighbour matching: for each lidar ts find closest odom ts
    indices = np.abs(odom_ts[:, None] - lidar_timestamps[None, :]).argmin(axis=0)

    poses = []
    for idx in indices:
        x, y, z         = odom[idx, 0:3]
        qx, qy, qz, qw  = odom[idx, 3:7]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
        T[:3,  3] = [x, y, z]
        if T_vehicle_lidar is not None:
            T = T @ T_vehicle_lidar
        poses.append(T)

    return poses


def _load_lidar_poses(
    lidar_poses_dir: Path,
    T_vehicle_lidar: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load precomputed LiDAR-synchronized poses produced by imu_pose_sync.

    poses.npy stores T_world_vehicle.  If T_vehicle_lidar is provided the
    returned poses are T_world_lidar = T_world_vehicle @ T_vehicle_lidar,
    consistent with _load_odom_poses.

    Invalid frames (NaN in poses.npy) are filled with the nearest valid pose
    so that the returned list is always fully populated.

    Returns:
        poses      List of N (4, 4) float64 matrices.
        valid_mask (N,) bool — False for frames that were originally invalid.
    """
    raw_poses  = np.load(lidar_poses_dir / "poses.npy")       # (N, 4, 4)
    valid_mask = np.load(lidar_poses_dir / "valid_mask.npy")  # (N,)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        raise RuntimeError(f"No valid poses found in {lidar_poses_dir}")

    poses: List[np.ndarray] = []
    for i in range(len(raw_poses)):
        if valid_mask[i]:
            T = raw_poses[i]
        else:
            nearest = valid_indices[np.argmin(np.abs(valid_indices - i))]
            T = raw_poses[nearest]
        if T_vehicle_lidar is not None:
            T = T @ T_vehicle_lidar
        poses.append(T)

    return poses, valid_mask


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

class TartanDriveSequence:
    """
    One TartanDrive recording.

    Args:
        seq_dir:           Directory containing <lidar_subdir>/ and optionally <odom_subdir>/.
        lidar_subdir:      Sub-directory with *.npy scan files (e.g. "livox", "velodyne_0").
        odom_subdir:       Sub-directory with odometry.npy + timestamps.txt.
                           If None or not found, poses will be unavailable.
        lidar_poses_subdir: Sub-directory with precomputed IMU-synchronized poses
                           (output of imu_pose_sync). When present, it takes priority
                           over odom_subdir. Set to None to always use odom.
        max_rad:           Range filter (metres).

    Exposes the same interface as KittiSequence.
    """

    def __init__(
        self,
        seq_dir: str,
        lidar_subdir: str = "livox",
        odom_subdir: Optional[str] = "super_odom",
        lidar_poses_subdir: Optional[str] = "lidar_poses",
        max_rad: float = 50.0,
        robot: Optional["Robot"] = None,
        T_vehicle_lidar: Optional[np.ndarray] = None,
        target_indices: Optional[set] = None,
    ):
        self.seq_dir   = Path(seq_dir)
        self.cloud_dir = self.seq_dir / lidar_subdir
        self.max_rad   = max_rad
        self.robot     = robot
        self.target_indices = target_indices

        if not self.cloud_dir.is_dir():
            raise FileNotFoundError(f"LiDAR directory not found: {self.cloud_dir}")

        self.cloud_files: List[str] = sorted(
            str(p)
            for p in self.cloud_dir.glob("*.npy")
            if not p.stem.endswith("_intensity")
        )
        if not self.cloud_files:
            raise FileNotFoundError(f"No .npy point cloud files found in {self.cloud_dir}")

        # Load lidar timestamps for pose matching
        lidar_ts_file = self.cloud_dir / "timestamps.txt"
        lidar_ts = (
            np.loadtxt(lidar_ts_file)
            if lidar_ts_file.exists()
            else None
        )

        # Load poses — prefer precomputed lidar_poses over raw odom
        self.poses: Optional[List[np.ndarray]] = None
        self.valid_mask: Optional[np.ndarray] = None

        lidar_poses_dir = self.seq_dir / lidar_poses_subdir if lidar_poses_subdir else None
        if lidar_poses_dir and (lidar_poses_dir / "poses.npy").exists():
            self.poses, self.valid_mask = _load_lidar_poses(lidar_poses_dir, T_vehicle_lidar)
            n_valid = int(self.valid_mask.sum())
            print(f"  [poses] {lidar_poses_dir.name}/ — "
                  f"{n_valid}/{len(self.valid_mask)} valid frames")
        elif odom_subdir and lidar_ts is not None:
            odom_dir = self.seq_dir / odom_subdir
            if (odom_dir / "odometry.npy").exists():
                self.poses = _load_odom_poses(odom_dir, lidar_ts, T_vehicle_lidar)
            else:
                print(f"  [WARN] Odom directory not found: {odom_dir} — no poses.")
        elif odom_subdir and lidar_ts is None:
            print(f"  [WARN] No timestamps.txt in {self.cloud_dir} — cannot match poses.")

    def get_scan(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (xyz, intensity): shapes (N,3) float32 and (N,) float32."""
        xyz_path = Path(self.cloud_files[idx])
        intensity_path = xyz_path.parent / (xyz_path.stem + "_intensity.npy")

        xyz = np.load(xyz_path).astype(np.float32)
        intensity = (
            np.load(intensity_path).astype(np.float32)
            if intensity_path.exists()
            else np.ones(len(xyz), dtype=np.float32)
        )

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

    def __len__(self) -> int:
        return len(self.cloud_files)

    @property
    def name(self) -> str:
        return self.seq_dir.name


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TartanDrive:
    """
    TartanDrive dataset.

    Discovers all sequences under root_dir that contain <lidar_subdir>/*.npy files.
    Handles flat (<root>/<seq>/livox/) and double-named (<root>/<seq>/<seq>/livox/)
    layouts.
    """

    def __init__(
        self,
        root_dir: str,
        lidar_subdir: str = "livox",
        odom_subdir: Optional[str] = "super_odom",
        lidar_poses_subdir: Optional[str] = "lidar_poses",
        max_rad: float = 50.0,
        robot: Optional["Robot"] = None,
        T_vehicle_lidar: Optional[np.ndarray] = None,
    ):
        self.root_dir          = Path(root_dir)
        self.lidar_subdir      = lidar_subdir
        self.odom_subdir       = odom_subdir
        self.lidar_poses_subdir = lidar_poses_subdir
        self.sequences: List[TartanDriveSequence] = []

        self._discover(max_rad, robot, T_vehicle_lidar)

        if not self.sequences:
            raise FileNotFoundError(
                f"No TartanDrive sequences found under '{root_dir}' "
                f"(looking for '<seq>/{lidar_subdir}/*.npy')."
            )

    def _discover(self, max_rad: float, robot: Optional["Robot"],
                  T_vehicle_lidar: Optional[np.ndarray]) -> None:
        seen: set = set()
        for lidar_dir in sorted(self.root_dir.rglob(self.lidar_subdir)):
            if not lidar_dir.is_dir():
                continue
            seq_dir = lidar_dir.parent
            if seq_dir in seen:
                continue
            seen.add(seq_dir)
            try:
                seq = TartanDriveSequence(
                    str(seq_dir),
                    lidar_subdir=self.lidar_subdir,
                    odom_subdir=self.odom_subdir,
                    lidar_poses_subdir=self.lidar_poses_subdir,
                    max_rad=max_rad,
                    robot=robot,
                    T_vehicle_lidar=T_vehicle_lidar,
                )
                self.sequences.append(seq)
            except FileNotFoundError:
                pass

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)

    def __getitem__(self, idx: int) -> TartanDriveSequence:
        return self.sequences[idx]
