"""
Precompute LiDAR-synchronized poses using IMU gyroscope integration.

For each LiDAR frame at timestamp t_k:
  - Rotation : integrate gyroscope (zero-order hold) from the nearest preceding
               odom pose to t_k.
  - Position : linear interpolation between the two bracketing odom poses.

Frames are marked invalid only when t_k falls outside the odom time range
(before the first or after the last odom sample).

Outputs written to <seq_dir>/<out_subdir>/:
  poses.npy       (N, 4, 4) float64  — SE3 matrix per LiDAR frame (NaN if invalid)
  valid_mask.npy  (N,)      bool     — False for invalid frames
  timestamps.txt  (N,)               — mirrors the LiDAR timestamps

Usage:
    python -m src.preprocessing.imu_pose_sync <seq_dir> [options]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (qx, qy, qz, qw) → 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def _rodrigues(omega_dt: np.ndarray) -> np.ndarray:
    """Rotation vector (ω·dt) → 3×3 rotation matrix via Rodrigues' formula."""
    angle = np.linalg.norm(omega_dt)
    if angle < 1e-12:
        return np.eye(3)
    axis = omega_dt / angle
    K = np.array([
        [       0, -axis[2],  axis[1]],
        [ axis[2],        0, -axis[0]],
        [-axis[1],  axis[0],        0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def compute_lidar_poses(
    lidar_ts: np.ndarray,
    odom_ts: np.ndarray,
    odom_pos: np.ndarray,
    odom_rot: np.ndarray,
    imu_ts: np.ndarray,
    imu_gyro: np.ndarray,
    T_vehicle_lidar: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        lidar_ts        (N,)       LiDAR frame timestamps.
        odom_ts         (M,)       Odometry timestamps (sorted).
        odom_pos        (M, 3)     Odometry positions.
        odom_rot        (M, 3, 3)  Odometry rotation matrices.
        imu_ts          (K,)       IMU timestamps (sorted).
        imu_gyro        (K, 3)     Angular velocity in body frame (rad/s).
        T_vehicle_lidar (4, 4)     Optional extrinsic: T_vehicle_lidar.

    Returns:
        poses      (N, 4, 4) float64  — NaN for frames outside odom range.
        valid_mask (N,)      bool.
    """
    N = len(lidar_ts)
    poses = np.full((N, 4, 4), np.nan, dtype=np.float64)
    valid = np.zeros(N, dtype=bool)

    for i, t_k in enumerate(lidar_ts):
        # Bracketing odom indices
        idx_after  = int(np.searchsorted(odom_ts, t_k, side="right"))
        idx_before = idx_after - 1

        # Only invalid if t_k falls outside the odom time range entirely
        if idx_before < 0 or idx_after >= len(odom_ts):
            continue

        t_before = odom_ts[idx_before]
        t_after  = odom_ts[idx_after]
        gap      = t_after - t_before

        # --- rotation: integrate gyro with zero-order hold ---
        R = odom_rot[idx_before].copy()

        # IMU indices strictly after t_before and at most t_k
        j_start = int(np.searchsorted(imu_ts, t_before, side="right"))
        j_end   = int(np.searchsorted(imu_ts, t_k,      side="right"))

        # Integration timeline: t_before | imu[j_start..j_end-1] | t_k
        t_nodes = np.empty(j_end - j_start + 2)
        t_nodes[0]  = t_before
        t_nodes[1:-1] = imu_ts[j_start:j_end]
        t_nodes[-1] = t_k

        for seg in range(len(t_nodes) - 1):
            dt = t_nodes[seg + 1] - t_nodes[seg]
            if dt <= 0:
                continue
            # Zero-order hold: last IMU sample at or before t_nodes[seg]
            j = np.clip(j_start - 1 + seg, 0, len(imu_gyro) - 1)
            R = R @ _rodrigues(imu_gyro[j] * dt)

        # --- position: linear interpolation ---
        alpha = (t_k - t_before) / gap
        p = odom_pos[idx_before] + alpha * (odom_pos[idx_after] - odom_pos[idx_before])

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = p

        if T_vehicle_lidar is not None:
            T = T @ T_vehicle_lidar

        poses[i] = T
        valid[i] = True

    return poses, valid


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_odom(
    odom_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts  = np.loadtxt(odom_dir / "timestamps.txt")
    raw = np.load(odom_dir / "odometry.npy")       # (M, 13)
    pos = raw[:, 0:3]
    rot = np.stack([
        _quat_to_rot(raw[j, 3], raw[j, 4], raw[j, 5], raw[j, 6])
        for j in range(len(raw))
    ])
    return ts, pos, rot


def _load_imu(imu_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    ts   = np.loadtxt(imu_dir / "timestamps.txt")
    data = np.load(imu_dir / "imu.npy")            # (K, 6): gyro_xyz | accel_xyz
    return ts, data[:, :3]


def _load_tf(tf_file: str, tf_key: str) -> np.ndarray:
    tfs = np.load(tf_file, allow_pickle=True).item()
    if tf_key not in tfs:
        raise KeyError(f"Key '{tf_key}' not found. Available: {list(tfs.keys())}")
    return tfs[tf_key].astype(np.float64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Precompute LiDAR-synchronized poses using IMU gyroscope integration."
    )
    p.add_argument("seq_dir",
                   help="Sequence directory containing odom / IMU / LiDAR subdirs.")
    p.add_argument("--odom-subdir",  default="super_odom",
                   help="Odometry sub-directory (default: super_odom).")
    p.add_argument("--imu-subdir",   default="multisense_imu",
                   help="IMU sub-directory (default: multisense_imu).")
    p.add_argument("--lidar-subdir", default="velodyne_1",
                   help="LiDAR sub-directory used for timestamps (default: velodyne_1).")
    p.add_argument("--out-subdir",   default="lidar_poses",
                   help="Output sub-directory (default: lidar_poses).")
    p.add_argument("--tf-file",      default=None,
                   help="Static transforms .npy file (optional).")
    p.add_argument("--tf-key",       default=None,
                   help="Key in --tf-file for T_vehicle_lidar (optional).")
    return p


def run(args: argparse.Namespace) -> None:
    seq_dir = Path(args.seq_dir)

    print(f"Odom   : {seq_dir / args.odom_subdir}")
    odom_ts, odom_pos, odom_rot = _load_odom(seq_dir / args.odom_subdir)
    print(f"  {len(odom_ts)} samples  ~{1/np.diff(odom_ts).mean():.0f} Hz")

    print(f"IMU    : {seq_dir / args.imu_subdir}")
    imu_ts, imu_gyro = _load_imu(seq_dir / args.imu_subdir)
    print(f"  {len(imu_ts)} samples  ~{1/np.diff(imu_ts).mean():.0f} Hz")

    print(f"LiDAR  : {seq_dir / args.lidar_subdir}")
    lidar_ts = np.loadtxt(seq_dir / args.lidar_subdir / "timestamps.txt")
    print(f"  {len(lidar_ts)} frames  ~{1/np.diff(lidar_ts).mean():.0f} Hz")

    T_vehicle_lidar = None
    if args.tf_file and args.tf_key:
        print(f"TF     : {args.tf_file}  key={args.tf_key}")
        T_vehicle_lidar = _load_tf(args.tf_file, args.tf_key)

    print(f"\nComputing {len(lidar_ts)} poses …")
    poses, valid = compute_lidar_poses(
        lidar_ts, odom_ts, odom_pos, odom_rot,
        imu_ts, imu_gyro,
        T_vehicle_lidar=T_vehicle_lidar,
    )

    out_dir = seq_dir / args.out_subdir
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "poses.npy",      poses)
    np.save(out_dir / "valid_mask.npy", valid)
    np.savetxt(out_dir / "timestamps.txt", lidar_ts, fmt="%.18e")

    n_valid = int(valid.sum())
    print(f"Done.  {n_valid}/{len(valid)} valid poses "
          f"({len(valid) - n_valid} dropped).")
    print(f"Output : {out_dir}")


if __name__ == "__main__":
    run(_parser().parse_args())
