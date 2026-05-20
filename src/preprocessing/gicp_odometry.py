"""
LiDAR odometry via KISS-ICP.

Builds a full trajectory by processing scans sequentially with KISS-ICP
(https://github.com/PRBonn/kiss-icp).  Much faster than Open3D GICP (~100×)
and handles motion deskewing natively when per-point timestamps are available.

Optional GPS z-correction (--gps-odom-subdir):
  KISS-ICP's z-translation is poorly constrained on flat terrain and drifts.
  When a GPS/odom source is available, the z-component of every pose can be
  replaced with GPS altitude interpolated to LiDAR timestamps.  XY and rotation
  from KISS-ICP are preserved.

Output written to <seq_dir>/<out_subdir>/:
  poses.npy       (N, 4, 4) float64  — T_world_lidar per frame (local odom frame)
  valid_mask.npy  (N,)      bool     — always True (KISS-ICP always produces a pose)
  timestamps.txt  (N,)               — mirrors LiDAR timestamps

Usage:
    python -m src.preprocessing.gicp_odometry <seq_dir> [options]
    python -m src.preprocessing.gicp_odometry <seq_dir> --gps-odom-subdir gps_odom
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP

from src.datasets.tartandrive import TartanDriveSequence, load_vehicle_lidar_tf


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def correct_z_with_gps(
    poses: np.ndarray,
    lidar_ts: np.ndarray,
    gps_odom_dir: Path,
) -> np.ndarray:
    """
    Replace the z-translation of each pose with GPS altitude.

    KISS-ICP's z-translation is poorly constrained on flat terrain (z is
    nearly unobservable geometrically), causing systematic upward drift.
    GPS provides reliable altitude at lower frequency.

    The corrected z is anchored to the first LiDAR frame so the output
    remains in a local reference frame:
        corrected_z[k] = gps_z_interp[k] - gps_z_interp[0]

    XY-translation and rotation are untouched.

    Args:
        poses:       (N, 4, 4) KISS-ICP poses (modified in place of copy).
        lidar_ts:    (N,) LiDAR timestamps (Unix seconds).
        gps_odom_dir: Directory with odometry.npy (col 2 = altitude) and timestamps.txt.

    Returns:
        corrected (N, 4, 4) poses.
    """
    gps_ts   = np.loadtxt(gps_odom_dir / "timestamps.txt")
    gps_odom = np.load(gps_odom_dir / "odometry.npy")   # (M, 13)
    gps_z    = gps_odom[:, 2]                            # altitude column

    # Linear interpolation; clamp to GPS time range
    gps_z_interp = np.interp(lidar_ts, gps_ts, gps_z)

    # Anchor to local frame (relative to first scan)
    gps_z_local = gps_z_interp - gps_z_interp[0]

    corrected = poses.copy()
    corrected[:, 2, 3] = gps_z_local
    return corrected


def build_kiss_trajectory(
    seq: TartanDriveSequence,
    max_range: float = 50.0,
    min_range: float = 1.0,
    voxel_size: float = 1.0,
    deskew: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process all scans with KISS-ICP and return the trajectory.

    Returns:
        poses      (N, 4, 4) float64
        valid_mask (N,)      bool  — all True (KISS-ICP always estimates a pose)
    """
    cfg = KISSConfig()
    cfg.data.max_range = max_range
    cfg.data.min_range = min_range
    cfg.data.deskew    = deskew
    cfg.mapping.voxel_size = float(voxel_size)

    kiss = KissICP(config=cfg)

    N = len(seq)
    poses = np.zeros((N, 4, 4), dtype=np.float64)

    for k in range(N):
        xyz, _ = seq.get_scan(k)
        # KISS-ICP expects (N,3) float64; timestamps = zeros when deskew disabled
        timestamps = np.zeros(len(xyz), dtype=np.float64)
        kiss.register_frame(xyz.astype(np.float64), timestamps)

        poses[k] = kiss.last_pose   # (4,4) SE3 — T_world_lidar[k]

        if (k + 1) % 200 == 0 or k == N - 1:
            print(f"  [{k+1:4d}/{N}]")

    valid_mask = np.ones(N, dtype=bool)
    return poses, valid_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a LiDAR odometry trajectory using KISS-ICP."
    )
    p.add_argument("seq_dir")
    p.add_argument("--lidar-subdir",       default="velodyne_1")
    p.add_argument("--odom-subdir",        default=None,
                   help="Odom sub-dir (optional, not used by KISS-ICP itself).")
    p.add_argument("--lidar-poses-subdir", default=None,
                   help="Precomputed poses (not used by KISS-ICP).")
    p.add_argument("--out-subdir",         default="gicp_poses")
    p.add_argument("--tf-file",            default=None)
    p.add_argument("--tf-key",             default=None)
    p.add_argument("--max-range",          default=50.0, type=float)
    p.add_argument("--min-range",          default=1.0,  type=float)
    p.add_argument("--voxel-size",         default=1.0, type=float,
                   help="KISS-ICP voxel size in metres (default: 1.0).")
    p.add_argument("--deskew",             action="store_true",
                   help="Enable motion deskewing (requires per-point timestamps).")
    p.add_argument("--gps-odom-subdir",    default=None,
                   help="Sub-directory with GPS odometry (e.g. 'gps_odom'). "
                        "When provided, the z-component of each pose is replaced "
                        "with interpolated GPS altitude to remove KISS-ICP z-drift.")
    return p


def main(args: argparse.Namespace) -> None:
    seq_dir = Path(args.seq_dir)

    T_vehicle_lidar = None
    if args.tf_file and args.tf_key:
        T_vehicle_lidar = load_vehicle_lidar_tf(args.tf_file, args.tf_key)

    seq = TartanDriveSequence(
        str(seq_dir),
        lidar_subdir=args.lidar_subdir,
        odom_subdir=args.odom_subdir or None,
        lidar_poses_subdir=args.lidar_poses_subdir or None,
        T_vehicle_lidar=T_vehicle_lidar,
    )
    print(f"Sequence : {seq.name}  ({len(seq)} scans)")
    print(f"Building KISS-ICP trajectory …\n")

    t0 = time.time()
    poses, valid = build_kiss_trajectory(
        seq,
        max_range=args.max_range,
        min_range=args.min_range,
        voxel_size=args.voxel_size,
        deskew=args.deskew,
    )
    elapsed = time.time() - t0
    print(f"\nKISS-ICP done. {len(seq)} frames in {elapsed:.1f}s  ({len(seq)/elapsed:.0f} scans/s).")

    lidar_ts_path = seq_dir / args.lidar_subdir / "timestamps.txt"
    lidar_ts = np.loadtxt(lidar_ts_path)

    if args.gps_odom_subdir:
        gps_dir = seq_dir / args.gps_odom_subdir
        if not (gps_dir / "odometry.npy").exists():
            print(f"[WARN] GPS odom not found at {gps_dir} — skipping z-correction.")
        else:
            z_before = poses[:, 2, 3]
            poses = correct_z_with_gps(poses, lidar_ts, gps_dir)
            z_after = poses[:, 2, 3]
            print(f"GPS z-correction applied ({gps_dir.name}/):")
            print(f"  z range before: [{z_before.min():.2f}, {z_before.max():.2f}] m  "
                  f"(drift {z_before[-1] - z_before[0]:+.2f} m)")
            print(f"  z range after : [{z_after.min():.2f}, {z_after.max():.2f}] m  "
                  f"(drift {z_after[-1] - z_after[0]:+.2f} m)")

    out_dir = seq_dir / args.out_subdir
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "poses.npy",      poses)
    np.save(out_dir / "valid_mask.npy", valid)
    np.savetxt(out_dir / "timestamps.txt", lidar_ts, fmt="%.18e")

    print(f"Output : {out_dir}")


if __name__ == "__main__":
    main(_parser().parse_args())
