"""
Pose verification using GICP.

For each pair of consecutive scans (k, k+1):
  1. Use the current pose estimate as initial guess for the relative transform.
  2. Run GICP to find the best-fit alignment.
  3. Report the residual between odom and GICP transforms.

Interpretation:
  - fitness ~ 1.0 and small residual  → poses are good.
  - low fitness                        → scans don't overlap (large motion, bad TF).
  - high fitness but large residual    → systematic TF error or odom drift.
  - residual consistent in direction   → systematic TF error.
  - residual random / growing          → odom noise or drift.

Usage:
    python -m src.preprocessing.verify_poses <seq_dir> [options]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from src.datasets.tartandrive import TartanDriveSequence, load_vehicle_lidar_tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return pcd


def _rotation_error_deg(R: np.ndarray) -> float:
    """Angular error of a rotation matrix relative to identity, in degrees."""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def run_gicp(
    pcd_src: o3d.geometry.PointCloud,
    pcd_dst: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    max_corr_dist: float,
    normal_radius: float,
) -> o3d.pipelines.registration.RegistrationResult:
    for pcd in (pcd_src, pcd_dst):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
    return o3d.pipelines.registration.registration_generalized_icp(
        pcd_src, pcd_dst,
        max_corr_dist,
        T_init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def verify(
    seq: TartanDriveSequence,
    indices: list[int],
    max_corr_dist: float = 1.0,
    normal_radius: float = 0.5,
    max_pts: int = 10_000,
) -> list[dict]:
    results = []

    for i in indices:
        if i + 1 >= len(seq):
            continue

        xyz_k,  _ = seq.get_scan(i)
        xyz_k1, _ = seq.get_scan(i + 1)

        # Downsample
        if len(xyz_k)  > max_pts: xyz_k  = xyz_k [::max(1, len(xyz_k)  // max_pts)]
        if len(xyz_k1) > max_pts: xyz_k1 = xyz_k1[::max(1, len(xyz_k1) // max_pts)]

        # Initial relative transform from poses:  T_k <- k+1
        T_odom = np.linalg.inv(seq.poses[i]) @ seq.poses[i + 1]

        pcd_k  = _to_pcd(xyz_k)
        pcd_k1 = _to_pcd(xyz_k1)

        reg = run_gicp(pcd_k1, pcd_k, T_odom, max_corr_dist, normal_radius)
        T_gicp = reg.transformation

        # Residual: how far GICP moved from the odom initial guess
        T_residual = np.linalg.inv(T_odom) @ T_gicp
        t_err = np.linalg.norm(T_residual[:3, 3])
        R_err = _rotation_error_deg(T_residual[:3, :3])

        results.append({
            "idx":       i,
            "fitness":   reg.fitness,
            "rmse":      reg.inlier_rmse,
            "t_err_m":   t_err,
            "R_err_deg": R_err,
            "T_odom":    T_odom,
            "T_gicp":    T_gicp,
        })

        print(
            f"  [{i:4d}->{i+1:4d}]  fitness={reg.fitness:.3f}  rmse={reg.inlier_rmse:.4f}  "
            f"Δt={t_err:.3f} m  ΔR={R_err:.2f}°"
        )

    return results


def summarize(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return

    fitness  = np.array([r["fitness"]   for r in results])
    t_errs   = np.array([r["t_err_m"]   for r in results])
    R_errs   = np.array([r["R_err_deg"] for r in results])

    print("\n--- Summary ---")
    print(f"  Pairs evaluated : {len(results)}")
    print(f"  Fitness         : mean={fitness.mean():.3f}  min={fitness.min():.3f}")
    print(f"  Δt odom→GICP    : mean={t_errs.mean():.3f} m   max={t_errs.max():.3f} m")
    print(f"  ΔR odom→GICP    : mean={R_errs.mean():.2f}°   max={R_errs.max():.2f}°")

    if fitness.mean() < 0.3:
        print("\n  [!] Low fitness — scans barely overlap.")
        print("      Likely cause: wrong TF (wrong sensor frame) or too large motion.")
    elif t_errs.mean() > 0.1 or R_errs.mean() > 2.0:
        print("\n  [!] High residual despite good fitness.")
        print("      Likely cause: systematic TF error or odom drift.")
    else:
        print("\n  [OK] Poses look consistent with scan-to-scan geometry.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify poses by comparing odometry with GICP scan-to-scan alignment."
    )
    p.add_argument("seq_dir",
                   help="Sequence directory.")
    p.add_argument("--lidar-subdir",       default="velodyne_1")
    p.add_argument("--odom-subdir",        default="super_odom")
    p.add_argument("--lidar-poses-subdir", default="lidar_poses",
                   help="Use precomputed poses if available (set to '' to force raw odom).")
    p.add_argument("--tf-file",            default=None)
    p.add_argument("--tf-key",             default=None)
    p.add_argument("--n-pairs",            default=20,  type=int,
                   help="Number of scan pairs to evaluate (evenly spaced).")
    p.add_argument("--max-corr-dist",      default=1.0, type=float,
                   help="GICP max correspondence distance (m).")
    p.add_argument("--normal-radius",      default=0.5, type=float)
    p.add_argument("--max-pts",            default=10_000, type=int,
                   help="Max points per scan (downsampled if larger).")
    return p


def main(args: argparse.Namespace) -> None:
    seq_dir = Path(args.seq_dir)

    T_vehicle_lidar = None
    if args.tf_file and args.tf_key:
        T_vehicle_lidar = load_vehicle_lidar_tf(args.tf_file, args.tf_key)
        print(f"TF: {args.tf_key}\n{T_vehicle_lidar.round(4)}\n")

    lidar_poses_subdir = args.lidar_poses_subdir or None
    seq = TartanDriveSequence(
        str(seq_dir),
        lidar_subdir=args.lidar_subdir,
        odom_subdir=args.odom_subdir,
        lidar_poses_subdir=lidar_poses_subdir,
        T_vehicle_lidar=T_vehicle_lidar,
    )
    print(f"Sequence: {seq.name}  ({len(seq)} scans)\n")

    indices = np.linspace(0, len(seq) - 2, args.n_pairs, dtype=int).tolist()
    results = verify(seq, indices, args.max_corr_dist, args.normal_radius, args.max_pts)
    summarize(results)


if __name__ == "__main__":
    main(_parser().parse_args())
