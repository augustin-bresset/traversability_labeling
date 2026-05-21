"""
Traversability labeling pipeline for LiDAR sequences.

For each scan in a sequence:
  1. Load the point cloud.
  2. Load robot poses (required — the sequence must provide them).
  3. Label each point as traversable (1) or not (0) based on robot footprint
     along the trajectory window.
  4. Save labels as <stem>.trav (binary uint8 array, same order as input points).

Usage:
    # RELLIS-3D split:
    python label_traversability.py --dataset rellis [--config configs/example_rellis.yaml] [--split train]

    # TartanDrive dataset:
    python label_traversability.py --dataset tartandrive [--config configs/example_tartan.yaml]

    # Single sequence (any KITTI-format dataset):
    python label_traversability.py --seq /data/myseq --cloud-subdir velodyne
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.rellis import KittiSequence, Rellis3D, RELLIS_CLOUD_SUBDIR
from src.datasets.tartandrive import TartanDrive, TartanDriveSequence, load_vehicle_lidar_tf
from src.robot import Robot
from src.traversability.labeler import TraversabilityLabeler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get(cfg: dict, *keys, default=None):
    """Nested dict access with a default."""
    val = cfg
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val


# ---------------------------------------------------------------------------
# Occupancy-grid helpers (used by method="grid")
# ---------------------------------------------------------------------------

def _build_grid(poses, resolution: float, half_size: float):
    """Allocate a 2D int16 grid covering the trajectory + one footprint margin."""
    all_x = np.array([p[0, 3] for p in poses])
    all_y = np.array([p[1, 3] for p in poses])
    margin = half_size + resolution
    ox = float(all_x.min() - margin)
    oy = float(all_y.min() - margin)
    nx = int(np.ceil((all_x.max() - all_x.min() + 2 * margin) / resolution)) + 2
    ny = int(np.ceil((all_y.max() - all_y.min() + 2 * margin) / resolution)) + 2
    return np.zeros((nx, ny), dtype=np.int16), ox, oy


def _rasterize(grid, tx: float, ty: float, R_2d, half_size: float,
               resolution: float, ox: float, oy: float, delta: int, is_round: bool):
    """Add delta (+1 or -1) to grid cells covered by the robot footprint at (tx, ty)."""
    r_cells = int(np.ceil(half_size / resolution))
    cx = int((tx - ox) / resolution)
    cy = int((ty - oy) / resolution)

    dg = np.arange(-r_cells, r_cells + 1)
    DGX, DGY = np.meshgrid(dg, dg, indexing='ij')          # (2r+1, 2r+1)

    WX = ox + (cx + DGX + 0.5) * resolution
    WY = oy + (cy + DGY + 0.5) * resolution
    DX = WX - tx
    DY = WY - ty

    if is_round:
        mask = DX ** 2 + DY ** 2 <= half_size ** 2
    else:
        # Rotate displacement into robot body frame to test axis-aligned footprint
        RX = DX * R_2d[0, 0] + DY * R_2d[1, 0]
        RY = DX * R_2d[0, 1] + DY * R_2d[1, 1]
        mask = (np.abs(RX) <= half_size) & (np.abs(RY) <= half_size)

    GX = (cx + DGX)[mask]
    GY = (cy + DGY)[mask]
    valid = (GX >= 0) & (GX < grid.shape[0]) & (GY >= 0) & (GY < grid.shape[1])
    grid[GX[valid], GY[valid]] += delta


def _update_footprint(grid, pose, labeler: TraversabilityLabeler,
                      resolution: float, ox: float, oy: float, delta: int):
    tx, ty = float(pose[0, 3]), float(pose[1, 3])
    _rasterize(grid, tx, ty, pose[:2, :2], labeler.half_size,
               resolution, ox, oy, delta, labeler.robot_shape == "round")


def _query_grid(grid, xyz: np.ndarray, pose, labeler: TraversabilityLabeler,
                resolution: float, ox: float, oy: float) -> np.ndarray:
    """Return uint8 label array for xyz (in scan-local frame)."""
    labels = np.zeros(len(xyz), dtype=np.uint8)
    height_mask = (xyz[:, 2] >= labeler.height_min) & (xyz[:, 2] <= labeler.height_max)
    if not height_mask.any():
        return labels
    R, t = pose[:3, :3], pose[:3, 3]
    xyz_world = (R @ xyz[height_mask].T).T + t
    gx = ((xyz_world[:, 0] - ox) / resolution).astype(np.int32)
    gy = ((xyz_world[:, 1] - oy) / resolution).astype(np.int32)
    valid = (gx >= 0) & (gx < grid.shape[0]) & (gy >= 0) & (gy < grid.shape[1])
    trav = np.zeros(height_mask.sum(), dtype=bool)
    trav[valid] = grid[gx[valid], gy[valid]] > 0
    labels[height_mask] = trav.astype(np.uint8)
    return labels


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(
    seq,
    labeler: TraversabilityLabeler,
    output_dir: Path,
    accum_window: int = 20,
    method: str = "accumulated",
    grid_resolution: float = 0.1,
) -> None:
    if method not in ("accumulated", "by_range", "grid"):
        raise ValueError(f"Unknown labeling method '{method}'. Use 'accumulated', 'by_range', or 'grid'.")

    n = len(seq)
    n_target = len(seq.target_indices) if seq.target_indices is not None else n
    print(f"  Sequence {seq.name}: {n} scans total, {n_target} to label  [method={method}]", flush=True)

    if not seq.has_poses():
        raise RuntimeError(
            f"Sequence {seq.name} has no poses — cannot label without odometry."
        )
    poses = seq.poses

    if len(poses) != n:
        raise ValueError(
            f"Pose count ({len(poses)}) != scan count ({n}) for sequence {seq.name}."
        )

    seq_out = output_dir / seq.name
    seq_out.mkdir(parents=True, exist_ok=True)

    MAX_PTS_PER_PAST_SCAN = 20_000
    trav_count = 0
    total_pts  = 0

    pbar = tqdm(total=n_target, desc="  Labeling      ", unit="scan")

    if method == "by_range":
        # Only the current scan is needed — no context window.
        for i, cloud_file in enumerate(seq.cloud_files):
            if seq.target_indices is not None and i not in seq.target_indices:
                continue
            pbar.update(1)
            xyz, _ = seq.get_scan(i)
            labels_i = labeler.label_scan_by_range(xyz, poses, i)
            out_path = seq_out / (Path(cloud_file).stem + ".trav")
            labels_i.tofile(str(out_path))
            trav_count += int(labels_i.sum())
            total_pts  += len(labels_i)

    elif method == "accumulated":
        # Sliding window: keep at most accum_window+1 scans in memory.
        # Non-target scans are still loaded as accumulation context.
        scan_window: dict = {}
        for i, cloud_file in enumerate(seq.cloud_files):
            xyz_i, _ = seq.get_scan(i)
            scan_window[i] = xyz_i
            for k in [k for k in list(scan_window) if k < i - accum_window]:
                del scan_window[k]

            if seq.target_indices is not None and i not in seq.target_indices:
                continue
            pbar.update(1)

            T_scan_world = np.linalg.inv(poses[i])
            all_xyz      = [xyz_i]
            all_origins  = [np.full(len(xyz_i), i, dtype=np.int32)]

            for k in range(max(0, i - accum_window), i):
                xyz_k = scan_window.get(k)
                if xyz_k is None:
                    continue
                if labeler.forward_accum:
                    fmask = labeler.forward_mask(xyz_k, poses, k)
                    xyz_k = xyz_k[fmask]
                if len(xyz_k) == 0:
                    continue
                step     = max(1, len(xyz_k) // MAX_PTS_PER_PAST_SCAN)
                xyz_k    = xyz_k[::step]
                T_scan_k = T_scan_world @ poses[k]
                R, t     = T_scan_k[:3, :3], T_scan_k[:3, 3]
                all_xyz.append(((R @ xyz_k.T).T + t).astype(np.float32))
                all_origins.append(np.full(len(xyz_k), k, dtype=np.int32))

            xyz_acc      = np.vstack(all_xyz)
            scan_origins = np.concatenate(all_origins)
            labels_acc   = labeler.label_accumulated(xyz_acc, scan_origins, poses, i)
            labels_i     = labels_acc[:len(xyz_i)]

            out_path = seq_out / (Path(cloud_file).stem + ".trav")
            labels_i.tofile(str(out_path))
            trav_count += int(labels_i.sum())
            total_pts  += len(labels_i)

    else:
        # Reverse pass with 2D occupancy grid.
        # Complexity: O(N×M) vs O(N×W×M) for the other methods.
        grid, ox, oy = _build_grid(poses, grid_resolution, labeler.half_size)
        mb = grid.nbytes / 1024 / 1024
        print(f"    Grid: {grid.shape[0]}×{grid.shape[1]} cells "
              f"@ {grid_resolution}m resolution  ({mb:.1f} MB)", flush=True)

        for i in range(n - 1, -1, -1):
            # Sliding-window update: add pose i+1, remove pose i+trajectory_window+1
            j_add = i + 1
            if j_add < n:
                _update_footprint(grid, poses[j_add], labeler, grid_resolution, ox, oy, +1)
            j_remove = i + labeler.trajectory_window + 1
            if j_remove < n:
                _update_footprint(grid, poses[j_remove], labeler, grid_resolution, ox, oy, -1)

            if seq.target_indices is not None and i not in seq.target_indices:
                continue
            pbar.update(1)

            xyz, _ = seq.get_scan(i)
            labels_i = _query_grid(grid, xyz, poses[i], labeler, grid_resolution, ox, oy)
            out_path = seq_out / (Path(seq.cloud_files[i]).stem + ".trav")
            labels_i.tofile(str(out_path))
            trav_count += int(labels_i.sum())
            total_pts  += len(labels_i)

    pbar.close()
    pct = 100.0 * trav_count / max(total_pts, 1)
    print(f"    Saved {n_target} label files -> {seq_out}")
    print(f"    Traversable: {trav_count}/{total_pts} pts ({pct:.1f} %)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label traversability in LiDAR sequences."
    )
    parser.add_argument("--config",  default="configs/example_rellis.yaml", help="Config YAML")
    parser.add_argument("--output",  default=None, help="Output directory (overrides config output_dir)")
    parser.add_argument(
        "--dataset", default=None, choices=["rellis", "tartandrive"],
        help="Dataset mode: 'rellis' or 'tartandrive' (reads root from config data.source)",
    )
    # RELLIS-3D split mode
    parser.add_argument("--split",   default=None,
                        help="RELLIS-3D split: train / val / test / all  "
                             "(all = every sequence, every frame, no .lst filtering)")
    # Single-sequence mode
    parser.add_argument("--seq",     default=None, help="Path to a single sequence directory")
    parser.add_argument(
        "--cloud-subdir", default=None,
        help=(
            f"Sub-directory with *.bin files inside --seq "
            f"(default: auto-detect RELLIS '{RELLIS_CLOUD_SUBDIR}', then 'velodyne')"
        ),
    )
    args = parser.parse_args()

    # Back-compat: --split implies --dataset rellis
    if args.split is not None and args.dataset is None:
        args.dataset = "rellis"

    if args.seq is None and args.dataset is None:
        parser.error(
            "Provide one of: --dataset rellis [--split ...], --dataset tartandrive, "
            "or --seq (single sequence)."
        )

    cfg = load_config(args.config)

    max_rad     = _get(cfg, "data",  "max_rad",     default=50.0)
    robot_shape = _get(cfg, "robot", "shape",       default="square")
    robot_size  = _get(cfg, "robot", "size",        default=1.0)
    height_min  = _get(cfg, "robot", "height_min",  default=-0.5)
    height_max  = _get(cfg, "robot", "height_max",  default=0.3)

    method      = _get(cfg, "setting", "method",       default="accumulated")

    traj_window   = _get(cfg, "setting", "accumulated", "trajectory_window", default=100)
    accum_window  = _get(cfg, "setting", "accumulated", "accum_window",      default=20)
    forward_accum = _get(cfg, "setting", "accumulated", "forward_accum",     default=False)

    lidar_range   = _get(cfg, "setting", "by_range", "lidar_range", default=None)

    grid_traj_window = _get(cfg, "setting", "grid", "trajectory_window", default=traj_window)
    grid_resolution  = _get(cfg, "setting", "grid", "resolution",        default=0.1)

    robot = Robot(
        shape=robot_shape, size=robot_size,
        height_min=height_min, height_max=height_max,
    )
    active_traj_window = grid_traj_window if method == "grid" else traj_window
    labeler = TraversabilityLabeler(
        robot_shape=robot_shape,
        robot_size=robot_size,
        height_min=height_min,
        height_max=height_max,
        trajectory_window=active_traj_window,
        forward_accum=forward_accum,
        lidar_range=lidar_range,
    )
    output_dir = Path(args.output) if args.output is not None else Path(
        _get(cfg, "setting", "output_dir", default="output/labels")
    )

    print(f"Robot         : shape={robot_shape}  size={robot_size} m")
    print(f"Forward accum : {forward_accum}")
    if method == "by_range":
        extra = f"  lidar_range={lidar_range} m"
    elif method == "grid":
        extra = f"  trajectory_window={grid_traj_window}  resolution={grid_resolution} m"
    else:
        extra = f"  trajectory_window={traj_window}  accum_window={accum_window}"
    print(f"Method        : {method}{extra}")
    print(f"Output        : {output_dir}\n")

    if args.seq is not None:
        # Single-sequence mode: works with any KITTI-format dataset
        seq_path = Path(args.seq)
        cloud_subdir = args.cloud_subdir
        if cloud_subdir is None:
            cloud_subdir = (
                RELLIS_CLOUD_SUBDIR
                if (seq_path / RELLIS_CLOUD_SUBDIR).exists()
                else "velodyne"
            )
        seq = KittiSequence(
            cloud_dir=str(seq_path / cloud_subdir),
            poses_file=str(seq_path / "poses.txt"),
            max_rad=max_rad,
            robot=robot,
        )
        process_sequence(seq, labeler, output_dir, accum_window, method, grid_resolution)

    elif args.dataset == "tartandrive":
        root_dir           = _get(cfg, "data", "source",             default="data/tartandrive_data/")
        lidar_subdir       = _get(cfg, "data", "lidar_subdir",       default="livox")
        odom_subdir        = _get(cfg, "data", "odom_subdir",        default="super_odom")
        lidar_poses_subdir = _get(cfg, "data", "lidar_poses_subdir", default="lidar_poses")
        tf_file            = _get(cfg, "data", "static_tf_file",     default=None)
        tf_key             = _get(cfg, "data", "lidar_tf_key",       default=None)
        T_vehicle_lidar = (
            load_vehicle_lidar_tf(tf_file, tf_key)
            if tf_file and tf_key else None
        )
        dataset = TartanDrive(root_dir=root_dir, lidar_subdir=lidar_subdir,
                              odom_subdir=odom_subdir, lidar_poses_subdir=lidar_poses_subdir,
                              max_rad=max_rad, robot=robot, T_vehicle_lidar=T_vehicle_lidar)
        print(f"Dataset : TartanDrive  root={root_dir}  lidar={lidar_subdir}  odom={odom_subdir}  ({len(dataset)} sequence(s))")
        for seq in dataset:
            process_sequence(seq, labeler, output_dir, accum_window, method, grid_resolution)

    else:
        # RELLIS-3D split mode
        root_dir = _get(cfg, "data", "source", default="data/rellis/")
        split    = args.split or "all"
        dataset  = Rellis3D(root_dir=root_dir, split=split, max_rad=max_rad, robot=robot)
        print(f"Dataset : RELLIS-3D  root={root_dir}  split={split}  ({len(dataset)} sequence(s))")
        for seq in dataset:
            process_sequence(seq, labeler, output_dir, accum_window, method, grid_resolution)

    print("\nDone.")


if __name__ == "__main__":
    main()
