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
from pathlib import Path

import numpy as np
import yaml

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
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(
    seq,
    labeler: TraversabilityLabeler,
    output_dir: Path,
    accum_window: int = 20,
) -> None:
    n = len(seq)
    n_target = len(seq.target_indices) if seq.target_indices is not None else n
    print(f"  Sequence {seq.name}: {n} scans total, {n_target} to label", flush=True)

    if not seq.has_poses():
        raise RuntimeError(
            f"Sequence {seq.name} has no poses — cannot label without odometry."
        )
    poses = seq.poses

    # Load all scans up-front — non-target frames are still needed as
    # accumulation context for nearby target frames.
    scans_xyz = []
    for i in range(n):
        xyz, _ = seq.get_scan(i)
        scans_xyz.append(xyz)

    if len(poses) != n:
        raise ValueError(
            f"Pose count ({len(poses)}) != scan count ({n}) for sequence {seq.name}."
        )

    seq_out = output_dir / seq.name
    seq_out.mkdir(parents=True, exist_ok=True)

    # How many past scans to accumulate before labeling.
    # The LiDAR has a blind spot directly under the vehicle at time t.
    # Past scans saw that ground before the robot reached it, so accumulating
    # them fills the gap and produces correct traversability labels.
    MAX_PTS_PER_PAST_SCAN = 20_000

    trav_count = 0
    total_pts  = 0

    for i, cloud_file in enumerate(seq.cloud_files):
        if seq.target_indices is not None and i not in seq.target_indices:
            continue  # not in split — skip computation and output

        T_scan_world = np.linalg.inv(poses[i])
        all_xyz      = [scans_xyz[i]]
        all_origins  = [np.full(len(scans_xyz[i]), i, dtype=np.int32)]

        for k in range(max(0, i - accum_window), i):
            xyz_k = scans_xyz[k]

            # Forward filter: drop points behind the robot at scan k.
            if labeler.forward_accum:
                fmask = labeler.forward_mask(xyz_k, poses, k)
                xyz_k = xyz_k[fmask]
            if len(xyz_k) == 0:
                continue

            step  = max(1, len(xyz_k) // MAX_PTS_PER_PAST_SCAN)
            xyz_k = xyz_k[::step]
            T_scan_k = T_scan_world @ poses[k]
            R, t = T_scan_k[:3, :3], T_scan_k[:3, 3]
            all_xyz.append(((R @ xyz_k.T).T + t).astype(np.float32))
            all_origins.append(np.full(len(xyz_k), k, dtype=np.int32))

        xyz_acc      = np.vstack(all_xyz)
        scan_origins = np.concatenate(all_origins)

        # label_accumulated only uses poses AFTER each point's origin scan,
        # preventing dynamic objects at former robot positions being mislabeled.
        labels_acc = labeler.label_accumulated(xyz_acc, scan_origins, poses, i)
        labels_i   = labels_acc[:len(scans_xyz[i])]

        out_path = seq_out / (Path(cloud_file).stem + ".trav")
        labels_i.tofile(str(out_path))

        trav_count += int(labels_i.sum())
        total_pts  += len(labels_i)

    pct = 100.0 * trav_count / max(total_pts, 1)
    print(f"    Saved {n} label files -> {seq_out}")
    print(f"    Traversable: {trav_count}/{total_pts} pts ({pct:.1f} %)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label traversability in LiDAR sequences."
    )
    parser.add_argument("--config",  default="configs/example_rellis.yaml", help="Config YAML")
    parser.add_argument("--output",  default="output/labels",               help="Output directory")
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

    max_rad       = _get(cfg, "data",    "max_rad",            default=50.0)
    forward_accum = _get(cfg, "setting", "forward_accum",      default=False)
    robot_shape   = _get(cfg, "robot",   "shape",              default="square")
    robot_size    = _get(cfg, "robot",   "size",               default=1.0)
    height_min    = _get(cfg, "robot",   "height_min",         default=-0.5)
    height_max    = _get(cfg, "robot",   "height_max",         default=0.3)
    traj_window   = _get(cfg, "robot",   "trajectory_window",  default=100)
    accum_window  = _get(cfg, "robot",   "accum_window",       default=20)

    robot = Robot(
        shape=robot_shape, size=robot_size,
        height_min=height_min, height_max=height_max,
    )
    labeler = TraversabilityLabeler(
        robot_shape=robot_shape,
        robot_size=robot_size,
        height_min=height_min,
        height_max=height_max,
        trajectory_window=traj_window,
        forward_accum=forward_accum,
    )
    output_dir = Path(args.output)

    print(f"Robot         : shape={robot_shape}  size={robot_size} m")
    print(f"Forward accum : {forward_accum}")
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
        process_sequence(seq, labeler, output_dir, accum_window)

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
            process_sequence(seq, labeler, output_dir, accum_window)

    else:
        # RELLIS-3D split mode
        root_dir = _get(cfg, "data", "source", default="data/rellis/")
        split    = args.split or "all"
        dataset  = Rellis3D(root_dir=root_dir, split=split, max_rad=max_rad, robot=robot)
        print(f"Dataset : RELLIS-3D  root={root_dir}  split={split}  ({len(dataset)} sequence(s))")
        for seq in dataset:
            process_sequence(seq, labeler, output_dir, accum_window)

    print("\nDone.")


if __name__ == "__main__":
    main()
