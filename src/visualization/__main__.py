"""
CLI entry point for the traversability viewer.

Usage:
    # RELLIS-3D (auto-detects os1_cloud_node_kitti_bin layout):
    python -m src.visualization --seq data/rellis/00000 --config configs/example_rellis.yaml

    # TartanDrive (pass dataset root or specific sequence dir):
    python -m src.visualization --seq data/tartandrive_data/ --config configs/example_tartan.yaml

    # Generic KITTI dataset (point clouds in velodyne/):
    python -m src.visualization --seq /data/kitti/00 --cloud-subdir velodyne

    # From pre-computed .trav files:
    python -m src.visualization --seq data/rellis/00000 --labels output/labels/00000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.datasets.kitti_sequence import KittiSequence
from src.datasets.rellis import RELLIS_CLOUD_SUBDIR
from src.datasets.tartandrive import TartanDriveSequence, load_vehicle_lidar_tf
from src.robot import Robot
from src.traversability.labeler import TraversabilityLabeler
from src.visualization.viewer import TraversabilityViewer


def _get(cfg: dict, *keys, default=None):
    val = cfg
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val


def _find_tartandrive_seq(root: Path, lidar_subdir: str) -> Path | None:
    """Return the first sequence directory that contains <lidar_subdir>/*.npy."""
    # Direct match
    if any((root / lidar_subdir).glob("*.npy")):
        return root
    # One level deep (e.g. root/<seq>/livox/)
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir() and any((candidate / lidar_subdir).glob("*.npy")):
            return candidate
    # Two levels deep (TartanDrive double-named structure: root/<seq>/<seq>/livox/)
    for top in sorted(root.iterdir()):
        if not top.is_dir():
            continue
        for candidate in sorted(top.iterdir()):
            if candidate.is_dir() and any((candidate / lidar_subdir).glob("*.npy")):
                return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive traversability viewer for LiDAR sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RELLIS-3D (auto-detects layout):
  python -m src.visualization --seq data/rellis/00000 --config configs/example_rellis.yaml

  # TartanDrive (dataset root or specific seq dir):
  python -m src.visualization --seq data/tartandrive_data/ --config configs/example_tartan.yaml

  # Generic KITTI dataset:
  python -m src.visualization --seq /data/kitti/sequences/00 --cloud-subdir velodyne

  # From pre-computed .trav files:
  python -m src.visualization --seq data/rellis/00000 \\
      --labels output/labels/00000 --config configs/example_rellis.yaml

  # Start at a specific scan index:
  python -m src.visualization --seq data/rellis/00000 --idx 100
        """,
    )
    parser.add_argument("--seq",    required=True, help="Path to the sequence directory (or dataset root for TartanDrive)")
    parser.add_argument("--config", default="configs/example_rellis.yaml", help="Config YAML")
    parser.add_argument("--labels", default=None, help="Directory with pre-computed .trav label files (optional)")
    parser.add_argument("--idx",    type=int, default=0, help="Starting scan index (default: 0)")
    parser.add_argument(
        "--cloud-subdir", default=None,
        help=(
            "Sub-directory containing point cloud files relative to --seq. "
            f"Auto-detected: RELLIS '{RELLIS_CLOUD_SUBDIR}', TartanDrive npy, then 'velodyne'."
        ),
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    max_rad       = _get(cfg, "data",    "max_rad",           default=50.0)
    forward_accum = _get(cfg, "setting", "forward_accum",     default=False)
    robot_shape   = _get(cfg, "robot",   "shape",             default="square")
    robot_size    = _get(cfg, "robot",   "size",              default=1.0)
    height_min    = _get(cfg, "robot",   "height_min",        default=-0.5)
    height_max    = _get(cfg, "robot",   "height_max",        default=0.3)
    traj_window   = _get(cfg, "robot",   "trajectory_window", default=100)
    robot = Robot(shape=robot_shape, size=robot_size,
                  height_min=height_min, height_max=height_max)
    lidar_subdir_cfg       = _get(cfg, "data", "lidar_subdir",        default=None)
    odom_subdir_cfg        = _get(cfg, "data", "odom_subdir",         default="super_odom")
    lidar_poses_subdir_cfg = _get(cfg, "data", "lidar_poses_subdir",  default="lidar_poses")
    tf_file_cfg            = _get(cfg, "data", "static_tf_file",      default=None)
    tf_key_cfg             = _get(cfg, "data", "lidar_tf_key",        default=None)
    T_vehicle_lidar  = (
        load_vehicle_lidar_tf(tf_file_cfg, tf_key_cfg)
        if tf_file_cfg and tf_key_cfg else None
    )

    seq_path     = Path(args.seq)
    cloud_subdir = args.cloud_subdir
    use_npy      = False  # TartanDrive flag

    if cloud_subdir is None:
        # 1. RELLIS layout
        for candidate in (seq_path, seq_path.parent / "Rellis-3D" / seq_path.name):
            if (candidate / RELLIS_CLOUD_SUBDIR).exists():
                seq_path = candidate
                cloud_subdir = RELLIS_CLOUD_SUBDIR
                break

        # 2. TartanDrive npy layout — check subdir from config first, then "livox"
        if cloud_subdir is None:
            for lidar_sub in filter(None, [lidar_subdir_cfg, "livox"]):
                found = _find_tartandrive_seq(seq_path, lidar_sub)
                if found is not None:
                    seq_path = found
                    cloud_subdir = lidar_sub
                    use_npy = True
                    break

        # 3. Standard KITTI velodyne/
        if cloud_subdir is None:
            cloud_subdir = "velodyne"
    else:
        # Explicit subdir: detect format by file extension
        cloud_dir_explicit = seq_path / cloud_subdir
        if cloud_dir_explicit.is_dir() and any(cloud_dir_explicit.glob("*.npy")):
            use_npy = True

    print(f"Loading sequence : {seq_path}")
    print(f"  Cloud sub-dir  : {cloud_subdir}  ({'npy' if use_npy else 'bin'})")

    if use_npy:
        seq = TartanDriveSequence(
            seq_dir=str(seq_path),
            lidar_subdir=cloud_subdir,
            odom_subdir=odom_subdir_cfg,
            lidar_poses_subdir=lidar_poses_subdir_cfg,
            max_rad=max_rad,
            robot=robot,
            T_vehicle_lidar=T_vehicle_lidar,
        )
    else:
        seq = KittiSequence(
            cloud_dir=str(seq_path / cloud_subdir),
            poses_file=str(seq_path / "poses.txt"),
            max_rad=max_rad,
            robot=robot,
        )

    print(f"  {len(seq)} scans found.")

    # Resolve poses
    poses = None
    if seq.has_poses():
        poses = seq.poses
        print(f"  Poses loaded ({len(poses)} entries).")
    elif args.labels is not None:
        print("  No poses — will load labels from files.")
    else:
        print(
            "  WARNING: no poses found.\n"
            "  Trajectory and on-the-fly labels will be unavailable.\n"
            "  Use --labels to load pre-computed .trav files."
        )

    labeler = TraversabilityLabeler(
        robot_shape=robot_shape,
        robot_size=robot_size,
        height_min=height_min,
        height_max=height_max,
        trajectory_window=traj_window,
        forward_accum=forward_accum,
    )

    label_dir = Path(args.labels) if args.labels else None

    print("Launching viewer …")
    TraversabilityViewer.launch(
        seq=seq,
        poses=poses,
        labeler=labeler,
        label_dir=label_dir,
        robot_shape=robot_shape,
        robot_size=robot_size,
        start_idx=args.idx,
    )


if __name__ == "__main__":
    main()
