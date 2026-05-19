"""
CLI entry point for the traversability viewer.

Usage:
    python -m src.visualization --seq data/rellis/00000 --config configs/example.yaml
    python -m src.visualization --seq data/rellis/00000 --config configs/example.yaml \\
        --labels output/labels/00000 --idx 50

If the sequence has a poses.txt file, labels are computed on-the-fly.
Without poses, the viewer shows only the raw point cloud (use --labels to
load pre-computed .trav files instead).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.datasets.rellis import Rellis3DSequence
from src.traversability.labeler import TraversabilityLabeler
from src.traversability.icp import compute_sequence_poses
from src.visualization.viewer import TraversabilityViewer


def _get(cfg: dict, *keys, default=None):
    val = cfg
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive traversability viewer for RELLIS-3D sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # On-the-fly labeling (requires poses.txt in the sequence directory):
  python -m src.visualization --seq data/rellis/00000 --config configs/example.yaml

  # From pre-computed .trav files:
  python -m src.visualization --seq data/rellis/00000 --config configs/example.yaml \\
      --labels output/labels/00000

  # Start at a specific scan index:
  python -m src.visualization --seq data/rellis/00000 --config configs/example.yaml --idx 100
        """,
    )
    parser.add_argument("--seq",    required=True, help="Path to the sequence directory (e.g. data/rellis/00000)")
    parser.add_argument("--config", default="configs/example.yaml", help="Config YAML (default: configs/example.yaml)")
    parser.add_argument("--labels", default=None, help="Directory with pre-computed .trav label files (optional)")
    parser.add_argument("--idx",    type=int, default=0, help="Starting scan index (default: 0)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    max_rad      = _get(cfg, "data",    "max_rad",          default=50.0)
    icp_required     = _get(cfg, "setting", "icp_required",     default=False)
    forward_labeling = _get(cfg, "setting", "forward_labeling", default=False)
    forward_dist     = _get(cfg, "setting", "forward_dist",     default=5.0)
    robot_shape  = _get(cfg, "robot",   "shape",            default="square")
    robot_size   = _get(cfg, "robot",   "size",             default=1.0)
    height_min   = _get(cfg, "robot",   "height_min",       default=-0.5)
    height_max   = _get(cfg, "robot",   "height_max",       default=0.3)
    traj_window  = _get(cfg, "robot",   "trajectory_window",default=100)

    # Load sequence — try path as-is, then with Rellis-3D/ inserted
    seq_path = Path(args.seq)
    if not (seq_path / "os1_cloud_node_kitti_bin").exists():
        candidate = seq_path.parent / "Rellis-3D" / seq_path.name
        if (candidate / "os1_cloud_node_kitti_bin").exists():
            seq_path = candidate
    print(f"Loading sequence: {seq_path}")
    seq = Rellis3DSequence(str(seq_path), max_rad=max_rad)
    print(f"  {len(seq)} scans found.")

    # Resolve poses
    poses = None
    if seq.has_poses():
        poses = seq.poses
        print(f"  Poses loaded from poses.txt ({len(poses)} entries).")
    elif args.labels is not None:
        print("  No poses.txt — will load labels from files.")
    elif icp_required:
        print("  No poses.txt — computing poses via ICP (this may take a moment) ...")
        scans_xyz = [seq.get_scan(i)[0] for i in range(len(seq))]
        poses = compute_sequence_poses(scans_xyz)
        print(f"  ICP done: {len(poses)} poses computed.")
    else:
        print(
            "  WARNING: no poses.txt found and icp_required=False.\n"
            "  Trajectory and on-the-fly labels will be unavailable.\n"
            "  Use --labels to load pre-computed .trav files."
        )

    labeler = TraversabilityLabeler(
        robot_shape=robot_shape,
        robot_size=robot_size,
        height_min=height_min,
        height_max=height_max,
        trajectory_window=traj_window,
        use_forward_labeling=forward_labeling,
        forward_dist=forward_dist,
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
