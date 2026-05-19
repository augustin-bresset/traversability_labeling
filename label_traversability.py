"""
Traversability labeling pipeline for RELLIS-3D.

For each scan in a sequence:
  1. Load the point cloud.
  2. Load or compute robot poses (ICP if icp_required=True).
  3. Label each point as traversable (1) or not (0) based on robot footprint
     along the trajectory window.
  4. Save labels as <stem>.trav (binary uint8 array, same order as .bin points).

Usage:
    python label_traversability.py [--config configs/example.yaml]
                                   [--split train]
                                   [--output output/labels]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.rellis import Rellis3D, Rellis3DSequence
from src.traversability.labeler import TraversabilityLabeler
from src.traversability.icp import compute_sequence_poses


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
    seq: Rellis3DSequence,
    labeler: TraversabilityLabeler,
    icp_required: bool,
    output_dir: Path,
    accum_window: int = 20,
) -> None:
    n = len(seq)
    print(f"  Sequence {seq.name}: {n} scans", flush=True)

    # Load all scans up-front (needed for ICP and label loop).
    scans_xyz = []
    for i in range(n):
        xyz, _ = seq.get_scan(i)
        scans_xyz.append(xyz)

    # Resolve poses.
    if seq.has_poses():
        poses = seq.poses
    elif icp_required:
        print(f"    No poses.txt found — computing poses via ICP ...", flush=True)
        poses = compute_sequence_poses(scans_xyz)
    else:
        raise RuntimeError(
            f"Sequence {seq.name} has no poses.txt and icp_required=False in config.\n"
            "Either provide a poses.txt file or set icp_required: True."
        )

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
        xyz_i, _ = scans_xyz[i], None  # already loaded

        # Accumulate past scans into scan i's frame
        T_scan_world = np.linalg.inv(poses[i])
        all_xyz = [scans_xyz[i]]

        for k in range(max(0, i - accum_window), i):
            xyz_k = scans_xyz[k]
            step  = max(1, len(xyz_k) // MAX_PTS_PER_PAST_SCAN)
            xyz_k = xyz_k[::step]
            T_scan_k = T_scan_world @ poses[k]
            R, t = T_scan_k[:3, :3], T_scan_k[:3, 3]
            all_xyz.append(((R @ xyz_k.T).T + t).astype(np.float32))

        xyz_acc = np.vstack(all_xyz)

        # Build scan_origins: current scan = i, past scan k → k
        scan_origins = np.concatenate([
            np.full(len(scans_xyz[i]), i, dtype=np.int32),
            *[
                np.full(
                    max(1, len(scans_xyz[k]) // MAX_PTS_PER_PAST_SCAN),
                    k,
                    dtype=np.int32,
                )
                for k in range(max(0, i - accum_window), i)
            ],
        ])

        # label_accumulated only uses poses AFTER each point's origin scan,
        # preventing dynamic objects at former robot positions being mislabeled.
        labels_acc = labeler.label_accumulated(xyz_acc, scan_origins, poses, i)
        labels_i   = labels_acc[:len(scans_xyz[i])]

        out_path = seq_out / (Path(cloud_file).stem + ".trav")
        labels_i.tofile(str(out_path))

        trav_count += int(labels_i.sum())
        total_pts  += len(labels_i)

    pct = 100.0 * trav_count / max(total_pts, 1)
    print(f"    Saved {n} label files → {seq_out}")
    print(f"    Traversable: {trav_count}/{total_pts} pts ({pct:.1f} %)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label traversability in RELLIS-3D point clouds."
    )
    parser.add_argument("--config",  default="configs/example.yaml", help="Config YAML")
    parser.add_argument("--split",   default="train",                 help="train / val / test")
    parser.add_argument("--output",  default="output/labels",         help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    root_dir     = _get(cfg, "data",    "source",       default="data/rellis/")
    max_rad      = _get(cfg, "data",    "max_rad",       default=50.0)
    icp_required      = _get(cfg, "setting", "icp_required",    default=False)
    forward_labeling  = _get(cfg, "setting", "forward_labeling", default=False)
    forward_dist      = _get(cfg, "setting", "forward_dist",     default=5.0)

    robot_shape  = _get(cfg, "robot", "shape",           default="square")
    robot_size   = _get(cfg, "robot", "size",            default=1.0)
    height_min   = _get(cfg, "robot", "height_min",      default=-0.5)
    height_max   = _get(cfg, "robot", "height_max",      default=0.3)
    traj_window  = _get(cfg, "robot", "trajectory_window", default=100)
    accum_window = _get(cfg, "robot", "accum_window",      default=20)

    dataset = Rellis3D(root_dir=root_dir, split=args.split, max_rad=max_rad)
    labeler = TraversabilityLabeler(
        robot_shape=robot_shape,
        robot_size=robot_size,
        height_min=height_min,
        height_max=height_max,
        trajectory_window=traj_window,
        use_forward_labeling=forward_labeling,
        forward_dist=forward_dist,
    )
    output_dir = Path(args.output)

    print(f"Dataset : {root_dir}  split={args.split}  ({len(dataset)} sequence(s))")
    print(f"Robot   : shape={robot_shape}  size={robot_size} m")
    print(f"ICP     : {icp_required}")
    print(f"Forward : {forward_labeling}  dist={forward_dist} m")
    print(f"Output  : {output_dir}\n")

    for seq in dataset:
        process_sequence(seq, labeler, icp_required, output_dir, accum_window)

    print("\nDone.")


if __name__ == "__main__":
    main()
