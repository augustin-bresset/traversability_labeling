# Traversability Labeling

Given the geometry of your robot, labels as traversable the points in the point cloud through which its trajectory passes.

## Quick start

### 1. Set up the environment

```bash
conda env create -f environment.yml   # create from scratch (first time)
conda activate trav_label
```

Or, if the `trav_label` environment already exists and you just want to sync dependencies:

```bash
conda activate trav_label
pip install numpy scipy open3d pyyaml matplotlib pytest
```

### 2. Configure

Edit the appropriate config to match your robot and dataset:

**RELLIS-3D** (`configs/example_rellis.yaml`):
```yaml
robot:
  shape: "square"
  size: 1.0

data:
  source: "data/rellis/"

setting:
  forward_accum: True
```

Expected layout:
```
data/rellis/
├── Rellis-3D/
│   ├── 00000/
│   │   ├── os1_cloud_node_kitti_bin/   # *.bin point clouds (XYZI float32)
│   │   └── poses.txt                   # KITTI-format poses (12 values/line)
│   └── 00001/ ...
├── pt_train.lst
└── pt_val.lst
```

**TartanDrive** (`configs/example_tartan.yaml`):
```yaml
data:
  source: "data/tartandrive_data/"
  lidar_subdir: "livox"   # change for other lidars
```

Expected layout:
```
data/tartandrive_data/
└── <recording>/
    └── <recording>/
        ├── livox/          # XXXXXX.npy (N,3) + XXXXXX_intensity.npy (N,)
        └── current_position/
```

### 3. Label traversability

```bash
# RELLIS-3D
python label_traversability.py --dataset rellis --split train --config configs/example_rellis.yaml

# TartanDrive
python label_traversability.py --dataset tartandrive --config configs/example_tartan.yaml
```

For each scan a binary `.trav` file (uint8, same point order as the input cloud) is written under `output/labels/<seq_id>/`.

Poses are required — provide a `poses.txt` (RELLIS) or equivalent odometry file. Without poses the pipeline will raise an error.

### 4. Visualise

```bash
# RELLIS-3D:
python -m src.visualization --seq data/rellis/Rellis-3D/00000 --config configs/example_rellis.yaml

# TartanDrive (pass dataset root, sequence is auto-detected):
python -m src.visualization --seq data/tartandrive_data/ --config configs/example_tartan.yaml

# From pre-computed .trav files:
python -m src.visualization --seq data/rellis/Rellis-3D/00000 \
    --labels output/labels/00000 --config configs/example_rellis.yaml

# Start at a specific scan:
python -m src.visualization --seq data/rellis/Rellis-3D/00000 --idx 50
```

**Viewer controls:**

| Key | Action |
|-----|--------|
| `->` / `L` | Next scan |
| `<-` / `H` | Previous scan |
| `T` | Cycle colour mode (Traversability / Intensity / Height) |
| `J` | Toggle trajectory (past + future) |
| `K` | Toggle robot footprint |
| `M` | Toggle traversable-point trail recording |
| `V` | Toggle forward-accumulation filter |
| `N` | Toggle forward-mask preview (show only forward-seen points) |
| `F` | Top-down view |
| `E` | LiDAR first-person view (camera at sensor, looking forward) |
| `R` | Reset camera |

**Legend panel** - each legend entry has a checkbox to show/hide that category independently (traversable points, ground, other points, past/future trajectory, robot footprint, traversable trail).

**Accumulate scans panel** - set *N* (how many past scans to stack) and *step* (every K-th scan).  With N=10 and step=5, the viewer stacks 10 scans spaced 5 apart, reaching 45 scans into the past without multiplying the point count.

**Forward accumulation** - when enabled, only points that were *ahead* of the robot at the time of each past scan are included in the accumulated cloud, preventing people following the vehicle from appearing as traversable.

**Traversable trail** - when recording is on, traversable points from each visited scan are accumulated in world coordinates (magenta). Useful to visualise the complete traversed path across the sequence. Capped at 500 k points; use *Clear* to reset.

