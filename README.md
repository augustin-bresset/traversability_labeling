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

### 3. Preprocessing — compute poses (if not already available)

Poses are required for labeling. If your sequence does not already have them, two preprocessing scripts are available.

#### Option A — KISS-ICP odometry (TartanDrive / any `.npy` sequence)

Builds a full trajectory with [KISS-ICP](https://github.com/PRBonn/kiss-icp). Much faster than Open3D GICP (~100×).

```bash
python -m src.preprocessing.gicp_odometry <seq_dir> \
    --lidar-subdir velodyne_1 \
    --out-subdir   gicp_poses \
    --voxel-size   1.0 \
    --max-range    50.0
```

Optional GPS z-correction (replaces KISS-ICP z, which drifts on flat terrain):

```bash
python -m src.preprocessing.gicp_odometry <seq_dir> \
    --lidar-subdir    velodyne_1 \
    --gps-odom-subdir gps_odom \
    --out-subdir      gicp_poses
```

Output written to `<seq_dir>/gicp_poses/`:
```
poses.npy        (N, 4, 4) float64 — T_world_lidar per frame
valid_mask.npy   (N,)      bool
timestamps.txt   (N,)
```

Then set `data.lidar_poses_subdir: "gicp_poses"` in the config.

#### Option B — IMU/odometry pose sync (TartanDrive)

Interpolates existing odometry poses to LiDAR timestamps using gyroscope integration for rotation and linear interpolation for position.  Useful when a GPS/wheel-odom source already provides a trajectory but is not synchronized with the LiDAR.

```bash
python -m src.preprocessing.imu_pose_sync <seq_dir> \
    --lidar-subdir velodyne_1 \
    --odom-subdir  gps_odom \
    --out-subdir   lidar_poses
```

Output written to `<seq_dir>/lidar_poses/`:
```
poses.npy        (N, 4, 4) float64 — NaN rows for invalid frames
valid_mask.npy   (N,)      bool    — False for out-of-range frames
timestamps.txt   (N,)
```

Then set `data.lidar_poses_subdir: "lidar_poses"` in the config.

---

**RELLIS-3D** — poses come directly from the `poses.txt` file included in the dataset (KITTI format, 12 values per line). No preprocessing needed.

---

### 4. Label traversability (batch)

```bash
# RELLIS-3D
python label_traversability.py --dataset rellis --split train --config configs/example_rellis.yaml

# TartanDrive
python label_traversability.py --dataset tartandrive --config configs/example_tartan.yaml
```

For each scan a binary `.trav` file (uint8, same point order as the input cloud) is written under `output/labels/<seq_id>/`.

Poses are required — provide a `poses.txt` (RELLIS) or equivalent odometry file. Without poses the pipeline will raise an error.

### 5. Visualise

```bash
# RELLIS-3D:
python -m src.visualization --seq data/rellis/Rellis-3D/00000 --config configs/example_rellis.yaml

# TartanDrive (pass dataset root, sequence is auto-detected):
python -m src.visualization --seq data/tartandrive_data/ --config configs/example_tartan.yaml

# From pre-computed .trav files:
python -m src.visualization --seq data/rellis/Rellis-3D/00000 \
    --labels output/labels/00000 --config configs/example_rellis.yaml

python -m src.visualization --seq data/tartandrive_data/ \
--labels output/output/labels/ --config configs/example_tartan.yaml

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

