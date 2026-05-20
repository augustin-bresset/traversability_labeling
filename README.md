# Traversability Labeling

Given the geometry of your robot, labels as traversable the points in the point cloud through which its trajectory passes.

## Quick start

### 1. Install dependencies

```bash
conda activate trav_loss
pip install scipy open3d
```

### 2. Configure

Edit `configs/example.yaml` to match your robot and dataset:

```yaml
robot:
  shape: "square"   # "square" or "round"
  size: 1.0         # footprint side length / diameter in metres

data:
  source: "data/rellis/"   # path to RELLIS-3D root

setting:
  icp_required: False   # True if no poses.txt is available
```

Expected dataset layout:

```
data/rellis/
├── Rellis-3D/
│   ├── 00000/
│   │   ├── os1_cloud_node_kitti_bin/   # *.bin point clouds (XYZI float32)
│   │   └── poses.txt                   # optional - KITTI-format poses (12 values/line)
│   ├── 00001/ ...
├── pt_train.lst
└── pt_val.lst
```

### 3. Label traversability

```bash
python label_traversability.py --split train --output output/labels
```

For each scan a binary `.trav` file (uint8, same point order as the `.bin`) is written under `output/labels/<seq_id>/`.

If no `poses.txt` is present, set `icp_required: True` in the config to estimate poses from the point clouds via ICP.

### 4. Visualise

```bash
# On-the-fly labeling (requires poses.txt):
python -m src.visualization --seq data/rellis/00000 --config configs/example.yaml

# From pre-computed .trav files:
python -m src.visualization --seq data/rellis/00000 \
    --labels output/labels/00000 --config configs/example.yaml

# Start at a specific scan:
python -m src.visualization --seq data/rellis/00000 --config configs/example.yaml --idx 50
```

**Viewer controls:**

| Key | Action |
|-----|--------|
| `->` / `L` | Next scan |
| `<-` / `H` | Previous scan |
| `T` | Cycle colour mode (Traversability / Intensity / Height) |
| `J` | Toggle trajectory |
| `K` | Toggle robot footprint |
| `M` | Toggle traversable-point trail recording |
| `F` | Top-down view |
| `R` | Reset camera |

**Accumulate scans panel** - set *N* (how many past scans to stack) and *step* (every K-th scan).  With N=10 and step=5, the viewer stacks 10 scans spaced 5 apart, reaching 45 scans into the past without multiplying the point count.

**Traversable trail** - when recording is on, traversable points from each visited scan are accumulated in world coordinates (magenta). Useful to visualise the complete traversed path across the sequence. Capped at 500 k points; use *Clear* to reset.

