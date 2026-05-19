import os
from pathlib import Path
import argparse
from collections import defaultdict



def go_to_split(root, split: str, max_depth: int = 10) -> Path:
    """
    Recursively(BFS-like) find a directory named `split` under root. 

    Args:
        root: dataset root
        split: split name (train/val/test)
        max_depth: safety limit to avoid infinite traversal

    Returns:
        Path to split directory

    Raises:
        FileNotFoundError if not found
    """

    root = Path(root).resolve()

    direct = root / split
    if direct.exists() and direct.is_dir():
        return direct

    queue = [(root, 0)]

    while queue:
        current, depth = queue.pop(0)

        if depth > max_depth:
            continue

        try:
            for child in current.iterdir():
                if child.is_dir():

                    if child.name == split:
                        return child

                    queue.append((child, depth + 1))

        except PermissionError:
            continue

    return None


ANCHOR_FILES = {"LICENSE", "CHANGELOG"}
MAP_EXTENSIONS = {"csv", "yaml", "json"}

def is_dataset_root(p: Path) -> bool:
    try:
        files = [f.name.lower() for f in p.iterdir() if f.is_file()]
    except PermissionError:
        return False

    if any(f in files for f in ANCHOR_FILES):
        return True

    if any(
        "mapping" in f and f.split(".")[-1] in MAP_EXTENSIONS
        for f in files
    ):
        return True

    return False


def find_dataset_root(path: str, max_depth: int = 5):
    """
    Search ONLY downward for dataset root.

    Args:
        path: starting directory
        max_depth: limit recursion depth

    Returns:
        Path or None
    """
    root = Path(path).resolve()

    def dfs(current: Path, depth: int):
        if depth > max_depth:
            return None

        if is_dataset_root(current):
            return current

        try:
            for child in current.iterdir():
                if child.is_dir():
                    res = dfs(child, depth + 1)
                    if res:
                        return res
        except PermissionError:
            pass

        return None

    return dfs(root, 0)

IMAGE_EXTENSIONS = {"jpg", "png"}
RAW_DATA_EXTENSIONS = {"bin", "pt", "csv", "npy", "label", "height_label"}

def is_leaf_data_dir(dir_path: Path) -> bool:
    """ Detect leaf directories. 
    """
    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
    
    return len(subdirs) == 0

def _is_data_file(f: Path) -> bool:
    return f.suffix.replace(".", "").lower() in RAW_DATA_EXTENSIONS

TYPE_NAMES = {"lidar", "camera", "imu", "label", "height"}

def _infer_type_from_path(path: str) -> str:
    p = path.lower()
    for type_name in TYPE_NAMES:
        if type_name in p:
            return type_name
    
def find_raw_data_dirs(path: str, max_up=10) -> dict[str, str]:
    """Find directories that contains raw datas.
    
    Return:
        {dir_path : type}

    For example find_raw_data_dirs("Goose3D") return :
    {
        "Goose3D/train/lidar/lidar-01-01-2000/" : "lidar",
        "Goose3D/train/lidar/lidar-02-01-2000/" : "lidar",
        ...
        }
    """

    root = Path(path).resolve()

    # 2. scan filesystem
    candidates = defaultdict(list)

    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if not _is_data_file(f):
            continue

        parent = f.parent

        if not is_leaf_data_dir(parent):
            continue

        dir_path = str(f.parent)
        dtype = _infer_type_from_path(str(f))

        candidates[dir_path].append(dtype)

    # 3. aggregate folder type
    result = defaultdict(list)

    for dir_path, types in candidates.items():
        type_score = defaultdict(int)

        for t in types:
            type_score[t] += 1

        best_type = max(type_score.items(), key=lambda x: x[1])[0]

        if best_type in TYPE_NAMES:
            result[best_type].append(dir_path)

    return dict(result)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find raw data directories in a dataset")
    parser.add_argument("--path", required=True, help="Path to dataset (any subfolder works)")
    parser.add_argument("--max-up", type=int, default=10, help="Max levels to search upward for dataset root")

    args = parser.parse_args()

    result = find_raw_data_dirs(args.path, args.max_up)


    print("\n[INFO] Detected raw data directories:\n")
    for k in result.keys():
        print(f" -- {k}")
        for v in result[k]:
            print(f"\t\t - {v}")
    print(f"\n[INFO] Total: {len(result)} directories")