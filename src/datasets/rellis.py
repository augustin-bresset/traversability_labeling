"""
RELLIS-3D dataset loader.

Two labeling modes (controlled by the --split argument):
  split='train'/'val'/'test'  — parse the matching pt_<split>.lst file;
                                 only frames listed there are written to disk,
                                 but all frames are loaded for accumulation context.
  split='all'                 — discover every sequence under Rellis-3D/ and
                                 label all frames (no .lst filtering).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

from .kitti_sequence import KittiSequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELLIS_CLOUD_SUBDIR = "os1_cloud_node_kitti_bin"


# ---------------------------------------------------------------------------
# .lst parser
# ---------------------------------------------------------------------------

def _parse_lst(lst_file: Path) -> Dict[str, Set[str]]:
    """
    Parse a RELLIS-3D split list file.

    Each line is a relative path such as:
        Rellis-3D/00000/os1_cloud_node_kitti_bin/000000000.bin

    Returns a dict {seq_id: {file_stem, ...}}.
    The seq_id is the first 5-digit directory component found in the path.
    """
    result: Dict[str, Set[str]] = {}
    with open(lst_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = Path(line).parts
            seq_id = next((p for p in parts if len(p) == 5 and p.isdigit()), None)
            if seq_id is None:
                continue
            result.setdefault(seq_id, set()).add(Path(line).stem)
    return result


# ---------------------------------------------------------------------------
# Sequence wrapper
# ---------------------------------------------------------------------------

class Rellis3DSequence(KittiSequence):
    """
    One RELLIS-3D sequence (point clouds in os1_cloud_node_kitti_bin/).

    target_indices — set of frame indices to label (None = all frames).
    """

    def __init__(
        self,
        seq_dir: str,
        max_rad: float = 50.0,
        robot=None,
        target_indices: Optional[Set[int]] = None,
    ):
        self.seq_dir = Path(seq_dir)
        super().__init__(
            cloud_dir=str(self.seq_dir / RELLIS_CLOUD_SUBDIR),
            poses_file=str(self.seq_dir / "poses.txt"),
            max_rad=max_rad,
            robot=robot,
            target_indices=target_indices,
        )

    @property
    def name(self) -> str:
        return self.seq_dir.name


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Rellis3D:
    """
    RELLIS-3D dataset.

    split='train'/'val'/'test'
        Reads the corresponding pt_<split>.lst file from root_dir.
        Only the listed frames are labelled; all frames are still loaded
        for accumulation context.

    split='all'
        Finds every sequence directory under root_dir (with or without the
        intermediate Rellis-3D/ sub-directory) and labels every frame.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        max_rad: float = 50.0,
        robot=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_rad = max_rad
        self.robot = robot
        self.sequences: List[Rellis3DSequence] = []

        if split == "all":
            self._load_all(max_rad)
        else:
            self._load_from_lst(split, max_rad)

        if not self.sequences:
            raise FileNotFoundError(
                f"No RELLIS-3D sequences found in '{root_dir}' for split='{split}'."
            )

    # ------------------------------------------------------------------

    def _iter_seq_dirs(self):
        """Yield every sequence directory that contains the RELLIS cloud subdir."""
        for base in (self.root_dir / "Rellis-3D", self.root_dir):
            if not base.is_dir():
                continue
            for d in sorted(base.iterdir()):
                if d.is_dir() and (d / RELLIS_CLOUD_SUBDIR).exists():
                    yield d

    def _find_seq(self, seq_id: str) -> Optional[Path]:
        for candidate in (
            self.root_dir / "Rellis-3D" / seq_id,
            self.root_dir / seq_id,
        ):
            if (candidate / RELLIS_CLOUD_SUBDIR).exists():
                return candidate
        return None

    def _load_all(self, max_rad: float) -> None:
        for seq_dir in self._iter_seq_dirs():
            self.sequences.append(Rellis3DSequence(str(seq_dir), max_rad=max_rad, robot=self.robot))

    def _load_from_lst(self, split: str, max_rad: float) -> None:
        lst_file = self.root_dir / f"pt_{split}.lst"
        if not lst_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {lst_file}\n"
                f"Use split='all' to label every frame without a .lst file."
            )
        targets = _parse_lst(lst_file)
        for seq_id, stems in sorted(targets.items()):
            seq_dir = self._find_seq(seq_id)
            if seq_dir is None:
                print(f"  [WARN] Sequence {seq_id} listed in {lst_file.name} not found — skipped.")
                continue
            seq = Rellis3DSequence(str(seq_dir), max_rad=max_rad, robot=self.robot)
            stem_to_idx = {Path(f).stem: i for i, f in enumerate(seq.cloud_files)}
            target_indices = {stem_to_idx[s] for s in stems if s in stem_to_idx}
            missing = stems - stem_to_idx.keys()
            if missing:
                print(f"  [WARN] {len(missing)} frames from .lst not found in {seq_id} — skipped.")
            seq.target_indices = target_indices
            self.sequences.append(seq)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)

    def __getitem__(self, idx: int) -> Rellis3DSequence:
        return self.sequences[idx]
