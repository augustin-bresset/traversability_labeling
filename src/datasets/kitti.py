import importlib
import os
import logging
import torch
import torch.nn.functional as F
from torch._C import Value
from torch_geometric.data import Data, Dataset
import logging
from pathlib import Path
from glob import glob
import pandas as pd

# Basic libs
import numpy as np
import yaml

from .dataset_mapping import go_to_split, find_dataset_root, find_raw_data_dirs

from .utils import heights_to_bins, class_mapping_da



class SemanticKITTI(Dataset):
    def __init__(
        self,
        root,
        split="training",
        transform=None,
        dataset_size=None,
        multiframe_range=None,
        da_flag=False,
        config=None,
        **kwargs,
    ):

        super().__init__(root, transform, None)

        self.split = split
        self.n_frames = 1
        self.da_flag = da_flag
        self.config = config
        self.multiframe_range = multiframe_range
        self.N_LABELS = self.config["nb_classes"] if self.config is not None else 11

        logging.info(f"SemanticKITTI - split {split}")

        # get the scenes
        assert split in ["train", "val", "test"]

        if split == "train":
            self.sequences = ["{:02d}".format(i) for i in range(11) if i != 8]
        elif split == "val":
            self.sequences = ["{:02d}".format(i) for i in range(11) if i == 8]
        elif split == "test":
            self.sequences = ["{:02d}".format(i) for i in range(11, 22)]
        else:
            raise ValueError("Unknown set for SemanticKitti data: ", split)

        # get the filenames
        self.all_files = []
        for sequence in self.sequences:
            self.all_files += [
                path
                for path in Path(
                    os.path.join(
                        self.root, "dataset", "sequences", sequence, "velodyne"
                    )
                ).rglob("*.bin")
            ]

        # Sort for verifying and parametrizing
        if split == "verifying" or split == "val" or split == "parametrizing":
            self.all_files = sorted(self.all_files, key=lambda i: str(i).lower())

        self.all_labels = []
        for fname in self.all_files:
            fname = str(fname).replace("/velodyne/", "/labels/")
            fname = str(fname).replace(".bin", ".label")
            self.all_labels.append(fname)

        # Read labels
        if self.n_frames == 1:
            config_file = os.path.join(self.root, "semantic-kitti.yaml")
        elif self.n_frames > 1:
            config_file = os.path.join(self.root, "semantic-kitti-all.yaml")
        else:
            raise ValueError("number of frames has to be >= 1")

        learning_map, learning_map_inv = class_mapping_da(config, config_file)

        self.learning_map = np.zeros(
            (np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32
        )

        for k, v in learning_map.items():
            self.learning_map[k] = v

        self.learning_map_inv = np.zeros(
            (np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32
        )
        for k, v in learning_map_inv.items():
            self.learning_map_inv[k] = v

    def get_weights(self):
        weights = torch.ones(self.N_LABELS)
        weights[0] = 0
        return weights

    @staticmethod
    def get_mask_filter_valid_labels(y):
        return y > 0

    @property
    def raw_file_names(self):
        return []

    def _download(self):  # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.all_files)

    def get_category(self, f_id):
        return str(self.all_files[f_id]).split("/")[-3]

    def get_object_name(self, f_id):
        return str(self.all_files[f_id]).split("/")[-1]

    def get_class_name(self, f_id):
        return "lidar"

    def get_save_dir(self, f_id):
        return os.path.join(
            str(self.all_files[f_id]).split("/")[-3],
            str(self.all_files[f_id]).split("/")[-2],
        )

    def get_filename(self, idx):
        return self.all_files[idx]

    def get(self, idx):
        """Get item."""

        fname_points = self.all_files[idx]
        frame_points = np.fromfile(fname_points, dtype=np.float32)
        pos = frame_points.reshape((-1, 4))
        intensities = pos[:, 3:]
        pos = pos[:, :3]

        if self.split in ["test", "testing"]:
            # Fake labels
            y = np.zeros((pos.shape[0],), dtype=np.int32)
        else:
            # Read labels
            label_file = self.all_labels[idx]
            frame_labels = np.fromfile(label_file, dtype=np.int32)
            y = frame_labels & 0xFFFF  # semantic label in lower half
            y = self.learning_map[y]

        # points are annotated only until 50 m
        mask = np.linalg.norm(pos, axis=1) < 50
        pos = pos[mask]
        y = y[mask]
        intensities = intensities[mask]

        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0], 1), dtype=torch.float)
        return Data(x=x, intensities=intensities, pos=pos, y=y, shape_id=idx)

