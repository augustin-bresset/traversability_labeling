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

from .dataset_mapping import go_to_split, find_dataset_root

from .utils import heights_to_bins



class Goose3D(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        max_samples=None,
        remap_cfg="goose_label_mapping.csv",
        max_rad=200,
        feat_dup=True,
        transform=None,
        is_challenge=False,
        strict=True,
        shuffle=True,
        map_trav_cfg=None,
        mode_trav=False,
        height_label=False,
    ):
        super().__init__(root_dir, transform, None)

        root_split = go_to_split(root_dir, split)

        if root_split is not None:
            self.root_dir = find_dataset_root(root_split)
        else:
            self.root_dir = find_dataset_root(root_dir)

        self.split = split
        self.max_samples = max_samples
        self.max_rad = max_rad
        self.feat_dup = feat_dup
        self.shuffle = shuffle
        self.remap_cfg = Path(remap_cfg)
        self.num_auxiliary_classes_ = 10
        self.is_challenge = is_challenge
        self.map_trav_cfg = Path(map_trav_cfg) if map_trav_cfg is not None else None
        self.mode_trav = mode_trav
        self.height_label = height_label
        self.labels_dir = "labels" if not is_challenge else "3d_challenge"

        if is_challenge:
            self.map_cfg = "goose_challenge_map.yaml"

        if self.root_dir is None:
            if strict:
                raise RuntimeError(
                f"Could not find dataset root from '{root_dir}'. "
                "Expected LICENSE / CHANGELOG / mapping file."
                )
            else:
                self.root_dir = root_dir
                logging.WARNING(f"[GOOSE] Falling back to provided root_dir: {root_dir}")


        self.cloud_files = sorted(
            glob(os.path.join(self.root_dir, "**", "lidar", "**", "*.bin"), recursive=True)
        )
        self.label_files = sorted(
            glob(os.path.join(self.root_dir, "**", "labels", "**", "*.label"), recursive=True)
        )
        if self.height_label:
            self.height_files = sorted(
                glob(os.path.join(self.root_dir, "**", "height_labels", "**", "*.height_label"), recursive=True)
            )

        if len(self.cloud_files) != len(self.label_files):
            raise ValueError(
                f"Number of cloud files and label files do not match: {len(self.cloud_files)} != {len(self.label_files)}"
            )

        # Load remap configuration
        if self.remap_cfg.suffix == ".csv":
            self.remap_dict = self._load_csv_mapping(
               os.path.join(self.root_dir, self.remap_cfg)
            )
        elif self.remap_cfg.suffix in {".yaml", ".yml"}:
            with open(os.path.join(self.root_dir, self.remap_cfg), "r") as stream:
                remap_data = yaml.safe_load(stream)
                self.remap_dict = remap_data["learning_map"]
        elif not self.mode_trav:
            raise ValueError(f"Unknown mapping format: {self.remap_cfg}")

        if self.mode_trav:
            with open(map_trav_cfg, "r") as strem:
                map_trav = yaml.safe_load(stream)
                self.map_trav_tensor = torch.tensor(
                    map_trav["traversable_map"], 
                    dtype=torch.long
                )

        self.num_classes_ = len(set(self.remap_dict.values()))
        print(f"GOOSE Number of classes: {self.num_classes_}")

        if self.max_samples is not None:
            if len(self.cloud_files) > self.max_samples:
                if self.shuffle:
                    ids = np.random.choice(
                        len(self.cloud_files), self.max_samples, replace=False
                    )
                else:
                    ids = np.arange(0, self.max_samples, 1, dtype=int)
                self.cloud_files = [self.cloud_files[i] for i in ids]
                self.label_files = [self.label_files[i] for i in ids]
                if self.height_label:
                    self.height_files = [self.height_files[i] for i in ids]

        self.id_to_cloud_file = dict(
            zip(range(len(self.cloud_files)), self.cloud_files)
        )
        self.id_to_label_file = dict(
            zip(range(len(self.label_files)), self.label_files)
        )
        if self.height_label:
            self.id_to_height_file = dict(
                zip(range(len(self.height_files)), self.height_files)
            )

    def _load_csv_mapping(self, mapping_file):
        """
        Expected format:
        class_name,label_key,has_instance,hex
        """
        df = pd.read_csv(mapping_file)

        if "label_key" not in df.columns:
            raise ValueError(f"Invalid mapping file: {mapping_file}")

        mapping = dict(zip(df["label_key"], df["label_key"]))

        return mapping

    def get_filenames(self):
        return self.cloud_files

    def get_bin_dir(self):
        return "lidar"

    def __len__(self):
        return len(self.cloud_files)

    def len(self):
        return len(self.cloud_files)

    def _download(self):  # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        file_name = self.id_to_cloud_file[idx]
        label_file = self.id_to_label_file[idx]

        pos = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(label_file, dtype=np.uint32).reshape((-1))

        # extract the semantic and instance label IDs
        labels = labels & 0xFFFF  # semantic label in lower half
        # inst_label = labels >> 16    # instance id in upper half

        coords, features = pos[:, :3], pos[:, 3:]

        if self.remap_dict is not None:
            labels = np.vectorize(lambda l: self.remap_dict.get(l, 0))(labels)

        assert len(coords) == len(
            labels
        ), f"{len(coords)} != {len(labels)} | {file_name}, {label_file}"

        mask = np.linalg.norm(coords, axis=1) < self.max_rad
        coords, features, labels = coords[mask], features[mask], labels[mask]

        coords = torch.from_numpy(coords).float()
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()

        intensities = torch.ones((coords.shape[0], 1), dtype=torch.float)

        if self.feat_dup:
            features = torch.cat([features, coords], dim=1)

        data_dict = dict(
            x=features,
            intensities=intensities,
            pos=coords,
            y=labels,
            shape_id=idx,
            pcd_file=file_name,
            label_file=label_file,
        )

        if self.height_label:
            height_labels = None
            if not self.is_challenge:
                height_file = self.id_to_height_file[idx]
                height_labels = np.fromfile(height_file, dtype=np.float32)[mask] # /!\ float64
                height_labels = heights_to_bins(height_labels)
                height_labels = torch.from_numpy(height_labels).long()

            data_dict["height_labels"] = height_labels        

        if self.mode_trav:
            data_dict["y"] = torch.isin(labels, self.map_trav_tensor)

        return Data(**data_dict)
