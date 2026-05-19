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

from .utils import heights_to_bins



class Outback(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        max_samples=5000,
        remap_cfg="outback_map.yaml",
        max_rad=50,
        transform=None,
        feat_dup=True,
    ):
        super().__init__(root_dir, transform, None)
        self.root_dir = root_dir
        self.max_depth = max_rad
        self.feat_dup = feat_dup
        self.remap_cfg = remap_cfg
        self.num_auxiliary_classes_ = 10

        if split != "all":
            self.split_csv = os.path.join(self.root_dir, f"{split}.csv")
            self.file_name_df = pd.read_csv(self.split_csv)
        else:
            train_df = pd.read_csv(os.path.join(self.root_dir, "train.csv"))
            val_df = pd.read_csv(os.path.join(self.root_dir, "val.csv"))
            self.file_name_df = pd.concat([train_df, val_df])

        # Load remap configuration
        with open(os.path.join(self.root_dir, self.remap_cfg), "r") as stream:
            remap_data = yaml.safe_load(stream)
            self.label_to_id = remap_data["learning_map"]

        self.num_classes_ = len(set(self.label_to_id.values()))
        print(f"GOOSE Number of classes: {self.num_classes_}")

        self.split = split

        self.max_samples = max_samples

    def __len__(self):
        return len(self.file_name_df)

    def get_filenames(self):
        return list(self.file_name_df["depth_files"])

    def get_label_files(self):
        return list(self.file_name_df["label_files"])

    def get_bin_dir(self):
        return "Depth"

    def len(self):
        return len(self.file_name_df)

    def _download(self):  # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def csv_to_pcd(self, depth_file, label_file, height_file):

        lidar_depths = np.genfromtxt(depth_file, delimiter=",", skip_header=1)
        labels = np.genfromtxt(label_file, delimiter=",", dtype=str)
        height_labels = np.float32(np.fromfile(height_file))
        height_labels = heights_to_bins(height_labels)

        # Define the vertical and horizontal FOV
        vertical_fov = (-30, 30)  # degrees
        horizontal_fov = (0, 360)  # degrees

        # Dataset shape
        depth_array = lidar_depths  # Example random depth values

        # Generate angle values
        vertical_angles = np.linspace(
            vertical_fov[0], vertical_fov[1], depth_array.shape[1]
        )  # 151 points
        horizontal_angles = np.linspace(
            horizontal_fov[0], horizontal_fov[1], depth_array.shape[0]
        )  # 900 points

        # Convert angles to radians
        vertical_angles = -np.radians(vertical_angles)
        horizontal_angles = np.radians(horizontal_angles)

        # Generate a meshgrid for the angles
        phi, theta = np.meshgrid(
            horizontal_angles, vertical_angles, indexing="ij"
        )  # Azimuth (phi) and Elevation (theta)

        # Compute 3D coordinates
        x = depth_array * np.cos(theta) * np.cos(phi)
        y = depth_array * np.cos(theta) * np.sin(phi)
        z = depth_array * np.sin(theta)

        # Reshape into a (N, 3) point cloud format
        points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

        # filter out points with depth = 0 and depth > self.max_depth

        mask = (depth_array.ravel() > 0) & (depth_array.ravel() < self.max_depth)
        points = points[mask]
        # print(points.shape)
        labels = labels.ravel()[mask]

        height_labels = height_labels[mask]

        labels = [
            self.label_to_id[label] if label in self.label_to_id else 0
            for label in labels
        ]

        assert len(points) == len(labels) == len(height_labels)

        return points, labels, height_labels

    def get(self, idx):

        depth_file = self.file_name_df["depth_files"][idx]
        label_file = self.file_name_df["label_files"][idx]
        height_file = self.file_name_df["height_files"][idx]

        points, labels, height_labels = self.csv_to_pcd(
            depth_file, label_file, height_file
        )

        features = np.ones_like(labels, dtype=np.float32)

        x = torch.ones((points.shape[0], 1), dtype=torch.float)
        points = torch.tensor(points, dtype=torch.float)
        features = torch.tensor(features, dtype=torch.float).unsqueeze(-1)
        labels = torch.tensor(labels, dtype=torch.long)
        height_labels = torch.tensor(height_labels, dtype=torch.long)

        if self.feat_dup:
            features = torch.cat([features, points], dim=1)

        return Data(
            x=features,
            intensities=x,
            pos=points,
            y=labels,
            shape_id=idx,
            height_labels=height_labels,
        )
