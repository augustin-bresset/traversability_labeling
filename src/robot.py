"""
Robot geometry.

Defines the physical shape of the robot and provides helpers to:
  - filter self-hit points from a LiDAR scan (a return inside the robot
    body is physically impossible and indicates a sensor artifact).
  - check whether the robot footprint covers a given set of XY positions
    (used by TraversabilityLabeler).
"""

from __future__ import annotations

import numpy as np


class Robot:
    """
    Robot shape parameters.

    Args:
        shape:      "square" (axis-aligned box) or "round" (cylinder).
        size:       Footprint side length (square) or diameter (round), in metres.
        height_min: Lowest extent of the robot body relative to the sensor, in metres.
        height_max: Highest extent of the robot body relative to the sensor, in metres.
    """

    def __init__(
        self,
        shape: str = "square",
        size: float = 1.0,
        height_min: float = -0.5,
        height_max: float = 0.3,
    ):
        if shape not in ("square", "round"):
            raise ValueError(f"Unknown robot shape '{shape}'. Use 'square' or 'round'.")
        self.shape      = shape
        self.size       = size
        self.half_size  = size / 2.0
        self.height_min = height_min
        self.height_max = height_max

    # ------------------------------------------------------------------

    def self_hit_mask(self, xyz: np.ndarray) -> np.ndarray:
        """
        Boolean mask: True for points that lie inside the robot body.

        A point inside the robot body is a physically impossible LiDAR return
        (the robot chassis blocks the beam).  These points should be discarded
        before any processing.

        xyz is assumed to be in the sensor/robot frame (sensor at origin).
        """
        z_inside = (xyz[:, 2] >= self.height_min) & (xyz[:, 2] <= self.height_max)

        if self.shape == "round":
            xy_inside = (xyz[:, 0] ** 2 + xyz[:, 1] ** 2) < self.half_size ** 2
        else:
            xy_inside = (np.abs(xyz[:, 0]) < self.half_size) & (np.abs(xyz[:, 1]) < self.half_size)

        return z_inside & xy_inside

    def in_footprint(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """
        Boolean mask: True where (dx, dy) falls within the robot's XY footprint.
        Used by TraversabilityLabeler.
        """
        if self.shape == "round":
            return dx ** 2 + dy ** 2 < self.half_size ** 2
        return (np.abs(dx) < self.half_size) & (np.abs(dy) < self.half_size)

    @staticmethod
    def from_config(cfg: dict) -> "Robot":
        """Build a Robot from a config dict (the 'robot' section of the YAML)."""
        return Robot(
            shape      = cfg.get("shape",      "square"),
            size       = cfg.get("size",       1.0),
            height_min = cfg.get("height_min", -0.5),
            height_max = cfg.get("height_max",  0.3),
        )
