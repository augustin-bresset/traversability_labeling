"""
Unit tests for TraversabilityLabeler.

All point coordinates are in scan-local frame (robot at origin),
not world frame.  Poses encode world-frame positions of each scan.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.traversability.labeler import TraversabilityLabeler


def _make_poses(positions_xy):
    """Build a list of identity-rotation 4x4 poses from (x, y) positions."""
    poses = []
    for x, y in positions_xy:
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = x
        T[1, 3] = y
        poses.append(T)
    return poses


class TestLabelScan:
    """label_scan: points in scan-local frame, future poses only."""

    def setup_method(self):
        self.labeler = TraversabilityLabeler(
            robot_shape="square",
            robot_size=1.0,    # half_size = 0.5 m
            height_min=-0.5,
            height_max=0.3,
            trajectory_window=100,
        )

    def test_point_ahead_is_traversable(self):
        # Robot drives forward, 0.3 m/step.  current_idx=5 → world [1.5, 0].
        # half_size=0.5 m, so a future pose at scan-local [0.3, 0] covers [-0.2, 0.8].
        # "Ahead" point at [0.2, 0, 0] (scan-local) → dx = 0.2-0.3 = -0.1 < 0.5 → traversable.
        # "Clearly behind" at [-2.0, 0, 0]: all future poses are at scan-local x >= 0.3,
        #   so dx = -2.0 - 0.3 = -2.3 → outside footprint → NOT traversable.
        positions = [(0.3 * i, 0.0) for i in range(20)]
        poses = _make_poses(positions)
        current_idx = 5

        xyz = np.array([
            [ 0.2, 0.0, 0.0],   # ahead → traversable (covered by pose k=6)
            [-2.0, 0.0, 0.0],   # clearly behind → NOT traversable
            [50.0, 0.0, 0.0],   # far ahead, never reached → not traversable
        ], dtype=np.float32)

        labels = self.labeler.label_scan(xyz, poses, current_idx=current_idx)

        assert labels[0] == 1, "Point ahead should be traversable"
        assert labels[1] == 0, "Point clearly behind should NOT be traversable"
        assert labels[2] == 0, "Far point should NOT be traversable"

    def test_height_filter(self):
        # Points above/below height range are never traversable.
        # Use 0.3 m steps so the future pose at scan-local [0.3, 0] is within footprint.
        positions = [(0.3 * i, 0.0) for i in range(20)]
        poses = _make_poses(positions)

        xyz = np.array([
            [0.1, 0.0,  0.5],   # above height_max=0.3 → not traversable
            [0.1, 0.0, -0.6],   # below height_min=-0.5 → not traversable
            [0.1, 0.0,  0.0],   # within range → traversable (dx=0.1-0.3=-0.2 < 0.5)
        ], dtype=np.float32)

        labels = self.labeler.label_scan(xyz, poses, current_idx=5)

        assert labels[0] == 0, "Above height_max should not be traversable"
        assert labels[1] == 0, "Below height_min should not be traversable"
        assert labels[2] == 1, "Within height range should be traversable"

    def test_round_footprint(self):
        labeler = TraversabilityLabeler(
            robot_shape="round",
            robot_size=1.0,
            height_min=-0.5,
            height_max=0.3,
            trajectory_window=100,
        )
        positions = [(0.1 * i, 0.0) for i in range(20)]
        poses = _make_poses(positions)

        xyz = np.array([
            [0.05,  0.0, 0.0],   # inside circle → traversable
            [0.05,  0.45, 0.0],  # inside circle (r < 0.5) → traversable
            [0.05,  0.55, 0.0],  # outside circle → not traversable
        ], dtype=np.float32)

        labels = labeler.label_scan(xyz, poses, current_idx=5)
        assert labels[0] == 1
        assert labels[1] == 1
        assert labels[2] == 0

    def test_no_future_poses(self):
        # At the last scan, traj_lo > traj_hi → nothing labeled.
        positions = [(float(i), 0.0) for i in range(5)]
        poses = _make_poses(positions)

        xyz = np.array([[0.1, 0.0, 0.0]], dtype=np.float32)
        labels = self.labeler.label_scan(xyz, poses, current_idx=4)
        assert labels[0] == 0, "Last scan has no future poses — nothing traversable"


class TestForwardLabeling:
    """use_forward_labeling=True labels height-filtered points in the forward corridor."""

    def setup_method(self):
        self.labeler = TraversabilityLabeler(
            robot_shape="square",
            robot_size=1.0,
            height_min=-0.5,
            height_max=0.3,
            trajectory_window=100,
            use_forward_labeling=True,
            forward_dist=5.0,
        )

    def test_forward_point_labeled(self):
        # Robot at idx=5 (world [1.5, 0]), next pose at [1.8, 0].
        # Forward direction in scan-local ≈ [+x].
        # A point at [2.0, 0, 0] in scan-local is 2 m ahead, within 5 m corridor,
        # within 0.5 m lateral half-size → must be labeled.
        positions = [(0.3 * i, 0.0) for i in range(20)]
        poses = _make_poses(positions)

        xyz = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
        labels = self.labeler.label_scan(xyz, poses, current_idx=5)
        assert labels[0] == 1, "Point ahead in corridor should be traversable"

    def test_forward_beyond_dist_not_labeled(self):
        # 6 m ahead > forward_dist=5 m → not labeled by forward pass.
        # Also beyond trajectory window reach → not labeled by poses either.
        positions = [(0.3 * i, 0.0) for i in range(20)]
        poses = _make_poses(positions)

        xyz = np.array([[6.0, 0.0, 0.0]], dtype=np.float32)
        labels = self.labeler.label_scan(xyz, poses, current_idx=5)
        assert labels[0] == 0, "Point beyond forward_dist should not be labeled"

    def test_forward_lateral_outside_footprint_not_labeled(self):
        # 2 m ahead but 1 m to the side (> 0.5 m half-size) → outside corridor.
        positions = [(0.3 * i, 0.0) for i in range(20)]
        poses = _make_poses(positions)

        xyz = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
        labels = self.labeler.label_scan(xyz, poses, current_idx=5)
        assert labels[0] == 0, "Point outside lateral footprint should not be labeled"

    def test_behind_point_not_labeled_by_forward(self):
        # Behind the robot → negative forward projection → not labeled by forward pass.
        # (Still won't be labeled by future poses either since robot moves away.)
        positions = [(0.3 * i, 0.0) for i in range(20)]
        poses = _make_poses(positions)

        xyz = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)
        labels = self.labeler.label_scan(xyz, poses, current_idx=5)
        assert labels[0] == 0, "Point behind robot should not be labeled by forward pass"

    def test_disabled_by_default(self):
        # Default labeler has use_forward_labeling=False: far-ahead point not labeled.
        labeler = TraversabilityLabeler(
            robot_shape="square", robot_size=1.0,
            height_min=-0.5, height_max=0.3, trajectory_window=2,
        )
        positions = [(0.3 * i, 0.0) for i in range(20)]
        poses = _make_poses(positions)

        # Point 4 m ahead, beyond trajectory window reach (window=2 → max 0.9 m in scan-local)
        xyz = np.array([[4.0, 0.0, 0.0]], dtype=np.float32)
        labels = labeler.label_scan(xyz, poses, current_idx=5)
        assert labels[0] == 0, "Forward labeling disabled: far-ahead point should not be labeled"


class TestLabelAccumulated:
    """label_accumulated: per-point future-only constraint via scan_origins."""

    def setup_method(self):
        self.labeler = TraversabilityLabeler(
            robot_shape="square",
            robot_size=1.0,
            height_min=-0.5,
            height_max=0.3,
            trajectory_window=100,
        )

    def test_past_scan_point_labeled_correctly(self):
        # Robot drives along x at 0.1 m/step.
        # current_idx = 10.  A point seen at scan k=8 at the robot's feet
        # (world pos [0.8, 0, 0]).  The robot passes over that world position
        # at scan k=8 itself, but the rule says only poses AFTER k=8 count,
        # i.e. k=9 (world [0.9, 0]) and k=10 (world [1.0, 0]).
        # In current frame (current_idx=10, world pos [1.0, 0]):
        #   scan-local of world [0.8, 0] = [0.8-1.0, 0] = [-0.2, 0, 0]
        # At k=9 (world [0.9,0]) relative to current frame = [-0.1, 0] → within 0.5m half.
        # So the point should be labeled traversable.
        positions = [(0.1 * i, 0.0) for i in range(20)]
        poses = _make_poses(positions)
        current_idx = 10

        # In current_idx=10 frame, world [0.8, 0] → local [-0.2, 0, 0]
        xyz_acc = np.array([[-0.2, 0.0, 0.0]], dtype=np.float32)
        scan_origins = np.array([8], dtype=np.int32)

        labels = self.labeler.label_accumulated(xyz_acc, scan_origins, poses, current_idx)
        assert labels[0] == 1, "Past scan ground point should be labeled traversable"

    def test_dynamic_object_at_past_robot_position(self):
        # A person was at world [0.8, 0] when scan k=8 was taken.
        # At k=9 the robot is at [0.9, 0] — 0.1 m away from [0.8, 0], within footprint.
        # BUT the person has moved away. Our rule says: only check poses > origin_k=8,
        # which includes k=9 where the robot was close. This tests the TEMPORAL rule:
        # if the robot DID pass through that position after observing it, it IS labeled.
        # The protection is that the robot won't drive through a standing person —
        # if a person is there the robot will deviate. This is a dataset-level guarantee,
        # not enforced in labeler logic.
        #
        # The test we CAN verify: a person BEHIND the robot in the CURRENT scan
        # will NOT be labeled because no future pose goes back to it.
        positions = [(float(i), 0.0) for i in range(20)]
        poses = _make_poses(positions)
        current_idx = 10

        # In current frame (robot at world [10, 0]):
        # A point at local [-2, 0, 0] = world [8, 0] = behind the robot.
        # Future poses (k=11...) all move forward (x > 10), never back to x=8.
        xyz_acc = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)
        # Origin = current scan → rule: only poses > 10, all at x>=11 → dx = local +1..
        # The point at local -2 means world 8; future poses at 11+ are 3+ m away → outside footprint.
        scan_origins = np.array([current_idx], dtype=np.int32)

        labels = self.labeler.label_accumulated(xyz_acc, scan_origins, poses, current_idx)
        assert labels[0] == 0, "Person behind robot should NOT be traversable"

    def test_scan_origins_enforce_future_only(self):
        # Robot moves 0.3 m/step.  current_idx=5 → world [1.5, 0].
        # We test with a point at scan-local [-0.5, 0, 0] (= world [1.0, 0]).
        #
        # Origin k=0 (allowed: poses k=1..19):
        #   k=3 → world [0.9, 0] → scan-local [-0.6, 0] → dx=-0.5-(-0.6)=0.1 < 0.5 → covered ✓
        #
        # Origin k=5 (allowed: poses k=6..19):
        #   k=6 → world [1.8, 0] → scan-local [0.3, 0] → dx=-0.5-0.3=-0.8 > 0.5 → NOT covered ✓
        #   All later poses move further right, so also not covered.
        positions = [(0.3 * i, 0.0) for i in range(20)]
        poses = _make_poses(positions)
        current_idx = 5

        xyz_acc = np.array([
            [-0.5, 0.0, 0.0],   # origin k=0: traversable (pose k=3 covers it)
            [-0.5, 0.0, 0.0],   # origin k=5: NOT traversable (robot moved past, no return)
        ], dtype=np.float32)
        scan_origins = np.array([0, 5], dtype=np.int32)

        labels = self.labeler.label_accumulated(xyz_acc, scan_origins, poses, current_idx)
        assert labels[0] == 1, "Early-observed point should be traversable (robot later drove over it)"
        assert labels[1] == 0, "Current-scan point that robot passed should NOT be traversable"
