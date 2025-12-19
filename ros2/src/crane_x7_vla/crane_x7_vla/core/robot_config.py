#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Robot configuration dataclasses."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RobotConfig:
    """CRANE-X7 robot specification.

    Centralizes all robot-specific constants that were previously
    hardcoded across multiple files.
    """

    # Robot identifier
    name: str = "crane_x7"

    # Degrees of freedom
    arm_dof: int = 7
    gripper_dof: int = 1
    action_dim: int = 8

    # Joint names
    arm_joint_names: List[str] = field(default_factory=lambda: [
        "crane_x7_shoulder_fixed_part_pan_joint",
        "crane_x7_shoulder_revolute_part_tilt_joint",
        "crane_x7_upper_arm_revolute_part_twist_joint",
        "crane_x7_upper_arm_revolute_part_rotate_joint",
        "crane_x7_lower_arm_fixed_part_joint",
        "crane_x7_lower_arm_revolute_part_joint",
        "crane_x7_wrist_joint",
    ])

    gripper_joint_names: List[str] = field(default_factory=lambda: [
        "crane_x7_gripper_finger_a_joint",
    ])

    # Initial positions
    initial_arm_position: List[float] = field(default_factory=lambda: [
        0.0, 1.57, 0.0, -2.8, 0.0, -1.05, 1.57
    ])
    initial_gripper_position: float = 0.0

    # Controller names
    arm_controller_name: str = "/crane_x7_arm_controller/follow_joint_trajectory"
    gripper_controller_name: str = "/crane_x7_gripper_controller/gripper_cmd"

    # Image settings
    default_image_size: Tuple[int, int] = (224, 224)

    @property
    def all_joint_names(self) -> List[str]:
        """Return all joint names in order (arm + gripper)."""
        return self.arm_joint_names + self.gripper_joint_names
