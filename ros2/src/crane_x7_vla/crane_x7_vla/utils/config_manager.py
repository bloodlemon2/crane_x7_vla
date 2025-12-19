#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Configuration management for VLA nodes."""

from rclpy.node import Node

from crane_x7_vla.core.robot_config import RobotConfig


class ConfigManager:
    """Manages ROS 2 parameters for VLA nodes."""

    # Default values matching crane_x7_robot.yaml
    DEFAULT_ARM_JOINT_NAMES = [
        "crane_x7_shoulder_fixed_part_pan_joint",
        "crane_x7_shoulder_revolute_part_tilt_joint",
        "crane_x7_upper_arm_revolute_part_twist_joint",
        "crane_x7_upper_arm_revolute_part_rotate_joint",
        "crane_x7_lower_arm_fixed_part_joint",
        "crane_x7_lower_arm_revolute_part_joint",
        "crane_x7_wrist_joint",
    ]
    DEFAULT_GRIPPER_JOINT_NAMES = [
        "crane_x7_gripper_finger_a_joint",
    ]
    DEFAULT_INITIAL_ARM_POSITION = [0.0, 1.57, 0.0, -2.8, 0.0, -1.05, 1.57]
    DEFAULT_INITIAL_GRIPPER_POSITION = 0.0
    DEFAULT_ARM_CONTROLLER = "/crane_x7_arm_controller/follow_joint_trajectory"
    DEFAULT_GRIPPER_CONTROLLER = "/crane_x7_gripper_controller/gripper_cmd"
    DEFAULT_IMAGE_SIZE = [224, 224]

    @classmethod
    def declare_robot_parameters(cls, node: Node) -> None:
        """Declare robot specification parameters.

        These parameters are typically loaded from crane_x7_robot.yaml.
        Default values are provided for backward compatibility.
        """
        # Robot identifier
        node.declare_parameter("robot.name", "crane_x7")

        # DOF
        node.declare_parameter("robot.arm_dof", 7)
        node.declare_parameter("robot.gripper_dof", 1)
        node.declare_parameter("robot.action_dim", 8)

        # Joint names
        node.declare_parameter("robot.arm_joint_names", cls.DEFAULT_ARM_JOINT_NAMES)
        node.declare_parameter("robot.gripper_joint_names", cls.DEFAULT_GRIPPER_JOINT_NAMES)

        # Initial positions
        node.declare_parameter("robot.initial_arm_position", cls.DEFAULT_INITIAL_ARM_POSITION)
        node.declare_parameter("robot.initial_gripper_position", cls.DEFAULT_INITIAL_GRIPPER_POSITION)

        # Controller names
        node.declare_parameter("robot.arm_controller_name", cls.DEFAULT_ARM_CONTROLLER)
        node.declare_parameter("robot.gripper_controller_name", cls.DEFAULT_GRIPPER_CONTROLLER)

        # Image settings
        node.declare_parameter("robot.default_image_size", cls.DEFAULT_IMAGE_SIZE)

    @staticmethod
    def load_robot_config(node: Node) -> RobotConfig:
        """Load robot configuration from ROS 2 parameters.

        Parameters should be declared with declare_robot_parameters() first.
        """
        image_size = node.get_parameter("robot.default_image_size").value

        return RobotConfig(
            name=node.get_parameter("robot.name").value,
            arm_dof=node.get_parameter("robot.arm_dof").value,
            gripper_dof=node.get_parameter("robot.gripper_dof").value,
            action_dim=node.get_parameter("robot.action_dim").value,
            arm_joint_names=list(node.get_parameter("robot.arm_joint_names").value),
            gripper_joint_names=list(node.get_parameter("robot.gripper_joint_names").value),
            initial_arm_position=list(node.get_parameter("robot.initial_arm_position").value),
            initial_gripper_position=node.get_parameter("robot.initial_gripper_position").value,
            arm_controller_name=node.get_parameter("robot.arm_controller_name").value,
            gripper_controller_name=node.get_parameter("robot.gripper_controller_name").value,
            default_image_size=tuple(image_size) if image_size else (224, 224),
        )
