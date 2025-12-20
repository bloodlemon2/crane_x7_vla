#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Configuration management for VLA nodes.

Robot configuration MUST be provided via crane_x7_robot.yaml.
No default values are provided - missing parameters will raise errors.
"""

from rclpy.node import Node

from crane_x7_vla.core.robot_config import RobotConfig


class ConfigManager:
    """Manages ROS 2 parameters for VLA nodes.

    All robot parameters must be loaded from crane_x7_robot.yaml.
    """

    @staticmethod
    def load_robot_config(node: Node) -> RobotConfig:
        """Load robot configuration from ROS 2 parameters.

        Parameters must be provided via crane_x7_robot.yaml.
        Missing parameters will raise ParameterNotDeclaredException.

        Returns:
            RobotConfig loaded from ROS 2 parameters
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
            default_image_size=tuple(image_size),
        )
