#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Configuration management for VLA nodes.

Robot configuration should be provided via crane_x7_robot.yaml.
Default values from RobotConfig are used if parameters are not provided.
"""

from rclpy.node import Node

from crane_x7_vla.core.robot_config import RobotConfig


class ConfigManager:
    """Manages ROS 2 parameters for VLA nodes.

    Robot parameters are loaded from crane_x7_robot.yaml with RobotConfig
    providing default values.
    """

    # Default config instance for getting default values
    _default_config = RobotConfig()

    @staticmethod
    def declare_robot_parameters(node: Node) -> None:
        """Declare all robot parameters with defaults from RobotConfig.

        This method MUST be called before load_robot_config().
        Parameters can be overridden via crane_x7_robot.yaml.

        Args:
            node: ROS 2 node instance
        """
        defaults = ConfigManager._default_config

        # Robot identifier
        node.declare_parameter('robot.name', defaults.name)

        # Degrees of freedom
        node.declare_parameter('robot.arm_dof', defaults.arm_dof)
        node.declare_parameter('robot.gripper_dof', defaults.gripper_dof)
        node.declare_parameter('robot.action_dim', defaults.action_dim)

        # Joint names
        node.declare_parameter('robot.arm_joint_names', defaults.arm_joint_names)
        node.declare_parameter('robot.gripper_joint_names', defaults.gripper_joint_names)

        # Initial positions
        node.declare_parameter('robot.initial_arm_position', defaults.initial_arm_position)
        node.declare_parameter('robot.initial_gripper_position', defaults.initial_gripper_position)

        # Controller names
        node.declare_parameter('robot.arm_controller_name', defaults.arm_controller_name)
        node.declare_parameter('robot.gripper_controller_name', defaults.gripper_controller_name)

        # Image settings
        node.declare_parameter('robot.default_image_size', list(defaults.default_image_size))

    @staticmethod
    def load_robot_config(node: Node) -> RobotConfig:
        """Load robot configuration from ROS 2 parameters.

        Parameters should be declared first via declare_robot_parameters().
        Values from crane_x7_robot.yaml will override defaults.

        Args:
            node: ROS 2 node instance

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
