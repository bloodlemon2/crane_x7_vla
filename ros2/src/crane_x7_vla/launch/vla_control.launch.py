#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch VLA inference and robot control nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    """Setup launch configuration."""
    # Get launch arguments
    model_path = LaunchConfiguration('model_path')
    task_instruction = LaunchConfiguration('task_instruction')
    robot_config_file = LaunchConfiguration('robot_config_file')
    config_file = LaunchConfiguration('config_file')
    use_flash_attention = LaunchConfiguration('use_flash_attention')
    device = LaunchConfiguration('device')
    auto_execute = LaunchConfiguration('auto_execute')
    image_topic = LaunchConfiguration('image_topic')
    set_initial_position = LaunchConfiguration('set_initial_position')

    nodes = []

    # Initial position node (optional, runs first)
    initial_position_node = Node(
        package='crane_x7_vla',
        executable='initial_position_node',
        name='initial_position_node',
        output='screen',
        parameters=[
            robot_config_file,
            config_file,
        ],
        condition=IfCondition(set_initial_position),
    )
    nodes.append(initial_position_node)

    # VLA inference node
    vla_inference_node = Node(
        package='crane_x7_vla',
        executable='vla_inference_node',
        name='vla_inference_node',
        output='screen',
        parameters=[
            robot_config_file,
            config_file,
            {
                'model_path': model_path,
                'task_instruction': task_instruction,
                'use_flash_attention': use_flash_attention,
                'device': device,
                'image_topic': image_topic,
            }
        ],
    )
    nodes.append(vla_inference_node)

    # Robot controller node
    robot_controller_node = Node(
        package='crane_x7_vla',
        executable='robot_controller',
        name='robot_controller',
        output='screen',
        parameters=[
            robot_config_file,
            config_file,
            {
                'auto_execute': auto_execute,
            }
        ],
    )
    nodes.append(robot_controller_node)

    return nodes


def generate_launch_description():
    """Generate launch description."""
    # Declare launch arguments
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to VLA model (e.g., /workspace/vla/outputs/<model_dir>/checkpoint-1500)'
    )

    declare_task_instruction = DeclareLaunchArgument(
        'task_instruction',
        default_value='pick up the object',
        description='Task instruction for the robot'
    )

    declare_robot_config_file = DeclareLaunchArgument(
        'robot_config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('crane_x7_vla'),
            'config',
            'crane_x7_robot.yaml'
        ]),
        description='Path to robot specification file'
    )

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('crane_x7_vla'),
            'config',
            'vla_config.yaml'
        ]),
        description='Path to VLA configuration file'
    )

    declare_use_flash_attention = DeclareLaunchArgument(
        'use_flash_attention',
        default_value='false',
        description='Enable Flash Attention 2'
    )

    declare_device = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device to run inference on (cuda or cpu)'
    )

    declare_auto_execute = DeclareLaunchArgument(
        'auto_execute',
        default_value='true',
        description='Automatically execute VLA-predicted actions'
    )

    declare_image_topic = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        description='RGB image topic for VLA inference'
    )

    declare_set_initial_position = DeclareLaunchArgument(
        'set_initial_position',
        default_value='true',
        description='Move robot to initial position before VLA inference'
    )

    return LaunchDescription([
        declare_model_path,
        declare_task_instruction,
        declare_robot_config_file,
        declare_config_file,
        declare_use_flash_attention,
        declare_device,
        declare_auto_execute,
        declare_image_topic,
        declare_set_initial_position,
        OpaqueFunction(function=launch_setup)
    ])
