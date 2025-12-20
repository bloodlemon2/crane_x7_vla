#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""ROS 2 node for setting CRANE-X7 to initial position before VLA inference."""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTolerance
from sensor_msgs.msg import JointState

from crane_x7_vla.utils.config_manager import ConfigManager


class InitialPositionNode(Node):
    """Node to move CRANE-X7 to training data initial position."""

    def __init__(self):
        super().__init__('initial_position_node')

        # Declare and load robot configuration
        ConfigManager.declare_robot_parameters(self)
        self.robot_config = ConfigManager.load_robot_config(self)

        # Declare node-specific parameters
        self.declare_parameter('execution_time', 3.0)  # Slower for initial movement
        self.declare_parameter('position_tolerance', 0.05)
        self.declare_parameter('wait_for_joint_states', True)

        # Get parameters (use robot config for controller names)
        self.arm_controller_name = self.robot_config.arm_controller_name
        self.gripper_controller_name = self.robot_config.gripper_controller_name
        self.execution_time = self.get_parameter('execution_time').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.wait_for_joint_states = self.get_parameter('wait_for_joint_states').value

        # Use robot config for positions and joint names
        self.initial_arm_position = self.robot_config.initial_arm_position
        self.initial_gripper_position = self.robot_config.initial_gripper_position
        self.arm_joint_names = self.robot_config.arm_joint_names

        # State
        self.joint_state_received = False
        self.arm_done = False
        self.gripper_done = False

        # Setup action clients
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.arm_controller_name
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            self.gripper_controller_name
        )

        # Subscribe to joint states to know when robot is ready
        if self.wait_for_joint_states:
            self.joint_state_sub = self.create_subscription(
                JointState,
                '/joint_states',
                self._joint_state_callback,
                10
            )

        self.get_logger().info('Initial Position Node starting...')
        self.get_logger().info(f'Target arm position: {self.initial_arm_position}')
        self.get_logger().info(f'Target gripper position: {self.initial_gripper_position}')

        # Start setup process
        self._setup()

    def _joint_state_callback(self, msg: JointState) -> None:
        """Track when joint states are available."""
        if not self.joint_state_received:
            self.joint_state_received = True
            self.get_logger().info('Joint states received, robot is ready')

    def _setup(self) -> None:
        """Wait for action servers and execute initial position."""
        # Wait for arm action server
        self.get_logger().info(f'Waiting for arm action server: {self.arm_controller_name}')
        if not self.arm_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('Arm action server not available')
            return

        self.get_logger().info('Arm action server connected')

        # Wait for gripper action server
        self.get_logger().info(f'Waiting for gripper action server: {self.gripper_controller_name}')
        if not self.gripper_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('Gripper action server not available')
            return

        self.get_logger().info('Gripper action server connected')

        # Wait for joint states if required
        if self.wait_for_joint_states:
            self.get_logger().info('Waiting for joint states...')
            timeout = 10.0
            start_time = self.get_clock().now()
            while not self.joint_state_received:
                rclpy.spin_once(self, timeout_sec=0.1)
                elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout:
                    self.get_logger().warn('Timeout waiting for joint states, proceeding anyway')
                    break

        # Execute initial position
        self.get_logger().info('Moving to initial position...')
        self._execute_arm_action()
        self._execute_gripper_action()

    def _execute_arm_action(self) -> None:
        """Execute arm trajectory to initial position."""
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.arm_joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = self.initial_arm_position
        point.time_from_start.sec = int(self.execution_time)
        point.time_from_start.nanosec = int((self.execution_time % 1) * 1e9)

        goal_msg.trajectory.points = [point]

        # Set tolerances
        goal_msg.goal_tolerance = []
        for joint_name in self.arm_joint_names:
            tolerance = JointTolerance()
            tolerance.name = joint_name
            tolerance.position = self.position_tolerance
            goal_msg.goal_tolerance.append(tolerance)

        # Send goal
        send_goal_future = self.arm_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._arm_goal_response_callback)

    def _execute_gripper_action(self) -> None:
        """Execute gripper action to initial position."""
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = float(self.initial_gripper_position)
        goal_msg.command.max_effort = 1.0

        # Send goal
        send_goal_future = self.gripper_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._gripper_goal_response_callback)

    def _arm_goal_response_callback(self, future) -> None:
        """Callback when arm goal is accepted or rejected."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Arm goal rejected')
            self.arm_done = True
            return

        self.get_logger().info('Arm goal accepted, executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._arm_result_callback)

    def _gripper_goal_response_callback(self, future) -> None:
        """Callback when gripper goal is accepted or rejected."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected')
            self.gripper_done = True
            return

        self.get_logger().info('Gripper goal accepted, executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._gripper_result_callback)

    def _arm_result_callback(self, future) -> None:
        """Callback when arm execution is complete."""
        self.arm_done = True
        try:
            result = future.result().result
            if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
                self.get_logger().info('Arm moved to initial position successfully')
            else:
                self.get_logger().warn(f'Arm action ended with code: {result.error_code}')
        except Exception as e:
            self.get_logger().error(f'Arm action failed: {e}')

        self._check_done()

    def _gripper_result_callback(self, future) -> None:
        """Callback when gripper execution is complete."""
        self.gripper_done = True
        try:
            result = future.result().result
            self.get_logger().info(
                f'Gripper moved to initial position: position={result.position:.3f}'
            )
        except Exception as e:
            self.get_logger().error(f'Gripper action failed: {e}')

        self._check_done()

    def _check_done(self) -> None:
        """Check if both arm and gripper have completed."""
        if self.arm_done and self.gripper_done:
            self.get_logger().info('Initial position setup complete!')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = InitialPositionNode()

    # Spin until both actions complete
    while rclpy.ok() and not (node.arm_done and node.gripper_done):
        rclpy.spin_once(node, timeout_sec=0.1)

    # Keep node alive briefly to ensure logs are flushed
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
