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


class InitialPositionNode(Node):
    """Node to move CRANE-X7 to training data initial position."""

    # Initial position based on training data (data/tfrecord_logs_2_10Hz)
    # Rounded to clean values for reproducibility
    INITIAL_ARM_POSITION = [
        0.0,    # joint1: crane_x7_shoulder_fixed_part_pan_joint (was 0.043)
        1.57,   # joint2: crane_x7_shoulder_revolute_part_tilt_joint (~π/2, was 1.611)
        0.0,    # joint3: crane_x7_upper_arm_revolute_part_twist_joint (was 0.063)
        -2.8,   # joint4: crane_x7_upper_arm_revolute_part_rotate_joint (was -2.812)
        0.0,    # joint5: crane_x7_lower_arm_fixed_part_joint (was -0.061)
        -1.05,  # joint6: crane_x7_lower_arm_revolute_part_joint (~-π/3, was -0.948)
        1.57,   # joint7: crane_x7_wrist_joint (~π/2, was 1.671)
    ]
    INITIAL_GRIPPER_POSITION = 0.0  # Open gripper (was -0.008)

    ARM_JOINT_NAMES = [
        'crane_x7_shoulder_fixed_part_pan_joint',
        'crane_x7_shoulder_revolute_part_tilt_joint',
        'crane_x7_upper_arm_revolute_part_twist_joint',
        'crane_x7_upper_arm_revolute_part_rotate_joint',
        'crane_x7_lower_arm_fixed_part_joint',
        'crane_x7_lower_arm_revolute_part_joint',
        'crane_x7_wrist_joint',
    ]

    def __init__(self):
        super().__init__('initial_position_node')

        # Declare parameters
        self.declare_parameter('arm_controller_name', '/crane_x7_arm_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_controller_name', '/crane_x7_gripper_controller/gripper_cmd')
        self.declare_parameter('execution_time', 3.0)  # Slower for initial movement
        self.declare_parameter('position_tolerance', 0.05)
        self.declare_parameter('wait_for_joint_states', True)

        # Get parameters
        self.arm_controller_name = self.get_parameter('arm_controller_name').value
        self.gripper_controller_name = self.get_parameter('gripper_controller_name').value
        self.execution_time = self.get_parameter('execution_time').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.wait_for_joint_states = self.get_parameter('wait_for_joint_states').value

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
        self.get_logger().info(f'Target arm position: {self.INITIAL_ARM_POSITION}')
        self.get_logger().info(f'Target gripper position: {self.INITIAL_GRIPPER_POSITION}')

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
        goal_msg.trajectory.joint_names = self.ARM_JOINT_NAMES

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = self.INITIAL_ARM_POSITION
        point.time_from_start.sec = int(self.execution_time)
        point.time_from_start.nanosec = int((self.execution_time % 1) * 1e9)

        goal_msg.trajectory.points = [point]

        # Set tolerances
        goal_msg.goal_tolerance = []
        for joint_name in self.ARM_JOINT_NAMES:
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
        goal_msg.command.position = float(self.INITIAL_GRIPPER_POSITION)
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
