#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""ROS 2 node for VLA model inference.

This node uses create_inference_core factory to automatically select
the appropriate inference backend (OpenVLA or Pi0/Pi0.5) based on
the model type detected from the checkpoint.
"""

from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge

from .vla_inference_core import create_inference_core


class ROS2LoggerAdapter:
    """Adapter to use ROS 2 logger with Python logging interface."""

    def __init__(self, ros_logger):
        self._logger = ros_logger

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warn(msg)

    def error(self, msg):
        self._logger.error(msg)

    def debug(self, msg):
        self._logger.debug(msg)


class VLAInferenceNode(Node):
    """ROS 2 node for VLA model inference (supports OpenVLA, Pi0, Pi0.5).

    This node subscribes to camera images and joint states,
    runs VLA inference using the appropriate backend, and publishes predicted actions.
    The model type is automatically detected from the checkpoint.

    Subscriptions:
        - /camera/color/image_raw (sensor_msgs/Image): RGB camera image
        - /joint_states (sensor_msgs/JointState): Robot joint states
        - /vla/update_instruction (std_msgs/String): Dynamic task instruction update

    Publishers:
        - /vla/predicted_action (std_msgs/Float32MultiArray): Predicted action (8-dim)

    Parameters:
        - model_path: Path to VLA model (local or HuggingFace Hub ID)
        - task_instruction: Task instruction for the robot
        - device: Inference device ('cuda' or 'cpu')
        - inference_rate: Inference rate in Hz
        - unnorm_key: Key for action normalization statistics (OpenVLA only)
    """

    def __init__(self):
        super().__init__('vla_inference_node')

        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_base_name', 'openvla')
        self.declare_parameter('task_instruction', 'pick up the object')
        self.declare_parameter('use_flash_attention', False)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('action_topic', '/vla/predicted_action')
        self.declare_parameter('inference_rate', 10.0)
        self.declare_parameter('center_crop', False)
        self.declare_parameter('unnorm_key', 'crane_x7')

        # Get parameters
        model_path = self.get_parameter('model_path').value
        model_base_name = self.get_parameter('model_base_name').value
        self.task_instruction = self.get_parameter('task_instruction').value
        use_flash_attention = self.get_parameter('use_flash_attention').value
        device = self.get_parameter('device').value
        self.image_topic = self.get_parameter('image_topic').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.action_topic = self.get_parameter('action_topic').value
        self.inference_rate = self.get_parameter('inference_rate').value
        self.center_crop = self.get_parameter('center_crop').value
        unnorm_key = self.get_parameter('unnorm_key').value

        # Initialize
        self.bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.latest_joint_state: Optional[JointState] = None

        # Create logger adapter for inference core
        logger_adapter = ROS2LoggerAdapter(self.get_logger())

        # Initialize VLA inference core using factory function
        # Automatically detects model type (OpenVLA, Pi0, Pi0.5)
        self.vla_core = create_inference_core(
            model_path=model_path,
            device=device,
            unnorm_key=unnorm_key,
            model_base_name=model_base_name,
            use_flash_attention=use_flash_attention,
            logger=logger_adapter,
        )

        # Load model
        if not self.vla_core.load_model():
            self.get_logger().error('Failed to load VLA model')

        # Setup subscriptions
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            10
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self._joint_state_callback,
            10
        )

        # Setup publishers
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            self.action_topic,
            10
        )

        # Subscribe to instruction updates
        self.update_instruction_sub = self.create_subscription(
            String,
            '/vla/update_instruction',
            self._update_instruction_callback,
            10
        )

        # Setup inference timer
        timer_period = 1.0 / self.inference_rate
        self.inference_timer = self.create_timer(timer_period, self._inference_callback)

        self.get_logger().info('VLA Inference Node initialized')
        self.get_logger().info(f'Model: {model_path}')
        self.get_logger().info(f'Task: {self.task_instruction}')
        self.get_logger().info(f'Inference rate: {self.inference_rate} Hz')

    def _image_callback(self, msg: Image) -> None:
        """Callback for RGB image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = cv_image
            self.get_logger().debug(
                f'Image received: shape={cv_image.shape}, '
                f'dtype={cv_image.dtype}, '
                f'mean={cv_image.mean():.2f}',
                throttle_duration_sec=2.0
            )
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def _joint_state_callback(self, msg: JointState) -> None:
        """Callback for joint states."""
        self.latest_joint_state = msg

    def _update_instruction_callback(self, msg: String) -> None:
        """Callback for updating task instruction."""
        self.task_instruction = msg.data
        self.get_logger().info(f'Updated task instruction: {self.task_instruction}')

    def _inference_callback(self) -> None:
        """Perform VLA inference and publish action."""
        if not self.vla_core.is_loaded:
            return

        if self.latest_image is None:
            self.get_logger().warn('No image received yet', throttle_duration_sec=5.0)
            return

        # Extract robot state from joint states
        state = self._extract_robot_state()

        # Define log callback
        def log_callback(msg: str):
            self.get_logger().info(msg)

        # Run inference using VLAInferenceCore
        action = self.vla_core.predict_action(
            image=self.latest_image,
            instruction=self.task_instruction,
            state=state,
            log_callback=log_callback
        )

        if action is not None:
            # Publish action
            action_msg = Float32MultiArray()
            action_msg.data = action.tolist()
            self.action_pub.publish(action_msg)
            self.get_logger().info(f'Published action: {action}')

    def _extract_robot_state(self) -> Optional[np.ndarray]:
        """Extract robot state from latest joint states.

        Returns:
            8-dim state array (7 arm joints + 1 gripper) or None if not available
        """
        if self.latest_joint_state is None:
            self.get_logger().warn('No joint state received yet', throttle_duration_sec=5.0)
            return None

        # CRANE-X7 joint order in JointState message
        # Expected: crane_x7_joint1 through crane_x7_joint7 + crane_x7_gripper_finger_a_joint
        expected_joints = [
            'crane_x7_shoulder_fixed_part_pan_joint',
            'crane_x7_shoulder_revolute_part_tilt_joint',
            'crane_x7_upper_arm_revolute_part_twist_joint',
            'crane_x7_upper_arm_revolute_part_rotate_joint',
            'crane_x7_lower_arm_fixed_part_joint',
            'crane_x7_lower_arm_revolute_part_joint',
            'crane_x7_wrist_joint',
            'crane_x7_gripper_finger_a_joint',
        ]

        joint_positions = self.latest_joint_state.position
        joint_names = self.latest_joint_state.name

        # Build state array in expected order
        state = np.zeros(8, dtype=np.float32)
        for i, joint_name in enumerate(expected_joints):
            if joint_name in joint_names:
                idx = joint_names.index(joint_name)
                state[i] = joint_positions[idx]
            else:
                self.get_logger().warn(
                    f'Joint {joint_name} not found in joint states',
                    throttle_duration_sec=10.0
                )

        return state


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = VLAInferenceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
