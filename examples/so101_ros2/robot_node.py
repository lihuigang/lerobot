#!/usr/bin/env python

"""ROS2 node exposing the SO-101 follower arm over standard interfaces.

Published topics:
    /so101/joint_states    sensor_msgs/JointState   (degrees, order = JOINT_NAMES)
    /so101/ee_pose         geometry_msgs/PoseStamped (base frame, metres)
    /so101/status          std_msgs/String           (human readable events)

Subscribed topics:
    /so101/joint_command   sensor_msgs/JointState   (degrees, clamped per step)
    /so101/gripper_command std_msgs/Float64         (0 = closed, 100 = open)
    /so101/ee_target       geometry_msgs/PoseStamped (position-only IK, flip guarded)

Services:
    /so101/emergency_stop  std_srvs/Trigger         (cut torque on all motors)
    /so101/set_torque      std_srvs/SetBool         (enable/disable torque)

Run inside the ROS2 container::

    source /opt/ros/jazzy/setup.bash
    python robot_node.py --port /dev/serial/by-id/usb-...-if00
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import SetBool, Trigger

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
GRIPPER_INDEX = JOINT_NAMES.index("gripper")
GRIPPER_MIN, GRIPPER_MAX = 0.0, 100.0


def _matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a (x, y, z, w) quaternion."""
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return 0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return (m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return (m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s


class RobotNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("so101_robot")
        self.max_joint_step = args.max_joint_step

        config = SOFollowerRobotConfig(
            port=args.port,
            id=args.id,
            calibration_dir=Path(args.calibration_dir),
            cameras={},
            max_relative_target=args.max_relative_target,
            disable_torque_on_disconnect=True,
        )
        self.robot = SO101Follower(config)
        self.robot.connect(calibrate=False)
        self.get_logger().info(f"connected follower on {args.port}")

        self.kinematics: RobotKinematics | None = None
        if args.urdf and Path(args.urdf).exists():
            self.kinematics = RobotKinematics(
                args.urdf,
                target_frame_name="gripper_frame_link",
                joint_names=JOINT_NAMES,
            )
            self.get_logger().info(f"kinematics loaded from {args.urdf}")
        else:
            self.get_logger().warn(f"URDF not found ({args.urdf}); ee_pose/ee_target disabled")

        self.pub_joints = self.create_publisher(JointState, "/so101/joint_states", 10)
        self.pub_ee = self.create_publisher(PoseStamped, "/so101/ee_pose", 10)
        self.pub_status = self.create_publisher(String, "/so101/status", 10)

        self.create_subscription(JointState, "/so101/joint_command", self._on_joint_command, 10)
        self.create_subscription(Float64, "/so101/gripper_command", self._on_gripper_command, 10)
        self.create_subscription(PoseStamped, "/so101/ee_target", self._on_ee_target, 10)

        self.create_service(Trigger, "/so101/emergency_stop", self._on_emergency_stop)
        self.create_service(SetBool, "/so101/set_torque", self._on_set_torque)

        self.create_timer(1.0 / max(args.publish_rate, 1.0), self._publish_state)
        self._status("robot node ready")

    # ------------------------------------------------------------------ helpers
    def _read_deg(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([float(obs[f"{name}.pos"]) for name in JOINT_NAMES], dtype=np.float64)

    def _send_deg(self, cmd: np.ndarray) -> None:
        cmd = np.asarray(cmd, dtype=np.float64).copy()
        cmd[GRIPPER_INDEX] = float(np.clip(cmd[GRIPPER_INDEX], GRIPPER_MIN, GRIPPER_MAX))
        self.robot.send_action({f"{name}.pos": float(cmd[i]) for i, name in enumerate(JOINT_NAMES)})

    def _status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)
        self.get_logger().info(text)

    # ---------------------------------------------------------- state publishing
    def _publish_state(self) -> None:
        try:
            joints = self._read_deg()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"state read failed: {exc}")
            return

        stamp = self.get_clock().now().to_msg()

        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = list(JOINT_NAMES)
        joint_state.position = [float(value) for value in joints]
        self.pub_joints.publish(joint_state)

        if self.kinematics is not None:
            transform = self.kinematics.forward_kinematics(joints)
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = "base_link"
            pose.pose.position.x = float(transform[0, 3])
            pose.pose.position.y = float(transform[1, 3])
            pose.pose.position.z = float(transform[2, 3])
            qx, qy, qz, qw = _matrix_to_quaternion(transform[:3, :3])
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            pose.pose.orientation.w = float(qw)
            self.pub_ee.publish(pose)

    # --------------------------------------------------------------- subscriptions
    def _on_joint_command(self, msg: JointState) -> None:
        try:
            cmd = self._read_deg()
        except Exception as exc:  # noqa: BLE001
            self._status(f"joint_command rejected (read failed): {exc}")
            return
        for name, position in zip(msg.name, msg.position):
            if name in JOINT_NAMES:
                cmd[JOINT_NAMES.index(name)] = float(position)
        try:
            self._send_deg(cmd)
        except Exception as exc:  # noqa: BLE001
            self._status(f"joint_command failed (write): {exc}")

    def _on_gripper_command(self, msg: Float64) -> None:
        try:
            cmd = self._read_deg()
            cmd[GRIPPER_INDEX] = float(msg.data)
            self._send_deg(cmd)
            self._status(f"gripper -> {float(msg.data):.1f}%")
        except Exception as exc:  # noqa: BLE001
            self._status(f"gripper_command failed: {exc}")

    def _on_ee_target(self, msg: PoseStamped) -> None:
        if self.kinematics is None:
            self._status("ee_target ignored: kinematics unavailable")
            return

        target = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64
        )
        try:
            current = self._read_deg()
            desired = self.kinematics.forward_kinematics(current)
            desired[:3, 3] = target
            solution = self.kinematics.inverse_kinematics(current, desired, orientation_weight=0.0)
        except Exception as exc:  # noqa: BLE001
            self._status(f"ee_target IK failed: {exc}")
            return

        delta = float(np.max(np.abs(solution[:5] - current[:5])))
        if delta > self.max_joint_step:
            self._status(
                f"ee_target rejected: joint delta {delta:.1f}deg > {self.max_joint_step:.1f}deg (flip guard)"
            )
            return

        cmd = current.copy()
        cmd[:5] = solution[:5]
        try:
            self._send_deg(cmd)
            self._status(f"ee_target -> ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) ok")
        except Exception as exc:  # noqa: BLE001
            self._status(f"ee_target write failed: {exc}")

    # ------------------------------------------------------------------- services
    def _on_emergency_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            self.robot.bus.disable_torque()
            response.success = True
            response.message = "torque disabled on all motors"
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"emergency_stop failed: {exc}"
        self._status(response.message)
        return response

    def _on_set_torque(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        try:
            if request.data:
                self.robot.bus.enable_torque()
                state = "enabled"
            else:
                self.robot.bus.disable_torque()
                state = "disabled"
            response.success = True
            response.message = f"torque {state}"
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"set_torque failed: {exc}"
        self._status(response.message)
        return response

    def shutdown(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="follower serial port (by-id path)")
    parser.add_argument("--id", default="my_awesome_follower_arm")
    parser.add_argument("--calibration-dir", default="/calibration/robots/so101_follower")
    parser.add_argument("--urdf", default="/urdfs/so101/so101_new_calib.urdf")
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--max-joint-step", type=float, default=22.0, help="IK flip guard, degrees")
    parser.add_argument("--publish-rate", type=float, default=20.0, help="state publish rate in Hz")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = RobotNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
