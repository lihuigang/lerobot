#!/usr/bin/env python

"""ROS2 camera node for the SO-101 front (head) and handeye (wrist) cameras.

This replaces the former HTTP preview service (``so101_camera_web.py``) and its
8765 port. One ``sensor_msgs/Image`` (bgr8) and one ``sensor_msgs/CameraInfo``
are published per camera:

    /so101/front/image_raw      /so101/front/camera_info
    /so101/handeye/image_raw    /so101/handeye/camera_info

Run inside the ROS2 container::

    source /opt/ros/jazzy/setup.bash
    python camera_node.py
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image

CAMERAS = {
    "front": {  # head camera (seen from above the table)
        "device": "/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0",
        "fourcc": "MJPG",
        # Manual exposure in 100 us units: 800 = 80 ms; this camera has no gain control.
        "exposure": 800.0,
        "width": 640,
        "height": 480,
        "frame_id": "front_optical_frame",
    },
    "handeye": {  # wrist camera (close-up of the jaws and tissue)
        "device": "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0",
        "fourcc": "YUYV",
        "exposure": None,  # keep the camera's automatic exposure
        "sharpness": 8.0,  # default 4 of 15
        "contrast": 34.0,  # default 27 of 64
        "width": 640,
        "height": 480,
        "rotate": 90,  # clockwise
        "output_width": 480,
        "output_height": 480,
        "frame_id": "handeye_optical_frame",
    },
}


class CameraWorker:
    """Background V4L2 capture thread that always keeps the latest frame."""

    def __init__(self, name: str, spec: dict, width: int, height: int, fps: int):
        self.name = name
        self.spec = spec
        self.width = width
        self.height = height
        self.fps = fps
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.frame: np.ndarray | None = None
        self.sequence = 0
        self.error: str | None = "connecting"
        self.thread = threading.Thread(target=self._run, name=f"cam-{name}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=3)

    def _set_error(self, message: str) -> None:
        with self.condition:
            self.error = message
            self.condition.notify_all()

    def _run(self) -> None:
        device = self.spec["device"]
        while not self.stop_event.is_set():
            if not Path(device).exists():
                self._set_error("device not found")
                self.stop_event.wait(2)
                continue

            capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not capture.isOpened():
                self._set_error("cannot open camera (busy?)")
                capture.release()
                self.stop_event.wait(2)
                continue

            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.spec["fourcc"]))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            if self.spec["exposure"] is not None:
                # This camera only accepts manual mode via auto_exposure=1 (0.25/2 are rejected).
                capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
                capture.set(cv2.CAP_PROP_EXPOSURE, self.spec["exposure"])
            if self.spec.get("gain") is not None:
                capture.set(cv2.CAP_PROP_GAIN, self.spec["gain"])
            if self.spec.get("sharpness") is not None:
                capture.set(cv2.CAP_PROP_SHARPNESS, self.spec["sharpness"])
            if self.spec.get("contrast") is not None:
                capture.set(cv2.CAP_PROP_CONTRAST, self.spec["contrast"])

            try:
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        self._set_error("read failed")
                        break
                    rotate = self.spec.get("rotate", 0)
                    if rotate == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rotate == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif rotate == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    output_width = self.spec.get("output_width")
                    output_height = self.spec.get("output_height")
                    if output_width and output_height:
                        height, width = frame.shape[:2]
                        scale = max(output_width / width, output_height / height)
                        new_width = max(output_width, int(round(width * scale)))
                        new_height = max(output_height, int(round(height * scale)))
                        resized = cv2.resize(
                            frame,
                            (new_width, new_height),
                            interpolation=cv2.INTER_AREA,
                        )
                        x = (new_width - output_width) // 2
                        y = (new_height - output_height) // 2
                        frame = resized[y : y + output_height, x : x + output_width]
                    with self.condition:
                        self.frame = frame
                        self.sequence += 1
                        self.error = None
                        self.condition.notify_all()
            finally:
                capture.release()

            self.stop_event.wait(1)

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self.condition:
            return self.sequence, self.frame


class CameraNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("so101_camera")
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        info_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.workers: dict[str, CameraWorker] = {}
        self.pub_image: dict[str, object] = {}
        self.pub_info: dict[str, object] = {}
        self.last_seq: dict[str, int] = {}

        for name, base_spec in CAMERAS.items():
            override = getattr(args, f"{name}_device")
            spec = {**base_spec, "device": override or base_spec["device"]}
            worker = CameraWorker(
                name,
                spec,
                spec.get("width", args.width),
                spec.get("height", args.height),
                spec.get("fps", args.fps),
            )
            self.workers[name] = worker
            self.pub_image[name] = self.create_publisher(Image, f"/so101/{name}/image_raw", image_qos)
            self.pub_info[name] = self.create_publisher(CameraInfo, f"/so101/{name}/camera_info", info_qos)
            self.last_seq[name] = -1
            worker.start()

        self.create_timer(1.0 / max(args.rate, 1.0), self._on_timer)
        self.get_logger().info("SO-101 camera node started (front + handeye)")

    def _on_timer(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for name, worker in self.workers.items():
            seq, frame = worker.latest()
            if frame is None or seq == self.last_seq[name]:
                continue
            self.last_seq[name] = seq
            self.pub_image[name].publish(self._to_image(frame, stamp, name))
            self.pub_info[name].publish(self._to_info(frame, stamp, name))

    def _to_image(self, frame: np.ndarray, stamp, name: str) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = CAMERAS[name]["frame_id"]
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * 3
        msg.data = np.ascontiguousarray(frame).tobytes()
        return msg

    def _to_info(self, frame: np.ndarray, stamp, name: str) -> CameraInfo:
        height, width = frame.shape[:2]
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = CAMERAS[name]["frame_id"]
        msg.width = width
        msg.height = height
        # Nominal pinhole intrinsics only; no per-camera calibration is loaded yet.
        focal = float(max(width, height))
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [focal, 0.0, width / 2.0, 0.0, focal, height / 2.0, 0.0, 0.0, 1.0]
        msg.p = [focal, 0.0, width / 2.0, 0.0, 0.0, focal, height / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def shutdown(self) -> None:
        for worker in self.workers.values():
            worker.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-device", default=None, help="override the front camera device path")
    parser.add_argument("--handeye-device", default=None, help="override the handeye camera device path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--rate", type=float, default=30.0, help="publish timer rate in Hz")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = CameraNode(args)
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
