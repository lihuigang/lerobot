#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
远程遥操作示例 - Async Teleoperation with Human-in-the-Loop

This script demonstrates how to use async inference with teleoperation input,
enabling remote control of robots with AI policy assistance.

Three modes are supported:
1. intervention: Human overrides AI when actively controlling
2. mix: AI and human inputs are blended with configurable ratio
3. direct: Full teleoperation without AI

Examples:

# 方式1: 游戏手柄干预模式
python examples/async_teleop/run_async_teleop.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --server_address=192.168.1.100:8080 \
    --policy_type=act \
    --pretrained_name_or_path=lerobot/act_aloha \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --teleop.type=gamepad \
    --teleop_mode=intervention \
    --task="pick up the red block"

# 方式2: 主从臂混合模式
python examples/async_teleop/run_async_teleop.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --server_address=192.168.1.100:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/my_model \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=blue \
    --teleop_mode=mix \
    --mix_alpha=0.5 \
    --task="assemble the parts"
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Any

import draccus
import torch

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import (
    TimedAction,
    get_logger,
    map_robot_keys_to_lerobot_features,
)
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    gamepad,
    keyboard,
    make_teleoperator_from_config,
    so_leader,
)
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.utils import TeleopEvents

from teleop_mixer import TeleopMixer, TeleopMode, create_teleop_mixer


@dataclass
class AsyncTeleopConfig:
    """Configuration for async teleoperation."""

    # Robot client config (embedded)
    robot: RobotClientConfig = field(metadata={"help": "Robot client configuration"})

    # Teleoperator configuration
    teleop: TeleoperatorConfig | None = field(
        default=None, metadata={"help": "Teleoperator configuration"}
    )

    # Teleoperation mode
    teleop_mode: str = field(
        default="intervention",
        metadata={
            "help": "Teleoperation mode: intervention (human overrides AI), "
            "mix (blend AI and human), direct (full teleop)"
        },
    )

    # Mix alpha (for mix mode)
    mix_alpha: float = field(
        default=0.5,
        metadata={"help": "Mixing ratio for mix mode (0=full AI, 1=full human)"},
    )

    # Intervention threshold
    intervention_threshold: float = field(
        default=0.1,
        metadata={"help": "Threshold to activate intervention"},
    )


@dataclass
class TeleopActionWrapper:
    """Wrapper for teleop actions with timing info."""

    action: dict[str, Any]
    timestep: int
    timestamp: float
    is_intervening: bool = False


class AsyncTeleopClient:
    """
    Async teleoperation client that combines AI policy with human teleoperation.

    This client runs two threads:
    1. Action receiver: receives AI actions from the server
    2. Control loop: reads teleop, combines with AI, and executes on robot
    """

    def __init__(self, config: AsyncTeleopConfig):
        self.config = config
        self.logger = get_logger("async_teleop_client")

        # Initialize the underlying robot client
        self.robot_client = RobotClient(config.robot)

        # Initialize teleoperator if provided
        self.teleop: Teleoperator | None = None
        if config.teleop is not None:
            self.teleop = make_teleoperator_from_config(config.teleop)
            self.logger.info(f"Teleoperator configured: {config.teleop.type}")

        # Initialize teleop mixer
        self.mixer = create_teleop_mixer(
            mode=config.teleop_mode,
            mix_alpha=config.mix_alpha,
            intervention_threshold=config.intervention_threshold,
        )

        # Teleop action queue
        self.teleop_action_queue: Queue[TeleopActionWrapper] = Queue()
        self.teleop_action_lock = threading.Lock()

        # Shutdown event
        self.shutdown_event = threading.Event()

    def start(self) -> bool:
        """Start the async teleop client."""
        if not self.robot_client.start():
            return False

        if self.teleop is not None:
            self.teleop.connect()
            self.logger.info("Teleoperator connected")

        return True

    def stop(self):
        """Stop the async teleop client."""
        self.shutdown_event.set()
        self.robot_client.stop()

        if self.teleop is not None:
            self.teleop.disconnect()
            self.logger.info("Teleoperator disconnected")

    def teleop_reader_thread(self):
        """Thread that reads teleop actions and puts them in the queue."""
        self.logger.info("Teleop reader thread starting")
        timestep = 0

        while not self.shutdown_event.is_set():
            loop_start = time.perf_counter()

            teleop_action = None
            is_intervening = False

            if self.teleop is not None:
                try:
                    teleop_action = self.teleop.get_action()

                    # Check for intervention events
                    if hasattr(self.teleop, "get_teleop_events"):
                        events = self.teleop.get_teleop_events()
                        is_intervening = events.get(TeleopEvents.IS_INTERVENTION, False)

                    # For intervention mode, check if human should intervene
                    if self.config.teleop_mode == "intervention":
                        is_intervening = self.mixer.should_intervene(teleop_action)

                except Exception as e:
                    self.logger.warning(f"Error reading teleop action: {e}")

            # Put action in queue
            action_wrapper = TeleopActionWrapper(
                action=teleop_action,
                timestep=timestep,
                timestamp=time.time(),
                is_intervening=is_intervening,
            )

            # Replace old action in queue (keep only latest)
            with self.teleop_action_lock:
                while not self.teleop_action_queue.empty():
                    self.teleop_action_queue.get_nowait()
                self.teleop_action_queue.put(action_wrapper)

            timestep += 1

            # Maintain teleop frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, self.config.robot.environment_dt - elapsed)
            time.sleep(sleep_time)

        self.logger.info("Teleop reader thread stopped")

    def control_loop(self, task: str):
        """
        Main control loop that combines AI and teleop actions.

        This loop:
        1. Gets AI action from robot_client's queue
        2. Gets human action from teleop queue
        3. Mixes them according to the configured mode
        4. Executes the mixed action on the robot
        """
        self.logger.info("Control loop starting")

        timestep = 0

        while not self.shutdown_event.is_set():
            loop_start = time.perf_counter()

            # Get teleop action
            teleop_action_wrapper = None
            with self.teleop_action_lock:
                if not self.teleop_action_queue.empty():
                    teleop_action_wrapper = self.teleop_action_queue.get_nowait()

            # Check if we should use teleop action
            use_teleop = False
            teleop_action = None
            is_intervening = False

            if teleop_action_wrapper is not None:
                teleop_action = teleop_action_wrapper.action
                is_intervening = teleop_action_wrapper.is_intervening
                use_teleop = teleop_action is not None

            # Get AI action from queue
            ai_action = None
            if self.robot_client.actions_available():
                try:
                    with self.robot_client.action_queue_lock:
                        if not self.robot_client.action_queue.empty():
                            timed_action = self.robot_client.action_queue.get_nowait()
                            ai_action = timed_action.get_action()
                            self.robot_client.latest_action = timed_action.get_timestep()
                except Exception as e:
                    self.logger.warning(f"Error getting AI action: {e}")

            # Mix AI and teleop actions
            if ai_action is not None:
                mixed_action = self.mixer.mix(ai_action, teleop_action, is_intervening)
            elif use_teleop and teleop_action is not None:
                # No AI action available, use teleop directly
                mixed_action = self.mixer._teleop_dict_to_tensor(
                    teleop_action,
                    torch.zeros(self.robot_client.robot.action_features.shape if hasattr(self.robot_client.robot.action_features, 'shape') else (6,)),
                )
            else:
                # No action available, skip this step
                time.sleep(self.config.robot.environment_dt)
                continue

            # Convert mixed action to robot action format
            robot_action = self._action_tensor_to_dict(mixed_action)

            # Send action to robot
            try:
                self.robot_client.robot.send_action(robot_action)
            except Exception as e:
                self.logger.error(f"Error sending action to robot: {e}")

            timestep += 1

            # Log status
            if timestep % 30 == 0:  # Log every 30 steps
                mode_str = "TELEOP" if use_teleop else "AI"
                if is_intervening:
                    mode_str = "INTERVENING"
                self.logger.info(
                    f"Step {timestep} | Mode: {mode_str} | "
                    f"Teleop: {teleop_action is not None} | AI: {ai_action is not None}"
                )

            # Maintain control frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, self.config.robot.environment_dt - elapsed)
            time.sleep(sleep_time)

        self.logger.info("Control loop stopped")

    def _action_tensor_to_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        """Convert action tensor to robot action dictionary."""
        action_features = self.robot_client.robot.action_features

        if hasattr(action_features, "names") and isinstance(action_features.names, dict):
            action = {key: action_tensor[i].item() for i, key in enumerate(action_features.names.keys())}
        else:
            action = {key: action_tensor[i].item() for i, key in enumerate(action_features)}

        return action

    def run(self, task: str):
        """Run the async teleop client."""
        if not self.start():
            self.logger.error("Failed to start client")
            return

        try:
            # Start teleop reader thread
            teleop_thread = threading.Thread(target=self.teleop_reader_thread, daemon=True)
            teleop_thread.start()

            # Start action receiver thread (from robot_client)
            action_receiver_thread = threading.Thread(
                target=self.robot_client.receive_actions, daemon=True
            )
            action_receiver_thread.start()

            # Run control loop in main thread
            self.control_loop(task)

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()


@draccus.wrap()
def async_teleop(cfg: AsyncTeleopConfig):
    """Main entry point for async teleoperation."""
    logging.info(f"Starting async teleop with mode: {cfg.teleop_mode}")
    logging.info(f"Mix alpha: {cfg.mix_alpha}")

    client = AsyncTeleopClient(cfg)
    client.run(task=cfg.robot.task)


if __name__ == "__main__":
    async_teleop()