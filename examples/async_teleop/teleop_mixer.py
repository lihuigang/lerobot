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
TeleopMixer - 混合AI策略输出和人类遥操作输入

This module provides the TeleopMixer class that combines AI policy actions
with human teleoperator actions to enable remote teleoperation scenarios.

三种混合模式:
1. intervention: 人类干预时完全覆盖AI输出
2. mix: 按比例混合AI和人类输入
3. direct: 完全使用人类输入（纯遥操作）
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class TeleopMode(str, Enum):
    """遥操作混合模式"""

    INTERVENTION = "intervention"  # 人类干预时完全覆盖AI输出
    MIX = "mix"  # 按比例混合AI和人类输入
    DIRECT = "direct"  # 完全使用人类输入（纯遥操作）


@dataclass
class TeleopMixerConfig:
    """TeleopMixer配置"""

    mode: TeleopMode = TeleopMode.INTERVENTION
    mix_alpha: float = 0.5  # 混合系数 (0=纯AI, 1=纯人类)
    intervention_threshold: float = 0.1  # 干预激活阈值（用于游戏手柄摇杆）


class TeleopMixer:
    """混合AI策略和人类遥操作输入"""

    def __init__(self, config: TeleopMixerConfig):
        self.config = config

    def mix(
        self,
        ai_action: torch.Tensor,
        teleop_action: dict[str, Any] | None,
        is_intervening: bool = False,
    ) -> torch.Tensor:
        """
        混合AI策略输出和人类遥操作输入

        Args:
            ai_action: AI策略输出的动作张量，形状 (action_dim,) 或 (1, action_dim)
            teleop_action: 人类遥操作设备的动作字典
            is_intervening: 人类是否正在干预

        Returns:
            混合后的动作张量
        """
        if self.config.mode == TeleopMode.DIRECT:
            return self._direct_mode(teleop_action, ai_action)
        elif self.config.mode == TeleopMode.INTERVENTION:
            return self._intervention_mode(ai_action, teleop_action, is_intervening)
        else:  # MIX mode
            return self._mix_mode(ai_action, teleop_action)

    def _direct_mode(
        self,
        teleop_action: dict[str, Any] | None,
        ai_action: torch.Tensor,
    ) -> torch.Tensor:
        """纯遥操作模式：完全使用人类输入"""
        if teleop_action is None:
            return ai_action  # 如果没有人类输入，使用AI动作

        return self._teleop_dict_to_tensor(teleop_action, ai_action)

    def _intervention_mode(
        self,
        ai_action: torch.Tensor,
        teleop_action: dict[str, Any] | None,
        is_intervening: bool,
    ) -> torch.Tensor:
        """干预模式：人类干预时完全覆盖AI输出"""
        if is_intervening and teleop_action is not None:
            return self._teleop_dict_to_tensor(teleop_action, ai_action)
        return ai_action

    def _mix_mode(
        self,
        ai_action: torch.Tensor,
        teleop_action: dict[str, Any] | None,
    ) -> torch.Tensor:
        """混合模式：按比例混合AI和人类输入"""
        if teleop_action is None:
            return ai_action

        teleop_tensor = self._teleop_dict_to_tensor(teleop_action, ai_action)
        alpha = self.config.mix_alpha

        return (1 - alpha) * ai_action + alpha * teleop_tensor

    def _teleop_dict_to_tensor(
        self,
        teleop_action: dict[str, Any],
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """将遥操作动作字典转换为张量"""
        # 处理不同类型的遥操作输入
        if "delta_x" in teleop_action:
            # 游戏手柄/键盘增量模式
            return self._delta_action_to_tensor(teleop_action, reference_tensor)
        elif any(".pos" in key for key in teleop_action):
            # 机械臂位置模式
            return self._position_action_to_tensor(teleop_action, reference_tensor)
        elif "linear_velocity" in teleop_action:
            # 移动机器人速度模式
            return self._velocity_action_to_tensor(teleop_action, reference_tensor)
        else:
            # 未知格式，返回AI动作
            return reference_tensor

    def _delta_action_to_tensor(
        self,
        teleop_action: dict[str, Any],
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """将增量动作转换为张量 (delta_x, delta_y, delta_z, gripper)"""
        device = reference_tensor.device
        dtype = reference_tensor.dtype

        # 获取动作维度
        if reference_tensor.ndim == 2:
            action_dim = reference_tensor.shape[1]
        else:
            action_dim = reference_tensor.shape[0]

        # 创建输出张量
        output = torch.zeros(action_dim, device=device, dtype=dtype)

        # 填充增量值
        output[0] = teleop_action.get("delta_x", 0.0)
        output[1] = teleop_action.get("delta_y", 0.0)
        output[2] = teleop_action.get("delta_z", 0.0)

        # 如果有夹爪动作
        if "gripper" in teleop_action and action_dim > 3:
            gripper_val = teleop_action["gripper"]
            # 将离散夹爪命令转换为连续值
            if isinstance(gripper_val, int):
                if gripper_val == 0:  # 关闭
                    output[3] = 0.0
                elif gripper_val == 2:  # 打开
                    output[3] = 1.0
                else:  # 保持
                    output[3] = reference_tensor[3].item() if reference_tensor.ndim == 1 else reference_tensor[0, 3].item()
            else:
                output[3] = float(gripper_val)

        if reference_tensor.ndim == 2:
            output = output.unsqueeze(0)

        return output

    def _position_action_to_tensor(
        self,
        teleop_action: dict[str, Any],
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """将位置动作转换为张量 (joint.pos格式)"""
        device = reference_tensor.device
        dtype = reference_tensor.dtype

        if reference_tensor.ndim == 2:
            action_dim = reference_tensor.shape[1]
            output = torch.zeros(action_dim, device=device, dtype=dtype)
            output = output.unsqueeze(0)
        else:
            action_dim = reference_tensor.shape[0]
            output = torch.zeros(action_dim, device=device, dtype=dtype)

        # 填充关节位置
        for i, key in enumerate(teleop_action.keys()):
            if i < action_dim:
                if reference_tensor.ndim == 2:
                    output[0, i] = teleop_action[key]
                else:
                    output[i] = teleop_action[key]

        return output

    def _velocity_action_to_tensor(
        self,
        teleop_action: dict[str, Any],
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """将速度动作转换为张量 (linear_velocity, angular_velocity)"""
        device = reference_tensor.device
        dtype = reference_tensor.dtype

        if reference_tensor.ndim == 2:
            action_dim = reference_tensor.shape[1]
            output = torch.zeros(action_dim, device=device, dtype=dtype)
            output = output.unsqueeze(0)
        else:
            action_dim = reference_tensor.shape[0]
            output = torch.zeros(action_dim, device=device, dtype=dtype)

        output[..., 0] = teleop_action.get("linear_velocity", 0.0)
        if action_dim > 1:
            output[..., 1] = teleop_action.get("angular_velocity", 0.0)

        return output

    def should_intervene(self, teleop_action: dict[str, Any] | None) -> bool:
        """
        判断人类是否应该干预

        对于游戏手柄，当摇杆移动超过阈值时激活干预
        """
        if teleop_action is None:
            return False

        if self.config.mode != TeleopMode.INTERVENTION:
            return True  # 非干预模式下总是使用人类输入

        # 检查是否有显著的移动输入
        delta_x = abs(teleop_action.get("delta_x", 0.0))
        delta_y = abs(teleop_action.get("delta_y", 0.0))
        delta_z = abs(teleop_action.get("delta_z", 0.0))

        max_delta = max(delta_x, delta_y, delta_z)
        return max_delta > self.config.intervention_threshold


def create_teleop_mixer(
    mode: str = "intervention",
    mix_alpha: float = 0.5,
    intervention_threshold: float = 0.1,
) -> TeleopMixer:
    """
    创建TeleopMixer实例的便捷函数

    Args:
        mode: 混合模式 (intervention, mix, direct)
        mix_alpha: 混合系数 (0=纯AI, 1=纯人类)
        intervention_threshold: 干预激活阈值

    Returns:
        TeleopMixer实例
    """
    config = TeleopMixerConfig(
        mode=TeleopMode(mode),
        mix_alpha=mix_alpha,
        intervention_threshold=intervention_threshold,
    )
    return TeleopMixer(config)