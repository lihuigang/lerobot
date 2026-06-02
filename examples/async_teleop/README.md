# 远程遥操作示例 - Async Teleoperation

本示例展示如何通过网络远程控制机器人，同时允许人类操作员通过本地遥操作设备进行干预。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     云端/远程服务器 (GPU)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Policy Server                          │  │
│  │  - 加载预训练策略模型                                      │  │
│  │  - 接收机器人观测数据                                      │  │
│  │  - 运行神经网络推理                                        │  │
│  │  - 返回AI动作序列                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↑↓ gRPC                             │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                    ┌──────────┼──────────┐
                    │   网络   │ (WiFi/5G) │
                    └──────────┼──────────
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                     机器人端 (边缘设备)                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               RobotClient + TeleopMixer                   │  │
│  │  - 连接物理机器人                                          │  │
│  │  - 连接本地遥操作设备 (游戏手柄/Leader臂)                   │  │
│  │  - 混合AI策略输出和人类输入                                │  │
│  │  - 发送观测到服务器，执行混合后的动作                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↑↓                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    物理机器人                              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↑↓                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  本地遥操作设备                            │  │
│  │  - 游戏手柄 (Gamepad)                                     │  │
│  │  - Leader机械臂 (SO-100/101)                              │  │
│  │  - 键盘 (Keyboard)                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 使用方式

### 方式1: 游戏手柄干预模式

AI策略自主运行，人类通过游戏手柄随时干预：

```bash
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
```

**游戏手柄控制**:
- **左摇杆**: X/Y方向移动干预
- **右摇杆**: Z方向移动干预
- **A按钮**: 打开夹爪
- **B按钮**: 关闭夹爪
- **LT按钮**: 激活干预模式（按住时人类输入生效）
- **RT按钮**: 停止当前任务

### 方式2: 主从臂混合模式

AI策略和Leader机械臂输入混合：

```bash
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
```

**混合模式说明**:
- `mix_alpha=0.5`: AI和人类输入各占50%
- `mix_alpha=0.0`: 纯AI策略
- `mix_alpha=1.0`: 纯人类遥操作

### 方式3: 纯远程遥操作模式

不使用AI策略，完全通过本地设备远程控制机器人：

```bash
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
    --teleop_mode=direct \
    --task="move to home position"
```

## 代码架构

### TeleopMixer类

混合AI策略输出和人类输入的核心组件：

```python
class TeleopMixer:
    """混合AI策略和人类遥操作输入"""
    
    def mix(
        self,
        ai_action: torch.Tensor,      # AI策略输出
        teleop_action: dict,          # 人类输入
        is_intervening: bool,         # 是否正在干预
        mode: str,                    # 混合模式
        alpha: float,                 # 混合系数
    ) -> torch.Tensor:                # 混合后的动作
        ...
```

### 混合策略

1. **Intervention模式**: 人类干预时完全覆盖AI输出
2. **Mix模式**: 按比例混合AI和人类输入
3. **Direct模式**: 完全使用人类输入（纯遥操作）

## 服务器端

服务器端代码不需要修改，使用标准的`policy_server.py`即可。

启动服务器：

```bash
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30