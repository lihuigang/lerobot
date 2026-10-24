# SO-101：Docker 一键命令

在 `lihuigang@192.168.1.9` 的交互式终端运行以下命令。标定、遥操、相机、采集、训练与推理均由 [脚本](scripts/so101_docker.sh) 启动**一次性 Docker 容器**；脚本每次重新解析串口和相机节点，不使用旧的 `so101-cuda` 容器。

> 安全：上次 Leader 肘关节约 `165°`，Follower 旧标定边界约 `135°`。重新核对标定和实际行程前，不要遥操、采集或推理，更不要强推关节。当前示例数据集尚未录制，因此也没有可用于推理的模型。

## 进入远端并检查

```bash
ssh -t lihuigang@192.168.1.9
cd ~/zhitai/github/lerobot
scripts/so101_docker.sh check
```

`check` 只检查设备映射并运行合成数据的 CUDA 一步训练，**不会打开相机或驱动机械臂**。应看到 `SO101_DOCKER_IO_OK` 和 `SO101_CUDA_SMOKE_OK gpu=Orin`。镜像若不存在，先运行 `scripts/so101_docker.sh build`；配方见 [Dockerfile](docker/Dockerfile.so101_cuda) 与 [依赖文件](docker/requirements.so101_cuda.txt)。不要在镜像内运行 `uv sync` 或安装通用 PyTorch，以免覆盖 Jetson CUDA 版本。

## 标定与遥操

标定脚本会先在容器内备份已有 JSON，再进入交互式标定：

```bash
scripts/so101_docker.sh calibrate-leader
scripts/so101_docker.sh calibrate-follower
```

已有 JSON 时，提示处输入 `c` 才会重新标定；直接回车会复用旧值。Follower 失力时要托住机械臂，缓慢扫过真实行程；标定后对比两臂同一姿态的肘关节读数。确认范围匹配、两臂姿态接近并清空工作区后，才运行：

```bash
SO101_ARM_READY=1 scripts/so101_docker.sh teleop
```

遇到持续的 `clamped to be safe` 警告，立即 Ctrl+C 停止并检查标定，不要加大限幅。

## 相机与采集

```bash
scripts/so101_docker.sh camera
# 浏览器打开 http://192.168.1.9:8765/；看完按 Ctrl+C

SO101_ARM_READY=1 scripts/so101_docker.sh record
```

脚本在打开相机前会停止旧的宿主网页预览服务，并检查是否仍有进程占用设备；容器预览与采集、推理不能同时运行。采集默认 50 回合、配置 15 fps，保存到宿主 `/media/lihuigang/zhitai/datasets/so101_pick_v1`，不上传 Hub。SSH 终端按 `n` 结束本回合、`r` 重录、`q` 保存并退出。此次容器预览实测前置约 7.5 fps、手眼约 20 fps；**录制前先确认两路相机在所选 `SO101_FPS` 下都能稳定达到目标帧率**，必要时降低共用帧率并使用新的数据集名称。

先试录到独立数据集，或继续已有采集：

```bash
SO101_ARM_READY=1 SO101_DATASET_NAME=so101_trial SO101_EPISODES=2 scripts/so101_docker.sh record
SO101_ARM_READY=1 SO101_DATASET_NAME=so101_trial scripts/so101_docker.sh record --resume
```

## CUDA 训练与推理

采集完成且数据集检查无误后训练。训练不映射串口或相机；默认 ACT、`batch_size=1`、5000 步：

```bash
scripts/so101_docker.sh train
```

模型默认保存到宿主 `/media/lihuigang/zhitai/outputs/so101_act_pick_v1/checkpoints/last/pretrained_model`。仅在标定问题解决、模型已生成且工作区清空后短时试运行：

```bash
SO101_ARM_READY=1 scripts/so101_docker.sh rollout
```

推理默认 20 秒，始终有人在机械臂旁；异常时立即 Ctrl+C 或切断电源。相机名称、分辨率、帧率与关节归一化方式必须和训练数据一致。镜像的入口、CUDA 计算与设备映射已验证；尚未执行真实采集、完整训练或实机推理。

运行 `scripts/so101_docker.sh --help` 可查看全部子命令。可通过 `SO101_DATASET_NAME`、`SO101_OUTPUT_NAME`、`SO101_TASK`、`SO101_FPS`、`SO101_EPISODES`、`SO101_STEPS`、`SO101_DURATION` 覆盖默认值；训练和推理须使用相同的数据集、输出名称和相机参数。
