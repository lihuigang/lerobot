#!/usr/bin/env python
"""Smoke-test Jetson CUDA and one synthetic SO-101 ACT training step.

This does not access the robot, cameras, or a real dataset.
"""

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")

    # Jetson's NVIDIA torch build needs to support the NumPy bridge used by datasets.
    np_array = np.ones((2, 2), dtype=np.float32)
    if torch.from_numpy(np_array).sum().item() != 4:
        raise RuntimeError("NumPy to PyTorch conversion failed")

    config = ACTConfig(
        input_features={
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.images.handeye": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        device="cuda",
        chunk_size=2,
        n_action_steps=2,
        dim_model=128,
        n_heads=4,
        dim_feedforward=256,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
        use_vae=False,
    )
    policy = ACTPolicy(config).to("cuda").train()
    batch = {
        "observation.images.front": torch.rand(1, 3, 64, 64, device="cuda"),
        "observation.images.handeye": torch.rand(1, 3, 64, 64, device="cuda"),
        "observation.state": torch.rand(1, 6, device="cuda"),
        "action": torch.rand(1, 2, 6, device="cuda"),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool, device="cuda"),
    }
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    loss, _ = policy(batch)
    if not torch.isfinite(loss):
        raise RuntimeError("ACT loss is not finite")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(
        "SO101_CUDA_SMOKE_OK",
        f"gpu={torch.cuda.get_device_name(0)}",
        f"torch={torch.__version__}",
        f"numpy={np.__version__}",
        f"act_loss={loss.detach().item():.6f}",
    )


if __name__ == "__main__":
    main()
