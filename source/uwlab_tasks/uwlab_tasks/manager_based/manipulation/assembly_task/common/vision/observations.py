# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Vision observation terms for assembly tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera


def process_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    process_image: bool = True,
    output_size: tuple = (224, 224),
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If process_image is True, returns ImageNet-normalized float tensor (B, C, H, W).
    If process_image is False, returns raw uint8 tensor (B, H, W, C).

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from.
        data_type: The data type to pull from the desired camera. Only "rgb" is supported.
        process_image: Whether to normalize and permute to CHW float. If False, returns HWC uint8.
        output_size: Spatial size to resize to as (H, W).

    Returns:
        The images produced at the last time-step.
    """
    assert data_type == "rgb", "Only RGB images are supported for now."
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    images = sensor.data.output[data_type].clone()

    start_dims = torch.arange(len(images.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    current_size = (images.shape[s + 1], images.shape[s + 2])

    images = images.to(dtype=torch.float32)
    images.div_(255.0).clamp_(0.0, 1.0)
    images = images.permute(start_dims + [s + 3, s + 1, s + 2])

    if current_size != output_size:
        images = F.interpolate(images, size=output_size, mode="bilinear", antialias=True)

    if not process_image:
        reverse_dims = torch.argsort(torch.tensor(start_dims + [s + 3, s + 1, s + 2]))
        images = images.permute(reverse_dims.tolist())
        images.mul_(255.0).clamp_(0, 255)
        images = images.to(dtype=torch.uint8)

    return images
