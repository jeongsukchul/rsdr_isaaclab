# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def randomize_dead_zone(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None):
    env.dead_zone_thresholds = (
        torch.rand((env.num_envs, 6), dtype=torch.float32, device=env.device) * env.default_dead_zone
    )
