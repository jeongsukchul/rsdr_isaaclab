# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Custom-Factory-PegInsert-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Custom-Factory-PegInsert-Direct-NoDR-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsert_NoDR_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Custom-Factory-PegInsert-Direct-ADR-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsert_ADR_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Custom-Factory-PegInsert-Direct-DORAEMON-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsert_DORAEMON_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Custom-Factory-PegInsert-Direct-GOFLOW-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsert_GOFLOW_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Custom-Factory-GearMesh-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskGearMeshCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Custom-Factory-GearMesh-Direct-NoDR-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskGearMesh_NoDR_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Custom-Factory-GearMesh-Direct-ADR-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskGearMesh_ADR_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Custom-Factory-GearMesh-Direct-DORAEMON-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskGearMesh_DORAEMON_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Custom-Factory-NutThread-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskNutThreadCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Custom-Factory-PegInsert-Direct-NoDR-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsert_NoDR_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Custom-Factory-PegInsert-Direct-ADR-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsert_ADR_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Custom-Factory-PegInsert-Direct-DORAEMON-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsert_DORAEMON_Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
