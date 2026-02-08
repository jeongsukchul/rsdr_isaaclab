# rsdr_isaaclab/tasks/manager_based/factory/mdp/randomization_cfg.py

from dataclasses import dataclass, field
from typing import Literal, Tuple, List, Union, Optional, Type
import torch
import numpy as np

# Import your task configs to use as defaults
from .factory_tasks_cfg import FactoryTask, PegInsert, GearMesh, NutThread

@dataclass
class RandomizationParamCfg:
    name: str
    size: int
    sampler_type: Literal["uniform", "gaussian", "custom"]
    
    # Bounds: Can be scalar (broadcast) or list (per-dimension)
    # Example for size=3: ([min_x, min_y, min_z], [max_x, max_y, max_z])
    hard_bounds: Tuple[Union[float, List[float]], Union[float, List[float]]]
    init_params: Tuple[Union[float, List[float]], Union[float, List[float]]]
    
    event_type: Literal["mass", "friction", "gravity", "stiffness", "damping", "reset_noise"]
    target_asset: str
    target_indices: Optional[List[int]] = None 
    indices: List[int] = field(default_factory=list)

@dataclass
class FactoryRandomizationCfg:
    params: List[RandomizationParamCfg] = field(default_factory=list)
    total_params: int = 0
    
    # [NEW] Pass the Task Config Class here (e.g. PegInsert) to inherit defaults
    task_class: Type[FactoryTask] = PegInsert 

    def __post_init__(self):
        # Instantiate the task config to access its default noise values
        task_cfg = self.task_class()
        
        if not self.params:
            self.params = [
                # --- Physics (Dynamics) ---
                # 1. Stiffness (7 DoF)
                # RandomizationParamCfg(
                #     name="stiffness", size=7, sampler_type="uniform",
                #     hard_bounds=([0.75]*7, [1.25]*7), 
                #     init_params=([1.]*7), # Start near nominal
                #     event_type="stiffness", target_asset="robot",
                # ),
                # # 2. Damping (7 DoF)
                # RandomizationParamCfg(
                #     name="damping", size=7, sampler_type="uniform",
                #     hard_bounds=([0.75]*7, [1.25]*7), 
                #     init_params=([1.0]*7),
                #     event_type="damping", target_asset="robot"
                # ),
                
                # --- Reset State (Inherited from Task Config) ---
                
                # 3. Fixed Asset Position Noise (3 dims)
                # Task Config: fixed_asset_init_pos_noise = [x, y, z] (range +/-)
                RandomizationParamCfg(
                    name="fixed_pos_noise", size=3, sampler_type="uniform",
                    hard_bounds=(
                        [-float(v) for v in task_cfg.fixed_asset_init_pos_noise], 
                        [float(v) for v in task_cfg.fixed_asset_init_pos_noise]
                    ),
                    init_params=(
                        [0.] * 3
                    ),
                    event_type="reset_noise", target_asset="fixed_asset"
                ),
                
                # 4. Fixed Asset Yaw Noise (1 dim)
                # Task Config: fixed_asset_init_orn_range_deg (scalar degrees)
                # We convert to radians: +/- range
                RandomizationParamCfg(
                    name="fixed_yaw_noise", size=1, sampler_type="uniform",
                    hard_bounds=(
                        [-float(np.deg2rad(task_cfg.fixed_asset_init_orn_range_deg))], 
                        [float(np.deg2rad(task_cfg.fixed_asset_init_orn_range_deg))]
                    ),
                    init_params=(
                        [0.] 
                    ),
                    event_type="reset_noise", target_asset="fixed_asset",
                ),

                # 5. Hand Init Pos Noise (3 dims)
                # Task Config: hand_init_pos_noise = [x, y, z]
                RandomizationParamCfg(
                    name="hand_init_pos_noise", size=3, sampler_type="uniform",
                    hard_bounds=(
                        [-v for v in task_cfg.hand_init_pos_noise], 
                        [v for v in task_cfg.hand_init_pos_noise]
                    ),
                    init_params=(
                        [0.] * 3
                    ),
                    event_type="reset_noise", target_asset="robot"
                ),

                # 6. Hand Init Orn Noise (3 dims)
                # Task Config: hand_init_orn_noise = [r, p, y]
                RandomizationParamCfg(
                    name="hand_init_orn_noise", size=3, sampler_type="uniform",
                    init_params= ([0.]*3),
                    hard_bounds=(
                        [-v for v in task_cfg.hand_init_orn_noise], 
                        [v for v in task_cfg.hand_init_orn_noise]
                    ),
                    event_type="reset_noise", target_asset="robot"
                ),
                
                # 7. Held Asset Pos Noise (3 dims)
                # Task Config: held_asset_pos_noise = [x, y, z]
                RandomizationParamCfg(
                    name="held_pos_noise", size=3, sampler_type="uniform",
                    init_params=([0.]*3), 
                    hard_bounds=(
                        [-v for v in task_cfg.held_asset_pos_noise], 
                        [v for v in task_cfg.held_asset_pos_noise]
                    ),
                    event_type="reset_noise", target_asset="held_asset"
                ),
            ]

        # AUTO-INDEXING LOGIC
        current_idx = 0
        for p in self.params:
            p.indices = list(range(current_idx, current_idx + p.size))
            current_idx += p.size
        self.total_params = current_idx