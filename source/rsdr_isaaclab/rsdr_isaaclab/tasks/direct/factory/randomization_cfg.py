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
    
    event_type: Literal[
        "mass",
        "friction",
        "gravity",
        "stiffness",
        "damping",
        "reset_noise",
        "pos_threshold",
        "rot_threshold",
        "ema",
    ]
    target_asset: str
    target_indices: Optional[List[int]] = None 
    indices: List[int] = field(default_factory=list)
    learning_indices: List[int] = field(default_factory=list)
    no_learning: bool = False
    visualize: bool = False

@dataclass
class FactoryRandomizationCfg:
    params: List[RandomizationParamCfg] = field(default_factory=list)
    total_params: int = 0
    total_learning_params: int = 0
    only_held_pos_noise_2d: bool = False #True
    # [NEW] Pass the Task Config Class here (e.g. PegInsert) to inherit defaults
    task_class: Type[FactoryTask] = PegInsert 

    def __post_init__(self):
        from .factory_env_cfg import CtrlCfg
        # Instantiate the task config to access its default noise values
        task_cfg = self.task_class()
        ctrl_cfg = CtrlCfg()
        if not self.params:
            self.params = [
                # --- Physics (Dynamics) ---
                # Stiffness (7 DoF)
                RandomizationParamCfg(
                    name="stiffness", size=6, sampler_type="uniform",
                    hard_bounds=([0.3]*6, [2.6]*6), 
                    init_params=([1.]*6), # Start near nominal
                    event_type="stiffness", target_asset="robot", no_learning=False, #visualize=True
                ),
                 # Friction
                RandomizationParamCfg(
                    name="robot_friction", size=1, sampler_type="uniform",
                    hard_bounds=([0.1], [10.0]), 
                    init_params=([0.75]),
                    event_type="friction", target_asset="robot", no_learning=True,#visualize=True
                ),

                RandomizationParamCfg(
                    name="held_friction", size=1, sampler_type="uniform",
                    hard_bounds=([0.1], [10.0]), 
                    init_params=([0.75]),
                    event_type="friction", target_asset="robot", no_learning=True,#visualize=True
                ),
                RandomizationParamCfg(
                    name="fixed_friction", size=1, sampler_type="uniform",
                    hard_bounds=([0.1], [10.0]), 
                    init_params=([0.75]),
                    event_type="friction", target_asset="robot", no_learning=True,#visualize=True
                ),
                # # # --- Reset State (Inherited from Task Config) ---
                
                # # # 3. Fixed Asset Position Noise (3 dims)
                # # # Task Config: fixed_asset_init_pos_noise = [x, y, z] (range +/-)
                RandomizationParamCfg(
                    name="fixed_pos_noise", size=3, sampler_type="uniform",
                    hard_bounds=(
                        [-float(v) for v in task_cfg.fixed_asset_init_pos_noise], 
                        [float(v) for v in task_cfg.fixed_asset_init_pos_noise]
                    ),
                    init_params=(
                        [0.] * 3
                    ),
                    event_type="reset_noise", target_asset="fixed_asset", no_learning=True,#visualize=True
                ),
                
                # # 4. Fixed Asset Yaw Noise (1 dim)
                # # Task Config: fixed_asset_init_orn_range_deg (scalar degrees)
                # # We convert to radians: +/- range,
                RandomizationParamCfg(
                    name="fixed_yaw_noise", size=1, sampler_type="uniform",
                    hard_bounds=(
                        [0], 
                        [float(np.deg2rad(task_cfg.fixed_asset_init_orn_range_deg))]
                    ),
                    init_params=(
                        [0.] 
                    ),
                    event_type="reset_noise", target_asset="fixed_asset", no_learning=True,#visualize=True
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
                    event_type="reset_noise", target_asset="robot", no_learning=False,#visualize=True
                ),

                # # 6. Hand Init Orn Noise (3 dims)
                # # Task Config: hand_init_orn_noise = [r, p, y]
                RandomizationParamCfg(
                    name="hand_init_orn_noise", size=1, sampler_type="uniform",
                    init_params= ([0.]*1),
                    hard_bounds=(
                        [-task_cfg.hand_init_orn_noise[2]], 
                        [task_cfg.hand_init_orn_noise[2]],
                    ),
                    event_type="reset_noise", target_asset="robot", no_learning=True,#visualize=True
                ),
                
                # 7. Held Asset Pos Noise (3 dims)
                # Task Config: held_asset_pos_noise = [x, y, z]
                #TODO : NutThread 의 경우 다르게
                RandomizationParamCfg(
                    name="held_pos_noise", size=2, sampler_type="uniform",
                    init_params=([0.]*2), 
                    hard_bounds=(
                        [-task_cfg.held_asset_pos_noise[0], -task_cfg.held_asset_pos_noise[2]], 
                        [task_cfg.held_asset_pos_noise[0], task_cfg.held_asset_pos_noise[2]], 
                    ),
                    event_type="reset_noise", target_asset="held_asset", no_learning=False,#visualize=True
                ),

                RandomizationParamCfg(
                    name="pos_threshold", size=3, sampler_type="uniform",
                    init_params=([1]*3), 
                    hard_bounds=(
                        [1/(1+v) for v in ctrl_cfg.pos_threshold_noise_level], 
                        [(1+v) for v in ctrl_cfg.pos_threshold_noise_level]
                    ),
                    event_type="pos_threshold", target_asset="robot", no_learning=True,#visualize=True
                ),
                RandomizationParamCfg(
                    name="rot_threshold", size=3, sampler_type="uniform",
                    init_params=([1]*3), 
                    hard_bounds=(
                        [1/(1+v) for v in ctrl_cfg.rot_threshold_noise_level], 
                        [(1+v) for v in ctrl_cfg.rot_threshold_noise_level]
                    ),
                    event_type="rot_threshold", target_asset="robot", no_learning=True,#visualize=True
                ),
                RandomizationParamCfg(
                    name="ema", size=1, sampler_type="uniform",
                    init_params=([ctrl_cfg.ema_factor]), 
                    hard_bounds=(
                        [ctrl_cfg.ema_factor_range[0]],
                        [ctrl_cfg.ema_factor_range[1]]
                    ),
                    event_type="ema", target_asset="robot", no_learning=True,#visualize=True
                ),
            ]
        if self.only_held_pos_noise_2d:
            self.params = [p for p in self.params if p.name == "held_pos_noise"]

        # AUTO-INDEXING LOGIC
        current_idx = 0
        current_learning_idx = 0
        for p in self.params:
            p.indices = list(range(current_idx, current_idx + p.size))
            if p.no_learning:
                p.learning_indices = []
            else:
                p.learning_indices = list(range(current_learning_idx, current_learning_idx + p.size))
                current_learning_idx += p.size
            current_idx += p.size
        self.total_params = current_idx
        self.total_learning_params = current_learning_idx
