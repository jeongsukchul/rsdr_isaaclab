from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union


@dataclass
class RandomizationParamCfg:
    name: str
    size: int
    sampler_type: Literal["uniform", "gaussian", "custom"]
    hard_bounds: Tuple[Union[float, List[float]], Union[float, List[float]]]
    init_params: Tuple[Union[float, List[float]], Union[float, List[float]]] | Union[float, List[float]]
    event_type: Literal["reset_noise", "mass", "friction", "gravity", "stiffness", "damping"]
    target_asset: str
    target_indices: Optional[List[int]] = None
    indices: List[int] = field(default_factory=list)
    visualize: bool = True


@dataclass
class FrankaCabinetRandomizationCfg:
    params: List[RandomizationParamCfg] = field(default_factory=list)
    total_params: int = 0

    def __post_init__(self):
        if not self.params:
            self.params = [
                RandomizationParamCfg(
                    name="robot_friction_scale",
                    size=1,
                    sampler_type="uniform",
                    hard_bounds=([0.1], [3.0]),
                    init_params=([1.0]),
                    event_type="friction",
                    target_asset="robot",
                    visualize=True,
                ),
                RandomizationParamCfg(
                    name="cabinet_friction_scale",
                    size=1,
                    sampler_type="uniform",
                    hard_bounds=([0.1], [3.0]),
                    init_params=([1.0]),
                    event_type="friction",
                    target_asset="cabinet",
                    visualize=True,
                ),
                RandomizationParamCfg(
                    name="robot_mass_scale",
                    size=1,
                    sampler_type="uniform",
                    hard_bounds=([0.1], [5.]),
                    init_params=([1.0]),
                    event_type="mass",
                    target_asset="robot",
                    visualize=True,
                ),
                RandomizationParamCfg(
                    name="cabinet_mass_scale",
                    size=1,
                    sampler_type="uniform",
                    hard_bounds=([0.1], [5.]),
                    init_params=([1.0]),
                    event_type="mass",
                    target_asset="cabinet",
                    visualize=True,
                ),
            ]

        current_idx = 0
        for p in self.params:
            p.indices = list(range(current_idx, current_idx + p.size))
            current_idx += p.size
        self.total_params = current_idx
