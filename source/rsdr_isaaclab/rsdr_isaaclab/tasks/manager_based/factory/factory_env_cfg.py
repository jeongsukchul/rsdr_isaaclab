# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as NoiseCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.envs.mdp as mdp

# --- Custom MDP Imports ---
from rsdr_isaaclab.tasks.manager_based.factory import mdp as factory_mdp
from rsdr_isaaclab.tasks.manager_based.factory.mdp import actions as factory_actions
from rsdr_isaaclab.tasks.manager_based.factory.mdp import events as factory_events
from rsdr_isaaclab.tasks.manager_based.factory.mdp import metrics as factory_metrics
from rsdr_isaaclab.tasks.manager_based.factory.factory_tasks_cfg import FactoryTask
# --- Task Config Import ---
from rsdr_isaaclab.tasks.manager_based.factory.factory_tasks_cfg import ASSET_DIR, PegInsert, GearMesh, NutThread, FactoryTask
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
# -------------------------------------------------------------------------
# Scene Configuration
# -------------------------------------------------------------------------
@configclass
class FactorySceneCfg(InteractiveSceneCfg):
    """Configuration for the Factory Scene (Robot + Assets)."""
    
    # 1. Robot (Franka Mimic)
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    # 2. Placeholders (Populated dynamically by _apply_task_settings)
    # Note: Explicitly define actuators={} to handle them as passive Articulations
    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=None,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=None,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=None,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=None,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    from isaaclab.assets import AssetBaseCfg
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", 
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05))
    )
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", #
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), #
            rot=(0.70711, 0.0, 0.0, 0.70711) #
        ),
    )

    # 3. Dome Light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0, #
            color=(0.75, 0.75, 0.75) #
        )
    )
# -------------------------------------
# ------------------------------------
# Action & Observation Configurations
# -------------------------------------------------------------------------


@configclass
class FactoryActionsCfg:
    """Factory Action Space using Exact FactoryEnv Logic."""
    arm_action = factory_actions.FactoryTaskSpaceControlCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_fingertip_centered",
        fixed_asset_name="fixed_asset",
        
        ema_factor = 0.2,

        pos_action_bounds = [0.05, 0.05, 0.05],
        rot_action_bounds = [1.0, 1.0, 1.0],

        pos_action_threshold = [0.02, 0.02, 0.02],
        rot_action_threshold = [0.097, 0.097, 0.097],

        reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01],
        reset_task_prop_gains = [300, 300, 300, 20, 20, 20],
        reset_rot_deriv_scale = 10.0,
        default_task_prop_gains = [100, 100, 100, 30, 30, 30],

        # Null space parameters.
        default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754],
        kp_null = 10.0,
        kd_null = 6.3246,
    )

@configclass
class FactoryObservationsCfg:
    """Factory Observations (Policy + Critic/State)."""
    
    # -------------------------------------------------------------------
    # POLICY (Actor)
    # -------------------------------------------------------------------
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. Fingertip Relative to Fixed Asset (Noisy)
        fingertip_pos_rel_fixed = ObsTerm(
            func=factory_mdp.body_pos_rel_fixed,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered"), 
                "body_name" : "panda_fingertip_centered",
            },
            noise=NoiseCfg(n_min=-0.001, n_max=0.001),
        )
        
        # 2. Fingertip Orientation
        fingertip_quat = ObsTerm(
            func=factory_mdp.body_world_orientation,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        
        # 3. Fingertip Position (Absolute)
        fingertip_pos = ObsTerm(
            func=factory_mdp.body_world_position,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )

        # 4. Finite Difference Velocities (Simulating real sensor noise)
        ee_linvel = ObsTerm(
            func=factory_mdp.FiniteDifferenceVelocityObs,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered"),
                "data_type": "linear"
            },
            noise=NoiseCfg(n_min=-0.01, n_max=0.01),
        )
        ee_angvel = ObsTerm(
            func=factory_mdp.FiniteDifferenceVelocityObs,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered"),
                "data_type": "angular"
            },
            noise=NoiseCfg(n_min=-0.01, n_max=0.01),
        )

        # 5. Previous Actions
        prev_actions = ObsTerm(func=mdp.last_action)

    # -------------------------------------------------------------------
    # CRITIC (State)
    # -------------------------------------------------------------------
    @configclass
    class CriticCfg(ObsGroup):
        # --- Robot State ---
        fingertip_pos = ObsTerm(
            func=factory_mdp.body_world_position,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        # Note: Critic gets the "True" relative distance (no noise)
        fingertip_pos_rel_fixed = ObsTerm(
            func=factory_mdp.body_pos_rel_fixed,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered"), 
                "body_name" : "panda_fingertip_centered",
            },
        )
        fingertip_quat = ObsTerm(
            func=factory_mdp.body_world_orientation,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        # Note: Critic gets TRUE physics velocity
        ee_linvel = ObsTerm(
            func=factory_mdp.body_world_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        ee_angvel = ObsTerm(
            func=factory_mdp.body_world_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        joint_pos = ObsTerm(
            func=factory_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
        )

        # --- Asset States ---
        held_pos = ObsTerm(
            func=factory_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("held_asset")},
        )
        held_quat = ObsTerm(
            func=factory_mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("held_asset")},
        )
        # Held relative to fixed (using generic function)
        held_pos_rel_fixed = ObsTerm(
            func=factory_mdp.body_pos_rel_fixed, 
            params={
                "robot_cfg": SceneEntityCfg("held_asset"), 
                "body_name" : "forge_round_peg_8mm",
            },
        )
        fixed_pos = ObsTerm(
            func=factory_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("fixed_asset")},
        )
        fixed_quat = ObsTerm(
            func=factory_mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("fixed_asset")},
        )

        # --- Controller Internals ---
        task_prop_gains = ObsTerm(
            func=factory_mdp.action_term_param,
            params={"action_term_name": "arm_action", "param_name": "task_prop_gains"},
        )
        pos_threshold = ObsTerm(
            func=factory_mdp.action_term_param,
            params={"action_term_name": "arm_action", "param_name": "pos_threshold"},
        )
        rot_threshold = ObsTerm(
            func=factory_mdp.action_term_param,
            params={"action_term_name": "arm_action", "param_name": "rot_threshold"},
        )
        prev_actions = ObsTerm(func=mdp.last_action)


    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]

# -------------------------------------------------------------------------
# Event Configurations (Resets & Randomization)
# -------------------------------------------------------------------------
@configclass
class FactoryEventCfg:
    """Events for Reset and Domain Randomization."""
    startup_layout = EventTerm(
        func=factory_events.reset_factory_assets_with_ik,
        mode="startup",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "task_cfg": None, 
        }
    )
    reset_layout = EventTerm(
        func=factory_events.reset_factory_assets_with_ik,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "task_cfg" : None, 
        }
    )

    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint.*"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    stabilize_inertia = EventTerm(
        func=factory_events.set_body_inertias,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "offset_value": 0.01,
        }
    )

    gui_debug_vis = EventTerm(
        func=factory_events.update_debug_vis,
        mode="interval",
        interval_range_s=(0.02,0.02)
    )


# -------------------------------------------------------------------------
# Base Environment Configuration
# -------------------------------------------------------------------------
@configclass
class FactoryEnvCfg(ManagerBasedRLEnvCfg):
    """Base Factory Environment Configuration."""
    
    # Managers
    scene: FactorySceneCfg = FactorySceneCfg(num_envs=128, env_spacing=2.0)
    observations: FactoryObservationsCfg = FactoryObservationsCfg()
    actions: FactoryActionsCfg = FactoryActionsCfg()
    events: FactoryEventCfg = FactoryEventCfg()
    task: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    # Environment Settings
    decimation: int = 8
    episode_length_s: float = 5.0 # Placeholder, overwritten bFdefault_y task settings

    @configclass
    class TerminationsCfg:
        # Time out based on episode_length_s
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    terminations: TerminationsCfg = TerminationsCfg()

    # Physics Settings (High Precision)
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.001,
            gpu_max_rigid_contact_count=2**23,
        ),
    )
    
    @configclass
    class CurriculumCfg:
        # Full Factory Metrics Logger (Time-to-Success + Success Rate)
        success = mdp.CurriculumTermCfg(
            func=factory_metrics.log_factory_success_metrics,
            params={
                "held_asset_cfg": SceneEntityCfg("held_asset"),
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            },
        )
        failure = mdp.CurriculumTermCfg(
            func=factory_metrics.log_grasp_stability,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "held_asset_cfg": SceneEntityCfg("held_asset"),
            }
        )
        log_reward_components = mdp.CurriculumTermCfg(
            func=factory_metrics.log_reward_components,
        )
        
        factory_stats = mdp.CurriculumTermCfg(
            func=factory_metrics.log_factory_statistics,
            params={
                "held_asset_cfg": SceneEntityCfg("held_asset"),
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                "task_cfg": None, 
            },
        )
        #For debug
        spawn_sanity_check = mdp.CurriculumTermCfg(
            func=factory_metrics.check_first_frame_stats,
        )
        # debug_obs_freshness = mdp.CurriculumTermCfg(
        #    func=factory_metrics.debug_observation_freshness,
        # )
        
    
    curriculum: CurriculumCfg = CurriculumCfg()

# -------------------------------------------------------------------------
# Dynamic Configuration Helpers
# -------------------------------------------------------------------------
def _build_rewards_cfg(task_cfg: FactoryTask) -> object:
    """Dynamically builds a RewardsCfg class using values from the FactoryTask config."""
    
    @configclass
    class TaskRewardsCfg:
        # Keypoint Rewards (Baseline, Coarse, Fine)
        kp_baseline = RewTerm(
            func=factory_mdp.factory_keypoint_optimization, weight=1.0,
            params={
                "factory_task_cfg": task_cfg, "keypoint_level": "baseline",
                "held_asset_cfg": SceneEntityCfg("held_asset"), "fixed_asset_cfg": SceneEntityCfg("fixed_asset")
            },
        )
        kp_coarse = RewTerm(
            func=factory_mdp.factory_keypoint_optimization, weight=1.0,
            params={
                "factory_task_cfg": task_cfg, "keypoint_level": "coarse",
                "held_asset_cfg": SceneEntityCfg("held_asset"), "fixed_asset_cfg": SceneEntityCfg("fixed_asset")
            },
        )
        kp_fine = RewTerm(
            func=factory_mdp.factory_keypoint_optimization, weight=1.0,
            params={
                "factory_task_cfg": task_cfg, "keypoint_level": "fine",
                "held_asset_cfg": SceneEntityCfg("held_asset"), "fixed_asset_cfg": SceneEntityCfg("fixed_asset")
            },
        )
        
        # Penalties
        action_penalty = RewTerm(
            func=factory_mdp.factory_action_l2_penalty, 
            weight=-task_cfg.action_penalty_ee_scale
        )
        action_rate = RewTerm(
            func=factory_mdp.factory_action_rate_l2_penalty, 
            weight=-task_cfg.action_grad_penalty_scale
        )

        # Engagement (Sparse)
        engagement = RewTerm(
            func=factory_mdp.factory_pose_alignment_reward, weight=1.0,
            params={
                "factory_task_cfg": task_cfg, "mode": "engagement",
                "held_asset_cfg": SceneEntityCfg("held_asset"), "fixed_asset_cfg": SceneEntityCfg("fixed_asset")
            },
        )
        # Success (Sparse)
        success = RewTerm(
            func=factory_mdp.factory_pose_alignment_reward, weight=10.0,
            params={
                "factory_task_cfg": task_cfg, "mode": "success",
                "held_asset_cfg": SceneEntityCfg("held_asset"), "fixed_asset_cfg": SceneEntityCfg("fixed_asset")
            },
        )

    return TaskRewardsCfg()


# rsdr_isaaclab/tasks/manager_based/factory/config/factory_env_cfg.py

def _apply_task_settings(env_cfg: FactoryEnvCfg, task_cfg: FactoryTask):
    """Updates the EnvCfg Scene with assets and Rewards from the Task Config."""
    
    # 1. Sync Episode Settings
    env_cfg.episode_length_s = task_cfg.duration_s
    env_cfg.curriculum.success.params["factory_task_cfg"] = task_cfg

    # 2. Update Held Asset (Adopts the FULL config from factory_tasks_cfg.py)

    env_cfg.scene.held_asset = task_cfg.held_asset

    env_cfg.scene.fixed_asset = task_cfg.fixed_asset
    if task_cfg.name == "gear_mesh":
        env_cfg.scene.small_gear = task_cfg.small_gear_cfg
        env_cfg.scene.large_gear = task_cfg.large_gear_cfg
    # 4. Inject Rewards
    env_cfg.rewards = _build_rewards_cfg(task_cfg)

    # 5. Inject Friction Events
    env_cfg.events.set_held_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (task_cfg.held_asset_cfg.friction, task_cfg.held_asset_cfg.friction),
            "dynamic_friction_range": (task_cfg.held_asset_cfg.friction, task_cfg.held_asset_cfg.friction),
            "num_buckets": 1,
            "restitution_range": (0.0, 0.0),
        }
    )
    env_cfg.events.set_fixed_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (task_cfg.fixed_asset_cfg.friction, task_cfg.fixed_asset_cfg.friction),
            "dynamic_friction_range": (task_cfg.fixed_asset_cfg.friction, task_cfg.fixed_asset_cfg.friction),
            "num_buckets": 1,
            "restitution_range": (0.0, 0.0),
        }
    )

    #FactoryEnv.__init__ -> set_body_inertias
    env_cfg.events.stabilize_inertia = EventTerm(
        func=factory_events.set_body_inertias,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "offset_value": 0.01,
        }
    )
    env_cfg.events.reset_layout.params["task_cfg"] = task_cfg
    env_cfg.events.startup_layout.params["task_cfg"] = task_cfg
    if task_cfg.name=="peg_insert":
        body_name="forge_round_peg_8mm"
    elif task_cfg.name=='gear_mesh':
        body_name="factory_gear_medium"
    elif task_cfg.name=="nut_thread":
        body_name="factory_nut_loose"
    else:
        raise ValueError(f"no such task :  {task_cfg.name}")
    env_cfg.observations.critic.held_pos_rel_fixed.params["body_name"] = body_name
    env_cfg.curriculum.factory_stats.params["task_cfg"] = task_cfg
# -------------------------------------------------------------------------
# Final Environment Configurations (Peg, Gear, Nut)
# -------------------------------------------------------------------------

@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        task = PegInsert()
        _apply_task_settings(self, task)

@configclass
class FactoryTaskGearMeshCfg(FactoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        task = GearMesh()
        _apply_task_settings(self, task)

@configclass
class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        task = NutThread()
        _apply_task_settings(self, task)
        # Enable unidirectional rotation for Nut Threading
        self.actions.arm_action.unidirectional_rot = True