# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asynchronous evaluator for RL-Games checkpoints.

Run this script in a separate process from training.
It watches a checkpoint directory and evaluates each new checkpoint with a larger eval env count.
"""

import argparse
import glob
import math
import os
import random
import re
import sys
import time
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Asynchronously evaluate RL-Games checkpoints.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments for evaluation.")
parser.add_argument("--eval_episodes", type=int, default=10, help="Number of eval episodes per checkpoint.")
parser.add_argument(
    "--deterministic",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="Use deterministic policy actions for evaluation.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing checkpoint .pth files.")
parser.add_argument(
    "--checkpoint_glob",
    type=str,
    default="last_*.pth",
    help="Glob for checkpoint files inside checkpoint_dir.",
)
parser.add_argument(
    "--include_best_checkpoint",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="Also include the fixed-name best checkpoint <config_name>.pth.",
)
parser.add_argument(
    "--run_dir",
    type=str,
    default=None,
    help="Training run directory name. Defaults to <task>-rl_games-seed=<seed>.",
)
parser.add_argument("--poll_interval", type=float, default=30.0, help="Seconds between checkpoint scans.")
parser.add_argument(
    "--mode",
    type=str,
    default="watch",
    choices=["watch", "all", "latest"],
    help="watch: run continuously, all: evaluate all checkpoints once and exit, latest: evaluate latest checkpoint once.",
)
parser.add_argument("--wandb-project-name", type=str, default="factory-peginsert", help="W&B project name.")
parser.add_argument("--wandb-entity", type=str, default="tjrcjf410-seoul-national-university", help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="Log asynchronous eval metrics to Weights and Biases.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import rsdr_isaaclab.tasks  # noqa: F401


def _to_float(v):
    if isinstance(v, (float, int)):
        return float(v)
    if torch.is_tensor(v):
        if v.numel() == 1:
            return float(v.item())
        return float(v.float().mean().item())
    try:
        return float(v)
    except Exception:
        return None


def _extract_step_from_ckpt_name(path: str) -> int | None:
    name = os.path.basename(path)
    for pat in (r"epoch[_=](\d+)", r"ep[_=](\d+)", r"frame[_=](\d+)", r"step[_=](\d+)"):
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    m = re.search(r"_(\d+)\.pth$", name)
    if m:
        return int(m.group(1))
    return None

def _extract_global_step_from_ckpt(path: str) -> int | None:
    """Try to read training global step (frames/steps) from RL-Games checkpoint."""
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return None

    # Common keys seen in RL-Games / IsaacLab training runs
    candidate_keys = [
        "frame", "frames", "total_frames", "step", "steps",
        "global_step", "env_steps", "num_frames",
    ]

    # sometimes nested
    candidate_paths = [
        ("frame",),
        ("frames",),
        ("total_frames",),
        ("step",),
        ("steps",),
        ("global_step",),
        ("env_steps",),
        ("num_frames",),
        ("stats", "frame"),
        ("stats", "frames"),
        ("counters", "frames"),
        ("counters", "steps"),
        ("config", "frame"),
    ]

    def get_nested(d, keys):
        x = d
        for k in keys:
            if not isinstance(x, dict) or k not in x:
                return None
            x = x[k]
        return x

    # 1) flat keys
    if isinstance(ckpt, dict):
        for k in candidate_keys:
            if k in ckpt:
                v = ckpt[k]
                if isinstance(v, (int, float)):
                    return int(v)

        # 2) nested keys
        for kp in candidate_paths:
            v = get_nested(ckpt, kp)
            if isinstance(v, (int, float)):
                return int(v)

    return None
def _find_checkpoints(ckpt_dir: str, checkpoint_glob: str, include_best: bool, best_name: str) -> list[str]:
    paths = glob.glob(os.path.join(ckpt_dir, checkpoint_glob))
    if include_best:
        best_path = os.path.join(ckpt_dir, f"{best_name}.pth")
        if os.path.exists(best_path):
            paths.append(best_path)

    # de-duplicate
    paths = list(set(paths))

    def sort_key(p: str):
        step = _extract_step_from_ckpt_name(p)
        # Put files without step at the end, but keep stable ordering by mtime
        step_key = step if step is not None else 10**18
        return (step_key, os.path.getmtime(p), p)

    return sorted(paths, key=sort_key)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]

    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", config_name))
    seed = agent_cfg["params"]["seed"]
    run_dir = args_cli.run_dir if args_cli.run_dir is not None else f"{args_cli.task}-rl_games-seed={seed}"
    ckpt_dir = (
        args_cli.checkpoint_dir
        if args_cli.checkpoint_dir is not None
        else os.path.join(log_root_path, run_dir, "nn")
    )
    env_cfg.log_dir = os.path.join(log_root_path, run_dir, "eval_async")
    print(f"[INFO] Watching checkpoints in: {ckpt_dir}")
    os.makedirs(env_cfg.log_dir, exist_ok=True)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    env = RlGamesVecEnvWrapper(raw_env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()

    wandb = None
    if args_cli.track:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb as _wandb

        wandb = _wandb
        group_name=  None
        if group_name is None:
            experiment_name = args_cli.wandb_name if args_cli.wandb_name is not None else run_dir
            exp_name = experiment_name.split("-Direct")[0]
            exp_name = exp_name.split("Factory-")[-1]
            sampler_name = (
                raw_env.unwrapped.sampler.name
                if hasattr(raw_env.unwrapped, "sampler") and raw_env.unwrapped.sampler is not None
                else "no_sampler"
            )
            group_name = f"{exp_name}_{sampler_name}"
            if sampler_name == "GMMVI":
                group_name += f"-beta={raw_env.unwrapped.sampler.beta}"
            elif sampler_name == "GOFLOW":
                group_name += f"-beta={1 / raw_env.unwrapped.sampler.alpha}-gamma={raw_env.unwrapped.sampler.beta}"
            elif sampler_name == "DORAEMON":
                group_name += (
                    f"-thres={raw_env.unwrapped.sampler.success_threshold}"
                    f"-rate={raw_env.unwrapped.sampler.success_rate_condition}"
                    f"-kl={raw_env.unwrapped.sampler.kl_upper_bound}"
                )
            elif sampler_name == "ADR":
                group_name += f"-thres={raw_env.unwrapped.sampler.success_threshold}"
        group_name +=f"-eval"
        wandb.init(
            project=args_cli.wandb_project_name,
            entity=args_cli.wandb_entity,
            name=args_cli.wandb_name if args_cli.wandb_name is not None else f"{run_dir}-async-eval",
            group=group_name,
            save_code=True,
        )
        agent_cfg['group'] = group_name
        if not wandb.run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg, "group":group_name})
    def run_eval(checkpoint_path: str) -> dict[str, float]:
        print(f"[INFO] Evaluating checkpoint: {checkpoint_path}")
        agent.restore(checkpoint_path)
        agent.reset()

        

        factory_env = raw_env.unwrapped
        factory_env.set_uniform_eval(True)
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()
        try:
            horizon = int(getattr(factory_env, "max_episode_length", 200))
            for _ in range(args_cli.eval_episodes):
                obs = env.reset()
                if isinstance(obs, dict):
                    obs = obs["obs"]
                if agent.is_rnn:
                    agent.init_rnn()

                for _ in range(horizon - 1):
                    with torch.no_grad():
                        obs_t = agent.obs_to_torch(obs)
                        actions = agent.get_action(obs_t, is_deterministic=args_cli.deterministic)
                        obs, _, dones, _ = env.step(actions)
                        if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                            for s in agent.states:
                                s[:, dones, :] = 0.0
        finally:
            if hasattr(factory_env, "_first_reset"):
                factory_env._first_reset = True
            env.reset()

        out = {}
        for k, v in dict(getattr(factory_env, "extras", {})).items():
            if "eval/" in k or k.startswith("eval"):
                val = _to_float(v)
                if val is not None:
                    out[k] = val
        return out

    evaluated = set()
    while simulation_app.is_running():
        checkpoints = _find_checkpoints(
            ckpt_dir=ckpt_dir,
            checkpoint_glob=args_cli.checkpoint_glob,
            include_best=args_cli.include_best_checkpoint,
            best_name=config_name,
        )
        if not checkpoints:
            if args_cli.mode == "watch":
                print(f"[INFO] No checkpoints found yet in {ckpt_dir}. Sleeping for {args_cli.poll_interval:.1f}s.")
                time.sleep(args_cli.poll_interval)
                continue
            print(f"[INFO] No checkpoints found in {ckpt_dir}. Exiting.")
            break

        pending = [p for p in checkpoints if p not in evaluated]
        if args_cli.mode == "latest":
            latest = checkpoints[-1]
            pending = [latest] if latest not in evaluated else []
        elif args_cli.mode == "all":
            pending = [p for p in checkpoints if p not in evaluated]

        for checkpoint in pending:
            import traceback
            try:
                metrics = run_eval(checkpoint)
            except Exception as exc:
                print(f"[WARN] Evaluation failed for {checkpoint}: {exc}")
                traceback.print_exc()
                evaluated.add(checkpoint)
                continue
            global_step = _extract_global_step_from_ckpt(checkpoint)

            print(f"[INFO] wandb_step={global_step} metrics={metrics}")

            if wandb is not None and metrics:
                payload = {k: v for k, v in metrics.items()}
                payload["async_eval/num_envs"] = float(args_cli.num_envs)
                payload["async_eval/eval_episodes"] = float(args_cli.eval_episodes)
                if global_step is not None:
                    wandb.log(payload, step=global_step)
                else:
                    wandb.log(payload)
            # step = _extract_step_from_ckpt_name(checkpoint)
            # if wandb is not None and metrics:
            #     payload = {k: v for k, v in metrics.items()}
            #     payload["async_eval/num_envs"] = float(args_cli.num_envs)
            #     payload["async_eval/eval_episodes"] = float(args_cli.eval_episodes)
            #     if step is not None:
            #         wandb.log(payload, step=step)
            #     else:
            #         wandb.log(payload)
            evaluated.add(checkpoint)

        if args_cli.mode in ("latest", "all"):
            break

        time.sleep(args_cli.poll_interval)

    env.close()
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
    simulation_app.close()
