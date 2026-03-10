import copy

import torch
import wandb
from rl_games.common.algo_observer import IsaacAlgoObserver

from rsdr_isaaclab.wandb_observer import IsaacWandbAlgoObserver


def _zeros_like_rnn(template, new_batch: int, device):
    """Create zero rnn state with matching structure and batch size."""
    if template is None:
        return None
    if torch.is_tensor(template):
        shape = list(template.shape)
        # RL-Games recurrent state is typically [num_layers, batch, hidden].
        # Keep layer dimension intact and replace the batch dimension.
        if len(shape) >= 2:
            shape[1] = new_batch
        else:
            shape[0] = new_batch
        return torch.zeros(shape, dtype=template.dtype, device=device)
    if isinstance(template, (list, tuple)):
        out = [_zeros_like_rnn(x, new_batch, device) for x in template]
        return type(template)(out)
    if isinstance(template, dict):
        return {k: _zeros_like_rnn(v, new_batch, device) for k, v in template.items()}
    raise TypeError(f"Unsupported rnn state type: {type(template)}")


def _clone_if_tensor(x):
    return x.clone() if torch.is_tensor(x) else copy.deepcopy(x)


class UniformEvalObserver(IsaacWandbAlgoObserver):


    def _get_env(self):
        v = self.algo.vec_env
        base = getattr(v, "env", None) or getattr(v, "_env", None) or v
        return getattr(base, "unwrapped", None)

    @staticmethod
    def _to_scalar(value):
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            return None
        if isinstance(value, (float, int)):
            return value
        return None

    def _log_scalar(self, key: str, value, step: int):
        scalar = self._to_scalar(value)
        if scalar is None:
            return
        if wandb.run is not None:
            wandb.log({key: float(scalar)}, step=int(step))

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.algo is None:
            return
        if epoch_num % self.eval_every != 0:
            return

        unwrapped_env = self._get_env()
        if unwrapped_env is None:
            raise RuntimeError("Could not find unwrapped env from vec_env.")

        saved = {}
        for k in ("frame", "total_frames", "agent_steps", "env_steps", "game_steps"):
            if hasattr(self.algo, k):
                saved[k] = getattr(self.algo, k)
        if hasattr(self.algo, "rnn_states"):
            saved["rnn_states"] = copy.deepcopy(self.algo.rnn_states)
        if hasattr(self.algo, "obs"):
            saved["obs"] = copy.deepcopy(self.algo.obs)
        saved_first_reset = getattr(unwrapped_env, "_first_reset", None)
        saved_train_extras = {
            k: v for k, v in dict(getattr(unwrapped_env, "extras", {})).items() if isinstance(k, str) and k.startswith("train/")
        }
        saved_env_state = {}
        for name in (
            "ep_return",
            "dr_context",
            "mapping",
            "reset_buf",
            "episode_length_buf",
            "progress_buf",
        ):
            if hasattr(unwrapped_env, name):
                saved_env_state[name] = _clone_if_tensor(getattr(unwrapped_env, name))

        eval_episode_returns = []

        try:
            unwrapped_env.set_uniform_eval(True)
            self.algo.set_eval()

            T = int(getattr(unwrapped_env, "max_episode_length", 200))
            num_actors = int(getattr(self.algo, "num_actors", unwrapped_env.num_envs))

            for _ in range(self.eval_episodes):
                obs = self.algo.env_reset()
                if hasattr(self.algo, "rnn_states"):
                    self.algo.rnn_states = _zeros_like_rnn(saved.get("rnn_states", None), num_actors, self.algo.device)

                ep_ret = None
                for _ in range(T):
                    obs_tensor = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs
                    processed_obs = self.algo._preproc_obs(obs_tensor)

                    input_dict = {
                        "is_train": False,
                        "prev_actions": None,
                        "obs": processed_obs,
                        "rnn_states": getattr(self.algo, "rnn_states", None),
                    }
                    with torch.no_grad():
                        res = self.algo.model(input_dict)

                    actions = res["mus"] if self.deterministic and "mus" in res else res["actions"]
                    obs, rew, dones, infos = self.algo.env_step(actions)
                    if "rnn_states" in res and hasattr(self.algo, "rnn_states"):
                        self.algo.rnn_states = res["rnn_states"]

                    rew = rew.view(-1)
                    if ep_ret is None:
                        ep_ret = torch.zeros_like(rew)
                    ep_ret += rew

                eval_episode_returns.append(ep_ret.mean())

            if eval_episode_returns:
                ret = torch.stack(eval_episode_returns).mean()
                wandb.log({"eval/uniform_return_mean" : ret.item()}, step=int(frame))

        finally:
            unwrapped_env.set_uniform_eval(False)

            obs = self.algo.env_reset()
            if hasattr(self.algo, "obs"):
                self.algo.obs = obs
            if saved_first_reset is not None:
                unwrapped_env._first_reset = saved_first_reset
            if hasattr(unwrapped_env, "extras"):
                # restore all train metrics exactly as they were before eval
                for k in list(unwrapped_env.extras.keys()):
                    if isinstance(k, str) and k.startswith("train/"):
                        unwrapped_env.extras.pop(k, None)
                for k, v in saved_train_extras.items():
                    unwrapped_env.extras[k] = v
            for name, value in saved_env_state.items():
                try:
                    if torch.is_tensor(value) and hasattr(getattr(unwrapped_env, name), "copy_"):
                        getattr(unwrapped_env, name).copy_(value)
                    else:
                        setattr(unwrapped_env, name, value)
                except Exception:
                    pass
            self.algo.set_train()
            for k, v in saved.items():
                try:
                    setattr(self.algo, k, v)
                except Exception:
                    pass
        for k, v in dict(getattr(unwrapped_env, "extras", {})).items():
            if "eval" in k:
                self._log_scalar(k, v, int(frame))
