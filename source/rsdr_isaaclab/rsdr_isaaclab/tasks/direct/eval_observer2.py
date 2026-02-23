import copy
import torch
from rl_games.common.algo_observer import IsaacAlgoObserver

def _zeros_like_rnn(template, new_batch: int, device):
    """template can be tensor or (list/tuple/dict) of tensors."""
    if template is None:
        return None
    if torch.is_tensor(template):
        shape = list(template.shape)
        shape[0] = new_batch
        return torch.zeros(shape, dtype=template.dtype, device=device)
    if isinstance(template, (list, tuple)):
        out = [_zeros_like_rnn(x, new_batch, device) for x in template]
        return type(template)(out)
    if isinstance(template, dict):
        return {k: _zeros_like_rnn(v, new_batch, device) for k, v in template.items()}
    raise TypeError(f"Unsupported rnn state type: {type(template)}")

def _mask_rnn_state(rnn_state, alive_mask_1d: torch.Tensor):
    """alive_mask_1d: shape (B,) with 1 for alive, 0 for done."""
    if rnn_state is None:
        return None
    if torch.is_tensor(rnn_state):
        # broadcast mask onto remaining dims
        view = [alive_mask_1d.shape[0]] + [1] * (rnn_state.ndim - 1)
        return rnn_state * alive_mask_1d.view(*view)
    if isinstance(rnn_state, (list, tuple)):
        out = [_mask_rnn_state(x, alive_mask_1d) for x in rnn_state]
        return type(rnn_state)(out)
    if isinstance(rnn_state, dict):
        return {k: _mask_rnn_state(v, alive_mask_1d) for k, v in rnn_state.items()}
    raise TypeError(f"Unsupported rnn state type: {type(rnn_state)}")

class SeparateEnvEvalObserver(IsaacAlgoObserver):
    def __init__(self, eval_env, eval_every=5, eval_episodes=1, deterministic=True, tb_prefix="eval/"):
        self.eval_env = eval_env
        self.eval_every = int(eval_every)
        self.eval_episodes = int(eval_episodes)
        self.deterministic = bool(deterministic)
        self.tb_prefix = tb_prefix
        self.algo = None

    def after_init(self, algo):
        self.algo = algo

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.algo is None or self.algo.writer is None or self.eval_env is None:
            return
        if epoch_num % self.eval_every != 0:
            return

        model = self.algo.model
        was_training = model.training
        model.eval()

        # eval batch size = eval env num_actors
        eval_num_actors = self.eval_env.unwrapped.num_envs if hasattr(self.eval_env, "unwrapped") else None
        if eval_num_actors is None:
            eval_num_actors = self.algo.num_actors  # fallback

        # create separate rnn state template (do NOT touch self.algo.rnn_states)
        train_rnn_template = getattr(self.algo, "rnn_states", None)


        # reset eval env (wrapper returns dict like training wrapper)
        obs = self.eval_env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs

        # fixed rollout length (matches your current style)
        T = getattr(self.eval_env.unwrapped, "max_episode_length", 200)

        total_return = None

        with torch.no_grad():
            for ep in range(self.eval_episodes):
                eval_rnn_states = _zeros_like_rnn(train_rnn_template, eval_num_actors, self.algo.device)
                act_dim = getattr(self.algo, "actions_num", None)
                prev_actions = torch.zeros((eval_num_actors, act_dim), device=self.algo.device)
                obs = self.eval_env.reset()
                if isinstance(obs, tuple):
                    obs, _ = obs

                ep_ret = None

                for t in range(T):
                    # unwrap obs to tensor expected by rl-games preprocessing
                    obs_tensor = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs
                    processed_obs = self.algo._preproc_obs(obs_tensor)

                    input_dict = {
                        "is_train": False,
                        "prev_actions": prev_actions,   # good for RNN policies that expect it
                        "obs": processed_obs,
                        "rnn_states": eval_rnn_states,
                    }

                    res = model(input_dict)

                    actions = res["actions"]
                    if self.deterministic and "mus" in res:
                        actions = res["mus"]

                    # step eval env directly (does not mutate training counters)
                    obs, rew, dones, infos = self.eval_env.step(actions)
                    if isinstance(obs, tuple):
                        obs, _ = obs

                    # update eval rnn state if returned
                    if "rnn_states" in res:
                        eval_rnn_states = res["rnn_states"]

                    # mask out done envs (reset hidden state for those env indices)
                    if dones is not None:
                        d = dones.view(-1).to(device=self.algo.device)
                        alive = (1.0 - d.float())
                        eval_rnn_states = _mask_rnn_state(eval_rnn_states, alive)
                        if prev_actions is not None:
                            prev_actions = prev_actions * alive.unsqueeze(1)

                    rew = rew.view(-1)
                    if ep_ret is None:
                        ep_ret = torch.zeros_like(rew)
                    ep_ret += rew

                total_return = ep_ret if total_return is None else (total_return + ep_ret)

        mean_return = (total_return / float(self.eval_episodes)).mean().item()
        self.algo.writer.add_scalar(f"{self.tb_prefix}return_mean", mean_return, frame)

        model.train(was_training)