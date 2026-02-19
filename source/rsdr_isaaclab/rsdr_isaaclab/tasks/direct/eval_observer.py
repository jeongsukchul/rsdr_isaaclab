import numpy as np
import torch
from rl_games.common.algo_observer import IsaacAlgoObserver

class UniformEvalObserver(IsaacAlgoObserver):
    def __init__(self, eval_every=1, eval_episodes=3, deterministic=True, tb_prefix=""):
        self.eval_every = int(eval_every)
        self.eval_episodes = int(eval_episodes)
        self.deterministic = bool(deterministic)
        self.tb_prefix = tb_prefix
        self.algo = None

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.algo is None or self.algo.writer is None:
            return
        if epoch_num % self.eval_every != 0:
            return

        v = self.algo.vec_env
        base = getattr(v, "env", None) or getattr(v, "_env", None) or v
        factory_env = getattr(base, "unwrapped", None)
        if factory_env is None:
            raise RuntimeError(f"Could not find unwrapped env. vec_env={type(v)}, base={type(base)}")

        factory_env.set_uniform_eval(True)
        self.algo.set_eval()
        metrics_accum = {}
        completed = 0
        rewards = None

        try:
            print("first reset with uniform dist!")
            obs = self.algo.env_reset()
            count = 0 
            act_dim = self.algo.actions_num  # usually exists in rl-games

            prev_actions = torch.zeros((self.algo.num_actors, act_dim), device=self.algo.device)
            T = factory_env.max_episode_length
            for ep in range(self.eval_episodes):
                obs = self.algo.env_reset()
                for t in range(T):
                    # res = self.algo.get_action_values(obs)
                    # actions = res["actions"]
                    processed_obs = self.algo._preproc_obs(obs['obs'])

                    input_dict = {
                        "is_train": False,              # <-- key change
                        "prev_actions": None,
                        "obs": processed_obs,
                        "rnn_states": self.algo.rnn_states,
                    }

                    with torch.no_grad():
                        res = self.algo.model(input_dict)

                    actions = res["actions"]           # stochastic, train-like
                    # prev_actions = actions.clone()
                    ep_before = factory_env.ep_return.clone()
                    obs, rew, dones, infos = self.algo.env_step(actions)
                    if "rnn_states" in res:
                        self.algo.rnn_states = res["rnn_states"]
                    rew = rew.view(-1)
                    if rewards is None:
                        rewards = torch.zeros_like(rew)
                    rewards += rew
        finally:
            factory_env.set_uniform_eval(False)
            factory_env._first_reset = True  # ensure env resets properly after eval
            obs = self.algo.env_reset()
            # # Many rl-games algos store current obs internally; keep it consistent
            if hasattr(self.algo, "obs"):
                self.algo.obs = obs
            self.algo.set_train()
        for k, v in dict(getattr(factory_env, "extras", {})).items():
            if "eval" in k:
                # print(f"Evalation: Logging {k}={v} from env infos.")
                self.writer.add_scalar(f"{k}/frame", v, frame)