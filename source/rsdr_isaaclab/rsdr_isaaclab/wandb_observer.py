from rl_games.algos_torch import torch_ext
import torch
import numpy as np
import wandb
from rsdr_isaaclab.tasks.direct.samplers.sampler import (
    render_param_distribution_image,
    render_pairwise_marginal_image,
)


class AlgoObserver:
    def __init__(self):
        pass

    def before_init(self, base_name, config, experiment_name):
        pass

    def after_init(self, algo):
        pass

    def process_infos(self, infos, done_indices):
        pass

    def after_steps(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        pass


class DefaultAlgoObserver(AlgoObserver):
    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.game_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)  
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not infos:
            return

        done_indices = done_indices.cpu().numpy()

        if not isinstance(infos, dict) and len(infos) > 0 and isinstance(infos[0], dict):
            for ind in done_indices:
                ind = ind.item()
                if len(infos) <= ind//self.algo.num_agents:
                    continue
                info = infos[ind//self.algo.num_agents]
                game_res = None
                if 'battle_won' in info:
                    game_res = info['battle_won']
                if 'scores' in info:
                    game_res = info['scores']

                if game_res is not None:
                    self.game_scores.update(torch.from_numpy(np.asarray([game_res])).to(self.algo.ppo_device))

        elif isinstance(infos, dict):
            if 'lives' in infos:
                # envpool
                done_indices = np.argwhere(infos['lives'] == 0).squeeze(1)

            for ind in done_indices:
                ind = ind.item()
                game_res = None
                if 'battle_won' in infos:
                    game_res = infos['battle_won']
                if 'scores' in infos:
                    game_res = infos['scores']
                if game_res is not None and len(game_res) > ind//self.algo.num_agents:
                    self.game_scores.update(torch.from_numpy(np.asarray([game_res[ind//self.algo.num_agents]])).to(self.algo.ppo_device))

    def after_clear_stats(self):
        self.game_scores.clear()

    # def after_print_stats(self, frame, epoch_num, total_time):
    #     if self.game_scores.current_size > 0 and self.writer is not None:
    #         mean_scores = self.game_scores.get_mean()
    #         self.writer.add_scalar('scores/mean', mean_scores, frame)
    #         self.writer.add_scalar('scores/iter', mean_scores, epoch_num)
    #         self.writer.add_scalar('scores/time', mean_scores, total_time)


class IsaacWandbAlgoObserver(AlgoObserver):
    """Log statistics from the environment along with the algorithm running stats."""

    def __init__(self, eval_every=1, eval_episodes=3, deterministic=True, tb_prefix="", log_step="epoch"):
        self.eval_every = int(eval_every)
        self.eval_episodes = int(eval_episodes)
        self.deterministic = bool(deterministic)
        self.tb_prefix = tb_prefix
        self.log_step = str(log_step)
        self.algo = None
        self.grid_n = 64
        self.vis_num_samples = 4096
        self._logged_initial_sampler_viz = False

        
    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.train_info_cache = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict):
            classname = self.__class__.__name__
            raise ValueError(f"{classname} expected 'infos' as dict. Received: {type(infos)}")
        # store episode information
        if "episode" in infos:
            self.ep_infos.append(infos["episode"])
        # log other variables directly
        if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, (float, int)):
                    self.direct_info[k] = float(v)
                elif isinstance(v, torch.Tensor) and len(v.shape) == 0:
                    self.direct_info[k] = float(v.item())

    def after_clear_stats(self):
        # clear stored buffers
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.algo is None or self.algo.writer is None:
            return
        self._maybe_log_initial_sampler_2d_viz(step=int(frame))
        if epoch_num % self.eval_every != 0:
            return
        # log scalars from the episode
        # if self.ep_infos:
        #     for key in self.ep_infos[0]:
        #         info_tensor = torch.tensor([], device=self.algo.device)
        #         for ep_info in self.ep_infos:
        #             # handle scalar and zero dimensional tensor infos
        #             if not isinstance(ep_info[key], torch.Tensor):
        #                 ep_info[key] = torch.Tensor([ep_info[key]])
        #             if len(ep_info[key].shape) == 0:
        #                 ep_info[key] = ep_info[key].unsqueeze(0)
        #             info_tensor = torch.cat((info_tensor, ep_info[key].to(self.algo.device)))
        #         value = torch.mean(info_tensor)
        #         self.writer.add_scalar("Episode/" + key, value, epoch_num)
        #     self.ep_infos.clear()
        # log scalars from env information collected during stepping
        train_metrics = dict(self.direct_info)
        # also pull latest env.extras so metrics emitted on reset are not missed
        train_metrics.update(self._collect_env_extra_scalars())
        for k, v in train_metrics.items():
            if k.startswith("train/"):
                self.train_info_cache[k] = v
        train_metrics.update(self.train_info_cache)
        payload = {k: v for k, v in train_metrics.items() if not k.startswith("eval/")}
        payload["train/_metric_count"] = float(len(payload))
        payload.update({"train/frame": float(frame), "train/epoch": float(epoch_num)})
        wandb.log(payload, step=int(frame))
        self.direct_info.clear()

        self._log_sampler_2d_viz(step=int(frame))
        
        # log mean reward/score from the env
        # if self.mean_scores.current_size > 0:
        #     mean_scores = self.mean_scores.get_mean()
        #     self.writer.add_scalar("scores/mean", mean_scores, frame)
        #     self.writer.add_scalar("scores/iter", mean_scores, epoch_num)
        #     self.writer.add_scalar("scores/time", mean_scores, total_time)

    def _get_factory_env(self):
        v = self.algo.vec_env
        base = getattr(v, "env", None) or getattr(v, "_env", None) or v
        return getattr(base, "unwrapped", None)

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, np.generic):
            return float(value.item())
        if torch.is_tensor(value) and value.numel() == 1:
            return float(value.item())
        return None

    def _collect_scalar_metrics(self, data, prefix=""):
        out = {}
        if isinstance(data, dict):
            for k, v in data.items():
                key = f"{prefix}/{k}" if prefix else str(k)
                if isinstance(v, dict):
                    out.update(self._collect_scalar_metrics(v, key))
                else:
                    scalar = self._to_scalar(v)
                    if scalar is not None:
                        out[key] = scalar
        return out

    def _collect_env_extra_scalars(self):
        env = self._get_factory_env()
        if env is None:
            return {}
        extras = dict(getattr(env, "extras", {}))
        return self._collect_scalar_metrics(extras)

    def _log_sampler_2d_viz(self, step: int):
        if wandb.run is None:
            return
        env = self._get_factory_env()
        if env is None:
            return
        sampler = getattr(env, "sampler", None)
        if sampler is None:
            return
        extras = dict(getattr(env, "extras", {}))
        contexts = extras.get("dr_samples", None)
        env_sampled_contexts = None
        if contexts is not None and torch.is_tensor(contexts) and contexts.ndim == 2:
            n_env = min(self.vis_num_samples, contexts.shape[0])
            env_sampled_contexts = contexts[:n_env].detach().to(device=sampler.device, dtype=torch.float32)

        sampler_sampled_contexts = sampler.sample_contexts(self.vis_num_samples).detach().to(
            device=sampler.device, dtype=torch.float32
        )

        pair_img_sampler = render_pairwise_marginal_image(
            sampler=sampler,
            context_source=sampler_sampled_contexts,
            samples=sampler_sampled_contexts,
            eval_samples=None,
            marginal_mc_samples=64,
            num_grid=80,
        )
        if pair_img_sampler is not None:
            wandb.log(
                {f"viz/{sampler.name}/pairwise/sampler": wandb.Image(pair_img_sampler)},
                step=step,
            )

        self._log_paramcfg_visualizations(
            step=step,
            sampler=sampler,
            env_sampled_contexts=env_sampled_contexts,
            sampler_sampled_contexts=sampler_sampled_contexts,
        )

    def _log_paramcfg_visualizations(
        self,
        step: int,
        sampler,
        env_sampled_contexts: torch.Tensor | None,
        sampler_sampled_contexts: torch.Tensor,
    ):
        params = [p for p in sampler.cfg.params if getattr(p, "visualize", False)]
        if not params:
            return

        for p in params:
            sampler_img = render_param_distribution_image(
                sampler=sampler,
                sampled_contexts=sampler_sampled_contexts,
                param_cfg=p,
                prefix=f"{sampler.name} sampler_draws(model.sample)",
                bins=40,
            )
            if sampler_img is not None:
                wandb.log(
                    {f"viz/{sampler.name}/param/{p.name}_sampler_draws": wandb.Image(sampler_img)},
                    step=step,
                )

            if env_sampled_contexts is None:
                continue

            env_img = render_param_distribution_image(
                sampler=sampler,
                sampled_contexts=env_sampled_contexts,
                param_cfg=p,
                prefix=f"{sampler.name} env.dr_samples",
                bins=40,
            )
            if env_img is not None:
                wandb.log(
                    {f"viz/{sampler.name}/param/{p.name}_env_dr_samples": wandb.Image(env_img)},
                    step=step,
                )

    def _maybe_log_initial_sampler_2d_viz(self, step: int):
        if self._logged_initial_sampler_viz:
            return
        self._log_sampler_2d_viz(step=step)
        self._logged_initial_sampler_viz = True
