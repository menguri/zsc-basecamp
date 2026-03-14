"""
PH2 Base Runner — manages wandb / tensorboard init, save-dir, policy_config.pkl,
and provides log helpers.

policy_config.pkl format is identical to ZSC-EVAL to maintain eval compatibility:
    (all_args, obs_space, share_obs_space, act_space)
"""
import os
import pickle
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import wandb
from loguru import logger

# ZSC-EVAL tensorboard writer
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


WANDB_KEY_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "wandb_info", "wandb_api_key"
)


def _load_wandb_key() -> str | None:
    try:
        with open(WANDB_KEY_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def _t2n(x):
    return x.detach().cpu().numpy()


class PH2BaseRunner:
    """
    Thin base class for PH2 runners.

    Subclasses must implement:
        run(), warmup(), collect_spec_rollout(), collect_ind_rollout(),
        eval()
    """

    def __init__(self, config: dict):
        self.all_args    = config["all_args"]
        self.envs        = config["envs"]
        self.eval_envs   = config.get("eval_envs", None)
        self.device      = config["device"]
        self.num_agents  = config["num_agents"]
        self.trainer     = config["trainer"]   # PH2Trainer instance

        args = self.all_args
        self.env_name        = args.env_name
        self.algorithm_name  = args.algorithm_name
        self.experiment_name = args.experiment_name
        self.use_centralized_V = args.use_centralized_V
        self.num_env_steps   = args.num_env_steps
        self.episode_length  = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.n_eval_rollout_threads = args.n_eval_rollout_threads
        self.hidden_size     = args.hidden_size
        self.recurrent_N     = args.recurrent_N
        self.use_linear_lr_decay = args.use_linear_lr_decay
        self.use_wandb       = args.use_wandb
        self.use_eval        = args.use_eval
        self.use_render      = getattr(args, "use_render", False)
        self.save_interval   = args.save_interval
        self.eval_interval   = args.eval_interval
        self.log_interval    = args.log_interval
        self.model_dir       = args.model_dir

        # ---- directories & logging ----
        run_dir: Path = config["run_dir"]
        if self.use_wandb:
            # init wandb
            api_key = _load_wandb_key()
            if api_key:
                os.environ["WANDB_API_KEY"] = api_key
            wandb.init(
                config=vars(args),
                project=getattr(args, "wandb_project", "zsc-basecamp"),
                entity=getattr(args, "wandb_entity", "m-personal-experiment"),
                name=f"ph2_{args.experiment_name}_seed{args.seed}",
                group=getattr(args, "layout_name", "unknown"),
                dir=str(run_dir),
                job_type="training",
                reinit=True,
                tags=getattr(args, "wandb_tags", []),
            )
            self.run_dir = str(run_dir)
        else:
            self.run_dir  = str(run_dir)
            log_dir = str(run_dir / "logs")
            os.makedirs(log_dir, exist_ok=True)
            if SummaryWriter is not None:
                self.writter = SummaryWriter(log_dir)

        self.save_dir = str(run_dir / "models")
        os.makedirs(self.save_dir, exist_ok=True)

        # ---- policy_config.pkl (ZSC-EVAL eval compat) ----
        share_obs_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )
        self.policy_config = (
            args,
            self.envs.observation_space[0],
            share_obs_space,
            self.envs.action_space[0],
        )
        pkl_path = os.path.join(self.run_dir, "policy_config.pkl")
        pickle.dump(self.policy_config, open(pkl_path, "wb"))
        logger.info(f"Policy config saved to {pkl_path}")

        # ---- restore if requested ----
        if self.model_dir is not None:
            self.trainer.restore(self.model_dir)

    # ------------------------------------------------------------------
    # abstract interface
    # ------------------------------------------------------------------
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # logging helpers (mirrors ZSC-EVAL Runner API)
    # ------------------------------------------------------------------
    def log_train(self, train_infos: dict, total_num_steps: int):
        for k, v in train_infos.items():
            if isinstance(v, Iterable):
                if len(v) == 0:
                    continue
                v = np.mean(v)
            if self.use_wandb:
                wandb.log({f"train/{k}": v}, step=total_num_steps)
            elif SummaryWriter is not None:
                self.writter.add_scalars(f"train/{k}", {k: v}, total_num_steps)

    def log_env(self, env_infos: dict, total_num_steps: int):
        for k, v in env_infos.items():
            if isinstance(v, Iterable):
                if len(v) == 0:
                    continue
                v = np.mean(v)
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            elif SummaryWriter is not None:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_ph2(self, spec_info: dict, ind_info: dict, extra: dict, total_num_steps: int):
        """Log PH2-specific metrics: spec/*, ind/*, and any extra scalars."""
        merged = {**spec_info, **ind_info, **extra}
        for k, v in merged.items():
            if isinstance(v, Iterable):
                if len(v) == 0:
                    continue
                v = np.mean(v)
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            elif SummaryWriter is not None:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------
    def save(self, step: int = 0):
        self.trainer.save(self.save_dir, step=step)
