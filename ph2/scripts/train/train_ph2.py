#!/usr/bin/env python
"""
PH2 training entry point.

Usage:
    python train_ph2.py \
        --env_name Overcooked \
        --algorithm_name ph2 \
        --experiment_name ph2_s1 \
        --layout_name cramped_room \
        --seed 1 \
        --n_rollout_threads 100 \
        --num_env_steps 10000000 \
        --use_centralized_V \
        --ph2_fixed_ind_prob 0.5 \
        --ph2_use_partner_pred \
        --use_wandb
"""
import os
import pprint
import socket
import sys
from pathlib import Path

import setproctitle
import torch
import wandb
from loguru import logger

# ---- make sure ZSC-EVAL and ph2 are on path ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_PH2_ROOT = os.path.join(_HERE, "..", "..")
_ZSC_ROOT = os.path.join(_PH2_ROOT, "..", "ZSC-EVAL")
for _p in [_PH2_ROOT, _ZSC_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args, OLD_LAYOUTS
from zsceval.utils.train_util import setup_seed

from ph2.ph2_config import get_ph2_args
from ph2.envs.overcooked.overcooked_adapter import make_train_env, make_eval_env
from ph2.algorithms.ph2.ph2_trainer import PH2Trainer
from ph2.runner.shared.overcooked_runner import PH2OvercookedRunner


os.environ.setdefault("WANDB_DIR",        os.getcwd() + "/wandb/")
os.environ.setdefault("WANDB_CACHE_DIR",  os.getcwd() + "/wandb/.cache/")
os.environ.setdefault("WANDB_CONFIG_DIR", os.getcwd() + "/wandb/.config/")


def parse_args(args):
    parser = get_config()
    parser = get_overcooked_args(parser)
    parser = get_ph2_args(parser)
    parser.allow_abbrev = False

    # zsceval.config defines algorithm_name choices that don't include "ph2".
    # Extend the existing choice list before argparse validation runs.
    for action in parser._actions:
        if getattr(action, "dest", None) == "algorithm_name" and action.choices is not None:
            if "ph2" not in action.choices:
                action.choices = list(action.choices) + ["ph2"]
            break

    # ph2 doesn't go through make_trainer_policy_cls — skip algorithm_name restriction
    all_args = parser.parse_args(args)

    # override algorithm_name to avoid ZSC-EVAL's assert
    all_args.algorithm_name = "ph2"

    # auto-set old_dynamics based on layout (same rule as ZSC-EVAL)
    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False

    # overcooked_version must be "new" (ph2 uses overcooked_new)
    all_args.overcooked_version = "new"
    if all_args.env_name == "Overcooked" and all_args.num_agents != 2:
        logger.warning(
            f"Overcooked expects 2 agents, but num_agents={all_args.num_agents}. "
            "Forcing num_agents=2 for PH2 compatibility."
        )
        all_args.num_agents = 2
    if all_args.env_name == "Overcooked" and all_args.cnn_layers_params is None:
        # Prevent default 5x5/no-padding CNN from crashing on small layouts (e.g., 5x4).
        all_args.cnn_layers_params = "32,3,1,1 64,3,1,1 32,3,1,1"

    # PH2 always runs with recurrent policy.
    # In this codebase, "--use_recurrent_policy" is defined as store_false,
    # so we force the intended values here to avoid accidental misconfiguration.
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False

    # parser compatibility: support legacy scalar `--entropy_coef`
    # by converting to ZSC-EVAL schedule format.
    entropy_coef_scalar = getattr(all_args, "entropy_coef", None)
    if entropy_coef_scalar is not None:
        all_args.entropy_coefs = [entropy_coef_scalar, entropy_coef_scalar]
        all_args.entropy_coef_horizons = [0, int(all_args.num_env_steps)]

    # Compatibility defaults with ZSC-EVAL runner/env scripts.
    compat_defaults = {
        "use_phi": False,
        "use_task_v_out": False,
        "stage": 1,
        "store_traj": False,
        "agent0_policy_name": "",
        "agent1_policy_name": "",
    }
    for k, v in compat_defaults.items():
        if not hasattr(all_args, k):
            setattr(all_args, k, v)

    return all_args


def get_ph2_base_run_dir() -> str:
    """
    PH2 stores results under zsc-basecamp/results/ph2 by default.
    Override with PH2_BASE_RUN_DIR if needed.
    """
    override = os.environ.get("PH2_BASE_RUN_DIR", "").strip()
    if override:
        return override
    repo_root = os.path.abspath(os.path.join(_PH2_ROOT, ".."))
    return os.path.join(repo_root, "results", "ph2")


def allocate_next_run_dir(run_root: Path) -> Path:
    """
    Allocate next run directory as run1, run2, ...
    Safe under concurrent launches by relying on atomic mkdir.
    """
    os.makedirs(str(run_root), exist_ok=True)
    idx = 1
    while True:
        candidate = run_root / f"run{idx}"
        try:
            os.makedirs(str(candidate), exist_ok=False)
            return candidate
        except FileExistsError:
            idx += 1


def main(args):
    all_args = parse_args(args)

    # ---- device ----
    if all_args.cuda and torch.cuda.is_available():
        logger.info("Using GPU")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        logger.info("Using CPU")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # ---- run dir ----
    base_run_dir = Path(get_ph2_base_run_dir())
    run_root = (
        base_run_dir
        / all_args.env_name
        / all_args.layout_name
        / all_args.experiment_name
    )
    run_dir = allocate_next_run_dir(run_root)

    all_args.run_dir = run_dir

    # ---- setproctitle ----
    setproctitle.setproctitle(
        f"ph2-{all_args.env_name}_{all_args.layout_name}-{all_args.experiment_name}"
        f"@{all_args.user_name}"
    )

    # ---- seed ----
    setup_seed(all_args.seed)

    # ---- envs ----
    logger.info(pprint.pformat(vars(all_args), compact=True))
    envs      = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
    num_agents = all_args.num_agents

    share_obs_space = (
        envs.share_observation_space[0]
        if all_args.use_centralized_V
        else envs.observation_space[0]
    )

    logger.info(
        f"Obs: {envs.observation_space[0].shape}  "
        f"Share-obs: {share_obs_space.shape}  "
        f"Act: {envs.action_space[0]}"
    )

    # ---- trainer ----
    trainer = PH2Trainer(
        all_args,
        envs.observation_space[0],
        share_obs_space,
        envs.action_space[0],
        device,
    )

    # ---- config dict ----
    config = {
        "all_args":  all_args,
        "envs":      envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device":    device,
        "run_dir":   run_dir,
        "trainer":   trainer,
    }

    # ---- runner ----
    runner = PH2OvercookedRunner(config)
    runner.run()

    # ---- cleanup ----
    envs.close()
    if eval_envs is not None and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        wandb.finish(quiet=True)
    elif hasattr(runner, "writter"):
        runner.writter.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    main(sys.argv[1:])
