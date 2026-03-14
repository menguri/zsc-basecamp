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
from zsceval.utils.train_util import get_base_run_dir, setup_seed

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

    # entropy_coef scalar convenience: if user passes single --entropy_coef
    # convert to ZSC-EVAL's list format
    if hasattr(all_args, "entropy_coef") and not hasattr(all_args, "_entropy_coef_set"):
        all_args.entropy_coefs = [all_args.entropy_coef, all_args.entropy_coef]
        all_args.entropy_coef_horizons = [0, int(all_args.num_env_steps)]

    return all_args


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
    base_run_dir = Path(get_base_run_dir())
    run_dir = (
        base_run_dir
        / all_args.env_name
        / all_args.layout_name
        / "ph2"
        / all_args.experiment_name
    )
    os.makedirs(str(run_dir), exist_ok=True)

    # wandb sets run_dir internally; for non-wandb add runX suffix
    if not all_args.use_wandb:
        exst = [
            int(str(f.name).split("run")[1])
            for f in run_dir.iterdir()
            if str(f.name).startswith("run")
        ] if run_dir.exists() else []
        curr_run = f"run{max(exst) + 1}" if exst else "run1"
        run_dir = run_dir / curr_run
        os.makedirs(str(run_dir), exist_ok=True)

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
