"""
Thin wrapper around ZSC-EVAL's overcooked_new environment.
Ensures share_obs = concat(obs_i, obs_j) rule is applied and
exposes a minimal interface compatible with the PH2 runner.
"""
import sys
import os

# Make sure ZSC-EVAL is importable
_ZSC_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ZSC-EVAL")
if _ZSC_ROOT not in sys.path:
    sys.path.insert(0, _ZSC_ROOT)

from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as _OvercookedNew
from zsceval.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocDummyBatchVecEnv
from zsceval.utils.train_util import setup_seed


def make_train_env(all_args, run_dir):
    """Create vectorised training environment."""
    def _safe_seed(env, seed):
        try:
            env.seed(seed)
        except (AttributeError, TypeError):
            # gym/gymnasium API difference: some Env bases do not expose seed().
            setup_seed(seed)

    def get_env_fn(rank):
        def init_env():
            env = _OvercookedNew(all_args, run_dir)
            _safe_seed(env, all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)],
            all_args.dummy_batch_size,
        )


def make_eval_env(all_args, run_dir):
    """Create vectorised evaluation environment."""
    def _safe_seed(env, seed):
        try:
            env.seed(seed)
        except (AttributeError, TypeError):
            setup_seed(seed)

    def get_env_fn(rank):
        def init_env():
            env = _OvercookedNew(all_args, run_dir, evaluation=True)
            _safe_seed(env, all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    n = all_args.n_eval_rollout_threads
    if n == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(n)],
            all_args.dummy_batch_size,
        )
