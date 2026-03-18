#!/usr/bin/env python
"""Behavioral Cloning (BC) training for ZSC-EVAL.

Uses GAMMA's pre-cached human demonstration data and ZSC-EVAL's overcooked_new
environment so the trained BC policy is eval-compatible with XP evaluation
(which also uses overcooked_new).

Human data source (pre-cached pickle):
    GAMMA/mapbt/envs/overcooked/dataset/formatted_human_trajectories/{layout}.pickle

Checkpoint output:
    <run_dir>/models/actor.pt       -- highest-priority file for _find_latest_actor_pt
    <run_dir>/policy_config.pkl     -- required by collect_runs in xp_eval.py
"""

import copy
import os
import pickle
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch
import torch.nn as nn
import wandb
from loguru import logger

from zsceval.config import get_config
from zsceval.envs.env_wrappers import ShareDummyVecEnv
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from zsceval.overcooked_config import get_overcooked_args
from zsceval.utils.train_util import get_base_run_dir, setup_seed

os.environ.setdefault("WANDB_DIR", os.getcwd() + "/wandb/")
os.environ.setdefault("WANDB_CACHE_DIR", os.getcwd() + "/wandb/.cache/")
os.environ.setdefault("WANDB_CONFIG_DIR", os.getcwd() + "/wandb/.config/")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(args, parser):
    parser = get_overcooked_args(parser)
    parser.add_argument(
        "--use_phi",
        default=False,
        action="store_true",
        help="Fix the main RL-policy agent index (unused for BC, required by Overcooked_new).",
    )
    parser.add_argument("--use_task_v_out", default=False, action="store_true")
    parser.add_argument(
        "--human_data_refresh",
        default=False,
        action="store_true",
        help="Re-extract human data from human_aware_rl.static (requires GAMMA env). "
             "By default uses the pre-cached pickle in GAMMA/mapbt/envs/overcooked/dataset/.",
    )
    parser.add_argument(
        "--human_data_split",
        type=str,
        default="2019-train",
        choices=["2019-train", "2019-test", "2020-train", "2024-train", "2024-test"],
    )
    parser.add_argument(
        "--human_layout_name",
        type=str,
        default=None,
        help="Layout name for human trajectories (default: same as --layout_name). "
             "Set to 'random3' when training counter_circuit_o_1order.",
    )
    parser.add_argument("--bc_validation_split", type=float, default=0.1)
    parser.add_argument("--bc_num_epochs", type=int, default=400)
    parser.add_argument("--bc_batch_size", type=int, default=128)

    all_args = parser.parse_known_args(args)[0]

    # Defaults
    if all_args.human_layout_name is None:
        all_args.human_layout_name = all_args.layout_name

    # BC is always non-recurrent (MLP actor, supervised training on fixed dataset)
    all_args.use_recurrent_policy = False
    all_args.use_naive_recurrent_policy = False
    all_args.algorithm_name = "bc"

    # Determine old_dynamics from layout (mirrors train_sp.py logic)
    from zsceval.overcooked_config import OLD_LAYOUTS
    all_args.old_dynamics = all_args.layout_name in OLD_LAYOUTS

    return all_args


# ---------------------------------------------------------------------------
# Human data
# ---------------------------------------------------------------------------

def _gamma_dataset_path(all_args) -> Path:
    """Return path to GAMMA's pre-cached formatted human trajectory pickle."""
    this_file = Path(__file__).resolve()
    # train_bc.py is at: ZSC-EVAL/zsceval/scripts/overcooked/train/
    # parents: [train/, overcooked/, scripts/, zsceval/, ZSC-EVAL/, zsc-basecamp/]
    zscbasecamp = this_file.parents[5]
    return (
        zscbasecamp
        / "GAMMA"
        / "mapbt"
        / "envs"
        / "overcooked"
        / "dataset"
        / "formatted_human_trajectories"
        / f"{all_args.human_layout_name}.pickle"
    )


def _raw_data_path(all_args) -> Path:
    """Path for the raw (non-featurized) trajectory pickle produced by extract_raw_trajectories.py."""
    this_file = Path(__file__).resolve()
    zscbasecamp = this_file.parents[5]
    return (
        zscbasecamp
        / "GAMMA" / "mapbt" / "envs" / "overcooked"
        / "dataset" / "formatted_human_trajectories"
        / f"{all_args.human_layout_name}_raw.pickle"
    )


def _extract_raw_via_gamma_python(all_args, raw_path: Path):
    """Subprocess call to GAMMA's python (which has human_aware_rl) to extract raw trajectories."""
    import subprocess
    this_file = Path(__file__).resolve()
    zscbasecamp = this_file.parents[5]
    gamma_python = zscbasecamp / ".zsc-gamma" / "bin" / "python3"
    extract_script = (
        zscbasecamp / "GAMMA" / "mapbt" / "scripts" / "train" / "extract_raw_trajectories.py"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(gamma_python), str(extract_script),
        "--layout_name", all_args.human_layout_name,
        "--human_data_split", all_args.human_data_split,
        "--output_path", str(raw_path),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _featurize_raw_with_zsceval(raw_path: Path, all_args, run_dir) -> tuple:
    """Load raw state dicts and featurize using ZSC-EVAL's overcooked_new env."""
    from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

    with open(raw_path, "rb") as f:
        raw_data = pickle.load(f)

    # Build a minimal env just for featurization
    env = Overcooked_new(all_args, run_dir, featurize_type=("bc", "bc"))
    base_env = env.base_env

    # Episodes come in pairs: (even_idx=player0_perspective, odd_idx=player1_perspective)
    # Same game state shared, but featurization and actions differ per player.
    inputs, targets = [], []
    for ep_idx, (ep_state_dicts, ep_actions) in enumerate(zip(raw_data["ep_states"], raw_data["ep_actions"])):
        player_idx = ep_idx % 2  # 0 for even episodes, 1 for odd episodes
        for state_dict, action_idx in zip(ep_state_dicts, ep_actions):
            state = OvercookedState.from_dict(state_dict)
            obs = base_env.featurize_state_mdp(state)[player_idx]
            inputs.append(obs.astype(np.float32))
            targets.append(int(action_idx))

    env.close()
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.int64)


def load_human_data(all_args, run_dir=None):
    """Return (inputs, targets) as float32/int64 numpy arrays.

    inputs : (N, obs_dim)  -- featurized state observations
    targets: (N,)          -- action indices

    When --human_data_refresh:
        1. Calls GAMMA python subprocess to extract raw trajectories (needs human_aware_rl)
        2. Re-featurizes using ZSC-EVAL's overcooked_new env (fixes GAMMA featurize bug)
    Otherwise:
        Loads pre-cached GAMMA pickle (featurized with GAMMA's env, minor soup feature mismatch).
    """
    if all_args.human_data_refresh:
        raw_path = _raw_data_path(all_args)
        logger.info(
            f"Refreshing human data: extracting raw trajectories via GAMMA python "
            f"(layout={all_args.human_layout_name}, split={all_args.human_data_split}) ..."
        )
        _extract_raw_via_gamma_python(all_args, raw_path)
        logger.info(f"Re-featurizing with ZSC-EVAL overcooked_new env ...")
        inputs, targets = _featurize_raw_with_zsceval(raw_path, all_args, run_dir)
    else:
        data_path = _gamma_dataset_path(all_args)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Pre-cached human data not found: {data_path}\n"
                f"Run with --human_data_refresh to extract from human_aware_rl, "
                f"or run GAMMA/sh_scripts/gamma/bc.sh first to generate the cache."
            )
        logger.warning(
            "Loading pre-cached GAMMA pickle (featurized with GAMMA's env). "
            "Minor soup feature mismatch possible. Use --human_data_refresh for exact match."
        )
        with open(data_path, "rb") as f:
            human_data = pickle.load(f)
        ep_states = human_data["ep_states"]
        ep_actions = human_data["ep_actions"]
        inputs = np.vstack(ep_states).astype(np.float32)
        targets = np.vstack(ep_actions).astype(np.int64).reshape(-1)

    logger.info(
        f"Human data loaded: {inputs.shape[0]} samples, "
        f"obs_dim={inputs.shape[1]}, unique_actions={np.unique(targets).tolist()}"
    )
    return inputs, targets


# ---------------------------------------------------------------------------
# BC Trainer
# ---------------------------------------------------------------------------

class BCTrainer:
    """Minimal BC trainer wrapping ZSC-EVAL's R_Actor.

    Uses R_Actor.evaluate_actions() to compute per-sample log-probabilities,
    so the loss is -mean(log_prob(target_action | obs)), i.e. cross-entropy.
    """

    def __init__(self, actor, optimizer, bc_batch_size: int,
                 recurrent_N: int, hidden_size: int, device):
        self.actor = actor
        self.optimizer = optimizer
        self.batch_size = bc_batch_size
        self.recurrent_N = recurrent_N
        self.hidden_size = hidden_size
        self.device = device

        self._train_inputs = None
        self._train_targets = None
        self._val_inputs = None
        self._val_targets = None

    def load_data(self, inputs: np.ndarray, targets: np.ndarray, validation_split: float = 0.1):
        N = len(inputs)
        perm = np.random.permutation(N)
        n_val = int(N * validation_split)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        self._train_inputs  = inputs[train_idx].copy()
        self._train_targets = targets[train_idx].copy()
        self._val_inputs    = inputs[val_idx].copy()
        self._val_targets   = targets[val_idx].copy()
        logger.info(
            f"BC data: train={len(self._train_inputs)}, val={len(self._val_inputs)} "
            f"(validation_split={validation_split})"
        )

    def _rnn_zeros(self, n: int) -> torch.Tensor:
        # R_Actor ignores rnn_states for non-recurrent policy; any shape is fine.
        return torch.zeros(n, self.recurrent_N, self.hidden_size, device=self.device)

    def fit_once(self) -> dict:
        inputs, targets = self._train_inputs, self._train_targets
        N = len(inputs)
        perm = np.random.permutation(N)
        correct, total, loss_sum = 0, 0, 0.0

        self.actor.train()
        for i in range(0, N, self.batch_size):
            idx = perm[i: i + self.batch_size]
            obs_t = torch.FloatTensor(inputs[idx]).to(self.device)
            act_t = torch.LongTensor(targets[idx]).to(self.device).unsqueeze(-1)  # (B, 1)
            masks = torch.ones(len(idx), 1, device=self.device)
            rnn_t = self._rnn_zeros(len(idx))

            # log_probs: (B, 1), CE loss = -mean(log_prob(target))
            log_probs, _, _ = self.actor.evaluate_actions(obs_t, rnn_t, act_t, masks)
            loss = -log_probs.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item() * len(idx)
            with torch.no_grad():
                actions, _, _ = self.actor(obs_t, rnn_t, masks, deterministic=True)
            correct += (actions.squeeze(-1) == act_t.squeeze(-1)).sum().item()
            total += len(idx)

        return {"training_loss": loss_sum / total, "training_accuracy": correct / total}

    @torch.no_grad()
    def validate(self) -> dict:
        inputs, targets = self._val_inputs, self._val_targets
        N = len(inputs)
        correct, total, loss_sum = 0, 0, 0.0

        self.actor.eval()
        for i in range(0, N, self.batch_size):
            obs_t = torch.FloatTensor(inputs[i: i + self.batch_size]).to(self.device)
            act_t = torch.LongTensor(targets[i: i + self.batch_size]).to(self.device).unsqueeze(-1)
            masks = torch.ones(len(obs_t), 1, device=self.device)
            rnn_t = self._rnn_zeros(len(obs_t))

            log_probs, _, _ = self.actor.evaluate_actions(obs_t, rnn_t, act_t, masks)
            loss_sum += (-log_probs.sum()).item()
            actions, _, _ = self.actor(obs_t, rnn_t, masks, deterministic=True)
            correct += (actions.squeeze(-1) == act_t.squeeze(-1)).sum().item()
            total += len(obs_t)

        return {"validation_loss": loss_sum / total, "validation_accuracy": correct / total}


# ---------------------------------------------------------------------------
# Reward evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_bc_reward(actor, all_args, run_dir, n_eval_episodes: int, device) -> dict:
    """Run the BC actor against itself and return mean sparse reward per episode."""
    n_envs = min(n_eval_episodes, 5)

    def _make_eval_env(rank):
        def init():
            env = Overcooked_new(all_args, run_dir, featurize_type=("bc", "bc"), evaluation=True)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init

    eval_envs = ShareDummyVecEnv([_make_eval_env(i) for i in range(n_envs)])
    num_agents = all_args.num_agents

    obs, _, _ = eval_envs.reset()
    obs = np.stack(obs)  # (n_envs, n_agents, obs_dim)

    rnn_states = np.zeros(
        (n_envs, num_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32
    )
    masks = np.ones((n_envs, num_agents, 1), dtype=np.float32)

    episode_sparse_r = []
    collected = 0

    actor.eval()
    while collected < n_eval_episodes:
        # Flatten (n_envs, n_agents, obs_dim) → (n_envs*n_agents, obs_dim)
        obs_flat = obs.reshape(n_envs * num_agents, -1)
        rnn_flat = rnn_states.reshape(n_envs * num_agents, all_args.recurrent_N, all_args.hidden_size)
        mask_flat = masks.reshape(n_envs * num_agents, 1)

        obs_t = torch.FloatTensor(obs_flat).to(device)
        rnn_t = torch.FloatTensor(rnn_flat).to(device)
        mask_t = torch.FloatTensor(mask_flat).to(device)

        actions, _, rnn_new = actor(obs_t, rnn_t, mask_t, deterministic=True)
        actions_np = actions.cpu().numpy().reshape(n_envs, num_agents, 1)
        rnn_states = rnn_new.cpu().numpy().reshape(n_envs, num_agents, all_args.recurrent_N, all_args.hidden_size)

        obs, _, _, dones, infos, _ = eval_envs.step(actions_np)
        obs = np.stack(obs)
        dones = np.array(dones)  # (n_envs, n_agents) or (n_envs,)

        # Episode done when all agents in an env are done
        env_done = dones.all(axis=-1) if dones.ndim > 1 else dones  # (n_envs,)
        for e, done in enumerate(env_done):
            if done and collected < n_eval_episodes:
                ep_info = infos[e]
                sparse_r = ep_info.get("episode", {}).get("ep_sparse_r", None)
                if sparse_r is not None:
                    episode_sparse_r.append(sparse_r)
                collected += 1
                # Reset rnn/masks for this env
                rnn_states[e] = 0.0
                masks[e] = 1.0

        masks[dones.all(axis=-1) if dones.ndim > 1 else dones] = 0.0

    eval_envs.close()

    if episode_sparse_r:
        return {
            "eval_sparse_r": float(np.mean(episode_sparse_r)),
            "eval_sparse_r_std": float(np.std(episode_sparse_r)),
        }
    return {"eval_sparse_r": 0.0, "eval_sparse_r_std": 0.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_env(all_args, run_dir):
    def init_env():
        env = Overcooked_new(all_args, run_dir, featurize_type=("bc", "bc"))
        env.seed(all_args.seed)
        return env
    return ShareDummyVecEnv([init_env])


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # --- device ---
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

    # --- run dir ---
    base_run_dir = Path(get_base_run_dir())
    run_dir = (
        base_run_dir
        / all_args.env_name
        / all_args.layout_name
        / "bc"
        / all_args.experiment_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- wandb / local run numbering ---
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name + "-new",
            entity=all_args.wandb_name,
            notes=socket.gethostname(),
            name=f"bc_{all_args.experiment_name}_seed{all_args.seed}",
            group=all_args.layout_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
            tags=getattr(all_args, "wandb_tags", []),
        )
    else:
        exst = (
            [
                int(d.name.split("run")[1])
                for d in run_dir.iterdir()
                if d.is_dir() and d.name.startswith("run")
            ]
            if run_dir.exists()
            else []
        )
        curr_run = f"run{max(exst) + 1}" if exst else "run1"
        run_dir = run_dir / curr_run
        run_dir.mkdir(parents=True, exist_ok=True)

    setproctitle.setproctitle(
        f"bc-{all_args.env_name}-{all_args.layout_name}"
        f"@{getattr(all_args, 'user_name', 'user')}"
    )
    setup_seed(all_args.seed)

    # --- env: get obs/act spaces for policy_config ---
    env = make_env(all_args, run_dir)
    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]
    env.close()
    logger.info(
        f"obs_space={obs_space.shape}  share_obs_space={share_obs_space.shape}  "
        f"act_space={act_space}"
    )

    # --- save policy_config.pkl (required by xp_eval.py collect_runs) ---
    with open(run_dir / "policy_config.pkl", "wb") as f:
        pickle.dump((all_args, obs_space, share_obs_space, act_space), f)
    logger.info(f"Saved policy_config.pkl → {run_dir}/policy_config.pkl")

    # --- build actor (no critic needed for BC) ---
    from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    args_for_policy = copy.deepcopy(all_args)
    setattr(args_for_policy, "eval_skip_critic", True)
    policy = R_MAPPOPolicy(
        args_for_policy, obs_space, share_obs_space, act_space, device=device
    )
    actor = policy.actor

    optimizer = torch.optim.Adam(
        actor.parameters(), lr=all_args.lr, eps=all_args.opti_eps
    )

    # --- human data ---
    inputs, targets = load_human_data(all_args, run_dir=run_dir)

    if inputs.shape[1] != obs_space.shape[0]:
        logger.warning(
            f"Human data obs_dim={inputs.shape[1]} != env obs_dim={obs_space.shape[0]}. "
            f"Featurize mismatch between GAMMA cache and overcooked_new? "
            f"Consider running with --human_data_refresh."
        )

    # --- trainer ---
    trainer = BCTrainer(
        actor=actor,
        optimizer=optimizer,
        bc_batch_size=all_args.bc_batch_size,
        recurrent_N=all_args.recurrent_N,
        hidden_size=all_args.hidden_size,
        device=device,
    )
    trainer.load_data(inputs, targets, validation_split=all_args.bc_validation_split)

    # --- checkpoint dir ---
    save_dir = run_dir / "models"
    save_dir.mkdir(exist_ok=True)

    eval_interval = all_args.eval_interval if all_args.use_eval else None
    n_eval_episodes = getattr(all_args, "eval_episodes", 10)

    best_sparse_r = -float("inf")
    best_val_acc  = -float("inf")
    last_eval_info = {}

    for epoch in range(1, all_args.bc_num_epochs + 1):
        train_info = trainer.fit_once()

        # --- validation (every epoch, matching GAMMA behavior) ---
        val_info = trainer.validate()
        if val_info["validation_accuracy"] > best_val_acc:
            best_val_acc = val_info["validation_accuracy"]
            torch.save(actor.state_dict(), save_dir / "actor_best_valid_acc.pt")

        # --- reward eval ---
        if eval_interval and (epoch % eval_interval == 0 or epoch == all_args.bc_num_epochs):
            last_eval_info = eval_bc_reward(actor, all_args, run_dir, n_eval_episodes, device)
            if last_eval_info["eval_sparse_r"] > best_sparse_r:
                best_sparse_r = last_eval_info["eval_sparse_r"]
                torch.save(actor.state_dict(), save_dir / "actor_best_sparse_r.pt")

        if epoch % all_args.log_interval == 0 or epoch == all_args.bc_num_epochs:
            reward_str = (
                f"  eval_sparse_r={last_eval_info['eval_sparse_r']:.3f}"
                if last_eval_info
                else ""
            )
            logger.info(
                f"Epoch {epoch:4d}/{all_args.bc_num_epochs}  "
                f"train_loss={train_info['training_loss']:.4f}  "
                f"train_acc={train_info['training_accuracy']:.3f}  "
                f"val_acc={val_info['validation_accuracy']:.3f}"
                + reward_str
            )
            if all_args.use_wandb:
                wandb.log({**train_info, **val_info, **last_eval_info, "epoch": epoch})

        if epoch % all_args.save_interval == 0:
            torch.save(actor.state_dict(), save_dir / f"actor_epoch{epoch}.pt")

    # Final checkpoint (fallback for eval if best_sparse_r was never updated)
    torch.save(actor.state_dict(), save_dir / "actor.pt")
    logger.info(
        f"Training done. best_sparse_r={best_sparse_r:.3f}  best_val_acc={best_val_acc:.3f}"
    )
    logger.info(f"Checkpoints: {save_dir}")

    if all_args.use_wandb:
        run.finish(quiet=True)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    main(sys.argv[1:])
