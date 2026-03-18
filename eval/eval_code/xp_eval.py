#!/usr/bin/env python3
import argparse
import copy
import csv
import itertools
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class RunModel:
    algo: str
    layout: str
    run_name: str
    run_dir: Path
    actor_path: Path          # agent0 (p0 position)
    policy_config_path: Path
    featurize_type: str
    # ph2 only: full ind actor (with _ph2_proj) and PartnerPredictionNet weights
    ind_actor_path: Optional[Path] = None
    pred_path: Optional[Path] = None
    # e3t only: position-specific actor for p1
    actor_agent1_path: Optional[Path] = None


def _sorted_run_dirs(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []

    runs = [p for p in base_dir.iterdir() if p.is_dir()]

    def _run_key(p: Path):
        name = p.name
        digits = "".join(ch for ch in name if ch.isdigit())
        if digits:
            return (0, int(digits), name)
        return (1, 0, name)

    return sorted(runs, key=_run_key)


def _find_policy_config(run_dir: Path) -> Optional[Path]:
    # Prefer config placed directly in run dir.
    direct = run_dir / "policy_config.pkl"
    if direct.exists():
        return direct

    # Fallback to recursive search (some runs keep files in subfolders).
    found = sorted(run_dir.rglob("policy_config.pkl"))
    if found:
        return found[0]
    return None


def _find_latest_actor_pt(run_dir: Path, algo: str = "") -> Optional[Path]:
    all_pts = [p for p in run_dir.rglob("*.pt") if p.is_file()]
    if not all_pts:
        return None

    # Exclude non-actor checkpoints first.
    def _is_actor_like(p: Path) -> bool:
        name = p.name.lower()
        if "critic" in name or "pred" in name:
            return False
        return True

    candidates = [p for p in all_pts if _is_actor_like(p)]
    if not candidates:
        candidates = all_pts

    # E3T: agent1 is a training-only lagging copy; always use agent0 (ego) for eval.
    if algo.lower() == "e3t":
        agent0 = [p for p in candidates if "agent0" in p.name.lower()]
        if agent0:
            candidates = agent0

    def _priority(p: Path) -> int:
        name = p.name.lower()
        if "actor" in name:
            return 3
        if "model" in name:
            return 2
        return 1

    return max(candidates, key=lambda p: (_priority(p), p.stat().st_mtime, str(p)))


def _find_e3t_agent1_pt(run_dir: Path) -> Optional[Path]:
    """e3t only: actor_agent1_*.pt for use when this model occupies p1 position."""
    candidates = [p for p in run_dir.rglob("*.pt") if "agent1" in p.name.lower()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))


def _find_latest_ind_actor_pt(run_dir: Path) -> Optional[Path]:
    """ph2 only: ind_actor_*.pt with full PH2Actor weights (includes _ph2_proj)."""
    candidates = [p for p in run_dir.rglob("ind_actor_*.pt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_latest_ind_pred_pt(run_dir: Path) -> Optional[Path]:
    """ph2 only: ind_pred_*.pt with PartnerPredictionNet weights."""
    candidates = [p for p in run_dir.rglob("ind_pred_*.pt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _algo_to_featurize(algo: str) -> str:
    if algo.lower() == "bc":
        return "bc"
    return "ppo"


def _candidate_model_dirs(models_root: Path, layout: str, algo: str) -> List[Path]:
    # New preferred layout:
    #   eval/models/<layout>/<algorithm>/run_x
    # Backward-compatible fallback:
    #   eval/models/<algorithm>/<layout>/run_x
    return [
        models_root / layout / algo,
        models_root / algo / layout,
    ]


def collect_runs(models_root: Path, algo: str, layout: str) -> List[RunModel]:
    for base in _candidate_model_dirs(models_root, layout, algo):
        run_dirs = _sorted_run_dirs(base)
        out: List[RunModel] = []
        for run_dir in run_dirs:
            policy_cfg = _find_policy_config(run_dir)
            actor = _find_latest_actor_pt(run_dir, algo=algo)
            if policy_cfg is None or actor is None:
                continue
            is_ph2 = algo.lower() == "ph2"
            is_e3t = algo.lower() == "e3t"
            out.append(
                RunModel(
                    algo=algo,
                    layout=layout,
                    run_name=run_dir.name,
                    run_dir=run_dir,
                    actor_path=actor,
                    policy_config_path=policy_cfg,
                    featurize_type=_algo_to_featurize(algo),
                    ind_actor_path=_find_latest_ind_actor_pt(run_dir) if is_ph2 else None,
                    pred_path=_find_latest_ind_pred_pt(run_dir) if is_ph2 else None,
                    actor_agent1_path=_find_e3t_agent1_pt(run_dir) if is_e3t else None,
                )
            )
        if out:
            return out
    return []


def _patched_policy_config(src_policy_config: Path, out_path: Path, algo: str) -> None:
    # Use eval-compatible rmappo policy loader for PH2 checkpoints.
    # PH2 export strips PH2-only projection head in actor.pt alias.
    with open(src_policy_config, "rb") as f:
        cfg = pickle.load(f)

    if not isinstance(cfg, tuple) or len(cfg) != 4:
        raise RuntimeError(f"Unexpected policy_config format in {src_policy_config}")

    args, obs_space, share_obs_space, act_space = cfg
    args_copy = copy.deepcopy(args)

    algo_lc = algo.lower()
    if algo_lc in ("ph2", "e3t"):
        # ph2 and e3t store non-standard algorithm_name in policy_config;
        # remap to rmappo so the standard R_MAPPOPolicy is used for eval.
        setattr(args_copy, "algorithm_name", "rmappo")
        if hasattr(args_copy, "use_single_network"):
            setattr(args_copy, "use_single_network", False)
    elif algo_lc == "fcp":
        # fcp stores algorithm_name='adaptive' (population-based training);
        # the exported actor is a standard MAPPO network, remap for eval.
        setattr(args_copy, "algorithm_name", "mappo")
        # fcp was trained with custom CNN params but policy_config stores None;
        # ZSC-EVAL default is (16,5,1,0) which mismatches the saved weights.
        if getattr(args_copy, "cnn_layers_params", None) is None:
            setattr(args_copy, "cnn_layers_params", "32,3,1,1 64,3,1,1 32,3,1,1")

    # Eval never uses the critic; skip its construction to avoid CNN dimension
    # errors when share_obs_space is too small for the default kernel sizes.
    setattr(args_copy, "eval_skip_critic", True)

    with open(out_path, "wb") as f:
        pickle.dump((args_copy, obs_space, share_obs_space, act_space), f)


def _build_population_yaml(
    tmp_dir: Path,
    model0: RunModel,
    model1: RunModel,
) -> Path:
    cfg0 = tmp_dir / "p0_policy_config.pkl"
    cfg1 = tmp_dir / "p1_policy_config.pkl"
    _patched_policy_config(model0.policy_config_path, cfg0, model0.algo)
    _patched_policy_config(model1.policy_config_path, cfg1, model1.algo)

    def _model_path_entry(model: RunModel, position: int) -> dict:
        # e3t uses position-specific actors: agent0 for p0, agent1 for p1.
        if model.algo.lower() == "e3t" and position == 1 and model.actor_agent1_path is not None:
            actor = model.actor_agent1_path
        else:
            actor = model.actor_path
        entry: dict = {"actor": str(actor)}
        if model.ind_actor_path is not None:
            entry["ind_actor"] = str(model.ind_actor_path)
        return entry

    pop_cfg = {
        "p0": {
            "policy_config_path": str(cfg0),
            "featurize_type": model0.featurize_type,
            "model_path": _model_path_entry(model0, position=0),
            **({"pred_model_path": str(model0.pred_path)} if model0.pred_path is not None else {}),
        },
        "p1": {
            "policy_config_path": str(cfg1),
            "featurize_type": model1.featurize_type,
            "model_path": _model_path_entry(model1, position=1),
            **({"pred_model_path": str(model1.pred_path)} if model1.pred_path is not None else {}),
        },
    }
    yml_path = tmp_dir / "population.yml"
    with open(yml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(pop_cfg, f, sort_keys=False)
    return yml_path


def _ensure_even_threads(n_threads: int) -> int:
    if n_threads < 2:
        return 2
    if n_threads % 2 == 1:
        return n_threads + 1
    return n_threads


def _parse_seeds(seeds: str) -> List[int]:
    out = []
    for s in seeds.split(","):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        out = [1]
    return out


def _run_one_eval(
    repo_root: Path,
    policy_pool_root: Path,
    population_yml: Path,
    layout: str,
    seed: int,
    eval_episodes: int,
    eval_threads: int,
    eval_steps: int,
    output_json: Path,
    cuda_visible_devices: str,
    viz: bool = False,
    gif_output_path: Optional[Path] = None,
) -> Dict[str, float]:
    eval_py = repo_root / "ZSC-EVAL" / "zsceval" / "scripts" / "overcooked" / "eval" / "eval.py"
    if not eval_py.exists():
        raise FileNotFoundError(f"Missing eval script: {eval_py}")

    cmd = [
        sys.executable,
        str(eval_py),
        "--env_name",
        "Overcooked",
        "--algorithm_name",
        "population",
        "--experiment_name",
        "xp_eval",
        "--layout_name",
        layout,
        "--user_name",
        "eval",
        "--num_agents",
        "2",
        "--seed",
        str(seed),
        "--episode_length",
        str(eval_steps),
        "--n_eval_rollout_threads",
        str(eval_threads),
        "--dummy_batch_size",
        str(eval_threads),
        "--eval_episodes",
        str(eval_episodes),
        "--population_size",
        "2",
        "--population_yaml_path",
        str(population_yml),
        "--agent0_policy_name",
        "p0",
        "--agent1_policy_name",
        "p1",
        "--eval_result_path",
        str(output_json),
        "--overcooked_version",
        "new",
        # In ZSC-EVAL parser, --use_wandb is store_false.
        # Pass this flag to disable wandb during batch XP evaluation.
        "--use_wandb",
    ]
    if viz:
        cmd.append("--use_render")

    env = os.environ.copy()
    env["POLICY_POOL"] = str(policy_pool_root)
    env.setdefault("EVOLVE_ACTOR_POOL", str(policy_pool_root))
    env["PYTHONPATH"] = f"{repo_root / 'ZSC-EVAL'}:{env.get('PYTHONPATH', '')}"
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    start_ts = time.time()
    subprocess.run(cmd, env=env, check=True)

    if viz and gif_output_path is not None:
        home = Path(os.path.expanduser("~"))
        eval_run_root = home / "ZSC" / "results" / "Overcooked" / layout / "population" / "xp_eval"
        gifs = [p for p in eval_run_root.rglob("*.gif") if p.is_file()] if eval_run_root.exists() else []
        if gifs:
            latest = max(gifs, key=lambda p: p.stat().st_mtime)
            # Prefer gifs generated by this eval call; fallback to latest overall.
            if latest.stat().st_mtime < (start_ts - 2):
                recent = [p for p in gifs if p.stat().st_mtime >= (start_ts - 2)]
                if recent:
                    latest = max(recent, key=lambda p: p.stat().st_mtime)
            gif_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest, gif_output_path)

    with open(output_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _reward_from_eval_json(eval_json: Dict[str, float]) -> Tuple[float, float, Optional[float], Optional[float]]:
    # eval.py already evaluates swapped seats in one run.
    d01 = eval_json.get("p0-p1-eval_ep_sparse_r", None)
    d10 = eval_json.get("p1-p0-eval_ep_sparse_r", None)
    vals = [v for v in [d01, d10] if v is not None]
    if not vals:
        # fallback to either metric
        v = eval_json.get("either-p0-eval_ep_sparse_r", 0.0)
        return float(v), 0.0, d01, d10
    vals_np = np.asarray(vals, dtype=np.float32)
    return float(vals_np.mean()), float(vals_np.var()), d01, d10


def _build_pairs(runs0: List[RunModel], runs1: List[RunModel], same_algo: bool) -> List[Tuple[RunModel, RunModel]]:
    if same_algo:
        # Full cross seed matrix including self-play: (0,0), (1,1), (1,2), (2,1), ...
        return list(itertools.product(runs0, runs0))
    return list(itertools.product(runs0, runs1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-play evaluator for zsc-basecamp/eval")
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--models_root", type=Path, required=True)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--algo0", type=str, required=True)
    parser.add_argument("--algo1", type=str, default="")
    parser.add_argument("--eval_mode", type=str, default="xp")
    parser.add_argument("--eval_steps", type=int, default=400)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--n_eval_threads", type=int, default=2)
    parser.add_argument("--eval_seeds", type=str, default="1")
    parser.add_argument("--cuda_visible_devices", type=str, default="")
    parser.add_argument("--viz", action="store_true", help="Enable visualization gif dump per pair.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.eval_mode != "xp":
        raise ValueError(f"Unsupported eval_mode={args.eval_mode}; only xp is supported.")

    algo0 = args.algo0.strip()
    algo1 = args.algo1.strip()
    same_algo_mode = (algo1 == "") or (algo1 == algo0)
    if algo1 == "":
        algo1 = algo0

    runs0 = collect_runs(args.models_root, algo0, args.layout)
    runs1 = collect_runs(args.models_root, algo1, args.layout)

    if not runs0:
        cand = ", ".join(str(p) for p in _candidate_model_dirs(args.models_root, args.layout, algo0))
        raise RuntimeError(f"No valid runs found for algo0={algo0}. Searched: {cand}")
    if not runs1:
        cand = ", ".join(str(p) for p in _candidate_model_dirs(args.models_root, args.layout, algo1))
        raise RuntimeError(f"No valid runs found for algo1={algo1}. Searched: {cand}")

    pairs = _build_pairs(runs0, runs1, same_algo_mode)
    eval_seeds = _parse_seeds(args.eval_seeds)
    eval_threads = _ensure_even_threads(args.n_eval_threads)

    if args.eval_episodes % eval_threads != 0:
        # Keep runner assumption simple.
        adjusted = max(eval_threads, (args.eval_episodes // eval_threads) * eval_threads)
        args.eval_episodes = adjusted

    combo_name = f"xp_{algo0}_{algo1}"
    combo_dir = args.results_root / args.layout / combo_name
    csv_dir = combo_dir / "csv"
    gif_dir = combo_dir / "gif"
    csv_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "xp_pairs.csv"

    if csv_path.exists() and not args.overwrite:
        raise FileExistsError(f"{csv_path} exists. Use --overwrite to replace it.")

    rows: List[Dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="xp_eval_pool_") as td:
        tmp_root = Path(td)
        for i, (m0, m1) in enumerate(pairs):
            for seed in eval_seeds:
                pair_tag = f"pair_{i:04d}_{m0.run_name}_{m1.run_name}_seed{seed}"
                pair_tmp = tmp_root / pair_tag
                pair_tmp.mkdir(parents=True, exist_ok=True)
                yml = _build_population_yaml(pair_tmp, m0, m1)

                out_json = csv_dir / f"{pair_tag}.json"
                out_gif = gif_dir / f"{pair_tag}.gif" if args.viz else None
                eval_json = _run_one_eval(
                    repo_root=args.repo_root,
                    policy_pool_root=pair_tmp,
                    population_yml=yml,
                    layout=args.layout,
                    seed=seed,
                    eval_episodes=args.eval_episodes,
                    eval_threads=eval_threads,
                    eval_steps=args.eval_steps,
                    output_json=out_json,
                    cuda_visible_devices=args.cuda_visible_devices,
                    viz=args.viz,
                    gif_output_path=out_gif,
                )

                reward_mean, reward_var, d01, d10 = _reward_from_eval_json(eval_json)
                rows.append(
                    {
                        "layout": args.layout,
                        "algo0": algo0,
                        "algo1": algo1,
                        "run0": m0.run_name,
                        "run1": m1.run_name,
                        "seed": seed,
                        "actor0": str(m0.actor_path),
                        "actor1": str(m1.actor_path),
                        "reward_mean": reward_mean,
                        "reward_var": reward_var,
                        "reward_std": float(np.sqrt(max(0.0, reward_var))),
                        "reward_seat_p0_p1": "" if d01 is None else float(d01),
                        "reward_seat_p1_p0": "" if d10 is None else float(d10),
                        "result_json": str(out_json),
                    }
                )

    fieldnames = [
        "layout",
        "algo0",
        "algo1",
        "run0",
        "run1",
        "seed",
        "actor0",
        "actor1",
        "reward_mean",
        "reward_var",
        "reward_std",
        "reward_seat_p0_p1",
        "reward_seat_p1_p0",
        "result_json",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        means = np.asarray([float(r["reward_mean"]) for r in rows], dtype=np.float32)
        summary = {
            "layout": args.layout,
            "algo0": algo0,
            "algo1": algo1,
            "num_rows": len(rows),
            "reward_mean": float(means.mean()),
            "reward_var": float(means.var()),
            "reward_std": float(means.std()),
            "csv_path": str(csv_path),
        }
    else:
        summary = {
            "layout": args.layout,
            "algo0": algo0,
            "algo1": algo1,
            "num_rows": 0,
            "reward_mean": 0.0,
            "reward_var": 0.0,
            "reward_std": 0.0,
            "csv_path": str(csv_path),
        }

    with open(csv_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
