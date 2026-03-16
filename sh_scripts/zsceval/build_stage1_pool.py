#!/usr/bin/env python3
"""
Build ZSC-EVAL Stage1 policy pools from local training results.

Supported targets:
  - fcp:  results/zsceval/Overcooked/<layout>/mappo/sp/run*/models/actor_periodic_*.pt
          -> policy_pool/<layout>/fcp/s1/sp/sp{i}_{init,mid,final}_actor.pt
  - mep:  results/zsceval/Overcooked/<layout>/mep/mep-S1-s<pop>/run*/models/mep{i}/actor_periodic_*.pt
          -> policy_pool/<layout>/mep/s1/mep-S1-s<pop>/mep{i}_{init,mid,final}_actor.pt
  - traj: results/zsceval/Overcooked/<layout>/traj/traj-S1-s<pop>/run*/models/traj{i}/actor_periodic_*.pt
          -> policy_pool/<layout>/traj/s1/traj-S1-s<pop>/traj{i}_{init,mid,final}_actor.pt
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


_STEP_RE = re.compile(r"actor_periodic_(\d+)\.pt$")


def _run_sort_key(p: Path) -> Tuple[int, str]:
    m = re.search(r"(\d+)$", p.name)
    if m:
        return (int(m.group(1)), p.name)
    return (10**9, p.name)


def _collect_periodic_ckpts(model_dir: Path) -> List[Tuple[int, Path]]:
    if not model_dir.exists():
        return []
    out: List[Tuple[int, Path]] = []
    for pt in model_dir.glob("actor_periodic_*.pt"):
        m = _STEP_RE.search(pt.name)
        if not m:
            continue
        out.append((int(m.group(1)), pt))
    out.sort(key=lambda x: x[0])
    return out


def _choose_init_mid_final(ckpts: List[Tuple[int, Path]]) -> Dict[str, Path]:
    if not ckpts:
        raise RuntimeError("No actor_periodic checkpoints found.")
    if len(ckpts) == 1:
        only = ckpts[0][1]
        return {"init": only, "mid": only, "final": only}
    init_step, init_pt = ckpts[0]
    final_step, final_pt = ckpts[-1]
    mid_target = (init_step + final_step) / 2.0
    mid_step, mid_pt = min(ckpts, key=lambda x: abs(x[0] - mid_target))
    _ = mid_step  # keep variable for readability
    return {"init": init_pt, "mid": mid_pt, "final": final_pt}


def _build_fcp(repo_root: Path, layout: str, population_size: int) -> None:
    src_base = repo_root / "results" / "zsceval" / "Overcooked" / layout / "mappo" / "sp"
    if not src_base.exists():
        raise RuntimeError(f"[fcp] Missing SP results dir: {src_base}")

    run_dirs = sorted([p for p in src_base.iterdir() if p.is_dir() and p.name.startswith("run")], key=_run_sort_key)
    if len(run_dirs) < population_size:
        raise RuntimeError(
            f"[fcp] Need at least {population_size} SP runs under {src_base}, found {len(run_dirs)}."
        )

    dst_dir = repo_root / "policy_pool" / layout / "fcp" / "s1" / "sp"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(1, population_size + 1):
        run_dir = run_dirs[idx - 1]
        ckpts = _collect_periodic_ckpts(run_dir / "models")
        chosen = _choose_init_mid_final(ckpts)
        for tag, src in chosen.items():
            dst = dst_dir / f"sp{idx}_{tag}_actor.pt"
            shutil.copy2(src, dst)


def _build_population_from_stage1(
    repo_root: Path,
    layout: str,
    algo: str,
    population_size: int,
) -> None:
    if algo not in ("mep", "traj"):
        raise ValueError(f"Unsupported algo={algo}")
    exp = f"{algo}-S1-s{population_size}"
    src_base = repo_root / "results" / "zsceval" / "Overcooked" / layout / algo / exp
    if not src_base.exists():
        raise RuntimeError(f"[{algo}] Missing Stage1 results dir: {src_base}")

    run_dirs = sorted([p for p in src_base.iterdir() if p.is_dir() and p.name.startswith("run")], key=_run_sort_key)
    if not run_dirs:
        raise RuntimeError(f"[{algo}] No run directories found under {src_base}")
    run_dir = run_dirs[-1]

    models_root = run_dir / "models"
    if not models_root.exists():
        raise RuntimeError(f"[{algo}] Missing models directory: {models_root}")

    dst_dir = repo_root / "policy_pool" / layout / algo / "s1" / exp
    dst_dir.mkdir(parents=True, exist_ok=True)

    prefix = algo
    for idx in range(1, population_size + 1):
        trainer_dir = models_root / f"{prefix}{idx}"
        if not trainer_dir.exists():
            raise RuntimeError(f"[{algo}] Missing trainer directory: {trainer_dir}")
        ckpts = _collect_periodic_ckpts(trainer_dir)
        chosen = _choose_init_mid_final(ckpts)
        for tag, src in chosen.items():
            dst = dst_dir / f"{prefix}{idx}_{tag}_actor.pt"
            shutil.copy2(src, dst)


def _check_complete(repo_root: Path, layout: str, algo: str, population_size: int) -> None:
    if algo == "fcp":
        base = repo_root / "policy_pool" / layout / "fcp" / "s1" / "sp"
        expected = [base / f"sp{i}_{tag}_actor.pt" for i in range(1, population_size + 1) for tag in ("init", "mid", "final")]
    elif algo in ("mep", "traj"):
        exp = f"{algo}-S1-s{population_size}"
        base = repo_root / "policy_pool" / layout / algo / "s1" / exp
        expected = [
            base / f"{algo}{i}_{tag}_actor.pt"
            for i in range(1, population_size + 1)
            for tag in ("init", "mid", "final")
        ]
    else:
        raise ValueError(f"Unsupported algo={algo}")

    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        head = "\n  ".join(missing[:10])
        raise RuntimeError(f"[{algo}] Pool build incomplete. Missing {len(missing)} files, e.g.:\n  {head}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage1 pool files from local zsceval run directories.")
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--algo", type=str, choices=["fcp", "mep", "traj"], required=True)
    parser.add_argument("--population_size", type=int, default=12)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    layout = args.layout
    pop = int(args.population_size)

    if args.algo == "fcp":
        _build_fcp(repo_root, layout, pop)
    elif args.algo in ("mep", "traj"):
        _build_population_from_stage1(repo_root, layout, args.algo, pop)
    else:
        raise ValueError(args.algo)

    _check_complete(repo_root, layout, args.algo, pop)
    print(f"[ok] built {args.algo} stage1 pool for layout={layout}, pop={pop}")


if __name__ == "__main__":
    main()

