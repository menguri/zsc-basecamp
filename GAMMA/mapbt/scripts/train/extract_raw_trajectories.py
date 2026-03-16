#!/usr/bin/env python
"""Extract raw (non-featurized) human trajectories and save as pickle.

Run with GAMMA's python (.zsc-gamma) which has human_aware_rl installed:
    .zsc-gamma/bin/python3 GAMMA/mapbt/scripts/train/extract_raw_trajectories.py \
        --layout_name cramped_room \
        --human_data_split 2019-train \
        --output_path /path/to/cramped_room_raw.pickle

The output pickle contains:
    {
        "ep_states":  list of episodes, each a list of state dicts (OvercookedState.to_dict())
        "ep_actions": list of episodes, each a list of [action_p0, action_p1]
    }

This raw format can be loaded and re-featurized by ZSC-EVAL's overcooked_new env.
"""

import argparse
import pickle
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_name", type=str, required=True)
    parser.add_argument(
        "--human_data_split",
        type=str,
        default="2019-train",
        choices=["2019-train", "2019-test", "2020-train", "2024-train", "2024-test"],
    )
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # Alias mapping: some layouts are stored under different names in human data
    _LAYOUT_ALIAS = {
        "forced_coordination": "random0",
        "counter_circuit_o_1order": "random3",
    }
    args.layout_name = _LAYOUT_ALIAS.get(args.layout_name, args.layout_name)

    # Load data split path
    if args.human_data_split == "2019-train":
        from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN as DATA_PATH
    elif args.human_data_split == "2019-test":
        from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TEST as DATA_PATH
    elif args.human_data_split == "2020-train":
        from human_aware_rl.static import CLEAN_2020_HUMAN_DATA_TRAIN as DATA_PATH
    elif args.human_data_split == "2024-train":
        from human_aware_rl.static import CLEAN_2024_HUMAN_DATA_TRAIN as DATA_PATH
    elif args.human_data_split == "2024-test":
        from human_aware_rl.static import CLEAN_2024_HUMAN_DATA_TEST as DATA_PATH
    else:
        raise ValueError(f"Unknown split: {args.human_data_split}")

    from human_aware_rl.human.process_dataframes import get_human_human_trajectories
    from overcooked_ai_py.mdp.actions import Action

    print(f"Extracting raw trajectories: layout={args.layout_name}, split={args.human_data_split}")

    data_params = {
        "layouts": [args.layout_name],
        "check_trajectories": False,
        "featurize_states": False,  # raw OvercookedState objects
        "data_path": DATA_PATH,
    }
    trajectories = get_human_human_trajectories(**data_params, silent=True)

    ep_states_raw = trajectories["ep_states"]    # list of lists of OvercookedState
    ep_actions_raw = trajectories["ep_actions"]  # list of lists of joint actions

    # Serialize states to dicts (cross-version compatible)
    ep_states_dicts = []
    for episode in ep_states_raw:
        ep_dicts = []
        for state in episode:
            if hasattr(state, "to_dict"):
                ep_dicts.append(state.to_dict())
            else:
                # Already a dict
                ep_dicts.append(state)
        ep_states_dicts.append(ep_dicts)

    # Normalize actions to list of [int, int] per timestep.
    # Each raw action is ONE player's direction tuple (dx, dy) or 'interact'.
    # We store [action_idx, action_idx] so the re-featurizer can create samples
    # from both player perspectives using the same tracked-player action.
    all_actions = list(Action.ALL_ACTIONS)

    def _direction_to_idx(action):
        """Convert a direction tuple (dx,dy) or 'interact' str to ALL_ACTIONS index."""
        if isinstance(action, str):
            return all_actions.index(action)
        elif isinstance(action, tuple):
            return all_actions.index(action)
        elif isinstance(action, int) and 0 <= action < len(all_actions):
            return action  # already a valid index
        else:
            # numpy scalar or other: try direct index lookup
            return all_actions.index(action)

    ep_actions_int = []
    for episode in ep_actions_raw:
        ep_ints = []
        for action in episode:
            ep_ints.append(_direction_to_idx(action))
        ep_actions_int.append(ep_ints)

    raw_data = {
        "ep_states": ep_states_dicts,
        "ep_actions": ep_actions_int,
        "layout_name": args.layout_name,
        "split": args.human_data_split,
    }

    with open(args.output_path, "wb") as f:
        pickle.dump(raw_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_ep = len(ep_states_dicts)
    n_steps = sum(len(ep) for ep in ep_states_dicts)
    print(f"Saved {n_ep} episodes ({n_steps} total steps) → {args.output_path}")


if __name__ == "__main__":
    main()
