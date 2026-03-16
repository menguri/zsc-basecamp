import argparse
import os
import os.path as osp

from loguru import logger

policy_pool_dir = "../policy_pool"

S1_POP_EXPS = {
    "fcp": {
        5: "sp",
    },
    "mep": {
        12: "mep-S1-s12",
    },
    "traj": {
        12: "traj-S1-s12",
    },
}

# 5-run source setting for all algs.
# stage2 target size differs by algorithm:
#   - fcp: 15 (= 5 * 3 checkpoints)
#   - mep/traj: 12 (= 4 * 3 checkpoints from S1 pop=12)
S1_TOTAL_SIZE = {
    "fcp": 5,
    "mep": 12,
    "traj": 12,
}

S2_POP_SIZE = {
    "fcp": 5,
    "mep": 4,
    "traj": 4,
}

N_REPEAT = 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=str)
    parser.add_argument("alg", type=str, choices=["fcp", "mep", "traj"])

    args = parser.parse_args()

    if args.layout == "all":
        layouts = [
            "random0",
            "random0_medium",
            "random1",
            "random3",
            "small_corridor",
            "unident_s",
            "random0_m",
            "random1_m",
            "random3_m",
            "academy_3_vs_1_with_keeper",
        ]
    else:
        layouts = [args.layout]

    for layout in layouts:
        total_size = S1_TOTAL_SIZE[args.alg]
        pop_size = S2_POP_SIZE[args.alg]
        exp = S1_POP_EXPS[args.alg][total_size]
        source_dir = osp.join(policy_pool_dir, layout, args.alg, "s1", exp)
        if not osp.isdir(source_dir):
            raise FileNotFoundError(f"Source dir not found: {source_dir}")
        pt_lst = os.listdir(source_dir)
        logger.debug(pt_lst)
        pop_alg = args.alg if args.alg != "fcp" else "sp"
        pt_lst.sort(key=lambda pt: int(pt.split("_", 1)[0][len(pop_alg) :]))
        if args.alg == "fcp":
            pt_lst = pt_lst[: total_size * 3]
            logger.info(f"pop size {len(pt_lst)}: {pt_lst}")
        yml_dir = osp.join(
            policy_pool_dir,
            layout,
            args.alg,
            "s2",
        )
        os.makedirs(yml_dir, exist_ok=True)
        for n_r in range(N_REPEAT):
            yml_path = osp.join(
                policy_pool_dir,
                layout,
                args.alg,
                "s2",
                f"train-s{pop_size*3}-{exp}-{n_r+1}.yml",
            )
            logger.info(f"Writing S2 yml for {exp} seed {n_r} in {yml_path}")
            yml = open(
                yml_path,
                "w",
                encoding="utf-8",
            )
            yml.write(
                f"""\
{args.alg}_adaptive:
    policy_config_path: {layout}/policy_config/rnn_policy_config.pkl
    featurize_type: ppo
    train: True
"""
            )
            for p_i in range(1, pop_size + 1):
                pt_i = (total_size // N_REPEAT * n_r + p_i - 1) % total_size + 1
                actor_names = [
                    f"{pop_alg}{pt_i}_init_actor.pt",
                    f"{pop_alg}{pt_i}_mid_actor.pt",
                    f"{pop_alg}{pt_i}_final_actor.pt",
                ]
                for actor_name in actor_names:
                    assert actor_name in pt_lst, (actor_name, pt_lst)
                yml.write(
                    f"""\
{pop_alg}{p_i}_1:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, args.alg, "s1", exp, actor_names[0])}
{pop_alg}{p_i}_2:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, args.alg, "s1", exp, actor_names[1])}
{pop_alg}{p_i}_3:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, args.alg, "s1", exp, actor_names[2])}
"""
                )
            yml.close()
