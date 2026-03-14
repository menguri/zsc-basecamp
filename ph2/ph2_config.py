"""
PH2 configuration.

Reuses ZSC-EVAL's get_config() + get_overcooked_args() and adds PH2-specific args.
MAPPO hyperparameter defaults are kept identical to ZSC-EVAL.
"""
import argparse


def get_ph2_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add PH2-specific arguments to an existing parser."""

    # ---- dual-policy schedule ----
    parser.add_argument(
        "--ph2_fixed_ind_prob",
        type=float,
        default=0.5,
        help=(
            "Fixed probability of an env being an ind-match episode. "
            "Set to -1 to use staged schedule (ph2_ratio_stage*)."
        ),
    )
    parser.add_argument("--ph2_ratio_stage1", type=int, default=2,
                        help="PH2 ind/spec ratio in stage 1 (0-33%% training).")
    parser.add_argument("--ph2_ratio_stage2", type=int, default=1,
                        help="PH2 ind/spec ratio in stage 2 (33-67%% training).")
    parser.add_argument("--ph2_ratio_stage3", type=int, default=2,
                        help="PH2 ind/spec ratio in stage 3 (67-100%% training).")

    # ---- partner prediction (E3T) ----
    parser.add_argument(
        "--ph2_use_partner_pred",
        action="store_true",
        default=True,
        help="Use E3T partner prediction loss for the ind policy (default: True).",
    )
    parser.add_argument(
        "--ph2_pred_loss_coef",
        type=float,
        default=1.0,
        help="Coefficient for partner prediction CE loss.",
    )
    parser.add_argument(
        "--ph2_history_len",
        type=int,
        default=5,
        help="Number of historical (obs, action) pairs fed to the partner predictor.",
    )

    # ---- wandb project / entity override ----
    parser.add_argument("--wandb_project", type=str, default="zsc-basecamp",
                        help="W&B project name.")
    parser.add_argument("--wandb_entity",  type=str, default="m-personal-experiment",
                        help="W&B entity (team/user) name.")
    parser.add_argument("--wandb_tags",    type=str, nargs="*", default=[],
                        help="W&B run tags.")

    return parser
