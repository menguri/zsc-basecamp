"""
PH2 configuration.

Reuses ZSC-EVAL's get_config() + get_overcooked_args() and adds PH2-specific args.
MAPPO hyperparameter defaults are kept identical to ZSC-EVAL.
"""
import argparse


def get_ph2_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add PH2-specific arguments to an existing parser."""
    existing_opts = set(getattr(parser, "_option_string_actions", {}).keys())

    def _add_if_missing(*option_strings, **kwargs):
        if any(opt in existing_opts for opt in option_strings):
            return
        parser.add_argument(*option_strings, **kwargs)
        existing_opts.update(option_strings)

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
    parser.add_argument(
        "--ph2_share_pred",
        action="store_true",
        default=False,
        help=(
            "Share PartnerPredictionNet parameters between spec and ind policies. "
            "If False (default), each policy has its own predictor trained independently."
        ),
    )

    # ---- blocked states (spec only) ----
    parser.add_argument(
        "--ph2_spec_use_blocked",
        action="store_true",
        default=False,
        help="Enable blocked-state penalty + actor input for spec policy.",
    )
    parser.add_argument(
        "--ph2_num_blocked_slots",
        type=int,
        default=1,
        help="K_max: maximum number of blocked obs per env per episode.",
    )
    parser.add_argument(
        "--ph2_blocked_pool_size",
        type=int,
        default=200,
        help="Max size of per-env FIFO pool from which blocked obs are sampled.",
    )
    # Penalty shape: omega * exp(-sigma * L2_dist)
    parser.add_argument(
        "--ph2_blocked_penalty_omega",
        type=float,
        default=10.0,
        help="Scale (omega) for blocked-state L2-distance penalty on spec reward.",
    )
    parser.add_argument(
        "--ph2_blocked_penalty_sigma",
        type=float,
        default=2.0,
        help="Decay rate (sigma) for blocked-state L2-distance penalty on spec reward.",
    )
    # V_gap-based sampling temperature
    parser.add_argument(
        "--ph2_vgap_beta",
        type=float,
        default=1.0,
        help=(
            "Temperature beta for V_gap softmax sampling of blocked states. "
            "Higher beta = more focused on states with small V_gap "
            "(states the policy has not yet learned to avoid)."
        ),
    )
    parser.add_argument(
        "--ph2_vgap_beta_schedule_enabled",
        action="store_true",
        default=True,
        help="Enable linear schedule for V_gap beta (start -> end).",
    )
    parser.add_argument(
        "--ph2_no_vgap_beta_schedule",
        action="store_false",
        dest="ph2_vgap_beta_schedule_enabled",
        help="Disable V_gap beta schedule and use fixed ph2_vgap_beta.",
    )
    parser.add_argument(
        "--ph2_vgap_beta_start",
        type=float,
        default=0.0,
        help="Start value of V_gap beta schedule.",
    )
    parser.add_argument(
        "--ph2_vgap_beta_end",
        type=float,
        default=1.0,
        help="End value of V_gap beta schedule.",
    )
    parser.add_argument(
        "--ph2_vgap_beta_horizon_env_steps",
        type=int,
        default=-1,
        help=(
            "Horizon (env steps) for V_gap beta schedule. "
            "If <=0, uses total PH2 training env steps."
        ),
    )
    parser.add_argument(
        "--ph2_epsilon",
        type=float,
        default=0.2,
        help=(
            "Random action probability for PH2 epsilon behavior "
            "(spec-spec, spec-ind spec slot, ind-ind)."
        ),
    )
    _add_if_missing(
        "--use_phi",
        default=False,
        action="store_true",
        help=(
            "Compatibility flag with ZSC-EVAL overcooked train scripts. "
            "Keeps a fixed policy-agent index when paired with external partner agents."
        ),
    )
    _add_if_missing(
        "--use_task_v_out",
        default=False,
        action="store_true",
        help="Compatibility flag for task-conditioned value heads.",
    )
    # ---- parser compatibility aliases ----
    # ZSC-EVAL uses `--entropy_coefs/--entropy_coef_horizons`.
    # Some legacy scripts still pass a scalar `--entropy_coef`.
    _add_if_missing(
        "--entropy_coef",
        type=float,
        default=None,
        help=(
            "Legacy alias. If set, overrides entropy schedule as "
            "--entropy_coefs <v> <v> and --entropy_coef_horizons 0 <num_env_steps>."
        ),
    )

    # ---- wandb project / entity override ----
    _add_if_missing(
        "--wandb_project",
        type=str,
        default="zsc-basecamp",
        help="W&B project name.",
    )
    _add_if_missing(
        "--wandb_entity",
        type=str,
        default="m-personal-experiment",
        help="W&B entity (team/user) name.",
    )
    _add_if_missing(
        "--wandb_tags",
        type=str,
        nargs="*",
        default=[],
        help="W&B run tags.",
    )

    return parser
