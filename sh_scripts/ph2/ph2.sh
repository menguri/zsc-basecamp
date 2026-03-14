#!/bin/bash
# sh_scripts/ph2/ph2.sh
# PH2 training script — 30M total steps (spec 15M + ind 15M)
#
# Usage:
#   bash sh_scripts/ph2/ph2.sh [LAYOUT]
#   GPU=1 bash sh_scripts/ph2/ph2.sh cramped_room
#   GPU=2 bash sh_scripts/ph2/ph2.sh asymmetric_advantages
#
# Blocked-state penalty: omega * exp(-sigma * L2_dist)
# Blocked-state sampling: V_gap softmax with temperature beta
# Spec/ind pred: separate by default (--ph2_share_pred to share)

source "$(dirname "$0")/../common_args.sh"
setup_dirs

LAYOUT=${1:-cramped_room}
PH2_DIR="${BASECAMP_DIR}/ph2"

echo "[ph2] layout=${LAYOUT}  GPU=${GPU}  seeds=${SEED_BEGIN}..${SEED_END}"

for seed in $(seq "${SEED_BEGIN}" "${SEED_END}"); do
    CUDA_VISIBLE_DEVICES="${GPU}" python "${PH2_DIR}/scripts/train/train_ph2.py" \
        --env_name "Overcooked" \
        --algorithm_name "ph2" \
        --experiment_name "ph2_s${seed}" \
        --layout_name "${LAYOUT}" \
        --seed ${seed} \
        --n_rollout_threads 100 \
        --n_eval_rollout_threads 5 \
        --num_env_steps 30000000 \
        --episode_length 400 \
        --hidden_size 64 \
        --lr 5e-4 \
        --critic_lr 5e-4 \
        --ppo_epoch 15 \
        --num_mini_batch 2 \
        --data_chunk_length 10 \
        --entropy_coef 0.01 \
        --reward_shaping_horizon 2500000 \
        --use_centralized_V \
        --use_recurrent_policy \
        --use_valuenorm \
        --use_policy_vhead \
        --ph2_fixed_ind_prob 0.5 \
        --ph2_use_partner_pred \
        --ph2_history_len 5 \
        --ph2_pred_loss_coef 1.0 \
        --ph2_spec_use_blocked \
        --ph2_num_blocked_slots 1 \
        --ph2_blocked_pool_size 200 \
        --ph2_blocked_penalty_omega 1.0 \
        --ph2_blocked_penalty_sigma 1.0 \
        --ph2_vgap_beta 1.0 \
        --save_interval 25 \
        --log_interval 5 \
        --eval_interval 25 \
        --use_eval \
        --use_wandb \
        --wandb_project "zsc-basecamp" \
        --wandb_entity "m-personal-experiment" \
        --wandb_tags "ph2" "${LAYOUT}" \
        --use_linear_lr_decay
done
