#!/bin/bash
# GAMMA - Behavior Cloning (BC)
# 환경: Overcooked + old_dynamics
# Usage: bash bc.sh [layout]

source "$(dirname "$0")/../common_args.sh"
setup_dirs

env="Overcooked"
algo="bc"
exp="bc"
num_agents=2

# layout → human_layout 매핑 (인간 데이터 레이아웃명이 다른 경우)
get_human_layout() {
    case $1 in
        counter_circuit_o_1order) echo "random3" ;;
        *) echo "$1" ;;
    esac
}

export POLICY_POOL="${GAMMA_POLICY_POOL}"

if [ -n "$1" ]; then
    run_layouts=("$1")
else
    run_layouts=("${LAYOUTS[@]}")
fi

for layout in "${run_layouts[@]}"; do
    human_layout=$(get_human_layout "${layout}")
    echo "=== GAMMA BC | layout=${layout}, human_layout=${human_layout} ==="
    for seed in $(seq ${SEED_BEGIN} ${SEED_END}); do
        echo "  seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python "${GAMMA_TRAIN_DIR}/train_overcooked_bc.py" \
            --env_name "${env}" \
            --algorithm_name "${algo}" \
            --experiment_name "${exp}" \
            --layout_name "${layout}" \
            --num_agents ${num_agents} \
            --seed "${seed}" \
            --n_training_threads 1 \
            --n_rollout_threads 1 \
            --num_mini_batch 1 \
            --episode_length 400 \
            --num_env_steps 10000000 \
            --reward_shaping_horizon 0 \
            --ppo_epoch 15 \
            --use_recurrent_policy \
            --old_dynamics \
            --human_data_refresh \
            --bc_num_epochs 100 \
            --bc_batch_size 128 \
            --lr 1e-2 \
            --human_layout_name "${human_layout}" \
            --save_interval 25 \
            --log_interval 10 \
            --use_eval --eval_stochastic --eval_interval 25 --eval_episodes 5 \
            --use_wandb \
            --wandb_name "${WANDB_ENTITY}"
    done
done
