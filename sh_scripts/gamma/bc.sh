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

# BC training amount (can be overridden at runtime)
# Example: BC_NUM_EPOCHS=200 BC_BATCH_SIZE=128 bash sh_scripts/gamma/bc.sh
BC_NUM_EPOCHS="${BC_NUM_EPOCHS:-400}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-128}"
BC_LR="${BC_LR:-1e-2}"
BC_SAVE_INTERVAL="${BC_SAVE_INTERVAL:-25}"
BC_EVAL_INTERVAL="${BC_EVAL_INTERVAL:-25}"
BC_EVAL_EPISODES="${BC_EVAL_EPISODES:-5}"
BC_LOG_INTERVAL="${BC_LOG_INTERVAL:-10}"
BC_SEED_BEGIN="${BC_SEED_BEGIN:-1}"
BC_SEED_END="${BC_SEED_END:-10}"

if [ -n "$1" ]; then
    run_layouts=("$1")
else
    run_layouts=("${LAYOUTS[@]}")
fi

for layout in "${run_layouts[@]}"; do
    human_layout=$(get_human_layout "${layout}")
    echo "=== GAMMA BC | layout=${layout}, human_layout=${human_layout} ==="
    for seed in $(seq "${BC_SEED_BEGIN}" "${BC_SEED_END}"); do
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
            --bc_num_epochs "${BC_NUM_EPOCHS}" \
            --bc_batch_size "${BC_BATCH_SIZE}" \
            --lr "${BC_LR}" \
            --human_layout_name "${human_layout}" \
            --save_interval "${BC_SAVE_INTERVAL}" \
            --log_interval "${BC_LOG_INTERVAL}" \
            --use_eval --eval_stochastic --eval_interval "${BC_EVAL_INTERVAL}" --eval_episodes "${BC_EVAL_EPISODES}" \
            --use_wandb \
            --wandb_name "${WANDB_ENTITY}"
    done
done
