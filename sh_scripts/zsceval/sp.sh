#!/bin/bash
# ZSC-EVAL - Self-Play (SP)
# 환경: Overcooked_new + old_dynamics
# Usage: bash sp.sh [layout]

source "$(dirname "$0")/../common_args.sh"
setup_dirs

env="Overcooked"
algo="mappo"
exp="sp"
num_agents=2

if [ -n "$1" ]; then
    run_layouts=("$1")
else
    run_layouts=("${LAYOUTS[@]}")
fi

ulimit -n 65536 2>/dev/null || ulimit -n 4096

for layout in "${run_layouts[@]}"; do
    echo "=== ZSC-EVAL SP | layout=${layout} ==="
    for seed in $(seq ${SEED_BEGIN} ${SEED_END}); do
        echo "  seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python "${ZSCEVAL_TRAIN_DIR}/train_sp.py" \
            --env_name "${env}" \
            --algorithm_name "${algo}" \
            --experiment_name "${exp}" \
            --layout_name "${layout}" \
            --num_agents ${num_agents} \
            --seed ${seed} \
            --n_training_threads 1 \
            --n_rollout_threads 100 \
            --dummy_batch_size 2 \
            --num_mini_batch 1 \
            --episode_length 400 \
            --num_env_steps 1e7 \
            --reward_shaping_horizon 1e8 \
            --overcooked_version new \
            --old_dynamics \
            --ppo_epoch 15 \
            --entropy_coefs 0.2 0.05 0.01 \
            --entropy_coef_horizons 0 5e6 1e7 \
            --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" \
            --use_recurrent_policy \
            --use_proper_time_limits \
            --save_interval 25 \
            --log_interval 10 \
            --use_eval --eval_interval 20 --n_eval_rollout_threads 10 \
            --use_wandb \
            --wandb_name "${WANDB_ENTITY}"
    done
done
