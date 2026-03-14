#!/bin/bash
# ZSC-EVAL - E3T (Efficient End-to-End Training)
# Usage: bash e3t.sh [layout]

source "$(dirname "$0")/../common_args.sh"
setup_dirs

ulimit -n 65536 2>/dev/null || ulimit -n 4096

if [ -n "$1" ]; then run_layouts=("$1"); else run_layouts=("${LAYOUTS[@]}"); fi

for layout in "${run_layouts[@]}"; do
    echo "=== ZSC-EVAL E3T | layout=${layout} ==="
    for seed in $(seq ${SEED_BEGIN} ${SEED_END}); do
        echo "  seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python "${ZSCEVAL_TRAIN_DIR}/train_e3t.py" \
            --env_name Overcooked --algorithm_name e3t \
            --experiment_name e3t \
            --layout_name "${layout}" --num_agents 2 \
            --seed ${seed} --n_training_threads 1 \
            --n_rollout_threads 100 --dummy_batch_size 2 \
            --num_mini_batch 1 --episode_length 400 \
            --num_env_steps 1e7 --reward_shaping_horizon 1e8 \
            --overcooked_version new --old_dynamics \
            --ppo_epoch 15 \
            --entropy_coefs 0.2 0.05 0.01 --entropy_coef_horizons 0 6e6 1e7 \
            --share_policy --random_index \
            --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" \
            --use_recurrent_policy \
            --epsilon 0.25 --weights_copy_factor 0.1 \
            --use_proper_time_limits \
            --save_interval 25 --log_interval 10 \
            --use_eval --eval_interval 20 --n_eval_rollout_threads 10 \
            --use_wandb --wandb_name "${WANDB_ENTITY}"
    done
done
