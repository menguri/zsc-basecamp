#!/bin/bash
# GAMMA - Self-Play (SP)
# 환경: Overcooked + old_dynamics
# 알고리즘: MAPPO (GAMMA 기준 하이퍼파라미터)
# Usage: bash sp.sh [layout]
#   layout 생략 시 5개 전체 순차 실행

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

for layout in "${run_layouts[@]}"; do
    echo "=== GAMMA SP | layout=${layout} ==="
    for seed in $(seq ${SEED_BEGIN} ${SEED_END}); do
        echo "  seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python "${GAMMA_TRAIN_DIR}/train_overcooked_sp.py" \
            --env_name ${env} \
            --algorithm_name ${algo} \
            --experiment_name "${exp}" \
            --layout_name ${layout} \
            --num_agents ${num_agents} \
            --seed ${seed} \
            --n_training_threads 1 \
            --n_rollout_threads 100 \
            --num_mini_batch 1 \
            --episode_length 400 \
            --num_env_steps 10000000 \
            --reward_shaping_horizon 5000000 \
            --ppo_epoch 15 \
            --entropy_coef 0.01 \
            --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" \
            --use_recurrent_policy \
            --old_dynamics \
            --save_interval 25 \
            --log_interval 10 \
            --use_eval --eval_stochastic --eval_interval 25 --eval_episodes 5 \
            --use_wandb \
            --wandb_name "${WANDB_ENTITY}"
    done
done
