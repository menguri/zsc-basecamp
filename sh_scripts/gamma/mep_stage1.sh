#!/bin/bash
# GAMMA - MEP Stage 1 (Population 생성)
# 환경: Overcooked + old_dynamics
# Usage: bash mep_stage1.sh [layout]
# NOTE: Stage1은 seed=1 고정

source "$(dirname "$0")/../common_args.sh"
setup_dirs

env="Overcooked"
algo="mep"
exp="mep-S1"
num_agents=2
population_size=12

export POLICY_POOL="${GAMMA_POLICY_POOL}"

if [ -n "$1" ]; then
    run_layouts=("$1")
else
    run_layouts=("${LAYOUTS[@]}")
fi

for layout in "${run_layouts[@]}"; do
    echo "=== GAMMA MEP Stage1 | layout=${layout}, pop_size=${population_size} ==="
    seed=1
    CUDA_VISIBLE_DEVICES=${GPU} python "${GAMMA_TRAIN_DIR}/train_overcooked_mep.py" \
        --env_name "${env}" \
        --algorithm_name "${algo}" \
        --experiment_name "${exp}" \
        --layout_name "${layout}" \
        --num_agents ${num_agents} \
        --seed ${seed} \
        --n_training_threads 1 \
        --n_rollout_threads 100 \
        --train_env_batch 100 \
        --num_mini_batch 1 \
        --episode_length 400 \
        --num_env_steps 10000000 \
        --reward_shaping_horizon 100000000 \
        --ppo_epoch 15 \
        --entropy_coef 0.01 \
        --mep_entropy_alpha 0.01 \
        --stage 1 \
        --population_size ${population_size} \
        --adaptive_agent_name mep_adaptive \
        --population_yaml_path "${GAMMA_POLICY_POOL}/config/${layout}/mep/s1/train.yml" \
        --old_dynamics \
        --save_interval 50 \
        --log_interval 1 \
        --use_wandb \
        --wandb_name "${WANDB_ENTITY}"
done
