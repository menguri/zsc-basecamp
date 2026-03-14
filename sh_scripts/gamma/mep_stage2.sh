#!/bin/bash
# GAMMA - MEP Stage 2 (Adaptive Agent 학습)
# 환경: Overcooked + old_dynamics
# Usage: bash mep_stage2.sh [layout]
# 전제조건: mep_stage1.sh 완료 후 POLICY_POOL에 population이 있어야 함

source "$(dirname "$0")/../common_args.sh"
setup_dirs

env="Overcooked"
algo="mep"
exp="mep-S2"
num_agents=2
population_size=36

export POLICY_POOL="${GAMMA_POLICY_POOL}"

if [ -n "$1" ]; then
    run_layouts=("$1")
else
    run_layouts=("${LAYOUTS[@]}")
fi

for layout in "${run_layouts[@]}"; do
    echo "=== GAMMA MEP Stage2 | layout=${layout}, pop_size=${population_size} ==="
    for seed in $(seq ${SEED_BEGIN} ${SEED_END}); do
        echo "  seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python "${GAMMA_TRAIN_DIR}/train_overcooked_adaptive.py" \
            --env_name "${env}" \
            --algorithm_name "${algo}" \
            --experiment_name "${exp}" \
            --layout_name "${layout}" \
            --num_agents ${num_agents} \
            --seed ${seed} \
            --n_training_threads 1 \
            --n_rollout_threads 200 \
            --train_env_batch 1 \
            --num_mini_batch 1 \
            --episode_length 400 \
            --num_env_steps 100000000 \
            --reward_shaping_horizon 100000000 \
            --ppo_epoch 15 \
            --stage 2 \
            --population_size ${population_size} \
            --adaptive_agent_name mep_adaptive \
            --use_agent_policy_id \
            --population_yaml_path "${GAMMA_POLICY_POOL}/config/${layout}/mep/s2/train.yml" \
            --old_dynamics \
            --save_interval 20 \
            --log_interval 10 \
            --use_wandb \
            --wandb_name "${WANDB_ENTITY}"
    done
done
