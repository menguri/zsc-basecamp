#!/bin/bash
# ZSC-EVAL - MEP Stage 2 (Adaptive Agent)
# 전제조건: mep_stage1.sh 완료
# Usage: bash mep_stage2.sh [layout]

source "$(dirname "$0")/../common_args.sh"
setup_dirs

env="Overcooked"
algo="mep"
population_size=12

# pop_size=12 기준
entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 2.5e7 5e7"
reward_shaping_horizon="5e7"
num_env_steps="5e7"
pop="mep-S1-s5"
mep_prioritized_alpha=0.5

export POLICY_POOL="${ZSCEVAL_POLICY_POOL}"
ulimit -n 65536 2>/dev/null || ulimit -n 4096

if [ -n "$1" ]; then run_layouts=("$1"); else run_layouts=("${LAYOUTS[@]}"); fi

for layout in "${run_layouts[@]}"; do
    echo "=== ZSC-EVAL MEP Stage2 | layout=${layout} ==="
    for seed in $(seq ${SEED_BEGIN} ${SEED_END}); do
        echo "  seed=${seed}"
        python "${ZSCEVAL_TRAIN_DIR}/train_adaptive.py" \
            --env_name "${env}" --algorithm_name "${algo}" \
            --experiment_name "mep-S2-s${population_size}" \
            --layout_name "${layout}" --num_agents 2 \
            --seed ${seed} --n_training_threads 1 \
            --n_rollout_threads 100 --dummy_batch_size 1 \
            --num_mini_batch 1 --episode_length 400 \
            --num_env_steps ${num_env_steps} \
            --reward_shaping_horizon ${reward_shaping_horizon} \
            --overcooked_version new --old_dynamics \
            --ppo_epoch 15 \
            --entropy_coefs ${entropy_coefs} \
            --entropy_coef_horizons ${entropy_coef_horizons} \
            --stage 2 \
            --mep_use_prioritized_sampling \
            --mep_prioritized_alpha ${mep_prioritized_alpha} \
            --population_size ${population_size} \
            --adaptive_agent_name mep_adaptive --use_agent_policy_id \
            --population_yaml_path "${ZSCEVAL_POLICY_POOL}/${layout}/mep/s2/train-s${population_size}-${pop}-${seed}.yml" \
            --use_proper_time_limits \
            --save_interval 25 --log_interval 1 \
            --use_eval --eval_interval 20 \
            --n_eval_rollout_threads $((population_size * 2)) --eval_episodes 5 \
            --use_wandb --wandb_name "${WANDB_ENTITY}"
    done
done
