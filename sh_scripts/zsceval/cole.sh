#!/bin/bash
# ZSC-EVAL - COLE (Cooperative Open-ended Learning)
# Usage: bash cole.sh [layout]

source "$(dirname "$0")/../common_args.sh"
setup_dirs

population_size=25
entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 2.5e7 5e7"
reward_shaping_horizon="5e7"
num_env_steps="5e7"
prioritized_alpha=1.5
generation_interval=20

export POLICY_POOL="${ZSCEVAL_POLICY_POOL}"
export EVOLVE_ACTOR_POOL="${HOME}/ZSC/tmp"
ulimit -n 65536 2>/dev/null || ulimit -n 4096

if [ -n "$1" ]; then run_layouts=("$1"); else run_layouts=("${LAYOUTS[@]}"); fi

for layout in "${run_layouts[@]}"; do
    echo "=== ZSC-EVAL COLE | layout=${layout}, pop=${population_size} ==="
    for seed in $(seq ${SEED_BEGIN} ${SEED_END}); do
        echo "  seed=${seed}"
        python "${ZSCEVAL_TRAIN_DIR}/train_cole.py" \
            --env_name Overcooked --algorithm_name cole \
            --experiment_name "cole-S2-s${population_size}" \
            --layout_name "${layout}" --num_agents 2 \
            --seed ${seed} --n_training_threads 1 \
            --n_rollout_threads 125 --dummy_batch_size 1 \
            --num_mini_batch 1 --episode_length 400 \
            --num_env_steps ${num_env_steps} \
            --reward_shaping_horizon ${reward_shaping_horizon} \
            --overcooked_version new --old_dynamics \
            --ppo_epoch 15 \
            --entropy_coefs ${entropy_coefs} \
            --entropy_coef_horizons ${entropy_coef_horizons} \
            --stage 2 \
            --population_size ${population_size} \
            --adaptive_agent_name cole_adaptive --use_agent_policy_id \
            --population_yaml_path "${ZSCEVAL_POLICY_POOL}/${layout}/cole/s1/train-s${population_size}-${seed}.yml" \
            --num_generation $((population_size * 2)) \
            --generation_interval ${generation_interval} \
            --prioritized_alpha ${prioritized_alpha} \
            --algorithm_type evolution \
            --use_proper_time_limits \
            --save_interval 25 --log_interval 1 \
            --use_eval --eval_interval 20 \
            --n_eval_rollout_threads $((population_size * 2 + 1)) --eval_episodes 5 \
            --use_wandb --wandb_name "${WANDB_ENTITY}"
    done
done
