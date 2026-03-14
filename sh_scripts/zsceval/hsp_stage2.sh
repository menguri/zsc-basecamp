#!/bin/bash
# ZSC-EVAL - HSP Stage 2
# 전제조건: MEP Stage1 population 완료
# Usage: bash hsp_stage2.sh [layout]

source "$(dirname "$0")/../common_args.sh"
setup_dirs

population_size=12
entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 2.5e7 5e7"
reward_shaping_horizon="5e7"
num_env_steps="5e7"
pop="hsp"
mep_exp="mep-S1-s5"

export POLICY_POOL="${ZSCEVAL_POLICY_POOL}"
ulimit -n 65536 2>/dev/null || ulimit -n 4096

if [ -n "$1" ]; then run_layouts=("$1"); else run_layouts=("${LAYOUTS[@]}"); fi

for layout in "${run_layouts[@]}"; do
    echo "=== ZSC-EVAL HSP Stage2 | layout=${layout} ==="
    for seed in $(seq ${SEED_BEGIN} ${SEED_END}); do
        echo "  seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} python "${ZSCEVAL_TRAIN_DIR}/train_adaptive.py" \
            --env_name Overcooked --algorithm_name adaptive \
            --experiment_name "hsp-S2-s${population_size}" \
            --layout_name "${layout}" --num_agents 2 \
            --seed ${seed} --n_training_threads 1 \
            --n_rollout_threads 100 --dummy_batch_size 1 \
            --num_mini_batch 1 --episode_length 400 \
            --num_env_steps ${num_env_steps} \
            --reward_shaping_horizon ${reward_shaping_horizon} \
            --overcooked_version new \
            --ppo_epoch 15 \
            --entropy_coefs ${entropy_coefs} \
            --entropy_coef_horizons ${entropy_coef_horizons} \
            --stage 2 \
            --population_size ${population_size} \
            --adaptive_agent_name hsp_adaptive --use_agent_policy_id \
            --population_yaml_path "${ZSCEVAL_POLICY_POOL}/${layout}/hsp/s2/train-s${population_size}-${pop}_${mep_exp}-${seed}.yml" \
            --use_proper_time_limits \
            --save_interval 25 --log_interval 1 \
            --use_eval --eval_interval 20 \
            --n_eval_rollout_threads $((population_size * 2)) --eval_episodes 5 \
            --use_wandb --wandb_name "${WANDB_ENTITY}"
    done
done
