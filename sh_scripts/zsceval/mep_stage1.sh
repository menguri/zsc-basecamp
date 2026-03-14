#!/bin/bash
# ZSC-EVAL - MEP Stage 1 (Population 생성, seed=1 고정)
# Usage: bash mep_stage1.sh [layout]

source "$(dirname "$0")/../common_args.sh"
setup_dirs

env="Overcooked"
algo="mep"
population_size=12
train_batch=125

export POLICY_POOL="${ZSCEVAL_POLICY_POOL}"
ulimit -n 65536 2>/dev/null || ulimit -n 4096

if [ -n "$1" ]; then run_layouts=("$1"); else run_layouts=("${LAYOUTS[@]}"); fi

for layout in "${run_layouts[@]}"; do
    echo "=== ZSC-EVAL MEP Stage1 | layout=${layout}, pop=${population_size} ==="
    ensure_zsceval_policy_config "${layout}"
    echo "  [prep] gen_pop_ymls.py ${layout} mep -s ${population_size}"
    run_zsceval_prep gen_pop_ymls.py "${layout}" mep -s "${population_size}"

    CUDA_VISIBLE_DEVICES=${GPU} python "${ZSCEVAL_TRAIN_DIR}/train_mep.py" \
        --env_name "${env}" --algorithm_name "${algo}" \
        --experiment_name "mep-S1-s${population_size}" \
        --layout_name "${layout}" --num_agents 2 \
        --seed 1 --n_training_threads 1 \
        --train_env_batch ${train_batch} --n_rollout_threads ${train_batch} --dummy_batch_size 1 \
        --num_mini_batch 1 --episode_length 400 \
        --num_env_steps 1e7 --reward_shaping_horizon 1e8 \
        --overcooked_version new \
        --ppo_epoch 15 \
        --entropy_coefs 0.2 0.05 0.01 --entropy_coef_horizons 0 6e6 1e7 \
        --stage 1 --mep_entropy_alpha 0.01 \
        --population_size ${population_size} --adaptive_agent_name mep_adaptive \
        --population_yaml_path "${ZSCEVAL_POLICY_POOL}/${layout}/mep/s1/train-s${population_size}.yml" \
        --use_proper_time_limits \
        --save_interval 25 --log_interval 1 \
        --use_eval --eval_interval 20 \
        --n_eval_rollout_threads $((population_size * 2)) --eval_episodes 10 \
        --use_wandb --wandb_name "${WANDB_ENTITY}"
done
