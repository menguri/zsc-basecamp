#!/bin/bash
# ZSC-EVAL - HSP Stage 2
# 전제조건: MEP Stage1(pop=12) + HSP Stage1 pool 준비 완료, Stage2 pop=12
# Usage: bash hsp_stage2.sh [layout]

source "$(dirname "$0")/../common_args.sh"
setup_dirs

population_size=12
s1_population_size=12
entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 2.5e7 5e7"
reward_shaping_horizon="5e7"
num_env_steps="5e7"
pop="hsp"
mep_exp="mep-S1-s12"
hsp_k=6

export POLICY_POOL="${ZSCEVAL_POLICY_POOL}"
ulimit -n 65536 2>/dev/null || ulimit -n 4096
BUILD_POOL_PY="${BASECAMP_DIR}/sh_scripts/zsceval/build_stage1_pool.py"

if [ -n "$1" ]; then run_layouts=("$1"); else run_layouts=("${LAYOUTS[@]}"); fi

for layout in "${run_layouts[@]}"; do
    echo "=== ZSC-EVAL HSP Stage2 | layout=${layout} ==="
    echo "  [prep] build mep stage1 pool from results (${layout}, mep)"
    python "${BUILD_POOL_PY}" \
        --repo_root "${BASECAMP_DIR}" \
        --layout "${layout}" \
        --algo mep \
        --population_size "${s1_population_size}"

    hsp_s1_dir="${ZSCEVAL_POLICY_POOL}/${layout}/hsp/s1/${pop}"
    if [ ! -d "${hsp_s1_dir}" ]; then
        echo "[error] Missing HSP stage1 pool dir: ${hsp_s1_dir}"
        echo "        Prepare HSP stage1 pool first (e.g., hsp*_final_w0_actor.pt files)."
        exit 1
    fi
    hsp_cnt=$(find "${hsp_s1_dir}" -maxdepth 1 -type f -name 'hsp*_final_w0_actor.pt' | wc -l)
    if [ "${hsp_cnt}" -eq 0 ]; then
        echo "[error] No HSP stage1 actor files found in ${hsp_s1_dir}"
        echo "        Expected pattern: hsp*_final_w0_actor.pt"
        exit 1
    fi

    ensure_zsceval_policy_config "${layout}"
    echo "  [prep] gen_hsp_S2_ymls.py -l ${layout} -s ${s1_population_size} -S ${population_size} -k ${hsp_k}"
    run_zsceval_prep gen_hsp_S2_ymls.py -l "${layout}" -s "${s1_population_size}" -S "${population_size}" -k "${hsp_k}"

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
