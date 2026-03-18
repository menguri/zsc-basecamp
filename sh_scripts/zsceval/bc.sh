#!/bin/bash
# sh_scripts/zsceval/bc.sh
# ZSC-EVAL - Behavioral Cloning (BC) using GAMMA's human demonstration data
#
# 훈련 환경: overcooked_new  (ph2/sp/mep 등 다른 에이전트들과 동일)
# 인간 데이터: GAMMA/mapbt/envs/overcooked/dataset/formatted_human_trajectories/{layout}.pickle
#             (human_aware_rl.static 에서 GAMMA bc.sh로 미리 캐시한 것 사용)
#
# 출력:
#   학습 결과 → ~/ZSC/results/Overcooked/{layout}/bc/bc/run{N}/
#   eval 복사 → eval/models/Overcooked/{layout}/bc/run{seed}/
#               (xp_eval.py의 _candidate_model_dirs가 바로 읽는 경로)
#
# Usage:
#   bash sh_scripts/zsceval/bc.sh [layout]
#   GPU=1 BC_NUM_EPOCHS=200 bash sh_scripts/zsceval/bc.sh cramped_room

source "$(dirname "$0")/../common_args.sh"
setup_dirs

# ── BC 하이퍼파라미터 (환경변수로 override 가능) ────────────────────────────
BC_NUM_EPOCHS="${BC_NUM_EPOCHS:-400}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-128}"
BC_LR="${BC_LR:-1e-2}"
BC_SAVE_INTERVAL="${BC_SAVE_INTERVAL:-100}"
BC_LOG_INTERVAL="${BC_LOG_INTERVAL:-10}"
BC_SEED_BEGIN="${BC_SEED_BEGIN:-1}"
BC_SEED_END="${BC_SEED_END:-5}"

# ── 레이아웃별 인간 데이터 이름 매핑 ──────────────────────────────────────
# counter_circuit_o_1order의 인간 데이터는 random3로 저장되어 있음
get_human_layout() {
    case $1 in
        counter_circuit_o_1order) echo "random3" ;;
        forced_coordination) echo "random0" ;;
        *) echo "$1" ;;
    esac
}

ulimit -n 65536 2>/dev/null || ulimit -n 4096

if [ -n "$1" ]; then
    run_layouts=("$1")
else
    run_layouts=("${LAYOUTS[@]}")
fi

for layout in "${run_layouts[@]}"; do
    human_layout=$(get_human_layout "${layout}")
    eval_models_dir="${BASECAMP_DIR}/eval/models/Overcooked/${layout}/bc"
    echo "=== ZSC-EVAL BC | layout=${layout}, human_layout=${human_layout} ==="

    for seed in $(seq "${BC_SEED_BEGIN}" "${BC_SEED_END}"); do
        echo "  seed=${seed}"

        CUDA_VISIBLE_DEVICES=${GPU} python "${ZSCEVAL_TRAIN_DIR}/train_bc.py" \
            --env_name Overcooked \
            --algorithm_name rmappo \
            --experiment_name bc \
            --layout_name "${layout}" \
            --num_agents 2 \
            --seed "${seed}" \
            --n_training_threads 1 \
            --n_rollout_threads 1 \
            --dummy_batch_size 1 \
            --num_mini_batch 1 \
            --episode_length 400 \
            --num_env_steps 10000000 \
            --reward_shaping_horizon 0 \
            --overcooked_version new \
            --human_layout_name "${human_layout}" \
            --human_data_refresh \
            --bc_num_epochs "${BC_NUM_EPOCHS}" \
            --bc_batch_size "${BC_BATCH_SIZE}" \
            --lr "${BC_LR}" \
            --save_interval "${BC_SAVE_INTERVAL}" \
            --log_interval "${BC_LOG_INTERVAL}" \
            --use_wandb \
            --wandb_name "${WANDB_ENTITY}"

        # ── 훈련 결과를 eval/models/ 로 복사 ────────────────────────────────
        # train_bc.py가 --use_wandb 없이 (ZSC-EVAL에서 --use_wandb = store_false)
        # 로컬 run 디렉터리에 저장함: ~/ZSC/results/Overcooked/{layout}/bc/bc/run{N}/
        zsc_results="${BASECAMP_DIR}/results/zsceval"
        run_root="${zsc_results}/Overcooked/${layout}/bc/bc"

        if [ -d "${run_root}" ]; then
            # 가장 최근에 수정된 run 디렉터리 선택
            latest_run=$(ls -td "${run_root}"/run* 2>/dev/null | head -n 1)
            if [ -n "${latest_run}" ]; then
                target_dir="${eval_models_dir}/run${seed}"
                mkdir -p "${target_dir}"

                # policy_config.pkl 복사 (루트에 저장됨)
                if [ -f "${latest_run}/policy_config.pkl" ]; then
                    cp "${latest_run}/policy_config.pkl" "${target_dir}/"
                fi

                # actor 복사 (마지막 epoch)
                if [ -f "${latest_run}/models/actor.pt" ]; then
                    cp "${latest_run}/models/actor.pt" "${target_dir}/actor.pt"
                    echo "  → eval 복사 완료: ${target_dir}"
                else
                    echo "  [WARNING] actor.pt를 찾지 못했습니다: ${latest_run}/models/"
                fi
                echo "    actor.pt: $(ls -lh "${target_dir}/actor.pt" 2>/dev/null | awk '{print $5}' || echo 'not found')"
            else
                echo "  [WARNING] run 디렉터리를 찾지 못했습니다: ${run_root}"
            fi
        else
            echo "  [WARNING] 결과 디렉터리 없음: ${run_root}"
        fi
    done

    echo "=== ${layout} BC 완료 ==="
    echo "  eval 모델 경로: ${eval_models_dir}/"
done
