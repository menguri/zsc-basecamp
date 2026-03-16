#!/bin/bash
# sh_scripts/ph2/ph2.sh
# PH2 training script — 30M total steps (spec 15M + ind 15M)
#
# Usage:
#   bash sh_scripts/ph2/ph2.sh [LAYOUT]
#   GPU=1 bash sh_scripts/ph2/ph2.sh cramped_room
#   GPUS=0,1,2,3,4 bash sh_scripts/ph2/ph2.sh
#
# Blocked-state penalty: omega * exp(-sigma * L2_dist)
# Blocked-state sampling: V_gap softmax with temperature beta
# Spec/ind pred: separate by default (--ph2_share_pred to share)

source "$(dirname "$0")/../common_args.sh"
setup_dirs

PH2_DIR="${BASECAMP_DIR}/ph2"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ph2][error] '${PYTHON_BIN}' not found. Activate your virtualenv or set PYTHON_BIN explicitly."
    exit 127
fi

PH2_FIXED_IND_PROB=${PH2_FIXED_IND_PROB:-0.5}
PH2_NUM_BLOCKED_SLOTS=${PH2_NUM_BLOCKED_SLOTS:-1}
PH2_BLOCKED_POOL_SIZE=${PH2_BLOCKED_POOL_SIZE:-200}
PH2_OMEGA=${PH2_OMEGA:-10.0}
PH2_SIGMA=${PH2_SIGMA:-2.0}
PH2_EPSILON=${PH2_EPSILON:-0.2}

PH2_VGAP_BETA=${PH2_VGAP_BETA:-1.0}
PH2_VGAP_BETA_SCHEDULE_ENABLED=${PH2_VGAP_BETA_SCHEDULE_ENABLED:-1}
PH2_VGAP_BETA_START=${PH2_VGAP_BETA_START:-0.0}
PH2_VGAP_BETA_END=${PH2_VGAP_BETA_END:-1.0}
PH2_VGAP_BETA_HORIZON_ENV_STEPS=${PH2_VGAP_BETA_HORIZON_ENV_STEPS:--1}
PH2_ENTROPY_COEFS=${PH2_ENTROPY_COEFS:-"0.01 0.01"}
PH2_ENTROPY_COEF_HORIZONS=${PH2_ENTROPY_COEF_HORIZONS:-"0 30000000"}
PH2_CNN_LAYERS_PARAMS=${PH2_CNN_LAYERS_PARAMS:-"32,3,1,1 64,3,1,1 32,3,1,1"}
PH2_DISABLE_WANDB=${PH2_DISABLE_WANDB:-0}
PH2_LOG_INTERVAL=${PH2_LOG_INTERVAL:-1}
PH2_EVAL_INTERVAL=${PH2_EVAL_INTERVAL:-25}

PH2_SHARE_PRED=${PH2_SHARE_PRED:-0}

if [ -n "$1" ]; then
    run_layouts=("$1")
else
    run_layouts=("${LAYOUTS[@]}")
fi

GPU_LIST_RAW="${GPUS:-${GPU:-0}}"
GPU_LIST_RAW="${GPU_LIST_RAW// /}"
if [ -z "${GPU_LIST_RAW}" ]; then
    GPU_LIST_RAW="0"
fi
IFS=',' read -r -a GPU_LIST <<< "${GPU_LIST_RAW}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    GPU_LIST=("0")
fi

run_one_seed() {
    local layout="$1"
    local seed="$2"
    local gpu="$3"
    read -r -a entropy_coefs_arr <<< "${PH2_ENTROPY_COEFS}"
    read -r -a entropy_horizons_arr <<< "${PH2_ENTROPY_COEF_HORIZONS}"

    local -a cmd=(
      "${PYTHON_BIN}" "${PH2_DIR}/scripts/train/train_ph2.py"
        --env_name "Overcooked" \
        --algorithm_name "ph2" \
        --experiment_name "ph2" \
        --layout_name "${layout}" \
        --num_agents 2 \
        --seed ${seed} \
        --n_rollout_threads 100 \
        --n_eval_rollout_threads 5 \
        --num_env_steps 15000000 \
        --episode_length 400 \
        --hidden_size 64 \
        --cnn_layers_params "${PH2_CNN_LAYERS_PARAMS}" \
        --lr 5e-4 \
        --critic_lr 5e-4 \
        --ppo_epoch 15 \
        --num_mini_batch 2 \
        --data_chunk_length 10 \
        --entropy_coefs "${entropy_coefs_arr[@]}" \
        --entropy_coef_horizons "${entropy_horizons_arr[@]}" \
        --reward_shaping_horizon 2500000 \
        --use_policy_vhead \
        --ph2_fixed_ind_prob "${PH2_FIXED_IND_PROB}" \
        --ph2_use_partner_pred \
        --ph2_history_len 5 \
        --ph2_pred_loss_coef 1.0 \
        --ph2_spec_use_blocked \
        --ph2_num_blocked_slots "${PH2_NUM_BLOCKED_SLOTS}" \
        --ph2_blocked_pool_size "${PH2_BLOCKED_POOL_SIZE}" \
        --ph2_blocked_penalty_omega "${PH2_OMEGA}" \
        --ph2_blocked_penalty_sigma "${PH2_SIGMA}" \
        --ph2_vgap_beta "${PH2_VGAP_BETA}" \
        --ph2_vgap_beta_start "${PH2_VGAP_BETA_START}" \
        --ph2_vgap_beta_end "${PH2_VGAP_BETA_END}" \
        --ph2_vgap_beta_horizon_env_steps "${PH2_VGAP_BETA_HORIZON_ENV_STEPS}" \
        --ph2_epsilon "${PH2_EPSILON}" \
        --save_interval 25 \
        --log_interval "${PH2_LOG_INTERVAL}" \
        --eval_interval "${PH2_EVAL_INTERVAL}" \
        --use_eval \
        --wandb_project "zsc-basecamp" \
        --wandb_entity "m-personal-experiment" \
        --wandb_tags "ph2" "${layout}" \
        --use_linear_lr_decay
    )

    if [ "${PH2_SHARE_PRED}" = "1" ]; then
        cmd+=(--ph2_share_pred)
    fi
    if [ "${PH2_VGAP_BETA_SCHEDULE_ENABLED}" != "1" ]; then
        cmd+=(--ph2_no_vgap_beta_schedule)
    fi
    # In this codebase, --use_wandb is store_false (passing it disables wandb).
    if [ "${PH2_DISABLE_WANDB}" = "1" ]; then
        cmd+=(--use_wandb)
    fi

    echo "[ph2][start] layout=${layout} seed=${seed} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${BASECAMP_DIR}:${PYTHONPATH}" "${cmd[@]}"
    local rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "[ph2][fail] layout=${layout} seed=${seed} gpu=${gpu} rc=${rc}"
        return "${rc}"
    fi
    echo "[ph2][done] layout=${layout} seed=${seed} gpu=${gpu}"
}

echo "[ph2] layouts=${run_layouts[*]}  gpus=${GPU_LIST[*]}  seeds=${SEED_BEGIN}..${SEED_END}"

for layout in "${run_layouts[@]}"; do
    echo "[ph2] ===== layout=${layout} ====="
    pids=()
    num_gpus=${#GPU_LIST[@]}

    for worker_idx in "${!GPU_LIST[@]}"; do
        gpu="${GPU_LIST[worker_idx]}"
        (
            for seed in $(seq $((SEED_BEGIN + worker_idx)) "${num_gpus}" "${SEED_END}"); do
                run_one_seed "${layout}" "${seed}" "${gpu}" || exit 1
            done
        ) &
        pids+=($!)
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done

    if [ "${failed}" -ne 0 ]; then
        echo "[ph2][error] layout=${layout} failed; stopping."
        exit 1
    fi
    echo "[ph2] ===== layout=${layout} complete ====="
done
