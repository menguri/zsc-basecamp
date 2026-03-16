#!/bin/bash
# zsc-basecamp/sh_scripts/common_args.sh
# 공통 환경 및 MAPPO 설정 (GAMMA 기준)
# source 해서 사용: source "$(dirname "$0")/../common_args.sh"

# ── 벤치마크 레이아웃 (5종) ─────────────────────────────────────────────────
LAYOUTS=(
    counter_circuit_o_1order
    forced_coordination
    coordination_ring
    cramped_room
    asymmetric_advantages
)

# ── GPU 설정 ─────────────────────────────────────────────────────────────────
# 외부에서 GPU=1 bash sp.sh 처럼 override 가능
GPU=${GPU:-0}

# ── 시드 ─────────────────────────────────────────────────────────────────────
SEED_BEGIN=1
SEED_END=5

# ── 경로 ─────────────────────────────────────────────────────────────────────
BASECAMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAMMA_ROOT="${BASECAMP_DIR}/GAMMA"
ZSCEVAL_ROOT="${BASECAMP_DIR}/ZSC-EVAL"

GAMMA_TRAIN_DIR="${GAMMA_ROOT}/mapbt/scripts/train"
ZSCEVAL_TRAIN_DIR="${ZSCEVAL_ROOT}/zsceval/scripts/overcooked/train"

# ── 결과 경로 정보 ─────────────────────────────────────────────────────────
# GAMMA 결과:   ${GAMMA_ROOT}/mapbt/scripts/results/{env}/{layout}/{algo}/{exp}/
# ZSC-EVAL 결과: ~/ZSC/results/{env}/{layout}/{algo}/{exp}/  (하드코딩됨)
#               심볼릭 링크로 zsc-basecamp/ 아래로 연결: 아래 setup_dirs 함수 참조

# ── Policy Pool 경로 ─────────────────────────────────────────────────────────
# GAMMA: ${GAMMA_ROOT}/mapbt/scripts/overcooked_population/
# ZSC-EVAL: ${BASECAMP_DIR}/policy_pool/  (zsc-basecamp 하위에 직접 생성)
GAMMA_POLICY_POOL="${GAMMA_ROOT}/mapbt/scripts/overcooked_population"
ZSCEVAL_POLICY_POOL="${BASECAMP_DIR}/policy_pool"

# ── Wandb 설정 ────────────────────────────────────────────────────────────────
# wandb_info/wandb_api_key 에서 API key 로드
_WANDB_KEY_FILE="${BASECAMP_DIR}/wandb_info/wandb_api_key"
if [ -f "${_WANDB_KEY_FILE}" ]; then
    export WANDB_API_KEY="$(cat ${_WANDB_KEY_FILE} | tr -d '[:space:]')"
fi

# entity/project 환경변수로 override → train script 내부의 wandb.init() 설정 무시
export WANDB_ENTITY="m-personal-experiment"
export WANDB_PROJECT="zsc-basecamp"

# ── 디렉터리 초기화 (첫 실행 시 한 번만 호출) ─────────────────────────────
setup_dirs() {
    # ZSC-EVAL 결과 디렉터리 심볼릭 링크
    # ~/ZSC/results/ → zsc-basecamp/results/zsceval/
    local zsc_results="${BASECAMP_DIR}/results/zsceval"
    local home_zsc="${HOME}/ZSC/results"
    if [ ! -L "${home_zsc}" ]; then
        mkdir -p "${zsc_results}"
        mkdir -p "${HOME}/ZSC"
        if [ -d "${home_zsc}" ]; then
            echo "[WARNING] ${home_zsc} already exists as a directory. Skipping symlink."
        else
            ln -s "${zsc_results}" "${home_zsc}"
            echo "[setup] Created symlink: ${home_zsc} -> ${zsc_results}"
        fi
    fi

    # Policy pool 디렉터리
    mkdir -p "${ZSCEVAL_POLICY_POOL}"
    mkdir -p "${GAMMA_POLICY_POOL}"
}

# ── ZSC-EVAL prep helper ──────────────────────────────────────────────────────
# Run prep scripts from ZSC-EVAL root so their relative paths (../policy_pool) resolve to zsc-basecamp/policy_pool.
run_zsceval_prep() {
    local script="$1"
    shift
    (
        cd "${ZSCEVAL_ROOT}" || exit 1
        python "zsceval/scripts/prep/${script}" "$@"
    )
}

# Ensure policy config pickles required by population YAMLs exist for a layout.
# This mirrors store_config.sh + mv_policy_config.sh behavior but without hardcoded layouts.
ensure_zsceval_policy_config() {
    local layout="$1"
    local policy_dir="${ZSCEVAL_POLICY_POOL}/${layout}/policy_config"
    local mlp_cfg="${policy_dir}/mlp_policy_config.pkl"
    local rnn_cfg="${policy_dir}/rnn_policy_config.pkl"

    if [ -f "${mlp_cfg}" ] && [ -f "${rnn_cfg}" ]; then
        return 0
    fi

    echo "  [prep] policy_config missing for ${layout}; generating store_config artifacts"
    mkdir -p "${policy_dir}"

    local -a store_args=(
        --env_name Overcooked
        --layout_name "${layout}"
        --num_agents 2
        --seed 1
        --n_training_threads 1
        --n_rollout_threads 50
        --dummy_batch_size 1
        --num_mini_batch 1
        --episode_length 400
        --num_env_steps 1e7
        --reward_shaping_horizon 1e8
        --overcooked_version new
        --ppo_epoch 15
        --entropy_coefs 0.2 0.05 0.001
        --entropy_coef_horizons 0 6e6 1e7
        --save_interval 25
        --log_interval 10
        --use_eval
        --eval_interval 20
        --n_eval_rollout_threads 10
        --use_proper_time_limits
        --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1"
        --use_wandb
    )

    CUDA_VISIBLE_DEVICES=${GPU} python "${ZSCEVAL_TRAIN_DIR}/train_sp.py" \
        --algorithm_name mappo \
        --experiment_name store_config_mlp \
        "${store_args[@]}" \
        --use_recurrent_policy

    CUDA_VISIBLE_DEVICES=${GPU} python "${ZSCEVAL_TRAIN_DIR}/train_sp.py" \
        --algorithm_name rmappo \
        --experiment_name store_config_rnn \
        "${store_args[@]}"

    local mlp_run_root="${HOME}/ZSC/results/Overcooked/${layout}/mappo/store_config_mlp"
    local rnn_run_root="${HOME}/ZSC/results/Overcooked/${layout}/rmappo/store_config_rnn"
    local mlp_run_dir
    local rnn_run_dir
    mlp_run_dir="$(find "${mlp_run_root}" -maxdepth 1 -type d -name 'run*' | sort -V | tail -n 1)"
    rnn_run_dir="$(find "${rnn_run_root}" -maxdepth 1 -type d -name 'run*' | sort -V | tail -n 1)"

    if [ -z "${mlp_run_dir}" ] || [ ! -f "${mlp_run_dir}/policy_config.pkl" ]; then
        echo "  [error] Failed to locate mlp policy_config.pkl under ${mlp_run_root}"
        return 1
    fi
    if [ -z "${rnn_run_dir}" ] || [ ! -f "${rnn_run_dir}/policy_config.pkl" ]; then
        echo "  [error] Failed to locate rnn policy_config.pkl under ${rnn_run_root}"
        return 1
    fi

    cp "${mlp_run_dir}/policy_config.pkl" "${mlp_cfg}"
    cp "${rnn_run_dir}/policy_config.pkl" "${rnn_cfg}"
    echo "  [prep] policy_config ready: ${policy_dir}"
}
