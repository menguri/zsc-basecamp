#!/bin/bash
# zsc-basecamp/sh_scripts/common_args.sh
# 공통 환경 및 MAPPO 설정 (GAMMA 기준)
# source 해서 사용: source "$(dirname "$0")/../common_args.sh"

# ── 벤치마크 레이아웃 (5종) ─────────────────────────────────────────────────
LAYOUTS=(
    cramped_room
    asymmetric_advantages
    coordination_ring
    forced_coordination
    counter_circuit_o_1order
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
