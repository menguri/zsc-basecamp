#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GAMMA_DIR="${BASE_DIR}/GAMMA"
ZSCEVAL_DIR="${BASE_DIR}/ZSC-EVAL"
GAMMA_VENV="${BASE_DIR}/.venv-gamma"
ZSCEVAL_VENV="${BASE_DIR}/.venv-zsceval"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[error] Python not found: ${PYTHON_BIN}"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[error] uv not found. Install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

PYTHON_PATH="$(command -v "${PYTHON_BIN}")"
PY_VER="$("${PYTHON_PATH}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[info] Using Python ${PY_VER} via ${PYTHON_BIN}"

if [ ! -d "${GAMMA_DIR}" ] || [ ! -d "${ZSCEVAL_DIR}" ]; then
    echo "[error] Run this script inside zsc-basecamp."
    exit 1
fi

create_venv() {
    local venv_path="$1"
    local setup_log
    setup_log="$(mktemp)"

    if [ "${FORCE_RECREATE:-0}" = "1" ] && [ -d "${venv_path}" ]; then
        echo "[setup] Recreating ${venv_path}"
        rm -rf "${venv_path}"
    fi

    if [ -d "${venv_path}" ] && [ ! -f "${venv_path}/bin/activate" ]; then
        echo "[setup] Found incomplete venv at ${venv_path}. Recreating."
        rm -rf "${venv_path}"
    fi

    if [ ! -d "${venv_path}" ]; then
        echo "[setup] Creating ${venv_path}"
        if ! "${PYTHON_BIN}" -m venv "${venv_path}" >"${setup_log}" 2>&1; then
            echo "[warn] python -m venv failed, trying uv venv fallback."
            tail -n 8 "${setup_log}" || true
            rm -rf "${venv_path}"
            uv venv --python "${PYTHON_PATH}" --no-managed-python --seed "${venv_path}"
        fi
    else
        echo "[setup] Reusing ${venv_path}"
    fi

    if [ ! -x "${venv_path}/bin/pip" ]; then
        echo "[error] ${venv_path} does not contain pip. Recreate with FORCE_RECREATE=1."
        rm -f "${setup_log}"
        exit 1
    fi

    rm -f "${setup_log}"
}

install_gamma() {
    echo "[setup] Installing GAMMA dependencies"
    uv pip install --python "${GAMMA_VENV}/bin/python" --upgrade pip setuptools wheel
    uv pip install --python "${GAMMA_VENV}/bin/python" -r "${GAMMA_DIR}/requirements.txt"
    uv pip install --python "${GAMMA_VENV}/bin/python" -e "${GAMMA_DIR}"
}

install_zsceval() {
    echo "[setup] Installing ZSC-EVAL dependencies"
    uv pip install --python "${ZSCEVAL_VENV}/bin/python" --upgrade pip setuptools wheel
    (
        cd "${ZSCEVAL_DIR}"
        uv pip install --python "${ZSCEVAL_VENV}/bin/python" -r requirements.txt
    )
}

create_venv "${GAMMA_VENV}"
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "[setup] SKIP_INSTALL=1 -> skipping package installation for GAMMA"
else
    install_gamma
fi

create_venv "${ZSCEVAL_VENV}"
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "[setup] SKIP_INSTALL=1 -> skipping package installation for ZSC-EVAL"
else
    install_zsceval
fi

cat <<EOF
[done] Virtual environments are ready.

Use these commands:
  source ${GAMMA_VENV}/bin/activate
  source ${ZSCEVAL_VENV}/bin/activate
EOF
