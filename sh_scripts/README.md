# ZSC Benchmark Shell Scripts

GAMMA와 ZSC-EVAL 알고리즘을 동일 환경 조건에서 실행하기 위한 스크립트 모음.

---

## 디렉터리 구조

```
sh_scripts/
├── common_args.sh          # 공통 설정 (레이아웃, 시드, 경로, wandb)
├── gamma/
│   ├── sp.sh               # Self-Play
│   ├── bc.sh               # Behavior Cloning
│   ├── mep_stage1.sh       # MEP Population 생성
│   └── mep_stage2.sh       # MEP Adaptive Agent
└── zsceval/
    ├── sp.sh               # Self-Play
    ├── fcp_stage2.sh       # FCP Adaptive Agent
    ├── mep_stage1.sh       # MEP Population 생성
    ├── mep_stage2.sh       # MEP Adaptive Agent
    ├── traj_stage1.sh      # TrajeDi Population 생성
    ├── traj_stage2.sh      # TrajeDi Adaptive Agent
    ├── hsp_stage2.sh       # HSP Adaptive Agent
    ├── cole.sh             # COLE
    └── e3t.sh              # E3T
```

---

## GAMMA 가상환경 세팅 (Python 3.12 + uv)

아래 절차는 현재 사용 중인 GAMMA 실행 환경 기준입니다.

```bash
cd zsc-basecamp

# 1) Python 3.12 가상환경 생성
uv venv .zsc-gamma --python 3.12

# 2) 가상환경 활성화
source .zsc-gamma/bin/activate

# 3) PyTorch 설치 (CUDA 12.1)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4) 필수 라이브러리 설치 (NumPy < 2.0)
uv pip install "numpy<2.0" absl-py dill scipy tqdm gym pettingzoo ipython pygame ipywidgets opencv-python \
wandb icecream setproctitle seaborn tensorboardX psutil slackweb pyastar2d einops h5py

# 5) GAMMA(mapbt) 설치
uv pip install -e ./GAMMA

# 6) overcooked_ai_py 설치
uv pip install -e ./GAMMA/mapbt/envs/overcooked/overcooked_berkeley
```

실행 예시:

```bash
cd zsc-basecamp
source .zsc-gamma/bin/activate
bash sh_scripts/gamma/sp.sh
```

---

## ZSC-EVAL 가상환경 세팅 (Python 3.9 + uv)

`ZSC-EVAL/environment.yml` 기반 설정을 `uv`로 구성한 절차입니다.

```bash
cd zsc-basecamp

# 1) Python 3.9 가상환경 생성
uv venv .zsc-zsceval --python 3.9
source .zsc-zsceval/bin/activate

# 2) PyTorch + CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3) ZSC-EVAL 의존성 설치 (requirements.txt 안에 -e . 포함)
cd ZSC-EVAL
uv pip install -r requirements.txt

# 4) environment.yml의 uwsgi 대응
uv pip install uwsgi

cd ..
```

실행 예시:

```bash
cd zsc-basecamp
source .zsc-zsceval/bin/activate
bash sh_scripts/zsceval/sp.sh
```

---

## GPU 지정

`common_args.sh`에서 `GPU=0` (기본값)으로 설정됩니다.
실행 시 환경변수로 override:

```bash
# GPU 0 사용 (기본)
bash sh_scripts/gamma/sp.sh

# GPU 1 사용
GPU=1 bash sh_scripts/gamma/sp.sh

# GPU 2에서 특정 레이아웃만
GPU=2 bash sh_scripts/zsceval/sp.sh cramped_room
```

`common_args.sh`의 기본값을 영구 변경하려면:
```bash
# common_args.sh 내 GPU 라인 수정
GPU=${GPU:-1}   # 기본값을 1로 변경
```

---

## 사전 준비

### 1. ZSC-EVAL `old_dynamics` 수정 ⚠️ 필수

ZSC-EVAL의 모든 train 스크립트는 layout명을 기반으로 `old_dynamics`를 자동 결정합니다.
벤치마크 레이아웃 5종은 기본적으로 `old_dynamics=False`로 설정되므로,
GAMMA와 동일한 환경을 만들려면 **`overcooked_config.py`를 수정해야 합니다.**

```python
# ZSC-EVAL/zsceval/overcooked_config.py
OLD_LAYOUTS = [
    "random0", "random0_medium", "random1", "random3",
    "small_corridor", "unident_s",
    # ↓ 아래 5개 추가
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]
```

### 2. 초기 디렉터리 설정

각 스크립트는 `setup_dirs`를 자동 호출하지만, 첫 실행 전 아래를 확인:

```bash
# ZSC-EVAL 결과 경로 심볼릭 링크 생성
# ~/ZSC/results/  →  zsc-basecamp/results/zsceval/
source sh_scripts/common_args.sh && setup_dirs
```

### 3. Wandb 설정

`wandb_info/wandb_api_key`의 키를 자동으로 읽어 사용합니다.
모든 실험은 `m-personal-experiment/zsc-basecamp` 프로젝트에 기록됩니다.
(`WANDB_PROJECT`, `WANDB_ENTITY` 환경변수로 override — train 스크립트 내부 project명 무시)

---

## 결과 저장 경로

| 프레임워크 | 결과 경로 |
|-----------|----------|
| **GAMMA** | `GAMMA/mapbt/scripts/results/{env}/{layout}/{algo}/{exp}/` |
| **ZSC-EVAL** | `~/ZSC/results/{env}/{layout}/{algo}/{exp}/` → 심볼릭 링크로 `zsc-basecamp/results/zsceval/` |
| **Policy Pool (GAMMA)** | `GAMMA/mapbt/scripts/overcooked_population/` |
| **Policy Pool (ZSC-EVAL)** | `zsc-basecamp/policy_pool/` |

---

## 환경 설정 요약

| 설정 | GAMMA | ZSC-EVAL |
|------|-------|----------|
| env class | `Overcooked` (Berkeley) | `Overcooked_new` |
| old_dynamics | ✅ `--old_dynamics` 명시 | ⚠️ `overcooked_config.py` 수정 필요 |
| episode_length | 400 | 400 |
| ppo_epoch | 15 | 15 |
| entropy | 0.01 (단일) | [0.2→0.05→0.01] 스케줄 (논문 재현) |
| 시드 | 1~5 | 1~5 |
| 레이아웃 | 5종 | 5종 |

---

## 실행 순서 (의존성 그래프)

### GAMMA

```
Step 1 (독립):  gamma/sp.sh          → SP 모델 생성 (seed 1-5)
Step 1 (독립):  gamma/bc.sh          → BC 모델 생성 (seed 1-5)
Step 2 (독립):  gamma/mep_stage1.sh  → MEP population 생성 (seed=1 고정)
Step 3 (S2 의존): gamma/mep_stage2.sh → MEP adaptive (Step 2 완료 후)
```

### ZSC-EVAL

```
Step 1 (독립):  zsceval/sp.sh           → SP 모델 생성 (seed 1-5)
Step 1 (독립):  zsceval/mep_stage1.sh   → MEP population (seed=1 고정)
Step 1 (독립):  zsceval/traj_stage1.sh  → TrajeDi population (seed=1 고정)
Step 1 (독립):  zsceval/e3t.sh          → E3T (독립 실행)
Step 1 (독립):  zsceval/cole.sh         → COLE (독립 실행)

Step 2 (SP 완료 후):
  zsceval/fcp_stage2.sh  → FCP adaptive (SP population 필요)

Step 2 (MEP S1 완료 후):
  zsceval/mep_stage2.sh  → MEP adaptive
  zsceval/hsp_stage2.sh  → HSP adaptive (MEP S1 population 필요)

Step 2 (TrajeDi S1 완료 후):
  zsceval/traj_stage2.sh → TrajeDi adaptive
```

### 전체 병렬 실행 권장 순서

```bash
# Phase 1: 독립 실행 (병렬 가능)
bash sh_scripts/gamma/sp.sh &
bash sh_scripts/gamma/bc.sh &
bash sh_scripts/gamma/mep_stage1.sh &
bash sh_scripts/zsceval/sp.sh &
bash sh_scripts/zsceval/mep_stage1.sh &
bash sh_scripts/zsceval/traj_stage1.sh &
bash sh_scripts/zsceval/e3t.sh &
bash sh_scripts/zsceval/cole.sh &
wait

# Phase 2: Stage1 완료 후
bash sh_scripts/gamma/mep_stage2.sh &
bash sh_scripts/zsceval/fcp_stage2.sh &
bash sh_scripts/zsceval/mep_stage2.sh &
bash sh_scripts/zsceval/hsp_stage2.sh &
bash sh_scripts/zsceval/traj_stage2.sh &
wait
```

---

## 단일 레이아웃 실행

```bash
# 특정 레이아웃만 실행 (예: cramped_room)
bash sh_scripts/gamma/sp.sh cramped_room
bash sh_scripts/zsceval/sp.sh cramped_room
```

---

## common_args.sh 주요 설정값

| 변수 | 값 | 설명 |
|------|----|------|
| `LAYOUTS` | 5종 배열 | 벤치마크 레이아웃 |
| `SEED_BEGIN / SEED_END` | 1 / 5 | 시드 범위 |
| `GAMMA_POLICY_POOL` | `GAMMA/mapbt/scripts/overcooked_population/` | GAMMA policy pool |
| `ZSCEVAL_POLICY_POOL` | `zsc-basecamp/policy_pool/` | ZSC-EVAL policy pool |
| `WANDB_ENTITY` | `m-personal-experiment` | wandb entity |
| `WANDB_PROJECT` | `zsc-basecamp` | wandb project |
