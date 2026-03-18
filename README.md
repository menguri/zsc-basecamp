# ZSC-Basecamp

**Zero-Shot Coordination (ZSC) 연구를 위한 통합 학습·평가 플랫폼.**
Overcooked 환경에서 다양한 ZSC 알고리즘을 동일 조건으로 훈련하고, cross-play 평가로 성능을 비교한다.
자체 개발 알고리즘 **PH2**의 구현 및 실험 기반으로도 사용된다.

---

## 디렉토리 구조

```
zsc-basecamp/
├── ph2/                    # PH2 알고리즘 구현 (독립 패키지)
├── GAMMA/                  # GAMMA 프레임워크 (NeurIPS 2024)
├── ZSC-EVAL/               # ZSC-EVAL 벤치마크 툴킷 (NeurIPS 2024 D&B)
├── sh_scripts/             # 학습 실행 스크립트
│   ├── common_args.sh      # 공통 환경변수 및 경로 설정
│   ├── zsceval/            # ZSC-EVAL 기반 알고리즘 스크립트
│   └── ph2/                # PH2 학습 스크립트
├── eval/                   # 평가 파이프라인
│   ├── eval-overcooked.sh  # 평가 실행 진입점
│   ├── eval_code/          # xp_eval.py 등 평가 코드
│   ├── models/             # 평가용 모델 (gitignore)
│   └── results/            # 평가 결과 CSV (gitignore)
├── policy_pool/            # ZSC-EVAL population YAML 및 policy config
├── results/                # 학습 결과 (gitignore)
│   ├── zsceval/            # ~/ZSC/results/ 심볼릭 링크 대상
│   └── ph2/                # PH2 학습 결과
└── wandb_info/             # WandB API 키 (gitignore)
```

---

## 환경: Overcooked 5종 레이아웃

| 레이아웃 | 특징 |
|---------|------|
| `cramped_room` | 좁은 공간, 충돌 회피 중요 |
| `coordination_ring` | 원형 동선, 순환 협력 필요 |
| `asymmetric_advantages` | 비대칭 능력, 역할 분담 |
| `forced_coordination` | 에이전트 간 강제 의존 |
| `counter_circuit_o_1order` | 양파 3개 단일 주문, 카운터 경유 |

- **Obs shape**: `(H, W, 20)` channels-last, 레이아웃마다 H×W 다름
- **Share obs**: `concat(obs_i, obs_j)` → critic 입력 (MAPPO 방식)
- **Action**: 6개 이산 (상/하/좌/우/정지/상호작용)
- **Episode**: 400 스텝, `overcooked_version=new`

---

## 지원 알고리즘

### Stage 1 — Population 생성 (10M 스텝)
| 알고리즘 | 설명 | 스크립트 |
|---------|------|---------|
| **SP** | Self-Play 베이스라인 | `zsceval/sp.sh` |
| **E3T** | End-to-End + 파트너 행동 예측 | `zsceval/e3t.sh` |
| **MEP Stage 1** | Maximum Entropy Population | `zsceval/mep_stage1.sh` |
| **TrajeDi Stage 1** | Trajectory Diversity PBT | `zsceval/traj_stage1.sh` |
| **BC** | Behavioral Cloning (인간 데이터) | `zsceval/bc.sh` |

### Stage 2 — Adaptive 에이전트 (50M 스텝)
| 알고리즘 | 설명 | 스크립트 |
|---------|------|---------|
| **FCP** | Fictitious Co-Play (Stage 1 population 사용) | `zsceval/fcp_stage2.sh` |
| **HSP** | Hidden-utility Self-Play | `zsceval/hsp_stage2.sh` |
| **MEP Stage 2** | MEP adaptive | `zsceval/mep_stage2.sh` |
| **TrajeDi Stage 2** | TrajeDi adaptive | `zsceval/traj_stage2.sh` |
| **COLE** | Cooperative Open-ended Learning | `zsceval/cole.sh` |

### 자체 개발
| 알고리즘 | 설명 | 스크립트 |
|---------|------|---------|
| **PH2** | Dual-policy PPO + 파트너 예측 (아래 상세 참조) | `ph2/ph2.sh` |

---

## PH2 알고리즘

> 상세 분석: [`ph2-info.md`](ph2-info.md)

### 핵심 아이디어

**두 개의 정책(spec / ind)을 동시에 교대로 학습**하는 듀얼-브랜치 MAPPO.

- **spec (specialist)**: spec-spec 자기대전에 특화. blocking penalty 적용 가능.
- **ind (individual)**: ind-ind 및 spec-ind 혼합 에피소드로 학습. 새 파트너 적응 담당.
- 한 update마다 `spec_step → ind_step` 순서로 진행. ind는 방금 업데이트된 spec을 파트너로 사용.

### 매칭 타입

```
ind-match (50%): ind ↔ ind  →  ind 업데이트
spec-match(50%): spec ↔ spec →  spec 업데이트
혼합 env에서 슬롯별 role mask로 업데이트 범위 분리
```

`PH2_FIXED_IND_PROB=0.5`가 기본값. 진행도 기반 3단계 스케줄(`PH2_RATIO_STAGE1/2/3`)로 변경 가능.

### 파트너 예측 (E3T-style)

`PartnerPredictionNet`이 과거 `(obs, action)` 히스토리(기본 5스텝)로 파트너의 다음 행동을 예측.
Cross-entropy loss로 보조 학습. spec/ind 각각 독립적인 predictor 보유.

### 손실 함수

```
L_total = L_actor(PPO clip) + VF_COEF * L_value(clipped) - ENT_COEF * H(π) + L_pred(CE)
```

critic은 `share_obs = concat(obs_i, obs_j)` 입력 (GAMMA/ZSC-EVAL MAPPO 표준 동일).

### StateReconNet (선택)

`USE_STATE_RECON=1` 플래그로 활성화. SP 학습 중 ego 히스토리로 게임 상태 재구성 + 파트너 행동 예측을 동시에 학습하는 보조 네트워크. 주요 학습 신호는 BCE(상태 재구성) + CE(행동 예측).

### 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `ph2_fixed_ind_prob` | 0.5 | ind-match 에피소드 비율 |
| `ph2_use_partner_pred` | True | 파트너 예측 활성화 |
| `ph2_history_len` | 5 | 예측 히스토리 윈도우 크기 |
| `ph2_pred_loss_coef` | 1.0 | 예측 loss 가중치 |
| `ph2_spec_use_blocked` | False | blocking penalty 활성화 |
| `ph2_blocked_penalty_omega` | 10.0 | 페널티 스케일 |
| `use_state_recon` | False | StateReconNet 동시 학습 |

---

## 학습 실행

### 공통 설정 (`sh_scripts/common_args.sh`)

```bash
# 첫 실행 시 심볼릭 링크 등 초기화
source sh_scripts/common_args.sh && setup_dirs
```

- `GPU=N` 환경변수로 GPU 지정 (기본 0)
- `WANDB_ENTITY`, `WANDB_PROJECT` 자동 설정
- ZSC-EVAL 결과는 `~/ZSC/results/` → `results/zsceval/`로 심볼릭 링크

### SP 학습

```bash
# 기본 (전체 레이아웃 순차 실행)
bash sh_scripts/zsceval/sp.sh

# 특정 레이아웃
bash sh_scripts/zsceval/sp.sh cramped_room

# StateReconNet 동시 학습
USE_STATE_RECON=1 bash sh_scripts/zsceval/sp.sh cramped_room
```

### PH2 학습

```bash
GPU=0 bash sh_scripts/ph2/ph2.sh cramped_room
```

### FCP (2-stage)

```bash
# Stage 1: SP population 생성
bash sh_scripts/zsceval/sp.sh

# Stage 2: FCP adaptive agent
bash sh_scripts/zsceval/fcp_stage2.sh
```

---

## 평가 실행

### 단일 알고리즘 self-play / cross-play

```bash
# SP self-play (asymmetric_advantages)
LAYOUT=asymmetric_advantages ALGO0=sp CUDA_VISIBLE_DEVICES=0 \
  bash eval/eval-overcooked.sh

# SP vs BC cross-play
LAYOUT=cramped_room ALGO0=sp ALGO1=bc CUDA_VISIBLE_DEVICES=0 \
  bash eval/eval-overcooked.sh
```

### 전체 레이아웃 × 알고리즘 일괄 실행 (GPU 분산)

```bash
# GPU 4~7에 28개 잡 분산 예시
CUDA_VISIBLE_DEVICES=4 LAYOUT=asymmetric_advantages ALGO0=sp GENERATE_SUMMARY=0 \
  nohup bash eval/eval-overcooked.sh > eval/logs/aa_sp.log 2>&1 &
```

### 결과 집계

```bash
.zsc-zsceval/bin/python eval/generate_summary.py --results_root eval/results
# → eval/results/xp_summary.csv 생성
```

### 주요 환경변수

| 변수 | 기본값 | 설명 |
|-----|-------|------|
| `LAYOUT` | `asymmetric_advantages` | 평가 레이아웃 |
| `ALGO0` | `ph2` | 기준 알고리즘 |
| `ALGO1` | `` (빈값 = self-play) | 상대 알고리즘 |
| `CUDA_VISIBLE_DEVICES` | `1` | GPU |
| `EVAL_EPISODES` | `5` | 에피소드 수 |
| `EVAL_SEEDS` | `10,11,12,13,14` | 평가 시드 |
| `OVERWRITE` | `1` | 기존 결과 덮어쓰기 |
| `VIZ` | `0` | GIF 저장 여부 |

---

## 모델 경로 규약

### 학습 결과
```
results/
  zsceval/Overcooked/<layout>/<algo>/<exp>/runX/
    models/actor_periodic_<steps>.pt
    policy_config.pkl
  ph2/Overcooked/<layout>/ph2/runX/
    models/ind_actor_periodic_<steps>.pt
    models/ind_pred_periodic_<steps>.pt
    policy_config.pkl
```

### 평가용 모델
```
eval/models/Overcooked/<layout>/
  sp/runX/    → actor_periodic_10000000.pt + policy_config.pkl
  e3t/runX/   → actor_agent0/1_periodic_10000000.pt + policy_config.pkl
  fcp/runX/   → actor_periodic_50000000.pt + policy_config.pkl
  ph2/runX/   → ind_actor_periodic_30000000.pt + ind_pred_periodic_30000000.pt + policy_config.pkl
  bc/runX/    → actor.pt + policy_config.pkl
```

---

## 의존성 및 환경 설정

```bash
# ZSC-EVAL 가상환경 (평가/ZSC-EVAL 훈련)
.zsc-zsceval/bin/python

# GAMMA 가상환경 (GAMMA 훈련)
.zsc-gamma/bin/python
```

WandB 키는 `wandb_info/wandb_api_key`에 저장 (gitignore 처리됨).
`common_args.sh` source 시 자동으로 `WANDB_API_KEY` 환경변수로 로드.

---

## 참고

- **GAMMA**: [NeurIPS 2024] Generative Agent Modeling for Multi-agent Adaptation
- **ZSC-EVAL**: [NeurIPS 2024 D&B Track] Zero-Shot Coordination Evaluation Benchmark
- **PH2 상세 분석**: [`ph2-info.md`](ph2-info.md)
