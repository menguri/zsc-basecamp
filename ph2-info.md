# PH2 알고리즘 분석 (experiments-stablock 기준)

## 0) 분석 범위

- 대상 코드: `ex-overcookedv2/experiments-stablock`
- 핵심 진입점:
  - `overcooked_v2_experiments/ppo/run.py` (PH2 분기)
  - `overcooked_v2_experiments/ppo/ippo_ph2.py` (PH2 듀얼 래퍼)
  - `overcooked_v2_experiments/ppo/ippo_ph2_core.py` (실제 PPO 업데이트/손실)
  - `overcooked_v2_experiments/ppo/models/rnn.py` (정책/가치/파트너예측 모델)

---

## 1) PH2가 실제로 어떻게 학습되는가

### 1-1. PH2 실행 분기

- `ALG_NAME`에 `"PH2"`가 포함되면 PH2 trainer를 사용함.
  - 근거: `overcooked_v2_experiments/ppo/run.py:166-173`
- PH2 기본 실험 설정은 `rnn-ph2.yaml`에서 정의됨.
  - `ALG_NAME: "PH2-E3T"`, `PH1_ENABLED: True`, `PH2_*` 스케줄 파라미터
  - 근거: `.../config/experiment/rnn-ph2.yaml:5-61`

### 1-2. 듀얼 정책 구조 (spec / ind)

- PH2는 내부적으로 정책을 2개 동시에 유지함:
  - specialist branch (`spec_cfg`, 로그 prefix는 `phase1`)
  - individual branch (`ind_cfg`, 로그 prefix는 `phase2`)
- 역할 플래그:
  - `PH2_ROLE="spec"` / `PH2_ROLE="ind"`
  - `PH2_MATCH_SCHEDULE=True`
  - 근거: `.../ippo_ph2.py:231-240`
- 한 update에서 순서:
  1. `spec_step`를 먼저 수행 (파트너로 ind 정책 사용)
  2. `ind_step`를 수행 (파트너로 방금 업데이트된 spec 정책 사용)
  - 근거: `.../ippo_ph2.py:403-419`

즉, PH2는 "한 정책만 학습"이 아니라 spec/ind 두 branch를 교대로 동기화하며 학습한다.

### 1-3. episode 매칭 타입 (spec-spec vs spec-ind vs ind-ind)

- PH2는 에피소드마다 매칭 타입을 섞는다.
- 매칭 확률은 `_phase2_ind_match_prob(update_step)`로 계산:
  - `PH2_FIXED_IND_PROB`가 있으면 그 값을 그대로 사용
  - 없으면 진행률 1/3 구간마다 `PH2_RATIO_STAGE1/2/3` 사용
  - 근거: `.../ippo_ph2_core.py:280-293`
- 초기 env별 매칭 상태도 동일 로직으로 샘플됨.
  - 근거: `.../ippo_ph2_core.py:2751-2757`

중요:
- 현재 `rnn-ph2.yaml` 기본값은 `PH2_FIXED_IND_PROB: 0.5`라서 stage ratio가 기본적으로는 무시됨.
  - 근거: `.../config/experiment/rnn-ph2.yaml:54-58`

### 1-4. 어떤 샘플로 누구를 업데이트하는가 (핵심)

- 롤아웃 시 `action_pick_mask`, `train_update_mask`를 만들어서
  - 어떤 actor 슬롯에서 learner action을 쓸지
  - 어떤 샘플을 손실 업데이트에 포함할지
  를 분리한다.
- 근거: `.../ippo_ph2_core.py:1135-1176`

구체적으로:

1. `PH2_ROLE=spec`
- ind-match env에서는 spec가 일부 슬롯 행동만 담당하지만, 업데이트는 spec-match 샘플 중심.
- 코드 주석 그대로: `spec updates only on spec-match samples`.
- 근거: `.../ippo_ph2_core.py:1152-1163`

2. `PH2_ROLE=ind`
- ind-ind에서는 양쪽 모두 ind가 수행/업데이트.
- spec-ind에서는 ind가 맡은 슬롯만 업데이트.
- 근거: `.../ippo_ph2_core.py:1166-1176`

### 1-5. 보상 경로

- 기본 보상은 `original + shaped * anneal_factor`.
- 단, `ind` branch 업데이트에서는 blocking penalty를 제외한 pure reward 경로를 사용함.
- 근거: `.../ippo_ph2_core.py:1513-1523`
- 실제 Transition에 들어가는 reward는 `reward_for_update`.
- 근거: `.../ippo_ph2_core.py:1650-1655`

### 1-6. 체크포인트 저장 방식

- PH2는 체크포인트에 `params_spec`, `params_ind`를 함께 저장함.
- 근거:
  - 저장: `.../ppo/main.py:248-257`, `267-276`
  - payload: `.../ppo/utils/store.py:70-73`

---

## 2) PH2에서 "무엇"이 학습되는가

### 2-1. 정책/가치 본체 (IPPO 형태)

- 모델은 shared encoder + GRU 위에 actor head와 critic head를 둠.
- actor logits: categorical policy
- critic: scalar value
- 근거: `.../models/rnn.py:331-353`

즉, 현재는 각 actor의 observation 기반 value를 학습하는 IPPO 구조(centralized critic 아님).

### 2-2. 파트너 모델링 분기

PH2-E3T에서 partner predictor가 함께 학습됨.

1. Action-prediction 모드 (`STATE_PREDICTION=False`)
- `predict_partner`로 파트너 행동 logits 예측
- CE loss 사용
- 근거:
  - 모델: `.../models/rnn.py:366-396`
  - 손실: `.../ippo_ph2_core.py:1977-1993`

2. State-prediction 모드 (`STATE_PREDICTION=True`)
- `predict_partner_state`가
  - `action_logits`
  - `next_z`
  - `context_z`
  를 출력
- CE(action) + MSE(next_z) 조합
- 근거:
  - 모델: `.../models/rnn.py:399-446`
  - 손실: `.../ippo_ph2_core.py:1937-1974`

### 2-3. Transition에 들어가는 학습 신호

- PH2 transition에는 일반 PPO 항목 외에 다음이 포함됨:
  - `obs_history`, `act_history`, `partner_action`, `z_state`, `next_partner_obs`, `blocked_states`
- 근거:
  - 구조체: `.../ippo_ph2_core.py:48-65`
  - 생성: `.../ippo_ph2_core.py:1650-1668`

---

## 3) PH2 손실 함수 구성

코드 기준 loss는 아래 합으로 구성됨.

## 3-1. GAE / target

- \(\delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)\)
- \(\hat{A}_t = \delta_t + \gamma \lambda (1-d_t)\hat{A}_{t+1}\)
- target = \(\hat{A}_t + V(s_t)\)
- 근거: `.../ippo_ph2_core.py:1847-1876`

## 3-2. PPO actor loss

- ratio = \(\exp(\log \pi_\theta - \log \pi_{\theta_{old}})\)
- clipped surrogate:
  - \(L^{clip} = -\min(r_t \hat{A}_t, \text{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)\)
- 근거: `.../ippo_ph2_core.py:2058-2077`

## 3-3. value loss (clipped value)

- \(V_\text{clip}=V_{old}+\text{clip}(V-V_{old},-\epsilon,+\epsilon)\)
- \(L_V = 0.5 \max((V-target)^2,(V_\text{clip}-target)^2)\)
- 근거: `.../ippo_ph2_core.py:2047-2054`

## 3-4. entropy bonus

- \(L_H = -\text{ENT\_COEF}\cdot \mathbb{E}[H(\pi)]\)
- 근거: `.../ippo_ph2_core.py:2078-2087`

## 3-5. partner prediction loss

1. Action prediction:
- \(L_{pred} = \alpha \cdot CE(\hat{a}_{partner}, a_{partner})\)
- 근거: `.../ippo_ph2_core.py:1984-1990`

2. State prediction:
- \(L_{pred} = \alpha\cdot CE(\hat{a}_{partner},a_{partner}) + \beta\cdot ||\hat{z}_{next}-z_{target}||^2\)
- 근거: `.../ippo_ph2_core.py:1949-1974`

## 3-6. 최종 손실

- \(L_{total} = L_{actor} + \text{VF\_COEF}\cdot L_V - \text{ENT\_COEF}\cdot H + L_{pred}\)
- 근거: `.../ippo_ph2_core.py:2083-2088`

추가:
- PH2에서는 `train_mask`로 샘플을 마스킹하여 role별 업데이트 범위를 제어함.
- 근거: `.../ippo_ph2_core.py:1894-1909`, `2136-2139`

---

## 4) MAPPO 전환 원칙 (GAMMA/ZSC-EVAL과 동일하게)

요구사항 기준: PH2를 MAPPO로 바꿀 때 **GAMMA/ZSC-EVAL에서 이미 쓰는 방식과 1:1로 맞춘다.**

핵심은 아래 3줄이다.

1. actor 입력은 기존 `obs` 유지
2. critic 입력은 `share_obs` 사용
3. `share_obs`는 GAMMA/ZSC-EVAL Overcooked처럼 **agent obs concat**으로 생성

### 4-1. share_obs 생성 규칙 (엔진 함수 의존 없이)

GAMMA/ZSC-EVAL overcooked는 이미 다음 규칙을 사용한다.

- 각 에이전트 관측을 채널축으로 이어붙여 share_obs 생성
  - `share_obs0 = concat(obs0, obs1)`
  - `share_obs1 = concat(obs1, obs0)`
  - 최종 `stack([share_obs0, share_obs1])`
- 근거:
  - GAMMA: `GAMMA/mapbt/envs/overcooked/Overcooked_Env.py:715-738`
  - ZSC-EVAL(new): `ZSC-EVAL/zsceval/envs/overcooked_new/Overcooked_Env.py:866-903`

즉 PH2-MAPPO에서도 이 규칙을 그대로 채택하면 되고,  
`overcooked_v2` 엔진의 global-state 전용 함수에 의존할 필요가 없다.

### 4-2. critic 경로 정합성 (GAMMA/ZSC-EVAL MAPPO와 동일)

GAMMA/ZSC-EVAL MAPPO는 정책에서 actor/critic 입력을 분리하고 critic은 `share_obs`를 받는다.

- 근거:
  - GAMMA policy: `GAMMA/mapbt/algorithms/r_mappo/algorithm/rMAPPOPolicy.py:30-42`
  - ZSC-EVAL policy: `ZSC-EVAL/zsceval/algorithms/r_mappo/algorithm/rMAPPOPolicy.py:62-100`
- runner에서도 `use_centralized_V=True`면 share_obs를 버퍼에 넣고 critic 계산에 사용
  - GAMMA: `GAMMA/mapbt/runner/shared/overcooked_runner.py:116-155`
  - ZSC-EVAL: `ZSC-EVAL/zsceval/runner/shared/overcooked_runner.py:192-261`
  - ZSC-EVAL base: `ZSC-EVAL/zsceval/runner/shared/base_runner.py:123-137`, `185-190`

따라서 PH2 쪽도 같은 인터페이스를 따르면 된다:

- rollout/transition에 `share_obs` 저장
- `get_values`와 value loss 계산 시 `share_obs` 사용
- actor, partner prediction, PH2 role-mask 로직은 유지

### 4-3. PH2 코드에 적용 시 바뀌는 최소 지점

1. Transition에 `share_obs`(필요시 `next_share_obs`) 추가
2. env step에서 `share_obs` 생성 후 transition에 저장 (obs concat 방식)
3. critic forward를 `share_obs` 입력으로 변경
4. `last_val`/GAE target 계산도 `share_obs` 기준으로 변경
5. minibatch permutation 시 `obs`, `share_obs` 동시 셔플

### 4-4. 손실식 자체는 그대로

- PPO actor/clipped value/entropy/pred loss 식은 동일
- 변경점은 value가 참조하는 입력만 바뀜:
  - IPPO: \(V(o_i)\)
  - MAPPO: \(V(\text{share\_obs}_i)\)

### 4-5. 구현 방향 결론

- PH2-MAPPO는 “새 알고리즘 추가”가 아니라,
  - **PH2 학습 스케줄/마스크는 그대로**
  - **critic 입력만 GAMMA/ZSC-EVAL MAPPO 표준(`share_obs=obs concat`)으로 교체**
  하면 된다.

---

## 5) 한 줄 요약

PH2는 "spec/ind 두 정책을 교대로 학습하는 듀얼 PPO+E3T"이며, MAPPO 전환의 본질은 **critic 입력을 GAMMA/ZSC-EVAL 방식의 share_obs(obs concat)로 바꾸는 것**이다. PH2 role/mask/스케줄은 그대로 재사용 가능하다.

---

## 6) `counter_circuit_o_1order`는 원본 counter circuit인가?

결론: **아니다.**

- `counter_circuit.layout`와 `counter_circuit_o_1order.layout`는 그리드/주문 정의가 다르다.
  - `counter_circuit.layout`: 토마토/양파 혼합 주문(`onion+tomato`, `onion+tomato+tomato`, ...)
  - `counter_circuit_o_1order.layout`: 양파 3개 단일 주문만 사용
- 근거:
  - `GAMMA/.../data/layouts/counter_circuit.layout`
  - `GAMMA/.../data/layouts/counter_circuit_o_1order.layout`

다만 실험 파이프라인에서는 인간 데이터 이름 `random3`를 `counter_circuit` 계열과 연결해 쓰는 관행이 있다.

- 예시 근거:
  - `GAMMA/mapbt/scripts/train_overcooked_bc.sh:5-7`
  - `ZSC-EVAL/zsceval/human_exp/overcooked_utils.py:13-18`

---

## 7) `zsc-basecamp/ph2` 독립 구현 계획 (Claude Opus 전달용)

요구사항 재정의:

- 구현 위치: `zsc-basecamp/ph2` 내부에서만 학습 파이프라인 동작
- 알고리즘 기반: **ZSC-EVAL의 MAPPO 네트워크/기본 하이퍼파라미터를 최대한 동일하게 사용**
- 환경: Overcooked 동작/관측/`share_obs` 규칙을 GAMMA/ZSC-EVAL과 동일하게 유지
- 금지: `ex-overcookedv2` 코드 직접 import/의존 금지
- 저장 포맷: GAMMA/ZSC-EVAL eval 파이프라인과 충돌 없도록 파일 형식/경로 규약 유지

### 7-1. 목표 디렉토리 구조

```text
zsc-basecamp/ph2/
  README.md
  ph2_config.py
  scripts/
    train/
      train_ph2.py
  runner/
    shared/
      base_runner.py
      overcooked_runner.py
  algorithms/
    r_mappo/
      algorithm/
        rMAPPOPolicy.py
      r_mappo.py
    ph2/
      ph2_trainer.py
      ph2_buffer.py
      ph2_utils.py
  envs/
    overcooked/
      overcooked_adapter.py
```

원칙:

- `algorithms/r_mappo/*`는 ZSC-EVAL 구현을 거의 그대로 가져오고, 변경은 최소화
- PH2 고유 로직(spec/ind role, match schedule, pred loss)은 `algorithms/ph2/*`에만 추가
- Overcooked 엔진은 `ZSC-EVAL/zsceval/envs/overcooked_new`를 호출/랩핑해 동일 동작 보장

### 7-2. 구현 단계 (실행 순서)

1. 베이스라인 고정
- ZSC-EVAL `r_mappo` policy/trainer, runner 인터페이스를 `ph2`로 복사
- 우선 PH2 없이 plain MAPPO 1-step 학습이 도는지 확인

2. 환경 어댑터 고정
- `overcooked_adapter.py`에서 obs/share_obs/action_space를 ZSC-EVAL과 동일하게 노출
- `share_obs = agent obs concat` 규칙을 강제 (`[obs_i, obs_j]` 순서 유지)

3. PH2 듀얼-브랜치 추가
- `ph2_trainer.py`에서 `spec_policy`, `ind_policy` 2개 유지
- update 한 번당 `spec_step -> ind_step` 순서 적용
- role/match 마스크로 샘플 사용 범위를 분리

4. 손실 결합
- PPO loss(clip/value/entropy)는 ZSC-EVAL MAPPO 코드 경로 재사용
- PH2 pred loss(action pred 또는 state pred)만 추가 합산
- critic 입력은 반드시 `share_obs`

5. 저장/복구 호환
- 기본 저장 규약 유지:
  - `models/actor.pt`
  - `models/critic.pt`
  - `policy_config.pkl`
- 듀얼 정책은 추가 파일로 확장:
  - `models/spec_actor.pt`, `models/spec_critic.pt`
  - `models/ind_actor.pt`, `models/ind_critic.pt`
- eval 호환을 위해 `actor.pt/critic.pt`는 기본 branch(예: `ind`)를 가리키도록 유지

6. 검증
- 기존 ZSC-EVAL eval 스크립트로 `actor.pt/critic.pt` 로딩 성공 확인
- PH2 전용 eval은 `spec_*`/`ind_*`를 선택적으로 로딩하도록 별도 옵션 제공

### 7-3. 기본 파라미터 정렬 규칙

- 시작점은 `ZSC-EVAL/zsceval/overcooked_config.py`와 동일
- 반드시 동일하게 유지할 항목:
  - `hidden_size`, `layer_N`, `use_orthogonal`, `use_valuenorm`
  - `ppo_epoch`, `clip_param`, `num_mini_batch`
  - `lr`, `critic_lr`, `opti_eps`, `weight_decay`
  - `episode_length`, `n_rollout_threads`, `num_env_steps`
- PH2 전용 항목만 신규 추가:
  - `ph2_role`, `ph2_fixed_ind_prob`, `ph2_ratio_stage1/2/3`
  - `ph2_pred_coef`, `ph2_state_pred`, `ph2_match_schedule`

### 7-4. 저장 포맷 호환 계약 (필수)

아래 4개는 기존 eval/restore 호환을 위한 최소 계약이다.

1. `policy_config.pkl` payload 형식:
- `(all_args, obs_space, share_obs_space, action_space)` 튜플 유지

2. 모델 파일명:
- 단일 정책 평가 경로는 기존과 동일하게 `actor.pt`, `critic.pt` 사용

3. 실행 결과 경로:
- `.../results/Overcooked/<layout>/<algo>/<exp>/runX/` 구조 유지

4. restore 동작:
- `model_dir`를 주면 기존 Runner restore 코드로 바로 로딩 가능해야 함

### 7-5. Claude Opus 작업 지시문(복붙용)

```text
Implement PH2 under zsc-basecamp/ph2 without importing ex-overcookedv2.
Reuse ZSC-EVAL MAPPO network and default hyperparameters as closely as possible.
Use the same Overcooked behavior and share_obs construction rule as GAMMA/ZSC-EVAL (agent obs concatenation).
Keep checkpoint/eval compatibility with GAMMA/ZSC-EVAL:
- policy_config.pkl tuple format
- models/actor.pt and models/critic.pt for default eval path
- optional spec/ind branch checkpoints as extra files
Deliver runnable train_ph2.py and compatibility smoke tests.
```
