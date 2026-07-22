# AutoResearch 워크북 — 실전 활용 가이드 (한 문서로 끝내기)

이 문서 하나로 AutoResearch 프레임워크를 **이해하고, 실행하고, 결과를 읽고, 내
문제로 바꿔 쓰고, 고장났을 때 고칠** 수 있게 만드는 것이 목표다. README와
`docs/HANDOFF.md`가 "무엇을 왜 만들었나(설계·불변식)"라면, 이 문서는 "그래서
어떻게 쓰나(실전·사례)"다.

작성 기준: contract schema v8 (`autoresearch-phase6c-tsp`, Euclidean-TSP 도메인),
Phase 1~6c 완료 상태. 본문의 콘솔 출력·숫자는 대부분 **실제로 이 저장소에서 방금
돌린 캠페인 `c20260718051332`** 의 결과다(§6).

---

## 목차

- [0. 30초 요약과 멘탈 모델](#0-30초-요약과-멘탈-모델)
- [1. AutoResearch가 실제로 하는 일](#1-autoresearch가-실제로-하는-일)
- [2. 설치와 사전 준비](#2-설치와-사전-준비)
- [3. 5분 퀵스타트 (복붙용)](#3-5분-퀵스타트-복붙용)
- [4. 꼭 알아야 할 8개 개념](#4-꼭-알아야-할-8개-개념)
- [5. 명령어 레퍼런스](#5-명령어-레퍼런스)
- [6. 사례 워크스루 — 실제 TSP 캠페인 한 판](#6-사례-워크스루--실제-tsp-캠페인-한-판)
- [7. 출력물 읽는 법](#7-출력물-읽는-법)
- [8. 심화 활용 레시피](#8-심화-활용-레시피)
- [9. 내 연구 문제로 도메인 바꾸기](#9-내-연구-문제로-도메인-바꾸기)
- [10. 트러블슈팅 & FAQ](#10-트러블슈팅--faq)
- [11. 건강 점검 (테스트 드릴)](#11-건강-점검-테스트-드릴)
- [12. 부록 — 레퍼런스 사전](#12-부록--레퍼런스-사전)

---

## 0. 30초 요약과 멘탈 모델

**한 줄:** AutoResearch는 "가설을 세우고 → 격리된 환경에서 코드/하이퍼파라미터를
바꿔 실행하고 → 신뢰할 수 있는 평가기로 채점하고 → 통과한 것만 남기는" 자율 연구
루프를, **평가기·계약·프로버넌스(증거 이력)를 에이전트보다 먼저** 세워서 돌린다.

**핵심 철학 (한 문장):** *신뢰할 수 있는 평가기가 없으면 에이전트를 아무리 늘려도
더 과학적이 되지 않는다.* 이 저장소의 모든 장치는 "그럴듯해 보이는 실패"와
"검증된 진전"을 구분하기 위해 존재한다.

AI/ML 하는 사람에게 익숙한 비유로 전체 그림:

| ML 세계 | AutoResearch |
|---|---|
| train / validation / test split | **dev / gate / test** 세 스플릿 (숨은 시드로 생성) |
| validation으로 하이퍼파라미터 튜닝 | **dev 스플릿**으로 keep/reject 탐색 |
| test는 논문 낼 때 딱 한 번 | **test 스플릿**은 `report`에서 딱 한 번 (사람 승인 필요) |
| validation overfitting 방지 | **blind gate** = 숨은 gate 스플릿으로 일반화 재확인 |
| ablation study | reject된 실험도 전부 원장(ledger)에 증거로 보존 |
| NAS / population-based training | **generation**마다 K개 가설 병렬 → 승자 1명 채택 |
| successive halving (HPO) | 값싼 smoke rung → 상위만 dev rung (§4) |
| RLHF의 reward hacking 방지 | 평가기가 자기보고 점수 무시, **직접 재계산** |

전체 파이프라인은 이 순서로 흐른다. 화살표 위 괄호는 "그 단계에서 새로 생기는
것"이다.

```
init ──(baseline·숨은 시드·보호 매니페스트)──▶ ground ──(문헌 근거 인증서)──▶
run ──(generation N판: 가설→실행→gate→채택)──▶ status ──(현황 대시보드)──▶
report ──(exit 3: 사람 승인 요구)──▶ approve ──(출판 의도 승인)──▶
report ──(test 다중시드 평가 + bootstrap CI + report.md 봉인)──▶ 끝
```

이 문서를 처음 본다면 §3(퀵스타트)를 그대로 복붙해 한 판 돌려보고, §6(사례
워크스루)에서 각 줄이 무슨 뜻인지 확인한 뒤, 필요할 때 §5·§7·§10을 사전처럼 찾아
보는 흐름을 권한다.

### 핵심 용어 먼저 (여기서 막히면 아래를 보라)

본문을 읽다 걸리기 쉬운 7개 단어만 미리 못 박아 둔다.

- **캠페인(campaign)** — 한 번의 "연구 프로젝트 단위". `init`으로 베이스라인을
  고정한 순간부터 `report`로 결과를 봉인할 때까지의 전체 실험 이력이 하나의
  캠페인이다. 고유 id(`c20260718051332` = 생성 시각)를 갖고 자기만의 숨은 시드·
  베이스라인·원장(ledger)을 가진다. `init --force`는 이전 캠페인을 폐기하고 새
  캠페인을 연다. → ML의 **실험 스윕(sweep) / run group**에 해당(같은 데이터 split·
  같은 baseline 기준으로 묶인 실험 묶음).
- **baseline(베이스라인)** — `init` 시점에 고정한 **출발점 성능**. 캠페인 내내 안
  바뀐다. 모든 개선은 이 대비로 잰다. (실측: dev tour 6,579,300.)
- **incumbent(현 챔피언)** — 지금까지 채택된 것 중 **가장 좋은 해/커밋**. 새 후보는
  incumbent를 기준으로 평가되고, 이기면 그 후보가 **새 incumbent**가 되어 main에
  머지된다. 처음엔 incumbent==baseline. → ML의 **best checkpoint / model selection의
  현재 최고**. (실측: 최종 incumbent = restarts=16 커밋 `4796a8c`.)
- **candidate(후보)** — 한 세대에서 제안된 가설의 실행 결과. incumbent에 도전한다.
  이 셋의 관계가 핵심이다: **baseline(고정) ← incumbent(갱신되는 챔피언) ←
  candidate(도전자)**.
- **restarts** — TSP solver의 하이퍼파라미터 하나. iterated local search에서 "지역
  최적(local optimum)에 갇히면 살짝 흔들어(perturbation) local search를 몇 번 더
  **재시작**할지"의 횟수. 많을수록 더 좋은 tour를 찾을 확률이 오르지만 **연산량이
  선형으로 늘어난다**(restarts=16 → 약 16배 연산). → ML의 **random restart best-of-N**
  (비볼록 최적화에서 여러 초기값으로 여러 번 돌려 제일 좋은 걸 고르기)과 같다.
- **봉인(seal)** — `report`가 미접촉 **test 스플릿**으로 최종 결과를 확정하는 것.
  이 순간의 숫자·통계·claim·그림을 `final_report` 레코드 + `claims.jsonl` +
  sha256 해시로 "이게 공식 결과다"라며 **불변으로 잠근다**(코드 용어 그대로 seal).
  논문에 test set을 한 번 쓰는 것과 같아, 다시 하려면 `--force`로 재승인해야 하고
  그 재사용 횟수가 **다중검정 공시**에 카운트된다.
- **dev / gate / test** — 숨은 시드로 만든 세 스플릿. dev=탐색(keep/reject),
  gate=일반화 재확인(blind), test=최종 보고 1회용. → **train / val / test** 그대로.

---

## 1. AutoResearch가 실제로 하는 일

### 1.1 지금 이 저장소가 풀고 있는 문제 (Euclidean-TSP)

현재 도메인은 **유클리드 TSP(외판원 문제)**다. 회귀·분류 같은 ML 문제가 아니라
조합최적화 문제라는 점이 중요하다.

- 평가기가 숨은 시드로 **도시 좌표 인스턴스**(도시 60개, 정수 격자 좌표)를 생성한다.
- solver(`src/train.py`)에게 **좌표만** 건네고, solver는 각 인스턴스에 대해
  **tour(도시 방문 순열)**를 반환한다.
- 평가기가 순열이 유효한지 검증하고 **tour 길이를 직접 재계산**한다. solver가
  자기 점수를 `reported_objectives`에 적어 보내도 **완전히 무시**한다 (점수 위조
  불가). 거리는 TSPLIB EUC_2D 정수 반올림이라 결정적·바이트 안정.
- 목표: `mean_tour_length`(평균 tour 길이) **최소화**.

편집 가능한 표면(`src/train.py`)에는 두 종류의 손잡이가 있다:

```python
# --- HYPERPARAMS-BEGIN (auto-patched; do not edit by hand) ---
HYPERPARAMS = {
    "use_nn_construction": True,   # nearest-neighbor 초기해 사용 여부
    "max_iterations": 20000,       # local search 반복 횟수
    "restarts": 1,                 # iterated local search 재시작 횟수
    "initial_temperature": 0.0,    # simulated annealing 초기 온도(0=순수 hill climbing)
    "cooling_rate": 0.995,         # 온도 감쇠율
    "segment_max": 3,              # or-opt 세그먼트 최대 길이
    "perturbation_strength": 4,    # 재시작 전 킥(perturbation) 강도
}
# --- HYPERPARAMS-END ---

NEIGHBORHOOD = "two_opt"   # or "or_opt" — 코드 수준 손잡이(LLM 코더만 바꿀 수 있음)
```

- **결정적 패처(patcher)**: 위 `HYPERPARAMS` 블록에서 **정확히 값 1개**를 바꾼다.
  (예: `restarts: 1 → 2`) — 완전 오프라인·결정적·무비용.
- **LLM 코딩 워커(coder)**: `src/**` 아무 데나 편집할 수 있다. 예를 들어
  `NEIGHBORHOOD`를 `two_opt → or_opt`로 바꾸거나, 수락 규칙을 바꾸거나, tabu
  메모리를 추가하는 등 **알고리즘 자체**를 손대는 "진짜 연구" 표면. `--proposer
  claude`일 때만 등장한다.

### 1.2 evaluator-first: 왜 평가기가 먼저인가

ML에서 reward model이 엉터리면 RLHF가 reward hacking으로 망가지듯, 여기서도
평가기가 뚫리면 나머지 전부가 무의미해진다. 그래서 이 저장소는 평가기를 **신뢰
경로(trusted path)**로 격리하고 다음을 구조적으로 보장한다:

- **루트 평가기 사본만 권위**를 갖는다. worktree(실험용 격리 작업트리) 안의
  `evaluation/` 복사본은 채점에 절대 쓰이지 않는다.
- **숨은 시드의 물리적 부재**: dev/gate/test 시드는 `evaluation/heldout_config.json`
  에 있는데 이 파일은 **git이 추적하지 않는다**. worktree는 추적 파일만
  체크아웃하므로 후보 작업공간에는 시드가 아예 없다. (train/val/test 누수 방지의
  물리적 버전)
- **nonce 에코**: 매 평가마다 새 nonce를 넘기고 평가기가 metrics에 그대로 되
  적어야 한다 → 위조 metrics 차단.
- **자기보고 무시**: solver의 `reported_objectives`는 버리고 평가기가 재계산.
- **SHA-256 매니페스트**: 보호 파일(`orchestrator.py`, `evaluation/**`,
  `research_contract.yaml`, `literature/**`, `assurance/**`, `sandbox/**` 등 23개)
  을 매 라운드 해시 대조.

### 1.3 무엇이 편집 가능하고 무엇이 보호되나

| 경로 | 상태 | 설명 |
|---|---|---|
| `src/**` | ✏️ 편집 대상 | solver 코드 (patcher/coder가 여기만 건드림) |
| `research_contract.yaml` | 🔒 보호 (0o444) | 연구 계약. 런 중 불변 |
| `orchestrator.py` | 🔒 보호 | 코디네이터·게이트·CLI |
| `evaluation/**` | 🔒 보호 | 평가기·데이터셋 (`heldout_config.json`은 git 미추적) |
| `literature/**` | 🔒 보호 | 문헌 엔진 + corpus (증거 프로버넌스) |
| `assurance/**` | 🔒 보호 | 통계·claim·보고서 생성 |
| `sandbox/**` | 🔒 보호 | 실행 격리 경계 |
| `protection/hashes.json` | 🔒 보호(git 추적) | SHA-256 매니페스트 |
| `experiments/**` | 런타임(gitignore) | 상태·원장·라운드·보고서 |
| `.worktrees/**` | 런타임(gitignore) | 실험별 격리 작업트리 |

> **"보호 파일을 일부러 고치려면"** → 반드시 §9.3 절차(`chmod u+w` → 편집 →
> `init --force`)를 따른다. 그냥 고치면 다음 `run`/`report`가 보호 위반으로 멈춘다.

---

## 2. 설치와 사전 준비

전제: `uv`(파이썬 패키지·가상환경 매니저)와 Python 3.13+ 툴체인.
(`.python-version`은 3.14로 고정.)

```bash
cd /Users/gyubin.son/workspace/dev/autoresearch
uv sync    # 의존성 설치: pyyaml + claude-agent-sdk 둘 뿐
```

- **오프라인 기본**: 휴리스틱 proposer + lexical 문헌 + scalar gate는 **SDK/네트워크
  전혀 안 쓴다.** 계정 로그인 없이 바로 돌아간다.
- **LLM 경로**(옵트인): `--proposer claude`, `--literature claude`,
  `--gate pairwise`, `--reviewer codex`는 로컬 Claude Code 로그인(별도 API 키 불필요)
  또는 로컬 `codex` 로그인을 재사용한다. 계정 사용량 한도에 걸릴 수 있고, 걸리면
  결정적 경로로 폴백한다.
- **container 샌드박스**(옵트인): Docker 데몬 + 미리 pull한 핀 이미지 필요(§8.5).

---

## 3. 5분 퀵스타트 (복붙용)

아무것도 모르고 그냥 한 판 돌려보고 싶다면 이걸 그대로 복붙하면 된다. 전부
오프라인·결정적이라 안전하다.

```bash
cd /Users/gyubin.son/workspace/dev/autoresearch
uv sync

# 1) 초기화: git·베이스라인·숨은 시드·보호 매니페스트 (이미 되어 있으면 생략)
uv run python orchestrator.py init

# 2) (선택) 문헌 근거 인증서 — 연구 질문이 어떤 선행연구 위에 서는지
uv run python orchestrator.py ground

# 3) 캠페인 실행: 3세대 병렬 포트폴리오 (휴리스틱·오프라인)
uv run python orchestrator.py run --generations 3

# 4) 현황 확인
uv run python orchestrator.py status

# 5) 보고 → 처음엔 사람 승인을 요구하며 exit 3 + request_id 를 낸다
uv run python orchestrator.py report        # exit 3, request_id 출력됨

# 6) 그 request_id 로 출판 의도 승인
uv run python orchestrator.py approve <위에서_나온_request_id>

# 7) 다시 보고 → 이번엔 test 스플릿 다중시드 평가 + bootstrap CI + report.md 봉인
uv run python orchestrator.py report

# 결과 보기
open experiments/report/report.md   # 또는: cat experiments/report/report.md
```

> **exit 3은 에러가 아니다.** "사람 승인이 필요하다"는 신호다. 스크립트에서는
> `0=봉인 완료 / 1=에러 / 2=인자 오류 / 3=승인 필요`로 구분해 처리하라.

전체 LLM 경로(옵트인)로 돌리고 싶으면:

```bash
uv run python orchestrator.py run --generations 3 \
  --proposer claude --literature claude --gate pairwise
```

---

## 4. 꼭 알아야 할 8개 개념

프레임워크를 제대로 쓰려면 이 8개만 확실히 잡으면 된다.

### 4.1 generation vs round(experiment)

- **round(=experiment)**: 가설 하나를 실행한 최소 단위. `rNNNN`(r0001, r0002…)로
  번호가 매겨지고, 저장소 예산은 `budgets.max_rounds`(기본 60)로 센다.
- **generation**: 한 판에 **K개(계약 `portfolio.parallel_branches`, 기본 8)의 가설을
  병렬로** 내놓고 승자 1명을 뽑는 단위. `run --generations N`의 N은 **generation**
  수다. 휴리스틱 proposer는 파라미터당 최대 1개 가설(약 6~7개)만 채우므로 K=8은
  상한이지 할당량이 아니다.

> 비유: generation = population-based training의 한 세대(여러 후보 동시 평가 후
> 선택), round = 그 세대 안의 개체 하나.

### 4.2 verdict ≠ decision (가장 헷갈리는 지점)

- **verdict**는 "과학적 판정"이다: dev 스플릿에서 무슨 일이 있었나.
- **decision**은 "채택 여부"다: 최종적으로 main에 머지됐나(`accept`/`reject`).

**`verdict=valid_positive`(dev에서 개선됨)이어도 `decision=REJECT`일 수 있다** — dev에서
좋아 보였지만 blind gate를 못 넘었거나, 같은 세대의 다른 후보가 승자가 됐기
때문이다. 한 세대에서 **gate 승자 딱 1명만** `accept`된다.

verdict 전체 목록:

| verdict | 뜻 | 과학적 신호? |
|---|---|---|
| `valid_positive` | dev 상대개선 ≥ `min_relative_improvement`(0.2%) | ✅ (gate 후보) |
| `valid_inconclusive` | 변화가 임계 미만 | ✅ (약한 신호) |
| `valid_negative` | 지표 악화 / 발산 / no_skill / infeasible / timeout / crash | ✅ (반증 증거) |
| `pruned` | successive halving 컷 (예산 결정) | ❌ (과학 아님) |
| `invalid_implementation` | 패치 실패 / 코더 기계적 실패 / 비결정성 | ❌ (기계 실패) |
| `contract_violation` | 보호 경로 접촉 / src symlink / 과대 diff | ❌ (프로토콜 위반) |
| `ff_conflict` | 승자인데 머지 직전 main이 움직임/더러워짐 | ❌ (인프라) |
| `aborted` | 평가기 인프라 crash / 중단 | ❌ (인프라) |

> **false-repair 금지 원칙**: 유효하게 패치된 후 발생한 런타임 실패(발산·timeout·
> no_skill·infeasible·dev 단계 실패)는 전부 **수리하지 않고** `valid_negative`로
> 증거화한다. "그럴듯해 보이게 고쳐서" 가짜 진전을 만드는 걸 구조적으로 막는다.
> 수리는 오직 **smoke 단계의 기계적 실패**(nonzero_exit / missing_artifact /
> malformed_artifact)에만, 아직 채점 가능한 답을 못 만든 경우에 한해 허용된다.

### 4.3 blind admission gate와 blindness (일반화 재확인)

한 세대에서 dev 개선 후보(`valid_positive`) 상위 `gate_top_k`(기본 2)명이 **gate
스플릿**(dev와 완전히 다른 숨은 시드)에서 다시 채점된다. incumbent(현 챔피언)의
gate 점수를 `gate_min_relative_improvement`(기본 0.1%)만큼 이겨야 **admitted(승인)**
되고, 승인된 후보 중 승자 1명만 main에 ff-merge된다.

> 왜? dev에서 미세하게 좋아졌지만 일반화 안 되는 후보(=validation overfitting)를
> 거른다. dev=validation, gate=또 다른 held-out validation이라고 보면 된다.

**blindness 불변식(매우 중요):** gate 점수는 오직 두 곳에만 존재한다 —
`record_type=gate` 원장 레코드와 `experiments/generations/gNNNN/gate/*.json`. gate
점수는 **insight, `best_primary`(항상 dev 점수), proposer 컨텍스트, search momentum,
콘솔 출력, report/claim 어디에도 절대 안 들어간다.** 콘솔에는 PASS/FAIL과 승자
run_id만 찍히고 점수는 "scores withheld"로 감춘다. (실측 예: §6의 g0003에서 dev
승자 점수는 6351315.225인데 gate 점수 6320049.425는 gate 레코드에만 있다.)

### 4.4 search momentum (Gome 스타일 방향 조향)

매 세대 시작 시, 원장의 experiment 레코드만 접어서 `{param}:{move}`(예:
`restarts:increase`)별 방향 점수를 만든다. **state에 저장하지 않고 매번 원장에서
재계산**하므로 crash에서 복구해도 replay==live가 자명하게 성립한다.

가중치(방향이 먹히는지의 신호):

| 결과 | 가중치 |
|---|---|
| `valid_positive` + accept | **+1.0** |
| `valid_positive` (gate 탈락) | +0.4 |
| `valid_negative` (악화/발산/timeout) | −1.0 (+발산 경계값 기록) |
| `valid_inconclusive` | −0.2 |
| `pruned` / invalid / contract_violation | 0.0 |

매 세대 경계마다 `momentum_decay`(0.5)로 감쇠. 휴리스틱 proposer는 이 momentum을
1순위, 문헌 stance를 2순위, 정적 우선순위를 3순위로 후보를 재정렬한다. 한
(param, direction)에서 연속 accept가 `accelerate_after`(2)회 쌓이면 **가속 스텝(제곱
스텝)**을 한 번 쓴다. §6의 사례에서 `restarts`가 1→2→4→**16**으로 점프하는 게 바로
이 가속(4²)이다. 또 K개 슬롯 중 최소 1개는 **momentum 0·문헌 미지지 방향**으로
강제 예약해 가설 붕괴(한 방향으로만 파는 것)를 막는다.

> 입력이 dev 신호와 accept/reject 비트뿐이라 **gate 점수가 구조적으로 못 들어온다**
> (blindness 유지). `refinement.enabled=false`면 이 조향이 꺼지고 Phase 3 동작과
> 바이트 동일해진다.

### 4.5 successive halving (예산 절약)

한 세대의 K개 브랜치가 전부 값싼 **smoke rung**(짧은 학습, dev split, 30초)을 돌고,
smoke 점수 상위 `max(min_keep, ceil(K·keep_fraction))` = `max(2, ceil(8·0.5))=4`개만
비싼 **dev rung**(전체 학습, 120초)으로 올라간다. 탈락은 verdict `pruned`.

> HPO의 successive halving 그대로다. **중요:** pruned는 **예산 결정이지 과학 아님** —
> insight를 증류하지 않고 momentum 가중치도 0이다. "pruned = 이 가설 실패"로 읽지
> 마라. 다만 재제안 방지를 위해 tested endpoint는 등록한다.

### 4.6 literature grounding (claim 수준 증거 그래프)

가설이 문헌 근거 위에 서게 한다. `literature/`는 **별도 통제 서비스**다:
orchestrator를 import하지 않고, corpus 외 어떤 파일도 읽지 않으며, 런타임에
아무것도 쓰지 않는다(폐포).

- **오프라인 corpus** (`literature/corpus/tsp_corpus.json`, `tsp-heuristics-v1`):
  TSP 휴리스틱 논문 13편 / claim 13개. 모순쌍, citation-laundering 트랩,
  prompt-injection 픽스처까지 포함. (지금은 큐레이션된 mock 스냅샷 —
  provenance 비어 있음. 실 문헌으로 갱신하려면 §8.6의 `ground --refresh`.)
- **기본 lexical**: 결정적 토큰 오버랩 검색 + 인용 BFS(1홉) + coverage 정지. 완전
  오프라인.
- **`--literature claude`**: 쿼리 분해·stance 판정·서술만 LLM이 맡되 **검색 실행은
  항상 결정적 백엔드**. 반-론더링 규칙: **LLM은 결정적 "supports"를 강등만 할 수
  있고 새로 부여할 수 없다.**
- 가설에는 **증거 id만** 실린다(`supporting_evidence_ids`, 화이트리스트 검증됨).
  claim 산문은 blindness 스캔 표면에 절대 안 들어간다. novelty는 숫자 없이 범주만:
  `replication / regime_extension / contradiction_test / unexplored`.

### 4.7 human approval gate (출판 아날로그)

미접촉 **test 스플릿**은 논문 낼 때 test set 한 번 쓰는 것과 같다. 그래서 처음
`report`(와 매 `--force` 재실행)는 **test 숫자를 계산하기 전에** 사람 승인을
요구한다:

1. `report` → 승인 요청을 원장에 적고 **exit 3 + request_id** 출력.
2. 사람이 의도(commit·dev 숫자·시드 계획·공시)를 확인.
3. `approve <request_id>` → 승인 기록.
4. 다시 `report` → 진행.

승인 상태는 state가 아니라 **원장에서 파생**된다. fingerprint(=incumbent·baseline
commit + contract·evaluator sha + 이전 봉인 수)가 캠페인이 전진하면 바뀌므로,
승인 후 `run`을 더 돌리면 **승인이 stale(무효)**이 되어 재승인이 필요하다.

> 코더 실행·예산 상향 같은 내부 작업은 일부러 승인 게이트를 안 건다 — 사소한
> 승인 피로가 쌓여 정작 중요한 "돌이킬 수 없는 결정"을 도장 찍듯 넘기게 되는 걸
> 막기 위해서다.

### 4.8 sandbox backend와 trust grade

후보 코드가 실제로 *실행*되는 지점(`evaluation/evaluate.py`의 `_run_train`)만 OS로
격리한다. 채점은 호스트 신뢰 경로에 남고 시드는 거기로 안 넘어간다.

- **`subprocess`(기본)**: OS 격리 없음. 현행 동작과 바이트 동일, Docker 불필요.
  **gate/test에서는 신뢰 등급이 아니다** — solver가 절대경로로 숨은 시드 파일을
  읽어 과적합할 수 있다. 그래서 gate/report마다 큰 경고가 뜬다.
- **`container`(옵트인)**: `docker run`으로 네트워크 차단·읽기전용 rootfs·권한
  드롭·비루트·리소스 제한·ephemeral PID 네임스페이스 + **숨은 시드 파일과 원장을
  마스킹**(컨테이너 FS에 물리적 부재). 이때만 gate/test 점수가 **trust-grade**다.

정책: `sandbox.require_container_for_trusted_splits: true`로 두면 subprocess에서
gate/report를 하드 에러로 막는다(기본은 경고만). report.md 헤더에도 신뢰 등급이
찍힌다. 자세한 셋업은 §8.5.

> 정직한 한계: **기본 subprocess에서 낸 숫자는 "정직해 보이지만 신뢰 등급은
> 아니다."** 결과를 남에게 주장하려면 container로 재현하라.

---

## 5. 명령어 레퍼런스

공통 규칙:
- 호출형: `uv run python orchestrator.py <subcommand> [flags]`.
- `init`/`run`/`report`/`ground`/`approve`는 **단일 인스턴스 잠금**(flock,
  `.orchestrator.lock`)을 잡는다 — 동시에 두 개 돌리면 두 번째가
  `error: another orchestrator process is running ...`로 실패. `status`/
  `verify-protection`은 잠금 없이 아무 때나 안전.
- exit code: `0`=성공, `1`=OrchestratorError(에러, stderr에 `error: ...`),
  `2`=인자 오류, `3`=report 승인 필요.

### 5.1 `init` — 초기화·베이스라인

```bash
uv run python orchestrator.py init [--force]
```

무엇을: git 저장소 준비(없으면 `git init`, gc.auto=0) → 계약과 평가기의 하드코딩
상수 **교차검증**(budget/metric/split/seed 상한/N_CITIES/sandbox backend — 어긋나면
`... drift`로 즉시 중단) → 숨은 dev/gate/test 시드 생성(`heldout_config.json`,
schema v4, test 시드 N=`finalist_seeds`개) → 보호 매니페스트 작성 → 초기 커밋 →
**베이스라인 dev 평가** → 보호 파일 읽기전용(0o444).

`--force`: `experiments/`를 **통째로 삭제**하고 시드·매니페스트 재생성 +
재베이스라인. 이전 캠페인 기록이 필요하면 먼저 백업. 계약/평가기/도메인을 바꾼
뒤에는 이걸로 새 캠페인을 시작해야 한다.

실측 출력(끝 두 줄):
```
[init] baseline mean_tour_length (dev) = 6579300.425000 at f31e2eadfaae
[init] protected files set read-only; ready: `uv run python orchestrator.py run --generations N`
```

주의: 이미 초기화됐는데 `--force` 없이 `init`하면
`already initialized (experiments/state.json exists); use --force to re-baseline`.

### 5.2 `ground` — 문헌 근거 인증서 (+ `--refresh` 유지보수)

```bash
uv run python orchestrator.py ground [--literature lexical|claude] [--model NAME]
```

무엇을: 계약의 `objective`에 대해 문헌 evidence flow를 돌려 **연구질문 인증서**를
만든다. `experiments/evidence/question_certificate.json`에 쓰고, evidence.jsonl에
`kind=question_grounding` 번들을 붙이고, 인증서 JSON을 stdout에 출력.

`--refresh`는 완전히 다른 **유지보수** 작업(실 API로 corpus 재생성) — §8.6 참조.

### 5.3 `run` — 캠페인 실행

```bash
uv run python orchestrator.py run \
  [--generations N] \                 # 기본 3
  [--proposer heuristic|claude] \     # 기본 heuristic (claude면 코더도 켜짐)
  [--model NAME] \                    # claude proposer/coder 모델 override
  [--max-budget-usd X] \              # claude proposer 제안당 상한, 기본 0.5
  [--gate scalar|pairwise] \          # 기본 scalar
  [--literature lexical|claude]       # 기본 lexical
```

무엇을: `--generations`판을 돈다. 각 세대: 보호검증 → momentum·문헌 그라운딩 →
K개 가설 제안 → 격리 worktree 병렬 실행(smoke rung → halving 컷 → dev rung) →
blind gate admission → (pairwise면) 심판 선택 → 승자 ff-merge → insight 증류.

**중요한 진입 조건:**
- 초기화 안 됐으면 `not initialized — run orchestrator.py init first`.
- main 작업트리에 **추적 파일 uncommitted 변경**이 있으면 거부(
  `... has uncommitted tracked changes ...; commit or restore them first`). 세대는
  HEAD에서 브랜치를 따므로 더러운 트리는 제안과 실제 실행을 어긋나게 한다.

정지 조건: `max_rounds`(60 도달) / `max_generations` / `stagnation`(4세대 연속
무승자) / `search_space_exhausted`(제안 없음).

플래그 의미의 원칙 — **계약이 "여부"를, CLI가 "방법"을 정한다.** 계약에서
`pairwise_gate.enabled=false`면 `--gate pairwise`는 조용히 무시되고 경고만 뜬다.
`--literature claude`도 `literature.enabled=false`면 무시.

실측 출력(§6 전체):
```
— generation g0001 —
  [r0001] restarts: 1 -> 2  mean_tour_length=6508432.5000  verdict=valid_positive  decision=ACCEPT
  ...
  [gate] candidates ['r0001'] -> winner r0001
...
generations executed: 3 (total 3; experiments 15); stop: requested generations done
mean_tour_length (dev): baseline 6579300.425000 -> best 6351315.225000 (+3.47% relative)
incumbent commit: 4796a8c70e16  stagnation: 0 generations
```

### 5.4 `status` — 현황 대시보드 (읽기전용)

```bash
uv run python orchestrator.py status
```

무엇을: 계약·지표·실험/세대 카운트·baseline/best(dev)·stagnation, 최근 12개 실험,
최근 5개 gate 결정(**점수 감춤**), search momentum(원장 파생), 문헌 통계, 승인/
리뷰 상태를 출력. 잠금 안 잡음 — `run` 도는 중에도 안전.

### 5.5 `report` — 최종 보고 (test 스플릿 1회성)

```bash
uv run python orchestrator.py report [--force] [--reviewer none|codex] [--model NAME]
```

무엇을(승인된 경우): sandbox preflight → 신뢰정책 경고/차단 → write-ahead
`report_attempt` 기록 → baseline·incumbent를 **N=`finalist_seeds`(5)개 test 시드**에서
평가 → paired bootstrap CI → 증거 감사(해석 불가 인용이면 하드 에러) →
`claims.jsonl` + `figures/*.svg` + `report.md` + `report.json` 작성 → (옵트인) codex
리뷰 → `final_report` 봉인 → 전체 report JSON을 stdout 출력.

- 승인 안 됐으면 **exit 3**(§4.7).
- 이미 봉인된 보고가 있는데 `--force` 없으면:
  `N final report(s) already exist — the test split is single-use. ...`. `--force`는
  새 의도(재승인 필요) + 다중검정 공시 카운터 증가.
- `--reviewer codex`는 `reviewer.enabled: true`도 있어야 실제로 돈다. 실패해도
  보고를 막지 않고 `status="unavailable"`로 기록(Claude로 폴백 안 함).

### 5.6 `approve` — 출판 의도 승인/거부

```bash
uv run python orchestrator.py approve <request_id> [--deny] [--reason "텍스트"]
```

무엇을: 원장에 승인/거부 결정을 append. `request_id`는 **고유 접두사**만 줘도 됨
(예: `approve 23ee75` OK). `--deny`로 거부, 나중에 같은 id를 `approve`하면 거부를
뒤집을 수 있다. fingerprint 신선도는 여기서 검사하지 않는다 — `report` 시점에
검사(그래서 stale 승인이 걸러진다).

### 5.7 `verify-protection` — 무결성 점검

```bash
uv run python orchestrator.py verify-protection
```

`OK — 23 protected files match the manifest`(exit 0) 또는 위반마다
`VIOLATION: <파일>`(exit 1). 잠금 없음.

---

## 6. 사례 워크스루 — 실제 TSP 캠페인 한 판

아래는 이 저장소에서 **실제로 방금 돌린** 캠페인 `c20260718051332`의 전 과정이다.
숫자·출력이 전부 실측이다.

### Step 0 — 출발점 확인

```bash
$ uv run python orchestrator.py status
contract:   autoresearch-phase6c-tsp
objective:  Minimize the mean held-out Euclidean-TSP tour length produced by the solver in src/train.py across h
metric:     mean_tour_length (minimize, min rel improvement 0.2%; gate epsilon 0.10%)
experiments: 0 / 60  generations: 0
baseline:   6579300.425000 (dev)
best:       6579300.425000 (dev) at f31e2eadfaae
stagnation: 0 / 4 generations

approval (report intent 3285689c7b88): none

$ uv run python orchestrator.py verify-protection
OK — 23 protected files match the manifest
```

읽는 법: 아직 실험 0개, incumbent==baseline(6,579,300). 지표는 최소화, dev 개선
임계 0.2%, gate 임계 0.10%.

### Step 1 — 문헌 그라운딩

```bash
$ uv run python orchestrator.py ground
```

출력(발췌)은 연구질문 인증서다. 핵심만:
```json
{
  "mode": "lexical",
  "evidence_counts_by_stance": {"supports": 7, "adjacent": 4, "contradicts": 1},
  "contradictions": [{"topic": "acceptance_criterion",
                      "evidence_ids": ["ev_cl-0501", "ev_cl-0601"]}],
  "coverage": {"stopped_because": "coverage_stable", "queries_run": 3,
               "topics_uncovered": ["initial_temperature"]}
}
```

읽는 법: corpus에서 support 7 / contradict 1 / adjacent 4개의 증거를 찾았고,
`acceptance_criterion` 주제에서 **모순쌍**(한 논문은 개선, 다른 논문은 악화 주장)을
탐지했다. coverage가 안정돼 3쿼리 만에 멈췄다.

### Step 2 — 3세대 캠페인 실행

```bash
$ uv run python orchestrator.py run --generations 3
```

stderr에 먼저 신뢰 경고가 3번 뜬다(gate 스플릿, subprocess 백엔드):
```
[warn] gate split runs under the 'subprocess' sandbox backend, which has NO
filesystem isolation: a candidate solver can read the held-out seed file by
absolute path and overfit the hidden instances, so its gate score is not
trust-grade. Set sandbox.backend: container for a trustworthy gate result.
```

그리고 세대별 진행:

```
— generation g0001 —
  [r0001] restarts: 1 -> 2               mean_tour_length=6508432.5000        verdict=valid_positive       decision=ACCEPT
  [r0002] use_nn_construction: True -> False  mean_tour_length=smoke 18318703.1000  verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [r0003] max_iterations: 20000 -> 50000  mean_tour_length=6573762.8750        verdict=valid_inconclusive   decision=REJECT
  [r0004] initial_temperature: 0.0 -> 0.5  mean_tour_length=smoke 7371016.5000   verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [r0005] cooling_rate: 0.995 -> 0.99     mean_tour_length=6579300.4250        verdict=valid_inconclusive   decision=REJECT
  [r0006] segment_max: 3 -> 4             mean_tour_length=6579300.4250        verdict=valid_inconclusive   decision=REJECT
  [r0007] perturbation_strength: 4 -> 8   mean_tour_length=smoke 7324408.0250   verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [gate] candidates ['r0001'] -> winner r0001
— generation g0002 —
  [r0008] restarts: 2 -> 4                mean_tour_length=6455142.9000        verdict=valid_positive       decision=ACCEPT
  [r0009] max_iterations: 20000 -> 8000   mean_tour_length=6575426.8500        verdict=valid_negative (metric_regression)  decision=REJECT
  [r0010] initial_temperature: 0.0 -> 1.0  mean_tour_length=smoke 7371016.5000   verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [r0011] cooling_rate: 0.995 -> 0.999    mean_tour_length=6508432.5000        verdict=valid_inconclusive   decision=REJECT
  [r0012] segment_max: 3 -> 2             mean_tour_length=smoke 7324408.0250   verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [r0013] perturbation_strength: 4 -> 2   mean_tour_length=smoke 7324408.0250   verdict=pruned (smoke_rank_below_cutoff)  decision=REJECT
  [gate] candidates ['r0008'] -> winner r0008
— generation g0003 —
  [r0014] restarts: 4 -> 16               mean_tour_length=6351315.2250        verdict=valid_positive       decision=ACCEPT
  [r0015] initial_temperature: 0.0 -> 2.0  mean_tour_length=6462573.6250        verdict=valid_inconclusive   decision=REJECT
  [gate] candidates ['r0014'] -> winner r0014

generations executed: 3 (total 3; experiments 15); stop: requested generations done
mean_tour_length (dev): baseline 6579300.425000 -> best 6351315.225000 (+3.47% relative)
incumbent commit: 4796a8c70e16  stagnation: 0 generations
```

**이 한 판에서 실제로 벌어진 과학 이야기:**

1. **halving이 값싸게 절반을 쳐냈다.** g0001에서 `use_nn_construction=False`,
   `initial_temperature=0.5`, `perturbation_strength=8`은 smoke rung 점수가 나빠서
   dev rung에 못 올라가고 `pruned`됐다(`mean_tour_length=smoke ...`로 표시). 이건
   "이 가설이 틀렸다"가 아니라 "예산상 여기서 컷"이라는 뜻이다.

2. **restarts가 유일하게 일관되게 먹혔다.** g0001에서 `restarts 1→2`만 dev를
   유의미하게(6,579,300 → 6,508,432) 개선해 gate를 통과했다. 다른 것들은
   inconclusive.

3. **momentum이 방향을 밀었다.** g0002에서 다시 `restarts 2→4`가 이겼고,
   g0003에서 momentum이 `restarts:increase`를 강하게 밀며 **가속 스텝**을 써서
   `restarts 4→16`(=4²)으로 점프해 또 이겼다 (momentum 표는 목록 아래 참고).

4. **valid_negative = 진짜 반증.** g0002의 `max_iterations 20000→8000`은 지표가
   악화돼(`metric_regression`) valid_negative가 됐다. 이건 "반복을 줄이면 나빠진다"는
   증거로 남고, momentum에서 `max_iterations:decrease: -0.50`으로 반영된다.

5. **정직한 해석:** 이 캠페인이 찾은 개선은 본질적으로 "재시작(=iterated local
   search)을 더 많이 하면 tour가 짧아진다"는 것이다. 실제로 맞는 말이지만 그 대가는
   **연산량 16배**다. `mean_tour_length`만 최소화하는 현 목표에서는 정당한 승리지만,
   "연산 예산 대비 품질"을 보고 싶다면 목표 함수에 `solve_seconds`를 넣어야 한다
   (§9의 교훈). 프레임워크는 이 트레이드오프를 숨기지 않고 report.md의 Limitations와
   secondary metric으로 드러낸다.

위 3번의 momentum을 `status`로 확인하면 이렇게 찍힌다 — `restarts:increase`가 +1.75로
최상위라, 연속 accept가 이 방향을 확신하게 만든 것이 눈에 보인다:

```text
search momentum (ledger-derived, dev signals only):
  restarts:increase: score +1.75  last=accepted
  max_iterations:decrease: score -0.50  last=valid_negative
  initial_temperature:increase: score -0.20  last=valid_inconclusive
  ...
```

### Step 3 — 보고 시도 (사람 승인 게이트)

```bash
$ uv run python orchestrator.py report
approval required before the test split is touched.
  request_id : 23ee75978e39
  intent     : incumbent 4796a8c70e16 vs baseline f31e2eadfaae, 5 test seed(s)
  review it, then: uv run python orchestrator.py approve 23ee75978e39
$ echo $?
3
```

exit 3. 아직 test 숫자는 하나도 계산 안 됐다. 원장에 `approval_request`만 적혔다.

### Step 4 — 승인

```bash
$ uv run python orchestrator.py approve 23ee75978e39
approve recorded for request_id 23ee75978e39
```

### Step 5 — 보고 봉인

```bash
$ uv run python orchestrator.py report
[warn] test split runs under the 'subprocess' sandbox backend ... not trust-grade ...
{ ... 전체 final_report JSON ... }
```

이 단계는 baseline·incumbent를 각 5개 test 시드(각 160 인스턴스)에서 재현하므로
시간이 좀 걸린다(수백 초). 끝나면 `experiments/report/`에 산출물이 쌓인다.

### Step 6 — 결과 읽기 (`report.md`)

`experiments/report/report.md` 실측 발췌:

```markdown
# AutoResearch Campaign Report — autoresearch-phase6c-tsp

Campaign c20260718051332 · baseline f31e2eadfaae · incumbent 4796a8c70e16 · 2026-07-18

Execution isolation: subprocess — NOT trust-grade: no filesystem isolation, so the
solver could read the held-out seed. Re-run under sandbox.backend: container to
trust these numbers.

## Headline result
**Status: verified.** Pooled mean test tour length 6554278.24 → 6303584.31; effect
250693.93 (3.82% relative), 95% CI [237076.58, 264382.16] across 5 hidden test
seed(s) × 800 instances. [claim_0001]

## Per-seed reproduction
| seed | baseline tour length | incumbent tour length | delta |
|------|----------------------|-----------------------|-------|
| s0 | 6585181.16 | 6313810.64 | 271370.52 |
| s1 | 6545216.19 | 6302670.83 | 242545.36 |
| s2 | 6571730.65 | 6331909.84 | 239820.81 |
| s3 | 6536335.89 | 6282253.64 | 254082.25 |
| s4 | 6532927.31 | 6287276.59 | 245650.73 |
Seed consistency (fraction improving): 100.00%.

## Admitted interventions
- restarts 1 -> 2 improved dev ... generation 1 and passed the blind admission gate. [claim_0003]
- restarts 2 -> 4 improved dev ... generation 2 and passed the blind admission gate. [claim_0004]
- restarts 4 -> 16 improved dev ... generation 3 and passed the blind admission gate. [claim_0005]

## Negative results
- Intervening on max_iterations did not help: 1 valid negative result(s) (failure
  classes: metric_regression). ... [claim_0006]

## Literature grounding
- Accepted change r0001/r0008/r0014 is grounded in 1 evidence record(s) from prior
  work (tsp:2018.0009). [claim_0007..0009]
```

**해석의 핵심 — `Status: verified`가 무슨 뜻인가:**
- 5개 **완전히 미접촉인 test 시드**에서 baseline 대비 평균 tour가 6,554,278 →
  6,303,584로 줄었다(effect 250,694, 3.82%).
- 95% bootstrap CI가 **[237076, 264382]로 0을 포함하지 않는다** → 개선이 통계적으로
  확실 → `verified`. (CI가 0을 걸치면 `inconclusive`, 상한<0이면 `refuted`, test
  시드가 하나라도 unclean이면 `unsupported`.)
- **seed consistency 100%** — 5개 시드 전부 개선. 우연이 아니다.
- 결정적으로, **dev에서의 +3.47%가 미접촉 test에서 +3.82%로 재현됐다.** blind
  gate가 dev-overfitting을 걸러낸 덕분에 dev 개선이 진짜로 일반화됐다는 실증이다.

> report.md의 **모든 숫자는 claim에서만 삽입**되고, 렌더 후 digit-scan이 추적 불가
> 숫자를 하드 거부한다(숫자-via-claim 불변식). 즉 "이 문서의 어떤 숫자든
> `experiments/claims.jsonl`의 어떤 claim에서 왔는지 추적 가능"이 구조적으로
> 보장된다.

---

## 7. 출력물 읽는 법

### 7.1 `experiments/` 트리 지도 (한 판 후)

```
experiments/
├── state.json                     # 재개 가능한 캠페인 포인터(현 상태)
├── ledger.jsonl                   # ★ 진실의 원천 (append-only, 9종 record_type)
├── baseline/metrics_dev.json      # init 베이스라인 평가
├── rounds/rNNNN/
│   ├── hypothesis.json            # 가설 인증서(원장 쓰기 전에 먼저 기록)
│   ├── metrics_smoke.json         # smoke rung 평가
│   └── metrics_dev.json           # dev rung 평가
├── generations/gNNNN/
│   ├── steering.json              # momentum·move_guidance·batch (순수 감사용)
│   ├── evidence.json              # 세대 문헌 그라운딩 스냅샷
│   └── gate/                      # ★ gate 점수가 존재하는 유일한 곳
│       ├── incumbent.json
│       ├── <run_id>.json          # 후보 gate-split 점수
│       └── <run_id>_dev_recheck.json  # 결정성 재검
├── evidence/
│   ├── evidence.jsonl             # 증거 메모리(원장과 별도, append-only)
│   └── question_certificate.json  # `ground` 인증서
├── claims.jsonl                   # report 시점 전체 재생성(5종 claim)
└── report/
    ├── report.md                  # ★ 사람이 읽는 결론
    ├── report.json                # final_report 레코드의 사본(기계용)
    ├── {baseline,incumbent}_test_sK.json  # 시드별 test 평가
    ├── figures/*.svg              # dev_trajectory / test_paired_rmse / verdict_mix
    └── review/<request_id>/       # codex 리뷰(옵트인일 때만)
```

루트 옆에: `insight_memory.json`(원장에서 재구성 가능), `artifacts/solution.json`
(평가마다 덮어씌워지는 임시 산출물 — **per-round 아카이브 아님**),
`.worktrees/`, `evaluation/heldout_config.json`(숨은 시드, git 미추적),
`protection/hashes.json`(매니페스트, git 추적).

> **전부 gitignore다.** 프로버넌스는 `hyp/*` git 브랜치 + `experiments/ledger.jsonl`
> 로만 살아남는다. `init --force`는 `experiments/`를 지우므로 필요하면 먼저 백업.

### 7.2 원장 record_type 사전

`experiments/ledger.jsonl`은 append-only JSONL. 이번 캠페인이 만든 것:
`baseline`×1, `experiment`×15, `gate`×3, `approval_request`×1, `approval_decision`×1,
`report_attempt`×1(그리고 봉인 후 `final_report`×1).

| record_type | 언제 | 핵심 필드 |
|---|---|---|
| `baseline` | init 1회 | commit, primary, metrics_path |
| `experiment` | 후보마다 | hypothesis(인증서), verdict, decision, primary, smoke_primary, best_primary_before/after, executor, failure_class, coder_family |
| `gate` | 세대마다(맨 먼저 기록) | candidates, results{run_id:점수}, incumbent_gate, winner, scalar_winner, mode, pairwise, reason, selection_rule |
| `correction` | 채택 후 머지 실패시 | corrects(무효화된 run_id), reason |
| `approval_request` | report exit 3 | request_id, fingerprint(5키), payload(dev숫자·시드계획·공시) |
| `approval_decision` | approve | request_id, decision(approve/deny), reason |
| `report_attempt` | test 쓰기 직전 | request_id, fingerprint (write-ahead 마커) |
| `review` | codex 리뷰시 | status, overall, review_path, review_sha256 |
| `final_report` | 봉인(맨 마지막) | test/ci/dev/costs/multiple_testing_disclosure/claims_sha256/… (report.json과 동일) |

### 7.3 세 가지 질문에 답 찾기

**Q1. "후보가 이겼나?"**
- `status`의 `last N gate decisions` 줄, 또는 `run` 콘솔의 `[gate] ... -> winner rNNNN`.
- 확실히 하려면 해당 세대의 `gate` 레코드: `winner` 필드 + `reason`
  (`beat the incumbent on the blind gate split`). admission 점수는 여기 `results`에만.
- 또는 `state.json`의 `best_commit`/`best_primary`가 움직였는지.

**Q2. "결론이 뭔가?"**
- 사람용: `experiments/report/report.md`의 `## Headline result` → `Status:` +
  effect + 95% CI.
- 기계용: `experiments/report/report.json`의 `primary_status` + `test.effect_abs` +
  `ci.abs`. `verified`(CI 하한>0)만 진짜 개선. `unsupported`는 "실패"가 아니라 "test
  시드 중 하나가 unclean이라 bootstrap 생략"이란 뜻.

**Q3. "무슨 근거로?"**
- 채택 experiment 레코드의 `hypothesis.supporting_evidence_ids` → 그 id들을
  `experiments/evidence/evidence.jsonl`에서 찾으면 `{canonical_paper_id, claim,
  stance, locator, ...}`가 나온다. (실측: r0014는 `ev_cl-0901`을 인용 →
  paper `tsp:2018.0009` §3.1 "multi-start restarts lower expected best route length".)
- report 시점에 **증거 감사**가 돌아, 채택 가설이 해석 불가한 증거를 인용하면
  `evidence audit failed: ...`로 보고가 하드 실패한다.

### 7.4 가설 인증서(hypothesis) 뜯어보기

실측 r0014(최종 승자)의 인증서:

```json
{
  "id": "h_r0014_restarts",
  "statement": "Changing restarts from 4 to 16 will minimize mean_tour_length by at least 0.2% relative to the incumbent.",
  "mechanism": "Independent multi-start restarts reduce the variance of the best tour found and lower the expected result, at a cost linear in the restart count.",
  "intervention": {"param": "restarts", "from": 4, "to": 16, "kind": "restarts_up"},
  "predicted_effect": "mean_tour_length improves from 6455142.9000 by >= 0.2% ...",
  "falsifier": "mean_tour_length fails to improve by >= 0.2% (or becomes degenerate) in the deterministic dev evaluation — the single decisive test",
  "minimal_test": "one smoke + one dev evaluation on the patched worktree",
  "supporting_evidence_ids": ["ev_cl-0901"],
  "nearest_prior_work": ["tsp:2015.0006", "tsp:2018.0009"],
  "proposer": "heuristic", "executor": "patcher"
}
```

이게 이 프레임워크의 "과학성"의 정수다: 모든 가설이 **statement / mechanism /
intervention / predicted_effect / falsifier(기각 조건) / minimal_test**를 갖는다.
Popper식 반증가능성을 강제한다.

---

## 8. 심화 활용 레시피

### 8.1 LLM proposer + 코딩 워커 (`--proposer claude`)

```bash
uv run python orchestrator.py run --generations 3 --proposer claude --max-budget-usd 0.5
```

무엇이 달라지나:
- 가설 생성을 Claude Agent SDK가 맡는다(tools 전면 비활성, JSON 스키마 강제, 검증
  실패시 휴리스틱 폴백). SDK/검증 실패시 `[warn] claude proposer failed (...);
  falling back to heuristic`.
- **코더가 켜진다**(`portfolio.max_coder_hypotheses`, 기본 1). 코더는 하이퍼파라미터로
  못 넘는 벽 — 예를 들어 `NEIGHBORHOOD`를 `two_opt→or_opt`로 바꾸거나 tabu를 추가 —
  을 위해 `src/**`를 편집한다.

**코더 격리(다층):**
- **PreToolUse 가드 훅**이 유일한 허용자. cwd로는 SDK 툴을 못 가두므로(절대경로
  허용) 훅이 모든 툴 호출을 realpath로 해석해 worktree 밖 Read/Glob/Grep과 `src/`
  밖 Write/Edit를 거부한다. **Bash·네트워크 툴은 아예 비활성.**
- `permission_mode="dontAsk"` + `allowed_tools=[]`로 **fail-closed**(훅이 에러·
  타임아웃이면 기본 거부).
- 코더 호출 전후로 **루트 지문(git status + 보호 매니페스트) 스냅샷 비교** → 절대
  경로로 루트를 건드리는 탈출을 탐지하면 `root working tree mutated during coder
  round ...`로 캠페인 중단.
- 이건 §8.5의 **실행 격리(container)와 별개 계층**이다. 코더 격리는 "편집 시점"을,
  container는 "실행 시점"을 막는다.

> 주의: LLM 경로는 계정 사용량 한도에 걸릴 수 있다. 걸리면 크래시 복구가 중단된
> 세대를 정리하므로 안전하고, 한도 리셋 후 재개하면 된다.

### 8.2 LLM 문헌 분석 (`--literature claude`)

```bash
uv run python orchestrator.py run --generations 3 --proposer claude --literature claude
```

LLM이 (1) 추가 검색 쿼리 분해, (2) 증거별 stance 판정 + 서술을 한다. **검색 실행은
항상 결정적 lexical 백엔드.** 반-론더링: LLM은 결정적 "supports"를 강등만 할 수
있고 새로 부여 못 한다(부여 시도는 `coverage.llm_supports_coerced`로 카운트). 실패
시 `[literature] LLM analyst failed (...); falling back to lexical grounding`.

### 8.3 pairwise gate (SciNav 스타일 blind 심판)

```bash
uv run python orchestrator.py run --generations 3 --proposer claude --gate pairwise
```

**admission은 어느 모드에서도 항상 결정적 스칼라 epsilon 규칙**(anti-overfitting
보증). admission을 통과한 후보가 2명 이상일 때만 익명화된 blind 심판단(N=3
다수결)이 승자를 고른다. 심판은 계약·가설 인증서(id만)·bounded 코드 diff·**dev**
메트릭만 보고 **gate 점수는 입력조차 못 받는다.** A/B 라벨은 sha256 패리티로
뒤섞어 position bias 상쇄, 판정은 enum 4지선다, 기권·과반미달·SDK실패·예산소진은
전부 결정적 스칼라로 폴백하며 `scalar_winner`를 항상 병기(divergence 감사).

> 콘솔: `[gate] candidates [...] -> winner rNNNN [pairwise: agreed with scalar]`
> (또는 `overrode scalar` / `scalar fallback`).

### 8.4 cross-model codex 적대 리뷰어 (ARIS)

진짜 이질 모델(OpenAI 계열)이 각 claim의 숫자를 raw test 데이터에 대조 감사한다.
pairwise 심판이 같은 Claude 계열이라 남는 상관 편향을 여기서 닫는다.

```bash
# 1) 계약에서 reviewer.enabled: true 로 (chmod 후, §9.3), 그리고 codex 로그인 준비
# 2) 옵트인 실행 (봉인된 보고를 다시 내는 것이므로 --force + 재승인 필요)
uv run python orchestrator.py report --force --reviewer codex
uv run python orchestrator.py approve <새 request_id>
uv run python orchestrator.py report --force --reviewer codex
```

- `codex exec`를 **read-only 샌드박스**에서 호출. 프롬프트에 심은 `echo_token`으로
  요청·응답을 묶어 재검증(불일치시 `echo_mismatch` 경고).
- **advisory**: 실패는 전부 `status="unavailable"`로 기록(코드: `codex_not_found`,
  `codex_not_authenticated`, `timeout`, `schema_violation` 등)될 뿐 보고를 막지 않고,
  **Claude 리뷰어로 조용히 폴백하지 않는다**(이질성 목적 보존).
- 결과는 `experiments/report/review/`와 `review` 원장 레코드에만. report.md엔 안
  들어간다(숫자-via-claim 스캔 일관성).

### 8.5 container 샌드박스로 trust-grade 만들기 (step by step)

기본 subprocess는 gate/test에서 신뢰 등급이 아니다. 진짜 격리를 원하면:

```bash
# 1) Docker 데몬 실행 (colima start 또는 Docker Desktop)
colima start

# 2) 이미지를 digest로 미리 pull — 런은 --network none이라 on-demand pull 불가!
docker pull python:3.14-slim@sha256:<digest>

# 3) 보호된 계약 잠금 해제 후 sandbox 블록 편집
chmod u+w research_contract.yaml
```

`research_contract.yaml`의 sandbox 블록:
```yaml
sandbox:
  backend: container
  image: "python:3.14-slim@sha256:<digest>"   # 반드시 digest 핀(재현성)
  memory_mb: 512
  cpus: 1.0
  pids_limit: 128
  require_container_for_trusted_splits: true   # subprocess로 gate/report 하드 차단
```

```bash
# 4) 매니페스트 재해시 + 재베이스라인 (baseline도 컨테이너에서 학습)
uv run python orchestrator.py init --force

# 5) 실행 — 각 후보 solver가 docker run 안에서 돈다
uv run python orchestrator.py run --generations 1
```

실제 docker argv(핵심 플래그):
```
docker run --rm --init --network none --read-only
  --tmpfs /tmp:rw,size=64m,noexec,nosuid,nodev
  --user 65534:65534 --cap-drop ALL --security-opt no-new-privileges
  --memory 512m --memory-swap 512m --cpus 1.0 --pids-limit 128
  -v <workspace>:/w:ro
  -v <fresh_artifacts>:/w/artifacts:rw               # 유일한 쓰기 표면
  -v <empty_mask>:/w/evaluation/heldout_config.json:ro   # 숨은 시드 마스킹
  --tmpfs /w/experiments:rw,size=1m,...              # gate 점수 원장 마스킹
  -v <instances>:/w/instances.json:ro               # 좌표만
  -e PYTHONHASHSEED=0 -e AUTORESEARCH_INSTANCES=/w/instances.json
  python:3.14-slim@sha256:<digest> python -s -B src/train.py
```

**fail-closed 규율:** 데몬/이미지가 없으면 **절대 subprocess로 조용히 폴백하지
않고** actionable 에러로 중단한다. 예:
- `Docker daemon not reachable — start it (colima start / Docker Desktop), or set sandbox.backend: subprocess in the contract`
- `pinned sandbox image absent — run `docker pull ...` first (runs are --network none, so the image cannot be pulled on demand)`

**provenance echo 검증:** 평가기가 `metrics["sandbox"].backend`를 요청 backend와
에코 대조. 불일치면 `ProtectionViolation: sandbox backend echo mismatch ...`
(구버전·우회 탐지). 성공하면 report.md 헤더가 `trust-grade`로 찍힌다.

> Linux면 `sandbox/runner.py`에 `--runtime=runsc` 한 줄 추가로 gVisor 격리로
> 승격(마운트/보안 모델 불변). backend는 캠페인 단위로 고정되므로 중간에 못 바꾼다.

### 8.6 실 문헌 corpus 새로고침 (`ground --refresh`)

현재 corpus는 큐레이션된 오프라인 mock이다. 실 문헌(OpenAlex/S2)으로 갱신하려면
**네트워크 호스트에서** 이 유지보수 작업을 돌린다. (캠페인은 얼린 스냅샷 위에서
결정적으로 도므로 refresh 때만 네트워크를 쓴다.)

```bash
export S2_API_KEY=...            # s2 fetch 때만. env 전용, 계약/커밋 절대 금지
chmod u+w literature/corpus/tsp_corpus.json
uv run python orchestrator.py ground --refresh \
  --source openalex --extractor claude --max-papers 60 --mailto you@example.com

git diff literature/corpus/tsp_corpus.json   # 사람이 태그 diff 리뷰
uv run python orchestrator.py init --force   # 재해시 + 재베이스라인
```

- 흐름: fetch → dedup(DOI>arXiv>title) → LLM(또는 결정적) claim 추출 →
  `.refresh.tmp`에 쓰고 재검증 → 유효하면 os.replace(무효면 옛 스냅샷 유지).
- **반-론더링 가드:** 주입된 abstract가 `effect=improves`(유일한 support 부여
  스탠스)를 못 만들게, 개선 단서가 없거나 인젝션 마커가 있으면 `conditional`로 강등.
  `DeterministicExtractor`는 절대 `improves`를 안 낸다.
- refresh 후 반드시 출력되는 **REVIEW BEFORE FREEZE** 블록이 support 부여 claim,
  injection-flagged 논문, 정책 탈락 claim 수를 보여준다 — **이 사람 리뷰가 동결
  게이트다.**
- 정직한 한계: `claude` extractor는 비결정론이라 실 refresh는 매번 새 diff.
  `_urllib_get`·실 SDK 호출은 실 네트워크에서만 최종 검증됨(오프라인 테스트 밖).

---

## 9. 내 연구 문제로 도메인 바꾸기

이 프레임워크의 진짜 가치는 "TSP를 푸는 것"이 아니라 **어떤 연구 문제든 이
안전장치 위에 얹을 수 있는 스캐폴드**라는 데 있다. 도메인 교체는 아래 4개를 함께
재설계하는 일이다.

### 9.1 무엇을 바꾸나 (4개 표면)

| 파일 | 역할 | 바꿀 때 |
|---|---|---|
| `src/train.py` | 편집 대상 solver/모델 | 새 알고리즘 + `HYPERPARAMS` 마커 블록 + (선택) 코드 손잡이 |
| `evaluation/dataset.py` | 인스턴스/데이터 생성 | 새 데이터 분포 + `SPLIT_SIZES` + `load_train`/`load_split`/`fingerprint` |
| `evaluation/evaluate.py` | **신뢰 평가기** | 새 지표 재계산 + 하드코딩 상수(budget/metric/split/N) + failure_class |
| `research_contract.yaml` | 계약 | `objective`, `primary_metric`, budgets, portfolio 등 |
| `literature/corpus/*.json` | (선택) 문헌 | 새 도메인 claim corpus (또는 `ground --refresh`) |

### 9.2 지켜야 할 설계 규칙 (안 지키면 보증이 무너진다)

1. **평가기는 자기보고를 믿지 마라.** solver 산출물을 검증하고 지표를 **직접
   재계산**하라. (TSP에서 tour 길이를 재계산하듯.) 이게 reward hacking 방지의 핵심.
2. **평가기 상수는 하드코딩하고 계약과 교차검증하라.** 평가기는 `evaluation/` 밖을
   신뢰하지 않는다. `init`이 계약과 평가기 상수(budget/metric/split/N)를 1회
   교차검증해 drift면 fail-fast한다.
3. **숨은 시드는 평가기에만.** dev/gate/test 시드는 `heldout_config.json`(git
   미추적)에 두고 인스턴스에는 **불투명 id(i0, i1…)만** 실어 solver로 넘겨라. 시드
   정수가 샌드박스로 넘어가는 구조를 만들지 마라(다른 스플릿 재생성 차단).
4. **목표 함수에 진짜 원하는 걸 넣어라.** §6의 교훈: `mean_tour_length`만
   최소화하면 "연산을 더 써서 이기는" 승리도 정당해진다. 연산 예산이 중요하면
   목표에 `solve_seconds`류를 반영하라. (평가기가 뭘 재느냐가 곧 "과학의 방향"이다.)
5. **`min_relative_improvement` / `gate_min_relative_improvement`는 노이즈보다 크게.**
   평가가 결정적이면 `>` 규칙이 1e-9 흔들림도 채택하므로 상대 epsilon을 쓴다.

### 9.3 보호 파일을 의도적으로 수정하는 절차

실행 중이 아닐 때:
```bash
chmod u+w <파일>                                 # 읽기전용 해제
# ... 편집 ...
uv run python orchestrator.py init --force       # 시드·매니페스트 재생성 + 재베이스라인
```

`--force`는 `experiments/`를 비우므로 이전 캠페인 기록이 필요하면 먼저 백업. 계약
schema나 heldout_config schema가 바뀌면 이전 상태에서 못 이어 돌리고 새 캠페인을
시작해야 한다.

> 팁: 큰 도메인 교체는 scratchpad에 rsync 복제 후 E2E로 드릴하고, 검증되면 실제
> 디렉토리에 반영하는 게 이 프로젝트의 관습이다.

---

## 10. 트러블슈팅 & FAQ

### 10.1 에러 메시지 → 대처

| 증상 / 메시지 | 원인 | 대처 |
|---|---|---|
| `exit 3` (report) | 사람 승인 필요(에러 아님) | `approve <request_id>` 후 재실행 |
| `approval pending for request_id ...` | 승인 대기 | 그 id를 `approve` |
| `report intent ... was denied` | `--deny`로 거부됨 | 같은 id를 `approve`로 뒤집기 |
| `N final report(s) already exist — test split is single-use` | 이미 봉인됨 | `report --force`(재승인 필요, 다중검정 공시 증가) |
| 승인했는데 다시 exit 3 | `run`을 더 돌려 **승인 stale** | 새 request_id를 재승인 |
| `main working tree has uncommitted tracked changes` | main 더러움 | 추적 변경 commit/restore (untracked는 무관) |
| `already initialized ...; use --force` | 이미 init됨 | 이어 돌리려면 그냥 `run`; 새 캠페인이면 `init --force`(기록 삭제됨) |
| `another orchestrator process is running (lock: ...)` | 동시 실행 | 다른 프로세스 종료; 죽은 프로세스면 `.orchestrator.lock` 확인 |
| `VIOLATION: <파일>` / `protected files modified: ...` | 보호 파일 변조 | 원복하거나 §9.3 절차로 정식 수정 |
| `... drift: contract ... vs evaluator ...` | 계약과 평가기 상수 불일치 | 계약↔평가기 값 맞추고 `init --force` |
| `evidence audit failed: <run> cites <id> ...` | 채택 가설이 해석 불가 증거 인용 | corpus/그라운딩 정합 확인 |
| `[warn] ... subprocess ... NOT trust-grade` | gate/test를 subprocess로 | 무시(현행) 또는 §8.5 container |
| `sandbox.backend 'container' ... image absent` | 이미지 미pull | `docker pull <digest>` 먼저 |
| `Docker daemon not reachable` | 데몬 꺼짐 | `colima start`/Docker Desktop, 또는 backend를 subprocess로 |
| `sandbox backend echo mismatch` | 평가기 우회/구버전 | 평가기 무결성 확인(`verify-protection`) |
| `literature ... is not writable (0o444)` | refresh인데 corpus 읽기전용 | `chmod u+w <corpus>` 후 재실행, 이후 `init --force` |

### 10.2 자주 묻는 것

**Q. `run --generations 3`인데 실험이 15개나 됐다?**
generation ≠ round. 한 세대에 최대 K=8 가설(휴리스틱은 파라미터당 1개라 ~6~7)이
돌아 experiments가 누적된다. 3세대 × ~5~7 = 15.

**Q. gate 점수를 콘솔/status에서 보고 싶다.**
못 본다(blindness). `experiments/generations/gNNNN/gate/*.json` 또는 `gate` 원장
레코드의 `results`를 직접 열어라. 트랜스크립트가 LLM 컨텍스트에 붙여넣어지는 걸
막기 위한 의도적 설계다.

**Q. `pruned`가 많이 뜨는데 실패인가?**
아니다. successive halving의 예산 컷이다(과학 아님). "이 방향이 틀렸다"가 아니라
"smoke rung에서 상위권이 아니라 dev rung 예산을 안 썼다"는 뜻.

**Q. `valid_positive`인데 왜 REJECT됐나?**
dev 개선은 맞지만 blind gate를 못 넘었거나 같은 세대의 다른 후보가 승자다. 세대당
accept는 1명뿐(§4.2).

**Q. `status`의 `primary_status: unsupported`?**
test 시드 중 하나가 unclean이라 bootstrap을 생략했다는 뜻 — "개선 실패"가 아니다.
`experiments/report/*_test_s*.json`의 failure_class를 확인.

**Q. LLM 경로 중 계정 한도에 걸렸다.**
크래시 복구가 중단 세대를 정리하므로 안전. 한도 리셋 후 재개. 문헌은 lexical로,
pairwise는 결정적 스칼라로 폴백해 세대를 막지 않는다.

**Q. 결과를 다른 사람에게 주장해도 되나?**
subprocess 백엔드 숫자는 "정직하지만 신뢰 등급 아님"이다. 주장하려면 §8.5
container로 재현해 report.md 헤더가 `trust-grade`로 찍히게 하라.

---

## 11. 건강 점검 (테스트 드릴)

프레임워크가 멀쩡한지 확인하려면 7개 드릴을 돌린다. **전부 완전 오프라인** —
Docker·네트워크·SDK·실 codex 불필요(전부 fake seam으로 대체). 각 파일은
`[ok  ] <name>` / `[FAIL] <name> — <detail>`를 줄마다 찍고, 전부 통과면 exit 0.

```bash
uv run python tests/test_phase2.py   # blindness / 코더 가드 fail-closed / gate 정확성 / 복구
uv run python tests/test_phase3.py   # 문헌 그라운딩 (결정성 / canary / 론더링 / 인젝션)
uv run python tests/test_phase4.py   # momentum fold / 조향 / halving / pairwise
uv run python tests/test_phase5.py   # bootstrap / claims / report digit-scan / 승인게이트 / codex stub
uv run python tests/test_phase6.py   # 샌드박스 docker argv·마스킹·fail-closed (Docker 불필요)
uv run python tests/test_phase6b.py  # 실 문헌 fetch·추출·스냅샷 (fake HTTP/LLM)
uv run python tests/test_phase6c.py  # TSP feasibility·재계산·seed 부재·blindness
```

CI 스모크 게이트로는 "7개 전부 exit 0"만 요구하면 된다. 주의: pytest가 아니라 각
파일을 직접 실행해야 하고, **저장소 루트에서** 돌려야 한다(실 계약·corpus를 읽으므로
CWD가 다르거나 corpus를 편집하면 카운트 드릴이 FAIL한다).

---

## 12. 부록 — 레퍼런스 사전

### 12.1 계약 필드 (현재 값)

`research_contract.yaml` (schema v8). 괄호는 현재 값.

```
primary_metric.name/direction/min_relative_improvement   (mean_tour_length / minimize / 0.002)
budgets.smoke_train_timeout_s / dev_train_timeout_s       (30 / 120)
budgets.max_rounds / repair_attempts                      (60 / 2)
portfolio.parallel_branches                               (8)   # 세대당 K
portfolio.gate_top_k                                      (2)   # gate로 보낼 dev 개선 후보 수
portfolio.gate_min_relative_improvement                  (0.001) # gate admission epsilon
portfolio.max_coder_hypotheses                           (1)   # 0이면 코더 끔
portfolio.max_generations                                (null)
portfolio.coder_max_turns / coder_max_budget_usd         (25 / 1.5)
portfolio.halving.enabled/keep_fraction/min_keep         (true / 0.5 / 2)
stop_conditions.stagnation_generations                   (4)
refinement.enabled/momentum_decay/exploit_fraction       (true / 0.5 / 0.75)
refinement.accelerate_after/evidence_steering            (2 / true)
pairwise_gate.enabled/judges/judge_model                 (true / 3 / claude-haiku-4-5)
pairwise_gate.judge_max_budget_usd                       (0.4)
literature.enabled/retriever/corpus_path                 (true / lexical / literature/corpus/tsp_corpus.json)
literature.max_evidence_per_generation/_per_hypothesis   (12 / 4)
literature.max_queries/stabilization_window/citation_hops (6 / 2 / 1)
literature.llm_max_budget_usd/_campaign_budget_usd       (0.5 / null)
literature.refresh.{sources,max_papers,extractor,...}    (openalex / 60 / claude / ...)
assurance.finalist_seeds/bootstrap_resamples/confidence  (5 / 10000 / 0.95)
reviewer.enabled/backend/timeout_s                       (false / codex / 300)
human_gate.enabled/require_approval_for                  (true / [first_report, force_report])
sandbox.backend/image/memory_mb/cpus/pids_limit          (subprocess / null / 512 / 1.0 / 128)
sandbox.require_container_for_trusted_splits             (false)
```

계약 검증 실패 예(로드시 하드 에러): `finalist_seeds`가 [1,16] 밖,
`bootstrap_resamples < 100`, `confidence_level`이 0.90/0.95/0.99가 아님,
`momentum_decay`가 (0,1) 밖, `judges`가 짝수/>5, `evidence_steering=true`인데
`literature.enabled=false` 등.

### 12.2 failure_class 사전

평가기가 붙이는 것: `invalid_workspace`, `evaluator_error`, `timeout`,
`nonzero_exit`, `missing_artifact`, `malformed_solution`, `infeasible_solution`(유효
tour 아님 — 과학적 negative), `no_skill`(identity-order tour보다 못함 — degenerate).
orchestrator가 붙이는 것: `metric_regression`(classify), `patch_failed`,
`oversized_diff`, `coder_unavailable`, `coder_error`, `nondeterministic`,
`smoke_rank_below_cutoff`(pruned), symlink/protected/editable 위반.

### 12.3 도메인 상수 (TSP)

`N_CITIES=60`, `GRID=1_000_000`, `TRAIN_SEED=20260401`(공개), `N_TRAIN_INSTANCES=40`,
`SPLIT_SIZES={dev:40, gate:40, test:160}`, `SOLVER_SEED=1337`, 거리=TSPLIB EUC_2D
정수 반올림. test는 `finalist_seeds`(5) × 160 = 800 인스턴스 풀로 paired bootstrap.

### 12.4 용어집 (엔지니어링 용어)

- **fail-closed**: 판단이 애매하거나 검사기가 에러/타임아웃이면 "거부"를 기본값으로
  삼는 것. (반대는 fail-open = 애매하면 통과.)
- **write-ahead**: 실제 작업 전에 "이걸 하겠다"를 먼저 원장에 적는 것. 도중에 죽어도
  복구가 무엇이 진행 중이었는지 안다. (DB의 WAL과 동일 개념.)
- **ff-merge (fast-forward merge)**: 갈래가 안 생기게 브랜치 포인터만 앞으로 옮기는
  머지. `main`에는 gate 통과 실험만 이렇게 쌓인다.
- **flock (파일 잠금)**: 파일에 잠금을 걸어 같은 작업이 동시에 두 번 돌지 못하게 함.
- **worktree**: 같은 git 저장소의 다른 커밋을 별도 폴더에 동시 체크아웃하는 기능.
  각 실험이 자기 worktree에서 격리 실행된다.
- **tmpfs**: 메모리에 올라가는 임시 파일시스템. 컨테이너에서 원장을 마스킹할 때
  "빈 tmpfs를 덮어씌워" 원본을 안 보이게 한다.
- **provenance echo**: 요청한 값(backend/nonce/split)을 결과에 그대로 되적게 해서
  위조·우회·구버전을 탐지하는 기법.
- **blindness (여기서)**: gate 점수가 탐색 신호·proposer·insight·보고서로 새지
  않도록 구조적으로 차단하는 불변식.
- **incumbent**: 현재 챔피언(best). 후보가 이걸 이겨야 새 incumbent가 된다.

---

### 다음에 뭘 할까 (추천)

1. **한 판 더, 조금 다르게:** `run --generations 4 --proposer claude`로 LLM 코더가
   `NEIGHBORHOOD`를 바꾸는 "알고리즘 수준" 개선을 시도하는지 보라(하이퍼파라미터
   벽을 코드로 넘는 실증).
2. **신뢰 등급 재현:** §8.5로 container 백엔드를 켜고 같은 캠페인을 재현해 report.md
   헤더가 `trust-grade`로 바뀌는지 확인.
3. **내 문제로 이식:** §9로 작은 도메인(예: 다른 조합최적화나 간단한 회귀)을 얹어
   보라. 목표 함수에 무엇을 넣느냐가 결과를 어떻게 바꾸는지(§6 교훈)를 직접 체감하는
   게 이 프레임워크를 이해하는 가장 빠른 길이다.

이 문서에서 못 찾은 세부는 `README.md`(사용 요약) → `docs/HANDOFF.md`(불변식 14개) →
`docs/BLUEPRINT.md`(설계 근거) → 코드 순으로 파고들면 된다.
