# AutoResearch — Phase 1: Constrained Executable Autoresearch Loop

2026 SOTA 블루프린트(Arbor / Gome / ERA / SciNav 종합)의 Phase 1 구현.
**Karpathy 스타일 keep/reject 루프 + 기본 Arbor 스타일 상태 관리**로, 에이전트
스웜보다 평가기(evaluator)·연구 계약(contract)·프로버넌스를 먼저 세우는 것이
목표다.

핵심 원칙: *신뢰할 수 있는 평가기 없이는 에이전트를 늘려도 더 과학적이 되지
않는다.* 이 저장소의 모든 구조는 "그럴듯해 보이는 실패"와 "검증된 진전"을
구분하는 데 맞춰져 있다.

## 빠른 시작

```bash
uv sync                                        # 의존성 설치 (pyyaml, claude-agent-sdk)
uv run python orchestrator.py init             # git 초기화 + 베이스라인 평가 + 보호 장치
uv run python orchestrator.py run --rounds 8   # keep/reject 루프 실행
uv run python orchestrator.py status           # 캠페인 상태 확인
uv run python orchestrator.py verify-protection
```

LLM 가설 생성기(Claude Agent SDK — 로컬 Claude Code 로그인 재사용, 별도 API 키
불필요):

```bash
uv run python orchestrator.py run --rounds 4 --proposer claude
```

## 루프가 하는 일

라운드마다:

1. 보호 manifest 검증 (protected 파일 SHA-256 대조)
2. **가설 인증서** 1개 생성 — statement / mechanism / intervention /
   predicted_effect / **falsifier**(기각 조건) / minimal_test.
   기본은 결정적 휴리스틱, `--proposer claude`면 Claude Agent SDK가 생성
   (툴 전면 비활성, JSON 스키마 강제 출력, 검증 실패 시 휴리스틱 폴백)
3. incumbent에서 git worktree 격리 생성 (`hyp/<campaign>/rNNNN-<param>` 브랜치 —
   캠페인 네임스페이스라 `init --force` 후에도 이전 캠페인 브랜치와 충돌 없음)
4. `src/train.py`의 HYPERPARAMS 마커 블록에서 **정확히 1개 파라미터** 치환
   (ast 파싱 + 라운드트립 검증; 기계적 실패만 제한 횟수 내 수리)
5. 평가 **전** 커밋 (커밋 = 순수 코드 변경)
6. protected/editable glob 검사 (평가 **전과 후** 모두)
7. **루트** 평가기 실행: smoke(2 epochs) → dev. worktree 안의 evaluation/
   사본은 절대 실행되지 않는다
8. 분류 → ledger에 판정 **선기록**(write-ahead) → 개선이면 `--ff-only` 머지,
   아니면 reject (브랜치는 프로버넌스로 보존)
9. 교훈을 `insight_memory.json`으로 증류, 상태 갱신, 정지 조건 확인

판정 분류:

| verdict | 의미 | 처리 |
|---|---|---|
| `valid_positive` | 상대 개선 ≥ `min_relative_improvement` | ff-merge (KEEP) |
| `valid_inconclusive` | 변화가 임계 미만 | reject, stagnation 증가 |
| `valid_negative` | 지표 악화 / NaN 발산 / no_skill / crash / timeout | reject — **수리 금지, 과학적 증거로 증류** |
| `invalid_implementation` | 패치 시점 기계적 실패만 | 라운드 무효 |
| `contract_violation` | protected 경로 접촉 | reject + 증거 보존 |

발산·crash·timeout이 `valid_negative`인 근거(귀속 규칙): 베이스라인이 init에서
실행 가능함이 증명됐고 패처가 결정적이므로, 유효한 패치 후의 런타임 실패는
전부 개입 탓이다. 평가 후 "수리"는 의도적으로 구현하지 않았다 — 유효한 부정적
결과를 긍정으로 바뀔 때까지 고치는 false repair를 차단하기 위해서다.

## Mock ML 태스크

순수 파이썬 합성 회귀 (외부 의존성 없음, 완전 결정적):

- 8개 피처, 이질적 스케일 `[1, 1, 5.0, 0.2, 1, 3.0, 1, 0.5]` (조건수 ~625) —
  `feature_scaling`이 실제 효과를 갖게 만든다
- `y = 0.3 + w·x + 0.3·x0·x1 + N(0, 0.25)` — 상호작용 항은 선형 모델이 못
  잡으므로 환원 불가능한 floor(~0.39)가 존재
- 상수 예측기 RMSE ~1.68, 베이스라인 ~0.51-0.54(held-out 시드에 따라 다름),
  unscaled lr ≥ 0.08에서 NaN 발산 → accept/reject/발산이 자연스럽게 섞인 탐색 표면

## 보호 모델 (Phase 1에서 닫은 것)

- **held-out 시드의 물리적 부재**: `evaluation/heldout_config.json`은 init 때
  생성되고 git이 추적하지 않는다. worktree는 추적 파일만 체크아웃하므로 후보
  워크스페이스에는 시드가 아예 존재하지 않는다
- **루트 평가기만 authoritative** + 평가기는 evaluation/ 밖을 신뢰하지 않음
  (예산 하드코딩 — init에서 contract와 1회 교차검증, dataset을 절대경로 import)
- **nonce 에코**: 오케스트레이터가 라운드마다 새 nonce를 전달, 평가기가
  metrics에 에코 (학습 서브프로세스에는 절대 전달 안 됨) → 위조 metrics 차단
- **격리된 학습 서브프로세스**: 처음부터 구성한 env(PATH, PYTHONHASHSEED=0),
  `-s -B` 플래그, 자체 세션 + 타임아웃 시 프로세스 그룹 SIGKILL
- **아티팩트 검증**: symlink 거부, 1MB 상한, 유한성 검사, train_rmse 재계산
- **SHA-256 manifest** 매 라운드 검증 + protected 파일 chmod 444 +
  worktree diff/status를 glob으로 평가 전·후 검사 + 평가기 자기 해시를
  metrics에 임베드해 manifest와 교차 대조
- **크래시 복구**: write-ahead ledger 기반 — 중단된 라운드는 aborted 처리
  (라운드 번호 소각), accepted인데 머지 안 된 경우 재실행. tested/stagnation/
  last_accepted는 ledger에서 통째로 재구성되므로 어느 시점에 죽어도 상태가
  ledger와 어긋나지 않는다. 머지가 불가능해진 accept는 기록 전에 강등되고,
  기록 후 머지 실패는 correction 레코드로 상쇄된다
- **단일 인스턴스 잠금**(flock): 동시 `run`이 진행 중인 worktree를 파괴하는
  것을 차단. rename을 통한 protected 파일 이동도 diff `--no-renames`로 탐지

### 정직한 한계 (Phase 2 과제)

- held-out 시드는 로컬 사용자가 루트에서 읽을 수 있다 (정책적 보호 수준).
  실제 격리는 blind admission gate + 컨테이너 샌드박스가 필요
- 학습 코드가 백그라운드 데몬을 남겨 평가 후 파일을 조작하는 TOCTOU류 공격,
  워크스페이스 밖 파일쓰기, 네트워크 접근은 OS 샌드박스 없이는 못 막는다
  (탐지는 일부 가능, 방지는 불가)
- 순차 단일 브랜치 (병렬 가설 포트폴리오는 Phase 2)

## 파일 구조

```
research_contract.yaml    # Layer 1 타입드 계약 — 불변, 절대 프로그램이 안 씀
orchestrator.py           # keep/reject 루프 (protected)
src/train.py              # 유일한 편집 가능 표면
evaluation/evaluate.py    # 보호된 평가기 → metrics.json
evaluation/dataset.py     # 합성 데이터 (train/held-out 시드 분리)
evaluation/heldout_config.json  # init 생성, untracked (worktree에 부재)
protection/hashes.json    # SHA-256 manifest (git 추적)
experiments/              # 런타임: state.json, ledger.jsonl, rounds/ (gitignored)
insight_memory.json       # ledger에서 재구성 가능한 파생 데이터 (gitignored)
.worktrees/               # 라운드별 격리 (gitignored)
```

프로버넌스 규약: `main`에는 accepted 실험만 ff-merge로 쌓인다. 실패한 실험도
`hyp/*` 브랜치 + `experiments/ledger.jsonl`에 전부 남는다 (유효한 부정적
결과는 증거다).

## protected 파일을 의도적으로 수정하려면

실행 중이 아닐 때: `chmod u+w <파일>` → 수정 → `uv run python orchestrator.py
init --force` (held-out 시드·manifest 재생성 + 재베이스라인). `--force`는
experiments/를 비우므로 이전 캠페인 기록이 필요하면 먼저 백업할 것.

## Phase 2 로드맵 (블루프린트 기준)

1. 가설 포트폴리오: 4–8 병렬 브랜치 + 코디네이터 (Claude Agent SDK 기반 —
   이 프로젝트는 LangGraph 대신 Claude Agent SDK를 오케스트레이션 런타임으로 쓴다)
2. LLM 코딩 워커 (RepairStrategy seam에 접속) + blind admission gate
3. 문헌 근거 엔진 (claim-level evidence, PaperQA2류)
4. Gome 스타일 branch-local directed update + SciNav 스타일 pairwise 평가
5. claim-evidence ledger 기반 보고서 생성 + cross-model 적대적 리뷰
