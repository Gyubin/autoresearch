# AutoResearch — Executable Autoresearch Loop (Phase 1 + 2 + 3)

2026 SOTA 블루프린트(Arbor / Gome / ERA / SciNav 종합)의 구현. **Karpathy 스타일
keep/reject 루프 + Arbor 스타일 상태 관리**에서 시작해, **병렬 가설 포트폴리오 +
blind admission gate + LLM 코딩 워커**(Phase 2), **claim 수준 문헌 그라운딩
(evidence graph)**(Phase 3)까지 확장했다. 에이전트 스웜보다
평가기(evaluator)·연구 계약(contract)·프로버넌스를 먼저 세우는 것이 원칙.

핵심 원칙: *신뢰할 수 있는 평가기 없이는 에이전트를 늘려도 더 과학적이 되지
않는다.* 이 저장소의 모든 구조는 "그럴듯해 보이는 실패"와 "검증된 진전"을
구분하는 데 맞춰져 있다.

## 빠른 시작

```bash
uv sync                                            # 의존성 (pyyaml, claude-agent-sdk)
uv run python orchestrator.py init                 # git 초기화 + 베이스라인 + 보호 장치
uv run python orchestrator.py ground               # 연구질문 인증서 (문헌 evidence flow)
uv run python orchestrator.py run --generations 4  # 병렬 포트폴리오 실행 (휴리스틱 + lexical 문헌)
uv run python orchestrator.py status               # 캠페인 상태 (문헌 통계 포함)
uv run python orchestrator.py report               # test 스플릿 최종 보고 + 증거 감사 (1회성)
uv run python orchestrator.py verify-protection
uv run python tests/test_phase2.py                 # Phase 2 드릴
uv run python tests/test_phase3.py                 # Phase 3 드릴 (문헌 그라운딩)
```

LLM 가설 생성기 + 코딩 워커 + 문헌 분석기(Claude Agent SDK — 로컬 Claude Code
로그인 재사용, 별도 API 키 불필요). 코더 가설은 `--proposer claude`일 때만,
LLM 문헌 경로는 `--literature claude`일 때만 켜진다 (기본은 완전 오프라인):

```bash
uv run python orchestrator.py run --generations 4 --proposer claude --literature claude
```

## Phase 2 — 병렬 포트폴리오 · Blind Gate · LLM 코더

한 **generation**마다 proposer가 서로 다른 병목을 노리는 **K개(기본 4)의 다양한
가설**을 한 번에 내놓고, 각각 격리된 worktree에서 **병렬 실행**된다
(concurrent.futures 스레드 + 서브프로세스 평가기; git의 공유 상태를 건드리는
worktree add/remove·branch 삭제·merge만 잠금으로 직렬화, 워커는 자기 worktree
안 작업만).

**두 단계 평가 (development / blind admission):**

- **dev 스플릿**은 탐색에 쓰인다 — 모든 K 후보를 generation 시작 시점의 incumbent
  기준으로 채점. 개선 후보(valid_positive) 상위 `gate_top_k`개만 다음 단계로.
- **gate 스플릿**은 숨겨진 별도 홀드아웃이다. 후보가 incumbent의 gate 점수를
  `gate_min_relative_improvement`만큼 이겨야 승자가 되고, 승자 1명만 main에
  ff-merge된다. 실제로 dev에서 미세하게 좋아졌지만 gate에서 일반화되지 않은
  후보를 걸러낸다 (development-set overfitting 방지).
- **blind 규약**: gate 점수는 `record_type=gate` 원장 레코드와 gate 메트릭 파일
  에만 존재한다. 절대 insight, `best_primary`(항상 dev 점수), proposer 컨텍스트,
  실험 레코드에 흘러들지 않는다. incumbent gate 점수는 커밋별로 캐시되며, 평가가
  결정적이라 이 캐시는 근사가 아니라 정확값이다.
- **test 스플릿**은 캠페인 끝의 `report`에서 **딱 한 번** 쓰인다. 재실행하려면
  `--force`가 필요하고 그 횟수가 다중검정 공시에 기록된다.

**LLM 코딩 워커 (executor="coder"):** 하이퍼파라미터로 못 넘는 벽 — 예컨대 선형
모델의 환원 불가 floor — 을 넘으려면 코드를 고쳐야 한다. proposer가 코더 가설을
낼 수 있고(`portfolio.max_coder_hypotheses`, 기본 1), ClaudeCoder가 worktree
안에서 `src/**`를 편집한다. 격리는 다층으로:

- **PreToolUse 가드 훅**이 유일한 허용자다. cwd는 SDK 툴을 가두지 못하므로
  (절대경로 허용), 훅이 모든 툴 호출을 realpath로 해석해 worktree 밖 Read/Glob/
  Grep과 `src/` 밖 Write/Edit를 거부한다. Bash·네트워크 툴은 아예 비활성.
- `permission_mode="dontAsk"` + `allowed_tools=[]`로 **fail-closed**: 훅이
  에러/타임아웃이면 기본 거부.
- 코더 호출 전후로 **루트 지문(git status + 보호 manifest)을 스냅샷 비교**해,
  절대경로로 루트를 건드리는 탈출을 탐지하면 캠페인을 중단.
- `src/` 밑 symlink, 과대 diff, 결정성 재검(gate 진입 후보는 dev를 2번 돌려
  bitwise 비교)까지 확인.

모델 클래스 확장은 **평가기 안에서 후보 코드를 실행하지 않고** 데이터로 처리한다:
아티팩트의 `feature_spec`(원본 피처 인덱스 곱들의 목록, 최대 32항·차수 3)을
평가기가 검증만 하고 스코어링에 적용한다.

## 루프가 하는 일

각 generation마다:

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
| `valid_positive` | dev 상대 개선 ≥ `min_relative_improvement` | gate 후보 (승자만 KEEP) |
| `valid_inconclusive` | 변화가 임계 미만 | reject, generation stagnation 증가 |
| `valid_negative` | 지표 악화 / NaN 발산 / no_skill / crash / timeout | reject — **수리 금지, 과학적 증거로 증류** |
| `invalid_implementation` | 기계적 실패 (패치 실패 / 코더 mechanical smoke 실패 / 비결정성) | 라운드 무효 |
| `contract_violation` | protected 경로 접촉 / src symlink / 과대 diff | reject + 증거 보존 |

`valid_positive`는 dev에서 개선됐다는 뜻일 뿐, 채택은 blind gate 통과가 조건이다
(한 generation에서 gate 승자 1명만 accept, 나머지 dev-improver는
`valid_positive`이되 decision=REJECT).

발산·crash·timeout이 `valid_negative`인 근거(귀속 규칙): 베이스라인이 init에서
실행 가능함이 증명됐으므로 유효한 개입 후의 런타임 실패는 전부 개입 탓이다. 평가
후 "수리"는 의도적으로 없다 — false repair 차단. 코더 가설의 수리는 **smoke 단계
기계적 실패**(nonzero_exit / missing_artifact / malformed_artifact)에만 허용되고,
아직 채점 가능한 모델을 못 만든 경우에 한한다. timeout·발산·no_skill·dev 단계
실패는 전부 증거로 남긴다.

## Mock ML 태스크

순수 파이썬 합성 회귀 (외부 의존성 없음, 완전 결정적):

- 8개 피처, 이질적 스케일 `[1, 1, 5.0, 0.2, 1, 3.0, 1, 0.5]` (조건수 ~625) —
  `feature_scaling`이 실제 효과를 갖게 만든다
- `y = 0.3 + w·x + 0.3·x0·x1 + N(0, 0.25)` — 상호작용 항은 **선형** 모델이 못
  잡으므로 하이퍼파라미터로는 못 넘는 환원 불가 floor(~0.39)가 존재
- 상수 예측기 RMSE ~1.68, 베이스라인 ~0.51-0.54, unscaled lr ≥ 0.08에서 NaN 발산
- **코더의 표적**: `src/train.py`의 `FEATURE_SPEC`을 확장해 x0·x1 같은 곱 항을
  더하면 floor를 뚫는다. 실측 데모에서 LLM 코더가 이 항을 추가해 dev RMSE
  ~0.25(=floor 아래)에 도달, 하이퍼파라미터만으로는 불가능한 개선을 blind gate
  통과로 채택했다 — Phase 2의 핵심 실증

## Phase 2에서 추가로 닫은 것

- **blind admission gate**: dev에서 좋아 보여도 숨겨진 gate 스플릿에서 일반화되지
  않으면 채택 안 됨 (development-set overfitting 방지). 3-way 스플릿(dev/gate/
  test)의 시드는 전부 서로 다르고 worktree에 부재.
- **gate 점수 blindness**: gate 값은 proposer 컨텍스트·insight·`best_primary`·
  실험 레코드 어디에도 안 들어감. `distill_insight`는 gate 레코드를 아예 안 읽고,
  자동 테스트가 이 불변식을 검사.
- **LLM 코더 격리**: PreToolUse 가드 훅(fail-closed) + 루트 지문 스냅샷 비교 +
  src symlink/과대 diff/결정성 재검 + smoke 전용 기계적 수리.
- **test 스플릿 1회성**: `report`는 재실행 시 다중검정 공시 카운터를 올림.
- **병렬 안전성**: git 공유 상태 변경만 잠금 직렬화, 워커 스레드는 상태/원장에
  안 씀(전부 배리어 이후 메인 스레드), gc.auto 비활성.

## Phase 3 — 문헌 그라운딩 (claim 수준 evidence graph)

가설이 문헌 근거 위에 서고, novelty·모순이 보고되고, 모든 인용이 감사 가능해진다.
`literature/`는 **별도 통제 서비스**다: orchestrator를 import하지 않고, 코퍼스
외에는 어떤 파일도 읽지 않으며(state·ledger·metrics 접근 불가 — 시그니처 수준
폐포), 런타임에 아무것도 쓰지 않는다.

- **오프라인 mock corpus** (`literature/corpus/mock_corpus.json`): 가공 논문
  13편 / claim 15개. 각 claim은 locator(섹션·표·쪽)·population·conditions·
  limitations·구조화 태그를 갖는다. 모순쌍(L2), 인용 순회로만 도달 가능한 부정
  결과, citation-laundering 트랩(트리 앙상블), prompt-injection 픽스처 포함.
  검색 백엔드는 `Retriever` 프로토콜 뒤라 실 API(OpenAlex/S2) 어댑터로 교체 가능.
- **이중 모드**: 기본은 결정적 lexical 검색(3개 인덱스 + 인용 BFS + coverage
  정지 — 재현 가능, 테스트 가능). `--literature claude`는 쿼리 분해·stance
  판정·novelty 서술을 LLM에 맡기되, **검색 실행은 항상 결정적 백엔드**이고
  LLM 판정은 결정적 "supports"를 강등만 할 수 있다(론더링 구조적 차단).
  LLM 경로는 `tools=[]` 구조화 출력 전용이라 문헌 텍스트가 코드·셸을 실행할
  표면이 없고, 실패 시 lexical로 폴백해 세대를 막지 않는다.
- **가설 인증서 확장**: `supporting_evidence_ids`(화이트리스트 검증된 id만) +
  `nearest_prior_work`. 가설에는 **id만** 실린다 — claim 산문은 ledger·insight의
  blindness 스캔 표면에 절대 유입되지 않는다. novelty는 숫자 점수 없이 범주만
  (`replication / regime_extension / contradiction_test / unexplored`).
- **evidence 메모리는 ledger와 별도**: `experiments/evidence/evidence.jsonl`
  (append-only, timestamp로 크래시 재시도 구분) + 세대별 스냅샷
  `experiments/generations/gNNNN/evidence.json`(멱등). gate 점수가 존재하기
  전(propose 직전, 메인 스레드)에만 계산·기록되므로 불변식 4(blindness)는
  구조와 시간 순서 양쪽으로 보존된다.
- **감사와 비용**: `report`가 채택 가설의 모든 인용을 (논문, claim, locator)로
  해석하고(해석 불가 = 하드 에러), proposal/coder/literature 3원 비용을 합산한다.
  캠페인 LLM 문헌 예산(`llm_max_campaign_budget_usd`)은 evidence.jsonl 합산으로
  강제되고 초과 시 lexical로 강등된다.

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

### 정직한 한계 (Phase 4~5 과제)

- held-out 시드는 로컬 사용자가 루트에서 읽을 수 있다 (정책적 보호 수준). 코더는
  가드 훅으로 못 읽지만, 진짜 격리는 컨테이너/가상화 샌드박스가 필요
- 코더/학습 코드가 백그라운드 데몬을 남겨 평가 후 파일을 조작하는 TOCTOU류 공격,
  워크스페이스 밖 파일쓰기(감지는 되나 방지는 아님), 네트워크 접근은 OS 샌드박스
  (gVisor/Firecracker/컨테이너) 없이는 못 막는다
- 문헌 엔진은 mock corpus 전용이다 — 실 API(OpenAlex/S2) 어댑터는 `Retriever`
  시임만 있고 미구현. 코더 가설의 개입 계열 분류는 키워드 매칭이라 보수적
  (모호하면 분류 포기 → unexplored)
- claim-evidence ledger 기반 보고서·cross-model 적대적 리뷰어는 아직 없음
  (블루프린트 Phase 4~5)

## 파일 구조

```
research_contract.yaml    # Layer 1 타입드 계약 (v3: portfolio/gate/literature) — 불변, protected
orchestrator.py           # 포트폴리오 코디네이터 + gate + 코더 + 문헌 시임 (protected)
literature/engine.py      # 문헌 엔진: corpus/검색/stance/novelty/LLM 분석기 (protected)
literature/corpus/mock_corpus.json  # 가공 논문 13편/claim 15개 (protected, git 추적)
src/train.py              # 편집 가능 표면 (HYPERPARAMS 블록 + FEATURE_SPEC)
evaluation/evaluate.py    # 보호된 평가기 → metrics.json (--split dev|gate|test)
evaluation/dataset.py     # 합성 데이터 (train 공개 + dev/gate/test 시드 분리)
evaluation/heldout_config.json  # init 생성, untracked (3개 숨은 시드, worktree에 부재)
protection/hashes.json    # SHA-256 manifest (git 추적)
tests/test_phase2.py      # 단위 드릴 (가드 훅 / stagnation / blindness / feature_spec)
tests/test_phase3.py      # 문헌 드릴 (결정성 / blindness canary / 론더링 / 인젝션 / 계약 v3)
experiments/              # 런타임: state.json, ledger.jsonl, rounds/, generations/,
                          #   evidence/ (evidence.jsonl + 인증서), report/ (gitignored)
insight_memory.json       # ledger에서 재구성 가능한 파생 데이터 (gitignored)
.worktrees/               # 실험별 격리 (gitignored)
```

프로버넌스 규약: `main`에는 gate를 통과한 실험만 ff-merge로 쌓인다. 나머지 실험도
`hyp/<campaign>/rNNNN-*` 브랜치 + `experiments/ledger.jsonl`에 전부 남는다 —
gate 결정은 `record_type=gate` 레코드로 별도 기록(점수는 blindness 때문에 원장
안에만, 콘솔엔 PASS/FAIL만).

## protected 파일을 의도적으로 수정하려면

실행 중이 아닐 때: `chmod u+w <파일>` → 수정 → `uv run python orchestrator.py
init --force` (dev/gate/test 시드·manifest 재생성 + 재베이스라인). `--force`는
experiments/를 비우므로 이전 캠페인 기록이 필요하면 먼저 백업할 것. Phase 3는
계약 schema가 v3(literature 블록)라, v2 이하의 계약·상태에서 이어 돌릴 수 없고
`init --force`로 새 캠페인을 시작해야 한다.

## 남은 로드맵 (블루프린트 기준)

Phase 1(제약된 keep/reject) + Phase 2(포트폴리오·blind gate·LLM 코더) +
Phase 3(claim 수준 문헌 그라운딩·mock corpus) 완료. 다음:

4. Gome 스타일 branch-local directed update + SciNav 스타일 pairwise 평가
   (스칼라 지표가 불충분할 때). 증거 기반 heuristic move 조향(현재는
   annotation-only)도 이 단계의 momentum 메모리와 함께 넣는 것이 자연스럽다
5. claim-evidence ledger 기반 보고서 생성 + cross-model 적대적 리뷰어 (실행기와
   다른 모델 계열) + 컨테이너/가상화 샌드박스로 실제 격리 + 실 문헌 API 어댑터
