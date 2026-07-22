# AutoResearch — Executable Autoresearch Loop (Phase 1 + 2 + 3 + 4 + 5 + 6a)

2026 SOTA 블루프린트(Arbor / Gome / ERA / SciNav 종합)의 구현. **Karpathy 스타일
keep/reject 루프 + Arbor 스타일 상태 관리**에서 시작해, **병렬 가설 포트폴리오 +
blind admission gate + LLM 코딩 워커**(Phase 2), **claim 수준 문헌 그라운딩
(evidence graph)**(Phase 3), **방향성 브랜치 정제 — Gome search momentum + 증거
기반 조향 + successive halving + SciNav pairwise gate**(Phase 4)까지 확장했다.
에이전트 스웜보다 평가기(evaluator)·연구 계약(contract)·프로버넌스를 먼저 세우는
것이 원칙.

핵심 원칙: *신뢰할 수 있는 평가기 없이는 에이전트를 늘려도 더 과학적이 되지
않는다.* 이 저장소의 모든 구조는 "그럴듯해 보이는 실패"와 "검증된 진전"을
구분하는 데 맞춰져 있다.

## 빠른 시작

```bash
uv sync                                            # 의존성 (pyyaml, claude-agent-sdk)
uv run python orchestrator.py init                 # git 초기화 + 베이스라인 + 보호 장치
uv run python orchestrator.py ground               # 연구질문 인증서 (문헌 evidence flow)
uv run python orchestrator.py run --generations 4  # 병렬 포트폴리오 실행 (휴리스틱 + lexical 문헌 + halving + momentum)
uv run python orchestrator.py status               # 캠페인 상태 (문헌 통계 + momentum + 승인/리뷰 포함)
uv run python orchestrator.py report               # test 스플릿 다중시드 보고 → 최초엔 승인 대기(exit 3 + request_id)
uv run python orchestrator.py approve <request_id> # 출판 의도 승인 → 이후 report가 다중시드 평가·claims·보고서 봉인
uv run python orchestrator.py report --reviewer codex  # (승인 후) cross-model codex 적대 리뷰 포함 (옵트인)
uv run python orchestrator.py verify-protection
uv run python tests/test_phase2.py                 # Phase 2 드릴
uv run python tests/test_phase3.py                 # Phase 3 드릴 (문헌 그라운딩)
uv run python tests/test_phase4.py                 # Phase 4 드릴 (momentum / 조향 / halving / pairwise)
uv run python tests/test_phase5.py                 # Phase 5 드릴 (bootstrap / claims / report / gate / reviewer / families)
uv run python tests/test_phase6.py                 # Phase 6a 드릴 (샌드박스 argv·마스킹·fail-closed, Docker 불필요)
uv run python tests/test_phase6b.py                # Phase 6b 드릴 (실 문헌 fetch·추출·스냅샷, 오프라인 fake HTTP/LLM)
uv run python tests/test_phase6c.py                # Phase 6c 드릴 (TSP feasibility·objective 재계산·seed 부재·blindness)
```

`--gate pairwise`를 붙이면 admission(스칼라 epsilon)은 그대로 두고, admission을
통과한 후보 사이의 승자 선택만 blind LLM 심판단이 맡는다 (기본은 결정적 스칼라):

```bash
uv run python orchestrator.py run --generations 4 --proposer claude --gate pairwise
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

- **오프라인 corpus** (`literature/corpus/tsp_corpus.json`, `tsp-heuristics-v1`):
  TSP 휴리스틱 논문 13편 / claim 13개. 각 claim은 locator(섹션·표·쪽)·population·
  conditions·limitations·구조화 태그를 갖는다. 모순쌍, 인용 순회로만 도달 가능한
  부정 결과, citation-laundering 트랩, prompt-injection 픽스처 포함. 이 스냅샷은
  큐레이션된 오프라인 corpus이며(provenance 비어 있음), 실 문헌으로 갱신하려면
  네트워크 호스트에서 `ground --refresh`(OpenAlex/S2 어댑터)로 재생성한다. 검색
  백엔드는 `Retriever` 프로토콜 뒤라 실 API 어댑터로 교체 가능.
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

## Phase 4 — 방향성 브랜치 정제

탐색이 무작정 넓어지지 않고 **관측된 방향으로** 좁혀지게 만든다. 세 부분 모두
gate blindness·false-repair·문헌 폐포 불변식을 구조적으로 유지한다.

- **Gome search momentum + 증거 조향** (오프라인·결정적): 매 세대 시작 시 원장의
  experiment 레코드만 접어(`extract_update_vectors` → `search_momentum_table`)
  `{param}:{move}`별 방향 점수를 만든다 — accept +1.0 / gate-탈락 dev-improver
  +0.4 / 지표악화 −1.0 / 발산·timeout −1.0(+경계값 기록) / inconclusive −0.2,
  세대 경계마다 감쇠. **state에 저장하지 않고** insight_memory처럼 매번 원장에서
  재계산하므로 crash recovery에서 replay==live가 자명하게 성립하고, dev 신호와
  accept/reject 비트만 입력이라 gate 점수가 구조적으로 못 들어온다. 휴리스틱
  proposer는 이 momentum을 1순위, 문헌 stance(supports<none<contradicts, 반박
  증거는 강등만 하고 제거는 안 함)를 2순위, 정적 우선순위를 3순위로 후보를
  재정렬하고, 연속 accept 후 가속 스텝·발산 경계로의 기하 이분을 추가한다. K개 중
  최소 1개는 momentum 0·문헌 미지지 방향으로 강제 예약해 가설 붕괴를 막는다.
  `refinement.enabled=false`면 Phase 3 제안 동작과 바이트 동일. 문헌 조향은
  엔진의 순수 메서드 `Grounding.move_guidance()`(범주형 enum + 증거 id만, 산문·
  숫자 없음)로 나오고 `attach()`는 그대로 annotation-only.
- **Successive halving** (오프라인): 세대의 K개 브랜치가 전부 값싼 smoke rung을
  돌고, smoke 점수 상위 `max(min_keep, ceil(K·keep_fraction))`개만 dev rung으로
  올라간다. 탈락은 새 verdict `pruned` — **과학적 증거가 아니라 예산 결정**이라
  insight를 증류하지 않고 momentum 가중치도 0이되, 재제안 방지를 위해 tested
  endpoint는 등록한다(_finish_generation·replay 양쪽 대칭). 코더의 기계적 수리
  루프는 smoke 스테이지 **안에** 있어 halving 컷 시점엔 수리할 것이 없다 —
  false-repair 경계가 구조적으로 유지된다.
- **SciNav pairwise gate** (LLM 옵트인, `--gate pairwise`): admission은 **항상**
  결정적 스칼라 epsilon 규칙(anti-overfitting 보증)이고, admission을 통과한
  후보가 2명 이상일 때만 익명화된 blind 심판단(N=3 다수결)이 승자를 고른다. LLM은
  admission을 절대 완화하지 못한다. 심판은 계약·가설 인증서(id만)·bounded 코드
  diff·**dev** 메트릭만 보고 **gate 점수는 입력조차 받지 못한다**(폐포로 blindness
  성립). 심판별 A/B 라벨은 sha256 파리티로 뒤섞어 position bias를 상쇄하고, 판정은
  enum 4지선다로 강제되며 untrusted 후보 자료 앞에 anti-injection 프레이밍을 둔다.
  기권·과반 미달·SDK 실패·캠페인 예산 소진은 전부 결정적 스칼라 선택으로 폴백하고,
  결정적 대응물 `scalar_winner`를 gate 레코드에 항상 병기해 divergence를 사후
  감사할 수 있다. gate 레코드는 `mode`/`scalar_winner`/`pairwise`(투표 상세) 확장.

## Phase 5 — Assurance + 보고 (claim-evidence ledger · 결정적 보고서 · cross-model 리뷰어 · human gate)

캠페인의 결과를 **로그된 증거로만 뒷받침되는 주장**으로 봉인하고, 그 주장을 이질
모델 계열이 적대적으로 감사하며, 미접촉 test 스플릿(출판 아날로그)을 건드리기 전에
human 승인을 요구한다. 전부 완전 오프라인·결정적이고(리뷰어만 옵트인·비결정),
gate 점수는 어떤 산출물에도 유입되지 않는다.

- **다중 시드 finalist 재현 + paired bootstrap CI**: `heldout_config`가 v3로
  test 스플릿에 숨은 시드 N개(`assurance.finalist_seeds`, 기본 5)를 갖고, 평가기가
  `--seed-index`로 시드별 데이터셋을 채점하며 test에서만 예제별 제곱오차를 방출한다.
  `assurance/stats.py`가 baseline·incumbent를 **같은 데이터셋에서 예제 단위로
  페어링**(fingerprint 동일성 검증)해 N×600 풀에서 pooled paired bootstrap으로
  RMSE 차의 신뢰구간을 낸다. RNG 시드는 campaign·commit에서 파생·로그돼 원장에서
  재현 가능하다. 후보가 하나도 채택되지 않았으면 incumbent==baseline이라 effect 0·
  CI [0,0]·status inconclusive로 정직하게 보고한다.
- **claim-evidence ledger** (`experiments/claims.jsonl`): report 시점에 원장·통계·
  계약에서 **전체 재생성**되는 파생 산출물(5종 결정적 규칙: 주 효과·캠페인 요약·
  채택 개선·부정 결과·문헌 그라운딩). `final_report`에 `claims_sha256`로 봉인된다.
- **결정적 report.md + SVG 그림**: 보고서의 모든 숫자는 claim/meta 값으로만 삽입되고
  렌더 후 digit-scan이 추적 안 되는 숫자를 하드 거부한다(숫자-via-claim 불변식).
  그림 3종은 stdlib SVG로 불변 로그에서 바이트 결정적으로 생성돼 sha256로 감사된다.
- **cross-model codex 적대 리뷰어** (`--reviewer codex`, 옵트인): OpenAI 계열
  `codex exec`가 read-only 샌드박스에서 각 claim의 숫자를 raw test 데이터에 대조
  감사한다(ARIS). 응답은 프롬프트에 심은 echo_token으로 요청과 묶고 재검증하며,
  실패는 전부 `status="unavailable"`로 기록될 뿐 보고를 막지 않고 Claude 리뷰어로
  조용히 폴백하지 않는다(진짜 이질성 보존 — Phase 4 pairwise 심판의 상관 편향 해소).
  결과는 `experiments/report/review/`와 `review` 원장 레코드에만 남고 report.md엔
  안 들어간다(숫자-via-claim 스캔 일관성 유지).
- **human 승인 gate**: 최초 `report`(와 `--force` 재실행)는 승인 요청을 원장에 적고
  **exit 3 + request_id**를 낸다. `approve <request_id>`로 사람이 의도(commit·dev
  숫자·시드 계획·공시)를 승인해야 진행된다. 승인 상태는 state가 아니라 원장에서
  파생되고, fingerprint에 이전 봉인 수가 들어가 `--force`마다 재승인을 강제하며
  캠페인이 전진하면 stale 승인이 무효가 된다.
- **momentum 코더 계열 분류**: Phase 4의 조대한 `coder:none` momentum 키를,
  코더 diff를 결정적으로 분류한 계열(`feature_spec_interaction` 등)로 세분한다.
  계열은 실험 레코드에 저장돼 replay==live가 유지된다.

## Phase 6a — 실행 격리 샌드박스 (진짜 OS 격리 · Blueprint Layer 5)

신뢰불가 후보 코드가 실제로 *실행*되는 지점은 `evaluation/evaluate.py`의 `_run_train`
하나다(채점은 호스트 신뢰 경로에 남고 held-out 시드는 거기로 안 넘어간다). Phase 6a는
이 실행만 OS 격리로 감싼다.

- **`Sandbox` 프로토콜** (`sandbox/runner.py` — protected·stdlib·평가기가 절대경로로
  로드): `SubprocessSandbox`(현행 동작 이식, 기본값 `subprocess`에서 바이트 동일) +
  `ContainerSandbox`(`docker run`). 문헌의 `Retriever` 심 패턴을 그대로 미러.
- **컨테이너 하드닝**: `--network none`(네트워크 차단), `--read-only`(읽기전용 rootfs),
  `--cap-drop ALL` + `--security-opt no-new-privileges`, 비루트 user,
  `--memory/--cpus/--pids-limit`, `--rm --init`(ephemeral PID 네임스페이스 →
  TOCTOU 데몬 잔존 불가). 워크스페이스는 `:ro`, artifacts만 별도 fresh 호스트 dir로
  rw 마운트(워크트리 원본 불변).
- **시드/원장 마스킹**: 워크스페이스가 ROOT든 워크트리든 `heldout_config.json`은 빈
  파일로, `experiments/`(gate 점수)는 tmpfs로 마스킹 → 컨테이너 FS에 물리적 부재.
- **fail-closed**: 데몬/이미지 부재 시 절대 `subprocess`로 조용히 폴백하지 않고
  actionable 에러로 중단(`run`/`init`/`report` 진입 preflight). 이미지는 미리
  `docker pull` 해둬야 한다(runs는 `--network none`이라 on-demand pull 불가).
- **계약 v6 `sandbox` 블록**: `backend`(subprocess|container)·`image`(digest 핀)·
  `memory_mb`·`cpus`·`pids_limit`. backend는 init에서 캠페인 단위로 고정 → baseline·
  후보가 동일 파이프라인이라 Phase 5 paired bootstrap 유효. 평가기가
  `SUPPORTED_SANDBOX_BACKENDS`를 선언, init이 계약과 교차검증. `metrics["sandbox"]`
  provenance를 `run_evaluator`가 요청 backend와 에코 대조(우회·구버전 탐지).
- **gVisor 드롭인**: Linux에서 `sandbox/runner.py`에 `--runtime=runsc` 한 줄 추가로
  syscall 격리 강화(마운트/보안 모델 불변).

옵트인 예 (Docker 데몬 실행 + 이미지 pull 후):

```yaml
# research_contract.yaml
sandbox:
  backend: container
  image: "python:3.14-slim@sha256:<digest>"
  memory_mb: 512
  cpus: 1.0
  pids_limit: 128
```

```bash
docker pull python:3.14-slim@sha256:<digest>   # --network none이라 미리 받아둠
uv run python orchestrator.py init --force     # baseline도 컨테이너에서 학습
uv run python orchestrator.py run --generations 1
```

## 보호 모델 (Phase 1에서 닫은 것)

- **held-out 시드의 물리적 부재**: `evaluation/heldout_config.json`은 init 때
  생성되고 git이 추적하지 않는다. worktree는 추적 파일만 체크아웃하므로 후보
  워크스페이스에는 시드가 아예 존재하지 않는다
- **루트 평가기만 authoritative** + 평가기는 evaluation/ 밖을 신뢰하지 않음
  (예산 하드코딩 — init에서 contract와 1회 교차검증, dataset을 절대경로 import)
- **nonce 에코**: 오케스트레이터가 라운드마다 새 nonce를 전달, 평가기가
  metrics에 에코 (학습 서브프로세스에는 절대 전달 안 됨) → 위조 metrics 차단
- **격리된 학습 서브프로세스**: 처음부터 구성한 env(PATH, PYTHONHASHSEED=0),
  `-s -B` 플래그, 자체 세션 + 타임아웃 시 프로세스 그룹 SIGKILL. **Phase 6a: 이
  실행이 `Sandbox` 프로토콜을 거치며, `sandbox.backend: container`에선 OS 격리로
  승격된다(위 Phase 6a 절 참조)**
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

### 정직한 한계 (Phase 6b/6c 과제)

- **진짜 격리는 Phase 6a에서 구현됐다** — 단 `sandbox.backend: container`를 옵트인
  했을 때만 강제된다. 이 backend에선 held-out 시드가 마스킹돼 컨테이너 FS에 부재하고,
  TOCTOU(백그라운드 데몬)·워크스페이스 밖 파일쓰기·네트워크가 `--network none`·읽기
  전용 rootfs·ephemeral PID 네임스페이스로 OS 차단된다. **기본값 `subprocess`에선**
  학습 서브프로세스가 정책적 보호 수준이라 위 위협들이 남는다(현행 동작·Docker 불필요).
  하드닝 보장을 원하면 계약에서 `container` + digest 핀 이미지로 전환하면 된다
- 문헌 엔진은 mock corpus 전용이다 — 실 API(OpenAlex/S2) 어댑터는 `Retriever`
  시임만 있고 미구현 (Phase 6b). 코더 가설의 개입 계열 분류는 키워드 매칭이라 보수적
  (모호하면 분류 포기 → unexplored)
- **pairwise 심판은 같은 Claude 계열**이라 상관된 편향은 제거되지 않는다 — 진짜
  이질적 리뷰어는 Phase 5의 cross-model codex 리뷰어(`--reviewer codex`)로 닫혔다.
  다만 codex 리뷰는 로컬 `codex` 로그인·계정 한도에 의존하는 옵트인·advisory 경로라,
  실패 시 리뷰 부재로 기록될 뿐 보고를 막지 않는다(비결정적이라 결정적 산출물의
  입력이 아니다)
- successive halving은 파라미터 수만큼(휴리스틱 one-per-param ≤6) 브랜치를 채우는
  현 규모에선 K=8일 때 실익이 크고, K=4에선 dev 평가 시간 절감 정도다
- 실 문헌 API(OpenAlex/S2) 어댑터(6b), 실제 연구 도메인 태스크 교체(6c)는 아직
  없음. gVisor/Firecracker 백엔드는 Linux에서 `sandbox/runner.py`에 `--runtime=runsc`
  한 줄로 드롭인 가능(후속). 코더 에이전트 자체의 컨테이너화도 후속 하드닝

## 파일 구조

```
research_contract.yaml    # Layer 1 타입드 계약 (v6: + sandbox 블록) — 불변, protected
orchestrator.py           # 코디네이터 + gate + 코더 + 문헌 시임 + momentum/halving + 승인 게이트 + 다중시드 report + 샌드박스 배선 (protected)
sandbox/runner.py         # Phase 6a 실행 격리 (protected): Sandbox 프로토콜 / SubprocessSandbox / ContainerSandbox / preflight
assurance/                # Phase 5 순수 패키지 (protected): stats/claims/report_md/figures/svgfig/reviewer/gate/families
literature/engine.py      # 문헌 엔진: corpus/검색/stance/novelty/move_guidance/LLM 분석기 (protected)
literature/corpus/tsp_corpus.json  # TSP 휴리스틱 논문 13편/claim 13개 (protected, git 추적)
src/train.py              # 편집 가능 표면 (HYPERPARAMS 블록 + FEATURE_SPEC)
evaluation/evaluate.py    # 보호된 평가기 → metrics.json (--split dev|gate|test, --seed-index, --sandbox-backend)
evaluation/dataset.py     # 합성 데이터 (train 공개 + dev/gate 시드 + test 시드 N개 분리)
evaluation/heldout_config.json  # init 생성, untracked (schema v3: dev/gate 시드 + test 시드 N개, worktree에 부재)
protection/hashes.json    # SHA-256 manifest (22개 파일, git 추적)
tests/test_phase2.py      # 단위 드릴 (가드 훅 / stagnation / blindness / feature_spec)
tests/test_phase3.py      # 문헌 드릴 (결정성 / blindness canary / 론더링 / 인젝션 / 계약)
tests/test_phase4.py      # 정제 드릴 (momentum / 조향 / halving / pruned / pairwise / 계약)
tests/test_phase5.py      # assurance 드릴 (bootstrap / claims / report digit-scan / gate / codex 리뷰어 stub / families)
tests/test_phase6.py      # 샌드박스 드릴 (subprocess 회귀 / container argv·마스킹 / fail-closed / provenance echo, Docker 불필요)
experiments/              # 런타임: state.json, ledger.jsonl, rounds/, generations/, evidence/,
                          #   claims.jsonl, report/(report.md · figures/ · review/) (gitignored)
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
experiments/를 비우므로 이전 캠페인 기록이 필요하면 먼저 백업할 것. Phase 6a는
계약 schema가 v6(+ sandbox 블록)이고 heldout_config는 v3(test 시드 N개)라, 이전
계약·상태에서 이어 돌릴 수 없고 `init --force`로 새 캠페인을 시작해야 한다. 보호
manifest는 22개 파일(assurance/ 9 + sandbox/ 2 포함)이다.

## 남은 로드맵 (블루프린트 기준)

Phase 1(제약된 keep/reject) + Phase 2(포트폴리오·blind gate·LLM 코더) +
Phase 3(claim 수준 문헌 그라운딩·mock corpus) + Phase 4(Gome search momentum·
증거 조향·successive halving·SciNav pairwise gate) + Phase 5(assurance + 보고 —
다중 시드 finalist 재현 + paired bootstrap CI, claim-evidence ledger, 결정적
report.md + SVG 그림, cross-model codex 적대 리뷰어, human 승인 gate) +
Phase 6a(실행 격리 샌드박스 — `Sandbox` 프로토콜, Docker `ContainerSandbox`로
`train.py` 실행을 OS 격리: 네트워크 차단·읽기전용 rootfs·시드/원장 마스킹·ephemeral
PID 네임스페이스, `subprocess` 기본/`container` 옵트인, fail-closed) +
Phase 6b(실 문헌 API — `literature/sources.py`의 OpenAlex 어댑터 + `ground --refresh`
로 fetch→LLM 추출→동결 corpus 스냅샷; 네트워크는 refresh 때만, 캠페인은 얼린 스냅샷
위에서 lexical로 결정론 유지) + Phase 6c(도메인을 Euclidean-TSP 휴리스틱으로 교체 —
평가기가 hidden seed로 인스턴스 생성→solver에 좌표만 전달→tour 길이 재계산) 완료.

### 실 문헌 corpus 새로고침 (`ground --refresh`) — 네트워크 호스트 런북

기계장치는 완비돼 있고 오프라인 테스트까지 됐다(핸들러 포함, `tests/test_phase6b.py`).
**실제 fetch만 네트워크 호스트에서** 돈다. OpenAlex는 keyless(polite pool용 `mailto`),
S2 키는 **env 전용**(`S2_API_KEY`, 계약·커밋 금지). 스냅샷은 refresh 때만 네트워크를 쓰고,
캠페인은 얼린 스냅샷을 결정적으로 읽는다. 반-laundering 가드: 주입된 abstract가
`effect=improves`(유일한 support 부여 스탠스)를 만들지 못하도록 개선 단서가 없거나 인젝션
마커가 있으면 `conditional`로 강등.

전제(네트워크 호스트): `api.openalex.org`/`api.semanticscholar.org` 아웃바운드 HTTPS,
`extractor: claude`면 Claude SDK 자격.

```bash
export S2_API_KEY=...            # s2 fetch 때만. env 전용, 계약/커밋 금지.
chmod u+w literature/corpus/tsp_corpus.json
# 옵션: --source openalex|s2|both|contract(기본)  --extractor claude|deterministic
#       --max-papers N  --mailto you@example.com
uv run python orchestrator.py ground --refresh --source openalex --mailto you@example.com
git diff literature/corpus/tsp_corpus.json   # 사람이 태그 diff 리뷰
# 리뷰 초점: effect=improves(support 부여) claim, injection_flagged 논문, dropped_claims_policy
# (선택) 실 소스 캠페인이면 research_contract.yaml 의 retriever 를 openalex|s2 로 (chmod 후).
#        lexical 로 두면 provenance assertion 생략 — 어떤 스냅샷이든 동작.
uv run python orchestrator.py init --force   # 매니페스트 재해시 + 재베이스라인 (experiments/ 초기화)
```

한계: `claude` extractor는 비결정론이라 실 refresh는 매번 새 diff(오프라인 경로만 재현
가능); `_urllib_get` seam·실 SDK 호출은 실 네트워크에서만 최종 검증됨(오프라인 테스트
커버 밖 — 핸들러의 chmod 게이트·validate-before-overwrite·REVIEW 출력만 오프라인 검증).

### 남은 후속 (정직한 한계)

- `literature/corpus/tsp_corpus.json`은 아직 큐레이션된 오프라인 스냅샷(provenance 비어
  있음)이다. 실제 문헌 반영은 위 런북대로 네트워크 호스트에서 `ground --refresh` → 태그
  diff 사람 리뷰 → `init --force` 후에 완성된다.
- evidence_steering(patcher)은 정렬 완료: `orchestrator.rank()`/explore 필터가
  `engine.PARAM_TO_FAMILY`로 raw param을 family로 변환해 문헌 조향이 실제로 동작한다
  (`tests/test_phase4.py`의 실 corpus end-to-end 드릴로 검증). 남은 정직한 공백 하나 —
  corpus는 `neighborhood_operator` 개선을 `add_operator` move로 태깅하는데 편집 표면
  (`segment_max`)은 increase/decrease만 내므로 그 claim은 어떤 무브와도 매칭되지 않는다
  ("무브 추가"에 해당하는 편집 표면이 없음).
- gate/test는 held-out 인스턴스에서 solver를 돌리므로 신뢰 채점은 `container` 백엔드에서만
  완전하다(subprocess는 seed 파일을 절대경로로 읽을 수 있음 — dev/smoke 전용). `subprocess`로
  gate/report를 돌리면 orchestrator가 **항상 경고**하고,
  `sandbox.require_container_for_trusted_splits: true`면 아예 fail-closed로 막는다.
  report.md 헤더에도 신뢰 등급이 찍힌다. 자세한 위협 모델은 `tests/test_phase6c.py` canary 참고.
