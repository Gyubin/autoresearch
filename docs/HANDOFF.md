# AutoResearch — 세션 인수인계 (Phase 6b 이어가기 전 먼저 읽기)

이 문서는 빈 세션이 이 프로젝트를 이어받기 위한 **단일 진입점**이다. 순서대로
읽으면 된다: 이 파일 → `docs/BLUEPRINT.md`(설계 근거·레이어 사양) →
`README.md`(사용법) → 필요한 코드.

작성 시점: 2026-07-18 (Phase 6a 반영 갱신). Phase 1~5 + 6a(실행 격리 샌드박스) 완료.

---

## 0. 30초 오리엔테이션

- **무엇**: 2026 SOTA 블루프린트(`docs/BLUEPRINT.md`) 기반 자율 연구 루프. 신뢰
  가능한 평가기·계약·provenance를 에이전트 스웜보다 먼저 세운다는 게 핵심 철학.
- **어디**: `/Users/gyubin.son/workspace/dev/autoresearch` (git 저장소, 브랜치 main).
- **스택**: Python 3.14 (`.python-version` 고정), uv 가상환경, 의존성은 pyyaml +
  claude-agent-sdk 뿐. **오케스트레이션은 LangGraph가 아니라 Claude Agent Python
  SDK로 구현** (사용자 확정 결정 — 블루프린트의 LangGraph 추천은 무시).
- **완료**: Phase 1(제약된 keep/reject 루프) + Phase 2(병렬 포트폴리오 + blind
  admission gate + LLM 코딩 워커) + Phase 3(claim 수준 문헌 그라운딩 —
  mock corpus 전용, 이중 모드) + Phase 4(방향성 브랜치 정제 — Gome search
  momentum + 증거 조향 + successive halving + SciNav pairwise gate) + Phase 5
  (assurance + 보고 — 다중 시드 finalist 재현 + paired bootstrap CI, claim-
  evidence ledger, 결정적 report.md + SVG 그림, cross-model codex 리뷰어,
  human 승인 gate, momentum 코더 계열 분류) + Phase 6a(실행 격리 — `sandbox/`
  패키지의 `Sandbox` 프로토콜, Docker `ContainerSandbox`로 `train.py` 실행을 OS
  격리: 네트워크 차단·읽기전용 rootfs·시드/원장 마스킹·ephemeral PID 네임스페이스.
  `subprocess` 기본(현행 동일)·`container` 옵트인, fail-closed).
- **다음**: Phase 6b(실 문헌 API OpenAlex/S2 어댑터 — `Retriever` 뒤 + fetch-once
  스냅샷 캐시로 결정론 유지), Phase 6c(mock 합성 회귀를 실제 연구 도메인으로
  교체 — 평가기·계약·코퍼스 재설계).

## 1. 지금 실행해보기

```bash
cd /Users/gyubin.son/workspace/dev/autoresearch
uv sync
uv run python orchestrator.py status              # 현재 캠페인 상태 (문헌 통계 + 승인/리뷰 포함)
uv run python orchestrator.py verify-protection   # 보호 파일 무결성 (20개 파일)
uv run python tests/test_phase2.py                # Phase 2 단위 드릴
uv run python tests/test_phase3.py                # Phase 3 문헌 드릴
uv run python tests/test_phase4.py                # Phase 4 정제 드릴 (momentum/조향/halving/pairwise)
uv run python tests/test_phase5.py                # Phase 5 드릴 (bootstrap/claims/report/gate/reviewer/families)
uv run python orchestrator.py ground              # 연구질문 인증서 (문헌 evidence flow)
uv run python orchestrator.py run --generations 2 # 병렬 세대 실행 (휴리스틱 + lexical + halving + momentum, SDK 불필요)
uv run python orchestrator.py run --generations 1 --proposer claude --literature claude --gate pairwise  # 전체 LLM 경로
uv run python orchestrator.py report              # (승인 필요) test 스플릿 다중시드 보고 → exit 3 + request_id
uv run python orchestrator.py approve <request_id># 출판 의도 승인 → 이후 report가 진행
uv run python orchestrator.py report --reviewer codex  # 승인 후 cross-model codex 리뷰 포함(옵트인, reviewer.enabled 필요)
```

**Phase 5 report 흐름**: `report`는 미접촉 test 스플릿을 건드리기 전에 human 승인
을 요구한다(출판 아날로그). 최초 `report`(와 `--force` 재실행)는 승인 요청을
원장에 적고 **exit 3 + request_id**를 출력하며, 사람이 `approve <request_id>`로
의도(commit·dev 숫자·시드 계획·공시)를 승인한 뒤에야 다중 시드 test 평가 →
paired bootstrap CI → `experiments/claims.jsonl` → `experiments/report/report.md`
+ `figures/*.svg` → (옵트인) codex 리뷰 → `final_report` 봉인이 일어난다. 승인은
state가 아니라 원장에서 파생되며(fingerprint = incumbent·baseline commit +
contract·evaluator sha + 이전 봉인 수), 캠페인이 전진하면 stale이 되어 재승인이
필요하다.

실행 단위는 라운드가 아니라 **generation**이다. 한 세대에 K개(계약
`portfolio.parallel_branches`, Phase 4에서 8) 가설이 병렬 실행된다. 코더 가설은
`--proposer claude`일 때만 나온다(휴리스틱은 완전 오프라인·결정적). 문헌
그라운딩은 계약(`literature.enabled`)이 *여부*를, `--literature {lexical,claude}`
가 *방법*을 결정한다 — 기본 lexical은 완전 오프라인이다. **Phase 4**: search
momentum·증거 조향·successive halving은 계약(`refinement.enabled`,
`portfolio.halving.enabled`)이 켜면 동작하고 전부 오프라인·결정적이다. pairwise
gate는 계약(`pairwise_gate.enabled`)이 *여부*를, `--gate {scalar,pairwise}`가
*방법*을 정한다 — 기본 scalar는 완전 오프라인이고, admission은 어느 모드에서도
결정적 스칼라 규칙이다.

## 2. 파일 지도

| 파일 | 역할 | protected? |
|---|---|---|
| `research_contract.yaml` | 타입드 계약 (schema v6: + sandbox 블록) | ✅ 불변 |
| `orchestrator.py` | 코디네이터 + gate + 코더 + 문헌 시임 + momentum/halving + 승인 게이트 + 다중시드 report + 샌드박스 배선(preflight·echo) + CLI | ✅ |
| `sandbox/` | Phase 6a 실행 격리 (`runner.py`): `Sandbox` 프로토콜, `SubprocessSandbox`(현행), `ContainerSandbox`(docker), `build_sandbox`, `preflight`. stdlib·절대경로 로드·런타임 파일IO 없음 | ✅ (§3 폐포) |
| `assurance/` | Phase 5 순수 패키지: `stats.py`(paired bootstrap), `claims.py`(claim ledger), `report_md.py`+`figures.py`+`svgfig.py`(결정적 보고/그림), `reviewer.py`(codex 어댑터), `gate.py`(승인 파생), `families.py`(코더 계열) | ✅ (§3-13 폐포) |
| `literature/engine.py` | 문헌 엔진: corpus 검증, LexicalRetriever, EvidenceEngine, move_of/move_guidance, ClaudeLiteratureAnalyst, FallbackAnalyst (~1000줄, stdlib) | ✅ |
| `literature/corpus/mock_corpus.json` | 가공 논문 13편/claim 15개 (모순쌍·론더링 트랩·인젝션 픽스처 포함) | ✅ (git 추적) |
| `evaluation/evaluate.py` | 보호된 평가기 → metrics.json (`--split dev|gate|test`) | ✅ |
| `evaluation/dataset.py` | 합성 데이터, `load_split`, `SPLIT_SIZES` | ✅ |
| `evaluation/heldout_config.json` | 숨은 dev/gate/test 시드 (init 생성, **git 미추적**) | — |
| `src/train.py` | 편집 가능 표면: HYPERPARAMS 블록 + FEATURE_SPEC | 편집 대상 |
| `protection/hashes.json` | protected 파일 SHA-256 manifest (22개: +sandbox/ 2) | ✅ (git 추적) |
| `tests/test_phase2.py` | 단위 드릴 (가드 훅/stagnation/blindness/feature_spec) | — |
| `tests/test_phase3.py` | 문헌 드릴 (결정성/blindness canary/론더링/인젝션/계약 v5) | — |
| `tests/test_phase4.py` | 정제 드릴 (momentum fold/조향/halving/pruned/pairwise/계약 v5) | — |
| `tests/test_phase5.py` | assurance 드릴 (bootstrap/claims/report digit-scan/gate 상태기계/codex 리뷰어 stub/families/record 불활성) | — |
| `tests/test_phase6.py` | 샌드박스 드릴 (subprocess 회귀/container argv 하드닝/시드·원장 마스킹/fail-closed preflight/타임아웃 teardown/계약 v6/no-drift/provenance echo — Docker 불필요) | — |
| `experiments/` | 런타임: state.json, ledger.jsonl, rounds/, generations/, evidence/, **claims.jsonl**, **report/(report.md·report.json·{role}_test_s{k}.json·figures/·review/)** | gitignored |
| `insight_memory.json` | ledger에서 재구성 가능한 파생 교훈 | gitignored |
| `.worktrees/` | 실험별 격리 worktree | gitignored |
| `docs/BLUEPRINT.md` | 원본 연구 문서 (Phase 5 설계 사양) | — |
| `docs/archive/` | 종료된 캠페인의 ledger/rounds 아카이브 | — |

`orchestrator.py`에서 찾을 것: `load_contract`(계약 v6 파싱 — 최상위 키
화이트리스트 + halving/refinement/pairwise_gate/assurance/reviewer/human_gate/sandbox
블록), `run_evaluator`(sandbox CLI 인자 전달 + backend echo 검증), `_sandbox_preflight`
(fail-closed, `run`/`init`/`report` 진입), `_load_evaluator_declarations`(sandbox
backends 교차검증),
`cmd_report`(승인 게이트 → 다중시드 test → stats → claims → report/figures →
리뷰 → seal), `cmd_approve`, `_reviewer_raw_test`, `extract_update_vectors`/
`search_momentum_table`/`_momentum_weight`(Phase 4a — ledger 파생 search momentum,
state 미영속), `run_generation`(세대 루프 — momentum→grounding→propose→attach→
2단 halving 풀→gate→영속화), `_experiment_smoke_stage`/`_experiment_dev_stage`/
`_apply_halving`/`_prune_record`(Phase 4b), `_run_gate`(admission — 결정적 스칼라
epsilon) + `_select_gate_winner`/`_scalar_gate_winner`/`PairwiseJudge`/
`_judge_campaign_spend`/`_candidate_diff`(Phase 4c — selection·blind 심판·예산),
`_finish_generation`(원장·머지·상태), `ClaudeCoder` + `_make_worktree_guard`(코더
격리), `distill_insight`(blindness 불변식 + pruned→None), `replay_ledger_fields`
(복구용 상태 재구성), `recover`(크래시 복구), `_build_literature`/`_build_judge`/
`_BudgetGuardedLiterature`(서비스 구성·캠페인 예산), `_sdk_structured_query`(공용
SDK 호출 — proposer/judge 공유), `cmd_report`(test 스플릿 + 증거 감사 + 4원 비용
합산). `literature/engine.py`에서 찾을 것: `load_corpus`(검증·콘텐츠 정책),
`move_of`(공용 move 어휘 — orchestrator가 import), `EvidenceEngine.ground/attach`
(10단계 플로우·단일 권위 증거 작성자 — attach는 여전히 annotation-only),
`Grounding.move_guidance`(Phase 4 조향 — 범주형·id만·순수 메서드), `_hyp_stance`
(안티 론더링 규칙), `_coder_family`(코더 가설 계열 분류 — 모호하면 포기),
`ClaudeLiteratureAnalyst`(LLM 경로, supports 강등만 허용).

## 3. 반드시 보존할 불변식 (깨면 Phase 1~4 보장이 무너진다)

1. **계약 불변**: `orchestrator.py`는 `research_contract.yaml`을 읽기만 한다.
   baseline은 `experiments/state.json`에 기록. 계약/평가기/dataset을 바꾸면
   schema drift로 init이 fail-fast하고, `init --force`로 새 캠페인을 시작해야 한다.
2. **평가기 authoritative는 루트 사본만**. 워크스페이스의 evaluation/ 사본은 절대
   스코어링에 안 쓴다. 평가기는 evaluation/ 밖을 신뢰 안 함(예산·지표·split을
   하드코딩, init에서 계약과 교차검증).
3. **held-out 시드의 물리적 부재**: 3개 시드(dev/gate/test)는 git 미추적이라
   worktree에 존재하지 않는다.
4. **gate blindness**: gate 점수는 `record_type=gate` 원장 레코드 + gate 메트릭
   파일에만. insight·`best_primary`(항상 dev 점수)·proposer 컨텍스트·실험 레코드·
   **search momentum·steering.json·move_guidance·pairwise 심판 패킷** 어디에도
   누출 금지. `distill_insight`는 gate 레코드를 안 읽는다. `tests/test_phase2.py`
   (리터럴 스캔)·`tests/test_phase3.py`(canary 0.424242)·`tests/test_phase4.py`
   (momentum/심판 canary)가 검사 — Phase 5에서 새 필드 추가 시 깨지 않게.
5. **false-repair 금지**: 유효 개입 후의 런타임 실패(발산/timeout/no_skill/dev
   단계 실패)는 전부 과학적 증거 = valid_negative, 수리 금지. 코더 수리는 smoke
   단계 기계적 실패(nonzero_exit/missing_artifact/malformed_artifact)에만.
   **Phase 4b**: 코더 수리 루프는 `_experiment_smoke_stage` 안에 있고 halving 컷은
   smoke 스테이지 종료 후 메인 스레드에서 일어나므로, halving 탈락(`pruned`)이
   수리 대상이 되는 일은 구조적으로 없다.
6. **코더 격리**: `permission_mode="dontAsk"` + `allowed_tools=[]` + PreToolUse
   가드 훅이 유일 허용자(fail-closed). 코더 호출 전후 `_root_fingerprint` 스냅샷
   비교로 루트 탈출 탐지 → 탐지 시 캠페인 중단. Bash·네트워크 툴 비활성.
7. **write-ahead 순서**: 세대당 gate 레코드 먼저 → K개 실험 레코드 → ff-merge →
   실패 시 correction 레코드. `replay_ledger_fields`는 세대별로 그룹핑해 stagnation
   재구성(평면 재구성이면 승리 세대가 stagnation K-1로 오염됨).
8. **병렬 안전성**: git 공유 상태 변경(worktree add/remove/prune, branch -D,
   merge)만 `Git._mutation_lock`으로 직렬화. 워커 스레드는 자기 worktree 안 작업만,
   state/ledger에 안 쓴다(전부 배리어 후 메인 스레드). gc.auto=0.
9. **문헌 엔진 폐포 (Phase 3)**: `literature/`는 orchestrator를 import하지 않고,
   코퍼스 외 어떤 파일도 읽지 않으며(`ground()` 시그니처가 state·ledger·metrics
   를 받을 수 없음), **런타임에 아무것도 쓰지 않는다**(protected 경로라 런타임
   캐시가 생기면 루트 지문이 코더 탈출 오탐으로 캠페인을 중단시킴). 증거는
   ledger와 **별도**의 `experiments/evidence/evidence.jsonl`에, gate 실행 전
   시점에만 기록된다. 가설에는 증거 **id만** 실린다(claim 산문은 blindness 스캔
   표면에 유입 금지). LLM 문헌 판정은 결정적 "supports"를 강등만 할 수 있다.
   `tests/test_phase3.py`가 이 불변식들을 드릴한다(gate canary 포함). **Phase 4**:
   `Grounding.move_guidance()`도 같은 폐포의 순수 메서드다(범주형 enum + 증거 id만,
   산문·숫자 없음).
10. **search momentum은 파생·미영속 (Phase 4a)**: momentum은 state에 저장되지
    않고 매 세대 ledger의 experiment 레코드에서 재계산된다(`extract_update_vectors`
    → `search_momentum_table`). 입력 폐포가 `replay_ledger_fields`와 동일(experiment
    레코드만, gate·correction·evidence 무시)이라 gate 점수가 구조적으로 못 들어오고
    replay==live가 자명하다. **새 필드를 state에 영속화하려는 유혹을 피할 것** —
    파생 유지가 crash recovery를 공짜로 만든다. dev 신호와 accept/reject 비트만
    쓴다(§3-4). steering.json은 읽는 코드 없는 순수 감사 산출물.
11. **pruned는 예산 결정이지 과학 아님 (Phase 4b)**: halving 탈락 verdict `pruned`
    는 `distill_insight`에서 None, search momentum 가중치 0. 하지만 tested endpoint
    등록은 한다 — `_finish_generation`과 `replay_ledger_fields` **양쪽에서 aborted만
    제외**하는 대칭을 유지해야 replay==live가 성립(비대칭이면 재제안 스래싱). smoke
    프록시 점수는 `smoke_primary`(dev-split 단축 학습, proposer 가시 무해)로 별도
    기록하고 `primary`(dev)는 None.
12. **pairwise는 selection만, admission은 결정적 (Phase 4c)**: admission(gate-split
    epsilon)은 어느 모드에서도 결정적 스칼라 규칙이고 LLM이 완화 못 한다. pairwise
    심판은 admission을 통과한 후보 사이에서 승자만 고른다(`_select_gate_winner`는
    항상 admitted 집합에서만 반환). 심판 입력 폐포는 계약·가설 인증서(id만)·bounded
    diff·**dev** 메트릭 — gate 점수·`state["gate"]`·gate 메트릭·`gate_record["results"]`
    는 절대 안 들어간다. 판정은 enum 4지선다로 강제, untrusted 후보 자료 앞에
    anti-injection 프레이밍. 기권·실패·예산소진은 결정적 스칼라로 폴백하고
    `scalar_winner`를 항상 병기(divergence 감사). gate 레코드의 `pairwise.cost_usd`는
    **세대별 delta**를 기록한다(심판 인스턴스가 세대 간 살아있어 `total_cost_usd`는
    누적기 — 누적값을 기록하면 `_judge_campaign_spend`가 prefix-sum을 재합산해
    2차 과다계상하고 재시작 시 원장이 자기모순).
13. **assurance 폐포 (Phase 5)**: `assurance/`는 stdlib만 import하고 orchestrator·
    literature·yaml을 import하지 않으며, **파일 IO를 하지 않는다**(literature보다
    엄격 — 데이터 in, 문자열/dict out). 유일 예외는 `reviewer.run_review`로,
    injectable runner로 `codex exec`를 부르고 자기 scratch workdir
    (`experiments/report/review/<request_id>/`)만 건드린다 — repo root·protected
    dir 절대 접근 금지. 모든 read/write는 orchestrator의 `cmd_report`가 하고 전부
    `experiments/**` 아래로만 간다(protected 경로에 런타임 쓰기가 생기면 root
    fingerprint가 코더 탈출로 오탐한다). wall clock·ambient randomness 없음
    (bootstrap 시드는 계약/commit 파생·로그, 타임스탬프는 인자로 전달)이라 claims·
    report.md·SVG가 바이트 결정적이고 sha256으로 감사된다. **gate 점수는 Phase 5
    산출물 어디에도 안 들어간다**(claims·report·리뷰어 패킷 전부 dev+test 숫자만).
    report.md의 모든 숫자는 claim/meta 값으로만 삽입되고 렌더 후 `scan_untraced_
    digits`가 검사한다. codex 리뷰는 **advisory** — 실패는 status "unavailable"로
    기록될 뿐 파이프라인을 막지 않고, Claude 리뷰어로 조용히 폴백하지 않는다
    (이질성 목적 보존). `tests/test_phase5.py`가 이 불변식들을 드릴한다.
14. **샌드박스 폐포·fail-closed·시드 부재** (Phase 6a): `sandbox/`는 신뢰 경로다 —
    stdlib만, orchestrator를 import하지 않으며, 런타임 파일IO는 후보의 스크래치
    (artifacts/log) 생성뿐이다. 평가기는 이 모듈을 **절대경로로** 로드해(sys.path
    경유 금지) 워크스페이스 사본이 격리 코드를 가릴 수 없다. `container` backend는
    시드·원장을 마스킹해 어떤 워크스페이스(ROOT 포함)에서도 시드가 컨테이너 FS에
    부재하고, 채점은 호스트에 남아 시드가 샌드박스로 안 들어간다. 데몬/이미지 부재
    시 **절대 `subprocess`로 조용히 폴백하지 않고** actionable 에러로 중단한다
    (reviewer 규율과 동일). backend는 init에서 고정 → baseline·후보 동일 파이프라인.
    `run_evaluator`는 backend echo를 검증한다. `tests/test_phase6.py`가 드릴한다.

## 4. Mock 태스크 (실증 수단)

순수 파이썬 합성 회귀. `y = 0.3 + w·x + 0.3·x0·x1 + N(0,0.25)`, 8개 이질 스케일
피처. **x0·x1 상호작용은 선형 모델이 못 잡아 하이퍼파라미터로는 못 넘는 floor
(~0.39)를 만든다.** 코더가 `src/train.py`의 `FEATURE_SPEC`에 곱 항([0,1] 등)을
추가하면 floor를 뚫는다(실증: 코더가 dev RMSE ~0.25 도달, gate 통과 후 merge).
평가기는 아티팩트의 `feature_spec`(원본 인덱스 곱 목록, 최대 32항·차수 3)을
**데이터로 검증·스코어링** — 후보 코드를 평가기 안에서 실행하지 않는다.

이 태스크는 실제 과학이 아니라 **파이프라인 실증용 대리 문제**다. Phase 5는
실제 연구 도메인으로 교체하거나, assurance 레이어를 이 태스크 위에 얹어
테스트할 수 있다.

## 5. Phase 진입 지점 (블루프린트 §2, §8 참조)

### Phase 3 — 문헌 그라운딩 (Layer 2) ✅ 완료
`literature/` 별도 통제 서비스(오프라인 mock corpus + `Retriever` 시임), 이중
모드(결정적 lexical 기본 / `--literature claude` 옵트인 — tools=[] 구조화 출력
전용, supports 강등만 허용), 가설 인증서 `supporting_evidence_ids`/
`nearest_prior_work`(id만·화이트리스트), 범주형 novelty, 모순 보고, coverage 정지,
`ground` 인증서, evidence 메모리(ledger와 별도), report 증거 감사 + 캠페인 문헌
예산. 남은 것: 실 API 어댑터(OpenAlex/S2)는 `Retriever` 프로토콜 뒤에 미구현.

### Phase 4 — 방향성 브랜치 정제 (Layer 4 심화) ✅ 완료
- **4a search momentum + 증거 조향**: `extract_update_vectors`/`search_momentum_table`
  (ledger 파생·미영속·§3-10), 휴리스틱 3단 정렬(momentum > 문헌 stance > 정적) +
  값 진행(가속 스텝·발산 경계 기하 이분) + explore 슬롯 강제(가설 붕괴 방어).
  이연했던 증거 조향은 `Grounding.move_guidance()`(순수·범주형·id만)로 넣고
  `attach()`는 annotation-only 유지. `refinement.enabled=false`면 Phase 3와 바이트
  동일. ClaudeProposer 프롬프트에 momentum/guidance 섹션 + 소프트 explore 재시도.
- **4b successive halving**: `run_experiment`를 smoke/dev 스테이지로 분리, 세대
  2단 풀(전원 smoke → `_apply_halving` 랭크 컷 → 생존자 dev). 탈락 verdict `pruned`
  (§3-11). K=8 + `halving.enabled`가 기본 계약.
- **4c SciNav pairwise gate**: `_run_gate` = admission(결정적 스칼라) +
  `_select_gate_winner`(selection). `PairwiseJudge`(N=3 blind 다수결, sha256 라벨
  스왑, enum 판정, anti-injection), `--gate pairwise` 옵트인, 스칼라 폴백 +
  `scalar_winner` 병기, 캠페인 예산 가드(§3-12).
- 적대적 리뷰: 세션 사용량 한도로 자동 4렌즈 중 1렌즈만 완주(robustness-recovery,
  pairwise 비용-누적 버그 1건 발견→수정, 회귀 드릴 추가). 나머지 렌즈는 인라인
  리뷰 + 재실행으로 보강.

### Phase 5 — assurance + 보고 (Layer 8, 9) ✅ 완료
- **다중 시드 finalist 재현**: heldout_config schema v3(test 시드 N개, `finalist_
  seeds`=5), `evaluate.py --seed-index`(test 전용) + `per_example_sq_errors`,
  `run_evaluator` seed_index 에코, `init`이 2+N 시드 생성 + `MAX_TEST_SEEDS`(=16,
  `MAX_FINALIST_SEEDS`와 교차검증).
- **paired bootstrap CI** (`assurance/stats.py`, stdlib): 시드별 데이터셋에서 예제
  단위로 baseline/incumbent 제곱오차를 페어링(fingerprint 동일성 assert), N×600
  풀에서 pooled 리샘플, 결정적 로그 시드. 계층 bootstrap은 기각(시드는 클러스터
  아님). incumbent==baseline은 aliasing→effect 0/CI[0,0]/inconclusive.
- **claim-evidence ledger** (`assurance/claims.py`): report 시점 전체 재생성,
  5종 결정적 규칙(primary_effect·campaign_summary·admitted_improvement·
  negative_result·literature_grounding), `final_report`에 `claims_sha256` 봉인.
- **결정적 report.md + SVG** (`assurance/report_md.py`+`figures.py`+`svgfig.py`):
  숫자는 claim/meta로만 삽입되고 렌더 후 digit-scan으로 강제, 그림 3종 sha256.
- **cross-model codex 리뷰어** (`assurance/reviewer.py`): `codex exec` 서브프로세스
  (read-only 샌드박스·stdin 패킷·`--output-schema`·echo_token 재검증), ARIS식
  claim 감사, advisory(실패=unavailable, Claude 폴백 없음). `--reviewer codex`
  옵트인 + `reviewer.enabled`.
- **human gate** (`assurance/gate.py`+`approve`): 출판(첫 report)·`--force`에 승인,
  원장 파생 상태, exit 3.
- **momentum 코더 계열** (`assurance/families.py`): 결정적 diff 분류, `coder_family`
  를 실험 레코드에 저장(replay==live).

### Phase 6a — 실행 격리 샌드박스 (Layer 5 심화) ✅ 완료
- **격리 경계는 `_run_train` 하나**: 신뢰불가 후보 코드가 실제로 *실행*되는 곳은
  `evaluation/evaluate.py`의 `_run_train`뿐이다(채점은 호스트 신뢰 경로에 남고 시드는
  거기로 안 넘어감). 그래서 이 실행만 격리하면 시드 읽기·네트워크·TOCTOU가 닫힌다.
- **`Sandbox` 프로토콜** (`sandbox/runner.py`, protected·stdlib·절대경로 로드):
  `SubprocessSandbox`(현행 동작 이식, `backend: subprocess` 바이트 동일 기본) +
  `ContainerSandbox`(`docker run` — `--network none`·`--read-only`·`--cap-drop ALL`·
  `--security-opt no-new-privileges`·비루트 user·`--memory/--cpus/--pids-limit`·
  `--rm --init`). `literature`의 `Retriever` 심 패턴을 그대로 미러.
- **시드/원장 마스킹**: 워크스페이스가 ROOT든 워크트리든 `heldout_config.json`은 빈
  파일로, `experiments/`(gate 점수)는 tmpfs로 마스킹 → 컨테이너 FS에서 물리적 부재.
  artifacts는 별도 fresh 호스트 dir만 rw로 마운트(워크트리 원본 불변).
- **fail-closed**: 데몬/이미지 부재 시 절대 `subprocess`로 조용히 폴백하지 않고
  actionable 에러로 중단(`sandbox.preflight`, `run`/`init`/`report` 진입 preflight).
  reviewer가 Claude로 안 떨어지는 것과 동일 규율.
- **계약 v6**: `sandbox` 블록(backend/image/memory_mb/cpus/pids_limit), `schema_version
  6`, `protected_globs`에 `sandbox/**`. backend는 init에서 캠페인 단위로 고정 →
  baseline·후보가 동일 파이프라인이라 Phase 5 paired bootstrap 유효. 평가기가
  `SUPPORTED_SANDBOX_BACKENDS`를 선언, init이 계약과 교차검증(budgets/metric와 동일).
- **provenance echo**: `metrics["sandbox"]={backend,image,isolated}`, `run_evaluator`가
  요청 backend와 에코 일치 검증(nonce/split 에코처럼) → 우회·구버전 탐지.
- **gVisor 후속 드롭인**: Linux에서 `sandbox/runner.py`에 `--runtime=runsc` 한 줄
  추가로 syscall 격리 강화(마운트/보안 모델 불변). Firecracker/코더 자체 컨테이너화는
  후속. 드릴: `tests/test_phase6.py`(Docker 불필요, fake runner로 argv/마스킹/preflight).

### Phase 6b/6c — 실 문헌 + 실제 도메인 (Layer 2 심화) ← 다음
- 실 문헌 API(OpenAlex/S2) 어댑터를 `Retriever` 프로토콜 뒤에 구현 + fetch-once
  스냅샷 캐시(결정론 유지) + `ground --refresh`.
- mock 합성 회귀를 실제 연구 도메인으로 교체(평가기·계약·문헌 코퍼스 재설계).

## 6. 개발 워크플로 관습 (이 프로젝트에서 지킨 것)

- **설계 먼저**: 큰 변경 전 Plan 서브에이전트로 설계안 받고 교정(git 함정, 상태
  원자성, SDK 사양 등 구체 질문). 코드 작성 전 수치·API를 실측/조사.
- **적대적 검증**: 구현 후 다중 렌즈 리뷰(correctness / 평가기 우회 / 스키마 정합 /
  견고성) → 발견별 반박 검증 → 확정 결함만 수정. Phase 2에서 recovery 렌즈만
  완주(0 확정), 나머지는 사용량 한도로 중단 — 직접 드릴로 대체 검증함. **Phase 3+
  에선 이 리뷰를 완주할 것.**
- **테스트를 실제 디렉토리 밖에서**: scratchpad에 rsync 복제 후 E2E 드릴, 검증되면
  실제 디렉토리에 반영.
- **protected 파일 수정 절차**: `chmod u+w <파일>` → 수정 → `init --force`(시드·
  manifest 재생성 + 재베이스라인, experiments/ 초기화됨).

## 7. 알려진 한계 / 주의 (블루프린트 §7, README 참조)

- **시드 읽기·네트워크·TOCTOU는 이제 `sandbox.backend: container`에서 OS로 막힌다**
  (Phase 6a): 컨테이너가 `heldout_config.json`을 마스킹(물리적 부재)하고 `--network
  none`·읽기전용 rootfs·ephemeral PID 네임스페이스로 데몬 잔존·워크스페이스 밖 쓰기를
  차단한다. **단 기본값은 `subprocess`** — 이 backend에선 아래의 정책적 한계가 그대로
  남는다(현행 동작 보존·Docker 불필요). 하드닝 보장을 원하면 계약에서 `container`로
  옵트인하고 이미지를 digest로 핀해야 한다.
- (`subprocess` backend) held-out 시드는 로컬 사용자가 루트에서 읽을 수 있고, 코더는
  가드 훅으로 못 읽지만 학습 서브프로세스는 정책적 보호 수준이다. TOCTOU(백그라운드
  데몬), 워크스페이스 밖 파일쓰기(탐지는 되나 방지는 아님), 네트워크는 OS 샌드박스
  (`container`) 없이는 못 막는다. codex 리뷰어의 read-only 샌드박스는 별개 계층.
- ClaudeCoder·pairwise 심판은 **계정 사용량 한도**에 걸릴 수 있다(SDK 에러).
  크래시 복구가 중단된 세대를 정리하므로 안전하지만, 한도 리셋 후 재개. 문헌 LLM
  경로는 lexical로, pairwise는 결정적 스칼라로 폴백해 세대를 막지 않는다(`mode`/
  `fallback_reason`로 기록). **주의**: Phase 4 적대적 리뷰 자체가 이 한도에 걸려
  자동 4렌즈 중 일부만 완주했다 — LLM 다중에이전트 리뷰를 돌릴 땐 한도를 감안할 것.
- 문헌 엔진은 mock corpus 전용(실 API 어댑터는 `Retriever` 시임만). 코더 가설의
  개입 계열 분류는 키워드 매칭 — 모호하면 분류를 포기한다(unexplored). pairwise
  심판은 같은 Claude 계열이라 상관 편향이 남는다(진짜 이질 리뷰어는 Phase 5).
- Phase 6b/6c 미구현: 실 문헌 API 어댑터(현재 `Retriever` 시임만), 실제 연구 도메인
  태스크 교체. codex 리뷰어는 로컬 `codex` 로그인·계정 한도에 의존하며(옵트인·
  advisory), 실패 시 리뷰 부재로 기록될 뿐 보고를 막지 않는다.
- **Phase 5 assurance는 §7의 시드 트러스트 경계를 상속한다 — 이 경계는 이제
  `sandbox.backend: container`로 닫을 수 있다(Phase 6a)**. `subprocess` backend에선
  학습 서브프로세스에 OS 샌드박스가 없어 적대적 코더가 `train.py`에서 worktree 밖을
  traverse해 숨은 시드(gate·test)를 읽거나 gitignore된 `experiments/`(원장 포함)에
  out-of-band로 쓸 수 있다. `container` backend에선 시드/원장이 마스킹되고 네트워크·
  worktree 밖 쓰기가 OS로 차단되므로 이 경로가 닫힌다. 어느 쪽이든 다중 시드 test
  재현·paired bootstrap·claim ledger·승인 게이트의 보증은 **정직한 후보 + 결정적
  파이프라인**에 대해 성립하고, headline `primary_effect`는 원장이 아니라 live
  bootstrap에서만 나오므로 원장 위조로 못 뒤집는다. 적대적 코더에 대한 강제는
  `container`에서 성립(적대 리뷰 evaluator-bypass 렌즈). 크래시로 봉인 못 한
  report_attempt는 다중검정 공시에 `crashed_report_attempts`로 보수적으로 집계된다.

## 8. 새 세션 시작 프롬프트 (예시)

> "이 디렉토리(`~/workspace/dev/autoresearch`)의 AutoResearch 시스템에서 Phase 6b
> (실 문헌 API OpenAlex/S2 어댑터 — `Retriever` 뒤 + fetch-once 스냅샷 캐시)를
> 이어서 구현하고 싶어. `docs/HANDOFF.md`와 `docs/BLUEPRINT.md`를 먼저 읽고, 현재
> 상태(6a 실행 격리 샌드박스까지 완료)를 파악한 뒤 설계안부터 제안해줘."

memory(`autoresearch-project-decisions`)에 구속력 있는 결정이 기록돼 있고 자동
recall되지만, 위 두 docs가 실제 사양의 원천이다.
