# AutoResearch — 세션 인수인계 (Phase 4~5 이어가기 전 먼저 읽기)

이 문서는 빈 세션이 이 프로젝트를 이어받기 위한 **단일 진입점**이다. 순서대로
읽으면 된다: 이 파일 → `docs/BLUEPRINT.md`(설계 근거·Phase 4~5 사양) →
`README.md`(사용법) → 필요한 코드.

작성 시점: 2026-07-17 (Phase 3 반영 갱신). Phase 1 + 2 + 3 완료.

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
  mock corpus 전용, 이중 모드).
- **다음**: Phase 4(방향성 브랜치 정제) → 5(assurance + 보고서).

## 1. 지금 실행해보기

```bash
cd /Users/gyubin.son/workspace/dev/autoresearch
uv sync
uv run python orchestrator.py status              # 현재 캠페인 상태 (문헌 통계 포함)
uv run python orchestrator.py verify-protection   # 보호 파일 무결성 (11개 파일)
uv run python tests/test_phase2.py                # Phase 2 단위 드릴
uv run python tests/test_phase3.py                # Phase 3 문헌 드릴
uv run python orchestrator.py ground              # 연구질문 인증서 (문헌 evidence flow)
uv run python orchestrator.py run --generations 2 # 병렬 세대 실행 (휴리스틱 + lexical 문헌, SDK 불필요)
uv run python orchestrator.py run --generations 1 --proposer claude --literature claude  # 전체 LLM 경로
uv run python orchestrator.py report              # test 스플릿 최종 보고 + 증거 감사 (1회성)
```

실행 단위는 라운드가 아니라 **generation**이다. 한 세대에 K개(계약
`portfolio.parallel_branches`, 기본 4) 가설이 병렬 실행된다. 코더 가설은
`--proposer claude`일 때만 나온다(휴리스틱은 완전 오프라인·결정적). 문헌
그라운딩은 계약(`literature.enabled`)이 *여부*를, `--literature {lexical,claude}`
가 *방법*을 결정한다 — 기본 lexical은 완전 오프라인이다.

## 2. 파일 지도

| 파일 | 역할 | protected? |
|---|---|---|
| `research_contract.yaml` | 타입드 계약 (schema v3: portfolio/gate/literature 블록) | ✅ 불변 |
| `orchestrator.py` | 코디네이터 + gate + ClaudeCoder + 문헌 시임 + CLI (~2900줄) | ✅ |
| `literature/engine.py` | 문헌 엔진: corpus 검증, LexicalRetriever, EvidenceEngine, ClaudeLiteratureAnalyst, FallbackAnalyst (~950줄, stdlib) | ✅ |
| `literature/corpus/mock_corpus.json` | 가공 논문 13편/claim 15개 (모순쌍·론더링 트랩·인젝션 픽스처 포함) | ✅ (git 추적) |
| `evaluation/evaluate.py` | 보호된 평가기 → metrics.json (`--split dev|gate|test`) | ✅ |
| `evaluation/dataset.py` | 합성 데이터, `load_split`, `SPLIT_SIZES` | ✅ |
| `evaluation/heldout_config.json` | 숨은 dev/gate/test 시드 (init 생성, **git 미추적**) | — |
| `src/train.py` | 편집 가능 표면: HYPERPARAMS 블록 + FEATURE_SPEC | 편집 대상 |
| `protection/hashes.json` | protected 파일 SHA-256 manifest (11개) | ✅ (git 추적) |
| `tests/test_phase2.py` | 단위 드릴 (가드 훅/stagnation/blindness/feature_spec) | — |
| `tests/test_phase3.py` | 문헌 드릴 (결정성/blindness canary/론더링/인젝션/계약 v3) | — |
| `experiments/` | 런타임: state.json, ledger.jsonl, rounds/, generations/, evidence/, report/ | gitignored |
| `insight_memory.json` | ledger에서 재구성 가능한 파생 교훈 | gitignored |
| `.worktrees/` | 실험별 격리 worktree | gitignored |
| `docs/BLUEPRINT.md` | 원본 연구 문서 (Phase 4~5 설계 사양) | — |
| `docs/archive/` | 종료된 캠페인의 ledger/rounds 아카이브 | — |

`orchestrator.py`에서 찾을 것: `load_contract`(계약 v3 파싱, literature 블록),
`run_generation`(세대 루프 — 문헌 grounding→propose→attach→영속화 시임 포함),
`run_experiment`(가설 1개 워커), `_run_gate`(blind gate), `_finish_generation`
(원장·머지·상태), `ClaudeCoder` + `_make_worktree_guard`(코더 격리),
`distill_insight`(blindness 불변식), `replay_ledger_fields`(복구용 상태 재구성),
`recover`(크래시 복구), `_build_literature`/`_BudgetGuardedLiterature`(문헌
서비스 구성·캠페인 예산), `cmd_ground`(연구질문 인증서), `cmd_report`(test
스플릿 + 증거 감사 + 3원 비용 합산). `literature/engine.py`에서 찾을 것:
`load_corpus`(검증·콘텐츠 정책), `EvidenceEngine.ground/attach`(10단계 플로우·
단일 권위 증거 작성자), `_hyp_stance`(안티 론더링 규칙), `_coder_family`
(코더 가설 계열 분류 — 모호하면 포기), `ClaudeLiteratureAnalyst`(LLM 경로,
supports 강등만 허용).

## 3. 반드시 보존할 불변식 (깨면 Phase 1+2 보장이 무너진다)

1. **계약 불변**: `orchestrator.py`는 `research_contract.yaml`을 읽기만 한다.
   baseline은 `experiments/state.json`에 기록. 계약/평가기/dataset을 바꾸면
   schema drift로 init이 fail-fast하고, `init --force`로 새 캠페인을 시작해야 한다.
2. **평가기 authoritative는 루트 사본만**. 워크스페이스의 evaluation/ 사본은 절대
   스코어링에 안 쓴다. 평가기는 evaluation/ 밖을 신뢰 안 함(예산·지표·split을
   하드코딩, init에서 계약과 교차검증).
3. **held-out 시드의 물리적 부재**: 3개 시드(dev/gate/test)는 git 미추적이라
   worktree에 존재하지 않는다.
4. **gate blindness**: gate 점수는 `record_type=gate` 원장 레코드 + gate 메트릭
   파일에만. insight·`best_primary`(항상 dev 점수)·proposer 컨텍스트·실험 레코드
   어디에도 누출 금지. `distill_insight`는 gate 레코드를 안 읽는다.
   `tests/test_phase2.py`가 이 불변식을 검사 — Phase 3~5에서 새 필드 추가 시 이
   테스트를 깨지 않게.
5. **false-repair 금지**: 유효 개입 후의 런타임 실패(발산/timeout/no_skill/dev
   단계 실패)는 전부 과학적 증거 = valid_negative, 수리 금지. 코더 수리는 smoke
   단계 기계적 실패(nonzero_exit/missing_artifact/malformed_artifact)에만.
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
   `tests/test_phase3.py`가 이 불변식들을 드릴한다(gate canary 포함).

## 4. Mock 태스크 (실증 수단)

순수 파이썬 합성 회귀. `y = 0.3 + w·x + 0.3·x0·x1 + N(0,0.25)`, 8개 이질 스케일
피처. **x0·x1 상호작용은 선형 모델이 못 잡아 하이퍼파라미터로는 못 넘는 floor
(~0.39)를 만든다.** 코더가 `src/train.py`의 `FEATURE_SPEC`에 곱 항([0,1] 등)을
추가하면 floor를 뚫는다(실증: 코더가 dev RMSE ~0.25 도달, gate 통과 후 merge).
평가기는 아티팩트의 `feature_spec`(원본 인덱스 곱 목록, 최대 32항·차수 3)을
**데이터로 검증·스코어링** — 후보 코드를 평가기 안에서 실행하지 않는다.

이 태스크는 실제 과학이 아니라 **파이프라인 실증용 대리 문제**다. Phase 3~5는
실제 연구 도메인으로 교체하거나, 문헌·assurance 레이어를 이 태스크 위에 얹어
테스트할 수 있다.

## 5. Phase 4~5 진입 지점 (블루프린트 §2, §8 참조)

### Phase 3 — 문헌 그라운딩 (Layer 2) ✅ 완료
구현됨: `literature/` 별도 통제 서비스(오프라인 mock corpus + `Retriever` 시임),
이중 모드(결정적 lexical 기본 / `--literature claude` 옵트인 — tools=[] 구조화
출력 전용, supports 강등만 허용), 가설 인증서 `supporting_evidence_ids`/
`nearest_prior_work`(id만·화이트리스트 검증), 범주형 novelty(숫자 점수 없음),
모순 보고, coverage 정지, `ground` 인증서, evidence 메모리(ledger와 별도,
`experiments/evidence/`), report 증거 감사 + 3원 비용 합산 + 캠페인 문헌 예산.
적대적 리뷰(4렌즈, 14 발견 → 13 확정 전부 수정) 완주. 남은 것: 실 API 어댑터
(OpenAlex/S2)는 미구현 — `Retriever` 프로토콜 뒤에 붙이면 된다.

### Phase 4 — 방향성 브랜치 정제 (Layer 4 심화)
목표: Gome식 업데이트 + SciNav pairwise. 진입 지점:
- `run_experiment` 후 실패/성공에서 "업데이트 벡터"(관측→근본원인→방향→bounded
  변경) 추출 → 같은 브랜치의 다음 세대 가설을 편향. `insight_memory`의 momentum을
  강화. **Phase 3에서 이연한 것**: 증거 기반 heuristic move 조향(현재 문헌은
  heuristic 가설에 annotation-only — `EvidenceEngine.attach`가 사후 부착만 한다).
  Gome momentum과 함께 이 단계에서 넣는 것이 자연스럽다.
- 스칼라 지표가 불충분한 산출물엔 pairwise 평가기 추가(다수 blind 심판, 다른 모델
  계열 권장 — ARIS). `_run_gate`를 pairwise 모드로 확장하거나 병렬 gate 추가.
- 비용 인지 스케줄링 + successive halving: 값싼 proxy(smoke)로 약한 브랜치 조기
  제거는 이미 부분 구현(smoke→dev), 이를 successive halving으로 일반화.

### Phase 5 — assurance + 보고 (Layer 8, 9)
목표: claim-evidence ledger + 결정적 보고서 + cross-model 리뷰어 + human gate.
진입 지점:
- 다중 시드 finalist 재현(계약에 seed 목록 추가) + 신뢰구간(paired bootstrap).
- `docs/BLUEPRINT.md`의 claim-evidence ledger 스키마로 `experiments/claims.jsonl`.
  보고서 생성기는 이 ledger 참조로만 숫자 삽입(현재 `cmd_report`를 확장).
- 결정적 그림 생성(불변 로그 산출물에서). LLM은 정성 기준·pairwise·그림 리뷰에만.
- **cross-model 적대 리뷰어**: executor(코더)와 다른 모델 계열로 원고 주장을 raw
  증거·claim ledger에 대조 감사(ARIS). 이 프로젝트의 adversarial review 워크플로
  패턴(아래 §6)을 재사용.
- human gate 노드: novelty 주장·출판·scope 변경·고 compute에 승인 interrupt.

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

- held-out 시드는 로컬 사용자가 루트에서 읽을 수 있다(정책적 보호). 코더는 가드
  훅으로 못 읽지만 진짜 격리는 컨테이너/가상화 샌드박스 필요 → Phase 5 이후.
- TOCTOU(백그라운드 데몬), 워크스페이스 밖 파일쓰기(탐지는 되나 방지는 아님),
  네트워크는 OS 샌드박스 없이 못 막는다.
- ClaudeCoder는 **계정 사용량 한도**에 걸릴 수 있다(SDK 에러). 크래시 복구가
  중단된 세대를 정리하므로 안전하지만, 한도 리셋 후 재개. 문헌 LLM 경로는
  실패 시 lexical로 폴백해 세대를 막지 않는다(`mode` 태그로 기록).
- 문헌 엔진은 mock corpus 전용(실 API 어댑터는 `Retriever` 시임만). 코더 가설의
  개입 계열 분류는 키워드 매칭 — 모호하면 분류를 포기한다(unexplored).
- Phase 4~5는 아직 미구현. claim ledger·cross-model 리뷰어·컨테이너 샌드박스 없음.

## 8. 새 세션 시작 프롬프트 (예시)

> "이 디렉토리(`~/workspace/dev/autoresearch`)의 AutoResearch 시스템에서 Phase 4
> (방향성 브랜치 정제)를 이어서 구현하고 싶어. `docs/HANDOFF.md`와
> `docs/BLUEPRINT.md`를 먼저 읽고, 현재 상태를 파악한 뒤 Phase 4 설계안부터
> 제안해줘."

memory(`autoresearch-project-decisions`)에 구속력 있는 결정이 기록돼 있고 자동
recall되지만, 위 두 docs가 실제 사양의 원천이다.
