# Autoresearch in 2026: SOTA Methods, Architecture, and Implementation Blueprint

> 원본 연구 문서 (사용자가 제공, assessment date 2026-07-16). 이 시스템의 설계
> 근거이자 Phase 3~5 구현의 참조 사양이다. **이 파일은 대화 밖에서 유일하게
> 남는 블루프린트 사본이다** — 새 세션은 여기서 아키텍처를 읽는다.

## Executive assessment

단일한 "SOTA autoresearch system"은 없다. 시스템마다 연구 라이프사이클의 다른
부분을 최적화한다:

- **Arbor, Gome, ERA, SciNav, AlphaEvolve, R&D-Agent** — 실행 가능한 코드/경험적
  산출물 위의 탐색에 집중.
- **AI Scientist v2, AutoResearchClaw, EvoScientist, ARIS** — 더 긴 end-to-end
  연구 워크플로.
- **Co-Scientist** — 가설 생성과 과학적 숙의.
- **Robin** — 가설 생성 + 실험 계획 + 생물 데이터 분석 통합.
- **PaperQA2, Ai2의 과학 검색 시스템** — 인용 근거 기반 문헌 종합.

가장 강한 아키텍처는 이들의 최고 아이디어를 종합한 것:

> **Arbor식 hypothesis-tree refinement(포트폴리오 수준) + Gome식 방향성 진단
> 업데이트(브랜치 내부) + SciNav식 pairwise 랭킹(신뢰 가능한 스칼라 지표가 없을
> 때) + Karpathy/ERA식 불변 평가 + EvoScientist식 성공·실패 메모리 + ARIS식
> claim-to-evidence 감사.**

고성능은 여러 챗 에이전트를 그룹 대화에 넣어서 나오지 않는다. 다음에서 나온다:

1. 신뢰 가능한 평가기.
2. 여러 가설에 대한 명시적 탐색.
3. 전략·구현·평가의 분리.
4. 격리되고 재현 가능한 실행.
5. 지속되지만 압축된 실험 메모리.
6. blind validation과 anti-overfitting 통제.
7. 로그된 증거로 뒷받침되는 주장만 하는 보고서 생성기.

벤치마크는 end-to-end 자율성이 아직 불안정함을 보여준다 (ResearchClawBench 최강
에이전트 21.5/50, AutoResearchBench deep discovery 9.39%, PaperBench 최강
21.0%). 따라서 2026의 올바른 배포 모델은 **전략적 human gate가 있는 제한된 자율
실험**이지, 무제한 자율 출판이 아니다.

---

## 1. SOTA 프레임워크 요약

### 1.1 Arbor — Hypothesis-Tree Refinement (HTR)
장수 coordinator + 단명 executor + 지속적 HTR 구조. 각 트리 노드는 가설 / 산출물
(Git 브랜치) / 관측 / 검증 상태 / 증류된 교훈 / 부모·자식을 묶는다. Executor는
격리된 Git worktree에서 실험 후 종료. 핵심: **지속 탐색 상태(실패 브랜치도 증거로
보존), 단명 워커(컨텍스트 무한 증가 방지), 산출물 계보, insight 역전파,
개발 평가와 admission 평가 분리.** MLE-Bench Lite 86.36% Any Medal (GPT-5.5,
2026-06 preprint). ablation: 트리/insight 제거 시 크게 저하 → 구조화된 메모리와
탐색이 이득의 원천.
→ **HTR을 최상위 제어 구조로 사용.**

### 1.2 Gome — "Reasoning as Gradient"
베이스 모델이 실패 진단을 잘할수록 exhaustive 트리 확장은 비효율. 최적화 개념을
매핑: gradient(무엇을 바꿔야 하는지 구조화된 진단), momentum(성공한 업데이트
방향 기억), distributed optimization(관련 업데이트를 구현하는 독립 추론 여러 개),
learning rate(코드 변경 규모). full MLE-Bench 35.1% any-medal (12h, V100).
핵심: 실험 후 구조화된 "업데이트 벡터"(관측된 실패 → 근본 원인 → 방향성 업데이트
→ 증거)를 뽑아 여러 관련 구현을 안내.
→ **매크로는 HTR, 유망 브랜치 내부는 Gome식 방향성 업데이트.**

### 1.3 ERA — Empirical Research Assistance (Nature)
LLM이 경험적 소프트웨어를 다시 쓰고, 여러 후보 생성, 트리 탐색으로 어느 후보를
더 탐색할지 결정. 문헌의 방법을 결합해 새 실행 가능 솔루션 생성. 9개 중 8개에서
published 결과 능가, 최상위 +14%. 핵심 원칙: **과학적 산출물·평가기·후보 결합
연산을 명시적으로.** (A의 전처리 + B의 목적함수 + C의 옵티마이저 + 새 정규화 등
조합). 스코어링 가능한 계산과학에 강한 템플릿.

### 1.4 SciNav (ICLR 2026)
신뢰 가능한 단일 스칼라 지표가 없을 때: 트리 탐색 + **pairwise 상대 판정** +
top-K 브랜치 선택. "연구 계약·증거·코드 diff·산출물을 볼 때 어느 후보가 더
과학적으로 타당한가? A/B/구분불가/둘다무효" — 무제약 1~10 점수보다 신뢰도 높음.
→ **결정적 평가가 없을 때 pairwise 랭킹 + 하드 유효성 검사 + 다수 blind 심판.**

### 1.5 AI Scientist v2
인간 템플릿 의존 제거. progressive agentic tree search로 아이디어→구현→실험→
분석/시각화→원고→자동 리뷰. VLM 피드백으로 그림 개선. ICLR 워크숍에 자율 논문 3편
제출, 1편이 평균 수용 기준 초과(워크숍 수용률 70%). 교훈: **인간 구조 제거는
일반성↑ 신뢰성↓.** 워크플로 커버리지·문서 생성의 참조로 쓰되 약한 부분은 더
엄격한 증거·평가 서비스로 교체.

### 1.6 AutoResearchClaw
구조화된 멀티에이전트 토론 + self-healing executor + **Pivot/Refine 결정** +
검증 가능한 결과·인용 보고 + 실패 크로스런 메모리. 7가지 human-intervention 모드
(신뢰도 기반 pause 포함). ARC-Bench에서 AI Scientist v2 대비 +54.7%(자체 벤치,
2026-05 preprint). 핵심: **Refine(가설은 여전히 그럴듯 → 구현/설계 수리) vs
Pivot(증거가 가설을 반박 → 과학적으로 다른 브랜치).** 런타임 에러 → 수리. 유효한
부정적 결과 → 과학적 수정/pivot(디버깅 아님).

### 1.7 EvoScientist / ARIS
- **EvoScientist**: 아이디어 메모리(유망 방향, 기각된 아이디어, novelty/feasibility
  교훈)와 실험 메모리(전처리·모델·학습·디버깅·평가 전략)를 **분리** → 코딩 트릭이
  과학 증거로 오인되는 걸 방지.
- **ARIS**: executor와 reviewer가 **다른 모델 계열**. assurance layer =
  integrity 검증 + result-to-claim 매핑 + 원고 주장을 raw 증거·claim ledger에
  대조 감사 + 수학 검사 + 렌더된 논문 시각 검사. 재사용 skill + research-wiki 메모리.
→ **EvoScientist의 메모리 분리 + ARIS의 claim ledger를 채택.**

### 1.8 Co-Scientist (Nature)
가설을 계속 생성·비판·랭킹·진화. 비동기 실행 + tournament로 유망 방향에 test-time
compute 집중. 역할: Generation, Reflection, Ranking, Evolution, Proximity,
Meta-review. scientist-in-the-loop 협력자.
→ **tournament + meta-review를 ideation 프런트엔드에 차용, 독립 실행 시스템과 연결.**

### 1.9 Robin
문헌 검색 + 데이터 분석 에이전트 통합. 외부 실험 루프(wet-lab)를 포함하는 성공적
과학 시스템 예시. 도메인 특화 — 일반 코딩 에이전트 프레임워크로 취급 금지.

### 1.10 R&D-Agent / AIDE / Karpathy / AlphaEvolve
- **R&D-Agent**(MS): Research Agent(아이디어·진단) + Development Agent(구현·실행·
  런타임 에러) 분리. full MLE-Bench 30.22%.
- **AIDE**: 후보를 코드-솔루션 트리로, draft/debug/improve 반복. **한 번에 의미
  있는 변경 하나, 브랜치 상태 요약**(무한 대화 유지 금지).
- **Karpathy autoresearch**: `train.py`만 수정 가능, `prepare.py`·평가는 read-only,
  고정 5분 예산, `val_bpb` 단일 지표, 개선은 유지·회귀는 되돌림. **evaluator-first
  autoresearch의 가장 명확한 실증.** ← 이 프로젝트 Phase 1의 직접 기반.
- **AlphaEvolve**: LLM 변형 + 자동 평가기 + 진화적 선택.

---

## 2. 권장 아키텍처 — 9개 레이어

```
연구 목표 → [1]연구 계약 → [2]문헌·증거 엔진 → 연구질문 인증서
→ [3]가설 포트폴리오 → [4]탐색 매니저/코디네이터
→ 단명 코딩 executor 여러 개 → [6]가시적 개발 평가기
→ [7]실험 원장 → insight·실패 증류 → (코디네이터로 피드백)
→ [6]blind admission gate → 승인 시 best 산출물/main
→ [6]클린 재현·시드·ablation → 최종 미접촉 평가 → [8]claim-evidence ledger
→ 보고서 → [9]적대적 리뷰 + human 승인
```

### Layer 1 — 연구 계약 (구현됨: research_contract.yaml)
타입드·버전드 계약. objective, primary_metric{name,direction,minimum_effect},
secondary_metrics, baseline, editable_globs, protected_globs, budgets,
validation{search/admission/final split}, stop_conditions. executor 안에
read-only 마운트.

### Layer 2 — 증거·문헌 엔진 (Phase 3 대상, 미구현)
단순히 유사 초록 10개 검색이 아니라 **evidence graph** 생성:
1. 연구질문을 개념·메커니즘·방법·데이터셋·결과어로 분해
2. lexical/semantic/citation/author 쿼리 생성
3. 다중 인덱스 검색
4. DOI/PubMed/arXiv 등 안정 ID로 canonicalize
5. 허용 시 full text 검색
6. page/section/table/figure locator와 함께 claim 수준 증거 추출
7. 참조·인용 논문 추적
8. supporting/contradicting/adjacent 식별
9. 최근접 prior claim 기반 novelty 보고(무근거 점수 아님)
10. 임의 검색 횟수가 아니라 증거 커버리지 안정화 시 정지

evidence record 스키마:
```json
{
  "evidence_id": "ev_0142",
  "canonical_paper_id": "doi:10.1234/example",
  "claim": "Method A improves minority-class recall under label imbalance.",
  "stance": "supports",
  "locator": {"section": "4.2", "table": "Table 3", "pages": [7, 8]},
  "population_or_dataset": "Dataset X",
  "conditions": "imbalance ratio >= 10:1",
  "limitations": ["single-site dataset", "no calibration analysis"]
}
```
스택: PaperQA2(로컬/큐레이트 full-text) + Semantic Scholar/OpenAlex/Crossref/
arXiv/PubMed/Europe PMC(discovery) + Ai2 Asta/ScholarQA + pgvector/Qdrant/Vespa/
OpenSearch(벡터) + 그래프/관계 edge 테이블(인용·claim 관계).
**문헌 텍스트가 코드·셸을 직접 실행하지 못하게 격리.**

### Layer 3 — 가설 인증서 (구현됨: Hypothesis certificate)
반증 가능한 가설:
```json
{
  "hypothesis_id": "h_008",
  "statement": "...", "mechanism": "...", "intervention": "...",
  "predicted_observations": {"macro_f1": "no decrease > 0.002", "ece": "-10%"},
  "falsifier": "ECE does not improve across >= 3 seeds.",
  "minimal_decisive_test": "...",
  "supporting_evidence_ids": ["ev_0142"], "nearest_prior_work": ["doi:..."],
  "risk": "low", "estimated_cost": 0.3
}
```

### Layer 4 — 하이브리드 탐색 매니저 (Phase 2에서 포트폴리오 구현)
- 매크로: 과학적으로 구별되는 여러 브랜치 유지 (같은 아이디어의 미세 변형 6개 금지).
- 브랜치 로컬: Gome식 방향성 업데이트(관측→근본원인→방향→bounded 변경), momentum.
- 비스칼라: pairwise top-K, Pareto frontier.
- acquisition: `a(h) = E[ΔM] + β·불확실성 + γ·novelty − λ·cost − ρ·risk`.
- 초기 정책: 초기 가설 8~16, 병렬 브랜치 4~8, 값싼 proxy 평가 먼저, successive
  halving, 60~80% exploit / 20~40% explore, 수리 2~3회, finalist 시드 3~5개.

### Layer 5 — 코딩 executor (Phase 2에서 patcher + ClaudeCoder 구현)
단명·격리(worktree+컨테이너)·가설 1개·평가기/숨은 데이터 변경 불가·main 직접
머지 불가·구조화된 결과 반환. 컨텍스트 패킷: 계약, 가설, repo map, 현재 baseline,
문헌 증거, 증류된 교훈 몇 개, 허용 경로/명령, admission 기준. **받지 말 것**: 숨은
gate 데이터, 최종 test 결과, secret, 전체 raw 대화, 무관 실험, 목표 변경 권한.

### Layer 6 — 평가 서비스 (핵심; Phase 1+2에서 3단계 구현)
- **개발 평가기**: coordinator/executor에 가시, 진단 상세, 탐색용.
- **blind admission gate**: 별도 서비스, 숨은 라벨·전체 로그·예제별 피드백 미노출,
  incumbent보다 일반화 잘 되는지만 선택.
- **최종 미접촉 평가기**: 최종 후보 freeze 후에만.
- 3-way split이 단일 hidden set 반복보다 안전.
- 보호: read-only 마운트, protected 파일 해시(전/후), diff allowlist, 기본 egress
  차단, 고정 CPU/GPU/RAM/wall-clock, pinned 이미지+lockfile, 데이터 해시·계보,
  구조화된 metrics.json(산문 추출 아님), NaN/leakage/중복/degenerate 검사, 커밋
  산출물에서 클린 재현. LLM 심판은 결정적 테스트가 가능하면 주 평가기로 쓰지 말 것.

### Layer 7 — 실험·insight 메모리 (Phase 1+2에서 ledger+insight 구현)
| 메모리 | 내용 |
|---|---|
| Evidence | 논문·claim·모순·인용 |
| Hypothesis | 가설·부모·상태·falsifier |
| Experiment | 커밋·환경·metrics·시드·산출물 |
| Insight | 성공·실패의 증류된 재사용 교훈 |
raw 챗 트랜스크립트를 장기 메모리로 쓰지 말 것. 유효한 실패 실험도 증거.

### Layer 8 — claim-evidence ledger (Phase 5 대상, 미구현)
보고서 작성 전 구축:
```json
{
  "claim_id": "claim_031",
  "text": "The proposed method improves macro-F1 by 1.2 percentage points.",
  "status": "verified",
  "supporting_runs": ["run_220","run_221","run_222"],
  "baseline_runs": ["run_012","run_013","run_014"],
  "effect_size": 0.012, "confidence_interval": [0.007, 0.017],
  "statistical_test": "paired bootstrap",
  "supporting_literature": ["ev_0142"], "limitations": ["single dataset family"]
}
```
보고서 생성기는 이 ledger 참조로만 숫자 삽입 가능. 표·그림은 불변 로그 산출물에서
직접 생성.

### Layer 9 — human gates (Phase 5 대상, 미구현)
연구질문 변경, 데이터 구매/수집, 값비싼 compute 시작, wet-lab/로봇, 생의학·화학·
안전 민감 실험, 공개 릴리스, novelty/임상 주장, 최종 원고 제출에 human 승인.
고레버리지 지점 개입이 완전 자율/완전 마이크로매니지보다 낫다(AutoResearchClaw).

---

## 3. 코딩 에이전트 루프 (Phase 1+2 구현)

```
SELECT 가설 → 현 incumbent에서 격리 worktree 생성 → 하나의 일관된 개입 PLAN
→ PATCH → protected/의존성 정책 VERIFY → LINT/TYPE/UNIT/SMOKE
→ 실행 실패 REPAIR(bounded) → 고정 예산 개발 실험 RUN → 구조화 metrics PARSE
→ CLASSIFY(무효구현 / 유효부정 / 유효불확정 / 유효긍정) → 산출물 COMMIT + provenance
→ 가설 트리에 insight DISTILL → 상위 후보를 blind admission gate로 → 승인만 MERGE
```
**수리 대상**: 문법·import·shape·설정·런타임 예외·산출물 경로. **수리 금지**:
반증된 가설·실패한 통계 결과·novelty 부재·올바르게 실행된 개입의 지표 회귀 → 이건
coordinator로 반환. **원자적 개입**: 실험당 일관된 개입 하나(아키텍처·옵티마이저·
augmentation·loss·batch·lr·calibration 동시 변경 금지).

---

## 4. 실용 스택
- **오케스트레이션**: (문서 권장 LangGraph) — **본 프로젝트는 Claude Agent SDK
  Python으로 구현**(사용자 결정). LangGraph 대신 SDK 런타임 사용.
- **코딩 백엔드**: OpenHands SDK(프로덕션 워커) / mini-swe-agent(최소 감사 가능
  워커) — 본 프로젝트는 ClaudeCoder(Agent SDK) 사용.
- **문헌**: PaperQA2 + Semantic Scholar/OpenAlex/Crossref/arXiv/PubMed + claim
  수준 evidence graph.
- **상태·산출물**: PostgreSQL + pgvector/Qdrant + S3/MinIO + Git + MLflow
  (본 프로젝트는 Phase 1~2에서 로컬 파일/JSONL/git으로 구현; 프로덕션 확장 시 위 스택).
- **실행 인프라**: Docker/gVisor/Kata/Firecracker + K8s/Slurm/Ray + Git worktrees
  + OCI digest + uv/Poetry/Conda/Nix lock + DVC/lakeFS. **egress 기본 차단, 문헌
  검색은 별도 통제 서비스로.**

---

## 6. 평가 전략 (Phase 5에서 자체 지표 계측)
| 능력 | 벤치마크 |
|---|---|
| 문헌 발견 | AutoResearchBench |
| 문헌 이해 | AstaBench |
| 과학 Python 생성 | ScienceAgentBench |
| ML 엔지니어링 | full MLE-Bench |
| 논문 재현 | PaperBench |
| end-to-end 재발견 | ResearchClawBench |
| 자기 도메인 | 과거 프로젝트 기반 private hidden benchmark |
기록할 지표: 유효 실험률, 코드 실행률, 개선률, blind-gate 수용률, 최종 test 개선,
재현률, citation precision/recall, claim-evidence 일관성, 수용 개선당 비용/compute,
human 개입 수, 탐색된 구별되는 가설 수, 올바르게 보존된 부정적 결과 비율,
평가기/데이터 정책 위반율.

---

## 7. 흔한 실패 모드 (설계가 방어하는 것들)
- **Evaluator hacking**: 산출물 개선 없이 점수만. → hidden 평가기, protected 해시,
  별도 프로세스, invariant 검사, 자원 회계, 수동 적대 테스트.
- **개발셋 overfitting**: 같은 dev 데이터에 수백 실험. → blind gate, 미접촉 final
  set, 주기적 refresh, nested validation, 실험수 인지 통계 보정.
- **가설 붕괴**: 모든 브랜치가 같은 접근의 미세 변형. → diversity-aware 선택,
  메커니즘 수준 클러스터링, novelty 항, 탐색 브랜치 최소 할당.
- **컨텍스트 오염**: 장수 코딩 에이전트에 모순 계획 축적. → 단명 executor, 압축
  브랜치 컨텍스트.
- **False repair**: 유효한 부정적 결과를 코딩 실패로 오인해 긍정될 때까지 수정.
  → 실행 유효성과 과학적 결과 구분, 실험 내 가설·평가기 의미 freeze.
- **Citation laundering**: 관련 있지만 주장을 뒷받침 안 하는 논문. → claim 수준
  증거 추출, 정확한 locator, 모순 검색, 독립 인용 검증.
- **Manuscript-first**: 신뢰 결과 전에 설득력 있는 서사부터. → 실험·claim ledger
  커버리지 충족 전 원고 생성 금지.
- **진실 없는 다중 에이전트 합의**: 여러 에이전트가 같은 무근거 가정 강화. → 독립
  증거, 결정적 평가, 이질적 리뷰어 모델, 적대 역할, blind 후보 라벨.

---

## 8. 구현 경로 (Phase 정의 — 이 프로젝트의 로드맵)
- **Phase 1 (완료)**: 제약된 실행 autoresearch. 저장소 1개, 불변 평가기 1개,
  스칼라 지표 1개, 코딩 워커 1개, worktree, 구조화 기록, 고정 예산. Karpathy식
  keep/reject 먼저.
- **Phase 2 (완료)**: 가설 포트폴리오. 타입드 인증서, 병렬 브랜치 4~8, coordinator,
  실패 분류, insight 증류, blind admission 평가. → Arbor형 루프.
- **Phase 3**: 문헌 그라운딩. PaperQA2 또는 동등 검색 서비스, canonical paper ID,
  claim 수준 증거, 인용 그래프 순회, novelty·모순 보고. **문헌 텍스트가 코드·셸
  실행 못하게.**
- **Phase 4**: 방향성 브랜치 정제. Gome식 진단 업데이트, momentum 메모리, SciNav식
  pairwise 선택(정성 산출물), 비용 인지 스케줄링 + successive halving.
- **Phase 5**: 과학적 assurance + 보고. 다중 시드 + 신뢰구간, baseline + ablation,
  claim-evidence ledger, 결정적 그림 생성, cross-model 적대 리뷰어, novelty·출판
  human 승인.

## Final recommendation (원문)
```
Orchestrator: LangGraph + PostgreSQL checkpointing   ← 본 프로젝트: Claude Agent SDK
Research policy: Arbor HTR + Gome branch-local updates + SciNav pairwise
Coding workers: OpenHands SDK / mini-swe-agent        ← 본 프로젝트: ClaudeCoder(SDK)
Literature: PaperQA2 + Semantic Scholar/OpenAlex/... + claim-level evidence graph
Execution: Git worktrees + Docker/gVisor/Firecracker + K8s/Slurm/Ray
Tracking: MLflow + PostgreSQL + S3/MinIO + dataset/OCI hashes
Assurance: immutable dev evaluator + blind admission gate + untouched final test
           + claim-evidence ledger + independent reviewer model
Human control: scope 변경·고 compute·새 데이터·안전 민감·novelty·출판에 승인 gate
```

> **가장 중요한 규칙: 에이전트 스웜 이전에 평가기·provenance 모델·실험 계약을
> 먼저 만든다.** 유효한 진전과 그럴듯한 실패를 구분할 신뢰 가능한 방법 없이는,
> 에이전트와 test-time compute를 더 넣어도 더 비싸고 더 설득력 있어질 뿐 더
> 과학적이 되지 않는다.
