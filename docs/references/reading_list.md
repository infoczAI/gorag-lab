# GoRAG Lab Reading List
> 우리의 관점(semantic layer + constraints + explainability + agent orchestration)에 가까운 레퍼런스만 모았습니다.
> 각 항목은 "왜 읽는가 / 우리 시스템 어디에 쓰는가" 기준으로 정리합니다.

---

## How to use this list
- **Why it matters**: gorag-lab 문서(01~05장) 중 어디를 강화하는지
- **Takeaway (1-liner)**: 핵심 한 줄
- **Use in our system**: 우리가 실제로 적용/응용할 지점
- **Details**: 논문/코드/블로그의 상세 내용

---

# 01) Ontology as a semantic layer (01_ontology)

## [Paper] OG-RAG: Ontology-Grounded Retrieval-Augmented Generation for LLMs (EMNLP 2025 / arXiv 2024)
- Link: [https://aclanthology.org/2025.emnlp-main.1674/](https://aclanthology.org/2025.emnlp-main.1674/)
- Why it matters: "Ontology는 문서가 아니라 기준면"이라는 관점을 retrieval 설계를 통해 증명한다.
- Takeaway (1-liner): "유사 문서 top-k" 대신, ontology-grounded knowledge unit의 최소 근거 집합을 구성한다.
- Use in our system:
  - (03_graph-rag) retrieval을 "개념 기반 컨텍스트 구성"으로 재정의할 때 참고
  - (01_ontology) TBox/ABox 분리 + grounding의 실질적 효과를 설명할 때 인용

??? info "Details (펼쳐보기)"
    ### 문제 정의
    LLM이 의료, 법률, 농업 등 특화 도메인 지식에 적응하지 못함. 구조화된 도메인 지식을 활용하지 못하는 문제.

    ### 핵심 아이디어
    검색을 "유사 문서 top-k" 대신 **온톨로지에 기반한 지식 단위의 최소 근거 집합**으로 재정의.

    ### 방법론
    1. 도메인 문서를 **하이퍼그래프**로 표현
    2. 각 하이퍼엣지가 온톨로지로 기반지워진 사실 지식 클러스터 캡슐화
    3. 쿼리에 대해 **최소한의 하이퍼엣지 집합**을 검색

    ### 실험 결과
    | 지표 | 개선율 |
    |------|--------|
    | 정확 사실 회수율 | 55%↑ |
    | 응답 정확성 | 40%↑ |
    | 속성 귀속 속도 | 30%↑ |
    | 사실 기반 추론 정확도 | 27%↑ |

---

## [Code] microsoft/ograg2 (reference implementation)
- Link: [https://github.com/microsoft/ograg2](https://github.com/microsoft/ograg2)
- Why it matters: "ontology-grounded retrieval"이 실제 파이프라인에서 어떤 artifact를 만들며 어디서 비용이 드는지 알 수 있다.
- Takeaway (1-liner): "grounding → hypergraph → minimal context"의 구현 포인트를 볼 수 있다.
- Use in our system:
  - concept-unit(팩트/규칙) 단위 컨텍스트 구성 / pruning 전략 비교

??? info "Details (펼쳐보기)"
    ### 목적
    OG-RAG 논문의 참조 구현. 도메인 특화 온톨로지로 LLM을 강화하여 사실 정확성 개선.

    ### 4대 핵심 기능
    1. **온톨로지 기반 검색** (Ontology-Grounded Retrieval)
    2. **하이퍼그래프 컨텍스트 표현** (Hypergraph Context Representation)
    3. **최적화 검색 알고리즘** (Optimized Context Retrieval Algorithm)
    4. **사실성 강화** (Enhanced Factual Accuracy)

    ### 모듈 구조
    ```
    ograg2/
    ├── knowledge_graph/     # 트리플 기반 지식 표현
    ├── ontology_mapping/    # 도메인 온톨로지-문서 연결
    ├── query_engine/        # RAG 쿼리 처리
    └── configs/             # API/모델/경로 설정
    ```

    ### 3단계 워크플로우
    1. `build_knowledge_graph.py` → 온톨로지 매핑 및 지식그래프 생성
    2. `query_llm.py` → LLM 쿼리 실행
    3. `test_answers.py` → 평가 메트릭(정확성, 충실성, 관련성 등) 검증

---

## [Blog] The Semantic Layer: A Reliable Map of the Enterprise Data Landscape (Graphwise, 2025)
- Link: [https://graphwise.ai/blog/the-semantic-layer-a-reliable-map-of-the-enterprise-data-landscape/](https://graphwise.ai/blog/the-semantic-layer-a-reliable-map-of-the-enterprise-data-landscape/)
- Why it matters: semantic layer를 (1) domain knowledge model + (2) enterprise KG로 나누어 보는 관점이다. (T-Box, A-Box에 대응됨)
- Takeaway (1-liner): "Semantic layer = domain model + enterprise KG"로 정의한 설계의 예시이다.
- Use in our system:
  - (02_knowledge-graph) KG를 "저장소"가 아니라 "의미 계층"으로 설명
  - (05_case-studies) ERP/업무 시스템 분석 사례의 설득력 강화

??? info "Details (펼쳐보기)"
    ### 핵심 정의
    Semantic Layer = 기업 데이터 환경의 **신뢰할 수 있는 지도**. 데이터와 비즈니스 앱 간의 중개 계층.

    ### 2가지 구성 요소
    | 구성 요소 | 대응 개념 | 역할 |
    |-----------|-----------|------|
    | Domain Knowledge Model | T-Box | 개념/관계의 스키마 정의 |
    | Enterprise KG | A-Box | 실제 인스턴스 데이터 |

    ### 주요 역할
    - 복잡한 데이터 구조를 이해하기 쉬운 형태로 변환
    - AI 및 시맨틱 검색 솔루션 구축 기반
    - LLM 성능 향상을 위한 컨텍스트 제공

---

# 02) Constraints & verification (Ontology is not a PDF) (01_ontology ↔ 02_knowledge-graph ↔ 04_agents)

## [Paper] xpSHACL: Explainable SHACL Validation using RAG and LLMs (VLDB 2025 workshop / arXiv 2025)
- Link: [https://arxiv.org/abs/2507.08432](https://arxiv.org/abs/2507.08432)
- Why it matters: 제약 검증을 "운영 가능한 품질장치"로 만들고, 위반을 사람이 이해 가능한 설명으로 변환하였다.
- Takeaway (1-liner): SHACL 위반을 "보고서 한 줄"이 아니라 "행동 가능한 설명"으로 만든다.
- Use in our system:
  - (04_agents) Verifier Agent의 핵심 기능 정의(= 제약 위반 설명/수정 가이드)
  - (01_ontology) "OWL+SHACL 병행"의 설득 근거

??? info "Details (펼쳐보기)"
    ### xpSHACL이란?
    SHACL 제약 위반에 대해 **상세하고, 다국어 지원, 인간이 읽을 수 있는 설명**을 생성하는 시스템.

    ### 문제 정의
    기존 SHACL 검증 보고서는 단순 영문 메시지로, 비전문가가 이해하기 어려움.

    ### 기술 구성 (3가지 요소 통합)
    ```
    ┌─────────────────────────────────────────────────┐
    │  규칙 기반 정당화 트리 (Justification Tree)      │
    │  → 위반의 논리적 근거 구조화                     │
    └─────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────┐
    │  RAG (Retrieval-Augmented Generation)           │
    │  → 관련 정보 검색으로 문맥 제공                  │
    └─────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────┐
    │  LLM                                            │
    │  → 자연스러운 설명문 생성                        │
    └─────────────────────────────────────────────────┘
    ```

    ### 핵심 혁신: Violation KG
    위반 지식 그래프 도입 → 설명을 캐싱/재사용하여 효율성·일관성 향상.

---

## [PDF] xpSHACL workshop PDF (VLDB LLM+Graph 2025)
- Link: [https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/LLM%2BGraph/LLMGraph-6.pdf](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/LLM%2BGraph/LLMGraph-6.pdf)
- Why it matters: xpSHACL의 핵심 아이디어를 빠르게 파악 가능(짧고 요약적).
- Takeaway (1-liner): "justification tree + RAG" 조합이 explainable constraint validation의 현실 해법.
- Use in our system:
  - PoC 단계에서 ontology의 중요성을 빠르게 증명할 때

??? info "Details (펼쳐보기)"
    ### 핵심 아이디어
    Justification Tree + RAG 조합으로 설명 가능한 제약 검증 구현.

    ### 접근 방식
    1. SHACL 위반 발생 시 **정당화 트리** 구성
    2. 트리의 각 노드에 대해 **RAG로 관련 컨텍스트** 검색
    3. LLM이 구조화된 정보를 바탕으로 **인간 친화적 설명** 생성

    > 상세 내용은 arXiv 논문 참조

---

## [Paper] G-SPEC: Graph-Symbolic Policy Enforcement and Control (arXiv 2025)
- Link: [https://arxiv.org/abs/2512.20275](https://arxiv.org/abs/2512.20275)
- Why it matters: Agentic 시스템의 위험을 "확률적 플래닝 + 결정적 검증(KG+SHACL)"로 분리해서 다룬다.
- Takeaway (1-liner): LLM의 plan/act는 불가피하게 흔들리므로, graph constraint로 '통과/차단'을 제어할 수 있어야 한다.
- Use in our system:
  - (04_agents) Guardrail/Verifier의 위치를 명확히(LLM plan 밖에서 deterministic gate)
  - "안전/컴플라이언스/거버넌스" 요구가 있는 고객 커뮤니케이션 근거

??? info "Details (펼쳐보기)"
    ### G-SPEC이란?
    **Graph-Symbolic Policy Enforcement and Control**. 5G/6G 네트워크 자율 운영을 위한 신경-기호 학습 프레임워크.

    ### 핵심 문제
    LLM 에이전트가 의도 기반 네트워킹을 제공하지만, **위상 환각(hallucination)과 정책 미준수**라는 확률적 위험 도입.

    ### 3가지 기둥
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                    결정론적 검증 계층                         │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │  NKG (Network Knowledge Graph)                          │ │
    │  │  → 네트워크 토폴로지/상태의 ground truth                  │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │                           ↕                                  │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │  TSLAM-4B (확률적 에이전트)                              │ │
    │  │  → 텔레콤 최적화 의사결정                                 │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │                           ↕                                  │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │  SHACL 제약조건                                          │ │
    │  │  → 정책 검증 규칙 (88개 정책)                             │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────────┘
    ```

    ### 아키텍처 핵심
    확률적 에이전트(TSLAM-4B)를 **결정론적 NKG와 SHACL 기반 정책 검증** 사이에 샌드위치.

    ### 성과
    | 지표 | 결과 |
    |------|------|
    | Safety Violations | Zero |
    | Remediation 성공률 | 94.1% (기준선 82.4%) |
    | 확장성 | O(k^1.2) (10K~100K 노드) |
    | 처리 오버헤드 | 142ms |

    ### 기여도 분석
    - 안전 향상의 **68%**: NKG 검증에서 파생
    - 안전 향상의 **24%**: SHACL 정책에서 파생

---

## [Code] NetoAI/G-SPEC-Framework (subset of SHACL policies 포함)
- Link: [https://github.com/NetoAI/G-SPEC-Framework](https://github.com/NetoAI/G-SPEC-Framework)
- Why it matters: "정책을 SHACL로 표현한다"가 실제로 어떤 형태가 되는지 예시와 함께 볼 수 있다.
- Takeaway (1-liner): 정책이 코드가 아니라 "graph traversal constraint"로 정리된다.
- Use in our system:
  - 도메인별 정책/제약을 룰 엔진처럼 운영하는 방안 비교

??? info "Details (펼쳐보기)"
    ### 목적
    G-SPEC 논문의 참조 구현. 5G 자율 네트워크에서 AI 에이전트의 안전성 보장을 위한 정책 검증 체계.

    ### SHACL 정책
    - 논문의 88개 정책 중 **대표적 subset** 포함
    - **4가지 제약 클래스** 모두 커버
    - 벤더 특정 독점 규칙은 제외

    ### 핵심 구조
    ```
    G-SPEC-Framework/
    ├── ontology/
    │   └── policies/        # SHACL 정책 저장
    ├── experiments/
    │   └── run_benchmark.py # 실험 재현
    └── requirements.txt
    ```

    ### 실행 환경
    - Python 3.9+
    - Neo4j 데이터베이스 (localhost:7687)
    - Apache 2.0 라이선스

---

# 03) Retrieval transparency & neuro-symbolic RAG (03_graph-rag ↔ 02_knowledge-graph)

## [Paper] Neurosymbolic Retrievers for Retrieval-Augmented Generation (arXiv 2026-01)
- Link: [https://arxiv.org/abs/2601.04568](https://arxiv.org/abs/2601.04568)
- Why it matters: "retriever가 불투명하면 디버깅/신뢰가 무너진다"는 문제의식이 동일하다.
- Takeaway (1-liner): retriever에 KG-path / symbolic feature를 섞어 "왜 이 문서를 골랐는지"를 설명한다.
- Use in our system:
  - (03_graph-rag) retrieval 단계에도 graph를 넣는 이유(설명/재현성)
  - "vector-only의 한계"를 단순 정확도 말고 투명성/디버깅 관점으로 강화

??? info "Details (펼쳐보기)"
    ### 문제 정의
    RAG 시스템의 검색기, 재순위기, 생성기가 **불투명**. 왜 이 문서를 골랐는지 설명 불가.

    ### 핵심 질문
    1. 검색기가 문서 선택을 명확히 설명할 수 있는가?
    2. 상징적 지식이 검색 투명성을 높일 수 있는가?

    ### 3가지 방법론

    #### 1. MAR (지식 변조 정렬 검색)
    모듈레이션 네트워크로 **해석 가능한 상징적 특성**으로 쿼리 임베딩 정제.

    #### 2. KG-Path RAG
    지식그래프 탐색으로 검색 품질과 해석 가능성 향상.
    ```
    Query → KG Path Traversal → Explainable Document Selection
    ```

    #### 3. Process Knowledge-infused RAG
    도메인별 도구로 검증된 워크플로우에 따라 검색 콘텐츠 재정렬.

    ### 검증
    정신건강 위험 평가 작업에서 **투명성과 성능 모두 향상**.

---

# 04) Agents & workflow orchestration (04_agents)

## [Doc/Code] LangGraph
- Link: [https://github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- Why it matters: "탐색은 실행이 아니라 의사결정 흐름"이라는 우리의 워크플로우 관점을 구현하기 좋은 primitive 제공.
- Takeaway (1-liner): stateful workflow에서 분기/반복/검증을 구조로 표현한다.
- Use in our system:
  - (04_agents) Planner / Explorer / Verifier / Composer 역할 분리
  - failure path(재시도/대체 경로)를 설계 자산으로 만들기

??? info "Details (펼쳐보기)"
    ### 정의
    장시간 실행되는 **상태 유지 에이전트**를 구축·관리·배포하는 저수준 오케스트레이션 프레임워크.

    ### 5대 핵심 기능

    | 기능 | 설명 |
    |------|------|
    | **Durable Execution** | 실패 시 정확한 중단점에서 자동 재개 |
    | **Human-in-the-Loop** | 실행 중 상태 검사/수정 가능 |
    | **포괄적 메모리 관리** | 단기 작업 메모리 + 세션 간 장기 저장소 |
    | **LangSmith 디버깅** | 실행 경로 추적, 상태 전이 캡처 |
    | **프로덕션 배포** | 확장 가능 인프라 제공 |

    ### 아키텍처
    ```python
    from langgraph.graph import StateGraph
    from typing import TypedDict

    class State(TypedDict):
        messages: list
        context: dict

    graph = StateGraph(State)
    graph.add_node("planner", planner_fn)
    graph.add_node("executor", executor_fn)
    graph.add_node("verifier", verifier_fn)
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", should_verify, {...})
    ```

    ### 설계 영감
    - Google Pregel
    - Apache Beam

    ### 생태계
    LangChain, LangSmith, LangGraph Studio와 통합.

---

# 05) Enterprise / ERP-like problems (KG is not just Graph RAG) (05_case-studies)

## [Blog] Graph Analytics in the Semantic Layer: Architectural Framework for Knowledge Intelligence (Enterprise Knowledge)
- Link: [https://enterprise-knowledge.com/graph-analytics-in-the-semantic-layer-architectural-framework-for-knowledge-intelligence/](https://enterprise-knowledge.com/graph-analytics-in-the-semantic-layer-architectural-framework-for-knowledge-intelligence/)
- Why it matters: "semantic layer → explainable, business-ready insight" 프레임을 엔터프라이즈 관점에서 설명한다.
- Takeaway (1-liner): 그래프는 단순 연결이 아니라 "의미 기반 분석 엔진"으로 semantic layer의 핵심이 된다.
- Use in our system:
  - (05_case-studies) ERP/업무 시스템 케이스에서 "왜 KG인가"를 비기술자에게 설득

??? info "Details (펼쳐보기)"
    ### 3대 아키텍처 요소

    #### 1. 메타데이터 그래프
    - 데이터 계보, 소유권, 보안 분류 추적
    - 예: 은행에서 고객 데이터가 수십 개 시스템을 거쳐 흐르는 과정 추적

    #### 2. 지식 그래프
    - 고객, 거래 등 실제 개체/관계를 온톨로지로 인코딩
    - 예: 소매업체에서 "고위험 고객" 정의를 위해 데이터 소스 간 관계 매핑

    #### 3. 분석 그래프
    - 속성 그래프(LPG)로 패턴·이상 탐지
    - 예: 금융기관의 사기 탐지 - SQL로는 놓치는 의심 패턴 발견

    ### 비즈니스 사례

    | 조직 | 활용 | 효과 |
    |------|------|------|
    | 글로벌 투자회사 | 지식 그래프로 M&A 실사 | 시간 단축 |
    | 보험 규제 기관 | 링크 분석 | 정교한 사기 적발 |
    | 정부 기관 | 그래프 기반 사건 분석 | 해결 시간 30% 단축 |

---

## [Blog] Ontologies, Context Graphs, and Semantic Layers: What AI Actually Needs in 2026 (Metadata Weekly)
- Link: [https://metadataweekly.substack.com/p/ontologies-context-graphs-and-semantic](https://metadataweekly.substack.com/p/ontologies-context-graphs-and-semantic)
- Why it matters: "semantic layer는 lookup용, ontology는 context+reasoning용"으로 선을 그어줌(우리 주장과 매우 유사).
- Takeaway (1-liner): BI용 semantic layer만으로는 AI가 필요로 하는 '의미/행동'이 안 나온다.
- Use in our system:
  - (01_ontology) 왜 ontology가 필요한가를 '현업 데이터 스택' 언어로 번역
  - (03_graph-rag) RAG가 대체 못 하는 지점을 명확히

??? info "Details (펼쳐보기)"
    ### Semantic Layer vs Ontology

    | 구분 | Semantic Layer | Ontology |
    |------|----------------|----------|
    | 출처 | BI 산업 | 지식 표현 분야 |
    | 초점 | 메트릭 정의의 일관성 | 개념 간 관계와 의미 추론 |
    | 예시 | `revenue = SUM(order_total)` | Gene Ontology, SNOMED CT |
    | 도구 | dbt MetricFlow (YAML) | OWL, SHACL |

    > **"의미는 측정과 동일하지 않다"**

    ### 핵심 통찰
    AI 시스템은 **대시보드를 소비하지 않음**. 개념, 관계, 추론 능력이 필요.
    → Semantic layer만으로는 AI가 도메인을 추론하도록 돕는 데 부족.

    ### 패러다임 전환
    - "메트릭-우선 패러다임이 끝나고 있다"
    - 의료/생명과학이 20년 전부터 온톨로지에 투자한 이유: **지식 표현이 비즈니스 크리티컬**했기 때문

    ### Context Graphs
    온톨로지의 진화 형태. 단순 데이터를 넘어 **의사결정 근거** 기록.
    - 예: "왜 할인이 승인되었는가?"를 추적

    ### 실무적 함의
    - YAML 메트릭 정의 ≠ 형식 온톨로지 엔지니어링
    - 진정한 지식 아키텍처 구축에는 도메인 전문성 + 형식 로직 필요
    - **AI가 이 구분을 강제하고 있음** → 지식 표현에 투자하는 조직이 경쟁 우위 확보

---

## [Blog] Knowledge Graph vs Graph Database: Key Differences (PuppyGraph)
- Link: [https://www.puppygraph.com/blog/knowledge-graph-vs-graph-database](https://www.puppygraph.com/blog/knowledge-graph-vs-graph-database)
- Why it matters: 외부 공유에서 가장 흔한 오해("그래프DB=KG?")를 깔끔하게 분리.
- Takeaway (1-liner): KG는 semantic layer(모델/의미), graph DB는 storage/query layer(저장/질의).
- Use in our system:
  - README/온보딩에서 개념 정리용(논쟁 방지)

??? info "Details (펼쳐보기)"
    ### 핵심 차이점

    | 구분 | Knowledge Graph | Graph Database |
    |------|-----------------|----------------|
    | **본질** | 의미론적 계층 (semantic layer) | 스토리지/쿼리 엔진 |
    | **초점** | 엔티티, 관계, **의미** 인코딩 | 구조적 연결 |
    | **추론** | 새로운 관계 발견 가능 | 명시적 데이터만 쿼리 |
    | **스키마** | 온톨로지 기반 | 스키마-온-리드 (유연) |
    | **쿼리 언어** | SPARQL, GraphQL | Cypher, Gremlin |
    | **우선순위** | 정확성, 의미의 정확성 | 처리량, 성능 |

    ### 사용 사례 비교

    #### Knowledge Graph 적합
    - GraphRAG, AI 애플리케이션
    - 데이터 통합/거버넌스
    - 복잡한 추론이 필요한 분석
    - 감사 추적성 필수 영역

    #### Graph Database 적합
    - 경로 분석, 패턴 매칭
    - 실시간 성능 필요 애플리케이션
    - 사기 탐지, 추천 엔진
    - 빠르게 변하는 데이터 구조

    ### 함께 사용하기
    두 기술은 **보완적**:
    - Knowledge Graph → 의미 제공
    - Graph Database → 빠른 쿼리 성능

    → 맥락과 성능을 모두 갖춘 시스템 구성 가능

---

# Appendix) Suggested reading paths

## Onboarding (1 week)
1) Graphwise semantic layer 글
2) OG-RAG (paper or overview)
3) Metadata Weekly 글
4) PuppyGraph "KG vs Graph DB"
5) xpSHACL (요약 PDF)

## Implementation (2–4 weeks)
1) LangGraph repo
2) og rag2 code
3) xpSHACL arXiv
4) Neurosymbolic Retrievers arXiv
5) G-SPEC paper/code (Verifier/guardrail 설계용)

---

# Notes (what we care about)
- We prefer references that treat ontology as:
  - (a) grounding surface for retrieval
  - (b) constraints for verification
  - (c) a source of explainability and governance
- We treat Graph RAG as:
  - an interface pattern on top of knowledge graph reasoning
  - not the definition of a knowledge graph system
