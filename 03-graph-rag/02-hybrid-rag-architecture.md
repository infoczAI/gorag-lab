# Hybrid RAG: Vector + Graph의 역할 분담

Vector RAG의 한계를 인식했다고 해서  
Vector를 버릴 필요는 없습니다.

GoRAG Lab에서는 Vector와 Graph를  
**대체 관계가 아니라 역할 분담 관계**로 봅니다.

이 문서에서는 Hybrid RAG 구조를 설명합니다.

---

## Hybrid RAG의 기본 전제

Hybrid RAG의 핵심 전제는 단순합니다.

- Vector는 “후보를 넓게 찾는 역할”
- Graph는 “의미적으로 좁히는 역할”

즉:
- Recall은 Vector
- Precision과 Consistency는 Graph

가 담당합니다.

---

## Hybrid RAG의 전형적인 흐름

일반적인 흐름은 다음과 같습니다.

1. 자연어 질의 입력
2. Vector 검색으로 관련 문서/엔티티 후보 확보
3. 후보를 Knowledge Graph의 개념에 매핑
4. Ontology 제약으로 유효성 검증
5. Graph 탐색으로 관련 맥락 확장
6. LLM을 통한 요약 및 설명

이 구조에서 Graph는:
> 검색 결과의 **의미적 필터이자 확장기** 역할을 합니다.

---

## 왜 Graph가 중간에 필요한가

Graph는 다음을 가능하게 합니다.

- 문서 간 관계를 따라가는 탐색
- 규칙과 예외의 적용
- “같은 질문, 같은 답”을 보장하는 기준
- 중간 추론 경로 확보

즉 Graph는:
> RAG 시스템의 **일관성과 설명성**을 담당합니다.

---

## Hybrid 구조에서의 실패 패턴

다음은 Hybrid RAG에서 흔한 실패입니다.

- Graph를 단순 캐시처럼 쓰는 경우
- Ontology 없이 그래프만 있는 경우
- Vector 결과를 그대로 LLM에 던지는 경우

이 경우 Hybrid RAG는:
> Vector RAG보다 복잡하지만  
> 더 나은 결과를 주지 못합니다.

---

## Hybrid RAG의 성공 조건

GoRAG Lab에서 중요하게 보는 조건은 다음입니다.

- Ontology가 Graph의 기준으로 작동하는가
- Graph 탐색이 질의 목적에 맞게 설계되었는가
- LLM이 “답 생성”이 아니라 “설명 요약”에 집중하는가

Hybrid RAG의 핵심은:
> 구성 요소의 수가 아니라  
> **역할 분담의 명확성**입니다.

---

## 요약

- Vector와 Graph는 경쟁 관계가 아니다
- Vector는 후보 탐색, Graph는 의미 통제
- Hybrid RAG는 구조가 명확할수록 강해진다

다음 문서에서는  
Graph 기반 탐색과 Agent 방식의 차이를 다룹니다.
