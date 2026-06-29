# 매뉴얼 음성 도우미 🎙️

> 가전·전자기기 사용법 음성 RAG 시스템

가전·전자기기 사용 중 생긴 질문을 **음성으로 물으면**, 제품 매뉴얼을 검색해
**정확한 답을 음성으로** 안내하는 핸즈프리 RAG 시스템입니다.

---

## 핵심 차별점

매뉴얼은 *"이 질문의 정답은 몇 페이지"* 가 명확히 정해지는 도메인입니다.
이 특성을 활용해 **검색 품질을 정량적으로 측정·개선**하는 것이 이 프로젝트의 핵심입니다.

| 구분 | 일반 매뉴얼 챗봇 | 본 프로젝트 |
|------|----------------|------------|
| 검색 방식 | 키워드 / 단순 임베딩 | **Hybrid(BM25+임베딩) + Reranker** |
| 품질 검증 | 정성 평가 위주 | **Recall@k · MRR 정량 측정** |
| 개선 과정 | 단일 구성 | **3단계 비교로 개선 입증** |
| 약점 분석 | 없음 | **질문유형별 실패 클러스터링** |
| 사용 경험 | 텍스트 입력 | **음성 입출력 핸즈프리** |

---

## 시스템 아키텍처

### 실시간 추론 파이프라인
```
모바일 웹앱 → STT(음성→텍스트) → 모델 식별 → RAG 검색 코어 → TTS(텍스트→음성) → 음성·텍스트 응답
                                              │
                          검색기(BM25+임베딩+Reranker) → 벡터 DB → LLM 답변생성(출처 표시)
```

### 데이터 파이프라인
- **① 오프라인 인덱싱 (사전 1회)** — 매뉴얼 PDF → 파싱·청킹(페이지·섹션 메타데이터) → 한국어 임베딩 → 벡터 DB 저장
- **② 온라인 추론 (질문마다)** — 질문 임베딩 → Top-k 검색·재순위 → 컨텍스트 구성 → 근거 기반 답변 생성

### RAG 3단계 (점진적 고도화)
| 단계 | 구성 | 핵심 |
|------|------|------|
| **Naive** | 단순 임베딩 Top-k 검색 | 베이스라인 |
| **Hybrid + Reranker** | BM25 + 임베딩 결합 후 재순위 | 키워드·의미 결합으로 정확도 향상 |
| **Agentic** | 모델 식별·재질의·다단계 검색 | 복합 질문 대응, 검색 실패 시 재시도 |

---

## 디렉토리 구조

```
manualgo/
├─ backend/          FastAPI 서버 (API + STT→RAG→TTS 오케스트레이션)
│  ├─ api/           라우터 (음성 질의, 헬스체크)
│  ├─ schemas/       Pydantic 요청/응답 모델
│  └─ services/      파이프라인 오케스트레이터
├─ rag/              RAG 검색 코어 (라이브러리)
│  ├─ indexing/      오프라인 인덱싱: PDF 파싱·청킹·임베딩·인덱스 구축
│  ├─ retrieval/     검색기 3단계 (naive / hybrid+reranker / agentic)
│  ├─ generation/    근거 기반 LLM 답변 생성 (출처 표시)
│  ├─ vectorstore/   벡터 DB 래퍼 (chroma / faiss)
│  └─ pipeline.py    온라인 추론 파이프라인
├─ speech/           STT (Whisper/CLOVA) · TTS (CLOVA/XTTS)
├─ evaluation/       오프라인 평가 모듈 (포트폴리오 핵심)
│  ├─ generate_evalset.py    평가셋 자동 생성 (LLM)
│  ├─ metrics.py             Recall@k · MRR
│  ├─ run_eval.py            단계별 성능 비교
│  ├─ failure_clustering.py  질문유형별 실패 클러스터링
│  └─ results/               평가 결과 산출물
├─ frontend/         모바일 웹앱 (마이크·답변·음성재생)
├─ data/
│  ├─ raw/manuals/   수집한 매뉴얼 PDF (원본)
│  ├─ processed/chunks/  파싱·청킹 결과
│  ├─ eval/          평가셋 (질문 ↔ 정답 페이지/섹션)
│  └─ index/         벡터 DB / BM25 인덱스
├─ notebooks/        탐색·평가 노트북
├─ scripts/          유틸 스크립트 (매뉴얼 수집 등)
└─ tests/            테스트
```

---

## 기술 스택

| 영역 | 후보 기술 |
|------|----------|
| 프론트엔드 | 모바일 웹앱 (반응형) / 필요 시 PWA |
| STT | Whisper (로컬 GPU) 또는 CLOVA Speech |
| 임베딩 | 한국어 특화 임베딩 모델 (예: BGE-M3) |
| 검색 | BM25 + 벡터 검색 + Reranker |
| LLM | API 또는 로컬 모델 |
| TTS | CLOVA Voice 또는 XTTS 계열 |
| 문서 처리 | PDF 파싱 · 청킹 · 인덱싱 파이프라인 |

---

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env        # 값 채우기

# 3. 매뉴얼 PDF를 data/raw/manuals/ 에 넣고 인덱스 구축
python -m rag.indexing.build_index

# 4. (평가) 평가셋 생성 후 단계별 성능 측정
python -m evaluation.generate_evalset
python -m evaluation.run_eval

# 5. 백엔드 서버 실행
uvicorn backend.main:app --reload
```

---

## 개발 일정 (2개월)

| 주차 | 주요 작업 | 산출물 |
|------|----------|--------|
| W1–2 | 매뉴얼 수집 · PDF 파싱 · 청킹/인덱싱 · Naive RAG | 검색 가능한 벡터 DB, 베이스라인 |
| W3 | 평가셋 자동 생성 · Recall@k/MRR 측정 환경 | 평가 노트북, 베이스라인 점수 |
| W4–5 | Hybrid + Reranker · 단계별 성능 비교 | 개선된 검색기, 비교 결과 |
| W6 | 실패 클러스터링 · 질문유형별 약점 분석 | 실패 분석 리포트 |
| W7 | STT/TTS 연동 · 모바일 웹앱 UI | 동작하는 음성 웹앱 |
| W8 | Agentic RAG · 통합 · 데모 정리 · 문서화 | 최종 데모, 포폴 정리 |

자세한 기획 내용은 [docs/planning.md](docs/planning.md) 참고.
