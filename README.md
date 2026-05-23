# 온점 (Onjeom) - AI 기반 문해력 향상 학습 웹

국어 교과 지문 기반 문제풀이 AI 튜터 시스템

## 프로젝트 개요

문해력 저하 문제를 해결하기 위한 AI 기반 맞춤형 학습 웹입니다.  
국어 교과 지문형 문제 데이터를 활용하여 LLM을 fine-tuning하고, 사용자의 어휘력·독해력·문해력을 분석하여 맞춤형 학습 경험을 제공합니다.

## 팀 정보

- **팀명**: 온점
- **소속**: 컴퓨터공학과
- **개발 기간**: 2026년 3월 ~

## 기술 스택

| 분류 | 기술 |
|------|------|
| LLM (국어 QA) | Qwen2.5-3B-Instruct + QLoRA fine-tuning |
| LLM (글쓰기 채점) | Llama-3.1-8B-Instruct + QLoRA (unsloth) |
| 학습 프레임워크 | unsloth, transformers, peft, trl, bitsandbytes |
| AI 서비스 | FastAPI |
| 벡터 DB | ChromaDB (RAG용) |
| 배포 | AWS EC2 |

## 레포지토리 구조

```
onjeom/
├── README.md
├── .gitignore
│
├── data/
│   └── korean_qa/
│       ├── preprocess.py          # AI Hub JSON → JSONL 변환 스크립트
│       └── README.md              # 데이터 출처 및 전처리 방법
│
├── models/
│   ├── korean_qa/                 # 국어 교과 QA 모델 (담당: 이성진)
│   │   ├── train.py               # Qwen2.5-3B QLoRA fine-tuning
│   │   └── inference.py           # 모델 추론
│   └── writing/                   # 글쓰기 채점 모델 (담당: 김우주)
│       ├── train.py               # Llama-3.1-8B QLoRA fine-tuning (unsloth)
│       └── inference.py           # WritingEvaluator 클래스
│
├── api/                           # AI API 서버 ← 팀원 필독
│   ├── README.md                  # 실행 및 테스트 방법
│   ├── app/
│   │   ├── main.py                # FastAPI 진입점
│   │   ├── core/                  # 모델 로딩 / 환경설정
│   │   ├── routers/
│   │   │   ├── korean_qa.py       # 국어 QA 라우터 (담당: 이성진)
│   │   │   └── writing.py         # 글쓰기 채점 라우터 (담당: 김우주)
│   │   ├── services/
│   │   │   └── writing_service.py # 글쓰기 채점 서비스 (담당: 김우주)
│   │   └── schemas/
│   │       └── writing.py         # 요청/응답 Pydantic 스키마 (담당: 김우주)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
└── notebooks/
    └── korean_qa_train.ipynb      # Colab/Kaggle 학습 노트북
```

## AI 서비스 API

| 엔드포인트 | 설명 | 담당 | 우선순위 |
|---|---|---|---|
| `POST /api/grading/grade` | 키워드 매칭 + LLM 2단계 자동 채점 | 이성진 | 필수 |
| `POST /api/tutor/ask` | RAG 기반 AI 튜터 질문 답변 | 이성진 | 필수 |
| `POST /api/tutor/explain` | 용어/문장 쉬운 설명 | 이성진 | 필수 |
| `POST /api/curriculum/generate` | 진단 결과 기반 커리큘럼 생성 | 이성진 | 필수 |
| `POST /api/indexing/index` | 콘텐츠 벡터 인덱싱 | 이성진 | 필수 |
| `POST /api/writing/evaluate` | 서술형 답안 자동 채점 + 피드백 | 김우주 | 필수 |
| `GET /health` | 서버 상태 확인 | - | - |

자세한 테스트 방법 → [`api/README.md`](./api/README.md)

### POST /api/writing/evaluate 상세

1단계 키워드 매칭 → 2단계 LLM 채점 → 점수 구간별 피드백 → 심층 분석(50점 미만)

| 응답 필드 | 설명 |
|---|---|
| `final_score` | 최종 점수 0~100 (키워드 40% + LLM 60%) |
| `feedback_type` | `EXCELLENT` / `GOOD` / `NEEDS_IMPROVEMENT` |
| `score_feedback` | 점수 구간별 안내 메시지 |
| `matched_keywords` | 포함된 키워드 목록 (프론트 초록 하이라이트용) |
| `missing_keywords` | 누락된 키워드 목록 (프론트 빨강 하이라이트용) |
| `deep_analysis` | 오류 유형 분류 + 개선 방향 (50점 미만 시에만 반환) |

## 모델

| 모델 | 베이스 | 담당 | 허브 |
|---|---|---|---|
| 국어 QA | Qwen2.5-3B-Instruct + QLoRA | 이성진 | [Onjeom/korean_qa](https://huggingface.co/Onjeom/korean_qa) |
| 글쓰기 채점 | Llama-3.1-8B-Instruct + QLoRA (unsloth) | 김우주 | [Onjeom/writing-ai](https://huggingface.co/Onjeom/writing-ai) |

## 데이터셋

### 국어 교과 지문형 문제 데이터 (AI Hub) — 담당: 이성진

- **규모**: 총 10,270 세트 (지문 + 주관식 문항 + 선택지 + 정답 + 해설)
- **라이선스**: AI Hub 이용약관 (재배포 불가, 직접 다운로드 필요)

| 학교급/학년 | 수량 |
|---|---|
| 중학교 1학년 | 3,001 |
| 중학교 2학년 | 2,307 |
| 중학교 3학년 | 2,936 |
| 고등학교 1학년 | 1,501 |
| 고등학교 2학년 | 525 |

### 글쓰기 평가 데이터 (AI Hub) — 담당: 김우주

- **라이선스**: AI Hub 이용약관 (재배포 불가, 직접 다운로드 필요)
- 세 데이터셋을 혼합하여 학습에 활용

| 데이터셋 | 설명 |
|---|---|
| 논술형 글쓰기 평가 데이터 | 주제에 대한 논리적 주장 및 근거 서술 평가 |
| 서술형 글쓰기 평가 데이터 | 지문 기반 내용 파악 및 서술형 답안 평가 |
| 주제별 글쓰기 평가 데이터 | 특정 주제에 대한 자유 글쓰기 평가 |

## 모델 학습 설정

### 국어 QA 모델 (담당: 이성진)

| 항목 | 값 |
|---|---|
| 베이스 모델 | Qwen/Qwen2.5-3B-Instruct |
| 양자화 | 4-bit QLoRA (NF4, double quant) |
| LoRA rank / alpha | 16 / 32 |
| Epochs | 2 |
| Learning rate | 1e-4 (cosine, warmup 5%) |
| Optimizer | paged_adamw_8bit |
| Max length | 1024 |
| 학습 환경 | RTX 4060 Ti 8GB |

### 글쓰기 채점 모델 (담당: 김우주)

| 항목 | 값 |
|---|---|
| 베이스 모델 | meta-llama/Meta-Llama-3.1-8B-Instruct |
| 프레임워크 | unsloth + trl SFTTrainer |
| 양자화 | 4-bit QLoRA (NF4) |
| LoRA rank / alpha | 32 / 32 |
| Epochs | 3 |
| Learning rate | 3e-5 (cosine, warmup 5%) |
| Optimizer | adamw_8bit |
| Max length | 1536 |
| 학습 데이터 | 논술형 + 서술형 + 주제별 글쓰기 평가 데이터 (AI Hub) |
| 점수 척도 | 1~4점 (인접 정확도 87.5%, Macro F1 0.5348) |
| 허브 | [Onjeom/writing-ai](https://huggingface.co/Onjeom/writing-ai) |

## 브랜치 전략

```
main          # 배포용 (직접 푸시 금지)
develop       # 통합 테스트용
feat/*        # 기능 개발
```

## 현재 진행 상황

### 공통
- [x] FastAPI AI 서비스 구조 구축
- [ ] Swagger 테스트 완료
- [ ] AWS EC2 배포

### 국어 QA (담당: 이성진)
- [x] 데이터 수집 및 전처리
- [x] Qwen2.5-3B QLoRA fine-tuning (v1, v2)
- [x] HuggingFace 모델 업로드 (Onjeom/korean_qa)
- [x] 채점 / RAG 튜터 / 커리큘럼 API 구현

### 글쓰기 채점 (담당: 김우주)
- [x] 글쓰기 평가 데이터 수집 (논술형 + 서술형 + 주제별, AI Hub)
- [x] Llama-3.1-8B QLoRA fine-tuning (unsloth)
- [x] HuggingFace 모델 업로드 (Onjeom/writing-ai)
- [x] FastAPI 채점 API 구현 (POST /api/writing/evaluate)
  - [x] 1단계 키워드 매칭 채점
  - [x] 2단계 LLM 채점
  - [x] 점수 구간별 피드백 (80~100 / 50~79 / 0~49)
  - [x] 포함/누락 키워드 응답 (프론트 하이라이트용)
  - [x] 50점 미만 오답 심층 분석 (CoT, 오류 유형 분류)
- [x] 백엔드(OnjeomBE) ↔ AI API 연동 코드 구현

## 참고 자료

**모델**
- [Qwen2.5 모델](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Meta Llama 3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [Unsloth](https://github.com/unslothai/unsloth)

**학습 프레임워크**
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)

**데이터셋 (AI Hub)**
- [국어 교과 지문형 문제 데이터](https://aihub.or.kr)
- [논술형 글쓰기 평가 데이터](https://aihub.or.kr)
- [서술형 글쓰기 평가 데이터](https://aihub.or.kr)
- [주제별 글쓰기 평가 데이터](https://aihub.or.kr)
