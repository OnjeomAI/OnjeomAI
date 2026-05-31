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
| 배포 | Kaggle (ngrok) / Docker + GPU 서버 |

## 레포지토리 구조

```
OnjeomAI/
├── README.md
├── .gitignore
├── kaggle_deploy.py               # Kaggle 배포 스크립트
│
├── data/
│   └── korean_qa/
│       ├── preprocess.py          # AI Hub JSON → JSONL 변환 스크립트
│       └── README.md              # 데이터 출처 및 전처리 방법
│
├── models/
│   ├── korean_qa/                 # 국어 교과 QA 모델 (담당: 이성진)
│   │   ├── train.py               # Qwen2.5-3B QLoRA fine-tuning
│   │   └── inference.py           # 단독 추론 스크립트 (서버 미사용)
│   └── writing/                   # 글쓰기 채점 모델 (담당: 김우주)
│       ├── train.py               # Llama-3.1-8B QLoRA fine-tuning (unsloth)
│       └── inference.py           # WritingEvaluator 클래스
│
└── api/                           # AI API 서버 ← 팀원 필독
    ├── README.md                  # 실행 및 테스트 방법
    ├── app/
    │   ├── main.py                # FastAPI 진입점
    │   ├── core/                  # 공통 (모델 로딩 / 환경설정)
    │   │   ├── config.py
    │   │   ├── model.py           # Korean QA 모델 로더 (Qwen2.5-3B)
    │   │   └── model_loader.py    # Writing 모델 로더 (Llama-3.1-8B)
    │   ├── korean_qa/             # 국어 QA 도메인 (담당: 이성진)
    │   │   ├── router.py
    │   │   ├── schemas/
    │   │   └── services/
    │   └── writing/               # 글쓰기 평가 도메인 (담당: 김우주)
    │       ├── router.py
    │       ├── irt_router.py
    │       ├── schemas/
    │       └── services/
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt
```

## AI 서비스 API

| 엔드포인트 | 설명 | 담당 | 상태 |
|---|---|---|---|
| `POST /api/grading/grade` | 키워드 매칭 + LLM 2단계 자동 채점 | 이성진 | 구현 완료 |
| `POST /api/tutor/ask` | RAG 기반 AI 튜터 질문 답변 | 이성진 | 구현 완료 |
| `POST /api/tutor/explain` | 용어·문장 쉬운 설명 | 이성진 | 구현 완료 |
| `POST /api/curriculum/generate` | 진단 결과 기반 커리큘럼 생성 | 이성진 | 구현 완료 |
| `POST /api/problems/generate` | AI 문제 자동 생성 (지문+질문+모범답안) | 이성진 | 구현 완료 |
| `POST /api/indexing/index` | 콘텐츠 벡터 인덱싱 | 이성진 | 구현 완료 |
| `POST /api/writing/evaluate` | 서술형 답안 자동 채점 + 피드백 | 김우주 | 구현 완료 |
| `POST /api/writing/curriculum/adjust` | 취약 역량 기반 동적 학습 경로 재조정 | 김우주 | 구현 완료 |
| `POST /api/writing/compare` | 이전·현재 답변 비교 및 성장 메시지 생성 | 김우주 | 구현 완료 |
| `POST /api/writing/weakness-report` | 역량별 약점 분석 리포트 생성 | 김우주 | 구현 완료 |
| `POST /api/writing/irt/estimate` | IRT 3PL 기반 학생 능력 수준 추정 (1PL 폴백 포함) | 김우주 | 구현 완료 |
| `POST /api/writing/curriculum-plan` | theta 기반 스테이지 결정 + 취약 역량 우선 문제 배치 | 김우주 | 구현 완료 |
| `GET /health` | 서버 상태 확인 | - | 구현 완료 |

자세한 테스트 방법 → [`api/README.md`](./api/README.md)

---

## Kaggle 배포

> GPU T4 x2, 인터넷 연결 ON

**셀 1 — 패키지 설치** (먼저 실행 후 완료 확인)

```python
!pip install -q fastapi uvicorn pyngrok pydantic-settings chromadb sentence-transformers peft accelerate bitsandbytes trl
!pip install -q "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
```

**셀 2 — 서버 실행**

```python
import os
import threading
import subprocess
import time
import requests
from pyngrok import ngrok

os.chdir("/kaggle/working")
os.system("rm -rf /kaggle/working/OnjeomAI")
os.system("git clone -b develop https://github.com/OnjeomAI/OnjeomAI.git /kaggle/working/OnjeomAI")

os.chdir("/kaggle/working/OnjeomAI/api")
os.makedirs("./chroma_db", exist_ok=True)

os.environ.pop("SKIP_MODEL_LOAD", None)
os.environ["BASE_MODEL"]      = "Qwen/Qwen2.5-3B-Instruct"
os.environ["ADAPTER_PATH"]    = "./models/korean_qa"
os.environ["CHROMA_PATH"]     = "./chroma_db"
os.environ["EMBEDDING_MODEL"] = "jhgan/ko-sroberta-multitask"

os.system("fuser -k 8000/tcp 2>/dev/null || true")
time.sleep(2)

def run_server():
    subprocess.run(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"])

threading.Thread(target=run_server, daemon=True).start()

# Qwen2.5-3B + Llama-3.1-8B 두 모델 로드 → 최대 10분 대기
print("모델 로딩 중... (최대 10분)")
for i in range(60):
    time.sleep(10)
    try:
        r = requests.get("http://localhost:8000/health", timeout=3)
        if r.status_code == 200:
            print(f"서버 준비 완료! ({(i + 1) * 10}초)")
            break
    except Exception:
        print(f"대기 중... {(i + 1) * 10}초")

ngrok.set_auth_token("YOUR_NGROK_TOKEN")
tunnel = ngrok.connect(8000)
print("AI 서버 URL:", tunnel.public_url)
```

### POST /api/writing/evaluate

서술형 답안을 받아 1단계 키워드 매칭 → 2단계 LLM 채점 순서로 자동 채점합니다.

**Request Body**

```json
{
  "passage_text": "지문 텍스트",
  "question_text": "문제 지시문",
  "model_answer": "모범 답안",
  "user_answer": "학생 답안 (최대 700자)",
  "keywords": [
    { "keyword": "핵심 키워드", "weight": 30 }
  ]
}
```

> `keywords` 생략 시 LLM 단독 채점으로 동작합니다.

**Response Body**

```json
{
  "keyword_score": 33,
  "raw_score": 2,
  "normalized_score": 50,
  "final_score": 43,
  "feedback": "LLM 채점 피드백 텍스트",
  "feedback_type": "NEEDS_IMPROVEMENT",
  "score_feedback": "이 개념부터 다시 보세요. 관련 학습 콘텐츠를 추천해드릴게요.",
  "matched_keywords": ["줄기"],
  "missing_keywords": ["수분", "가시"],
  "deep_analysis": {
    "error_types": ["내용 누락", "어휘 부족"],
    "analysis": "오류 원인 상세 분석",
    "improvement": "구체적 개선 방향"
  }
}
```

| 응답 필드 | 타입 | 설명 |
|---|---|---|
| `keyword_score` | int \| null | 1단계 키워드 기반 점수 (0~100). keywords 미입력 시 null |
| `raw_score` | int | LLM 원점수 (1~4점) |
| `normalized_score` | int | LLM 정규화 점수 (25 / 50 / 75 / 100) |
| `final_score` | int | **최종 점수** (키워드 40% + LLM 60%, 또는 LLM 단독) |
| `feedback_type` | string | `EXCELLENT`(80~100) / `GOOD`(50~79) / `NEEDS_IMPROVEMENT`(0~49) |
| `score_feedback` | string | 점수 구간별 안내 메시지 |
| `matched_keywords` | string[] | 포함된 키워드 (프론트 초록 하이라이트용) |
| `missing_keywords` | string[] | 누락된 키워드 (프론트 빨강 하이라이트용) |
| `deep_analysis` | object \| null | 오류 유형 분류 + 개선 방향 (final_score < 50 시에만 반환) |

### POST /api/writing/curriculum/adjust

역량별 최근 점수 이력을 분석하여 취약 역량을 탐지하고, 커리큘럼 재조정 메시지를 생성합니다.  
3회 연속 50점 미만인 역량을 취약 역량으로 판정합니다.

**Request Body**

```json
{
  "competency_history": [
    { "competency": "factual", "scores": [75, 80, 90] },
    { "competency": "inferential", "scores": [40, 35, 42] }
  ]
}
```

**Response Body**

```json
{
  "needs_adjustment": true,
  "weak_competencies": ["추론적 독해"],
  "adjustment_message": "추론적 독해 점수가 낮아요. 관련 문제를 먼저 배치했어요.",
  "recommended_focus": "추론적 독해 문제를 집중적으로 연습하는 것을 권장합니다."
}
```

역량 값: `factual` / `inferential` / `critical` / `vocabulary` / `logical`

---

### POST /api/writing/compare

이전 답변과 현재 답변을 비교하여 성장 메시지와 키워드 변화를 반환합니다.

**Request Body**

```json
{
  "question_text": "문제 지시문",
  "model_answer": "모범 답안",
  "previous_answer": "이전 학생 답안",
  "previous_score": 50,
  "current_answer": "현재 학생 답안",
  "current_score": 75,
  "keywords": [
    { "keyword": "광합성", "weight": 50 }
  ]
}
```

**Response Body**

```json
{
  "score_diff": 25,
  "is_improved": true,
  "growth_message": "저번엔 '광합성' 키워드가 없었는데 이번엔 포함됐어요. 성장했어요!",
  "newly_included_keywords": ["광합성"],
  "still_missing_keywords": [],
  "analysis": "이전 답변과 현재 답변 비교 분석 텍스트"
}
```

---

### POST /api/writing/weakness-report

역량별 평균 점수를 분석하여 약점 리포트와 개선 권장사항을 생성합니다.

**Request Body**

```json
{
  "competency_scores": [
    { "competency": "factual", "score": 80 },
    { "competency": "inferential", "score": 42 },
    { "competency": "critical", "score": 65 },
    { "competency": "vocabulary", "score": 55 },
    { "competency": "logical", "score": 38 }
  ]
}
```

**Response Body**

```json
{
  "weak_competencies": [
    { "competency": "추론적 독해", "score": 42, "level": "취약" },
    { "competency": "논리 구조 파악", "score": 38, "level": "취약" },
    { "competency": "비판적 독해", "score": 65, "level": "보통" }
  ],
  "report": "추론적 독해와 논리 구조 파악 영역에서 집중적인 학습이 필요합니다.",
  "recommendations": ["논리 구조 파악 문제를 집중 연습하세요.", "추론적 독해 문제를 집중 연습하세요."],
  "priority_competency": "논리 구조 파악"
}
```

| 응답 필드 | 타입 | 설명 |
|---|---|---|
| `weak_competencies` | object[] | 50점 미만(취약) / 50~69점(보통) 역량 목록 |
| `report` | string | LLM 생성 약점 분석 리포트 (2~3문장) |
| `recommendations` | string[] | 역량별 개선 권장사항 |
| `priority_competency` | string \| null | 가장 집중해야 할 역량 (최저 점수 기준) |

---

### POST /api/writing/irt/estimate

학생의 문제 응답 이력을 바탕으로 **3PL IRT 모델**로 능력 수준을 추정합니다.  
문제별 `a_param` / `b_param` / `c_param`을 전달하면 완전한 3PL로 동작하고, 미전달 시 1PL (Rasch) 폴백으로 동작합니다.

**파라미터 매핑 (1PL 폴백 시)**

| difficulty | b 파라미터 | 설명 |
|---|---|---|
| 1 | -2.0 | 매우 쉬움 |
| 2 | -1.0 | 쉬움 |
| 3 | 0.0 | 보통 |
| 4 | 1.0 | 어려움 |
| 5 | 2.0 | 매우 어려움 |

> 이진화 기준: score ≥ 50 → 정답

**Request Body**

```json
{
  "responses": [
    { "difficulty": 2, "score": 70 },
    { "difficulty": 3, "score": 45, "a_param": 1.5, "b_param": 0.3, "c_param": 0.1 },
    { "difficulty": 4, "score": 30 }
  ]
}
```

| 요청 필드 | 타입 | 설명 |
|---|---|---|
| `difficulty` | int | 문제 난이도 (1~5, 필수) |
| `score` | int | 학생 점수 (0~100, 필수) |
| `a_param` | float \| null | 3PL 변별도 (0.1~4.0, 미입력 시 1.0 고정) |
| `b_param` | float \| null | 3PL 난이도 파라미터 (−4~4, 미입력 시 difficulty 자동 매핑) |
| `c_param` | float \| null | 3PL 추측도 (0.0~0.5, 미입력 시 0.0 고정) |

**Response Body**

```json
{
  "theta": -0.2094,
  "se": 0.7633,
  "ability_level": "중",
  "next_difficulty": 3
}
```

| 응답 필드 | 타입 | 설명 |
|---|---|---|
| `theta` | float | EAP 능력 추정치 (표준화 척도, μ=0 σ=1) |
| `se` | float | 추정 표준오차 (낮을수록 정확) |
| `ability_level` | string | 능력 수준: `하` / `중하` / `중` / `중상` / `상` |
| `next_difficulty` | int | 권장 다음 문제 난이도 (1~5) |

**능력 수준 구간**

| theta 범위 | 능력 수준 | 권장 난이도 |
|---|---|---|
| θ < −1.5 | 하 | 1 |
| −1.5 ≤ θ < −0.5 | 중하 | 2 |
| −0.5 ≤ θ < 0.5 | 중 | 3 |
| 0.5 ≤ θ < 1.5 | 중상 | 4 |
| θ ≥ 1.5 | 상 | 5 |

---

### POST /api/writing/curriculum-plan

IRT 능력 추정치(theta)를 기반으로 스테이지를 결정하고, 취약 역량(50점 미만) reading_type 문제를 우선 배치하여 주간 커리큘럼 플랜을 생성합니다.

**theta → 스테이지 매핑**

| theta 범위 | 스테이지 |
|---|---|
| θ < −0.5 | [1] |
| −0.5 ≤ θ < 0.0 | [1, 2] |
| 0.0 ≤ θ < 0.5 | [2, 3] |
| θ ≥ 0.5 | [3, 4] |

> 스테이지별 배정 문제 수: `daily_goal × 7`

**Request Body**

```json
{
  "theta": -0.21,
  "daily_goal": 3,
  "competency_scores": {
    "FACTUAL": 80,
    "INFERENTIAL": 42,
    "CRITICAL": 65,
    "CREATIVE": 55
  },
  "available_problems": [
    { "id": 101, "difficulty": 2, "reading_type": "INFERENTIAL" },
    { "id": 102, "difficulty": 1, "reading_type": "FACTUAL" }
  ]
}
```

| 요청 필드 | 타입 | 설명 |
|---|---|---|
| `theta` | float | IRT 능력 추정치 |
| `daily_goal` | int | 일일 목표 문제 수 (≥ 1) |
| `competency_scores` | object | 역량별 점수 `{역량명: 점수(0~100)}` |
| `available_problems` | object[] | 배정 가능한 전체 문제 목록 |

**Response Body**

```json
{
  "plan": {
    "1": [101, 105, 108],
    "2": [102, 106, 109]
  }
}
```

| 응답 필드 | 타입 | 설명 |
|---|---|---|
| `plan` | object | 스테이지별 문제 ID 목록 `{stage: [problemId, ...]}` |

---

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

- **원본 규모**: 논술형 16,010건 + 서술형 32,006건 + 주제별 16,001건
- **균형 샘플링**: 가장 적은 주제별(16,001건) 기준으로 세 데이터셋 균등 조정 → 합계 **48,003건**
- **실제 학습 사용**: 점수 레이블 1~4점 (5점 데이터 없음), 점수별 2,000개씩 균등 샘플링
  - 학습 데이터: **8,000건** (점수별 2,000건 × 4)
  - 검증 데이터: **1,000건** (나머지 데이터에서 랜덤 샘플링)
- **라이선스**: AI Hub 이용약관 (재배포 불가, 직접 다운로드 필요)

**논술형 글쓰기 평가 데이터** (원본 소계: 16,010건)  
주제에 대한 논리적 주장·근거 서술 평가

| 학교급/학년 | 수량 |
|---|---|
| 초등학교 5학년 | 3,932 |
| 초등학교 6학년 | 3,151 |
| 중학교 1학년 | 3,856 |
| 중학교 2학년 | 2,173 |
| 중학교 3학년 | 2,898 |

**서술형 글쓰기 평가 데이터** (원본 소계: 32,006건)  
지문 기반 내용 파악 및 서술형 답안 평가

| 학교급/학년 | 수량 |
|---|---|
| 초등학교 5학년 | 7,451 |
| 초등학교 6학년 | 7,096 |
| 중학교 1학년 | 6,111 |
| 중학교 2학년 | 6,042 |
| 중학교 3학년 | 5,306 |

**주제별 글쓰기 평가 데이터** (원본 소계: 16,001건 ← 균형 기준)  
특정 주제에 대한 자유 글쓰기 평가

| 연령대 | 수량 |
|---|---|
| 20~30대 | 7,994 |
| 40대 이상 | 8,007 |

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
| 학습 데이터 | 논술형 + 서술형 + 주제별 (균형 샘플링 48,003건 → 학습 8,000 / 검증 1,000 랜덤) |
| 점수 척도 | 1~4점 (5점 데이터 없음, 인접 정확도 87.5%, Macro F1 0.5348) |
| 학습 환경 | RTX 5070 Ti 16GB |
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
- [x] korean_qa / writing 도메인 폴더 분리
- [x] Kaggle 배포 스크립트 작성 (`kaggle_deploy.py`)
- [ ] Swagger 테스트 완료

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
- [x] 동적 학습 경로 재조정 API 구현 (POST /api/writing/curriculum/adjust)
  - [x] 3회 연속 50점 미만 역량 자동 탐지
  - [x] LLM 기반 재조정 메시지 및 개선 방향 생성
- [x] 답변 변화 추적 API 구현 (POST /api/writing/compare)
  - [x] 이전·현재 답변 키워드 변화 분석
  - [x] LLM 기반 성장 메시지 생성
- [x] 약점 분석 리포트 API 구현 (POST /api/writing/weakness-report)
  - [x] 역량별 취약/보통 수준 분류 (취약 <50, 보통 50~69)
  - [x] 우선 집중 역량 자동 선정
  - [x] LLM 기반 약점 리포트 및 권장사항 생성
- [x] IRT 능력 추정 API 구현 (POST /api/writing/irt/estimate)
  - [x] difficulty(1~5) → b 파라미터(-2~2) 매핑 (1PL 폴백)
  - [x] EAP 알고리즘 (N(0,1) 사전분포 × 161점 theta 격자)
  - [x] 능력 수준 5단계 분류 (하/중하/중/중상/상) 및 권장 난이도 반환
  - [x] 3PL 확장 — per-item a/b/c 파라미터 수신 시 완전한 3PL 모델 적용
- [x] OnjeomBE ↔ AI API IRT 연동
  - [x] `AiIrtService` 인터페이스 + `AiIrtServiceImpl` (RestClient) + `AiIrtServiceMock` 구현
  - [x] `DiagnosticService.calculateTheta()` → IRT API 호출로 교체 (TODO 완료)
  - [x] 진단 완료 시 문제별 a/b/c 파라미터 포함하여 AI API 전달
  - [x] `Problem` 엔티티에 `aParam` / `bParam` / `cParam` nullable 컬럼 추가 (3PL 준비)
- [x] 커리큘럼 플랜 생성 API 구현 (POST /api/writing/curriculum-plan)
  - [x] theta 기반 스테이지 결정 (4구간)
  - [x] 취약 역량(50점 미만) reading_type 문제 우선 배치
  - [x] 스테이지별 `daily_goal × 7` 문제 배정

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
