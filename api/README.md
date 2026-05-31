# 온점 AI Service

국어 독해 채점 · AI 튜터 · 커리큘럼 생성 · 글쓰기 평가 API 서버입니다.

---

## 담당

| 도메인 | 파일 | 담당 |
|--------|------|------|
| korean_qa | `app/korean_qa/` | 이성진 |
| writing | `app/writing/` | 김우주 |
| 공통 | `app/main.py`, `app/core/` | 공통 |

---

## 시작하기

### 1. 환경 설정

```bash
git clone https://github.com/OnjeomAI/OnjeomAI.git
cd api

pip install -r requirements.txt

cp .env.example .env
```

> HuggingFace 모델 자동 다운로드를 위해 로그인 필요 (최초 1회)
> ```bash
> huggingface-cli login
> ```

### 2. 서버 실행

```bash
# 전체 모델 로딩 (GPU 필요, 1~2분 소요)
uvicorn app.main:app --reload

# 모델 없이 빠른 테스트 (키워드 채점만 동작)
SKIP_MODEL_LOAD=1 uvicorn app.main:app --reload

# writing 모델만 mock으로 (korean_qa 모델은 실제 로딩)
MOCK_MODEL=true uvicorn app.main:app --reload
```

`모델 로딩 완료!` 메시지가 뜨면 준비 완료.

### 3. API 문서 확인

브라우저에서 열기: `http://localhost:8000/docs`

---

## API 엔드포인트

### Korean QA (이성진)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/grading/grade` | 주관식 답변 2단계 채점 |
| POST | `/api/tutor/ask` | RAG 기반 AI 튜터 질문 |
| POST | `/api/tutor/explain` | 용어/문장 쉬운 설명 |
| POST | `/api/curriculum/generate` | 맞춤형 커리큘럼 생성 |
| POST | `/api/problems/generate` | AI 문제 자동 생성 |
| POST | `/api/indexing/index` | 벡터 임베딩 인덱싱 |

### Writing (김우주)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/writing/evaluate` | 서술형 답안 자동 채점 |
| POST | `/api/writing/curriculum/adjust` | 동적 학습 경로 재조정 |
| POST | `/api/writing/compare` | 이전·현재 답변 변화 추적 |
| POST | `/api/writing/weakness-report` | 역량별 약점 분석 리포트 |
| POST | `/api/writing/curriculum-plan` | theta 기반 커리큘럼 플랜 |
| POST | `/api/writing/explain-term` | 용어/문장 쉬운 설명 |
| POST | `/api/writing/irt/estimate` | IRT 3PL 학생 능력 추정 |

### 공통

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 헬스체크 |

---

## 폴더 구조

```
api/
├── app/
│   ├── main.py                  # FastAPI 진입점
│   ├── core/
│   │   ├── config.py            # 환경변수 설정
│   │   ├── model.py             # Korean QA 모델 로더 (Qwen2.5-3B)
│   │   └── model_loader.py      # Writing 모델 로더 (Llama-3.1-8B)
│   ├── korean_qa/               # 이성진 담당
│   │   ├── router.py
│   │   ├── schemas/
│   │   │   ├── grading.py
│   │   │   ├── tutor.py
│   │   │   ├── curriculum.py
│   │   │   ├── indexing.py
│   │   │   └── problem.py
│   │   └── services/
│   │       ├── grading_service.py
│   │       ├── rag_service.py
│   │       ├── curriculum_service.py
│   │       └── problem_service.py
│   └── writing/                 # 김우주 담당
│       ├── router.py
│       ├── irt_router.py
│       ├── schemas/
│       │   ├── writing.py
│       │   └── irt.py
│       └── services/
│           ├── writing_service.py
│           └── irt_service.py
├── models/                      # LoRA 어댑터 (gitignore)
├── chroma_db/                   # Vector DB (gitignore)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Kaggle 배포

```python
import os, subprocess
from pyngrok import ngrok

# 1. 레포 클론
os.system("git clone -b develop https://github.com/OnjeomAI/OnjeomAI.git /kaggle/working/OnjeomAI")
os.makedirs("/kaggle/working/OnjeomAI/api/chroma_db", exist_ok=True)
os.chdir("/kaggle/working/OnjeomAI/api")

# 2. 패키지 설치
os.system("pip install -r requirements.txt -q")

# 3. ngrok 터널
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
tunnel = ngrok.connect(8000)
print("AI 서버 URL:", tunnel.public_url)

# 4. 서버 실행
subprocess.Popen(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"])
```

---

## Docker 실행

```bash
docker compose up
```

GPU 드라이버 및 nvidia-container-toolkit 설치 필요.
