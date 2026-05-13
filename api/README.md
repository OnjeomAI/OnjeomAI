# 온점 AI Service

국어 독해 채점 · AI 튜터 · 커리큘럼 생성 API 서버입니다.

---

## 담당

| 파일 | 담당 |
|---|---|
| `app/routers/korean_qa.py` | 이성진 |
| `app/services/grading_service.py` | 이성진 |
| `app/services/rag_service.py` | 이성진 |
| `app/services/curriculum_service.py` | 이성진 |
| `app/routers/writing.py` | 김우주 |
| `app/services/writing_service.py` | 김우주 |
| `app/main.py`, `app/core/`, `app/schemas/` | 공통 |

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
# 모델 포함 정상 실행
uvicorn app.main:app --reload

# 빠른 재시작 (라우터/스키마 수정 시, 모델 로딩 생략)
SKIP_MODEL_LOAD=1 uvicorn app.main:app --reload
```

처음 실행 시 모델 자동 다운로드 (약 5~10분 소요).  
`모델 로딩 완료!` 메시지가 뜨면 준비된 거예요.

### 3. API 문서 확인

브라우저에서 열기: `http://localhost:8000/docs`

---

## API 테스트

### 공통 방법

1. `http://localhost:8000/docs` 접속
2. 테스트할 엔드포인트 클릭
3. **Try it out** 버튼 클릭
4. 예시 데이터 붙여넣고 **Execute** 클릭

---

### 1. 주관식 자동 채점 (담당: 이성진)

**엔드포인트**: `POST /api/grading/grade`

```json
{
  "passage": "사막은 강수량이 매우 적은 지역으로, 일교차가 크고 식물이 거의 자라지 않는다. 선인장은 두꺼운 줄기에 수분을 저장하여 이런 환경에 적응했다.",
  "question": "선인장이 사막 환경에서 살아남을 수 있는 이유를 서술하시오.",
  "model_answer": "선인장은 두꺼운 줄기에 수분을 저장하는 구조를 가지고 있어 강수량이 적은 사막에서도 생존할 수 있다.",
  "keywords": [
    {"keyword": "수분 저장", "weight": 50},
    {"keyword": "두꺼운 줄기", "weight": 30},
    {"keyword": "사막", "weight": 20}
  ],
  "student_answer": "선인장은 줄기에 물을 저장해서 사막에서 살 수 있다."
}
```

응답: `score`, `stage1_score`, `found_keywords`, `missing_keywords`, `feedback`, `grade_reason`

---

### 2. AI 튜터 질문 (담당: 이성진)

**엔드포인트**: `POST /api/tutor/ask`

```json
{
  "question": "추론적 독해란 무엇인가요?",
  "context": null
}
```

> 콘텐츠 인덱싱 후에는 해당 지문 기반으로 답변합니다.

---

### 3. 용어 설명 (담당: 이성진)

**엔드포인트**: `POST /api/tutor/explain`

```json
{
  "term": "역설법",
  "context": "글쓴이는 역설법을 사용하여 주제를 강조했다."
}
```

---

### 4. 커리큘럼 생성 (담당: 이성진)

**엔드포인트**: `POST /api/curriculum/generate`

```json
{
  "theta": -0.5,
  "daily_goal": 10,
  "weak_areas": ["추론적 이해", "비판적 독해"]
}
```

> `theta`: IRT 능력 추정치 (-3 ~ 3), 낮을수록 초급

---

### 5. 콘텐츠 벡터 인덱싱 (담당: 이성진)

**엔드포인트**: `POST /api/indexing/index`

```json
{
  "content_id": "test-001",
  "passage": "사막은 강수량이 매우 적은 지역으로...",
  "question": "선인장이 사막 환경에서 살아남을 수 있는 이유를 서술하시오.",
  "model_answer": "선인장은 두꺼운 줄기에 수분을 저장...",
  "keywords": ["수분 저장", "두꺼운 줄기", "사막"]
}
```

---

### 6. 헬스 체크

**엔드포인트**: `GET /health`  →  `{"status": "ok"}`

---

## 폴더 구조

```
api/
├── app/
│   ├── main.py                      # FastAPI 진입점 (공통)
│   ├── core/
│   │   ├── config.py                # 환경변수 (공통)
│   │   └── model.py                 # Qwen 모델 로딩 (공통)
│   ├── routers/
│   │   ├── korean_qa.py             # 국어 QA 라우터 (이성진)
│   │   └── writing.py               # 글쓰기 평가 라우터 (김우주)
│   ├── services/
│   │   ├── grading_service.py       # 채점 로직 (이성진)
│   │   ├── rag_service.py           # RAG 파이프라인 (이성진)
│   │   ├── curriculum_service.py    # 커리큘럼 생성 (이성진)
│   │   └── writing_service.py       # 글쓰기 평가 로직 (김우주)
│   └── schemas/                     # 요청/응답 타입 (공통)
├── models/                          # LoRA 어댑터 (gitignore)
├── chroma_db/                       # Vector DB (gitignore)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Docker로 실행

```bash
docker compose up
```

GPU 드라이버 및 nvidia-container-toolkit 설치 필요.
