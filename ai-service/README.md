# 온점 AI Service

국어 독해 채점 · AI 튜터 · 커리큘럼 생성 API 서버입니다.

---

## 시작하기

### 1. 환경 설정

```bash
git clone [repo 주소]
cd ai-service

pip install -r requirements.txt

cp .env.example .env
```

> HuggingFace 모델 자동 다운로드를 위해 로그인 필요 (최초 1회)
> ```bash
> huggingface-cli login
> ```

### 2. 서버 실행

```bash
uvicorn app.main:app --reload
```

처음 실행 시 모델을 자동으로 다운로드합니다 (약 5~10분 소요).  
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

### 1. 주관식 자동 채점

**엔드포인트**: `POST /api/grading/grade`

**예시 요청**:
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

**확인할 것**:
- `score`: 최종 점수 (0~100)
- `stage1_score`: 키워드 매칭 1단계 점수
- `found_keywords`: 포함된 키워드 목록
- `missing_keywords`: 누락된 키워드 목록
- `feedback`: AI 피드백 메시지
- `grade_reason`: 채점 근거

---

### 2. AI 튜터 질문

> 콘텐츠가 인덱싱되어 있어야 참고 자료 기반 답변이 가능합니다.  
> 인덱싱 전에도 일반 답변은 동작합니다.

**엔드포인트**: `POST /api/tutor/ask`

**예시 요청**:
```json
{
  "question": "추론적 독해란 무엇인가요?",
  "context": null
}
```

**확인할 것**:
- `answer`: AI 튜터 답변
- `sources`: 참조한 콘텐츠 ID 목록

---

### 3. 용어 설명

**엔드포인트**: `POST /api/tutor/explain`

**예시 요청**:
```json
{
  "term": "역설법",
  "context": "글쓴이는 역설법을 사용하여 주제를 강조했다."
}
```

**확인할 것**:
- `explanation`: 쉬운 말로 풀어쓴 설명

---

### 4. 커리큘럼 생성

**엔드포인트**: `POST /api/curriculum/generate`

**예시 요청**:
```json
{
  "theta": -0.5,
  "daily_goal": 10,
  "weak_areas": ["추론적 이해", "비판적 독해"]
}
```

> `theta`: IRT 능력 추정치 (-3 ~ 3)  
> 낮을수록 초급, 높을수록 고급

**확인할 것**:
- `total_days`: 전체 학습 일수
- `summary`: 커리큘럼 요약
- `plans`: 일별 학습 계획 목록

---

### 5. 콘텐츠 벡터 인덱싱

> 관리자가 문제를 등록할 때 호출하는 API입니다.  
> 인덱싱 완료 후 AI 튜터 검색에 반영됩니다.

**엔드포인트**: `POST /api/indexing/index`

**예시 요청**:
```json
{
  "content_id": "test-001",
  "passage": "사막은 강수량이 매우 적은 지역으로, 일교차가 크고 식물이 거의 자라지 않는다. 선인장은 두꺼운 줄기에 수분을 저장하여 이런 환경에 적응했다.",
  "question": "선인장이 사막 환경에서 살아남을 수 있는 이유를 서술하시오.",
  "model_answer": "선인장은 두꺼운 줄기에 수분을 저장하는 구조를 가지고 있어 강수량이 적은 사막에서도 생존할 수 있다.",
  "keywords": ["수분 저장", "두꺼운 줄기", "사막"]
}
```

**확인할 것**:
- `status`: `"indexing"` (백그라운드 처리)
- `content_id`: 인덱싱된 콘텐츠 ID

인덱싱 후 `/api/tutor/ask`에서 같은 주제로 질문하면 해당 지문을 참고해서 답변합니다.

---

### 6. 헬스 체크

**엔드포인트**: `GET /health`

```json
{"status": "ok"}
```

서버 정상 동작 여부 확인용입니다.

---

## 프로젝트 구조

```
ai-service/
├── app/
│   ├── main.py              # FastAPI 진입점
│   ├── core/
│   │   ├── config.py        # 환경변수 (.env)
│   │   └── model.py         # Qwen 모델 로딩
│   ├── api/                 # 엔드포인트 라우터
│   ├── services/            # 비즈니스 로직
│   └── schemas/             # 요청/응답 타입
├── models/korean_qa/        # LoRA 어댑터 (gitignore됨)
├── chroma_db/               # Vector DB 데이터 (gitignore됨)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Docker로 실행 (선택)

```bash
docker compose up
```

GPU 드라이버 및 nvidia-container-toolkit 설치 필요.
