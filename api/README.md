# 온점 AI API

FastAPI 기반 AI 서비스 서버입니다.

## 실행 방법

### 로컬 실행

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Docker 실행 (GPU 필요)

```bash
cd api
docker-compose up --build
```

## API 문서

서버 기동 후 → http://localhost:8000/docs

## 엔드포인트

### POST /api/writing/evaluate — 서술형 답안 자동 채점

**Request**
```json
{
  "passage_text": "선인장은 사막에 사는 식물이다.",
  "question_text": "선인장이 사막에서 살 수 있는 이유를 서술하시오.",
  "model_answer": "선인장은 줄기에 물을 저장하고 잎이 가시로 변해 수분 손실을 줄이기 때문이다.",
  "user_answer": "선인장은 물을 저장해서 살 수 있다."
}
```

**Response**
```json
{
  "raw_score": 2,
  "normalized_score": 50,
  "feedback": "답안이 핵심 내용을 일부 포함하고 있으나 잎이 가시로 변하는 이유 등 구체적인 근거가 부족합니다. ..."
}
```

| 필드 | 설명 |
|------|------|
| `raw_score` | 원점수 1~4점 |
| `normalized_score` | 정규화 점수 (25/50/75/100) |
| `feedback` | 개선 방향 피드백 |

### GET /health

```json
{"status": "ok"}
```

## 채점 기준

| 점수 | 기준 |
|------|------|
| 4점 | 핵심 요구사항 완벽히 파악, 전반적 흐름 우수 |
| 3점 | 지시문 이해했으나 근거가 평이하거나 논리 깊이 부족 |
| 2점 | 키워드만 나열, 근거 심각하게 부족 |
| 1점 | 무의미한 반복, 꼼수 명백 |

## 백엔드 연동

백엔드(`AiScoringService`)는 `normalized_score` 값을 점수로 사용합니다.

```java
// AiScoringServiceImpl.java 예시
int score = response.getNormalizedScore();  // 25 / 50 / 75 / 100
```
