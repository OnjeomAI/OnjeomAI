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

---

## 엔드포인트

### POST /api/writing/evaluate — 서술형 답안 자동 채점

2단계 채점 파이프라인 + 점수 구간별 피드백 + 심층 분석을 한 번의 호출로 반환합니다.

**Request**
```json
{
  "passage_text": "선인장은 사막에 사는 식물이다.",
  "question_text": "선인장이 사막에서 살 수 있는 이유를 서술하시오.",
  "model_answer": "선인장은 줄기에 물을 저장하고 잎이 가시로 변해 수분 손실을 줄이기 때문이다.",
  "user_answer": "선인장은 물을 저장해서 살 수 있다.",
  "keywords": [
    { "keyword": "줄기", "weight": 30 },
    { "keyword": "수분", "weight": 40 },
    { "keyword": "가시", "weight": 30 }
  ]
}
```

> `keywords` 생략 시 LLM 단독 채점으로 폴백합니다.

**Response**
```json
{
  "keyword_score": 33,
  "raw_score": 2,
  "normalized_score": 50,
  "final_score": 43,
  "feedback": "물 저장 기능은 언급했으나 잎의 변형과 수분 손실 억제 내용이 누락되었습니다.",
  "feedback_type": "NEEDS_IMPROVEMENT",
  "score_feedback": "이 개념부터 다시 보세요. 관련 학습 콘텐츠를 추천해드릴게요.",
  "matched_keywords": ["줄기"],
  "missing_keywords": ["수분", "가시"],
  "deep_analysis": {
    "error_types": ["내용 누락", "어휘 부족"],
    "analysis": "학생이 줄기의 수분 저장 기능은 언급했으나, 잎이 가시로 변형되어 증산을 억제한다는 핵심 메커니즘이 빠졌습니다.",
    "improvement": "잎이 가시로 변한 이유(증산 억제)와 기공 개폐 방식을 추가로 서술하면 완성도가 높아집니다."
  }
}
```

#### 응답 필드 설명

| 필드 | 설명 |
|---|---|
| `keyword_score` | 1단계 키워드 기반 점수 (0~100). keywords 미입력 시 null |
| `raw_score` | LLM 원점수 (1~4점) |
| `normalized_score` | LLM 정규화 점수 (25/50/75/100) |
| `final_score` | **최종 점수** (키워드 40% + LLM 60% 가중 평균, 또는 LLM 단독) |
| `feedback_type` | `EXCELLENT` / `GOOD` / `NEEDS_IMPROVEMENT` |
| `score_feedback` | 점수 구간별 안내 메시지 |
| `matched_keywords` | 포함된 핵심 키워드 → 프론트에서 초록 하이라이트 |
| `missing_keywords` | 누락된 핵심 키워드 → 프론트에서 빨강 하이라이트 |
| `deep_analysis` | 오답 심층 분석 (final_score < 50 시에만 반환, 그 외 null) |

#### 점수 구간별 피드백

| 구간 | feedback_type | score_feedback |
|---|---|---|
| 80~100점 | `EXCELLENT` | 심화 학습 추천 |
| 50~79점 | `GOOD` | 보완 포인트 제시 |
| 0~49점 | `NEEDS_IMPROVEMENT` | 관련 학습 콘텐츠 안내 |

#### deep_analysis 오류 유형

`final_score < 50` 일 때만 반환됩니다.

| 오류 유형 | 설명 |
|---|---|
| `개념 혼동` | 핵심 개념을 잘못 이해한 경우 |
| `어휘 부족` | 필요한 용어를 사용하지 못한 경우 |
| `논리 비약` | 인과 관계나 근거가 불충분한 경우 |
| `내용 누락` | 필수 내용을 빠뜨린 경우 |

---

### GET /health

```json
{"status": "ok"}
```

---

## 백엔드 연동 예시

```java
// AiScoringServiceImpl.java
WritingEvaluateResponse response = aiApiClient.evaluate(request);
int score = response.getFinalScore();          // 0~100
String feedback = response.getScoreFeedback(); // 점수 구간 메시지
```
