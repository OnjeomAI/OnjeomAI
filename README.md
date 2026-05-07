# 온점 (Onjeom) - AI 기반 문해력 향상 학습 웹

> 국어 교과 지문 기반 문제풀이 AI 튜터 시스템

---

## 프로젝트 개요

문해력 저하 문제를 해결하기 위한 AI 기반 맞춤형 학습 웹입니다.  
국어 교과 지문형 문제 데이터를 활용하여 LLM을 fine-tuning하고, 사용자의 어휘력·독해력·문해력을 분석하여 맞춤형 학습 경험을 제공합니다.

---

## 팀 정보

- **팀명**: 온점
- **소속**: 컴퓨터공학과
- **개발 기간**: 2026년 3월 ~

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| LLM (국어 QA) | Qwen2.5-3B-Instruct + LoRA fine-tuning |
| LLM (글쓰기 평가) | Llama 3.1 + fine-tuning |
| 학습 프레임워크 | transformers, peft, trl, bitsandbytes |
| 데이터 처리 | Python, HuggingFace Datasets |
| 백엔드 | FastAPI |
| 벡터 DB | ChromaDB / FAISS (RAG용) |
| 프론트엔드 | 미정 |

---

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
│   │   ├── train.py               # Qwen2.5-3B LoRA fine-tuning
│   │   └── inference.py           # 모델 추론
│   └── writing/                   # 글쓰기 평가 모델 (팀원 담당)
│       ├── train.py               # Llama3.1 fine-tuning
│       └── inference.py
│
├── api/
│   ├── app.py                     # FastAPI 메인
│   ├── korean_qa_router.py        # 국어 QA 라우터
│   └── writing_router.py          # 글쓰기 평가 라우터
│
└── notebooks/
    └── korean_qa_train.ipynb      # Colab/Kaggle 학습 노트북
```

---

## 데이터셋

### 국어 교과 지문형 문제 데이터 (AI Hub)

- **출처**: [AI Hub - 국어 교과 지문형 문제 데이터](https://aihub.or.kr)
- **라이선스**: 저작권 구매 데이터 (재배포 불가)
- **규모**: 총 10,270 세트 (문항 + 지문 + 정답/오답 + 해설)
- **구성**:

| 학교급/학년 | 수량 | 비율 |
|------------|------|------|
| 중학교 1학년 | 3,001 | 29.2% |
| 중학교 2학년 | 2,307 | 22.5% |
| 중학교 3학년 | 2,936 | 28.6% |
| 고등학교 1학년 | 1,501 | 14.6% |
| 고등학교 2학년 | 525 | 5.1% |
| **합계** | **10,270** | **100%** |

- **난이도 분포**: 상(3.8%) / 중(26.7%) / 하(69.6%)
- **포맷**: 원천데이터(PNG) + 라벨링데이터(JSON)

### 데이터 전처리 결과

```
전체 JSON 파일: 17,453개 (Training + Validation)
→ Training: 16,426건
→ Validation: 1,027건
```

### 데이터 사용 방법

1. [AI Hub](https://aihub.or.kr)에서 직접 다운로드
2. 전처리 스크립트 실행:

```bash
python data/korean_qa/preprocess.py \
  --base_path "다운로드경로/data" \
  --save_path "data/korean_qa/output"
```

---

## 모델 학습

### 국어 QA 모델 (Qwen2.5-3B-Instruct + LoRA)

**fine-tuning 데이터 포맷**:
```json
{
  "instruction": "다음 지문을 읽고 문항에 답하시오.",
  "input": "[지문]\n...\n\n[문항]\n...",
  "output": "정답: ②\n\n해설: ...",
  "metadata": {
    "난이도": "하",
    "학교급": "중학교",
    "학년": "1학년"
  }
}
```

**학습 설정**:
```
베이스 모델: Qwen/Qwen2.5-3B-Instruct
양자화: 4bit (NF4, double quant)
LoRA rank: 16
LoRA alpha: 32
Target modules: q_proj, v_proj, k_proj, o_proj
학습 epoch: 5
배치 사이즈: 1 (gradient accumulation 8)
학습률: 2e-4
최적화: adamw_torch
학습 환경: RTX 4060 Ti (VRAM 16GB)
```

**학습 실행**:
```bash
pip install transformers trl peft accelerate bitsandbytes
python models/korean_qa/train.py
```

---

## AI 기능 명세

| 기능 | 설명 | 우선순위 |
|------|------|---------|
| RAG 기반 AI 튜터 | 국어 독해 자료 기반 질문 답변 (Vector DB + LLM) | 필수 |
| LLM 자동 채점 | 키워드 기반 1차 채점 + LLM 문맥 분석 2차 채점 (0~100점) | 필수 |
| 점수 구간별 피드백 | 80~100/50~79/0~49점 구간별 맞춤 피드백 생성 | 우대 |
| 오답 심층 분석 | Chain-of-Thought로 오류 유형 분류 및 재설명 | 우대 |
| 동적 학습 경로 재조정 | 연속 저득점 시 커리큘럼 자동 재배열 | 선택 |
| 답변 변화 추적 | 재풀이 시 이전 답변과 비교하여 성장 메시지 생성 | 우대 |
| 내 답변 vs 모범답안 비교 | 키워드 하이라이트로 누락/포함 시각화 | 우대 |

---

## API 구조

```
POST /qa/answer          # 문제 풀이 요청
POST /qa/grade           # 주관식 채점 요청
POST /qa/feedback        # 피드백 생성
POST /writing/evaluate   # 글쓰기 평가 (팀원 담당)
GET  /user/progress      # 학습 현황 조회
```

---

## 브랜치 전략

```
main                  # 배포용 (직접 푸시 금지)
dev                   # 통합 테스트용
feature/korean-qa     # 국어 QA 모델 작업 (이성진)
feature/writing       # 글쓰기 평가 모델 (팀원)
```

---

## 현재 진행 상황

- [x] 데이터 수집 (AI Hub 국어 교과 지문형 문제 데이터)
- [x] 데이터 전처리 (JSON → JSONL 변환, Train/Val 분리)
- [x] 베이스 모델 선정 (Qwen2.5-3B-Instruct)
- [x] LoRA fine-tuning 코드 작성
- [ ] 로컬 GPU (RTX 4060 Ti)에서 학습 완료
- [ ] 모델 성능 평가
- [ ] RAG 파이프라인 구축
- [ ] 자동 채점 시스템 구현

---

## 참고 자료

- [AI Hub - 국어 교과 지문형 문제 데이터](https://aihub.or.kr)
- [Qwen2.5 모델](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [TRL - Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- 한옥영. (2023). 생성형 AI 기반 학습자 맞춤형 교육 시스템 설계. 컴퓨터교육학회 논문지, 26(6)
