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
| LLM (글쓰기 평가) | Llama 3.1 + fine-tuning |
| 학습 프레임워크 | transformers, peft, trl, bitsandbytes |
| AI 서비스 | FastAPI |
| 벡터 DB | ChromaDB (RAG용) |
| 배포 | AWS EC2 |
| 프론트엔드 | 미정 |

## 레포지토리 구조

```
onjeom/
├── README.md
├── .gitignore
│
├── ai-service/                    # AI API 서버 (담당: 이성진)
│   ├── README.md                  # 실행 및 테스트 방법 ← 팀원 필독
│   ├── app/
│   │   ├── main.py                # FastAPI 진입점
│   │   ├── core/                  # 모델 로딩 / 환경설정
│   │   ├── api/                   # 엔드포인트 라우터
│   │   ├── services/              # 채점·RAG·커리큘럼 로직
│   │   └── schemas/               # 요청/응답 타입
│   ├── models/                    # LoRA 어댑터 (gitignore됨)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
└── datasets/
    └── 29.국어 교과 지문형 문제 데이터/
        ├── train_v2.py            # Qwen2.5-3B QLoRA 학습 스크립트
        └── test_model.py          # 모델 추론 테스트
```

## AI 서비스 API

| 엔드포인트 | 설명 | 우선순위 |
|---|---|---|
| `POST /api/grading/grade` | 키워드 매칭 + LLM 2단계 자동 채점 | 필수 |
| `POST /api/tutor/ask` | RAG 기반 AI 튜터 질문 답변 | 필수 |
| `POST /api/tutor/explain` | 용어/문장 쉬운 설명 | 필수 |
| `POST /api/curriculum/generate` | 진단 결과 기반 커리큘럼 생성 | 필수 |
| `POST /api/indexing/index` | 콘텐츠 벡터 인덱싱 | 필수 |
| `GET /health` | 서버 상태 확인 | - |

자세한 테스트 방법 → [`ai-service/README.md`](./ai-service/README.md)

## 모델

| 모델 | 베이스 | 허브 |
|---|---|---|
| 국어 QA | Qwen2.5-3B-Instruct + QLoRA | [Onjeom/korean_qa](https://huggingface.co/Onjeom/korean_qa) |
| 글쓰기 평가 | Llama 3.1 (예정) | - |

## 데이터셋

**국어 교과 지문형 문제 데이터 (AI Hub)**

- **규모**: 총 10,270 세트 (지문 + 주관식 문항 + 선택지 + 정답 + 해설)
- **라이선스**: AI Hub 이용약관 (재배포 불가, 직접 다운로드 필요)

| 학교급/학년 | 수량 |
|---|---|
| 중학교 1학년 | 3,001 |
| 중학교 2학년 | 2,307 |
| 중학교 3학년 | 2,936 |
| 고등학교 1학년 | 1,501 |
| 고등학교 2학년 | 525 |

## 모델 학습 설정

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

## 브랜치 전략

```
main          # 배포용 (직접 푸시 금지)
develop       # 통합 테스트용
feature/*     # 기능 개발
```

## 현재 진행 상황

- [x] 데이터 수집 및 전처리
- [x] Qwen2.5-3B QLoRA fine-tuning (v1, v2)
- [x] HuggingFace 모델 업로드 (Onjeom/korean_qa)
- [x] FastAPI AI 서비스 구조 구축
- [x] 채점 / RAG 튜터 / 커리큘럼 API 구현
- [ ] Swagger 테스트 완료
- [ ] AWS EC2 배포
- [ ] 글쓰기 평가 모델 (팀원 담당)

## 참고 자료

- [AI Hub - 국어 교과 지문형 문제 데이터](https://aihub.or.kr)
- [Qwen2.5 모델](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
