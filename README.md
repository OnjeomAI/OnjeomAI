# OnjeomAI

> 온점의 AI 모델 레포지토리 — 문해력 채점 및 피드백 생성

---

## 프로젝트 소개

온점 AI는 사용자의 글쓰기 답변을 분석하여 **채점 평가**와 **맞춤형 피드백**을 자동으로 생성하는 모델입니다.  
Llama 3.1을 기반으로 fine-tuning하여 국어 교과 도메인에 특화된 글쓰기 평가 시스템을 구현합니다.

---

## 모델 구조

### 기반 모델 : Llama 3.1

표준 디코더 전용 트랜스포머 구조를 기반으로 하며, 단일 모델 구조를 채택해 학습 안정성을 높였습니다.  
SFT(Supervised Fine-Tuning)와 DPO(Direct Preference Optimization)를 반복 적용하여 성능을 점진적으로 향상시켰습니다.

| Task | 모델 | 방식 |
|:----:|:----:|------|
| 채점 평가 | Llama 3.1 | 인코더 + 분류 헤더 |
| 피드백 생성 | Llama 3.1 | 디코더 생성 |

**Input** : 질문, 답변, 평가지표  
**Output** : 평가 점수 (채점) / 피드백 텍스트 (생성)

---

## 학습 데이터

> AI Hub에서 회원가입 후 직접 다운로드해주세요.

| 데이터 | 건수 | Train | Validation | Test | 링크 |
|--------|-----:|------:|-----------:|-----:|:----:|
| 국어 교과 지문형 문제 | - | - | - | - | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM010&aihubDataSe=data&dataSetSn=71857) |
| 논술형 글쓰기 평가 | 20,012건 | 80% | 10% | 10% | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM010&aihubDataSe=data&dataSetSn=71819) |
| 서술형 글쓰기 평가 | 40,006건 | 80% | 10% | 10% | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM010&aihubDataSe=data&dataSetSn=71818) |
| 주제별 글쓰기 평가 | 20,003건 | 80% | 10% | 10% | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM010&aihubDataSe=data&dataSetSn=71817) |
| 학술논문 이해 데이터 | 추후 추가 예정 | - | - | - | - |

다운로드 후 `data/` 폴더에 넣어주세요.

---

## 모델 성능

| 데이터 | Task | 지표 | 목표 | 결과 |
|--------|------|------|:----:|:----:|
| 논술형 | 채점 평가 | F1 Score | 0.6 이상 | **0.95** ✅ |
| 논술형 | 피드백 생성 | Perplexity | 10% 이상 감소 | **25.86% 감소** ✅ |
| 서술형 | 채점 평가 | F1 Score | 0.6 이상 | **0.94** ✅ |
| 서술형 | 피드백 생성 | Perplexity | 10% 이상 감소 | **24.08% 감소** ✅ |
| 주제별 | 채점 평가 | F1 Score | 0.6 이상 | **0.95** ✅ |
| 주제별 | 피드백 생성 | Perplexity | 10% 이상 감소 | **19.67% 감소** ✅ |

---

## 학습 환경

| 항목 | 사양 |
|------|------|
| CPU | Intel® Xeon® Gold 6226R |
| Memory | 256GB |
| GPU | NVIDIA A100 80GB PCIe × 2 |
| OS | Ubuntu 20.04 |
| Python | 3.8.10 |
| Framework | CUDA 12.2, PyTorch 2.5.1 |
| 학습 알고리즘 | Transformer |
| 학습 조건 | epoch=60, batch_size=1, optimizer=AdamW |

---

## 설치 방법

```bash
git clone https://github.com/OnjeomAI/OnjeomAI.git
cd OnjeomAI
pip install -r requirements.txt
```

---

## 기술 스택

![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12.2-76B900?style=flat-square)

---

<div align="center">
  <sub>온점 · 영남대학교 컴퓨터공학과 · 2026</sub>
</div>