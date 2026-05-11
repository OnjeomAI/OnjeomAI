# 국어 교과 지문형 문제 데이터

## 데이터 출처
- **AI Hub**: [국어 교과 지문형 문제 데이터](https://www.aihub.or.kr)
- 저작권 문제로 원천 데이터는 포함하지 않습니다.
- 직접 다운로드 후 `data/Training/라벨링데이터/` 경로에 배치하세요.

## 전처리

```bash
python data/korean_qa/preprocess.py
```

JSON → JSONL 변환 결과는 `output/korean_qa_v2/` 에 저장됩니다.

## 프롬프트 형식

**입력**
```
[지문]
{passage}

[문항]
{question}
① ...
② ...
③ ...
④ ...
⑤ ...
```

**출력**
```
정답: ② 선택지 텍스트

해설: 설명
```
