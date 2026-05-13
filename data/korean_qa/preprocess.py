"""
AI Hub 국어 교과 지문형 문제 JSON → JSONL 변환 스크립트

사용법:
    python data/korean_qa/preprocess.py
    python data/korean_qa/preprocess.py --data_dir data/Training/라벨링데이터 --output_dir output/korean_qa_v2
"""
import os, json, random, glob, argparse

CHOICE_SYMBOLS = ["①", "②", "③", "④", "⑤"]


def parse_sample(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    fields = {}
    for item in data.get("learning_data_info", []):
        name  = item["class_name"]
        texts = [ci["text_description"] for ci in item.get("class_info_list", [])]
        fields[name] = "\n".join(texts).strip()

    passage  = fields.get("지문", "")
    question = fields.get("문항", "")
    wrong    = fields.get("오답", "")
    answer   = fields.get("정답", "")
    explain  = fields.get("해설", "")

    if not passage or not question or not answer:
        return None

    choices = (wrong + "\n" + answer).strip() if wrong else answer
    inp = f"[지문]\n{passage}\n\n[문항]\n{question}\n{choices}"
    out = f"정답: {answer}"
    if explain:
        out += f"\n\n해설: {explain}"

    return {"input": inp, "output": out}


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def main(data_dir, output_dir, train_ratio=0.9, seed=42):
    random.seed(seed)

    json_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True))
    print(f"JSON 파일 {len(json_files)}개 발견")

    if not json_files:
        raise FileNotFoundError(f"JSON 파일 없음: {data_dir}")

    samples, errors = [], 0
    for path in json_files:
        s = parse_sample(path)
        if s:
            samples.append(s)
        else:
            errors += 1

    print(f"파싱 완료 — 유효: {len(samples)}개 / 오류: {errors}개")

    random.shuffle(samples)
    split = int(len(samples) * train_ratio)

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path   = os.path.join(output_dir, "val.jsonl")

    save_jsonl(samples[:split], train_path)
    save_jsonl(samples[split:], val_path)

    print(f"저장 완료 — Train: {split}건 ({train_path})")
    print(f"저장 완료 — Val:   {len(samples) - split}건 ({val_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/Training/라벨링데이터")
    parser.add_argument("--output_dir", default="output/korean_qa_v2")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.train_ratio, args.seed)
