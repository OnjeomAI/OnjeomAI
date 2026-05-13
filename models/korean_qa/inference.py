"""
국어 교과 QA 모델 추론

사용법 (단독 실행):
    python models/korean_qa/inference.py --mode interactive
    python models/korean_qa/inference.py --mode eval --val output/korean_qa_v2/val.jsonl --n 100

모듈로 사용:
    from models.korean_qa.inference import KoreanQAModel
    model = KoreanQAModel()
    result = model.predict(input_text)
"""
import os, json, random, re, torch, argparse

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_DIR = os.path.join(BASE_DIR, "output", "korean_qa_v2", "final")
BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"

CHOICE_RE = re.compile(r"[①②③④⑤]")


class KoreanQAModel:
    def __init__(self, adapter_dir=ADAPTER_DIR):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )
        self.model     = PeftModel.from_pretrained(base, adapter_dir)
        self.tokenizer = tokenizer
        self.model.eval()

    def predict(self, input_text: str, max_new_tokens: int = 128) -> dict:
        messages = [
            {"role": "system", "content": "다음 지문을 읽고 문항에 답하시오."},
            {"role": "user",   "content": input_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        choice = CHOICE_RE.search(generated)
        return {
            "raw":    generated,
            "choice": choice.group() if choice else None,
        }


def eval_accuracy(model: KoreanQAModel, val_path: str, n: int = 100, seed: int = 42):
    random.seed(seed)
    with open(val_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]

    n = min(n, len(samples))
    subset = random.sample(samples, n)

    correct = 0
    for i, s in enumerate(subset, 1):
        result = model.predict(s["input"])
        pred   = result["choice"]
        gold   = (CHOICE_RE.search(s["output"]) or type("", (), {"group": lambda self: None})()).group()
        hit    = (pred == gold) if (pred and gold) else False
        correct += hit
        print(f"[{i:3d}/{n}] 정답:{gold}  예측:{pred}  {'O' if hit else 'X'}")
        print(f"  [모델 출력] {result['raw']}")
        print(f"  [정답 해설] {s['output']}")
        print()

    acc = correct / n * 100
    print(f"\n정답률: {correct}/{n} = {acc:.1f}%")
    return acc


def interactive(model: KoreanQAModel):
    print("=== 인터랙티브 테스트 (exit 입력 시 종료) ===")
    print("지문 + 문항 + 선택지 입력 후 빈 줄(엔터 두 번) 입력\n")
    while True:
        lines = []
        while True:
            line = input()
            if line.lower() == "exit":
                return
            if line == "" and lines:
                break
            lines.append(line)
        user_input = "\n".join(lines).strip()
        if not user_input:
            continue
        result = model.predict(user_input)
        print(f"\n[모델 출력]\n{result['raw']}\n{'─'*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["eval", "interactive"], default="interactive")
    parser.add_argument("--val",     default=os.path.join(BASE_DIR, "output", "korean_qa_v2", "val.jsonl"))
    parser.add_argument("--n",       type=int, default=100)
    parser.add_argument("--adapter", default=ADAPTER_DIR)
    args = parser.parse_args()

    qa_model = KoreanQAModel(adapter_dir=args.adapter)

    if args.mode == "eval":
        eval_accuracy(qa_model, val_path=args.val, n=args.n)
    else:
        interactive(qa_model)
