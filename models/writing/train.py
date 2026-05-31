"""
글쓰기 채점 모델 학습 스크립트
Base  : meta-llama/Meta-Llama-3.1-8B-bnb-4bit  (unsloth 최적화 버전)
Method: QLoRA (4-bit NF4) via unsloth + trl SFTTrainer
Hub   : Onjeom/writing-ai
"""

import json
import re
from dataclasses import dataclass, field

from datasets import Dataset
from trl import SFTTrainer, TrainingArguments, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
import torch


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    base_model: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    hub_model_id: str = "Onjeom/writing-ai"
    output_dir: str = "./checkpoints/writing"
    data_path: str = "../../data/writing/train.jsonl"

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0         # unsloth 권장값
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # 학습
    max_seq_length: int = 1536
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    optim: str = "adamw_8bit"
    fp16: bool = True
    logging_steps: int = 20
    save_steps: int = 200
    push_to_hub: bool = True


# ──────────────────────────────────────────────
# 프롬프트 상수
# ──────────────────────────────────────────────

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{response}"
)

# 유연한 채점 기준 (학습 시 instruction으로 사용)
RELAXED_INSTRUCTION = (
    "주어진 지시문과 학생의 답안을 분석하여, 부족한 점과 개선 방향을 포함한 피드백을 작성하고 "
    "맨 마지막에 1점부터 4점 사이의 최종 점수를 부여하시오.\n\n"
    "[유연하고 관대한 채점 기준]\n"
    "- 4점: 지시문의 핵심 요구사항을 잘 파악하였고 전반적인 흐름이 우수한 답안 (사소한 결함은 너그럽게 만점 처리)\n"
    "- 3점: 지시문은 이해했으나 근거가 다소 평이하거나 논리의 깊이가 아쉬운 일반적인 답안\n"
    "- 2점: 지시문의 키워드만 겨우 나열했거나 주장의 근거가 심각하게 부족한 답안\n"
    "- 1점 (최하점): 같은 말을 무의미하게 반복하거나 꼼수가 명백한 답안"
)


def build_full_instruction(question_text: str, model_answer: str) -> str:
    return (
        f"{RELAXED_INSTRUCTION}\n\n"
        f"[문제]\n{question_text}\n\n"
        f"[모범 답안]\n{model_answer}"
    )


def format_sample(sample: dict) -> dict:
    instruction = build_full_instruction(
        sample["question_text"], sample["model_answer"]
    )
    user_answer = sample["student_answer"][:700]   # OOM 방지
    response = f"{sample['feedback']}\n\n[최종 점수: {sample['score']}]"

    text = ALPACA_PROMPT.format(
        instruction=instruction,
        input=user_answer,
        response=response,
    ) + "<|end_of_text|>"

    return {"text": text}


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    cfg = TrainConfig()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
        dtype=None,          # auto-detect
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 데이터 로드
    raw = []
    with open(cfg.data_path, encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line.strip()))

    dataset = Dataset.from_list(raw).map(
        format_sample, remove_columns=Dataset.from_list(raw[:1]).column_names
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        optim=cfg.optim,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        args=training_args,
    )

    trainer.train()

    # HuggingFace Hub 업로드
    model.save_pretrained_merged(
        cfg.output_dir,
        tokenizer,
        save_method="lora",
    )
    if cfg.push_to_hub:
        model.push_to_hub_merged(
            cfg.hub_model_id,
            tokenizer,
            save_method="lora",
        )
        print(f"업로드 완료: https://huggingface.co/{cfg.hub_model_id}")


if __name__ == "__main__":
    main()