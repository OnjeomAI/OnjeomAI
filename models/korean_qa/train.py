"""
국어 교과 QA 모델 — Qwen2.5-3B QLoRA fine-tuning (v2)

사용법:
    python models/korean_qa/train.py

사전 조건:
    - data/korean_qa/preprocess.py 실행 완료
    - output/korean_qa_v2/train.jsonl, val.jsonl 존재
"""
import os, json, torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD_ADAPTER = os.path.join(BASE_DIR, "output", "korean_qa", "final")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output", "korean_qa_v2")
FINAL_DIR   = os.path.join(OUTPUT_DIR, "final")
TRAIN_JSONL = os.path.join(OUTPUT_DIR, "train.jsonl")
VAL_JSONL   = os.path.join(OUTPUT_DIR, "val.jsonl")

BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
NUM_EPOCHS  = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import transformers.trainer
transformers.trainer.check_torch_load_is_safe = lambda: None


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def make_messages(d):
    return {"messages": [
        {"role": "system",    "content": "다음 지문을 읽고 문항에 답하시오."},
        {"role": "user",      "content": d["input"]},
        {"role": "assistant", "content": d["output"]},
    ]}


print("데이터셋 로딩...")
train_dataset = Dataset.from_list([make_messages(d) for d in load_jsonl(TRAIN_JSONL)])
val_dataset   = Dataset.from_list([make_messages(d) for d in load_jsonl(VAL_JSONL)])
print(f"Train: {len(train_dataset)} / Val: {len(val_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(OLD_ADAPTER, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"베이스 모델 로딩: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.float16,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

print(f"기존 어댑터 로딩: {OLD_ADAPTER}")
model = PeftModel.from_pretrained(model, OLD_ADAPTER, is_trainable=True)

for _, param in model.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)

model.print_trainable_parameters()

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    bf16=False,
    save_steps=200,
    eval_steps=200,
    logging_steps=50,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    max_length=1024,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=sft_config,
    processing_class=tokenizer,
)

trainer.train(resume_from_checkpoint=True)

trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"저장 완료: {FINAL_DIR}")
