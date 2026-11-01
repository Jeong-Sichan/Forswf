"""QLoRA format adaptation. Run on a GPU after checking the model license/card."""

import argparse
from pathlib import Path

from prompting import messages, read_examples, render, require_local_dataset, require_pinned_revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Reviewed JSONL training data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="KORMo-Team/KORMo-10B-sft")
    parser.add_argument("--revision", required=True, help="Pinned Hugging Face model commit SHA")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=1536)
    args = parser.parse_args()

    require_pinned_revision(args.revision)
    require_local_dataset(args.train)
    if not Path(args.output).resolve().is_relative_to("/data"):
        raise ValueError("LoRA 모델 출력은 /data 사용자 영역에 저장해 주세요")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA 학습에는 CUDA GPU가 필요합니다")
    rows = read_examples(args.train)
    if len({row["expression"].strip() for row in rows}) < 50:
        raise ValueError("서로 다른 표현 50개 미만으로 학습 결과를 주장하지 마세요. 검토한 데이터를 더 모아 주세요.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )

    class Examples(torch.utils.data.Dataset):
        def __init__(self):
            self.items = []
            for row in rows:
                prompt = render(
                    tokenizer,
                    messages(row["expression"], row["situation"], row["relationship"], row["language"]),
                    True,
                )
                completion = row["answer"].strip() + tokenizer.eos_token
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                answer_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
                if len(prompt_ids) + len(answer_ids) > args.max_length:
                    raise ValueError(f"사례가 max-length를 초과합니다: {row['expression'][:30]}")
                self.items.append({
                    "input_ids": prompt_ids + answer_ids,
                    "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
                    "labels": [-100] * len(prompt_ids) + answer_ids,
                })

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        return {
            key: torch.tensor([
                item[key] + [(-100 if key == "labels" else 0 if key == "attention_mask" else tokenizer.pad_token_id)]
                * (width - len(item[key])) for item in batch
            ]) for key in ("input_ids", "attention_mask", "labels")
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            warmup_ratio=0.03,
            logging_steps=5,
            save_strategy="no",
            gradient_checkpointing=True,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            optim="paged_adamw_8bit",
            report_to="none",
        ),
        train_dataset=Examples(),
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
