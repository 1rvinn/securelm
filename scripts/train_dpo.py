#!/usr/bin/env python
"""Stage 4 — DPO training (QLoRA) on data/dpo_pairs.jsonl.

Runs on the rented A100 (Linux). Loads Llama-3.1-8B-Instruct in 4-bit (NF4),
attaches LoRA adapters to attention+MLP projections, and trains with TRL's
DPOTrainer. No reference model is loaded separately: with a PEFT config the
reference is the base model with adapters disabled (the memory win). Saves the
LoRA adapter (Stage 5 serves base + adapter via vLLM — no merge needed).

Env (A100; torch 2.8.0 already in the RunPod image):
    pip install "trl==1.7.1" "peft==0.19.1" "bitsandbytes==0.49.2" \
                "accelerate==1.14.0" "datasets==5.0.0"

Run:
    export HF_HOME=/workspace/hf; export HF_TOKEN=hf_...
    python train_dpo.py --data /workspace/dpo_pairs.jsonl \
                        --output /workspace/checkpoints/dpo-llama31-8b
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

LLAMA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]


def main() -> int:
    ap = argparse.ArgumentParser(description="QLoRA DPO training")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--data", default="/workspace/dpo_pairs.jsonl")
    ap.add_argument("--output", default="/workspace/checkpoints/dpo-llama31-8b")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)          # QLoRA LR (DPOConfig default 1e-6 is full-FT)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)       # effective batch 16
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--max-prompt-len", type=int, default=1024)  # jailbreak/base64 prompts can be long
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--no-quantize", action="store_true",
                    help="Load the base model in plain bf16 instead of 4-bit "
                         "NF4 (still LoRA, NOT full fine-tuning). Everything "
                         "else — LoRA config, ref_model=None+peft, DPO "
                         "hyperparameters — is unchanged.")
    args = ap.parse_args()

    # --- dataset (conversational: prompt/chosen/rejected as message lists) ---
    ds = load_dataset("json", data_files=args.data, split="train")
    keep = [c for c in ("prompt", "chosen", "rejected") if c in ds.column_names]
    ds = ds.select_columns(keep)  # drop our metadata columns; TRL wants only these
    print(f"Loaded {len(ds)} DPO pairs from {args.data}")

    # --- tokenizer ---
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"                     # required by DPOTrainer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- base model: 4-bit NF4 (QLoRA) or plain bf16 (--no-quantize) ---
    # Quantization only affects the FROZEN base weights' precision; the LoRA
    # adapters (and the ref_model=None adapter-disable trick) are identical
    # either way. --no-quantize is still LoRA, NOT full fine-tuning.
    if args.no_quantize:
        print("Loading base model in bf16 (no quantization) — LoRA adapters only.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print("Loading base model in 4-bit NF4 (QLoRA).")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    model.config.use_cache = False                # required with gradient checkpointing

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LLAMA_TARGET_MODULES,
    )

    # Build config args, then keep only those this TRL version's DPOConfig
    # accepts — arg names drift across TRL releases (e.g. max_prompt_length).
    desired_args = dict(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        beta=args.beta,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        max_prompt_length=args.max_prompt_len,
        bf16=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        precompute_ref_log_probs=True,            # cache ref log-probs, free the ref path
    )
    valid = set(DPOConfig.__dataclass_fields__)
    dropped = [k for k in desired_args if k not in valid]
    if dropped:
        print(f"[warn] this TRL's DPOConfig doesn't accept {dropped} — dropping them "
              f"(the trainer will use its defaults for those).")
    cfg = DPOConfig(**{k: v for k, v in desired_args.items() if k in valid})

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                           # PEFT: reference = adapter-disabled base
        args=cfg,
        train_dataset=ds,
        processing_class=tok,                      # renamed from tokenizer= in current TRL
        peft_config=peft_cfg,
    )

    trainer.train()

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory allocated: {peak_gb:.2f} GB")

    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"\nSaved LoRA adapter to {args.output}")
    print("Stage 5 will serve it with: vllm serve <base> --enable-lora "
          f"--lora-modules dpo={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
