import os
import sys
import time
import math
import json
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from engram_qwen35_integration import Qwen35WithEngram, EngramConfig

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_NAME = "Qwen/Qwen3.5-2B-Base"
DATASET = "EleutherAI/wikitext_document_level"
DATASET_NAME = "wikitext-103-raw-v1"
BASE_DIR = os.path.expanduser("~/Documents/Engram")


def load_test_data(tokenizer, seq_len=256, batch_size=2):
    ds = load_dataset(DATASET, DATASET_NAME, split="test", token=HF_TOKEN)

    def tokenize_fn(examples):
        all_ids = []
        for text in examples["page"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            while len(ids) > 0:
                chunk = ids[:seq_len]
                if len(chunk) < 10:
                    break
                pad_id = tokenizer.pad_token_id or 0
                padded = chunk + [pad_id] * (seq_len - len(chunk))
                all_ids.append(padded)
                ids = ids[seq_len:]
        return {"input_ids": all_ids}

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names, batch_size=500, num_proc=4)

    def collate_fn(batch):
        ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
        labels = ids.clone()
        pad_id = tokenizer.pad_token_id or 0
        labels[labels == pad_id] = -100
        return ids, labels

    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    return loader


def compute_perplexity(model_fn, loader, label=""):
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    t0 = time.time()

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.cuda()
            labels = labels.cuda()
            logits = model_fn(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100, reduction="sum"
            )
            n_valid = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += n_valid
            n_batches += 1

            if n_batches % 100 == 0:
                ppl = math.exp(total_loss / total_tokens)
                print(f"  [{label}] Batch {n_batches}: running ppl={ppl:.2f}")

    elapsed = time.time() - t0
    avg_ce = total_loss / total_tokens
    ppl = math.exp(avg_ce)
    return {"ce_loss": avg_ce, "perplexity": ppl, "tokens": total_tokens, "time_s": elapsed, "batches": n_batches}


def eval_baseline(loader, model_name=MODEL_NAME):
    print("\n" + "=" * 70)
    print(f"  BASELINE: {model_name} (no Engram)")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, token=HF_TOKEN, trust_remote_code=True
    ).cuda()
    model.eval()
    results = compute_perplexity(lambda ids: model(ids).logits, loader, "baseline")
    print(f"  Perplexity: {results['perplexity']:.2f}  CE Loss: {results['ce_loss']:.4f}  Tokens: {results['tokens']:,}")
    del model
    torch.cuda.empty_cache()
    return results


def eval_engram(loader, exp_name, ckpt_name=None):
    exp_dir = os.path.join(BASE_DIR, "experiments", exp_name)
    mmap_dir = os.path.join(exp_dir, "embedding_tables")

    if ckpt_name is None:
        for candidate in ["final.pt", "latest.pt"]:
            if os.path.exists(os.path.join(exp_dir, "checkpoints", candidate)):
                ckpt_name = candidate
                break
        else:
            print(f"  SKIP {exp_name}: no checkpoint found")
            return None

    ckpt_path = os.path.join(exp_dir, "checkpoints", ckpt_name)
    print(f"\n{'='*70}")
    print(f"  ENGRAM: {exp_name} ({ckpt_name})")
    print("=" * 70)

    ckpt = torch.load(ckpt_path, weights_only=False)
    layer_ids = ckpt["config"]["layers"]
    model_name = ckpt["config"].get("model", MODEL_NAME)
    multiplier = ckpt["config"].get("multiplier", 5)
    is_full_ft = ckpt["config"].get("full_finetune", False)

    tokenizer_for_vocab = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
    vocab_size = len(tokenizer_for_vocab)
    cfg = EngramConfig(
        layer_ids=layer_ids,
        tokenizer_name_or_path=model_name,
        engram_vocab_size=[vocab_size * multiplier, vocab_size * multiplier],
    )
    model = Qwen35WithEngram(cfg, mmap_dir=mmap_dir)

    if is_full_ft:
        print(f"  Mode: FULL FINE-TUNE + Engram")
        model.load_state_dict(ckpt["model_state"], assign=True)
    else:
        print(f"  Mode: Frozen base + Engram only")
        model.freeze_base_model()
        model.engram_modules.load_state_dict(ckpt["engram_state"])

    model.cuda()
    model.eval()

    print(f"  Layers: {layer_ids}  Step: {ckpt['step']}  Train loss: {ckpt['loss']:.4f}")

    results = compute_perplexity(lambda ids: model(ids, use_mmap=True), loader, exp_name)
    results["layers"] = layer_ids
    results["step"] = ckpt["step"]
    results["checkpoint"] = ckpt_name
    results["full_finetune"] = is_full_ft
    print(f"  Perplexity: {results['perplexity']:.2f}  CE Loss: {results['ce_loss']:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="qwen35_2b_fullft_10k")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN, trust_remote_code=True)
    loader = load_test_data(tokenizer, batch_size=2)

    baseline = eval_baseline(loader, args.model)

    all_results = {"baseline": baseline, "engram": None}

    engram = eval_engram(loader, args.experiment)
    if engram:
        all_results["engram"] = engram

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':25} {'Layers':12} {'PPL':>8} {'CE Loss':>10} {'vs Base':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
    base_ppl = baseline["perplexity"]
    print(f"  {'Baseline':25} {'---':12} {base_ppl:>8.2f} {baseline['ce_loss']:>10.4f} {'---':>10}")
    if engram:
        delta = engram["perplexity"] - base_ppl
        pct = (engram["perplexity"] / base_ppl - 1) * 100
        label = "FullFT+Engram" if engram.get("full_finetune") else "Engram(frozen)"
        print(f"  {label:25} {str(engram['layers']):12} {engram['perplexity']:>8.2f} {engram['ce_loss']:>10.4f} {delta:>+8.2f} ({pct:+.1f}%)")
    print(f"{'='*70}")

    exp_tag = args.experiment
    out_path = args.output or os.path.join(BASE_DIR, f"eval_{exp_tag}_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
