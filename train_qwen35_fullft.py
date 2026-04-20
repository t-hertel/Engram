import os
import sys
import time
import json
import argparse
import torch
import numpy as np
import bitsandbytes as bnb
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from engram_qwen35_integration import Qwen35WithEngram, EngramConfig

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def get_lr(step, warmup_steps, max_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))


def tokenize_texts(texts, tokenizer, seq_len):
    pad_id = tokenizer.pad_token_id or 0
    all_ids = []
    batch_size = 2000
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encodings = tokenizer(batch, add_special_tokens=False, truncation=False, padding=False)
        for ids in encodings["input_ids"]:
            pos = 0
            while pos < len(ids):
                chunk = ids[pos:pos + seq_len]
                if len(chunk) < 10:
                    break
                padded = chunk + [pad_id] * (seq_len - len(chunk))
                all_ids.append(padded)
                pos += seq_len
    return all_ids


def load_domain_data(domain, tokenizer, seq_len=256, split="train", max_examples=None):
    base_dir = os.path.expanduser("~/Documents/Engram")
    cache_dir = os.path.join(base_dir, "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    model_tag = "qwen35_2b"
    cache_key = f"{model_tag}_{domain}_{split}_{seq_len}"
    if max_examples:
        cache_key += f"_max{max_examples}"
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.exists(cache_path):
        print(f"  Loading cached {domain} from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    print(f"  Tokenizing {domain} (no cache)...")
    texts = []

    if domain == "wikitext":
        from datasets import load_dataset
        ds = load_dataset("EleutherAI/wikitext_document_level", "wikitext-103-raw-v1", split=split, token=HF_TOKEN)
        texts = [x["page"] for x in ds if len(x["page"]) > 50]
    elif domain == "math":
        from datasets import load_dataset
        ds = load_dataset("meta-math/MetaMathQA", split="train")
        texts = [x["query"] + "\n" + x["response"] for x in ds if len(x.get("query", "")) > 10]
        if split == "test":
            texts = texts[-5000:]
    elif domain == "code":
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split=split)
        texts = [x["prompt"] + "\n" + x["completion"] for x in ds if len(x.get("prompt", "")) > 5]
    elif domain == "science":
        from datasets import load_dataset
        ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
        texts = [x["title"] + ". " + x["abstract"] for x in ds if len(x.get("abstract", "")) > 50]
        if split == "test":
            texts = texts[-5000:]

    if max_examples and len(texts) > max_examples:
        texts = texts[:max_examples]

    all_ids = tokenize_texts(texts, tokenizer, seq_len)
    torch.save(all_ids, cache_path)
    print(f"  Cached {domain}: {len(all_ids)} sequences -> {cache_path}")
    return all_ids


def main():
    os.environ["TMPDIR"] = os.path.expanduser("~/.tmp")
    os.environ["TEMP"] = os.path.expanduser("~/.tmp")
    os.environ["TMP"] = os.path.expanduser("~/.tmp")

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--domains", type=str, default="wikitext")
    parser.add_argument("--layers", type=str, default="1,6,12,17")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--use_mmap", action="store_true")
    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--engram_lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--max_examples_per_domain", type=int, default=50000)
    parser.add_argument("--multiplier", type=int, default=5)
    args = parser.parse_args()

    layer_ids = [int(x) for x in args.layers.split(",")]
    domains = [d.strip() for d in args.domains.split(",")]

    base_dir = os.path.expanduser("~/Documents/Engram")
    exp_dir = os.path.join(base_dir, "experiments", args.name)
    mmap_dir = os.path.join(exp_dir, "embedding_tables")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_file = os.path.join(exp_dir, "training_log.jsonl")
    output_log = os.path.join(exp_dir, "train_output.log")
    os.makedirs(mmap_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    model_name = "Qwen/Qwen3.5-2B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
    vocab_size = len(tokenizer)

    cfg = EngramConfig(
        layer_ids=layer_ids,
        tokenizer_name_or_path=model_name,
        engram_vocab_size=[vocab_size * args.multiplier, vocab_size * args.multiplier],
    )
    model = Qwen35WithEngram(cfg, mmap_dir=mmap_dir)

    # DO NOT freeze base model — full fine-tune (no grad checkpointing needed for 2B)
    model.cuda()
    model.train()

    trainable_base = sum(p.numel() for p in model.base_model.parameters() if p.requires_grad)
    trainable_engram = sum(p.numel() for p in model.engram_modules.parameters() if p.requires_grad)
    trainable_total = trainable_base + trainable_engram

    tokenizer = model.tokenizer

    # Separate param groups: base model (low lr) + engram (high lr)
    base_params = list(model.base_model.parameters())
    engram_params = list(model.engram_modules.parameters())

    optimizer = bnb.optim.AdamW8bit([
        {"params": base_params, "lr": args.base_lr, "weight_decay": 0.01},
        {"params": engram_params, "lr": args.engram_lr, "weight_decay": 0.01},
    ], betas=(0.9, 0.95))

    print(f"\n=== Loading domains: {domains} ===")
    all_train_ids = []
    domain_counts = {}
    for domain in domains:
        ids = load_domain_data(domain, tokenizer, args.seq_len, "train", args.max_examples_per_domain)
        domain_counts[domain] = len(ids)
        all_train_ids.extend(ids)
        print(f"  {domain}: {len(ids)} sequences")

    print(f"  Total training sequences: {len(all_train_ids)}")

    train_tensors = torch.tensor(all_train_ids, dtype=torch.long)
    perm = torch.randperm(len(train_tensors))
    train_tensors = train_tensors[perm]

    train_dataset = TensorDataset(train_tensors)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: (
            torch.stack([x[0] for x in batch]),
            (lambda ids: ids.clone().masked_fill_(
                ids == (tokenizer.pad_token_id or 0), -100))(
                torch.stack([x[0] for x in batch]))
        ),
        num_workers=0, pin_memory=True,
    )

    start_step = 0
    losses_history = []

    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, weights_only=False)
        model.load_state_dict(ckpt["model_state"], assign=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        losses_history = ckpt.get("losses_history", [])
        print(f"Resumed at step {start_step}, loss was {losses_history[-1]:.4f}")

    print(f"\n{'='*70}")
    print(f"  Experiment: {args.name}")
    print(f"  Model: {model_name}")
    print(f"  Mode: FULL FINE-TUNE + Engram")
    print(f"  Domains: {domains}")
    print(f"  Domain counts: {domain_counts}")
    print(f"  Layers: {layer_ids}")
    print(f"  Trainable base params: {trainable_base:,}")
    print(f"  Trainable engram params: {trainable_engram:,}")
    print(f"  Trainable total: {trainable_total:,}")
    print(f"  Base LR: {args.base_lr}  Engram LR: {args.engram_lr}")
    print(f"  Multiplier: {args.multiplier}x")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"{'='*70}\n")

    print(f"{'Step':>6} {'Loss':>9} {'Avg100':>9} {'LR_base':>10} {'LR_eng':>10} {'tok/s':>8} {'ms/step':>8} {'GPU(GB)':>8}")
    print("-" * 82)

    torch.cuda.reset_peak_memory_stats()
    train_iter = iter(train_loader)
    t_run_start = time.time()
    recent_losses = []

    for step in range(start_step, args.steps):
        lr_base = get_lr(step, args.warmup, args.steps, args.base_lr)
        lr_engram = get_lr(step, args.warmup, args.steps, args.engram_lr)
        optimizer.param_groups[0]["lr"] = lr_base
        optimizer.param_groups[1]["lr"] = lr_engram

        torch.cuda.synchronize()
        t0 = time.time()

        try:
            input_ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, labels = next(train_iter)

        input_ids = input_ids.cuda()
        labels = labels.cuda()

        logits = model(input_ids, use_mmap=args.use_mmap)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        loss_val = loss.item()

        torch.cuda.synchronize()
        t1 = time.time()

        losses_history.append(loss_val)
        recent_losses.append(loss_val)
        if len(recent_losses) > 100:
            recent_losses.pop(0)

        step_ms = (t1 - t0) * 1000
        toks_per_sec = (args.batch_size * args.seq_len) / (step_ms / 1000) if step_ms > 0 else 0
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9
        avg100 = np.mean(recent_losses)

        if step % args.log_every == 0 or step == start_step:
            line = f"{step:>6} {loss_val:>9.4f} {avg100:>9.4f} {lr_base:>10.7f} {lr_engram:>10.6f} {toks_per_sec:>8.0f} {step_ms:>8.1f} {gpu_mem:>8.2f}"
            print(line)
            with open(output_log, "a") as f:
                f.write(line + "\n")
            sys.stdout.flush()

        log_entry = {
            "step": step, "loss": loss_val, "avg100": avg100,
            "lr_base": lr_base, "lr_engram": lr_engram,
            "tokens_per_sec": toks_per_sec, "step_ms": step_ms,
            "gpu_gb": gpu_mem, "experiment": args.name, "domains": domains,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if step % args.save_every == 0 and step > start_step:
            ckpt_data = {
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "step": step, "loss": loss_val, "losses_history": losses_history,
                "config": {
                    "layers": layer_ids, "base_lr": args.base_lr,
                    "engram_lr": args.engram_lr, "seq_len": args.seq_len,
                    "batch_size": args.batch_size, "name": args.name,
                    "model": model_name, "domains": domains,
                    "multiplier": args.multiplier, "full_finetune": True,
                },
            }
            torch.save(ckpt_data, os.path.join(ckpt_dir, f"step_{step}.pt"))
            torch.save(ckpt_data, latest_ckpt)
            print(f"  >> Checkpoint saved: step {step}")
            sys.stdout.flush()

    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "step": args.steps, "loss": losses_history[-1],
        "losses_history": losses_history,
        "config": {
            "layers": layer_ids, "base_lr": args.base_lr,
            "engram_lr": args.engram_lr, "seq_len": args.seq_len,
            "batch_size": args.batch_size, "name": args.name,
            "model": model_name, "domains": domains,
            "multiplier": args.multiplier, "full_finetune": True,
        },
    }, final_path)

    total_time = time.time() - t_run_start
    config_out = {
        "name": args.name, "layers": layer_ids, "domains": domains,
        "domain_counts": domain_counts, "total_sequences": len(all_train_ids),
        "steps": args.steps, "batch_size": args.batch_size, "seq_len": args.seq_len,
        "base_lr": args.base_lr, "engram_lr": args.engram_lr,
        "warmup": args.warmup, "trainable_base": trainable_base,
        "trainable_engram": trainable_engram,
        "trainable_total": trainable_total, "multiplier": args.multiplier,
        "full_finetune": True,
        "total_time_s": total_time, "final_loss": losses_history[-1],
        "best_loss": min(losses_history), "initial_loss": losses_history[0],
        "improvement": losses_history[0] - losses_history[-1],
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    summary = f"""
{'='*70}
  Experiment Complete: {args.name}
  Model: {model_name}
  Mode: FULL FINE-TUNE + Engram
  Domains: {domains}
{'='*70}
  Total steps:    {args.steps}
  Total time:     {total_time/60:.1f} minutes
  Initial loss:   {losses_history[0]:.4f}
  Final loss:     {losses_history[-1]:.4f}
  Best loss:      {min(losses_history):.4f}
  Peak GPU:       {torch.cuda.max_memory_allocated() / 1e9:.2f} GB
{'='*70}
"""
    print(summary)
    with open(output_log, "a") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
