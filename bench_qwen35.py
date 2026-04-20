import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE_DIR = os.path.expanduser("~/Documents/Engram")
MODEL_NAME = "Qwen/Qwen3.5-2B-Base"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def score_sequence(model, input_ids):
    with torch.no_grad():
        if hasattr(model, 'engram_modules'):
            logits = model(input_ids, use_mmap=True)
        else:
            logits = model(input_ids).logits
        return torch.nn.functional.log_softmax(logits.float(), dim=-1)


def eval_hellaswag(model, tokenizer, max_examples=None):
    print("  Loading HellaSwag...")
    ds = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, example in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            ctx = example["ctx"]
            endings = example["endings"]
            label = int(example["label"]) if example["label"] != "" else 0
            ctx_enc = tokenizer.encode(ctx, add_special_tokens=False)
            scores = []
            for ending in endings:
                full = ctx + " " + ending
                ids = tokenizer.encode(full, add_special_tokens=False)
                ctx_len = len(ctx_enc)
                if len(ids) <= ctx_len:
                    scores.append(-100.0)
                    continue
                input_ids = torch.tensor([ids], dtype=torch.long).cuda()
                log_probs = score_sequence(model, input_ids)
                score = 0.0
                for t in range(ctx_len, len(ids)):
                    score += log_probs[0, t - 1, ids[t]].item()
                scores.append(score)
            pred = int(np.argmax(scores))
            if pred == label:
                correct += 1
            total += 1
            if total % 500 == 0:
                print(f"    HellaSwag {total}: acc={correct/total:.4f}")
    acc = correct / total if total > 0 else 0
    print(f"  HellaSwag: {acc:.4f} ({correct}/{total})")
    return acc, total


def eval_piqa(model, tokenizer, max_examples=None):
    print("  Loading PIQA...")
    ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, example in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            goal = example["goal"]
            sol1 = example["sol1"]
            sol2 = example["sol2"]
            label = int(example["label"])
            goal_enc = tokenizer.encode(goal, add_special_tokens=False)
            goal_len = len(goal_enc)
            scores = []
            for sol in [sol1, sol2]:
                ids = tokenizer.encode(goal + " " + sol, add_special_tokens=False)
                if len(ids) <= goal_len:
                    scores.append(-100.0)
                    continue
                input_ids = torch.tensor([ids], dtype=torch.long).cuda()
                log_probs = score_sequence(model, input_ids)
                score = sum(log_probs[0, t - 1, ids[t]].item() for t in range(goal_len, len(ids)))
                scores.append(score)
            pred = int(np.argmax(scores))
            if pred == label:
                correct += 1
            total += 1
            if total % 500 == 0:
                print(f"    PIQA {total}: acc={correct/total:.4f}")
    acc = correct / total if total > 0 else 0
    print(f"  PIQA: {acc:.4f} ({correct}/{total})")
    return acc, total


def eval_arc_challenge(model, tokenizer, max_examples=None):
    print("  Loading ARC-Challenge...")
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, example in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            question = example["question"]
            choices = example["choices"]
            texts = choices["text"]
            labels_raw = choices["label"]
            answer_key = example["answerKey"]
            label_to_idx = {l: idx for idx, l in enumerate(labels_raw)}
            if answer_key not in label_to_idx:
                total += 1
                continue
            correct_idx = label_to_idx[answer_key]
            q_enc = tokenizer.encode(question, add_special_tokens=False)
            q_len = len(q_enc)
            scores = []
            for choice_text in texts:
                ids = tokenizer.encode(question + " " + choice_text, add_special_tokens=False)
                if len(ids) <= q_len:
                    scores.append(-100.0)
                    continue
                input_ids = torch.tensor([ids], dtype=torch.long).cuda()
                log_probs = score_sequence(model, input_ids)
                score = sum(log_probs[0, t - 1, ids[t]].item() for t in range(q_len, len(ids)))
                scores.append(score)
            pred = int(np.argmax(scores))
            if pred == correct_idx:
                correct += 1
            total += 1
            if total % 500 == 0:
                print(f"    ARC-C {total}: acc={correct/total:.4f}")
    acc = correct / total if total > 0 else 0
    print(f"  ARC-Challenge: {acc:.4f} ({correct}/{total})")
    return acc, total


def run_baseline(tasks, task_fns):
    print("\n" + "=" * 70)
    print(f"  BASELINE: {MODEL_NAME} (no Engram)")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, token=HF_TOKEN, trust_remote_code=True
    ).cuda().eval()

    results = {}
    for name in tasks:
        if name in task_fns:
            acc, n = task_fns[name](model, tokenizer)
            results[name] = {"acc": acc, "total": n}

    del model
    torch.cuda.empty_cache()
    return results


def run_engram(exp_name, tasks, task_fns):
    print("\n" + "=" * 70)
    print(f"  ENGRAM: {exp_name}")
    print("=" * 70)
    from engram_qwen35_integration import Qwen35WithEngram, EngramConfig

    exp_dir = os.path.join(BASE_DIR, "experiments", exp_name)

    ckpt_name = None
    for candidate in ["final.pt", "latest.pt"]:
        if os.path.exists(os.path.join(exp_dir, "checkpoints", candidate)):
            ckpt_name = candidate
            break

    if ckpt_name is None:
        print("  SKIP: no checkpoint found")
        return None

    ckpt_path = os.path.join(exp_dir, "checkpoints", ckpt_name)
    ckpt = torch.load(ckpt_path, weights_only=False)
    layer_ids = ckpt["config"]["layers"]
    multiplier = ckpt["config"].get("multiplier", 5)
    model_name = ckpt["config"].get("model", MODEL_NAME)

    tokenizer_for_vocab = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
    vocab_size = len(tokenizer_for_vocab)

    cfg = EngramConfig(
        layer_ids=layer_ids,
        tokenizer_name_or_path=model_name,
        engram_vocab_size=[vocab_size * multiplier, vocab_size * multiplier],
    )
    mmap_dir = os.path.join(exp_dir, "embedding_tables")
    model = Qwen35WithEngram(cfg, mmap_dir=mmap_dir)

    is_full_ft = ckpt["config"].get("full_finetune", False)
    if is_full_ft:
        print(f"  Mode: FULL FINE-TUNE + Engram")
        model.load_state_dict(ckpt["model_state"], assign=True)
    else:
        print(f"  Mode: Frozen base + Engram only")
        model.freeze_base_model()
        model.engram_modules.load_state_dict(ckpt["engram_state"])

    model.cuda()
    model.eval()
    tokenizer = model.tokenizer

    print(f"  Checkpoint: {ckpt_name}  Step: {ckpt['step']}  Layers: {layer_ids}")

    results = {}
    for name in tasks:
        if name in task_fns:
            acc, n = task_fns[name](model, tokenizer)
            results[name] = {"acc": acc, "total": n}

    results["checkpoint"] = ckpt_name
    results["step"] = ckpt["step"]
    results["full_finetune"] = is_full_ft

    del model
    torch.cuda.empty_cache()
    return results


def main():
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="qwen35_2b_fullft_10k")
    parser.add_argument("--task", type=str, default="all", help="hellaswag, piqa, arc_challenge, or all")
    args = parser.parse_args()

    if args.task == "all":
        tasks = ["hellaswag", "piqa", "arc_challenge"]
    else:
        tasks = [t.strip() for t in args.task.split(",")]

    task_fns = {
        "hellaswag": eval_hellaswag,
        "piqa": eval_piqa,
        "arc_challenge": eval_arc_challenge,
    }

    all_results = {}

    print("Phase 1: Baseline evaluation")
    baseline = run_baseline(tasks, task_fns)
    all_results["baseline"] = baseline

    print("\nPhase 2: Engram evaluation")
    engram = run_engram(args.experiment, tasks, task_fns)
    if engram:
        all_results["engram"] = engram

    out_path = os.path.join(BASE_DIR, f"benchmark_{args.experiment}_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  SUMMARY ({MODEL_NAME})")
    print("=" * 70)
    print(f"  {'Experiment':15} {'HellaSwag':>12} {'PIQA':>12} {'ARC-C':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

    b = all_results["baseline"]
    for task in tasks:
        if task in b:
            print(f"  {'Baseline':15} {b.get('hellaswag',{}).get('acc',float('nan')):>12.4f} {b.get('piqa',{}).get('acc',float('nan')):>12.4f} {b.get('arc_challenge',{}).get('acc',float('nan')):>12.4f}")
            break
    else:
        pass

    if "engram" in all_results and all_results["engram"]:
        e = all_results["engram"]
        is_full_ft = e.get("full_finetune", False)
        label = "FullFT+Engram" if is_full_ft else "Engram(frozen)"
        print(f"  {label:15} {e.get('hellaswag',{}).get('acc',float('nan')):>12.4f} {e.get('piqa',{}).get('acc',float('nan')):>12.4f} {e.get('arc_challenge',{}).get('acc',float('nan')):>12.4f}")
        print(f"  {'Delta':15}", end="")
        for task in tasks:
            if task in e and task in b:
                delta = e[task]["acc"] - b[task]["acc"]
                print(f" {delta:>+11.4f}", end="")
        print()

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
