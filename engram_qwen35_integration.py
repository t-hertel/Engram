import os
import math
import mmap
from dataclasses import dataclass, field
from typing import List, Optional
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import normalizers, Regex


@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "Qwen/Qwen3.5-4B-Base"
    vocab_size: int = -1
    engram_vocab_size: List[int] = field(default_factory=list)
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 8, 16, 23])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    hidden_size: int = -1
    num_layers: int = -1

    def resolve(self, base_model=None):
        if base_model is not None:
            config = base_model.config
            if hasattr(config, 'text_config'):
                tc = config.text_config
                self.hidden_size = tc.hidden_size
                self.num_layers = tc.num_hidden_layers
                self.vocab_size = tc.vocab_size
            else:
                self.hidden_size = config.hidden_size
                self.num_layers = config.num_hidden_layers
                self.vocab_size = config.vocab_size
        if self.vocab_size <= 0:
            tok = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
            self.vocab_size = len(tok)
        if self.pad_id < 0:
            self.pad_id = self.vocab_size - 3
        if not self.engram_vocab_size:
            self.engram_vocab_size = [self.vocab_size * 5, self.vocab_size * 5]


class CompressedTokenizer:
    def __init__(self, tokenizer_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []
        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "\ufffd" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
                if isinstance(key, list):
                    key = key[0] if key else str(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text
            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]
        return lookup, len(new_tokens)

    def compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    def __init__(self, config: EngramConfig):
        self.vocab_size_per_ngram = config.engram_vocab_size
        self.max_ngram_size = config.max_ngram_size
        self.n_head_per_ngram = config.n_head_per_ngram
        self.pad_id = config.pad_id
        self.layer_ids = config.layer_ids

        self.compressed_tokenizer = CompressedTokenizer(config.tokenizer_name_or_path)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[config.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}
        rng_base = np.random.default_rng(config.seed)
        for layer_id in self.layer_ids:
            base_seed = int(config.seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64,
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self._calc_vocab_across_layers()

    def _calc_vocab_across_layers(self):
        seen_primes = set()
        result = {}
        for layer_id in self.layer_ids:
            all_ngram = []
            for ngram in range(2, self.max_ngram_size + 1):
                head_sizes = []
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                start = vocab_size - 1
                for _ in range(self.n_head_per_ngram):
                    p = find_next_prime(start, seen_primes)
                    seen_primes.add(p)
                    head_sizes.append(p)
                    start = p
                all_ngram.append(head_sizes)
            result[layer_id] = all_ngram
        return result

    def _get_ngram_hashes(self, input_ids, layer_id):
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k):
            if k == 0:
                return x
            return np.pad(x, ((0, 0), (k, 0)), mode='constant', constant_values=self.pad_id)[:, :T]

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        all_hashes = []

        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        compressed = self.compressed_tokenizer.compress(input_ids)
        return {lid: self._get_ngram_hashes(compressed, lid) for lid in self.layer_ids}


class OffloadedEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices_flat, weight_np, offsets, D, orig_shape):
        rows = weight_np[indices_flat]
        ctx.save_for_backward(indices_flat)
        ctx.weight_np = weight_np
        ctx.D = D
        ctx.orig_shape = orig_shape
        return torch.from_numpy(rows.copy()).cuda()

    @staticmethod
    def backward(ctx, grad_output):
        indices_flat = ctx.saved_tensors[0]
        grad_cpu = grad_output.cpu().numpy().astype(np.float16)
        np.add.at(ctx.weight_np, indices_flat.numpy(), grad_cpu)
        return None, None, None, None, None


class OffloadedEmbedding:
    def __init__(self, list_of_N, D, mmap_path):
        self.list_of_N = list_of_N
        self.D = D
        self.num_heads = len(list_of_N)
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.offsets = torch.tensor(offsets, dtype=torch.long)

        total_N = sum(list_of_N)
        self.total_N = total_N
        byte_size = total_N * D * 2

        self.mmap_path = mmap_path
        os.makedirs(os.path.dirname(mmap_path), exist_ok=True)

        if not os.path.exists(mmap_path) or os.path.getsize(mmap_path) < byte_size:
            print(f"  Creating mmap file: {mmap_path} ({byte_size / 1e9:.2f} GB)")
            with open(mmap_path, 'wb') as f:
                f.write(b'\x00' * byte_size)

        self.file = open(mmap_path, 'r+b')
        self.mmap_obj = mmap.mmap(self.file.fileno(), byte_size)
        self.weight_np = np.frombuffer(self.mmap_obj, dtype=np.float16).reshape(total_N, D)
        self.weight_gpu = None

    def to_gpu(self):
        if self.weight_gpu is None:
            self.weight_gpu = torch.from_numpy(self.weight_np.copy()).cuda()
        return self.weight_gpu

    def lookup(self, hash_ids, use_mmap=False):
        shifted = hash_ids + self.offsets.to(hash_ids.device)
        if use_mmap:
            indices_flat = shifted.cpu().numpy().flatten()
            result = OffloadedEmbeddingFunction.apply(
                torch.from_numpy(indices_flat), self.weight_np, self.offsets, self.D, shifted.shape
            )
            return result.view(*shifted.shape, self.D)
        else:
            w = self.weight_gpu if self.weight_gpu is not None else self.to_gpu()
            return torch.nn.functional.embedding(shifted, w)

    def close(self):
        self.mmap_obj.close()
        self.file.close()


class ShortConv(nn.Module):
    def __init__(self, hidden_size, kernel_size=4, dilation=1, norm_eps=1e-5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.act = nn.SiLU()

    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.norm(x)
        x_t = x_norm.transpose(1, 2)
        y = self.conv(x_t)[..., :T]
        y = self.act(y)
        return y.transpose(1, 2)


class EngramModule(nn.Module):
    def __init__(self, config: EngramConfig, layer_id: int, mmap_dir: str):
        super().__init__()
        self.layer_id = layer_id
        self.config = config

        self.hash_mapping = NgramHashMapping(config)

        list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[layer_id] for x in y]
        D = config.n_embed_per_ngram // config.n_head_per_ngram

        mmap_path = os.path.join(mmap_dir, f"engram_layer_{layer_id}.bin")
        self.offloaded_embed = OffloadedEmbedding(list_of_N, D, mmap_path)

        self.register_buffer("offsets", self.offloaded_embed.offsets)

        engram_hidden = (config.max_ngram_size - 1) * config.n_embed_per_ngram

        self.value_proj = nn.Linear(engram_hidden, config.hidden_size)
        self.key_proj = nn.Linear(engram_hidden, config.hidden_size)
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.norm2 = nn.RMSNorm(config.hidden_size)

        self.short_conv = ShortConv(
            config.hidden_size,
            kernel_size=config.kernel_size,
            dilation=config.max_ngram_size,
        )

    def forward(self, hidden_states, input_ids_np, use_mmap=False):
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()

        hash_ids = torch.from_numpy(
            self.hash_mapping.hash(input_ids_np)[self.layer_id]
        ).to(hidden_states.device)
        embeddings = self.offloaded_embed.lookup(hash_ids, use_mmap=use_mmap).float().flatten(start_dim=-2)

        key = self.norm1(self.key_proj(embeddings))
        query = self.norm2(hidden_states)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.config.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid()

        value = gate * self.value_proj(embeddings)
        conv_out = self.short_conv(value)
        output = value + conv_out
        return output.to(orig_dtype)


class Qwen35WithEngram(nn.Module):
    def __init__(self, config: EngramConfig, mmap_dir: str):
        super().__init__()
        self.config = config
        model_name = config.tokenizer_name_or_path.split("/")[-1]
        print(f"Loading {model_name} base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.tokenizer_name_or_path, dtype=torch.bfloat16, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, trust_remote_code=True)
        config.resolve(self.base_model)

        print("Initializing Engram modules...")
        self.engram_modules = nn.ModuleDict({
            str(lid): EngramModule(config, lid, mmap_dir)
            for lid in config.layer_ids
        })

        self._input_ids = None
        self._use_mmap = False
        self._hooks = []

        for lid in config.layer_ids:
            hook = self.base_model.model.layers[lid].register_forward_pre_hook(
                self._make_hook(lid)
            )
            self._hooks.append(hook)

        embed_size = sum(
            m.offloaded_embed.total_N * m.offloaded_embed.D
            for m in self.engram_modules.values()
        )
        non_embed = sum(
            p.numel() for m in self.engram_modules.values() for p in m.parameters()
        )
        base_params = sum(p.numel() for p in self.base_model.parameters())

        print(f"\n=== Model Summary ===")
        print(f"Base model params: {base_params:,} ({base_params * 2 / 1e9:.2f} GB bf16)")
        print(f"Engram embedding table (NVMe/RAM): {embed_size:,} ({embed_size * 2 / 1e9:.2f} GB fp16)")
        print(f"Engram trainable params (GPU): {non_embed:,} ({non_embed * 2 / 1e6:.2f} MB fp16)")
        print(f"Engram hooks at layers: {config.layer_ids}")

    def _make_hook(self, layer_id):
        def hook_fn(module, args):
            hidden = args[0] if len(args) > 0 else None
            if self._input_ids is not None and hidden is not None:
                engram_out = self.engram_modules[str(layer_id)](
                    hidden, self._input_ids, use_mmap=self._use_mmap
                )
                new_hidden = hidden + engram_out
                return (new_hidden,) + args[1:]
            return args
        return hook_fn

    def forward(self, input_ids, use_mmap=False):
        self._input_ids = input_ids.cpu().numpy()
        self._use_mmap = use_mmap
        logits = self.base_model(input_ids).logits
        self._input_ids = None
        return logits

    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("Frozen base model. Only Engram modules trainable.")

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_engram_params(self):
        return list(self.engram_modules.parameters())


if __name__ == "__main__":
    cfg = EngramConfig()
    mmap_dir = os.path.expanduser("~/Documents/Engram/experiments/qwen35_4b_test/embedding_tables")
    os.makedirs(mmap_dir, exist_ok=True)

    model = Qwen35WithEngram(cfg, mmap_dir=mmap_dir)
    model.freeze_base_model()
    model.cuda()

    trainable = model.count_trainable()

    text = "The quick brown fox jumps over the lazy dog."
    inputs = model.tokenizer(text, return_tensors="pt").input_ids.cuda()

    print(f"\n=== GPU Forward Pass ===")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    import time
    t0 = time.time()

    with torch.no_grad():
        logits = model(inputs, use_mmap=False)

    torch.cuda.synchronize()
    t1 = time.time()
    gpu_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"Input: {inputs.shape} -> Output: {logits.shape}")
    print(f"Forward pass: {(t1 - t0) * 1000:.1f} ms")
    print(f"Peak GPU memory: {gpu_mem:.2f} GB")
    print(f"Trainable params: {trainable:,}")
