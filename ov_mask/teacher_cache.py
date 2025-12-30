import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from tqdm import tqdm

from ovmask.utils import tokenize_prompts, last_token_indices, select_last_logits


@dataclass
class TeacherMeta:
    n: int
    vocab: int
    shard_size: int
    dtype: str  # stored dtype, e.g. "float16"


def _teacher_dir(base_dir: str, task: str, split: str, variant: str) -> str:
    d = os.path.join(base_dir, task, split, variant)
    os.makedirs(d, exist_ok=True)
    return d


def teacher_exists(base_dir: str, task: str, split: str, variant: str) -> bool:
    d = _teacher_dir(base_dir, task, split, variant)
    return os.path.exists(os.path.join(d, "meta.json"))


@torch.no_grad()
def build_teacher_cache(
    base_dir: str,
    task: str,
    split: str,
    variant: str,
    prompts: List[str],
    teacher_model,
    shard_size: int,
    force: bool,
    logger,
) -> None:
    """
    Stores logp_teacher = log_softmax(logits_last) as float16 on disk in shards.

    Layout:
      teacher_cache/{task}/{split}/{variant}/
        meta.json
        shard_00000.pt
        shard_00001.pt
        ...
    """
    d = _teacher_dir(base_dir, task, split, variant)
    meta_path = os.path.join(d, "meta.json")

    if (not force) and os.path.exists(meta_path):
        logger.info(f"[teacher] exists, skipping: {d}")
        return

    logger.info(f"[teacher] building cache: task={task} split={split} variant={variant} n={len(prompts)}")

    # We'll run teacher on its device
    device = next(teacher_model.parameters()).device
    tokenizer = teacher_model.tokenizer

    all_shards = []
    shard_idx = 0

    # pick a reasonably large forward batch for teacher; we reuse your training batch_size later anyway
    forward_bs = 64

    current = []
    for i in tqdm(range(0, len(prompts), forward_bs), desc=f"teacher {task}/{split}/{variant}"):
        batch_prompts = prompts[i : i + forward_bs]
        input_ids, attn_mask = tokenize_prompts(tokenizer, batch_prompts, device=device)
        logits = teacher_model(input_ids)  # [b, seq, vocab]
        last_idx = last_token_indices(attn_mask)
        logits_last = select_last_logits(logits, last_idx)  # [b, vocab]
        logp = torch.log_softmax(logits_last.float(), dim=-1).to(torch.float16).cpu()  # store fp16 on disk

        # append to shard buffer
        for row in logp:
            current.append(row)
            if len(current) == shard_size:
                shard = torch.stack(current, dim=0).contiguous()  # [shard, vocab]
                torch.save(shard, os.path.join(d, f"shard_{shard_idx:05d}.pt"))
                shard_idx += 1
                current = []

    if len(current) > 0:
        shard = torch.stack(current, dim=0).contiguous()
        torch.save(shard, os.path.join(d, f"shard_{shard_idx:05d}.pt"))
        shard_idx += 1

    # meta
    vocab = int(logp.shape[-1]) if len(prompts) > 0 else int(teacher_model.cfg.d_vocab)
    meta = TeacherMeta(n=len(prompts), vocab=vocab, shard_size=shard_size, dtype="float16")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, indent=2)

    logger.info(f"[teacher] done: {d} (shards={shard_idx})")


class TeacherShardReader:
    """
    Random access reader for teacher logp stored in shards on disk.
    Keeps a small in-memory CPU cache of recently used shard tensors.
    """

    def __init__(self, base_dir: str, task: str, split: str, variant: str, max_cached_shards: int = 2):
        self.dir = _teacher_dir(base_dir, task, split, variant)
        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Teacher meta not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.n = int(meta["n"])
        self.vocab = int(meta["vocab"])
        self.shard_size = int(meta["shard_size"])
        self.max_cached_shards = max_cached_shards

        self._cache: Dict[int, torch.Tensor] = {}  # shard_id -> tensor [shard, vocab] on CPU
        self._lru: List[int] = []

    def _load_shard(self, shard_id: int) -> torch.Tensor:
        if shard_id in self._cache:
            # refresh LRU
            if shard_id in self._lru:
                self._lru.remove(shard_id)
            self._lru.append(shard_id)
            return self._cache[shard_id]

        path = os.path.join(self.dir, f"shard_{shard_id:05d}.pt")
        shard = torch.load(path, map_location="cpu")  # float16 CPU

        self._cache[shard_id] = shard
        self._lru.append(shard_id)

        # evict if needed
        while len(self._lru) > self.max_cached_shards:
            evict = self._lru.pop(0)
            if evict in self._cache:
                del self._cache[evict]
        return shard

    def get(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        indices: [batch] dataset indices (0..n-1)
        Returns float32 tensor on device: [batch, vocab]
        """
        idx_cpu = indices.detach().cpu().long()
        out = torch.empty((idx_cpu.numel(), self.vocab), dtype=torch.float32, device=device)

        shard_ids = (idx_cpu // self.shard_size).tolist()
        in_shard = (idx_cpu % self.shard_size).tolist()

        # group by shard
        by_shard: Dict[int, List[Tuple[int, int]]] = {}
        for out_i, (sid, off) in enumerate(zip(shard_ids, in_shard)):
            by_shard.setdefault(sid, []).append((out_i, off))

        for sid, pairs in by_shard.items():
            shard = self._load_shard(sid)  # [shard_size, vocab] float16 CPU
            rows = torch.stack([shard[off] for (_, off) in pairs], dim=0)  # float16 CPU
            rows = rows.to(device=device, dtype=torch.float32)
            for j, (out_i, _) in enumerate(pairs):
                out[out_i] = rows[j]

        return out
