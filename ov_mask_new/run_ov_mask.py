#!/usr/bin/env python3
"""
OV-mask (E1 Part A–C, OV-only) for GPT-2 small ("gpt2") using HuggingFace Transformers.

Pipeline per task (IOI / GT / GP):
A) Cache (per split): baseline clean logits @ target, corrupt nu(z,1) @ target per layer/head
B) Per-head SVD of W_aug=[W_O; b_share] and choose minimal rank r such that trunc-only KL<=1e-6 (calib subset)
C) Learn mask m=sigmoid(mask_logits) over kept components; mix clean/corrupt in SVD space; minimize KL + L1(m)

Implementation notes:
- We hook each layer's attn.c_proj (output projection). In HF GPT-2 this is Conv1D (y = x @ W + b).
- PyTorch forward hooks can return a modified output tensor.
- We "replace per-head OV writes" by adding summed per-head deltas at the layer attention output (post c_proj) at target_pos.
"""

from __future__ import annotations

import argparse, json, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ------------------------- Config -------------------------

@dataclass
class Cfg:
    model_name: str = "gpt2"
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_head: int = 64
    d_aug: int = 65

    batch_size: int = 64
    lr: float = 1e-2
    weight_decay: float = 1e-9
    l1_weight: float = 6e-5
    max_epochs: int = 80
    early_stop_patience: int = 12
    active_threshold: float = 5e-2
    # active_threshold: float = 0.5

    trunc_kl_eps: float = 1e-4
    calib_take: int = 64          # how many train examples to use for truncation
    calib_batch: int = 32         # batch size for truncation eval

    seed: int = 0
    cache_dtype: torch.dtype = torch.float16
    compute_dtype: torch.dtype = torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)

def kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(-1)


# ------------------------- Task adapters -------------------------

def _trail_space(s: str) -> str:
    return str(s).rstrip() + " "

def load_examples(task: str, csv_path: Path) -> List[Dict]:
    df = pd.read_csv(csv_path)
    cols = set(df.columns)

    # Already-normalized format
    if {"prompt_clean", "prompt_corrupt"}.issubset(cols):
        out = []
        for _, r in df.iterrows():
            ex = {"prompt_clean": _trail_space(r["prompt_clean"]),
                  "prompt_corrupt": _trail_space(r["prompt_corrupt"])}
            for k in ["label_clean","label_corrupt"]:
                if k in cols: ex[k] = str(r[k])
            out.append(ex)
        return out

    task = task.lower()
    out = []

    if task == "ioi":
        req = {"ioi_sentences_input","corr_ioi_sentences_input","ioi_sentences_labels","corr_ioi_sentences_labels"}
        if not req.issubset(cols): raise ValueError(f"IOI csv missing {req}, got {sorted(cols)}")
        for _, r in df.iterrows():
            out.append({
                "prompt_clean": _trail_space(r["ioi_sentences_input"]),
                "prompt_corrupt": _trail_space(r["corr_ioi_sentences_input"]),
                "label_clean": str(r["ioi_sentences_labels"]),
                "label_corrupt": str(r["corr_ioi_sentences_labels"]),
            })
        return out

    if task == "gp":
        req = {"prefix","pronoun","corr_prefix","corr_pronoun"}
        if not req.issubset(cols): raise ValueError(f"GP csv missing {req}, got {sorted(cols)}")
        def strip_last(text: str, last: str) -> str:
            t = str(text).rstrip()
            last = str(last).strip()
            if t.lower().endswith(" " + last.lower()):
                return t[:-(len(last)+1)]
            parts = t.split()
            return " ".join(parts[:-1]) if len(parts) else t
        for _, r in df.iterrows():
            out.append({
                "prompt_clean": _trail_space(strip_last(r["prefix"], r["pronoun"])),
                "prompt_corrupt": _trail_space(strip_last(r["corr_prefix"], r["corr_pronoun"])),
                "label_clean": str(r["pronoun"]).strip(),
                "label_corrupt": str(r["corr_pronoun"]).strip(),
            })
        return out

    if task == "gt":
        req = {"prefix","century","corr_prefix","corr_century"}
        if not req.issubset(cols): raise ValueError(f"GT csv missing {req}, got {sorted(cols)}")
        def strip_last_tok(text: str) -> str:
            t = str(text).rstrip()
            parts = t.split()
            return " ".join(parts[:-1]) if len(parts) else t
        for _, r in df.iterrows():
            out.append({
                "prompt_clean": _trail_space(strip_last_tok(r["prefix"])),
                "prompt_corrupt": _trail_space(strip_last_tok(r["corr_prefix"])),
                "label_clean": str(r["century"]).strip(),
                "label_corrupt": str(r["corr_century"]).strip(),
            })
        return out

    raise ValueError("task must be one of: ioi, gt, gp")


# ------------------------- Hooks -------------------------

class CProjInputCacher:
    """Caches per-layer z (concat->heads) at target_pos from each layer's attn.c_proj input."""
    def __init__(self, model: GPT2LMHeadModel, cfg: Cfg, target_pos: torch.Tensor):
        self.model, self.cfg, self.target_pos = model, cfg, target_pos
        self.handles = []
        self.z_by_layer: Dict[int, torch.Tensor] = {}

    def _make(self, layer: int):
        cfg = self.cfg
        def pre_hook(_mod, inputs):
            x = inputs[0]  # [B,L,d_model]
            B, L, _ = x.shape
            z = x.view(B, L, cfg.n_heads, cfg.d_head)
            idx = torch.arange(B, device=x.device)
            pos = self.target_pos.to(x.device)
            self.z_by_layer[layer] = z[idx, pos].detach()  # [B,n_heads,64]
        return pre_hook

    def __enter__(self):
        for l in range(self.cfg.n_layers):
            self.handles.append(self.model.transformer.h[l].attn.c_proj.register_forward_pre_hook(self._make(l)))
        return self

    def __exit__(self, *args):
        for h in self.handles: h.remove()
        self.handles = []


class OVIntervention:
    """
    Forward hook on each layer's attn.c_proj that adjusts output at target_pos:
      out[target] += sum_h (y_mod - y_full)
    """
    def __init__(self, model: GPT2LMHeadModel, cfg: Cfg,
                 basis: Dict[Tuple[int,int], Dict[str, torch.Tensor]],
                 W_blocks: Dict[Tuple[int,int], Dict[str, torch.Tensor]],
                 mask_logits: nn.ParameterDict):
        self.model, self.cfg = model, cfg
        self.basis, self.W_blocks, self.mask_logits = basis, W_blocks, mask_logits
        self.handles = []
        self.target_pos: Optional[torch.Tensor] = None
        self.nu_corrupt_layers: Optional[torch.Tensor] = None  # [B, n_layers, n_heads, 65]
        self.mode: str = "masked"  # or "trunc_only"
        self.only: Optional[Tuple[int,int]] = None  # (layer, head) for truncation calibration

    def _make(self, layer: int):
        cfg = self.cfg
        def hook(_mod, inputs, output):
            if self.target_pos is None or self.nu_corrupt_layers is None: return output
            x = inputs[0]  # [B,L,768]
            B, L, _ = x.shape
            idx = torch.arange(B, device=x.device)
            pos = self.target_pos.to(x.device)

            z_t = x.view(B, L, cfg.n_heads, cfg.d_head)[idx, pos]  # [B,n_heads,64]
            ones = torch.ones((B, cfg.n_heads, 1), device=z_t.device, dtype=z_t.dtype)
            nu_clean = torch.cat([z_t, ones], dim=-1)  # [B,n_heads,65]
            nu_corr = self.nu_corrupt_layers[:, layer].to(z_t.device)  # [B,n_heads,65]

            delta = torch.zeros((B, cfg.d_model), device=z_t.device, dtype=output.dtype)
            for h in range(cfg.n_heads):
                key = (layer, h)
                if self.only is not None and key != self.only:
                    continue

                W = self.W_blocks[key]["W"].to(z_t.device)          # [64,768]
                b = self.W_blocks[key]["b_share"].to(z_t.device)    # [768]
                y_full = z_t[:, h] @ W + b                          # [B,768]

                U = self.basis[key]["U"].to(z_t.device)             # [65,r]
                S = self.basis[key]["S"].to(z_t.device)             # [r]
                V = self.basis[key]["V"].to(z_t.device)             # [768,r]

                a_clean = nu_clean[:, h].to(cfg.compute_dtype) @ U.to(cfg.compute_dtype)
                if self.mode == "trunc_only":
                    a_mix = a_clean
                else:
                    a_corr = nu_corr[:, h].to(cfg.compute_dtype) @ U.to(cfg.compute_dtype)
                    m = torch.sigmoid(self.mask_logits[f"l{layer}_h{h}"]).to(cfg.compute_dtype)
                    a_mix = m * a_clean + (1 - m) * a_corr

                y_mod = (a_mix * S.to(cfg.compute_dtype)) @ V.to(cfg.compute_dtype).T
                y_mod = y_mod.to(output.dtype)
                delta = delta + (y_mod - y_full)

            out = output.clone()
            out[idx, pos] = out[idx, pos] + delta
            return out
        return hook

    def __enter__(self):
        for l in range(self.cfg.n_layers):
            self.handles.append(self.model.transformer.h[l].attn.c_proj.register_forward_hook(self._make(l)))
        return self

    def __exit__(self, *args):
        for h in self.handles: h.remove()
        self.handles = []


# ------------------------- Part A: caches -------------------------

def build_cache(cfg: Cfg, model: GPT2LMHeadModel, tok: GPT2TokenizerFast,
                examples: List[Dict], out_path: Path, device: torch.device) -> None:
    ensure_dir(out_path.parent)
    prompts_clean = [e["prompt_clean"] for e in examples]
    prompts_corr  = [e["prompt_corrupt"] for e in examples]

    enc_c = tok(prompts_clean, return_tensors="pt", padding=True, truncation=True)
    enc_k = tok(prompts_corr,  return_tensors="pt", padding=True, truncation=True)

    toks_c = enc_c["input_ids"].long()
    tpos_c = enc_c["attention_mask"].sum(1).long() - 1

    toks_k = enc_k["input_ids"].long()
    tpos_k = enc_k["attention_mask"].sum(1).long() - 1

    N = toks_c.shape[0]
    idxs = torch.arange(N)

    logits_list, nu_list = [], []
    model.eval()

    for start in tqdm(range(0, N, cfg.batch_size), desc=f"cache {out_path.stem}"):
        b = idxs[start:start+cfg.batch_size]
        inp_c, tp_c = toks_c[b].to(device), tpos_c[b].to(device)
        inp_k, tp_k = toks_k[b].to(device), tpos_k[b].to(device)

        with torch.no_grad():
            lc = model(inp_c).logits
            lt = lc[torch.arange(inp_c.size(0), device=device), tp_c]
            logits_list.append(lt.to(cfg.cache_dtype).cpu())

        with torch.no_grad():
            with CProjInputCacher(model, cfg, tp_k) as cacher:
                _ = model(inp_k).logits
                nus = []
                for l in range(cfg.n_layers):
                    z = cacher.z_by_layer[l]  # [B,n_heads,64]
                    ones = torch.ones((z.size(0), cfg.n_heads, 1), device=z.device, dtype=z.dtype)
                    nus.append(torch.cat([z, ones], -1))
                nu_list.append(torch.stack(nus, 1).to(cfg.cache_dtype).cpu())

    torch.save({
        "tokens_clean": toks_c,
        "target_pos": tpos_c,
        "logits_base_at_target": torch.cat(logits_list, 0),
        "nu_corrupt_at_target": torch.cat(nu_list, 0),
        "meta": {"created_at": now(), "n": int(N)},
    }, out_path)


# ------------------------- Part B: SVD + truncation -------------------------

def extract_W_blocks(cfg: Cfg, model: GPT2LMHeadModel) -> Dict[Tuple[int,int], Dict[str, torch.Tensor]]:
    """
    For each (layer, head): W_block = c_proj.weight[h*d_head:(h+1)*d_head, :]  and b_share=b/n_heads.
    Conv1D uses weight shaped [in_dim, out_dim] and forward is x@W + b.
    """
    out = {}
    for l in range(cfg.n_layers):
        attn = model.transformer.h[l].attn
        W = attn.c_proj.weight.detach().float().cpu()   # [768,768] (in,out)
        b = attn.c_proj.bias.detach().float().cpu()
        b_share = b / cfg.n_heads
        for h in range(cfg.n_heads):
            out[(l,h)] = {"W": W[h*cfg.d_head:(h+1)*cfg.d_head].contiguous(),
                          "b_share": b_share.contiguous()}
    return out

def load_basis_from_dir(cfg: Cfg, svd_dir: Path) -> Dict[Tuple[int,int], Dict[str, torch.Tensor]]:
    basis = {}
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            obj = torch.load(svd_dir / f"layer{l:02d}_head{h:02d}.pt", map_location="cpu")
            basis[(l,h)] = {"U": obj["U_keep"], "S": obj["S_keep"], "V": obj["V_keep"]}
    return basis

def choose_ranks(cfg: Cfg, model: GPT2LMHeadModel, device: torch.device,
                 W_blocks: Dict[Tuple[int,int], Dict[str, torch.Tensor]],
                 calib_tokens: torch.Tensor, calib_tpos: torch.Tensor, calib_base_logits: torch.Tensor,
                 svd_dir: Path) -> Dict[Tuple[int,int], Dict[str, torch.Tensor]]:

    ensure_dir(svd_dir)
    p_base = probs(calib_base_logits.float()).cpu()

    class CalibDS(torch.utils.data.Dataset):
        def __len__(self): return calib_tokens.size(0)
        def __getitem__(self, i): return {"x": calib_tokens[i], "t": calib_tpos[i], "i": i}

    dl = torch.utils.data.DataLoader(CalibDS(), batch_size=cfg.calib_batch, shuffle=False, num_workers=0)

    def eval_rank(layer: int, head: int, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, r: int) -> float:
        key = (layer, head)
        basis = {key: {"U": U[:, :r], "S": S[:r], "V": V[:, :r]}}
        mask = nn.ParameterDict({f"l{layer}_h{head}": nn.Parameter(torch.zeros((r,), device=device))})

        model.eval()
        with OVIntervention(model, cfg, basis, W_blocks, mask) as interv:
            interv.mode = "trunc_only"
            interv.only = key
            kls = []
            for batch in dl:
                x = batch["x"].to(device)
                t = batch["t"].to(device)
                idxs = batch["i"].long()
                interv.target_pos = t
                interv.nu_corrupt_layers = torch.zeros((x.size(0), cfg.n_layers, cfg.n_heads, cfg.d_aug), device=device)
                with torch.no_grad():
                    logits = model(x).logits
                    lt = logits[torch.arange(x.size(0), device=device), t].cpu()
                kls.append(float(kl(p_base[idxs], probs(lt)).mean()))
            return float(np.mean(kls))

    basis_out = {}
    for l in tqdm(range(cfg.n_layers), desc="SVD layers"):
        for h in range(cfg.n_heads):
            W = W_blocks[(l,h)]["W"]
            b = W_blocks[(l,h)]["b_share"]
            W_aug = torch.cat([W, b.unsqueeze(0)], 0)  # [65,768]
            U, S, Vh = torch.linalg.svd(W_aug, full_matrices=True)
            V = Vh.T[:, :cfg.d_aug]  # [768,65]

            lo, hi, best = 1, cfg.d_aug, cfg.d_aug
            while lo <= hi:
                mid = (lo + hi) // 2
                if eval_rank(l, h, U, S, V, mid) <= cfg.trunc_kl_eps:
                    best = mid; hi = mid - 1
                else:
                    lo = mid + 1

            obj = {"U_keep": U[:, :best].contiguous(),
                   "S_keep": S[:best].contiguous(),
                   "V_keep": V[:, :best].contiguous(),
                   "rank": best,
                   "keep_idx": list(range(best)),
                   "drop_idx": list(range(best, cfg.d_aug))}
            torch.save(obj, svd_dir / f"layer{l:02d}_head{h:02d}.pt")
            basis_out[(l,h)] = {"U": obj["U_keep"], "S": obj["S_keep"], "V": obj["V_keep"]}
    return basis_out


# ------------------------- Part C: training -------------------------

def init_masks(cfg: Cfg, basis: Dict[Tuple[int,int], Dict[str, torch.Tensor]], device: torch.device) -> nn.ParameterDict:
    pdict = nn.ParameterDict()
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            r = int(basis[(l,h)]["S"].numel())
            pdict[f"l{l}_h{h}"] = nn.Parameter(torch.zeros((r,), device=device))
    return pdict

def sparsity(cfg: Cfg, mask_logits: nn.ParameterDict) -> Dict[str, float]:
    m_all = torch.cat([torch.sigmoid(p).detach().flatten().cpu() for p in mask_logits.values()])
    n_active = int((m_all > cfg.active_threshold).sum())
    n_learnable = int(m_all.numel())
    S_rel = 1.0 - n_active / max(1, n_learnable)
    S_full = 1.0 - n_active / float(cfg.n_layers * cfg.n_heads * cfg.d_aug)
    return {"n_active": n_active, "n_learnable": n_learnable, "S_rel": float(S_rel), "S_full": float(S_full)}

def train(cfg: Cfg, model: GPT2LMHeadModel, device: torch.device,
          basis: Dict[Tuple[int,int], Dict[str, torch.Tensor]],
          W_blocks: Dict[Tuple[int,int], Dict[str, torch.Tensor]],
          train_cache: dict, val_cache: dict, out_dir: Path) -> nn.ParameterDict:

    ensure_dir(out_dir / "plots"); ensure_dir(out_dir / "masks")
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists(): metrics_path.unlink()

    mask_logits = init_masks(cfg, basis, device)
    opt = torch.optim.AdamW(mask_logits.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def make_dl(cache: dict, shuffle: bool):
        class DS(torch.utils.data.Dataset):
            def __len__(self): return cache["tokens_clean"].size(0)
            def __getitem__(self, i):
                return (cache["tokens_clean"][i].long(),
                        cache["target_pos"][i].long(),
                        cache["logits_base_at_target"][i],
                        cache["nu_corrupt_at_target"][i])
        def collate(batch):
            x, t, lb, nu = zip(*batch)
            return (torch.stack(x), torch.stack(t), torch.stack(lb), torch.stack(nu))
        return torch.utils.data.DataLoader(DS(), batch_size=cfg.batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate)

    train_dl = make_dl(train_cache, True)
    val_dl = make_dl(val_cache, False)

    for p in model.parameters(): p.requires_grad_(False)
    model.eval()

    best_val, bad = float("inf"), 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    tr_curve, va_curve, act_curve = [], [], []

    with OVIntervention(model, cfg, basis, W_blocks, mask_logits) as interv:
        for epoch in range(1, cfg.max_epochs + 1):
            # train
            mask_logits.train()
            kls = []
            for x, t, lb, nu in train_dl:
                x, t = x.to(device), t.to(device)
                lb = lb.to(device).float()
                nu = nu.to(device).float()

                interv.target_pos = t
                interv.nu_corrupt_layers = nu
                interv.mode = "masked"
                interv.only = None

                logits = model(x).logits
                lt = logits[torch.arange(x.size(0), device=device), t]
                loss_kl = kl(probs(lb), probs(lt)).mean()
                m_all = torch.cat([torch.sigmoid(p).reshape(-1) for p in mask_logits.values()])
                # loss = loss_kl + cfg.l1_weight * m_all.mean()
                loss = loss_kl + cfg.l1_weight * m_all.sum()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                kls.append(float(loss_kl.detach().cpu()))
            tr = float(np.mean(kls))

            # val
            mask_logits.eval()
            with torch.no_grad():
                kls = []
                for x, t, lb, nu in val_dl:
                    x, t = x.to(device), t.to(device)
                    lb = lb.to(device).float()
                    nu = nu.to(device).float()
                    interv.target_pos = t
                    interv.nu_corrupt_layers = nu
                    interv.mode = "masked"
                    interv.only = None
                    logits = model(x).logits
                    lt = logits[torch.arange(x.size(0), device=device), t]
                    kls.append(float(kl(probs(lb), probs(lt)).mean().cpu()))
                va = float(np.mean(kls))

            sp = sparsity(cfg, mask_logits)
            tr_curve.append(tr); va_curve.append(va); act_curve.append(sp["n_active"])

            rec = {"time": now(), "epoch": epoch, "train_kl": tr, "val_kl": va, **sp}
            with open(metrics_path, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
            print(f"[{epoch:03d}] train_KL={tr:.4e} val_KL={va:.4e} n_active={sp['n_active']} S_rel={sp['S_rel']:.3f}")

            if va < best_val - 1e-12:
                best_val, bad = va, 0
                best_state = {k: v.detach().clone().cpu() for k, v in mask_logits.items()}
            else:
                bad += 1
                if bad >= cfg.early_stop_patience:
                    print("Early stopping."); break

    if best_state is not None:
        for k in mask_logits.keys(): mask_logits[k].data.copy_(best_state[k].to(device))

    # save masks
    with torch.no_grad():
        for l in range(cfg.n_layers):
            for h in range(cfg.n_heads):
                m = torch.sigmoid(mask_logits[f"l{l}_h{h}"]).cpu()
                torch.save({"m": m}, out_dir / "masks" / f"layer{l:02d}_head{h:02d}.pt")

    # plots
    def save_plot(y1, y2, path, yscale=None, labels=("train","val"), ylabel=""):
        plt.figure()
        plt.plot(range(1,len(y1)+1), y1, label=labels[0])
        if y2 is not None: plt.plot(range(1,len(y2)+1), y2, label=labels[1])
        if yscale: plt.yscale(yscale)
        plt.xlabel("epoch"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
        plt.savefig(path, dpi=200); plt.close()

    save_plot(tr_curve, va_curve, out_dir/"plots/kl_curves.png", yscale="log", ylabel="KL(base||masked)")
    save_plot(act_curve, None, out_dir/"plots/n_active.png", ylabel="n_active")

    with torch.no_grad():
        m_all = torch.cat([torch.sigmoid(p).cpu().flatten() for p in mask_logits.values()]).numpy()
    plt.figure(); plt.hist(m_all, bins=50); plt.xlabel("m"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(out_dir/"plots/m_hist.png", dpi=200); plt.close()

    return mask_logits


# ------------------------- Receptors -------------------------

def receptors(cfg: Cfg, model: GPT2LMHeadModel, tok: GPT2TokenizerFast,
              basis: Dict[Tuple[int,int], Dict[str, torch.Tensor]], out_dir: Path, top_k: int = 20) -> None:
    rec_dir = out_dir / "receptors"; ensure_dir(rec_dir)
    W_U = model.lm_head.weight.detach().float().cpu().T  # [768,V]
    for l in tqdm(range(cfg.n_layers), desc="receptors"):
        for h in range(cfg.n_heads):
            V = basis[(l,h)]["V"].float().cpu()  # [768,r]
            comps = []
            for k in range(V.size(1)):
                scores = (V[:,k].unsqueeze(0) @ W_U).squeeze(0)
                topv, topi = torch.topk(scores, top_k)
                botv, boti = torch.topk(-scores, top_k)
                comps.append({
                    "k": k,
                    "top": [{"tok": tok.decode([int(i)]), "score": float(s)} for i,s in zip(topi, topv)],
                    "bottom": [{"tok": tok.decode([int(i)]), "score": float(-s)} for i,s in zip(boti, botv)],
                })
            with open(rec_dir / f"layer{l:02d}_head{h:02d}.json", "w", encoding="utf-8") as f:
                json.dump({"layer": l, "head": h, "rank": int(V.size(1)), "components": comps}, f, ensure_ascii=False, indent=2)


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ioi","gt","gp"])
    ap.add_argument("--data_dir", default="data_main")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_root", default="outputs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stage", default="all", choices=["all","a","b","c","receptors"])
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Cfg(seed=args.seed)
    if args.max_epochs is not None: cfg.max_epochs = args.max_epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    set_seed(cfg.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_root) / args.task
    cache_dir = out_dir / "cache"; svd_dir = out_dir / "svd"
    ensure_dir(cache_dir); ensure_dir(svd_dir)

    tok = GPT2TokenizerFast.from_pretrained(cfg.model_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = torch.device(args.device)

    model = GPT2LMHeadModel.from_pretrained(cfg.model_name).to(device).eval()

    train_ex = load_examples(args.task, data_dir/args.train_csv)
    val_ex   = load_examples(args.task, data_dir/args.val_csv)
    test_ex  = load_examples(args.task, data_dir/args.test_csv)

    train_cache_p = cache_dir/"train.pt"
    val_cache_p   = cache_dir/"val.pt"
    test_cache_p  = cache_dir/"test.pt"

    if args.stage in ("all","a"):
        build_cache(cfg, model, tok, train_ex, train_cache_p, device)
        build_cache(cfg, model, tok, val_ex,   val_cache_p,   device)
        build_cache(cfg, model, tok, test_ex,  test_cache_p,  device)

    train_cache = torch.load(train_cache_p, map_location="cpu")
    val_cache   = torch.load(val_cache_p,   map_location="cpu")

    W_blocks = extract_W_blocks(cfg, model)

    basis: Optional[Dict[Tuple[int,int], Dict[str, torch.Tensor]]] = None
    if args.stage in ("all","b"):
        n = min(cfg.calib_take, int(train_cache["tokens_clean"].size(0)))
        basis = choose_ranks(cfg, model, device, W_blocks,
                             train_cache["tokens_clean"][:n],
                             train_cache["target_pos"][:n],
                             train_cache["logits_base_at_target"][:n],
                             svd_dir)

    if basis is None:
        basis = load_basis_from_dir(cfg, svd_dir)

    if args.stage in ("all","c"):
        _ = train(cfg, model, device, basis, W_blocks, train_cache, val_cache, out_dir)

    if args.stage in ("all","receptors"):
        receptors(cfg, model, tok, basis, out_dir)

    print("Done. Outputs:", out_dir)

if __name__ == "__main__":
    main()
