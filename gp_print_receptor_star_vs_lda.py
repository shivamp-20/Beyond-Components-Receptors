#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Print STAR vs LDA composite OV logit receptor token lists + 'mass' diagnostics.

Reads:
  - outputs/gp/svd_cache.pt
  - outputs/gp/masks.pt
  - Exp2 results JSON containing idx_list + w (LDA weights)

Prints:
  - Top / bottom tokens (default 25) for STAR and LDA receptors
  - Count of gender tokens in top/bottom
  - 'Mass' metrics using |receptor| (receptor is not a probability distribution)
  - Score + rank of ' he' and ' she'
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer


def _safe_decode(tok: GPT2TokenizerFast, tid: int) -> str:
    try:
        return tok.decode([int(tid)], clean_up_tokenization_spaces=False)
    except TypeError:
        return tok.decode([int(tid)])


def _topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), int(x.numel()))
    return torch.topk(x, k=k, largest=True).indices


def _bottomk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), int(x.numel()))
    return torch.topk(x, k=k, largest=False).indices


def _rank_of_token(scores: torch.Tensor, token_id: int, descending: bool = True) -> int:
    token_id = int(token_id)
    if descending:
        return int((scores > scores[token_id]).sum().item()) + 1
    return int((scores < scores[token_id]).sum().item()) + 1


def _parse_csvish_list(s: str) -> List[str]:
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _make_token_ids(tok: GPT2TokenizerFast, strs: List[str], strict_single_token: bool) -> List[int]:
    out: List[int] = []
    for p in strs:
        ids = tok.encode(p, add_special_tokens=False)
        if strict_single_token and len(ids) != 1:
            raise ValueError(f"String {p!r} is not a single token: ids={ids}")
        if len(ids) == 1:
            out.append(int(ids[0]))
    return out


def _format_token_table(
    tok: GPT2TokenizerFast, scores: torch.Tensor, ids: List[int], gender_set: set
) -> List[Tuple[str, float, str]]:
    rows = []
    for tid in ids:
        tstr = _safe_decode(tok, tid)
        sc = float(scores[tid].item())
        tag = "G" if tid in gender_set else "-"
        rows.append((repr(tstr), sc, tag))
    return rows


def _mass_metrics(scores: torch.Tensor, top_ids: List[int], bottom_ids: List[int], gender_set: set) -> Dict[str, float]:
    abs_scores = scores.abs()
    total_abs = float(abs_scores.sum().item()) + 1e-30

    top_abs = float(abs_scores[top_ids].sum().item())
    bottom_abs = float(abs_scores[bottom_ids].sum().item())

    gender_abs_total = float(abs_scores[list(gender_set)].sum().item()) if gender_set else 0.0

    top_gender_ids = [i for i in top_ids if i in gender_set]
    bottom_gender_ids = [i for i in bottom_ids if i in gender_set]

    top_gender_abs = float(abs_scores[top_gender_ids].sum().item()) if top_gender_ids else 0.0
    bottom_gender_abs = float(abs_scores[bottom_gender_ids].sum().item()) if bottom_gender_ids else 0.0

    return {
        "mass_topK": top_abs / total_abs,
        "mass_bottomK": bottom_abs / total_abs,
        "gender_mass_total": gender_abs_total / total_abs,
        "gender_mass_within_topK": top_gender_abs / (top_abs + 1e-30),
        "gender_mass_within_bottomK": bottom_gender_abs / (bottom_abs + 1e-30),
        "gender_mass_in_topK_over_total": top_gender_abs / total_abs,
        "gender_mass_in_bottomK_over_total": bottom_gender_abs / total_abs,
    }


def _count_gender(ids: List[int], gender_set: set) -> int:
    return sum(1 for t in ids if t in gender_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="outputs/gp")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--i_star", type=int, default=0)
    ap.add_argument("--exp2_results_json", type=str, required=True)
    ap.add_argument("--topK", type=int, default=25)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--male_strings", type=str, default=" he,He, his,His, him,Him, himself,Himself")
    ap.add_argument("--female_strings", type=str, default=" she,She, her,Her, hers,Hers, herself,Herself")
    ap.add_argument("--strict_single_token_gender", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    svd_path = out_dir / "svd_cache.pt"
    masks_path = out_dir / "masks.pt"
    if not svd_path.exists():
        raise FileNotFoundError(f"Missing: {svd_path}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Missing: {masks_path}")

    # tokenizer + pronoun ids
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    he_ids = tok.encode(" he", add_special_tokens=False)
    she_ids = tok.encode(" she", add_special_tokens=False)
    if len(he_ids) != 1 or len(she_ids) != 1:
        raise ValueError(f"' he'/' she' not single tokens: he={he_ids}, she={she_ids}")
    he_id = int(he_ids[0])
    she_id = int(she_ids[0])

    male_ids = _make_token_ids(tok, _parse_csvish_list(args.male_strings), bool(args.strict_single_token_gender))
    female_ids = _make_token_ids(tok, _parse_csvish_list(args.female_strings), bool(args.strict_single_token_gender))
    gender_set = set(male_ids) | set(female_ids) | {he_id, she_id}

    # model (need W_U)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    model = HookedTransformer.from_pretrained("gpt2-small", device=args.device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # gp loader (ddp or non-ddp)
    try:
        import train_gp_masks_and_dump_ov_logit_receptors_ddp as gp  # type: ignore
    except Exception:
        import train_gp_masks_and_dump_ov_logit_receptors as gp  # type: ignore

    _qk, ov, _mlp_in, _mlp_out, _rank_total_ov = gp.load_svd_cache(str(svd_path), device=args.device)
    masks_dict = torch.load(str(masks_path), map_location="cpu")
    mask_vec = masks_dict["ov"][args.layer][args.head].detach().float().cpu()

    svd = ov[args.layer][args.head]

    # exp2 json -> idx_list + w
    exp2_path = Path(args.exp2_results_json)
    if not exp2_path.exists():
        raise FileNotFoundError(f"Missing: {exp2_path}")
    with exp2_path.open("r", encoding="utf-8") as f:
        exp2 = json.load(f)

    idx_list = exp2.get("idx_list") or exp2.get("config", {}).get("idx_list")
    w = exp2.get("w") or exp2.get("w_vec") or exp2.get("fit", {}).get("w")
    if idx_list is None or w is None:
        raise ValueError("exp2_results_json must contain idx_list and w (check keys in your JSON).")

    idx_list = [int(x) for x in idx_list]
    w = torch.tensor([float(x) for x in w], device=args.device, dtype=torch.float32)
    if w.numel() != len(idx_list):
        raise ValueError(f"w has len {w.numel()} but idx_list has len {len(idx_list)}")

    # SVD pieces
    V_sel = torch.stack([svd.Vh[j, :] for j in idx_list], dim=0).to(device=args.device, dtype=torch.float32)
    v_star = svd.Vh[int(args.i_star), :].to(device=args.device, dtype=torch.float32)
    v_lda = (w @ V_sel).to(device=args.device, dtype=torch.float32)

    # receptors
    W_U = model.W_U.to(dtype=torch.float32)
    rec_star = (v_star @ W_U).detach().cpu()
    rec_lda = (v_lda @ W_U).detach().cpu()

    def print_block(name: str, scores: torch.Tensor) -> Dict[str, object]:
        top_ids = _topk_indices(scores, args.topK).tolist()
        bottom_ids = _bottomk_indices(scores, args.topK).tolist()

        print("\n" + "=" * 100)
        print(name)
        print("=" * 100)
        print(f'[TOKENS] " he"={he_id}  " she"={she_id}')
        print(f"[SCORE] he={float(scores[he_id].item()):.6g}  she={float(scores[she_id].item()):.6g}")
        print(f"[RANK ] he={_rank_of_token(scores, he_id, True)}  she={_rank_of_token(scores, she_id, True)}")

        print(f"[TOP{args.topK}] gender_count={_count_gender(top_ids, gender_set)}/{args.topK}")
        for tstr, sc, tag in _format_token_table(tok, scores, top_ids, gender_set):
            print(f"  {tstr:>16s}  {sc: .6g}  {tag}")

        print(f"[BOT{args.topK}] gender_count={_count_gender(bottom_ids, gender_set)}/{args.topK}")
        for tstr, sc, tag in _format_token_table(tok, scores, bottom_ids, gender_set):
            print(f"  {tstr:>16s}  {sc: .6g}  {tag}")

        mm = _mass_metrics(scores, top_ids, bottom_ids, gender_set)
        print("[MASS | abs(score)]")
        for k, v in mm.items():
            print(f"  {k:28s} = {v:.6g}")

        return {"top_ids": top_ids, "bottom_ids": bottom_ids, "mass": mm}

    # config header
    print("\n[CONFIG]")
    print(f"layer={args.layer} head={args.head} i_star={args.i_star}")
    print(f"exp2_results_json={str(exp2_path)}")
    print(f"k={len(idx_list)}  idx_list={idx_list}")
    print("idx_list details:")
    for j in idx_list:
        print(f"  idx={j:4d}  mask={float(mask_vec[j].item()):.6g}  sigma={float(svd.S[j].detach().cpu().item()):.6g}")
    print(f"w summary: max|w|={float(w.abs().max().item()):.6g}")

    star_out = print_block("STAR receptor = Vh[i_star] @ W_U", rec_star)
    lda_out = print_block("LDA receptor = (w @ V_sel) @ W_U", rec_lda)

    # comparison
    print("\n" + "-" * 100)
    print("[COMPARISON] LDA - STAR (mass deltas)")
    star_mass = star_out["mass"]
    lda_mass = lda_out["mass"]
    for key in star_mass.keys():
        dv = float(lda_mass[key]) - float(star_mass[key])
        print(f"  Δ {key:25s} = {dv:+.6g}")

    print("\n[INTERPRET]")
    print("If these are higher for LDA than STAR, LDA emphasizes gender tokens more:")
    print("  - gender_mass_within_topK")
    print("  - gender_mass_in_topK_over_total")


if __name__ == "__main__":
    main()
