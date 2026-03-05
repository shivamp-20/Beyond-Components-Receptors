#!/usr/bin/env python3
# CIVET_FEASIBILITY_TEST1 — VERSION 4 — 2025-03-05
# If you see this line printed below, you have the correct file.
print(">>> SCRIPT VERSION: civet_feasibility_test1.py VERSION 4 (2025-03-05)")

"""
CIVET Feasibility Test 1: The Discrimination Test
===================================================
Tests whether conformal prediction can discriminate between correct and wrong
interpretations of SAE features. Run on Kaggle with a single T4 GPU.
"""

import sys
import time
import json
import gc
import warnings
import importlib
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def quantile_higher(arr, q):
    """np.quantile with method='higher', works across all numpy versions."""
    try:
        return float(np.quantile(arr, q, method="higher"))
    except TypeError:
        return float(np.quantile(arr, q, interpolation="higher"))


# ================================================================
print("=" * 80)
print("CIVET FEASIBILITY TEST 1: THE DISCRIMINATION TEST")
print("=" * 80)
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print()

# ================================================================
# STEP 0 — Discover and select an SAE
# ================================================================
print("=" * 80)
print("STEP 0: Discovering available SAEs in SAELens")
print("=" * 80)

import sae_lens
print(f"sae_lens version: {sae_lens.__version__}")

from sae_lens import SAE

# --- Try every known way to get the pretrained SAE directory ---
sae_directory = None

# Method 1-2: known function paths via importlib (no bare import at module level)
for method_path in [
    "sae_lens.pretrained_saes.get_pretrained_saes_directory",
    "sae_lens.get_pretrained_saes_directory",
]:
    parts = method_path.rsplit(".", 1)
    mod_path, func_name = parts[0], parts[1]
    try:
        mod = importlib.import_module(mod_path)
        func = getattr(mod, func_name)
        sae_directory = func()
        print(f"  Loaded via {method_path}()")
        break
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        print(f"  {method_path} -> {type(e).__name__}: {e}")

# Method 3: scan sae_lens top-level namespace
if sae_directory is None:
    print("  Scanning sae_lens namespace...")
    for attr_name in sorted(dir(sae_lens)):
        if any(kw in attr_name.lower() for kw in ["pretrained", "directory", "registry", "catalog"]):
            obj = getattr(sae_lens, attr_name)
            if callable(obj):
                try:
                    result = obj()
                    if isinstance(result, dict) and len(result) > 0:
                        sae_directory = result
                        print(f"  Found via sae_lens.{attr_name}() -> {len(result)} entries")
                        break
                except Exception:
                    pass

# Method 4: scan known submodules
if sae_directory is None:
    for submod_name in ["toolkit.pretrained_saes", "toolkit", "config", "utils"]:
        try:
            submod = importlib.import_module(f"sae_lens.{submod_name}")
            for attr_name in dir(submod):
                if "directory" in attr_name.lower() or "pretrained" in attr_name.lower():
                    obj = getattr(submod, attr_name)
                    if callable(obj):
                        try:
                            result = obj()
                            if isinstance(result, dict) and len(result) > 0:
                                sae_directory = result
                                print(f"  Found via sae_lens.{submod_name}.{attr_name}()")
                                break
                        except Exception:
                            pass
            if sae_directory is not None:
                break
        except (ImportError, ModuleNotFoundError):
            pass

# --- Search directory for GPT-2 Small residual stream SAE ---
chosen_release = None
chosen_sae_id = None

if sae_directory is not None:
    print(f"\nSAE directory has {len(sae_directory)} entries.")
    print("First 40 keys:")
    for i, k in enumerate(sorted(sae_directory.keys())):
        if i < 40:
            print(f"  {k}")
        elif i == 40:
            print(f"  ... ({len(sae_directory) - 40} more)")
            break

    candidates = []
    for key, val in sae_directory.items():
        kl = key.lower()
        if "gpt2" not in kl:
            continue
        if "::" in key:
            rel, sid = key.split("::", 1)
            if "resid" in sid.lower():
                candidates.append((rel, sid))
        else:
            sae_ids = []
            if hasattr(val, "saes_map"):
                sae_ids = list(val.saes_map.keys())
            elif isinstance(val, dict) and "saes_map" in val:
                sae_ids = list(val["saes_map"].keys())
            elif isinstance(val, dict):
                sae_ids = list(val.keys())
            for sid in sae_ids:
                if "resid" in sid.lower():
                    candidates.append((key, sid))

    print(f"\nGPT-2 residual stream candidates: {len(candidates)}")
    for c in candidates[:15]:
        print(f"  release={c[0]}  sae_id={c[1]}")

    best_score = -1
    for rel, sid in candidates:
        score = 0
        for layer in [8, 7, 6]:
            if f"blocks.{layer}." in sid:
                score += 10 + (10 - abs(layer - 7))
                break
        if "16384" in sid:
            score += 5
        elif "32768" in sid:
            score += 3
        if "jb" in rel.lower():
            score += 2
        if score > best_score:
            best_score = score
            chosen_release = rel
            chosen_sae_id = sid
else:
    print("\nCould not load SAE directory from any known API path.")

# --- Last resort: brute-force known good IDs ---
if chosen_release is None:
    print("\nTrying known SAE IDs by brute force...")
    known = [
        ("gpt2-small-res-jb", "blocks.8.hook_resid_pre_16384"),
        ("gpt2-small-res-jb", "blocks.7.hook_resid_pre_16384"),
        ("gpt2-small-res-jb", "blocks.6.hook_resid_pre_16384"),
        ("gpt2-small-res-jb", "blocks.8.hook_resid_pre"),
        ("gpt2-small-resid-pre-v5-32k", "blocks.8.hook_resid_pre_32768"),
    ]
    for rel, sid in known:
        try:
            print(f"  Trying: release={rel}  sae_id={sid} ...")
            result = SAE.from_pretrained(release=rel, sae_id=sid)
            # New API returns just SAE; old API returns tuple
            if isinstance(result, tuple):
                del result
            else:
                del result
            chosen_release, chosen_sae_id = rel, sid
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  SUCCESS!")
            break
        except Exception as e:
            print(f"    Failed: {e}")

print(f"\n>>> CHOSEN: release={chosen_release}  sae_id={chosen_sae_id}")
if chosen_release is None:
    print("\nFATAL: No usable SAE found.")
    print("sae_lens attrs:", [x for x in dir(sae_lens) if not x.startswith("_")])
    sys.exit(1)


# ================================================================
# STEP 1 — Load model and SAE
# ================================================================
print("\n" + "=" * 80)
print("STEP 1: Loading model and SAE")
print("=" * 80)

from transformer_lens import HookedTransformer

model_name = "gpt2-small" if "gpt2" in chosen_release.lower() else "pythia-70m-deduped"
print(f"Loading model: {model_name}")
model = HookedTransformer.from_pretrained(model_name)

print(f"Loading SAE: release={chosen_release}, sae_id={chosen_sae_id}")
result = SAE.from_pretrained(release=chosen_release, sae_id=chosen_sae_id)
if isinstance(result, tuple):
    sae_obj = result[0]
else:
    sae_obj = result

model = model.to(device)
sae_obj = sae_obj.to(device)


# --- Robustly extract SAE config (attribute names change across versions) ---
def get_sae_attr(sae, *candidates, default=None):
    """Try multiple attribute names on sae and sae.cfg."""
    for attr in candidates:
        # Try sae.cfg.attr
        if hasattr(sae, "cfg") and hasattr(sae.cfg, attr):
            return getattr(sae.cfg, attr)
        # Try sae.attr directly
        if hasattr(sae, attr):
            return getattr(sae, attr)
    return default


# Hook name (where SAE hooks into the model)
sae_hook_name = get_sae_attr(sae_obj, "hook_name", "hook_point", "hook_point_name")
if sae_hook_name is None:
    # Parse from sae_id: "blocks.8.hook_resid_pre" → "blocks.8.hook_resid_pre"
    sae_hook_name = chosen_sae_id
    print(f"  (hook_name not in config; using sae_id as hook name: {sae_hook_name})")

# Hook layer
sae_hook_layer = get_sae_attr(sae_obj, "hook_layer", "hook_point_layer")
if sae_hook_layer is None:
    # Parse from hook name: "blocks.8.hook_resid_pre" → 8
    import re
    m = re.search(r"blocks\.(\d+)\.", sae_hook_name)
    sae_hook_layer = int(m.group(1)) if m else 8
    print(f"  (hook_layer not in config; parsed from hook name: {sae_hook_layer})")

# SAE width
sae_width = get_sae_attr(sae_obj, "d_sae", "d_hidden", "n_features")
if sae_width is None:
    # Infer from weight matrix
    if hasattr(sae_obj, "W_enc"):
        sae_width = sae_obj.W_enc.shape[-1]
    elif hasattr(sae_obj, "W_dec"):
        sae_width = sae_obj.W_dec.shape[0]
    else:
        sae_width = "unknown"
    print(f"  (d_sae not in config; inferred from weights: {sae_width})")

print(f"\nModel: {model.cfg.model_name}")
print(f"SAE hook name: {sae_hook_name}")
print(f"SAE hook layer: {sae_hook_layer}")
print(f"SAE width: {sae_width}")

# Print all cfg attributes for debug record
if hasattr(sae_obj, "cfg"):
    print(f"SAE config type: {type(sae_obj.cfg).__name__}")
    cfg_attrs = {k: str(getattr(sae_obj.cfg, k))[:80] for k in dir(sae_obj.cfg)
                 if not k.startswith("_") and not callable(getattr(sae_obj.cfg, k, None))}
    print(f"SAE config attrs: {cfg_attrs}")


# ================================================================
# STEP 2 — Extract SAE feature activations
# ================================================================
print("\n" + "=" * 80)
print("STEP 2: Loading dataset + extracting activations")
print("=" * 80)

from datasets import load_dataset

dataset = load_dataset("NeelNanda/pile-10k", split="train")
print(f"Dataset: {len(dataset)} documents")

all_text = " ".join(dataset["text"][:200])
tokens = model.to_tokens(all_text, prepend_bos=True)[0]

N_TOKENS = min(50000, len(tokens))
tokens = tokens[:N_TOKENS]
print(f"Tokens: {N_TOKENS}")

SEQ_LEN = 128
n_seq = N_TOKENS // SEQ_LEN
tokens_batched = tokens[: n_seq * SEQ_LEN].reshape(n_seq, SEQ_LEN)
print(f"Sequences: {n_seq} x {SEQ_LEN} = {n_seq * SEQ_LEN} positions")

BATCH = 4
hook_name = sae_hook_name
hook_layer = sae_hook_layer

all_feature_acts = []
all_token_ids = []

sae_obj.eval()
model.eval()

t0 = time.time()
with torch.no_grad():
    for bs in range(0, n_seq, BATCH):
        be = min(bs + BATCH, n_seq)
        bt = tokens_batched[bs:be].to(device)

        _, cache = model.run_with_cache(
            bt, stop_at_layer=hook_layer + 1, names_filter=[hook_name],
        )
        residual = cache[hook_name]

        # Robust encode across SAELens versions
        try:
            fa = sae_obj.encode(residual)
            if isinstance(fa, tuple):
                fa = fa[0]
            elif hasattr(fa, "feature_acts"):
                fa = fa.feature_acts
        except Exception:
            out = sae_obj(residual)
            if hasattr(out, "feature_acts"):
                fa = out.feature_acts
            elif isinstance(out, tuple):
                fa = out[1] if len(out) > 1 else out[0]
            else:
                fa = out

        fa_flat = fa.reshape(-1, fa.shape[-1])
        tid_flat = bt.reshape(-1)

        if bs == 0:
            print(f"  First batch: feature_acts shape={fa.shape}, "
                  f"min={fa_flat.min().item():.4f}, max={fa_flat.max().item():.4f}")
            assert fa_flat.min().item() >= -1e-6, "Negative activations!"

        all_feature_acts.append(fa_flat.cpu())
        all_token_ids.append(tid_flat.cpu())

        del cache, residual, fa, fa_flat
        torch.cuda.empty_cache()

        done_pct = be / n_seq * 100
        if (bs // BATCH) % 25 == 0:
            print(f"  {be}/{n_seq} sequences ({done_pct:.0f}%) [{time.time()-t0:.1f}s]")

all_feature_acts = torch.cat(all_feature_acts, dim=0).numpy()
all_token_ids = torch.cat(all_token_ids, dim=0).numpy()

N = all_feature_acts.shape[0]
n_features = all_feature_acts.shape[1]
print(f"\nDone in {time.time()-t0:.1f}s")
print(f"Activations: {all_feature_acts.shape}  Tokens: {all_token_ids.shape}")

print(f"\n--- Sanity: Activation stats ---")
print(f"  min={all_feature_acts.min():.6f}  max={all_feature_acts.max():.6f}  "
      f"mean={all_feature_acts.mean():.6f}  frac_nonzero={(all_feature_acts > 0).mean():.6f}")
assert all_feature_acts.min() >= -1e-6, "Negative activations!"
print("  Non-negative: OK")

del model, sae_obj
torch.cuda.empty_cache()
gc.collect()
print("  GPU freed.")


# ================================================================
# STEP 3 — Three-way split
# ================================================================
print("\n" + "=" * 80)
print("STEP 3: Three-way data split (40/30/30)")
print("=" * 80)

np.random.seed(42)
perm = np.random.permutation(N)
n_disc = int(0.4 * N)
n_cal = int(0.3 * N)

idx_disc = perm[:n_disc]
idx_cal = perm[n_disc : n_disc + n_cal]
idx_test = perm[n_disc + n_cal :]

print(f"  Discover: {len(idx_disc)}  Cal: {len(idx_cal)}  Test: {len(idx_test)}")
assert len(np.intersect1d(idx_disc, idx_cal)) == 0
assert len(np.intersect1d(idx_disc, idx_test)) == 0
assert len(np.intersect1d(idx_cal, idx_test)) == 0
print("  Disjoint: OK")

tok_disc = all_token_ids[idx_disc]
tok_cal = all_token_ids[idx_cal]
tok_test = all_token_ids[idx_test]


# ================================================================
# STEP 4 — Feature selection (10 easy, high-purity features)
# ================================================================
print("\n" + "=" * 80)
print("STEP 4: Semi-automated feature selection")
print("=" * 80)

acts_disc_all = all_feature_acts[idx_disc]

act_freq = (acts_disc_all > 0).mean(axis=0)
alive_idx = np.where(act_freq > 0.01)[0]
print(f"Alive features (>1%): {len(alive_idx)} / {n_features}")

from transformers import AutoTokenizer
tok_name = "gpt2" if "gpt2" in model_name else "EleutherAI/pythia-70m-deduped"
tokenizer = AutoTokenizer.from_pretrained(tok_name)
V = tokenizer.vocab_size
print(f"Vocab size: {V}")

print("Computing purity scores...")
t0 = time.time()
purity = np.zeros(len(alive_idx))
dom_tok = np.zeros(len(alive_idx), dtype=np.int64)

for ai, fi in enumerate(alive_idx):
    fa = acts_disc_all[:, fi]
    total = fa.sum()
    if total == 0:
        continue
    sums = np.bincount(tok_disc, weights=fa, minlength=V)
    top = int(np.argmax(sums))
    purity[ai] = sums[top] / total
    dom_tok[ai] = top
    if (ai + 1) % 500 == 0:
        print(f"  {ai+1}/{len(alive_idx)} [{time.time()-t0:.1f}s]")

print(f"Done in {time.time()-t0:.1f}s")

order = np.argsort(-purity)
sel_feats = []
used_toks = set()

for oi in order:
    fi = alive_idx[oi]
    dt = dom_tok[oi]
    if dt in used_toks:
        continue
    if purity[oi] < 0.3:
        break
    sel_feats.append(int(fi))
    used_toks.add(dt)
    if len(sel_feats) >= 10:
        break

if len(sel_feats) < 10:
    for oi in order:
        fi = alive_idx[oi]
        if int(fi) not in sel_feats and purity[oi] > 0.15:
            sel_feats.append(int(fi))
        if len(sel_feats) >= 10:
            break

print(f"\nSelected {len(sel_feats)} features:")
print("=" * 90)
for i, fi in enumerate(sel_feats):
    ai = int(np.where(alive_idx == fi)[0][0])
    fa = acts_disc_all[:, fi]
    means = np.bincount(tok_disc, weights=fa, minlength=V)
    counts = np.bincount(tok_disc, minlength=V).astype(float)
    counts[counts == 0] = 1.0
    means = means / counts
    top5 = np.argsort(-means)[:5]
    top5_str = [(repr(tokenizer.decode([t])), f"{means[t]:.4f}") for t in top5]

    print(f"F{i+1}: sae_idx={fi}, purity={purity[ai]:.3f}, "
          f"act_freq={act_freq[fi]:.4f}, dominant={repr(tokenizer.decode([int(dom_tok[ai])]))}")
    print(f"    Top-5: {top5_str}")

print("\n>>> HUMAN CHECK: Each feature should have a clear dominant token.")


# ================================================================
# STEP 5 — Fit 3 interpretations + conformal test per feature
# ================================================================
print("\n" + "=" * 80)
print("STEP 5: Interpretations + conformal testing")
print("=" * 80)

alpha = 0.05
results = {}

uniq_tok = np.unique(tok_disc)
tok2col = {int(t): c for c, t in enumerate(uniq_tok)}
n_tok_feat = len(uniq_tok)
print(f"One-hot dim: {n_tok_feat}")


def build_oh(tids):
    rows, cols, vals = [], [], []
    for r, t in enumerate(tids):
        c = tok2col.get(int(t))
        if c is not None:
            rows.append(r)
            cols.append(c)
            vals.append(1.0)
    return csr_matrix((vals, (rows, cols)), shape=(len(tids), n_tok_feat))


t0 = time.time()
X_disc = build_oh(tok_disc)
X_cal = build_oh(tok_cal)
X_test = build_oh(tok_test)
print(f"One-hot matrices built in {time.time()-t0:.1f}s. Shape: {X_disc.shape}")

feat1_R = {}

for i, fi in enumerate(sel_feats):
    print(f"\n{'='*60}")
    print(f"Feature {i+1} (SAE index {fi})")
    print(f"{'='*60}")

    ad = all_feature_acts[idx_disc, fi]
    ac = all_feature_acts[idx_cal, fi]
    at = all_feature_acts[idx_test, fi]

    sp = (ad > 0).mean()
    thresh = float(np.median(ad)) if sp > 0.05 else 0.0
    yd = (ad > thresh).astype(int)
    print(f"  sparsity={sp:.4f}, thresh={thresh:.6f}, pos_rate={yd.mean():.4f}")

    amin, amax = float(ad.min()), float(ad.max())
    if amax == amin:
        print(f"  SKIP: constant activation")
        continue

    na_c = np.clip((ac - amin) / (amax - amin), 0, 1)
    na_t = np.clip((at - amin) / (amax - amin), 0, 1)

    # Interp 1: CORRECT
    lr_c = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=5000, random_state=42)
    lr_c.fit(X_disc, yd)
    gc_cal = lr_c.predict_proba(X_cal)[:, 1]
    gc_test = lr_c.predict_proba(X_test)[:, 1]
    tr_acc = accuracy_score(yd, lr_c.predict(X_disc))
    n_nz = int(np.sum(np.abs(lr_c.coef_[0]) > 1e-8))
    print(f"  CORRECT: train_acc={tr_acc:.4f}, nonzero_coefs={n_nz}")
    if tr_acc < 0.70:
        print(f"  WARNING: low train accuracy")

    # Interp 2: WRONG-BUT-PLAUSIBLE (cyclic)
    other_fi = sel_feats[(i + 1) % len(sel_feats)]
    ad_o = all_feature_acts[idx_disc, other_fi]
    sp_o = (ad_o > 0).mean()
    th_o = float(np.median(ad_o)) if sp_o > 0.05 else 0.0
    yd_o = (ad_o > th_o).astype(int)

    lr_w = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=5000, random_state=42)
    lr_w.fit(X_disc, yd_o)
    gw_cal = lr_w.predict_proba(X_cal)[:, 1]
    gw_test = lr_w.predict_proba(X_test)[:, 1]
    print(f"  WRONG-PLAUSIBLE: trained on feature {other_fi}")

    # Interp 3: CLEARLY WRONG (random)
    rng = np.random.RandomState(42 + fi)
    w_r = rng.randn(n_tok_feat) / np.sqrt(n_tok_feat)
    b_r = rng.randn()

    def rand_g(X):
        z = X.dot(w_r) + b_r
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    gr_cal = rand_g(X_cal)
    gr_test = rand_g(X_test)

    # Conformal testing
    feat_res = {"feature_index": int(fi), "feature_rank": i + 1}
    names = ["correct", "wrong_plausible", "clearly_wrong"]
    g_cals = [gc_cal, gw_cal, gr_cal]
    g_tests = [gc_test, gw_test, gr_test]

    for nm, g_c, g_t in zip(names, g_cals, g_tests):
        Rc_aa = np.abs(na_c - g_c)
        Rt_aa = np.abs(na_t - g_t)
        nc = len(Rc_aa)
        ql = min((1 - alpha) * (1 + 1 / nc), 1.0)
        qh_aa = quantile_higher(Rc_aa, ql)
        cov_aa = float(np.mean(Rt_aa <= qh_aa))
        med_Rt = float(np.median(Rt_aa))
        pv_aa = float((np.sum(Rc_aa >= med_Rt) + 1) / (nc + 1))

        Rc_cc = ((ac > thresh) != (g_c > 0.5)).astype(float)
        Rt_cc = ((at > thresh) != (g_t > 0.5)).astype(float)
        qh_cc = quantile_higher(Rc_cc, ql)
        cov_cc = float(np.mean(Rt_cc <= qh_cc))
        med_Rt_cc = float(np.median(Rt_cc))
        pv_cc = float((np.sum(Rc_cc >= med_Rt_cc) + 1) / (nc + 1))

        feat_res[f"{nm}_aa_q_hat"] = qh_aa
        feat_res[f"{nm}_aa_coverage"] = cov_aa
        feat_res[f"{nm}_aa_p_value"] = pv_aa
        feat_res[f"{nm}_aa_mean_cal"] = float(Rc_aa.mean())
        feat_res[f"{nm}_aa_mean_test"] = float(Rt_aa.mean())
        feat_res[f"{nm}_cc_q_hat"] = qh_cc
        feat_res[f"{nm}_cc_coverage"] = cov_cc
        feat_res[f"{nm}_cc_p_value"] = pv_cc
        feat_res[f"{nm}_cc_mean_cal"] = float(Rc_cc.mean())
        feat_res[f"{nm}_cc_mean_test"] = float(Rt_cc.mean())

        print(f"\n  [{nm}] AA: cov={cov_aa:.4f} q={qh_aa:.4f} p={pv_aa:.4f} "
              f"Rcal={Rc_aa.mean():.4f} Rtest={Rt_aa.mean():.4f}")
        print(f"  [{nm}] CC: cov={cov_cc:.4f} q={qh_cc:.4f} p={pv_cc:.4f} "
              f"Rcal={Rc_cc.mean():.4f} Rtest={Rt_cc.mean():.4f}")

        if i == 0:
            feat1_R[f"{nm}_cal"] = Rc_aa.copy()
            feat1_R[f"{nm}_test"] = Rt_aa.copy()

    qc = feat_res["correct_aa_q_hat"]
    print(f"\n  Sanity: q_hat_correct={qc:.4f} {'OK' if 0 < qc < 1 else 'WARNING'}")

    results[f"feature_{i+1}"] = feat_res


# ================================================================
# STEP 6 — Summary table + Pass/Fail
# ================================================================
print("\n\n" + "=" * 100)
print("SUMMARY TABLE: FEASIBILITY TEST 1 RESULTS")
print("=" * 100)
hdr = f"{'Feat':<6} {'Type':<20} {'AA Cov':<10} {'AA p':<10} {'CC Cov':<10} {'CC p':<10} {'AA mR':<10}"
print(hdr)
print("-" * 100)

c_cov, w_cov, r_cov = [], [], []
c_pv, w_pv, r_pv = [], [], []
gaps = []

for i in range(len(sel_feats)):
    k = f"feature_{i+1}"
    if k not in results:
        continue
    r = results[k]
    for tp in ["correct", "wrong_plausible", "clearly_wrong"]:
        print(f"F{i+1:<5} {tp:<20} {r[f'{tp}_aa_coverage']:<10.4f} {r[f'{tp}_aa_p_value']:<10.4f} "
              f"{r[f'{tp}_cc_coverage']:<10.4f} {r[f'{tp}_cc_p_value']:<10.4f} "
              f"{r[f'{tp}_aa_mean_test']:<10.4f}")
    c_cov.append(r["correct_aa_coverage"])
    w_cov.append(r["wrong_plausible_aa_coverage"])
    r_cov.append(r["clearly_wrong_aa_coverage"])
    c_pv.append(r["correct_aa_p_value"])
    w_pv.append(r["wrong_plausible_aa_p_value"])
    r_pv.append(r["clearly_wrong_aa_p_value"])
    gaps.append(r["correct_aa_coverage"] - r["wrong_plausible_aa_coverage"])
    print()

nf = len(c_cov)
c_cov = np.array(c_cov); w_cov = np.array(w_cov); r_cov = np.array(r_cov)
c_pv = np.array(c_pv); w_pv = np.array(w_pv); r_pv = np.array(r_pv)
gaps = np.array(gaps)

print("\n" + "=" * 80)
print("PASS / FAIL EVALUATION")
print("=" * 80)

c1_hard_fail = int(np.sum(c_cov < 0.85))
print(f"\n[C1: Correct coverage >= 0.93]")
print(f"  Passing: {int(np.sum(c_cov >= 0.93))}/{nf}")
print(f"  Hard fail (<0.85): {c1_hard_fail}/{nf}")
print(f"  Values: {c_cov.round(4).tolist()}")
print(f"  {'>>> KILL' if c1_hard_fail > 3 else '>>> OK'}")

c2_rejected = int(np.sum((r_cov <= 0.50) | (r_pv < 0.05)))
c2_fail = int(np.sum(r_cov > 0.85))
print(f"\n[C2: Clearly wrong rejected]")
print(f"  Rejected: {c2_rejected}/{nf}")
print(f"  High cov (>0.85): {c2_fail}/{nf}")
print(f"  Coverages: {r_cov.round(4).tolist()}")
print(f"  P-values: {r_pv.round(4).tolist()}")
if c2_fail > 3:
    print(f"  >>> KILL")
elif c2_rejected < 8:
    print(f"  >>> WARNING: only {c2_rejected}/{nf} rejected")
else:
    print(f"  >>> OK")

c3_dist = int(np.sum(gaps > 0.05))
print(f"\n[C3: Wrong-plausible distinguishable]")
print(f"  Distinguishable: {c3_dist}/{nf}")
print(f"  Gaps: {gaps.round(4).tolist()}")
print(f"  {'>>> KILL/REDESIGN' if c3_dist < 3 else f'>>> OK ({c3_dist}/{nf})'}")

kills = []
if c1_hard_fail > 3:
    kills.append("C1: correct interps rejected too often")
if c2_fail > 3:
    kills.append("C2: random interps not rejected")
if c3_dist < 3:
    kills.append("C3: cannot distinguish wrong-plausible from correct")

print(f"\n{'='*80}")
if not kills:
    print("OVERALL VERDICT: PASS -- Proceed to Feasibility Test 2")
else:
    print(f"OVERALL VERDICT: FAIL ({len(kills)} criteria)")
    for kf in kills:
        print(f"  - {kf}")
print("=" * 80)


# ================================================================
# STEP 7 — Plots
# ================================================================
print("\n" + "=" * 80)
print("STEP 7: Generating plots")
print("=" * 80)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(nf); w = 0.25
ax.bar(x - w, c_cov, w, label="Correct", color="#2ecc71")
ax.bar(x, w_cov, w, label="Wrong-plausible", color="#e67e22")
ax.bar(x + w, r_cov, w, label="Clearly wrong", color="#e74c3c")
ax.axhline(0.95, color="k", ls="--", lw=1, label="Target (0.95)")
ax.axhline(0.85, color="gray", ls=":", lw=1, label="Kill (0.85)")
ax.set_xlabel("Feature"); ax.set_ylabel("AA Coverage")
ax.set_title("Feasibility Test 1: Conformal Coverage by Interpretation Type")
ax.set_xticks(x); ax.set_xticklabels([f"F{i+1}" for i in range(nf)])
ax.legend(loc="lower left"); ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("test1_coverage_by_type.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: test1_coverage_by_type.png")

fig, ax = plt.subplots(figsize=(12, 5))
bar_cols = ["#2ecc71" if g > 0.05 else "#e74c3c" for g in gaps]
ax.bar(range(nf), gaps, color=bar_cols)
ax.axhline(0.05, color="k", ls="--", lw=1, label="Min gap (0.05)")
ax.axhline(0.0, color="gray", ls="-", lw=0.5)
ax.set_xlabel("Feature"); ax.set_ylabel("Gap (Correct - Wrong)")
ax.set_title("Feasibility Test 1: Discriminative Power per Feature")
ax.set_xticks(range(nf)); ax.set_xticklabels([f"F{i+1}" for i in range(nf)])
ax.legend()
plt.tight_layout()
plt.savefig("test1_coverage_gap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: test1_coverage_gap.png")

if feat1_R:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    r1 = results.get("feature_1", {})
    for j, (nm, lab, col) in enumerate([
        ("correct", "Correct", "#2ecc71"),
        ("wrong_plausible", "Wrong-plausible", "#e67e22"),
        ("clearly_wrong", "Clearly wrong", "#e74c3c"),
    ]):
        ax = axes[j]
        ck, tk = f"{nm}_cal", f"{nm}_test"
        if ck in feat1_R:
            ax.hist(feat1_R[ck], bins=50, alpha=0.5, density=True, label="Cal", color="steelblue")
            ax.hist(feat1_R[tk], bins=50, alpha=0.5, density=True, label="Test", color=col)
        cv = r1.get(f"{nm}_aa_coverage", float("nan"))
        pv = r1.get(f"{nm}_aa_p_value", float("nan"))
        ax.set_title(f"{lab}\ncov={cv:.3f}, p={pv:.4f}")
        ax.set_xlabel("R_AA"); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    plt.suptitle("Feature 1: Nonconformity Score Distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig("test1_score_distributions_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: test1_score_distributions_f1.png")

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w, c_pv, w, label="Correct", color="#2ecc71")
ax.bar(x, w_pv, w, label="Wrong-plausible", color="#e67e22")
ax.bar(x + w, r_pv, w, label="Clearly wrong", color="#e74c3c")
ax.axhline(0.05, color="k", ls="--", lw=1, label="alpha=0.05")
ax.set_xlabel("Feature"); ax.set_ylabel("p-value")
ax.set_title("Feasibility Test 1: Conformal P-values by Interpretation Type")
ax.set_xticks(x); ax.set_xticklabels([f"F{i+1}" for i in range(nf)])
ax.legend()
plt.tight_layout()
plt.savefig("test1_pvalues.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: test1_pvalues.png")


# ================================================================
# STEP 8 — Save JSON
# ================================================================
print("\n" + "=" * 80)
print("STEP 8: Saving results")
print("=" * 80)

res_out = {}
for k, v in results.items():
    res_out[k] = {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, np.ndarray))}

res_out["summary"] = {
    "n_features": nf,
    "n_tokens": int(N),
    "n_discover": int(len(idx_disc)),
    "n_cal": int(len(idx_cal)),
    "n_test": int(len(idx_test)),
    "alpha": alpha,
    "correct_coverages_aa": c_cov.tolist(),
    "wrong_coverages_aa": w_cov.tolist(),
    "random_coverages_aa": r_cov.tolist(),
    "correct_pvalues_aa": c_pv.tolist(),
    "wrong_pvalues_aa": w_pv.tolist(),
    "random_pvalues_aa": r_pv.tolist(),
    "coverage_gaps": gaps.tolist(),
    "model": model_name,
    "sae_release": chosen_release,
    "sae_id": chosen_sae_id,
    "verdict": "PASS" if not kills else "FAIL",
    "kill_flags": kills,
}

with open("test1_results.json", "w") as f:
    json.dump(res_out, f, indent=2)
print("Saved: test1_results.json")

print("\n\nCOMPACT RESULTS (copy this):")
print(json.dumps(res_out["summary"], indent=2))

print("\n\n" + "=" * 80)
print("OUTPUT FILES:")
print("  test1_coverage_by_type.png")
print("  test1_coverage_gap.png")
print("  test1_score_distributions_f1.png")
print("  test1_pvalues.png")
print("  test1_results.json")
print()
print("TO DISPLAY IMAGES, run in the NEXT Kaggle cell:")
print("  from IPython.display import Image, display")
print("  for f in ['test1_coverage_by_type.png','test1_coverage_gap.png',")
print("            'test1_score_distributions_f1.png','test1_pvalues.png']:")
print("      print(f); display(Image(filename=f))")
print()
print(f"Done! {nf} features tested.")
