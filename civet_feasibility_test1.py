#!/usr/bin/env python3
"""
CIVET Feasibility Test 1: The Discrimination Test
===================================================
Tests whether conformal prediction can discriminate between correct and wrong
interpretations of SAE features. Run on Kaggle with a single T4 GPU.

Outputs:
  - Printed summary tables, sanity checks, pass/fail evaluation
  - test1_coverage_by_type.png
  - test1_coverage_gap.png
  - test1_pvalues.png
  - test1_score_distributions_f1.png
  - test1_results.json
"""

import sys
import time
import json
import gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; saves to file
import matplotlib.pyplot as plt

# Helper: np.quantile 'method' param was called 'interpolation' before numpy 1.22
def _quantile_higher(arr, q):
    """Compute quantile with method='higher', compatible across numpy versions."""
    try:
        return float(np.quantile(arr, q, method="higher"))
    except TypeError:
        return float(np.quantile(arr, q, interpolation="higher"))

print("=" * 80)
print("CIVET FEASIBILITY TEST 1: THE DISCRIMINATION TEST")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# ============================================================
# STEP 0: Discover available SAEs in SAELens
# ============================================================
print("=" * 80)
print("STEP 0: Discovering available SAE releases in SAELens")
print("=" * 80)

from sae_lens import SAE
from sae_lens.pretrained_saes import get_pretrained_saes_directory

sae_directory = get_pretrained_saes_directory()

# Print all available releases for the record
print(f"\nTotal SAE releases/entries available: {len(sae_directory)}")
print("\nAll release/SAE keys (showing first 50):")
for i, key in enumerate(sorted(sae_directory.keys())):
    if i < 50:
        print(f"  {key}")
    elif i == 50:
        print(f"  ... and {len(sae_directory) - 50} more")

# The directory structure varies by SAELens version:
#   - Newer: flat dict mapping "release::sae_id" -> PretrainedSAEInfo
#   - Older: nested dict mapping release -> {saes_map: {sae_id: ...}}
# We handle both.

# Strategy: look for keys containing "gpt2-small" and "resid"
print("\n--- Searching for GPT-2 Small residual stream SAEs ---")

# Collect candidate (release, sae_id) pairs
candidates = []

for key, val in sae_directory.items():
    key_lower = key.lower()
    
    # Check if this is a flat-style key like "gpt2-small-res-jb::blocks.8.hook_resid_pre_16384"
    if "::" in key:
        release_part, sae_id_part = key.split("::", 1)
        if "gpt2" in release_part.lower() and ("resid" in key_lower or "res" in key_lower):
            candidates.append((release_part, sae_id_part, key))
    else:
        # Might be a release name with nested saes_map
        if "gpt2" in key_lower:
            # Try to get sae_ids from this release
            sae_ids = []
            if hasattr(val, "saes_map"):
                sae_ids = list(val.saes_map.keys())
            elif isinstance(val, dict) and "saes_map" in val:
                sae_ids = list(val["saes_map"].keys())
            elif isinstance(val, dict):
                # Maybe it's directly mapping sae_id -> info
                sae_ids = list(val.keys())
            
            for sid in sae_ids:
                if "resid" in sid.lower() or "res" in sid.lower():
                    candidates.append((key, sid, f"{key}::{sid}"))

print(f"Found {len(candidates)} GPT-2 residual stream SAE candidates")
for c in candidates[:20]:
    print(f"  release={c[0]}  sae_id={c[1]}")
if len(candidates) > 20:
    print(f"  ... and {len(candidates) - 20} more")

# Pick the best candidate: layer 6-8, prefer 16K width
chosen_release = None
chosen_sae_id = None
best_score = -1

for release, sae_id, full_key in candidates:
    score = 0
    sid_lower = sae_id.lower()
    # Prefer layers 6-8
    for layer in [8, 7, 6]:
        if f"blocks.{layer}." in sae_id or f"layer_{layer}" in sid_lower:
            score += 10 + (10 - abs(layer - 7))  # prefer layer 7-8
            break
    # Prefer 16K width
    if "16384" in sae_id or "16k" in sid_lower:
        score += 5
    elif "32768" in sae_id or "32k" in sid_lower:
        score += 3
    # Prefer "res-jb" (Joseph Bloom's popular SAEs)
    if "jb" in release.lower():
        score += 2
    
    if score > best_score:
        best_score = score
        chosen_release = release
        chosen_sae_id = sae_id

# If no GPT-2 SAE found, try Pythia
if chosen_release is None:
    print("\n--- No GPT-2 Small SAE found. Searching for Pythia-70M ---")
    for key, val in sae_directory.items():
        key_lower = key.lower()
        if "pythia" in key_lower and "70m" in key_lower:
            if "::" in key:
                release_part, sae_id_part = key.split("::", 1)
                if "resid" in sae_id_part.lower():
                    chosen_release = release_part
                    chosen_sae_id = sae_id_part
                    break
            else:
                sae_ids = []
                if hasattr(val, "saes_map"):
                    sae_ids = list(val.saes_map.keys())
                elif isinstance(val, dict):
                    sae_ids = list(val.keys())
                resid_ids = [s for s in sae_ids if "resid" in s.lower()]
                if resid_ids:
                    chosen_release = key
                    chosen_sae_id = resid_ids[0]
                    break

print(f"\n>>> CHOSEN SAE RELEASE: {chosen_release}")
print(f">>> CHOSEN SAE ID: {chosen_sae_id}")

if chosen_release is None or chosen_sae_id is None:
    # Last resort: print everything and let user pick
    print("\nERROR: Could not auto-select an SAE. Printing full directory for manual selection:")
    for rname, rinfo in sae_directory.items():
        print(f"\n  Release: {rname}")
        try:
            if hasattr(rinfo, "saes_map"):
                for sid in list(rinfo.saes_map.keys())[:10]:
                    print(f"    {sid}")
            elif isinstance(rinfo, dict):
                for sid in list(rinfo.keys())[:10]:
                    print(f"    {sid}")
        except Exception:
            print(f"    (could not enumerate)")
    sys.exit(1)

# ============================================================
# STEP 1: Load Model and SAE
# ============================================================
print("\n" + "=" * 80)
print("STEP 1: Loading model and SAE")
print("=" * 80)

from transformer_lens import HookedTransformer

# Determine model name from release
model_name = "gpt2-small" if "gpt2" in chosen_release.lower() else "pythia-70m-deduped"
print(f"Loading model: {model_name}")
model = HookedTransformer.from_pretrained(model_name)

print(f"Loading SAE: release={chosen_release}, sae_id={chosen_sae_id}")
sae, cfg_dict, sparsity_data = SAE.from_pretrained(
    release=chosen_release,
    sae_id=chosen_sae_id,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
sae = sae.to(device)

print(f"\nModel: {model.cfg.model_name}")
print(f"SAE hook: {sae.cfg.hook_name}")
print(f"SAE hook layer: {sae.cfg.hook_layer}")
print(f"SAE width (d_sae): {sae.cfg.d_sae}")
print(f"Device: {device}")

# ============================================================
# STEP 2: Load dataset and extract activations
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: Loading dataset and extracting SAE activations")
print("=" * 80)

from datasets import load_dataset

dataset = load_dataset("NeelNanda/pile-10k", split="train")
print(f"Dataset loaded: {len(dataset)} documents")

# Concatenate and tokenize
all_text = " ".join(dataset["text"][:200])
tokens = model.to_tokens(all_text, prepend_bos=True)
tokens = tokens[0]  # (seq_len,)

N_TOKENS = min(50000, len(tokens))
tokens = tokens[:N_TOKENS]
print(f"Total tokens after truncation: {N_TOKENS}")

# Process in batches
SEQ_LEN = 128
n_sequences = N_TOKENS // SEQ_LEN
tokens_batched = tokens[:n_sequences * SEQ_LEN].reshape(n_sequences, SEQ_LEN)
actual_N = n_sequences * SEQ_LEN
print(f"Sequences: {n_sequences} x {SEQ_LEN} = {actual_N} token positions")

# Batch size in terms of sequences per forward pass
BATCH_SEQ = 4  # number of sequences per batch (conservative for T4)

all_feature_acts = []
all_token_ids = []

sae.eval()
model.eval()

hook_name = sae.cfg.hook_name
hook_layer = sae.cfg.hook_layer

t0 = time.time()
with torch.no_grad():
    for batch_start in range(0, n_sequences, BATCH_SEQ):
        batch_end = min(batch_start + BATCH_SEQ, n_sequences)
        batch_tokens = tokens_batched[batch_start:batch_end].to(device)

        # Run model up to the needed layer
        _, cache = model.run_with_cache(
            batch_tokens,
            stop_at_layer=hook_layer + 1,
            names_filter=[hook_name],
        )

        residual = cache[hook_name]  # (batch, seq_len, d_model)

        # Get SAE feature activations (handle different SAELens API versions)
        try:
            feature_acts = sae.encode(residual)
            # Some versions return a tuple/named tuple
            if isinstance(feature_acts, tuple):
                feature_acts = feature_acts[0]
            elif hasattr(feature_acts, "feature_acts"):
                feature_acts = feature_acts.feature_acts
        except Exception:
            # Fallback: use forward pass
            sae_out = sae(residual)
            if isinstance(sae_out, tuple):
                feature_acts = sae_out[1] if len(sae_out) > 1 else sae_out[0]
            elif hasattr(sae_out, "feature_acts"):
                feature_acts = sae_out.feature_acts
            else:
                feature_acts = sae_out

        # Flatten
        feature_acts_flat = feature_acts.reshape(-1, feature_acts.shape[-1])
        token_ids_flat = batch_tokens.reshape(-1)

        all_feature_acts.append(feature_acts_flat.cpu())
        all_token_ids.append(token_ids_flat.cpu())

        # Print shape info on first batch for sanity
        if batch_start == 0:
            print(f"  First batch — feature_acts shape: {feature_acts.shape}")
            print(f"  First batch — feature_acts_flat shape: {feature_acts_flat.shape}")
            print(f"  Feature activations min={feature_acts_flat.min().item():.4f}, "
                  f"max={feature_acts_flat.max().item():.4f}")
            assert feature_acts_flat.min().item() >= -1e-6, "Negative activations in first batch!"

        del cache, residual, feature_acts, feature_acts_flat
        torch.cuda.empty_cache()

        if (batch_start // BATCH_SEQ) % 20 == 0:
            elapsed = time.time() - t0
            done = batch_end
            pct = done / n_sequences * 100
            print(f"  Processed {done}/{n_sequences} sequences ({pct:.0f}%) [{elapsed:.1f}s]")

# Stack all batches
all_feature_acts = torch.cat(all_feature_acts, dim=0).numpy()  # (N, d_sae)
all_token_ids = torch.cat(all_token_ids, dim=0).numpy()        # (N,)

N = all_feature_acts.shape[0]
n_features = all_feature_acts.shape[1]
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")
print(f"Feature activations shape: {all_feature_acts.shape}")
print(f"Token IDs shape: {all_token_ids.shape}")

# ============================================================
# SANITY CHECK: Activations are non-trivial
# ============================================================
print("\n--- Sanity Check: Activation statistics ---")
print(f"  Min activation: {all_feature_acts.min():.6f}")
print(f"  Max activation: {all_feature_acts.max():.6f}")
print(f"  Mean activation: {all_feature_acts.mean():.6f}")
print(f"  Fraction of non-zero entries: {(all_feature_acts > 0).mean():.6f}")
assert all_feature_acts.min() >= -1e-6, "ERROR: Negative activations detected!"
print("  >> Activations are non-negative. OK.")

# Free GPU memory — we only need CPU from here on
del model, sae
torch.cuda.empty_cache()
gc.collect()
print("  >> GPU memory freed (model + SAE unloaded).")

# ============================================================
# STEP 3: Three-way data split
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: Three-way data split")
print("=" * 80)

np.random.seed(42)
indices = np.random.permutation(N)
n_discover = int(0.4 * N)
n_cal = int(0.3 * N)

idx_discover = indices[:n_discover]
idx_cal = indices[n_discover:n_discover + n_cal]
idx_test = indices[n_discover + n_cal:]

print(f"  Discover set: {len(idx_discover)} tokens")
print(f"  Calibration set: {len(idx_cal)} tokens")
print(f"  Test set: {len(idx_test)} tokens")

# Sanity: splits are disjoint
assert len(np.intersect1d(idx_discover, idx_cal)) == 0, "Discover/Cal overlap!"
assert len(np.intersect1d(idx_discover, idx_test)) == 0, "Discover/Test overlap!"
assert len(np.intersect1d(idx_cal, idx_test)) == 0, "Cal/Test overlap!"
print("  >> Splits are disjoint. OK.")

tokens_discover = all_token_ids[idx_discover]
tokens_cal = all_token_ids[idx_cal]
tokens_test = all_token_ids[idx_test]

# ============================================================
# STEP 4: Semi-automated feature selection (10 easy features)
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: Semi-automated feature selection")
print("=" * 80)

acts_discover = all_feature_acts[idx_discover]  # (n_discover, n_features)

# Step 4a: Find alive features (>1% activation frequency)
activation_frequency = (acts_discover > 0).mean(axis=0)
alive_mask = activation_frequency > 0.01
alive_indices = np.where(alive_mask)[0]
print(f"Alive features (>1% activation): {len(alive_indices)} / {n_features}")

# Step 4b: Compute purity for each alive feature using vectorized bincount
# We need: for each feature, the token with the highest total activation, and
# purity = that token's share of total activation.
# We already have the tokenizer from the model, but we freed the model.
# Re-load tokenizer only (lightweight)
from transformers import AutoTokenizer
tokenizer_name = "gpt2" if "gpt2" in model_name else "EleutherAI/pythia-70m-deduped"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
vocab_size = tokenizer.vocab_size
print(f"Vocab size: {vocab_size}")

print("Computing purity scores for alive features...")
t0 = time.time()

purity_scores = np.zeros(len(alive_indices))
dominant_tokens = np.zeros(len(alive_indices), dtype=np.int64)

for ai, feat_idx in enumerate(alive_indices):
    feat_acts = acts_discover[:, feat_idx]
    total_act = feat_acts.sum()
    if total_act == 0:
        purity_scores[ai] = 0.0
        dominant_tokens[ai] = -1
        continue

    # Use bincount to sum activations per token type
    token_act_sums = np.bincount(tokens_discover, weights=feat_acts, minlength=vocab_size)
    top_token = np.argmax(token_act_sums)
    purity_scores[ai] = token_act_sums[top_token] / total_act
    dominant_tokens[ai] = top_token

    if (ai + 1) % 500 == 0:
        print(f"  Computed purity for {ai + 1}/{len(alive_indices)} features [{time.time()-t0:.1f}s]")

print(f"Purity computation done in {time.time()-t0:.1f}s")

# Step 4c: Select top features by purity with diverse dominant tokens
sorted_order = np.argsort(-purity_scores)

selected_features = []
selected_dominant_tokens_set = set()

for idx in sorted_order:
    feat_idx = alive_indices[idx]
    dom_tok = dominant_tokens[idx]
    if dom_tok in selected_dominant_tokens_set:
        continue
    if purity_scores[idx] < 0.3:
        break
    selected_features.append(int(feat_idx))
    selected_dominant_tokens_set.add(dom_tok)
    if len(selected_features) >= 10:
        break

# Relax diversity if needed
if len(selected_features) < 10:
    for idx in sorted_order:
        feat_idx = alive_indices[idx]
        if int(feat_idx) not in selected_features and purity_scores[idx] > 0.15:
            selected_features.append(int(feat_idx))
        if len(selected_features) >= 10:
            break

print(f"\nSelected {len(selected_features)} features for testing:")
print("=" * 90)

# Print feature selection table with top-5 tokens
for i, feat_idx in enumerate(selected_features):
    ai = np.where(alive_indices == feat_idx)[0][0]
    dom_tok = dominant_tokens[ai]
    purity = purity_scores[ai]

    # Top-5 tokens by mean activation
    feat_acts = acts_discover[:, feat_idx]
    token_act_means = np.bincount(tokens_discover, weights=feat_acts, minlength=vocab_size)
    token_counts = np.bincount(tokens_discover, minlength=vocab_size).astype(float)
    token_counts[token_counts == 0] = 1.0  # avoid div by zero
    token_act_means = token_act_means / token_counts
    
    top5 = np.argsort(-token_act_means)[:5]
    top5_strs = [repr(tokenizer.decode([t])) for t in top5]
    top5_means = [f"{token_act_means[t]:.4f}" for t in top5]

    print(f"Feature {i+1}: index={feat_idx}, purity={purity:.3f}")
    print(f"  Dominant token: {repr(tokenizer.decode([int(dom_tok)]))}")
    print(f"  Activation frequency: {activation_frequency[feat_idx]:.4f}")
    print(f"  Top-5 tokens (by mean act): {list(zip(top5_strs, top5_means))}")
    print()

print(">>> HUMAN CHECK: Do these features look interpretable? Each should have")
print("    a clear dominant token. If any look weird, note it but we proceed anyway.")
print()

# ============================================================
# STEP 5: Fit interpretations and run conformal tests
# ============================================================
print("=" * 80)
print("STEP 5: Fitting 3 interpretations per feature + conformal testing")
print("=" * 80)

alpha = 0.05
results = {}

# Build one-hot token features ONCE (same for all features)
unique_tokens_disc = np.unique(tokens_discover)
token_to_col = {int(t): c for c, t in enumerate(unique_tokens_disc)}
n_token_features = len(unique_tokens_disc)
print(f"Unique tokens in discover set: {n_token_features}")

def build_onehot(token_ids):
    rows, cols, vals = [], [], []
    for r, t in enumerate(token_ids):
        t = int(t)
        if t in token_to_col:
            rows.append(r)
            cols.append(token_to_col[t])
            vals.append(1.0)
    return csr_matrix((vals, (rows, cols)), shape=(len(token_ids), n_token_features))

print("Building one-hot matrices (one-time cost)...")
t0_oh = time.time()
X_disc = build_onehot(tokens_discover)
X_cal = build_onehot(tokens_cal)
X_test = build_onehot(tokens_test)
print(f"  Done in {time.time()-t0_oh:.1f}s. X_disc shape: {X_disc.shape}")

# We'll store full R distributions for Feature 1 for the histogram plot
feat1_R_distributions = {}

for i, feat_idx in enumerate(selected_features):
    print(f"\n{'='*60}")
    print(f"Processing Feature {i+1} (SAE index {feat_idx})")
    print(f"{'='*60}")

    # --- Extract activations ---
    acts_disc = all_feature_acts[idx_discover, feat_idx]
    acts_cal = all_feature_acts[idx_cal, feat_idx]
    acts_test = all_feature_acts[idx_test, feat_idx]

    # --- Binarization threshold ---
    sparsity_rate = (acts_disc > 0).mean()
    if sparsity_rate > 0.05:
        threshold = float(np.median(acts_disc))
    else:
        threshold = 0.0

    y_disc = (acts_disc > threshold).astype(int)
    y_cal = (acts_cal > threshold).astype(int)
    y_test = (acts_test > threshold).astype(int)
    print(f"  Sparsity rate: {sparsity_rate:.4f}, Threshold: {threshold:.6f}")
    print(f"  Positive rate (discover): {y_disc.mean():.4f}")

    # --- Normalization ---
    a_min = float(acts_disc.min())
    a_max = float(acts_disc.max())
    if a_max == a_min:
        print(f"  WARNING: Feature {feat_idx} has constant activation. Skipping.")
        continue

    norm_acts_disc = np.clip((acts_disc - a_min) / (a_max - a_min), 0, 1)
    norm_acts_cal = np.clip((acts_cal - a_min) / (a_max - a_min), 0, 1)
    norm_acts_test = np.clip((acts_test - a_min) / (a_max - a_min), 0, 1)

    # Sanity
    assert norm_acts_cal.min() >= 0.0 and norm_acts_cal.max() <= 1.0 + 1e-9
    assert norm_acts_test.min() >= 0.0 and norm_acts_test.max() <= 1.0 + 1e-9

    # ======================================================
    # INTERPRETATION 1: CORRECT
    # ======================================================
    lr_correct = LogisticRegression(
        penalty="l1", solver="saga", C=0.1, max_iter=5000, random_state=42
    )
    lr_correct.fit(X_disc, y_disc)

    g_correct_cal = lr_correct.predict_proba(X_cal)[:, 1]
    g_correct_test = lr_correct.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_disc, lr_correct.predict(X_disc))
    n_nonzero = int(np.sum(np.abs(lr_correct.coef_[0]) > 1e-8))
    print(f"  Correct interp: train_acc={train_acc:.4f}, nonzero_coefs={n_nonzero}")

    # Sanity: train accuracy should be decent for easy features
    if train_acc < 0.70:
        print(f"  WARNING: Low train accuracy ({train_acc:.3f}). Feature may not be easy.")

    # ======================================================
    # INTERPRETATION 2: WRONG-BUT-PLAUSIBLE (cyclic pairing)
    # ======================================================
    other_feat_idx = selected_features[(i + 1) % len(selected_features)]
    acts_disc_other = all_feature_acts[idx_discover, other_feat_idx]

    sparsity_other = (acts_disc_other > 0).mean()
    thresh_other = float(np.median(acts_disc_other)) if sparsity_other > 0.05 else 0.0
    y_disc_other = (acts_disc_other > thresh_other).astype(int)

    lr_wrong = LogisticRegression(
        penalty="l1", solver="saga", C=0.1, max_iter=5000, random_state=42
    )
    lr_wrong.fit(X_disc, y_disc_other)

    g_wrong_cal = lr_wrong.predict_proba(X_cal)[:, 1]
    g_wrong_test = lr_wrong.predict_proba(X_test)[:, 1]
    print(f"  Wrong-plausible interp: trained on feature {other_feat_idx}")

    # ======================================================
    # INTERPRETATION 3: CLEARLY WRONG (random weights)
    # ======================================================
    rng = np.random.RandomState(42 + feat_idx)
    w_random = rng.randn(n_token_features) / np.sqrt(n_token_features)
    b_random = rng.randn()

    def random_interp(X_sparse):
        logits = X_sparse.dot(w_random) + b_random
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

    g_random_cal = random_interp(X_cal)
    g_random_test = random_interp(X_test)

    # ======================================================
    # CONFORMAL TESTING
    # ======================================================
    interp_names = ["correct", "wrong_plausible", "clearly_wrong"]
    g_cals = [g_correct_cal, g_wrong_cal, g_random_cal]
    g_tests = [g_correct_test, g_wrong_test, g_random_test]

    feature_results = {"feature_index": int(feat_idx), "feature_rank": i + 1}

    for name, g_cal, g_test in zip(interp_names, g_cals, g_tests):
        # --- AA Score ---
        R_cal_aa = np.abs(norm_acts_cal - g_cal)
        R_test_aa = np.abs(norm_acts_test - g_test)

        n_cal_pts = len(R_cal_aa)
        q_level = min((1 - alpha) * (1 + 1 / n_cal_pts), 1.0)
        q_hat_aa = _quantile_higher(R_cal_aa, q_level)

        coverage_aa = float(np.mean(R_test_aa <= q_hat_aa))

        median_R_test_aa = float(np.median(R_test_aa))
        p_value_aa = float((np.sum(R_cal_aa >= median_R_test_aa) + 1) / (n_cal_pts + 1))

        # --- CC Score ---
        R_cal_cc = ((acts_cal > threshold) != (g_cal > 0.5)).astype(float)
        R_test_cc = ((acts_test > threshold) != (g_test > 0.5)).astype(float)

        q_hat_cc = _quantile_higher(R_cal_cc, q_level)
        coverage_cc = float(np.mean(R_test_cc <= q_hat_cc))

        median_R_test_cc = float(np.median(R_test_cc))
        p_value_cc = float((np.sum(R_cal_cc >= median_R_test_cc) + 1) / (n_cal_pts + 1))

        # Store
        feature_results[f"{name}_aa_q_hat"] = q_hat_aa
        feature_results[f"{name}_aa_coverage"] = coverage_aa
        feature_results[f"{name}_aa_p_value"] = p_value_aa
        feature_results[f"{name}_aa_mean_cal"] = float(R_cal_aa.mean())
        feature_results[f"{name}_aa_mean_test"] = float(R_test_aa.mean())
        feature_results[f"{name}_cc_q_hat"] = q_hat_cc
        feature_results[f"{name}_cc_coverage"] = coverage_cc
        feature_results[f"{name}_cc_p_value"] = p_value_cc
        feature_results[f"{name}_cc_mean_cal"] = float(R_cal_cc.mean())
        feature_results[f"{name}_cc_mean_test"] = float(R_test_cc.mean())

        print(
            f"\n  [{name}] AA: coverage={coverage_aa:.4f}, q_hat={q_hat_aa:.4f}, "
            f"p={p_value_aa:.4f}, mean_R_cal={R_cal_aa.mean():.4f}, mean_R_test={R_test_aa.mean():.4f}"
        )
        print(
            f"  [{name}] CC: coverage={coverage_cc:.4f}, q_hat={q_hat_cc:.4f}, "
            f"p={p_value_cc:.4f}, mean_R_cal={R_cal_cc.mean():.4f}, mean_R_test={R_test_cc.mean():.4f}"
        )

        # Save full R distributions for Feature 1 for histogram plot
        if i == 0:
            feat1_R_distributions[f"{name}_cal_aa"] = R_cal_aa.copy()
            feat1_R_distributions[f"{name}_test_aa"] = R_test_aa.copy()

    # Sanity check: q_hat should be reasonable
    q_correct = feature_results["correct_aa_q_hat"]
    if q_correct <= 0 or q_correct >= 1:
        print(f"  WARNING: q_hat for correct interp is {q_correct:.4f} (expected 0.2-0.7)")
    else:
        print(f"\n  Sanity: q_hat_correct={q_correct:.4f} (expected ~0.2-0.7). OK.")

    results[f"feature_{i+1}"] = feature_results

# ============================================================
# STEP 6: Summary table and Pass/Fail evaluation
# ============================================================
print("\n\n" + "=" * 100)
print("SUMMARY TABLE: FEASIBILITY TEST 1 RESULTS")
print("=" * 100)
print(
    f"\n{'Feature':<10} {'Type':<20} {'AA Coverage':<14} {'AA p-value':<12} "
    f"{'CC Coverage':<14} {'CC p-value':<12} {'AA mean_R':<12}"
)
print("-" * 100)

correct_coverages_aa = []
wrong_coverages_aa = []
random_coverages_aa = []
correct_pvalues_aa = []
wrong_pvalues_aa = []
random_pvalues_aa = []
coverage_gaps = []

for i in range(len(selected_features)):
    key = f"feature_{i+1}"
    if key not in results:
        continue
    r = results[key]

    for interp_type in ["correct", "wrong_plausible", "clearly_wrong"]:
        cov_aa = r[f"{interp_type}_aa_coverage"]
        p_aa = r[f"{interp_type}_aa_p_value"]
        cov_cc = r[f"{interp_type}_cc_coverage"]
        p_cc = r[f"{interp_type}_cc_p_value"]
        mean_r = r[f"{interp_type}_aa_mean_test"]

        print(
            f"F{i+1:<9} {interp_type:<20} {cov_aa:<14.4f} {p_aa:<12.4f} "
            f"{cov_cc:<14.4f} {p_cc:<12.4f} {mean_r:<12.4f}"
        )

    correct_coverages_aa.append(r["correct_aa_coverage"])
    wrong_coverages_aa.append(r["wrong_plausible_aa_coverage"])
    random_coverages_aa.append(r["clearly_wrong_aa_coverage"])
    correct_pvalues_aa.append(r["correct_aa_p_value"])
    wrong_pvalues_aa.append(r["wrong_plausible_aa_p_value"])
    random_pvalues_aa.append(r["clearly_wrong_aa_p_value"])
    coverage_gaps.append(r["correct_aa_coverage"] - r["wrong_plausible_aa_coverage"])
    print()

n_feat = len(correct_coverages_aa)
correct_coverages_aa = np.array(correct_coverages_aa)
wrong_coverages_aa = np.array(wrong_coverages_aa)
random_coverages_aa = np.array(random_coverages_aa)
correct_pvalues_aa = np.array(correct_pvalues_aa)
wrong_pvalues_aa = np.array(wrong_pvalues_aa)
random_pvalues_aa = np.array(random_pvalues_aa)
coverage_gaps = np.array(coverage_gaps)

# --- PASS / FAIL ---
print("\n" + "=" * 80)
print("PASS / FAIL EVALUATION")
print("=" * 80)

# Criterion 1
c1_pass = int(np.sum(correct_coverages_aa >= 0.93))
c1_fail_hard = int(np.sum(correct_coverages_aa < 0.85))
print(f"\n[Criterion 1: Correct interp coverage >= 0.93]")
print(f"  Features passing: {c1_pass}/{n_feat}")
print(f"  Features with coverage < 0.85: {c1_fail_hard}/{n_feat}")
print(f"  Coverages: {correct_coverages_aa.round(4).tolist()}")
if c1_fail_hard > 3:
    print(f"  >>> KILL: {c1_fail_hard} features have coverage < 0.85 (threshold: >3 kills)")
else:
    print(f"  >>> OK")

# Criterion 2
c2_rejected = int(np.sum((random_coverages_aa <= 0.50) | (random_pvalues_aa < 0.05)))
c2_fail = int(np.sum(random_coverages_aa > 0.85))
print(f"\n[Criterion 2: Clearly wrong rejected]")
print(f"  Features rejected (cov<=0.50 or p<0.05): {c2_rejected}/{n_feat}")
print(f"  Features with coverage > 0.85: {c2_fail}/{n_feat}")
print(f"  Coverages: {random_coverages_aa.round(4).tolist()}")
print(f"  P-values: {random_pvalues_aa.round(4).tolist()}")
if c2_fail > 3:
    print(f"  >>> KILL: {c2_fail} features have random coverage > 0.85")
elif c2_rejected < 8:
    print(f"  >>> WARNING: Only {c2_rejected}/{n_feat} random interps rejected (want >=8)")
else:
    print(f"  >>> OK")

# Criterion 3
c3_distinguishable = int(np.sum(coverage_gaps > 0.05))
print(f"\n[Criterion 3: Wrong-plausible distinguishable from correct]")
print(f"  Features where correct coverage > wrong+0.05: {c3_distinguishable}/{n_feat}")
print(f"  Coverage gaps (correct - wrong): {coverage_gaps.round(4).tolist()}")
if c3_distinguishable < 3:
    print(f"  >>> KILL/REDESIGN: Cannot distinguish wrong-plausible from correct")
else:
    print(f"  >>> OK (distinguishable for {c3_distinguishable}/{n_feat} features)")

# Overall verdict
print(f"\n{'='*80}")
kill_flags = []
if c1_fail_hard > 3:
    kill_flags.append("Criterion 1 FAILED: correct interps rejected too often")
if c2_fail > 3:
    kill_flags.append("Criterion 2 FAILED: random interps not rejected")
if c3_distinguishable < 3:
    kill_flags.append("Criterion 3 FAILED: cannot distinguish wrong-plausible from correct")

if len(kill_flags) == 0:
    print("OVERALL VERDICT: PASS — Proceed to Feasibility Test 2")
else:
    print(f"OVERALL VERDICT: FAIL ({len(kill_flags)} criteria failed)")
    for flag in kill_flags:
        print(f"  - {flag}")
print("=" * 80)

# ============================================================
# STEP 7: Plots
# ============================================================
print("\n" + "=" * 80)
print("STEP 7: Generating plots")
print("=" * 80)

# PLOT 1: Bar chart of AA coverage by interpretation type
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(n_feat)
width = 0.25

ax.bar(x - width, correct_coverages_aa, width, label="Correct", color="#2ecc71")
ax.bar(x, wrong_coverages_aa, width, label="Wrong-plausible", color="#e67e22")
ax.bar(x + width, random_coverages_aa, width, label="Clearly wrong", color="#e74c3c")

ax.axhline(y=0.95, color="black", linestyle="--", linewidth=1, label="Target coverage (0.95)")
ax.axhline(y=0.85, color="gray", linestyle=":", linewidth=1, label="Kill threshold (0.85)")
ax.set_xlabel("Feature")
ax.set_ylabel("Empirical Coverage (AA score)")
ax.set_title("Feasibility Test 1: Conformal Coverage by Interpretation Type")
ax.set_xticks(x)
ax.set_xticklabels([f"F{i+1}" for i in range(n_feat)])
ax.legend(loc="lower left")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("test1_coverage_by_type.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: test1_coverage_by_type.png")

# PLOT 2: Coverage gap per feature
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#2ecc71" if g > 0.05 else "#e74c3c" for g in coverage_gaps]
ax.bar(range(n_feat), coverage_gaps, color=colors)
ax.axhline(y=0.05, color="black", linestyle="--", linewidth=1, label="Min discriminative gap (0.05)")
ax.axhline(y=0.0, color="gray", linestyle="-", linewidth=0.5)
ax.set_xlabel("Feature")
ax.set_ylabel("Coverage Gap (Correct - Wrong-plausible)")
ax.set_title("Feasibility Test 1: Discriminative Power per Feature")
ax.set_xticks(range(n_feat))
ax.set_xticklabels([f"F{i+1}" for i in range(n_feat)])
ax.legend()
plt.tight_layout()
plt.savefig("test1_coverage_gap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: test1_coverage_gap.png")

# PLOT 3: Score distributions for Feature 1
if feat1_R_distributions:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    r1 = results.get("feature_1", {})

    for idx_p, (name, label, color) in enumerate([
        ("correct", "Correct", "#2ecc71"),
        ("wrong_plausible", "Wrong-plausible", "#e67e22"),
        ("clearly_wrong", "Clearly wrong", "#e74c3c"),
    ]):
        ax = axes[idx_p]
        cal_key = f"{name}_cal_aa"
        test_key = f"{name}_test_aa"
        if cal_key in feat1_R_distributions and test_key in feat1_R_distributions:
            ax.hist(
                feat1_R_distributions[cal_key], bins=50, alpha=0.5, density=True,
                label="Cal", color="steelblue",
            )
            ax.hist(
                feat1_R_distributions[test_key], bins=50, alpha=0.5, density=True,
                label="Test", color=color,
            )
        cov = r1.get(f"{name}_aa_coverage", float("nan"))
        pv = r1.get(f"{name}_aa_p_value", float("nan"))
        ax.set_title(f"{label}\ncov={cov:.3f}, p={pv:.4f}")
        ax.set_xlabel("Nonconformity score R_AA")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle("Feature 1: Nonconformity Score Distributions (AA)", fontsize=13)
    plt.tight_layout()
    plt.savefig("test1_score_distributions_f1.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: test1_score_distributions_f1.png")

# PLOT 4: P-values comparison
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(n_feat)
width = 0.25
ax.bar(x - width, correct_pvalues_aa, width, label="Correct", color="#2ecc71")
ax.bar(x, wrong_pvalues_aa, width, label="Wrong-plausible", color="#e67e22")
ax.bar(x + width, random_pvalues_aa, width, label="Clearly wrong", color="#e74c3c")
ax.axhline(y=0.05, color="black", linestyle="--", linewidth=1, label="α = 0.05")
ax.set_xlabel("Feature")
ax.set_ylabel("Conformal p-value")
ax.set_title("Feasibility Test 1: Conformal P-values by Interpretation Type")
ax.set_xticks(x)
ax.set_xticklabels([f"F{i+1}" for i in range(n_feat)])
ax.legend()
plt.tight_layout()
plt.savefig("test1_pvalues.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: test1_pvalues.png")

# ============================================================
# STEP 8: Save JSON results
# ============================================================
print("\n" + "=" * 80)
print("STEP 8: Saving JSON results")
print("=" * 80)

results_clean = {}
for k, v in results.items():
    results_clean[k] = {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, np.ndarray))}

results_clean["summary"] = {
    "n_features_tested": n_feat,
    "n_tokens_total": int(N),
    "n_discover": int(len(idx_discover)),
    "n_cal": int(len(idx_cal)),
    "n_test": int(len(idx_test)),
    "alpha": alpha,
    "correct_coverages_aa": correct_coverages_aa.tolist(),
    "wrong_coverages_aa": wrong_coverages_aa.tolist(),
    "random_coverages_aa": random_coverages_aa.tolist(),
    "correct_pvalues_aa": correct_pvalues_aa.tolist(),
    "wrong_pvalues_aa": wrong_pvalues_aa.tolist(),
    "random_pvalues_aa": random_pvalues_aa.tolist(),
    "coverage_gaps": coverage_gaps.tolist(),
    "model": model_name,
    "sae_release": chosen_release,
    "sae_id": chosen_sae_id,
    "overall_verdict": "PASS" if len(kill_flags) == 0 else "FAIL",
    "kill_flags": kill_flags,
}

with open("test1_results.json", "w") as f:
    json.dump(results_clean, f, indent=2)
print("Saved: test1_results.json")

# Print compact version for easy copy-paste
print("\n\nCOMPACT RESULTS (copy this):")
print(json.dumps(results_clean["summary"], indent=2))

# Print file listing
print("\n\n" + "=" * 80)
print("OUTPUT FILES (download these from Kaggle):")
print("=" * 80)
print("  1. test1_coverage_by_type.png")
print("  2. test1_coverage_gap.png")
print("  3. test1_score_distributions_f1.png")
print("  4. test1_pvalues.png")
print("  5. test1_results.json")
print()
print("TO VIEW IMAGES INLINE, run this in the NEXT Kaggle cell:")
print("  from IPython.display import Image, display")
print("  for f in ['test1_coverage_by_type.png', 'test1_coverage_gap.png',")
print("            'test1_score_distributions_f1.png', 'test1_pvalues.png']:")
print("      print(f'\\n--- {f} ---')")
print("      display(Image(filename=f))")
print()
print("TO VIEW JSON, run in the NEXT Kaggle cell:")
print("  import json")
print("  with open('test1_results.json') as f: print(json.dumps(json.load(f), indent=2))")
print()
print("Done! Total features tested:", n_feat)
