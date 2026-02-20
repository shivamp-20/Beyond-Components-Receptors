# Dark Receptor Characterization + Predictive Receptor Margin
# Instructions for coding window

## TASK

Write `gp_dark_receptor_prm.py` that characterizes the dark receptor (L11H1sv1) 
discovered by OCA and tests whether it adds error-predictive power beyond the 
gender-aligned receptors.

## CONTEXT

I'm working on mechanistic interpretability of GPT-2 small on a gender pronoun (GP) 
prediction task. Previous experiments established:
- R1 (L10H9, sv0, polarity +1): promotion receptor, gender-aligned, AUC=0.964
- R3 (L9H7, sv1, polarity -1): inhibition receptor, gender-aligned, AUC=0.958
- L11H1sv1 (polarity +1): "dark receptor" — AUC=0.526 (near chance for gender) but 
  ablation drops accuracy by 8.17%. Causally critical, correlatively invisible.
- R2 (L11H8, sv6): confirmed ghost, ignore it.

The 3 independent directions (R1, R3, dark) capture R²=0.9815 of model logit variance.

## FILES AVAILABLE

- `svd_cache.pt` or `ov_svd_cache.pt` in `outputs/gp/`: SVD of OV matrices
- `test_gp.csv` in `data_main/`: test set (columns: prefix, pronoun, corr_prefix, corr_pronoun, name, corr_name)
- `train_gp_masks_and_dump_ov_logit_receptors_ddp.py`: has data loading code (expand_rows, etc.)
- `gp_exp4_receptor_shapley_ablation.py`: has receptor loading code
- TransformerLens installed, Kaggle T4 GPUs

## HOW TO RUN

```bash
python gp_dark_receptor_prm.py \
  --data_dir data_main \
  --csv test_gp.csv \
  --out_dir outputs/gp \
  --receptors "10,9,0,+1;9,7,1,-1" \
  --dark_receptor "11,1,1,+1" \
  --batch_size 64 \
  --device cuda \
  --use_both 1
```

Note: --receptors has R1 and R3 only (NOT R2). --dark_receptor is separate.

## WHAT THE SCRIPT DOES

### Step 0: Setup
- Load GPT-2 small via TransformerLens
- Load SVD cache
- Load receptor directions:
  - r1 = Vh[0] from ov[10][9], unit vector (768,)
  - r3 = Vh[1] from ov[9][7], unit vector (768,)
  - r_dark = Vh[1] from ov[11][1], unit vector (768,)
- Load test data with use_both=1 (clean + corrupted → 612 examples)
- Tokenize, compute decision positions (last token before padding)

### Step 1: Collect residuals and attention (ONE forward pass)
- Use model.run_with_cache to get:
  - Final residual stream at decision position: resid_final (N, 768)
  - Attention weights of L11H1 at decision position: attn_L11H1 (N, seq_len)
    Hook name: "blocks.11.attn.hook_pattern" → shape (N, n_heads, seq_len, seq_len)
    Extract head 1, at decision position row → (N, seq_len)
- Compute clean model predictions:
  - Apply ln_final to resid_final
  - Compute logits via W_U
  - he_logit = logits[:, he_id], she_logit = logits[:, she_id]
  - pred = (he_logit > she_logit).long()
  - label = y (from data: +1 for he, -1 for she → convert to binary: 1 for he, 0 for she)
  - correct = (pred == label)
  - error = ~correct
- Print baseline accuracy (should be ~91.34%)

### Step 2: Compute receptor activations
For each example i:
```python
g1[i] = resid_final[i] @ r1        # R1 activation (scalar)
g3[i] = resid_final[i] @ r3        # R3 activation (scalar)
g_dark[i] = resid_final[i] @ r_dark # dark receptor activation (scalar)
```
All are arrays of shape (N,).

### Step 3: Dark receptor token profile
Compute the logit-space profile of the dark receptor:
```python
# Get unembedding matrix
W_U = model.W_U  # shape (d_model, vocab_size) or (vocab_size, d_model) — check!
# In TransformerLens: model.W_U has shape (d_model, d_vocab)
# So: logit_profile = r_dark @ W_U → shape (vocab_size,)
logit_profile = r_dark @ model.W_U  # (vocab_size,)

# If there's a bias: logit_profile += model.b_U (but bias is same for all directions, 
# so it doesn't affect the ranking — skip it)

# Top 30 and bottom 30 tokens
top_idx = logit_profile.argsort(descending=True)[:30]
bot_idx = logit_profile.argsort(descending=False)[:30]
```
Print the tokens and their logit values.

Also compute for R1 and R3 for comparison:
```python
logit_r1 = r1 @ model.W_U
logit_r3 = r3 @ model.W_U
```
Print their top-10 tokens too (should be gendered — he/she/him/her etc.)

### Step 4: Dark receptor activation analysis
```python
# Split by context
he_mask = (y == 1)  # he-context examples
she_mask = (y == -1)  # she-context examples

# Stats
print(f"g_dark — he-context: mean={g_dark[he_mask].mean():.4f}, std={g_dark[he_mask].std():.4f}")
print(f"g_dark — she-context: mean={g_dark[she_mask].mean():.4f}, std={g_dark[she_mask].std():.4f}")
print(f"g_dark — correct: mean={g_dark[correct].mean():.4f}, std={g_dark[correct].std():.4f}")
print(f"g_dark — error: mean={g_dark[error].mean():.4f}, std={g_dark[error].std():.4f}")

# Correlation with sentence length (number of non-pad tokens)
seq_lengths = attn_mask.sum(dim=1).numpy()
corr_length = np.corrcoef(np.abs(g_dark), seq_lengths)[0,1]
print(f"Correlation |g_dark| vs sentence length: {corr_length:.4f}")
```

### Step 5: Attention pattern analysis for L11H1
```python
# attn_L11H1 has shape (N, seq_len) — attention weights at decision position
# Average over all examples
avg_attn = attn_L11H1.mean(dim=0)  # (seq_len,)

# Also split by correct vs error
avg_attn_correct = attn_L11H1[correct].mean(dim=0)
avg_attn_error = attn_L11H1[error].mean(dim=0)

# For each example, find the argmax attention position
# Map positions to token types: name, verb, pronoun placeholder, other
# We know the GP sentence structure, so relative positions are meaningful
```
For the attention plot, show the average attention pattern for a few representative examples,
and the average over all examples. Label the token positions.

### Step 6: Predictive Receptor Margin
```python
# y_binary: 1 for he, 0 for she (for AUC computation)
# y_signed: +1 for he, -1 for she (for margin computation)

# Gender margin (polarity-weighted)
# R1 pol=+1, R3 pol=-1
# gender_score = (+1)*g1 + (-1)*g3 = g1 - g3
# When gender_score > 0 → predict he, < 0 → predict she
gender_score = g1 - g3  # shape (N,)

# Signed margin: positive = agrees with label, negative = disagrees
signed_margin = y_signed * gender_score  # (N,)

# PRM₂: confidence from gender receptors only
prm2 = np.abs(gender_score)  # |g1 - g3|

# PRM₃: confidence from gender + dark
prm3 = prm2 * np.abs(g_dark)  # |g1 - g3| × |g_dark|

# PRM_dark: dark receptor alone
prm_dark = np.abs(g_dark)

# Error prediction: lower confidence → higher error risk
# So we use NEGATIVE confidence as the "error risk score"
# AUC-ROC: predict error (binary) from -prm2, -prm3, -prm_dark
from sklearn.metrics import roc_auc_score, precision_recall_curve

error_binary = error.numpy().astype(int)  # 1 = error, 0 = correct

auc_prm2 = roc_auc_score(error_binary, -prm2)
auc_prm3 = roc_auc_score(error_binary, -prm3)
auc_dark = roc_auc_score(error_binary, -prm_dark)

# Also try: dark receptor sign matters? Maybe it's not |g_dark| but g_dark itself
# Test both signs
auc_dark_pos = roc_auc_score(error_binary, -g_dark)   # low g_dark → error
auc_dark_neg = roc_auc_score(error_binary, g_dark)     # high g_dark → error
auc_dark_abs = roc_auc_score(error_binary, -np.abs(g_dark))  # low |g_dark| → error
print(f"Dark receptor error AUC: -g_dark={auc_dark_pos:.4f}, +g_dark={auc_dark_neg:.4f}, -|g_dark|={auc_dark_abs:.4f}")
# Use whichever is highest for the rest

print(f"\nError Prediction AUC-ROC:")
print(f"  PRM₂ (gender only):        {auc_prm2:.4f}")
print(f"  PRM₃ (gender × dark):      {auc_prm3:.4f}")
print(f"  PRM_dark (dark only):       {auc_dark:.4f}")
print(f"  Improvement (PRM₃ - PRM₂): {auc_prm3 - auc_prm2:+.4f}")
```

### Step 7: Precision@k curves
```python
n_errors = error_binary.sum()
N = len(error_binary)

# For each predictor, rank examples from most error-prone to least
# (lowest confidence first)
for name, scores in [("PRM2", prm2), ("PRM3", prm3), ("PRM_dark", prm_dark)]:
    order = np.argsort(scores)  # ascending = lowest confidence first
    cumulative_errors = np.cumsum(error_binary[order])
    precision_at_k = cumulative_errors / np.arange(1, N+1)
    # Store for plotting
```

### Step 8: Error type decomposition
```python
# Thresholds: 25th percentile of each metric across ALL examples (not just errors)
gender_thr = np.percentile(prm2, 25)
dark_thr = np.percentile(prm_dark, 25)

# Classify ALL examples into quadrants
type_G  = (prm2 < gender_thr) & (prm_dark >= dark_thr)   # gender weak, dark ok
type_D  = (prm2 >= gender_thr) & (prm_dark < dark_thr)    # gender ok, dark weak
type_GD = (prm2 < gender_thr) & (prm_dark < dark_thr)     # both weak
type_OK = (prm2 >= gender_thr) & (prm_dark >= dark_thr)   # both ok

print(f"\n=== ERROR TYPE DECOMPOSITION ===")
print(f"Gender threshold (25th pctile of |g1-g3|): {gender_thr:.4f}")
print(f"Dark threshold (25th pctile of |g_dark|): {dark_thr:.4f}")
print(f"")
for name, mask in [("Type G (gender fail)", type_G), 
                    ("Type D (dark fail)", type_D),
                    ("Type GD (both fail)", type_GD), 
                    ("Type OK (neither)", type_OK)]:
    n_total = mask.sum()
    n_err = (mask & error_binary.astype(bool)).sum()
    rate = n_err / n_total * 100 if n_total > 0 else 0
    print(f"  {name}: {n_total} examples, {n_err} errors ({rate:.1f}% error rate)")

# Also report: of all errors, what fraction are each type?
print(f"\nAmong {n_errors} total errors:")
for name, mask in [("Type G", type_G), ("Type D", type_D), 
                    ("Type GD", type_GD), ("Type OK", type_OK)]:
    n_err = (mask & error_binary.astype(bool)).sum()
    frac = n_err / n_errors * 100 if n_errors > 0 else 0
    print(f"  {name}: {n_err} ({frac:.1f}%)")
```

### Step 9: Cosine checks (sanity)
```python
print(f"\n=== RECEPTOR GEOMETRY ===")
print(f"cos(R1, R3) = {(r1 @ r3).item():.4f}")
print(f"cos(R1, dark) = {(r1 @ r_dark).item():.4f}")
print(f"cos(R3, dark) = {(r3 @ r_dark).item():.4f}")
# Dark should be nearly orthogonal to both R1 and R3 (cos ≈ 0)
```

## PLOTS (save as PNG)

### Plot 1: Dark receptor token profile
- Two subplots side by side
- Left: bar chart of top-20 tokens by logit_dark value (label each bar with token string)
- Right: bar chart of bottom-20 tokens (i.e. most negative logit_dark values)
- Title: "L11H1sv1 (Dark Receptor) — Logit Space Profile"
- For comparison, annotate where "he", "she", "him", "her" fall in the ranking

### Plot 2: 2D Error Landscape (THE MONEY FIGURE)
- x-axis: |PRM₂| = |g₁ - g₃| (gender margin)
- y-axis: |g_dark| (dark receptor activation)
- Blue dots: correct predictions
- Red dots: errors (make them larger/brighter so visible against blue)
- Draw dashed lines at the 25th percentile thresholds (gender_thr horizontal, dark_thr vertical)
  Actually: gender_thr should be VERTICAL (x-axis) and dark_thr should be HORIZONTAL (y-axis)
- Label the four quadrants: "G fail" (left-top), "D fail" (right-bottom), "GD fail" (left-bottom), "OK" (right-top)
- Title: "Error Landscape: Gender Margin vs Dark Receptor"

### Plot 3: Precision@k Curves
- x-axis: k (number of examples flagged, from 1 to N)
- y-axis: precision = (# actual errors in top-k) / k
- Three lines: PRM₂ (red), PRM₃ (blue), PRM_dark (green)
- Also draw a horizontal dashed line at the base error rate (n_errors/N ≈ 8.7%)
- Title: "Error Prediction: Precision@k"
- Add legend with AUC-ROC values in the label

### Plot 4: Error Type Distribution
- Two subplots:
- Left: stacked bar or pie chart showing ALL examples by type (G, D, GD, OK) with error counts overlaid
- Right: bar chart showing error RATE per type (% of examples in that type that are errors)
- Title: "Error Types by Receptor Failure Mode"

### Plot 5: L11H1 Attention Pattern
- Two subplots:
- Left: average attention pattern (over all examples) as a bar chart over positions
  x-axis = relative token position (0, 1, 2, ..., seq_len-1)
  y-axis = attention weight
  Color or label notable positions: subject name, verb, object, pronoun position
  IMPORTANT: Since examples have different lengths and structures, this is tricky.
  Simplest approach: for the MOST COMMON sentence structure (likely structure 0),
  filter to only those examples, and plot the attention pattern.
  Print the actual tokens for one representative example so we can label the x-axis.
- Right: same but split by correct (blue) vs error (red)
- Title: "L11H1 Attention at Decision Position"

### Plot 6: Activation distributions
- Three subplots in a row:
- Left: histogram of g1 values, split by he-context (blue) vs she-context (red)
- Middle: histogram of g3 values, same split
- Right: histogram of g_dark values, same split
- The first two should show clear separation (gendered). The third should show overlap (not gendered).
- Title: "Receptor Activation Distributions by Gender Context"

## OUTPUT FORMAT

Save all numerical results to `outputs/gp/dark_receptor_prm_results.json`:
```json
{
  "baseline_accuracy": 0.9134,
  "n_examples": 612,
  "n_errors": 54,
  "receptor_cosines": {"r1_r3": ..., "r1_dark": ..., "r3_dark": ...},
  "dark_token_profile": {"top20": [...], "bottom20": [...]},
  "error_prediction_auc": {"prm2": ..., "prm3": ..., "dark_only": ...},
  "error_types": {
    "type_G": {"n_total": ..., "n_errors": ..., "error_rate": ...},
    "type_D": {"n_total": ..., "n_errors": ..., "error_rate": ...},
    "type_GD": {"n_total": ..., "n_errors": ..., "error_rate": ...},
    "type_OK": {"n_total": ..., "n_errors": ..., "error_rate": ...}
  },
  "dark_activation_stats": {
    "he_context": {"mean": ..., "std": ...},
    "she_context": {"mean": ..., "std": ...},
    "correct": {"mean": ..., "std": ...},
    "error": {"mean": ..., "std": ...}
  }
}
```

## SANITY CHECKS

1. Baseline accuracy should be ~91.34% (54 errors out of 612)
2. cos(R1, dark) and cos(R3, dark) should both be small (< 0.1) — confirming dark is orthogonal
3. R1 and R3 token profiles should show gendered tokens (he/she/him/her in top-10)
4. Dark receptor token profile should NOT show gendered tokens prominently
5. PRM₂ AUC should be meaningfully above 0.5 (gender margin does predict errors)
6. Type OK errors should have the lowest error rate of all types
7. g_dark should NOT separate cleanly by he/she context (that's the whole point — it's not gendered)
