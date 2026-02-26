# Phase 2A Follow-Up: Dark Receptor IOI Characterization

## Why We're Doing This

Phase 2A revealed that the dark receptor (L11H1sv1) — which was "dark" for GP gender (AUC=0.526) — has transfer AUC=0.776 on IOI, explaining 11.5% of IOI logit variance (corr=−0.339). This is the strongest individual finding in the paper. But before we can claim it in a paper, we need to:

1. **Rule out confounds** — is the signal driven by sentence length, position, or template type rather than genuine relational-role encoding?
2. **Establish causality** — correlation isn't causation. Does REMOVING the dark receptor direction from IOI residual streams actually hurt IOI performance? (The GP causal test gave AccDrop=8.17% for dark. What's the IOI equivalent?)
3. **Identify what L11H1 specifically contributes** — the layer curve jumps from 0.653 (layer 10) to 0.776 (layer 11). How much of the signal is accumulated from earlier heads vs written by L11H1 at layer 11?

This follow-up addresses all three. It transforms the finding from "interesting correlation" to "causal cross-task mechanism."

---

## What Makes This Novel

No existing mech interp paper has shown that a direction identified as causally important by one task's circuit analysis is ALSO causally important for a different task — where the direction carries NO discriminative signal for the original task's label. The dark receptor has:
- GP AUC = 0.526 (no gender discrimination)
- GP AccDrop = 8.17% (causally important for GP)
- IOI transfer AUC = 0.776 (predicts IOI correctness)
- IOI causal effect = ? (this experiment)

If IOI AccDrop is meaningful, we have a direction that is causally important for TWO different tasks despite being "dark" (non-discriminative) for the original task's label. This is a new type of circuit component: **task-general structural infrastructure**.

---

## The Mathematics

### Part 1: Confound Controls

We need to show that the dark receptor's IOI correlation isn't driven by surface features.

**Confound A: Sentence length.** IOI sentences range from 12-18 words (mean 14.9, std 1.6). If dark activation correlates with sentence length, and sentence length correlates with IOI difficulty, the dark-IOI association is confounded.

$$r_{\text{confound}} = \text{corr}(g_{\text{dark}}^{(\text{final})}, \; \text{len}_i)$$

If |r| < 0.1, sentence length is not a confound.

**Partial correlation:** Remove the linear effect of sentence length from both dark activation and IOI logit diff, then measure the residual correlation:

$$r_{\text{partial}} = \text{corr}(g_{\text{dark}} - \hat{g}_{\text{dark|len}}, \; \Delta\text{logit} - \widehat{\Delta\text{logit}}_{\text{|len}})$$

where $\hat{g}_{\text{dark|len}}$ is the OLS prediction of dark activation from sentence length. If $r_{\text{partial}} \approx r_{\text{raw}}$ (≈ −0.339), the signal survives controlling for length.

**Confound B: Template type.** Three templates: "gave" (N=690), "decided to give" (N=230), "wanted to give" (N=79). Compute dark transfer AUC separately for each template. If AUC is similar across templates → not a template confound.

**Confound C: IO position.** IO appears first in 478/1000 examples, second in 522. Compute dark transfer AUC for IO-first vs IO-second. If similar → not a position confound.

**Confound D: Prediction position.** The token index of the prediction position varies. Measure:

$$r_{\text{pos}} = \text{corr}(g_{\text{dark}}^{(\text{final})}, \; \text{pred\_pos}_i)$$

### Part 2: Causal Ablation on IOI

This is the most important part. For each receptor k ∈ {R1, R3, Dark}:

1. Take the cached final residual stream: $x_i^{(\text{final})} \in \mathbb{R}^{768}$
2. Project out the receptor direction:

$$\tilde{x}_i = x_i - (x_i^\top v_k) \, v_k$$

3. Apply layer normalization:

$$\hat{x}_i = \text{LayerNorm}(\tilde{x}_i)$$

Note: LayerNorm parameters (weight γ, bias β) come from model.ln_final.

4. Compute logits:

$$\text{logits}_i = \hat{x}_i \cdot W_U + b_U$$

5. Extract IO and S logits, compute new IOI accuracy:

$$\text{IOI\_Acc}_{\text{ablated}} = \frac{1}{N}\sum_i \mathbf{1}[\text{logits}_i[\text{IO}_i] > \text{logits}_i[\text{S}_i]]$$

6. Compute AccDrop:

$$\text{AccDrop}_k^{\text{IOI}} = \text{IOI\_Acc}_{\text{original}} - \text{IOI\_Acc}_{\text{ablated}}$$

**The key comparison table:**

| Receptor | GP AccDrop | IOI AccDrop |
|----------|-----------|------------|
| R1 | 3.59% | ? |
| R3 | 1.31% | ? |
| Dark | 8.17% | ? |

**Predictions:**
- R1 IOI AccDrop ≈ 0% (R1 is GP-specific, removing it shouldn't hurt IOI)
- R3 IOI AccDrop ≈ 0% (same reasoning)
- Dark IOI AccDrop > 0% (if the correlation is causal)

If Dark IOI AccDrop is meaningful (say 1-5%), we have proved causal cross-task relevance.

**Cross-task Causal Fidelity Ratio:**

$$\text{CFR}_k^{\text{IOI}} = \frac{\text{IOI\_AccDrop}_k}{\text{IOI\_AUC}_k - 0.5}$$

For the dark receptor on GP: CFR = 8.17% / (0.526 - 0.5) = 3.14 (very high — big causal effect per unit of discriminability).
For the dark receptor on IOI: CFR = IOI_AccDrop / (0.776 - 0.5) = IOI_AccDrop / 0.276.

### Part 3: Layer Delta Analysis

The dark receptor's IOI AUC jumps from 0.653 (layer 10) to 0.776 (layer 11). Layer 11 contains L11H1 (the dark receptor's host head). The jump tells us how much L11H1 specifically contributes.

**Delta computation:** For each example i, compute:

$$\delta_i = g_{\text{dark}}^{(11)}(i) - g_{\text{dark}}^{(10)}(i)$$

This is the change in dark activation caused by layer 11 processing (which includes L11H1 and MLP11 and layer norm). If δ correlates with IOI logit diff, then layer 11 is specifically writing IOI-relevant information along v_dark.

$$r_{\delta} = \text{corr}(\delta_i, \; \Delta\text{logit}_{\text{IOI},i})$$

Also compute: AUC of δ for predicting IOI correctness.

**Fraction of signal from L11H1:** What fraction of the dark receptor's total IOI-relevant signal comes from the layer 10→11 jump?

$$\text{L11\_fraction} = \frac{r_{\delta}^2}{r_{\text{total}}^2} \approx \frac{\text{corr}(\delta, \Delta\text{logit})^2}{\text{corr}(g_{\text{dark}}^{(11)}, \Delta\text{logit})^2}$$

If this is high (>0.5), most of the dark receptor's IOI signal is written at layer 11. If low, it's accumulated from earlier layers.

### Part 4: IOI Logit Diff Decomposition

Similar to GP's Type D gap analysis. Decompose IOI logit difference into dark receptor component and residual:

$$\Delta\text{logit}_{\text{IOI}} = \beta_d \cdot g_{\text{dark}} + \epsilon$$

where β_d comes from OLS regression. Report:
- β_d (effect size)
- R² = 0.1149 (already known)
- For the 28 IOI errors: what is the dark receptor activation? Are errors concentrated at extreme dark values?

---

## What We Expect

| Test | Expected if signal is real | Expected if confound |
|------|--------------------------|---------------------|
| Sentence length confound | |r| < 0.1 | |r| > 0.3 |
| Partial corr (controlling length) | ≈ −0.33 (unchanged) | drops toward 0 |
| Template split AUC | Similar across templates | Varies greatly |
| IO position split AUC | Similar for first/second | Differs |
| Dark IOI AccDrop | 1-5% (meaningful) | ≈ 0% |
| R1 IOI AccDrop | ≈ 0% | — |
| R3 IOI AccDrop | ≈ 0% | — |
| Layer delta corr | Significant | Near 0 |

**What could go wrong:**
1. **Dark AccDrop ≈ 0 on IOI.** Would mean the correlation is real but not causal — the residual stream carries IOI signal along v_dark, but IOI has redundant pathways that compensate. Still publishable as a correlation finding, but weaker.
2. **Sentence length confound.** Would reduce the story to "dark receptor encodes sentence complexity." Still interesting but less novel. Partial correlation would tell us.
3. **Template confound.** If dark AUC is 0.9 for "gave" but 0.5 for "decided to give," it's encoding template identity. Less interesting.

---

## Outputs

### Numbers to Print

```
============================================================
PHASE 2A FOLLOW-UP: DARK RECEPTOR IOI CHARACTERIZATION
============================================================

=== PART 1: CONFOUND CONTROLS ===
Sentence length:
  corr(g_dark, sentence_length):      X.XXXX
  corr(IOI_logit_diff, sent_length):  X.XXXX
  Partial corr (dark, IOI | length):  X.XXXX  (raw: -0.3389)

Prediction position:
  corr(g_dark, pred_position):        X.XXXX

Template split:
  Template "gave" (N=690):         dark AUC = X.XXXX
  Template "decided" (N=230):      dark AUC = X.XXXX
  Template "wanted" (N=79):        dark AUC = X.XXXX

IO position:
  IO first (N=478):  dark AUC = X.XXXX
  IO second (N=522): dark AUC = X.XXXX

=== PART 2: CAUSAL ABLATION ===
Original IOI accuracy:  97.20%
After projecting out R1:    XX.XX%  (AccDrop = X.XX%)
After projecting out R3:    XX.XX%  (AccDrop = X.XX%)
After projecting out Dark:  XX.XX%  (AccDrop = X.XX%)

Cross-task CFR:
  Dark GP CFR:  3.14
  Dark IOI CFR: X.XX

=== PART 3: LAYER DELTA ===
Layer 10→11 delta stats:
  mean(δ):  X.XX
  std(δ):   X.XX
  corr(δ, IOI_logit_diff):  X.XXXX
  AUC(δ, IOI_correct):      X.XXXX
  L11 fraction:              X.XX%

=== PART 4: ERROR ANALYSIS ===
28 IOI errors:
  Dark mean (errors):  X.XX
  Dark mean (correct): X.XX
  Dark diff (d'):      X.XX
============================================================
```

### Plots (5 plots)

**Plot 1: Confound control summary (4-panel)**
- Panel A: scatter of dark activation vs sentence length (with correlation annotated)
- Panel B: bar chart of dark AUC by template type
- Panel C: bar chart of dark AUC by IO position (first vs second)
- Panel D: scatter of dark activation vs prediction position
- Title: "Dark Receptor IOI Signal: Confound Controls"

**Plot 2: Causal ablation comparison (bar chart)**
- Grouped bars: GP AccDrop vs IOI AccDrop for each receptor (R1, R3, Dark)
- 6 bars total (2 per receptor)
- Title: "Causal Ablation: GP vs IOI"

**Plot 3: Layer delta analysis**
- Left panel: scatter of δ (layer 10→11 dark delta) vs IOI logit diff
- Right panel: histogram of δ for IOI-correct vs IOI-error
- Title: "Layer 11 Contribution to Dark Receptor IOI Signal"

**Plot 4: Cross-task causal profile (the money figure)**
- 3×2 grid showing: rows = R1, R3, Dark; columns = GP, IOI
- Each cell: a small bar showing AccDrop and AUC
- Shows at a glance: R1/R3 are GP-specific (high GP AccDrop, zero IOI AccDrop), Dark is cross-task (meaningful AccDrop in both)
- Title: "Cross-Task Causal Profile of Three Circuit Channels"

**Plot 5: Dark receptor activation by IOI error status**
- Violin or box plot: dark activation for IOI-correct vs IOI-error
- With individual points overlaid (especially for the 28 errors)
- Title: "Dark Receptor Activation: IOI Correct vs Error"

---

## Pseudocode

```
PHASE 2A FOLLOW-UP: DARK RECEPTOR IOI CHARACTERIZATION
========================================================
Inputs needed:
  - GPT-2 Small model (TransformerLens HookedTransformer)
  - svd_cache.pt
  - test_1k_ioi.csv
  - Phase 2A cached data:
      * all_residuals (1000, 13, 768) — if saved from Phase 2A
      * io_token_ids, s_token_ids — if saved
      * all_logits_io, all_logits_s — if saved
      * ioi_correct — if saved
    If NOT saved: we need to rerun forward passes (Phase 2A code)

  NOTE TO CODING CLAUDE: If you have the Phase 2A tensors saved
  (e.g., as a .pt file), load them. If not, run the forward pass
  loop from Phase 2A first to collect all_residuals, then proceed.

Constants:
  R1 = (10, 9, 0), R3 = (9, 7, 1), Dark = (11, 1, 1)
  GP AccDrop: R1=3.59%, R3=1.31%, Dark=8.17%
  IOI original accuracy: 97.20%
  Dark corr with IOI logit diff: -0.3389
  Dark R²: 0.1149

SETUP:
======
import torch, numpy as np, pandas as pd
from transformer_lens import HookedTransformer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

model = HookedTransformer.from_pretrained("gpt2-small")
tokenizer = model.tokenizer
svd_cache = torch.load("svd_cache.pt")

# Extract receptor directions (same as Phase 2A)
def get_direction(layer, head, sv_idx):
    V = svd_cache[(layer, head)]['V']
    v = V[sv_idx, :]
    return v / v.norm()

v_R1 = get_direction(10, 9, 0)
v_R3 = get_direction(9, 7, 1)
v_dark = get_direction(11, 1, 1)

# Load IOI data
ioi_df = pd.read_csv("data_main/test_1k_ioi.csv")
n_examples = len(ioi_df)

# === EITHER load cached Phase 2A data OR rerun forward passes ===
# If cached:
# all_residuals = torch.load("phase2a_residuals.pt")  # (1000, 13, 768)
# io_token_ids = torch.load("phase2a_io_tokens.pt")
# s_token_ids = torch.load("phase2a_s_tokens.pt")
# all_logits_io = torch.load("phase2a_logits_io.pt")
# all_logits_s = torch.load("phase2a_logits_s.pt")
# ioi_correct = (all_logits_io > all_logits_s).numpy().astype(int)

# If NOT cached, rerun Phase 2A forward passes to collect all_residuals
# [insert Phase 2A forward pass code here]

# Compute IOI logit diff and receptor projections
ioi_logit_diff = (all_logits_io - all_logits_s).numpy()
ioi_correct = (all_logits_io > all_logits_s).numpy().astype(int)

g_dark_all_layers = (all_residuals @ v_dark.cpu()).numpy()  # (1000, 13)
g_dark_final = g_dark_all_layers[:, -1]
g_dark_layer10 = g_dark_all_layers[:, 10]  # layer index 10 = post block 10
g_R1_final = (all_residuals[:, -1, :] @ v_R1.cpu()).numpy()
g_R3_final = (all_residuals[:, -1, :] @ v_R3.cpu()).numpy()

def safe_auc(labels, scores):
    try:
        auc = roc_auc_score(labels, scores)
        return max(auc, 1 - auc)
    except:
        return 0.5


==============================
PART 1: CONFOUND CONTROLS
==============================

# Sentence length
sentence_lengths = np.array([len(row['ioi_sentences_input'].split()) 
                             for _, row in ioi_df.iterrows()])

# Prediction position (token count)
pred_positions = np.array([len(tokenizer.encode(row['ioi_sentences_input'])) 
                           for _, row in ioi_df.iterrows()])

# Template type
templates = []
for _, row in ioi_df.iterrows():
    inp = row['ioi_sentences_input']
    if 'decided to give' in inp:
        templates.append('decided')
    elif 'wanted to give' in inp:
        templates.append('wanted')
    elif 'gave' in inp:
        templates.append('gave')
    else:
        templates.append('other')
templates = np.array(templates)

# IO position (first or second mention)
io_is_first = []
for _, row in ioi_df.iterrows():
    inp = row['ioi_sentences_input']
    io = row['ioi_sentences_labels']
    s = row['ioi_sentences_labels_wrong']
    io_is_first.append(inp.find(io) < inp.find(s))
io_is_first = np.array(io_is_first)

# --- Sentence length confound ---
r_dark_len = np.corrcoef(g_dark_final, sentence_lengths)[0, 1]
r_logit_len = np.corrcoef(ioi_logit_diff, sentence_lengths)[0, 1]
PRINT: f"corr(g_dark, sentence_length): {r_dark_len:.4f}"
PRINT: f"corr(IOI_logit_diff, sentence_length): {r_logit_len:.4f}"

# Partial correlation: dark vs IOI logit diff, controlling for sentence length
# Residualize both variables on sentence length
reg_dark = LinearRegression().fit(sentence_lengths.reshape(-1,1), g_dark_final)
resid_dark = g_dark_final - reg_dark.predict(sentence_lengths.reshape(-1,1))
reg_logit = LinearRegression().fit(sentence_lengths.reshape(-1,1), ioi_logit_diff)
resid_logit = ioi_logit_diff - reg_logit.predict(sentence_lengths.reshape(-1,1))
partial_corr_len = np.corrcoef(resid_dark, resid_logit)[0, 1]
PRINT: f"Partial corr (dark, IOI_logit | sentence_length): {partial_corr_len:.4f} (raw: -0.3389)"

# --- Prediction position confound ---
r_dark_pos = np.corrcoef(g_dark_final, pred_positions)[0, 1]
r_logit_pos = np.corrcoef(ioi_logit_diff, pred_positions)[0, 1]
PRINT: f"corr(g_dark, pred_position): {r_dark_pos:.4f}"
PRINT: f"corr(IOI_logit_diff, pred_position): {r_logit_pos:.4f}"

# Partial correlation controlling for BOTH length and position
X_confounds = np.column_stack([sentence_lengths, pred_positions])
reg_dark2 = LinearRegression().fit(X_confounds, g_dark_final)
resid_dark2 = g_dark_final - reg_dark2.predict(X_confounds)
reg_logit2 = LinearRegression().fit(X_confounds, ioi_logit_diff)
resid_logit2 = ioi_logit_diff - reg_logit2.predict(X_confounds)
partial_corr_both = np.corrcoef(resid_dark2, resid_logit2)[0, 1]
PRINT: f"Partial corr (dark, IOI | length+position): {partial_corr_both:.4f}"

# --- Template type split ---
for tmpl in ['gave', 'decided', 'wanted']:
    mask = (templates == tmpl)
    n = mask.sum()
    if n > 20:
        auc = safe_auc(ioi_correct[mask], g_dark_final[mask])
        r = np.corrcoef(g_dark_final[mask], ioi_logit_diff[mask])[0, 1]
        PRINT: f"Template '{tmpl}' (N={n}): dark AUC={auc:.4f}, corr={r:.4f}"

# --- IO position split ---
for label, mask in [("IO first", io_is_first), ("IO second", ~io_is_first)]:
    n = mask.sum()
    auc = safe_auc(ioi_correct[mask], g_dark_final[mask])
    r = np.corrcoef(g_dark_final[mask], ioi_logit_diff[mask])[0, 1]
    PRINT: f"{label} (N={n}): dark AUC={auc:.4f}, corr={r:.4f}"


==============================
PART 2: CAUSAL ABLATION ON IOI
==============================

# For each receptor, project out its direction from the final residual,
# apply ln_final + W_U, and recompute IOI accuracy.

# Get model components for manual logit computation
ln_final_weight = model.ln_final.w.cpu()  # (768,)
ln_final_bias = model.ln_final.b.cpu()    # (768,)
W_U = model.W_U.cpu()                      # (768, vocab_size)
b_U = model.b_U.cpu()                      # (vocab_size,)

# Manual layer norm
def manual_ln(x, weight, bias):
    # x: (batch, 768) or (768,)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / (var + 1e-5).sqrt()
    return x_norm * weight + bias

# Get final residual streams (post last block, before ln_final)
final_resid = all_residuals[:, -1, :].clone()  # (1000, 768) — NOTE: check if this 
# is pre-ln_final or post-ln_final. In TransformerLens, blocks.11.hook_resid_post 
# is AFTER block 11 but BEFORE ln_final. This is what we want.

# Original logits (sanity check)
original_logits = manual_ln(final_resid, ln_final_weight, ln_final_bias) @ W_U + b_U
original_io_logits = torch.tensor([original_logits[i, io_token_ids[i]] for i in range(n_examples)])
original_s_logits = torch.tensor([original_logits[i, s_token_ids[i]] for i in range(n_examples)])
original_acc = (original_io_logits > original_s_logits).float().mean().item()
PRINT: f"Sanity check - reconstructed IOI accuracy: {original_acc:.4f} (should be ~0.972)"

# Causal ablation for each receptor
for name, v_k in [("R1", v_R1), ("R3", v_R3), ("Dark", v_dark)]:
    v_k_cpu = v_k.cpu()
    
    # Project out v_k from each example's final residual
    projections = (final_resid @ v_k_cpu).unsqueeze(1)  # (1000, 1)
    ablated_resid = final_resid - projections * v_k_cpu.unsqueeze(0)  # (1000, 768)
    
    # Recompute logits
    ablated_logits = manual_ln(ablated_resid, ln_final_weight, ln_final_bias) @ W_U + b_U
    
    # Extract IO and S logits
    abl_io = torch.tensor([ablated_logits[i, io_token_ids[i]] for i in range(n_examples)])
    abl_s = torch.tensor([ablated_logits[i, s_token_ids[i]] for i in range(n_examples)])
    
    abl_acc = (abl_io > abl_s).float().mean().item()
    acc_drop = original_acc - abl_acc
    
    # Also compute ablated logit diff
    abl_logit_diff = (abl_io - abl_s).numpy()
    mean_logit_diff_drop = ioi_logit_diff.mean() - abl_logit_diff.mean()
    
    PRINT: f"{name}: IOI Acc after ablation = {abl_acc:.4f}, AccDrop = {acc_drop:.4f} ({acc_drop*100:.2f}%)"
    PRINT: f"  Mean logit diff: {abl_logit_diff.mean():.4f} (original: {ioi_logit_diff.mean():.4f}, drop: {mean_logit_diff_drop:.4f})"

# Cross-task CFR for dark receptor
# GP CFR = GP_AccDrop / (GP_AUC - 0.5) = 0.0817 / (0.526 - 0.5) = 3.14
gp_cfr_dark = 0.0817 / (0.526 - 0.5)
# IOI CFR = IOI_AccDrop / (IOI_AUC - 0.5)
ioi_acc_drop_dark = ...  # from above computation
ioi_cfr_dark = ioi_acc_drop_dark / (0.776 - 0.5) if ioi_acc_drop_dark > 0 else 0
PRINT: f"Cross-task CFR (Dark): GP={gp_cfr_dark:.2f}, IOI={ioi_cfr_dark:.2f}"


==============================
PART 3: LAYER DELTA ANALYSIS
==============================

# Delta = dark activation at layer 11 minus layer 10
# This isolates what layer 11 (including L11H1) contributes

# NOTE on layer indexing:
# all_residuals[:, l, :] = residual after block l (0-indexed)
# So layer 10 = all_residuals[:, 10, :], layer 11 = all_residuals[:, 11, :]
# The dark receptor's host head L11H1 is in block 11.
# delta = g_dark(after block 11) - g_dark(after block 10)

delta_dark = g_dark_all_layers[:, 11] - g_dark_all_layers[:, 10]

PRINT: f"Layer 10→11 delta: mean={delta_dark.mean():.4f}, std={delta_dark.std():.4f}"

# Correlation of delta with IOI logit diff
r_delta = np.corrcoef(delta_dark, ioi_logit_diff)[0, 1]
PRINT: f"corr(delta_dark, IOI_logit_diff): {r_delta:.4f}"

# AUC of delta for predicting IOI correctness
delta_auc = safe_auc(ioi_correct, delta_dark)
PRINT: f"AUC(delta_dark, IOI_correct): {delta_auc:.4f}"

# L11 fraction: what fraction of the dark signal comes from layer 11?
r_total = np.corrcoef(g_dark_final, ioi_logit_diff)[0, 1]  # should be ~-0.339
l11_fraction = (r_delta**2) / (r_total**2) if r_total != 0 else 0
PRINT: f"L11 fraction (r²_delta / r²_total): {l11_fraction:.4f}"

# Also: correlation of g_dark at layer 10 with IOI logit diff
r_layer10 = np.corrcoef(g_dark_all_layers[:, 10], ioi_logit_diff)[0, 1]
PRINT: f"corr(g_dark_layer10, IOI_logit_diff): {r_layer10:.4f}"
PRINT: f"corr(g_dark_layer11, IOI_logit_diff): {r_total:.4f}"
PRINT: f"Signal gain from L11: {abs(r_total) - abs(r_layer10):.4f}"


==============================
PART 4: ERROR ANALYSIS
==============================

correct_mask = ioi_correct.astype(bool)
error_mask = ~correct_mask

PRINT: f"N errors: {error_mask.sum()}"
PRINT: f"Dark mean (correct): {g_dark_final[correct_mask].mean():.4f} (std={g_dark_final[correct_mask].std():.4f})"
PRINT: f"Dark mean (error):   {g_dark_final[error_mask].mean():.4f} (std={g_dark_final[error_mask].std():.4f})"

# Cohen's d for dark receptor between correct and error
d_prime = (g_dark_final[correct_mask].mean() - g_dark_final[error_mask].mean()) / g_dark_final.std()
PRINT: f"Cohen's d (correct vs error): {d_prime:.4f}"

# For errors: which templates?
error_templates = templates[error_mask]
for tmpl in ['gave', 'decided', 'wanted']:
    n_err = (error_templates == tmpl).sum()
    n_total = (templates == tmpl).sum()
    PRINT: f"  Template '{tmpl}': {n_err}/{n_total} errors ({n_err/n_total*100:.1f}%)"


==============================
PLOTS
==============================

# PLOT 1: Confound controls (2x2 panel)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: dark vs sentence length
ax = axes[0, 0]
ax.scatter(sentence_lengths + np.random.uniform(-0.2, 0.2, n_examples), 
           g_dark_final, alpha=0.3, s=10, c='steelblue')
ax.set_xlabel("Sentence length (words)")
ax.set_ylabel("g_dark (final layer)")
ax.set_title(f"Dark vs Length (r={r_dark_len:.3f})")

# Panel B: dark AUC by template
ax = axes[0, 1]
tmpl_names = ['gave', 'decided', 'wanted']
tmpl_aucs = [safe_auc(ioi_correct[templates==t], g_dark_final[templates==t]) for t in tmpl_names]
tmpl_ns = [(templates==t).sum() for t in tmpl_names]
bars = ax.bar(range(3), tmpl_aucs, color=['steelblue', 'coral', 'gray'])
ax.set_xticks(range(3))
ax.set_xticklabels([f"{t}\n(N={n})" for t, n in zip(tmpl_names, tmpl_ns)])
ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel("Dark Transfer AUC")
ax.set_title("Dark AUC by Template Type")
for i, v in enumerate(tmpl_aucs):
    ax.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=10)

# Panel C: dark AUC by IO position
ax = axes[1, 0]
auc_first = safe_auc(ioi_correct[io_is_first], g_dark_final[io_is_first])
auc_second = safe_auc(ioi_correct[~io_is_first], g_dark_final[~io_is_first])
n_first = io_is_first.sum()
n_second = (~io_is_first).sum()
bars = ax.bar([0, 1], [auc_first, auc_second], color=['steelblue', 'coral'])
ax.set_xticks([0, 1])
ax.set_xticklabels([f"IO first\n(N={n_first})", f"IO second\n(N={n_second})"])
ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel("Dark Transfer AUC")
ax.set_title("Dark AUC by IO Position")
for i, v in enumerate([auc_first, auc_second]):
    ax.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=10)

# Panel D: dark vs prediction position
ax = axes[1, 1]
ax.scatter(pred_positions + np.random.uniform(-0.2, 0.2, n_examples),
           g_dark_final, alpha=0.3, s=10, c='steelblue')
ax.set_xlabel("Prediction position (tokens)")
ax.set_ylabel("g_dark (final layer)")
ax.set_title(f"Dark vs Position (r={r_dark_pos:.3f})")

plt.suptitle("Dark Receptor IOI Signal: Confound Controls", fontsize=14, fontweight='bold')
plt.tight_layout()
Save: phase2a_followup_plot1_confounds.png


# PLOT 2: Causal ablation comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(3)
width = 0.35
gp_accdrops = [3.59, 1.31, 8.17]  # GP AccDrop in %
ioi_accdrops = [ioi_accdrop_R1*100, ioi_accdrop_R3*100, ioi_accdrop_dark*100]
bars1 = ax.bar(x - width/2, gp_accdrops, width, label='GP AccDrop', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, ioi_accdrops, width, label='IOI AccDrop', color='coral', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['R1\n(L10H9sv0)', 'R3\n(L9H7sv1)', 'Dark\n(L11H1sv1)'])
ax.set_ylabel("AccDrop (%)")
ax.set_title("Cross-Task Causal Ablation: GP vs IOI")
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f"{bar.get_height():.2f}%", ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.2f}%", ha='center', fontsize=9)
Save: phase2a_followup_plot2_causal_ablation.png


# PLOT 3: Layer delta analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: delta vs IOI logit diff
ax = axes[0]
colors = ['steelblue' if c else 'coral' for c in ioi_correct]
ax.scatter(delta_dark, ioi_logit_diff, c=colors, alpha=0.3, s=10)
ax.set_xlabel("δ_dark (layer 10→11)")
ax.set_ylabel("IOI logit difference")
ax.set_title(f"L11H1 Contribution (corr={r_delta:.3f})")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Right: delta distribution correct vs error
ax = axes[1]
ax.hist(delta_dark[correct_mask], bins=30, alpha=0.6, color='steelblue', 
        label='IOI correct', density=True)
if error_mask.sum() > 0:
    ax.hist(delta_dark[error_mask], bins=15, alpha=0.6, color='coral',
            label='IOI error', density=True)
ax.set_xlabel("δ_dark (layer 10→11)")
ax.set_ylabel("Density")
ax.set_title(f"L11 Delta by IOI Correctness (AUC={delta_auc:.3f})")
ax.legend()
Save: phase2a_followup_plot3_layer_delta.png


# PLOT 4: Cross-task causal profile (the money figure)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
receptors = ['R1', 'R3', 'Dark']
gp_aucs = [0.964, 0.958, 0.526]
ioi_aucs = [0.512, 0.575, 0.776]
gp_ads = [3.59, 1.31, 8.17]
ioi_ads = [ioi_accdrop_R1*100, ioi_accdrop_R3*100, ioi_accdrop_dark*100]

for idx, (name, ax) in enumerate(zip(receptors, axes)):
    x = np.arange(2)
    width = 0.35
    
    # AUC bars
    ax2 = ax.twinx()
    bars_auc = ax.bar(x - width/2, [gp_aucs[idx], ioi_aucs[idx]], width, 
                       color='steelblue', alpha=0.5, label='AUC')
    bars_ad = ax2.bar(x + width/2, [gp_ads[idx], ioi_ads[idx]], width,
                       color='coral', alpha=0.5, label='AccDrop %')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['GP', 'IOI'])
    ax.set_ylabel("AUC")
    ax2.set_ylabel("AccDrop (%)")
    ax.set_title(f"{name}")
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.set_ylim(0.3, 1.0)
    
    # Annotate
    for b in bars_auc:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha='center', fontsize=8)
    for b in bars_ad:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                 f"{b.get_height():.2f}%", ha='center', fontsize=8)

plt.suptitle("Cross-Task Causal Profile: Three Circuit Channels", fontsize=13, fontweight='bold')
plt.tight_layout()
Save: phase2a_followup_plot4_cross_task_profile.png


# PLOT 5: Dark activation by IOI error status
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
data_correct = g_dark_final[correct_mask]
data_error = g_dark_final[error_mask]
parts = ax.violinplot([data_correct, data_error], positions=[0, 1], showmeans=True, showmedians=True)
# Overlay individual error points
ax.scatter(np.ones(error_mask.sum()) + np.random.uniform(-0.05, 0.05, error_mask.sum()),
           data_error, c='red', s=30, zorder=5, alpha=0.7, label='Individual errors')
ax.set_xticks([0, 1])
ax.set_xticklabels([f"IOI Correct\n(N={correct_mask.sum()})", f"IOI Error\n(N={error_mask.sum()})"])
ax.set_ylabel("g_dark (final layer)")
ax.set_title(f"Dark Receptor: IOI Correct vs Error (d'={d_prime:.3f})")
ax.legend()
Save: phase2a_followup_plot5_error_analysis.png


==============================
FINAL SUMMARY
==============================

PRINT: "="*60
PRINT: "PHASE 2A FOLLOW-UP: DARK RECEPTOR IOI — SUMMARY"
PRINT: "="*60
PRINT: "CONFOUND CONTROLS:"
PRINT: f"  corr(dark, sent_length):        {r_dark_len:.4f}"
PRINT: f"  corr(dark, pred_position):      {r_dark_pos:.4f}"
PRINT: f"  Partial corr (controlling all): {partial_corr_both:.4f} (raw: -0.3389)"
PRINT: f"  Template AUCs: gave={tmpl_aucs[0]:.3f}, decided={tmpl_aucs[1]:.3f}, wanted={tmpl_aucs[2]:.3f}"
PRINT: f"  IO position: first={auc_first:.3f}, second={auc_second:.3f}"
PRINT: ""
PRINT: "CAUSAL ABLATION (AccDrop %):"
PRINT: f"  R1:   GP={3.59:.2f}%  IOI={ioi_accdrop_R1*100:.2f}%"
PRINT: f"  R3:   GP={1.31:.2f}%  IOI={ioi_accdrop_R3*100:.2f}%"
PRINT: f"  Dark: GP={8.17:.2f}%  IOI={ioi_accdrop_dark*100:.2f}%"
PRINT: ""
PRINT: "LAYER DELTA:"
PRINT: f"  corr(delta_L11, IOI_logit): {r_delta:.4f}"
PRINT: f"  L11 fraction:               {l11_fraction:.4f}"
PRINT: ""
PRINT: "ERROR ANALYSIS:"
PRINT: f"  Dark mean correct: {g_dark_final[correct_mask].mean():.2f}"
PRINT: f"  Dark mean error:   {g_dark_final[error_mask].mean():.2f}"
PRINT: f"  Cohen's d:         {d_prime:.3f}"
PRINT: "="*60

# Save results JSON
import json
results = {
    "confound_controls": {
        "corr_dark_length": float(r_dark_len),
        "corr_dark_position": float(r_dark_pos),
        "partial_corr_controlling_both": float(partial_corr_both),
        "template_aucs": {t: float(a) for t, a in zip(tmpl_names, tmpl_aucs)},
        "io_position_aucs": {"first": float(auc_first), "second": float(auc_second)}
    },
    "causal_ablation": {
        "R1_ioi_accdrop": float(ioi_accdrop_R1),
        "R3_ioi_accdrop": float(ioi_accdrop_R3),
        "dark_ioi_accdrop": float(ioi_accdrop_dark),
        "R1_gp_accdrop": 0.0359,
        "R3_gp_accdrop": 0.0131,
        "dark_gp_accdrop": 0.0817
    },
    "layer_delta": {
        "corr_delta_ioi": float(r_delta),
        "auc_delta": float(delta_auc),
        "l11_fraction": float(l11_fraction)
    },
    "error_analysis": {
        "dark_mean_correct": float(g_dark_final[correct_mask].mean()),
        "dark_mean_error": float(g_dark_final[error_mask].mean()),
        "cohens_d": float(d_prime)
    }
}
with open("phase2a_followup_results.json", "w") as f:
    json.dump(results, f, indent=2)
PRINT: "[SAVE] phase2a_followup_results.json"
```

---

## What You Need To Do

1. **Check if Phase 2A data is cached.** If you saved `all_residuals`, `io_token_ids`, `s_token_ids`, `all_logits_io`, `all_logits_s` from Phase 2A, load them. If not, the code must rerun the forward passes (same as Phase 2A).

2. **TransformerLens model components:** The causal ablation needs `model.ln_final.w`, `model.ln_final.b`, `model.W_U`, `model.b_U`. Print these shapes to verify:
   - `model.ln_final.w.shape` should be `(768,)`
   - `model.W_U.shape` should be `(768, 50257)`

3. **Layer indexing:** Verify that `all_residuals[:, 11, :]` is the post-block-11 residual (before ln_final). In TransformerLens, `blocks.11.hook_resid_post` is after block 11 but before ln_final.

4. **Sanity check:** The reconstructed IOI accuracy from manual ln_final + W_U should match the original 97.2%. If it doesn't, the layer norm computation is wrong.

5. **Runtime:** ~5 minutes if Phase 2A data is cached (pure computation, no forward passes). ~25 minutes if forward passes need to rerun.

---

## After Running: Send Back

1. All printed output
2. The 5 plots
3. The results JSON
4. Any errors/warnings

---

## What This Experiment Is NOT

This is NOT the "next experiment after the dark receptor follow-up." This IS the follow-up. After this, the remaining experiments from the priority list are:
1. Pipeline Formalization (writing, 2 hrs)
2. RIM Formalization (writing, 1 hr)
3. Noise Robustness (experiment, 2-3 hrs)
4. Counterfactual Patching (experiment, 2-3 hrs)
5. Attention Failure Diagnosis (experiment, 2 hrs)
6. Second Task Pipeline (experiment, half day)

These are all separate, independent experiments. We move to them AFTER this follow-up.
