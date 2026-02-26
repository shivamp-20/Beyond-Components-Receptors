# IDEA 1: Causal Fidelity Ratio (CFR) — Full Specification

---

## 1. WHAT WE ARE DOING

We are building a cheap, general-purpose diagnostic that separates **causally real** logit receptors from **ghost receptors** — directions that pass every correlational test but have zero causal contribution to the model's prediction.

The mask training procedure in Beyond Components identifies logit receptors by correlation: it trains a learnable mask weight for every (layer, head, sv_idx) triple, and directions with high mask weight are declared "important." But we discovered in Exp 4 (Shapley) that R2 — which had mask weight ≥ 0.9 and AUC = 0.952 — has Shapley φ = −0.003. It is a **ghost**: correlated with the task output but causally inert.

The problem: Shapley values require 2^K evaluations (K = number of receptors). For K = 3, that's 8 — trivial. But with mask threshold > 0.9, you told me there are **200+ receptors**. 2^200 is impossible.

We propose CFR: a metric that requires only **K forward passes** (one per receptor) and correctly identifies ghosts.

---

## 2. WHY IT COMES FROM OUR RESULTS

This is not an abstract idea. It comes directly from a specific experimental finding:

- **Exp 4 (Shapley):** R2 (L11H8, sv6) has AUC = 0.952 but Shapley φ = −0.003.
- **Why R2 is a ghost:** cos(R1, R2) = +0.566. R2's direction is partially aligned with R1's. When the model writes gender signal along R1's direction, R2 picks up that signal because its direction overlaps. It's a **bystander** — it observes the signal but doesn't produce it.
- **The implication:** Mask training cannot distinguish bystanders from actors. Any direction that overlaps with a causally real receptor will have high correlation with the task. The higher the overlap (cosine similarity), the stronger the ghost effect.

**Our prediction for the full set:** Most of the 200+ mask-identified receptors are ghosts. They're bystanders whose directions overlap with the 2-4 directions that actually matter. If we can show this — that going from 200+ correlated receptors to 2-4 causal receptors — that's a striking finding about the sparsity of true causal structure, and a methodological warning about trusting correlation-based receptor identification.

---

## 3. THE METHOD

### 3.1 What a "receptor" is (recap for the coding window)

A receptor is defined by a (layer, head, sv_idx) triple. The mask training script performs SVD on the augmented OV matrix for every attention head:

$$W_{\text{aug}}^{(\text{OV})} = U \Sigma V^T$$

Each singular triplet (U[:, k], Σ[k], Vh[k, :]) is a "component" of that head. The **write direction** is Vh[k, :] ∈ ℝ^{d_model} — this is the direction in residual stream space that this component writes into. The mask weight for (layer, head, sv_idx=k) tells you how much this direction contributes to the task-relevant logit difference during mask training.

### 3.2 What "ablating a receptor" means

Given a receptor with write direction r_k = Vh[sv_idx] (unit vector in ℝ^768 for GPT-2 small):

**Ablation = project out r_k from the final residual stream.**

For each example i, let x_final(i) ∈ ℝ^768 be the residual stream at the decision position (last token position) after the final transformer block (block 11), BEFORE ln_final.

The ablated residual is:

$$x_{\text{abl}}(i) = x_{\text{final}}(i) - (x_{\text{final}}(i) \cdot r_k) \cdot r_k$$

This removes ALL signal along r_k, regardless of which component wrote it (host head, other heads, MLPs — everything).

Then recompute the logits:

$$\text{logits}_{\text{abl}}(i) = \text{ln\_final}(x_{\text{abl}}(i)) \cdot W_U + b_U$$

And read off he_logit and she_logit to determine the model's prediction.

### 3.3 The accuracy drop

For the full dataset (use **test_gp.csv** — this is the held-out test set, not the train set we used for all previous experiments):

$$\text{AccDrop}_k = \text{Acc}_{\text{clean}} - \text{Acc}_{\text{ablated}_k}$$

Where:
- Acc_clean = accuracy of the unmodified model
- Acc_ablated_k = accuracy after projecting out receptor k

If AccDrop_k ≈ 0, the receptor is a ghost — removing it doesn't hurt the model.
If AccDrop_k > 0, the receptor is causally real — the model needs that direction.

### 3.4 The Causal Fidelity Ratio (CFR)

$$\text{CFR}_k = \frac{\text{AccDrop}_k}{\max_j \text{AccDrop}_j}$$

This normalizes against the strongest receptor, giving a value in [0, 1]:
- CFR = 1.0: this is the most causally important receptor (the anchor).
- CFR ≈ 0: ghost.
- CFR > 0 but < 1: real but secondary.

**Ghost threshold: CFR < 0.05.**

### 3.5 Why CFR and not just AccDrop?

Because AccDrop depends on the dataset size, the baseline accuracy, and the task difficulty. By normalizing against the max, CFR becomes a relative measure that's comparable across tasks and models. A receptor with CFR = 0.02 is a ghost regardless of whether the max AccDrop is 3% or 30%.

### 3.6 Additional metric: AccDrop itself

We also report the raw AccDrop for each receptor. This gives the absolute causal contribution in accuracy points.

---

## 4. WHAT WE RUN ON

**Dataset: test_gp.csv** (the held-out test set).

Why test, not train? Because:
1. Train set (train_1k_gp.csv) was used for mask training — the mask weights are optimized on it. Using train would conflate "mask correlation" with "causal contribution."
2. Test set gives an unbiased estimate of causal contribution.
3. We use use_both=1 (clean + corrupted), same as all previous experiments, for consistency.

**Which receptors to test:**

We enumerate ALL (layer, head, sv_idx) triples where the trained mask weight exceeds a threshold. We test multiple thresholds:

| Threshold | Approx. count (from your numbers) |
|-----------|-----------------------------------|
| > 0.90 | ~200+ |
| > 0.95 | (somewhere in between) |
| > 0.98 | ~80 |
| > 0.99 | ~32 |

**For the actual experiment, we use ALL receptors with mask > 0.90.** This gives us the full picture — we can always filter to higher thresholds afterward.

**Determining polarity:** Each receptor needs a polarity (+1 or −1) for the AUC computation. We determine this automatically:
- Project the test set residuals onto the receptor direction.
- Compute the correlation with the label (he=+1, she=−1).
- If correlation > 0: polarity = +1 (male-on-top). If < 0: polarity = −1 (female-on-top).

This is important because we don't manually assign polarity to 200 receptors — we let the data determine it.

---

## 5. WHAT WE EXPECT

### Main prediction: Extreme sparsity

Out of 200+ receptors with mask > 0.9, we predict:
- **2-5 receptors** with CFR > 0.05 (causally real).
- **195+ receptors** with CFR < 0.05 (ghosts).

This would mean the mask training procedure has a ~97-99% false positive rate for causal relevance. The correlation-based identification vastly overestimates the number of real receptors.

### Why we predict this

1. **Geometric argument:** In ℝ^768, any direction with significant cosine similarity to a causally real receptor will inherit its correlation with the task. If the real receptors span a 2-3 dimensional subspace, then ANY direction with a non-trivial projection onto this subspace will have nonzero mask weight. There are many such directions.

2. **Empirical evidence from K=3:** We already showed this for R2. R2 has cos(R1, R2) = +0.566, which is high enough to give it AUC = 0.952 — nearly as good as the causally real R1 (AUC = 0.962). But Shapley = −0.003. If this phenomenon scales, most of the 200+ receptors are "R2-like."

3. **The Shapley values:** R1 = 0.148, R3 = 0.085, R2 = −0.003. Only 2 out of 3 are real. If we extrapolate this 2/3 ratio to 200+... we'd get ~130 real, which seems too many. But the K=3 case was already filtered to the TOP 3 by mask weight. The 200+ include many weaker directions. The ratio of real/ghost should be much more extreme for the full set.

### What would make this result publishable

The key plot is the **CFR distribution histogram**: x-axis = CFR value, y-axis = count of receptors. If this shows a massive spike at CFR ≈ 0 with a tiny tail of 2-5 receptors at CFR > 0.05, that's the money figure.

The key number is: **"Of 200+ correlation-identified receptors, only N are causally real."**

If N ≤ 5, this is a strong result about the sparsity of true causal structure.

### What would make it even better

If the causally real receptors (CFR > 0.05) turn out to be EXACTLY our R1 and R3 (plus maybe 1-2 others), that validates our original receptor selection AND shows that the method rediscovers the known circuit from scratch.

### What if the prediction is wrong?

If we find 20+ receptors with CFR > 0.05, that means the circuit is more distributed than we thought. This is still a finding — it means the gender pronoun task uses many directions, not just 2-3. It would change the paper's narrative but not kill it.

---

## 6. WHAT THIS ADDS TO THE PAPER

### 6.1 Methodological contribution

CFR is a **reusable tool**. Anyone using Beyond Components on any task can apply CFR to prune their receptor list. The method is:
1. Train masks (Beyond Components, existing).
2. Enumerate receptors above threshold.
3. For each, compute single-ablation AccDrop.
4. Normalize to get CFR.
5. Keep only CFR > 0.05.

This is a 5-line addition to any logit receptor pipeline. It's cheap (K forward passes), general, and catches a systematic failure mode of correlation-based identification.

### 6.2 Connection to our previous experiments

- **Exp 4 (Shapley):** CFR generalizes the ghost detection from K=3 (where Shapley was feasible) to K=200+ (where Shapley is impossible). It's the scalable version.
- **Exp 2 (Cross-talk):** The ghosts are explained by the same geometric overlap that causes cross-talk. High cosine similarity → high cross-talk → high correlation → ghost.
- **Phase diagram:** After CFR pruning, we can redo the phase diagram with only the real receptors. If the pruned set still explains R² ≥ 0.98 of the logit variance, then the ghosts were truly redundant.

### 6.3 The claim in the paper

"Correlation-based receptor identification (mask training) overestimates the number of causally relevant receptors by an order of magnitude. We introduce the Causal Fidelity Ratio (CFR), a lightweight diagnostic requiring only K forward passes, that correctly identifies ghost receptors — directions that inherit task correlation through geometric overlap with true causal receptors but contribute nothing to the model's computation. Applied to the gender pronoun task, CFR prunes 200+ mask-identified receptors to N causally real ones, revealing extreme sparsity in the true causal circuit."

---

## 7. NOVELTY AND SUBNOVELTIES

### 7.1 The main novelty
Nobody has proposed a scalable causal ghost detection method for logit receptors. The Beyond Components paper identifies receptors by mask weight. We show this is insufficient and provide the fix.

### 7.2 Subnovelties

1. **The false positive rate of correlation-based identification.** Quantifying how many ghost receptors mask training produces. This is a finding about the method, not just our task.

2. **The geometric explanation for ghosts.** We can predict which receptors are ghosts from their cosine similarity to the top-CFR receptors, without running any ablations. If cos(r_ghost, r_real) > threshold → ghost. This is a SECOND, even cheaper method.

3. **The cosine-vs-CFR scatter plot.** For each receptor, plot its max cosine similarity to any CFR > 0.05 receptor (x-axis) vs its own CFR (y-axis). Prediction: strong negative correlation — high overlap with a real receptor → low CFR. This confirms the geometric explanation.

4. **The coverage check.** After pruning to only CFR > 0.05 receptors, recompute R² of receptor-vs-model logit. If R² stays ≥ 0.95 with just 2-5 receptors (down from 200+), the ghosts were truly redundant. The information they carried was ALREADY in the real receptors.

5. **AUC-vs-CFR comparison.** Plot AUC (x-axis) vs CFR (y-axis) for all receptors. Ghosts will cluster in the top-left: high AUC, low CFR. Real receptors in the top-right: high AUC, high CFR. This visually demonstrates that correlation (AUC) and causation (CFR) diverge.

---

## 8. WHAT WE NEED PRINTED AND PLOTTED

### 8.1 Text output (give to me to interpret)

```
=== CFR GHOST DETECTION ===
Clean baseline accuracy: XX.X%
Number of receptors tested: XXX (mask threshold > 0.90)

[TABLE] All receptors sorted by CFR descending:
Rank | Layer | Head | SV_idx | MaskWeight | AUC | AccDrop | CFR | Polarity | Ghost?
1    | ...
2    | ...
...

[SUMMARY]
Receptors with CFR > 0.05: N
Receptors with CFR < 0.05: M  (ghosts)
Ghost fraction: M/(N+M) = XX.X%

Known receptors in top-CFR list:
  R1 (L10H9 sv0): CFR = ...
  R3 (L9H7 sv1):  CFR = ...
  R2 (L11H8 sv6): CFR = ...

[COVERAGE CHECK]
R² of top-CFR receptors vs model logit: ...
R² of all receptors vs model logit: ...

[COSINE ANALYSIS]
For each ghost receptor, its max cosine to any real receptor:
  Mean: ...
  Median: ...
For each real receptor, its max cosine to any other real receptor:
  Mean: ...
```

### 8.2 Plots (save as PNG, send to me)

**Plot 1: CFR Distribution Histogram**
- x-axis: CFR value (0 to 1), bins of 0.02
- y-axis: count of receptors
- Color: red bars for CFR < 0.05 (ghosts), blue bars for CFR ≥ 0.05 (real)
- Title: "Causal Fidelity Ratio Distribution (N=XXX receptors, mask > 0.90)"
- This is THE money figure.

**Plot 2: AUC vs CFR Scatter**
- x-axis: AUC (0.5 to 1.0)
- y-axis: CFR (0 to 1)
- Each point is a receptor. Color by ghost/real (CFR threshold 0.05).
- Mark R1, R2, R3 with special symbols (star or triangle) and labels.
- Draw the ghost threshold line at CFR = 0.05.
- Title: "Correlation (AUC) vs Causation (CFR)"

**Plot 3: Cosine-to-Nearest-Real vs CFR**
- x-axis: max |cos(r_k, r_j)| where j is any receptor with CFR > 0.05
- y-axis: CFR of receptor k
- Color by ghost/real.
- Fit a trend line (should be negatively correlated for ghosts — high overlap = low CFR).
- Title: "Geometric Overlap Predicts Ghost Status"

**Plot 4: AccDrop Bar Chart (top 20 receptors)**
- x-axis: receptor label (L{l}H{h}sv{sv})
- y-axis: AccDrop in percentage points
- Sorted descending.
- Color bars by ghost/real.
- Title: "Single-Ablation Accuracy Drop (Top 20)"

**Plot 5: CFR at Different Mask Thresholds**
- Make 4 sub-panels: mask > 0.90, > 0.95, > 0.98, > 0.99
- Each sub-panel: bar chart showing number of real (blue) vs ghost (red) receptors
- Title: "Ghost Detection at Different Mask Thresholds"

---

## 9. PSEUDOCODE FOR THE CODING WINDOW

```python
# === CFR Ghost Detection Experiment ===
#
# Inputs:
#   - svd_cache.pt (or ov_svd_cache.pt): contains ov[layer][head] with .Vh, .S, .U, .r
#   - masks.pt: contains trained mask weights per (layer, head, sv_idx)
#   - test_gp.csv: held-out test set
#   - GPT-2 small via TransformerLens
#
# Steps:
#
# 1. Load model, SVD cache, masks
#    model = HookedTransformer.from_pretrained("gpt2-small")
#    ov = load svd_cache (same code as all previous experiments)
#    masks = torch.load("masks.pt")  -- need to understand the structure
#
# 2. Enumerate all receptors with mask > 0.90
#    For each (layer, head):
#      For each sv_idx in range(ov[layer][head].r):
#        mask_weight = get_mask_weight(masks, layer, head, sv_idx)
#        if mask_weight > 0.90:
#          receptor_list.append((layer, head, sv_idx, Vh[sv_idx]))
#
# 3. Load test data
#    Use test_gp.csv with use_both=1 (same expand_rows as previous experiments)
#    Tokenize all examples, get tokens, last_idx, labels (y: +1 for he, -1 for she)
#
# 4. Collect clean final residuals (ONE forward pass)
#    Hook at "blocks.11.hook_resid_post" to get the post-block-11 residual
#    For each batch:
#      Run forward pass, collect residual at last_idx
#    Result: resid_clean of shape (N, d_model)
#
# 5. Compute clean accuracy (baseline)
#    For each example:
#      Apply ln_final to resid_clean[i]
#      Compute logits = ln_final_out @ W_U + b_U   (or model.unembed)
#      he_logit = logits[he_id], she_logit = logits[she_id]
#      pred = "he" if he_logit > she_logit else "she"
#      correct if pred matches label
#    baseline_acc = fraction correct
#
# 6. For each receptor k in receptor_list:
#    r_k = Vh[sv_idx] (unit vector, d_model)
#    For each example i:
#      projection = resid_clean[i] dot r_k
#      resid_abl[i] = resid_clean[i] - projection * r_k
#      Apply ln_final, compute logits, check prediction
#    acc_ablated_k = fraction correct
#    AccDrop_k = baseline_acc - acc_ablated_k
#
#    Also compute:
#    g_k[i] = resid_clean[i] dot r_k  (receptor activation)
#    Determine polarity: if corr(g_k, labels) > 0: pol=+1, else pol=-1
#    Compute AUC: polarity-adjusted, same method as Exp 0
#
# 7. Compute CFR for all receptors
#    max_drop = max(AccDrop_k for all k)
#    CFR_k = AccDrop_k / max_drop  (handle max_drop = 0 edge case)
#    ghost_k = (CFR_k < 0.05)
#
# 8. Compute cosine similarities between all receptors
#    For each pair (j, k): cos_jk = r_j dot r_k
#    For each receptor k: max_cos_to_real[k] = max over j where CFR_j > 0.05 of |cos_jk|
#
# 9. Coverage check
#    Real receptor directions: R_real = stack of all r_k where CFR_k > 0.05
#    For each example:
#      g_real = R_real @ resid_clean[i]  (activations on real receptors)
#      linear_combination = weighted sum (fit a linear regression: g_real -> model_logit_diff)
#    R² of this fit = coverage
#
# 10. Print all tables, save all plots
```

### Important implementation details for the coding window:

**Loading masks.pt:** The mask training script saves masks as a dictionary. The exact structure depends on the code, but typically it's something like:
- `masks["ov"]` or `masks["mask"]` containing a nested structure `masks[layer][head]` which is a tensor of shape (rank,) — one mask weight per sv_idx.
- Or it might be a flat tensor indexed differently. The coding window needs to look at the mask training script's save code to determine the exact format.

**CRITICAL: Tell the coding window to FIRST print the structure of masks.pt** (`torch.load("masks.pt")` and print keys, shapes, types) before writing the main code. This avoids guessing.

**ln_final application:** Use `model.ln_final(x)` — don't manually access weights. This works regardless of LayerNorm variant.

**Unembed:** After ln_final, do `logits = model.unembed(ln_out)` where ln_out has shape (d_model,). Or manually: `logits = ln_out @ model.W_U + model.b_U`. The TransformerLens API might differ slightly — the coding window should check.

**Residual collection:** Hook at `blocks.11.hook_resid_post` or equivalently, use `model.run_with_cache` and extract `resid_post` at the last block.

**Batching the ablation:** The ablation is a simple vector projection — it can be done in bulk:
```python
# For all N examples at once:
proj = (resid_clean @ r_k).unsqueeze(-1) * r_k.unsqueeze(0)  # (N, d_model)
resid_abl = resid_clean - proj  # (N, d_model)
```
This is fast — no need for individual forward passes per ablation. The expensive part is the ln_final + unembed per ablation (N × d_model matrix multiply), but that's still just a matrix operation, not a full model forward pass. The single forward pass to collect resid_clean is the only real cost.

**BUT WAIT — there's a subtlety with ln_final.** LayerNorm is not a linear operation. The norm of x_abl differs from x_clean because we removed a component. So ln_final(x_abl) ≠ ln_final(x_clean) − ln_final(projection). We MUST reapply ln_final to the ablated residual. This means:
```python
# Correct:
ln_out_abl = model.ln_final(resid_abl)  # (N, d_model), recomputed
logits_abl = ln_out_abl @ model.W_U + model.b_U
```
This is still cheap — ln_final is just a normalization, not a transformer block.

**AUC computation:** For each receptor, compute AUC using:
```python
from sklearn.metrics import roc_auc_score
# labels: +1 for he, -1 for she
# g_k: receptor activation (projection of residual onto r_k)
# polarity: +1 or -1
auc = roc_auc_score((labels + 1) // 2, polarity * g_k)
```
Where `(labels + 1) // 2` converts {-1, +1} to {0, 1}.

---

## 10. WHAT TO TELL THE CODING WINDOW

Give the coding window this briefing:

---

**TASK: Write a Python script `gp_cfr_ghost_detection.py` that runs the Causal Fidelity Ratio (CFR) ghost detection experiment.**

**Context:** I am working on mechanistic interpretability of GPT-2 small on a gender pronoun (GP) prediction task. I have:
- Trained mask weights (`masks.pt`) from the Beyond Components logit receptors method
- SVD cache (`svd_cache.pt` or `ov_svd_cache.pt`) containing OV matrix SVDs for all attention heads
- Test data (`test_gp.csv`) with columns: prefix, pronoun, corr_prefix, corr_pronoun, name, corr_name
- TransformerLens installed
- Two T4 GPUs on Kaggle

**What the script does:**
1. Loads the model (GPT-2 small via TransformerLens), SVD cache, and masks
2. Enumerates ALL (layer, head, sv_idx) triples where mask weight > 0.90
3. Runs ONE forward pass on test_gp.csv to collect the final residual stream (after block 11, before ln_final) at the decision position
4. For each receptor: projects out its direction from the residual, reapplies ln_final + unembed, measures accuracy
5. Computes CFR = AccDrop / max(AccDrop)
6. Computes AUC and polarity for each receptor
7. Computes cosine similarities between all receptors
8. Prints comprehensive tables and saves 5 plots

**CRITICAL FIRST STEP:** Before writing the main code, the script should print the structure of `masks.pt` (keys, shapes, types of all entries) so we understand the format. Wrap this in a try/except so if the format is unexpected, we get a clear error.

**Files to give the coding window:**
1. `train_gp_masks_and_dump_ov_logit_receptors_ddp.py` (the mask training script — so it can understand masks.pt format and use the data loading functions)
2. `gp_exp4_receptor_shapley_ablation.py` (the Shapley script — so it can reuse the receptor loading and data expansion code)
3. `test_gp.csv` (or tell it where to find it: `data_main/test_gp.csv`)

**How to run:**
```bash
python gp_cfr_ghost_detection.py \
  --data_dir data_main \
  --csv test_gp.csv \
  --out_dir outputs/gp \
  --mask_threshold 0.90 \
  --batch_size 64 \
  --device cuda \
  --use_both 1 \
  --known_receptors "10,9,0,+1;11,8,6,+1;9,7,1,-1"
```

The `--known_receptors` flag marks R1, R2, R3 in the output so we can verify they're correctly identified.

**Output requirements:**
- Print the full table of all receptors sorted by CFR
- Print summary statistics (number real, number ghost, ghost fraction)
- Print where R1, R2, R3 land in the ranking
- Save 5 plots as PNG files (detailed specifications in the document I'll provide)
- Save results as JSON for future analysis

---

## 11. SANITY CHECKS

The coding window should implement these checks:

1. **Baseline accuracy should match previous experiments.** On test_gp.csv with use_both=1, the model accuracy should be approximately 89-90%. If it's very different, something is wrong with data loading or decision position computation.

2. **R1 (L10H9 sv0) should have the highest or near-highest CFR.** We know from Shapley that R1 has φ = 0.148 (largest). Its AccDrop should be the largest or close to it.

3. **R2 (L11H8 sv6) should have CFR < 0.05.** We know it's a ghost from Shapley. If CFR says it's real, our method is broken.

4. **R3 (L9H7 sv1) should have CFR > 0.05 but less than R1.** Shapley φ = 0.085.

5. **The sum of AccDrops should NOT equal the total margin.** Single ablations don't compose linearly (because of non-orthogonality). This is expected and not a bug.

6. **No receptor should have negative AccDrop significantly.** A small negative AccDrop (< 0.5%) is fine (noise). A large negative would mean ablation HELPS, which would be very surprising and worth investigating.

---

## 12. HOW THIS CONNECTS TO EVERYTHING ELSE

| Previous finding | How CFR connects |
|-----------------|------------------|
| R2 is ghost (Exp 4, Shapley) | CFR should confirm: R2 CFR < 0.05 |
| cos(R1, R2) = 0.566 (Exp 4) | Geometric explanation: R2 is ghost BECAUSE of this overlap |
| cos(R1, R3) = −0.775 (Exp 4) | Despite high overlap, R3 is NOT a ghost — it has independent computation |
| R1 inherits from embedding (Exp 0c) | R1 should be one of the top-CFR receptors — inherited signals are real |
| R3 computes at layer 9 (Exp 0c) | R3 should be second top-CFR — computed signals are real |
| R² = 0.980 (Type D gap) | After CFR pruning to real receptors, R² should stay ≥ 0.95 |
| 41.2% computational coupling (Exp 2) | The non-orthogonality that causes coupling is the SAME non-orthogonality that creates ghosts |

---

## 13. WHAT MAKES THIS PUBLISHABLE (HONEST ASSESSMENT)

**If we find ≤ 5 real receptors out of 200+:** This is a clean, striking result. The claim "mask training has a 97% false positive rate for causal relevance" is memorable, quantitative, and practically important. It's a warning to the field AND a solution (CFR). Publishability: **8/10**.

**If we find 10-20 real receptors out of 200+:** Still a good result. The circuit is more distributed than we thought, but CFR still provides massive pruning. Less dramatic but still useful. Publishability: **6/10**.

**If we find 50+ real receptors:** Our method works (it still prunes), but the headline finding is weaker. The circuit is genuinely distributed. We'd need to reframe: "even in a distributed circuit, CFR identifies a hierarchy." Publishability: **4/10** for this finding alone, but it changes the paper's narrative in interesting ways.

**Regardless of outcome:** The CFR method itself is a contribution. It's cheap, general, and the paper can recommend it as a standard post-processing step after mask training. The experiment is worth running.
