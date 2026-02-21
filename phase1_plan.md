# Phase 1: Causal Subspace Geometry

## What Are We Doing (Big Picture)

We have established that the Mask → CFR → OCA pipeline compresses 207 mask-identified directions down to 3 causally independent directions (R1, R3, dark receptor) that explain R² = 0.9815 of the logit variance. But we have not yet asked: **what is the geometric structure of the space these directions live in?**

This is a single unified experiment with four tightly connected parts (A-D), all answerable from saved tensors + model weights, with **zero forward passes**. The entire experiment runs in one Kaggle script.

---

## Why This Is Not Trivial (The Intellectual Gap)

Here is the problem our current results leave open:

**Gap 1: CFR as implicit decorrelation.** We showed that 194/207 directions are ghosts. We showed that ghosts have high cosine with R1. But we never *quantified* whether CFR systematically strips out geometric correlation. If CFR is functioning as an implicit decorrelation step — removing directions that are geometric echoes of real receptors — then the mean pairwise |cosine| among survivors should drop sharply compared to the pre-CFR set. This is a testable, non-obvious prediction. It is not guaranteed: CFR filters on causal fidelity, not on geometry. If it happens to decorrelate, that tells us something deep about the relationship between causal relevance and geometric independence in transformer circuits.

**Gap 2: Dimensionality of the causal subspace.** OCA found 3 independent directions from 13 survivors. But OCA uses a hard cosine threshold (τ = 0.3). It doesn't tell us what the *continuous* dimensionality of the 13-direction subspace is. If we stack the 13 survivor vectors as rows of a matrix R₁₃ ∈ ℝ^{13×768} and compute its SVD, the singular value spectrum tells us exactly how many dimensions these 13 directions actually span. If the effective rank is 3, then the 13 directions approximately lie in a 3D subspace of ℝ^768. This is a *much stronger* claim than "OCA found 3" — it means the CFR survivors are inherently low-rank, not just prunable by a threshold.

**Gap 3: Connection to superposition.** The superposition hypothesis (Elhage et al. 2022, Anthropic) says models represent more features than they have dimensions by superimposing features in overlapping directions. Our setting gives a concrete, causally-validated instance: 13 gender-relevant directions with nonzero causal fidelity living in (approximately) 3 dimensions of ℝ^768. If the SVD confirms this, we can define a **Receptor Superposition Index** that quantifies the compression ratio. This is novel because existing superposition work operates on toy models or SAE features — we would be measuring superposition from *causally validated* circuit directions.

**Gap 4: Mechanistic explanation of inherited vs. computed.** We showed that R1 commits at the embedding layer (inherited) and R3 commits at layer 9 (computed). We have a verbal explanation: "the embedding already contains gender information along R1's direction." But we never *tested* this. If we extract the actual name token embeddings from W_E and project them onto R1's write direction, we should see that male names project positively and female names project negatively (or vice versa, depending on polarity). If we then compute the mean gender direction in embedding space (d_gender = mean(male embeddings) - mean(female embeddings)) and measure its cosine with R1's write direction, a high |cosine| would *prove* that R1 inherits from a pre-existing gender axis in the embedding. Conversely, R3's direction should have low cosine with d_gender, because R3 is *computed* at layer 9 — the embedding has no signal along R3's direction.

This last point is what makes it non-trivial. The claim "R1 inherits from the embedding" was so far supported by the commitment analysis (sign agreement from layer -1). But sign agreement only tells you the *sign* is stable — it doesn't tell you *why*. The embedding projection directly tests the mechanism: R1's write direction is aligned with a pre-existing gender axis in the learned embedding table.

---

## How This Connects to Previous Work

This experiment sits naturally after CFR+OCA and before IOI transfer (Phase 2). The logical flow is:

1. **CFR** asks: "which directions are causally real?" → 13 survivors
2. **OCA** asks: "which survivors are independent?" → 3 basis directions
3. **Phase 1 (this)** asks: "what is the geometry of the space these directions span?" → dimensionality, decorrelation, superposition, embedding alignment
4. **Phase 2 (IOI)** asks: "do these directions generalize to a different task?" → inherited-vs-computed dissociation

Phase 1 provides the geometric foundation that makes Phase 2 interpretable. Without Phase 1, if R1 transfers to IOI and R3 doesn't, we'd say "because one is inherited and one is computed." With Phase 1, we can say "because R1's write direction aligns with the pre-existing gender axis in the embedding (cos = X), which is task-independent, while R3's direction is orthogonal to the embedding gender axis (cos = Y), confirming it is a task-specific computation."

---

## How This Differs From What Exists in the Literature

**Cosine analysis of model directions:** Many papers compute pairwise cosines between features or directions (e.g., SAE feature geometry papers). What makes ours different is that we do it *across stages of a causal filtering pipeline*. We compare 207×207 (raw) vs. 13×13 (post-CFR) vs. 3×3 (post-OCA) cosine structure. The claim is that causal filtering acts as decorrelation — this has not been shown before.

**SVD of feature sets:** SVD/PCA of activation matrices is common. But SVD of *write direction vectors* from causally-validated receptors, specifically to measure superposition, is not. Existing superposition measurements come from (a) toy models with known ground truth (Elhage et al.), (b) SAE dictionary directions, or (c) probing directions. Ours comes from OV-SVD write directions filtered through a causal pipeline. Different provenance, same theoretical framework.

**Embedding gender probing:** Gender probes on word embeddings exist (Bolukbasi et al. 2016 and follow-ups). But those probe static word embeddings, not contextual transformer embeddings. And they don't connect the embedding gender axis to specific *circuit directions* inside the model. Our contribution is showing that a specific OV write direction (R1) aligns with the embedding gender axis, which mechanistically explains why that direction carries gender signal from layer 0.

---

## The Four Parts

### Part A: Pairwise Cosine Structure

**Math.** For each set S ∈ {all 207, 13 CFR survivors, 3 OCA survivors}, let V_S = {v₁, ..., v_|S|} be the write direction vectors (each ∈ ℝ^768). Compute the pairwise absolute cosine matrix:

$$C_{ij} = |\cos(v_i, v_j)| = \frac{|v_i^\top v_j|}{\|v_i\| \|v_j\|}$$

Report:
- Mean off-diagonal |cos| for each set
- Max off-diagonal |cos| for each set
- Histograms of off-diagonal |cos| values
- The 3×3 cosine matrix among OCA survivors (R1, R3, dark) — this should show them as approximately orthogonal

**Key prediction:** Mean |cos| should drop from pre-CFR to post-CFR, and drop again from post-CFR to post-OCA. This would prove that CFR+OCA is not just selecting "the causally relevant ones" but also selecting directions that are *geometrically independent*.

**Why this matters:** If the drop is large (say from 0.3 to 0.1), it means ghosts are ghosts *precisely because* they point in similar directions to real receptors. The causal filtering and geometric decorrelation would be two sides of the same coin. If the drop is small, the ghost phenomenon is about something other than geometry.

### Part B: Singular Value Spectrum (Superposition Measurement)

**Math.** Stack the 13 CFR survivor direction vectors as rows of a matrix:

$$R_{13} = \begin{bmatrix} v_1^\top \\ v_2^\top \\ \vdots \\ v_{13}^\top \end{bmatrix} \in \mathbb{R}^{13 \times 768}$$

Compute its SVD: R₁₃ = UΣVᵀ, where Σ = diag(σ₁, σ₂, ..., σ₁₃) with σ₁ ≥ σ₂ ≥ ... ≥ σ₁₃ ≥ 0.

Define:
- **Cumulative variance explained:** $\rho(k) = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{13} \sigma_i^2}$
- **Effective rank:** $k^* = \min\{k : \rho(k) \geq 0.95\}$
- **Receptor Superposition Index (RSI):** $\text{RSI} = 1 - \frac{k^*}{K}$ where K=13

Interpretation:
- If k* = 3: "13 causally-relevant gender directions live in a 3D subspace." RSI = 1 - 3/13 = 0.77 (77% superposition).
- If k* = 5: less superposition than expected. RSI = 0.62.
- If k* = 13: no superposition at all. RSI = 0.

Also do SVD on the full 207-direction matrix R₂₀₇ ∈ ℝ^{207×768} and compare effective ranks. If the 207-set also has effective rank ~3, that confirms the entire mask-identified set is a massively bloated representation of a 3D signal.

**Additional analysis I'm adding:** Compute the **Stable Rank** (a smooth version of effective rank):

$$\text{srank}(R) = \frac{\|R\|_F^2}{\|R\|_2^2} = \frac{\sum \sigma_i^2}{\sigma_1^2}$$

Stable rank is more robust than threshold-based effective rank and has a nice theoretical interpretation.

**Key prediction:** I expect k* = 3 (or possibly 4, with the dark receptor contributing a slightly different direction). The singular value plot should show σ₁ >> σ₂ >> σ₃ >> σ₄ ≈ ... ≈ σ₁₃ ≈ 0, with a clear "elbow" at k=3.

**Why this matters for publishability:** This directly connects our circuit analysis to the superposition hypothesis — one of the most discussed theoretical questions in mechanistic interpretability. We provide the first measurement of superposition from *causally validated* directions rather than toy models or SAE dictionaries.

### Part C: Embedding Geometry Audit

**Math.** Extract the token embedding matrix W_E ∈ ℝ^{50257×768}. For each name in our dataset, get its token embedding e_name = W_E[token_id, :].

Partition names into male-associated (he-fraction > 0.7) and female-associated (he-fraction < 0.3).

Compute the **embedding gender direction:**

$$d_{\text{gender}} = \frac{\bar{e}_{\text{male}} - \bar{e}_{\text{female}}}{\|\bar{e}_{\text{male}} - \bar{e}_{\text{female}}\|}$$

where $\bar{e}_{\text{male}}$ = mean of male name embeddings, $\bar{e}_{\text{female}}$ = mean of female name embeddings.

Then measure alignment:
- cos(v_R1, d_gender): how aligned R1's write direction is with the embedding gender axis
- cos(v_R3, d_gender): how aligned R3's write direction is with the embedding gender axis
- cos(v_dark, d_gender): how aligned the dark receptor is with the embedding gender axis

Also project all name embeddings onto each receptor direction and plot:
- Scatter: e_name · v_R1 (y-axis) vs. he-fraction (x-axis)
- Scatter: e_name · v_R3 (y-axis) vs. he-fraction (x-axis)
- Scatter: e_name · v_dark (y-axis) vs. he-fraction (x-axis)

**Additional analysis I'm adding:** Compute the **embedding AUC** — use the projection e_name · v_Rk as a classifier for gender, compute AUC. This gives us "how well does the *embedding alone* separate genders along this receptor direction." Compare:
- Embedding AUC for R1 direction (expected: high, ~0.8+)
- Embedding AUC for R3 direction (expected: low, ~0.55)
- Embedding AUC for dark direction (expected: ~0.5)

This is a single number that summarizes the alignment story.

**Another addition:** Compute **d_gender's projection onto the 3-receptor subspace.** What fraction of the embedding gender direction lives inside span(v_R1, v_R3, v_dark)? This tells us whether the circuit receptors capture the full gender axis from the embedding or only a component of it.

$$\text{captured\_fraction} = \|P_{\mathcal{B}} \, d_{\text{gender}}\|^2$$

where P_B is the projection onto the span of the 3 receptor directions.

**Key prediction:** cos(v_R1, d_gender) should be high (|cos| > 0.5, possibly much higher). cos(v_R3, d_gender) should be near zero (|cos| < 0.15). This would prove the inherited-vs-computed distinction at the geometric level: R1 "sees" the embedding gender signal because its write direction is aligned with it. R3 doesn't see it because its direction is orthogonal to it — R3's gender signal is created *de novo* by L9H7's attention computation.

**Why this matters:** This closes the explanatory loop. We showed (Exp 0c) that R1 commits from the embedding. Now we show *why* — because its write direction already has high cosine with the gender axis that exists in the embedding table. This is a mechanistic explanation, not just a behavioral observation.

### Part D: Receptor Rank Spectrum

**Math.** From the OCA convergence data, we have R²(k) for k = 1, 2, 3 (the cumulative R² when adding directions in order of AccDrop). Compute:

$$\rho(k) = R^2(k) - R^2(k-1)$$

with R²(0) = 0. This is the **marginal explanatory gain** per direction.

Plot as a bar chart: x-axis = direction (ordered by AccDrop), y-axis = ρ(k).

Also compute the **information concentration ratio:**

$$\text{ICR} = \frac{\rho(1)}{\sum_k \rho(k)} = \frac{R^2(1)}{R^2(3)}$$

This tells us how concentrated the explanatory power is in the top direction.

**Key prediction:** The dark receptor (highest AccDrop) will have the largest marginal ρ, followed by R1, then R3. The spectrum should be steeply decaying.

**Note:** This part is simpler than A-C and mainly serves to produce a clean visualization of the "three-direction sufficiency" claim.

---

## What We Expect (Summary of Predictions)

| Quantity | Expected Value | If Violated |
|----------|---------------|-------------|
| Mean |cos| (207 set) | ~0.2-0.3 (high) | Directions are more random than expected |
| Mean |cos| (13 CFR set) | ~0.15-0.2 (lower) | CFR doesn't decorrelate |
| Mean |cos| (3 OCA set) | ~0.1-0.15 (lowest) | OCA doesn't decorrelate further |
| Effective rank (13 survivors) | 3 (maybe 4) | More independent dims than expected |
| Effective rank (207 set) | 3-5 | Gender signal spreads wider than 3D |
| RSI | ~0.75 (if k*=3) | Less superposition than claimed |
| cos(v_R1, d_gender) | |cos| > 0.5 | R1 does NOT inherit from embedding |
| cos(v_R3, d_gender) | |cos| < 0.15 | R3 IS aligned with embedding (not computed) |
| cos(v_dark, d_gender) | |cos| < 0.1 | Dark receptor encodes embedding gender |
| Embedding AUC for R1 | >0.8 | R1 direction doesn't separate names |
| Embedding AUC for R3 | ~0.55 | R3 separates even in embedding |

**What could go wrong (and what it would mean):**

1. **Effective rank = 6-7 instead of 3:** Would mean the CFR survivors span more dimensions than OCA captures. OCA's threshold (τ=0.3) might be too aggressive. Still interesting — would motivate a softer pruning method.

2. **cos(v_R1, d_gender) ≈ 0:** Would *disprove* the inheritance hypothesis. R1's early commitment would need a different explanation (perhaps it's not the embedding gender signal but some other structure). This would be a surprising and publishable negative result.

3. **cos(v_R3, d_gender) > 0.5:** Would mean R3 also reads from the embedding gender axis. The "computed" label would be wrong. The commitment analysis showed R3 commits at layer 9, but maybe it *partially* inherits with the embedding signal getting amplified at layer 9 rather than created from scratch.

Any of these violations would be interesting. We win either way.

---

## Publishability Impact Assessment

**Before Phase 1:** The paper claims "3 directions explain 98% of the logit variance" with CFR+OCA as methodology. This is a clean pipeline contribution.

**After Phase 1 (assuming predictions hold):**
- The paper additionally claims: "these 3 directions span a 3D causal subspace in ℝ^768, with 13 causally-relevant directions exhibiting RSI=0.77 superposition." This connects to the superposition hypothesis literature.
- The paper proves: "R1 inherits from the embedding gender axis (cos=X), explaining its layer-0 commitment, while R3's direction is orthogonal to the embedding gender axis (cos=Y), confirming it is a task-specific computation." This completes the mechanistic story.
- The paper shows: "causal filtering (CFR) acts as implicit geometric decorrelation, dropping mean |cos| from X to Y." This is a novel observation about the relationship between causal relevance and geometric structure.

**Rating: +1.5 to publishability** (from 6/10 → 7.5/10 for main conference). The superposition connection and embedding alignment proof are the strongest additions. The decorrelation observation is a nice bonus.

---

## Sanity Check: Is This Circular?

One worry: are we just measuring things that are tautologically true?

**Part A (cosine drop):** NOT circular. CFR filters on AccDrop/AUC, not on geometry. Ghosts could have low cosine with real receptors and still be ghosts (e.g., through a different mechanism). The fact that ghosts are geometrically correlated with real receptors is an empirical finding, not a tautology.

**Part B (SVD):** NOT circular. OCA pruned with a hard threshold. The SVD tells us about the continuous rank structure. If 13 directions lie in a 5D subspace, OCA's 3D output would be an underfitting. The SVD is a strictly more informative analysis.

**Part C (embedding alignment):** NOT circular. Commitment analysis showed R1's sign is stable from layer 0. But many directions could have stable signs from the embedding without being aligned with the gender axis (e.g., a direction orthogonal to gender that happens to have consistent signs). The embedding projection tests a specific mechanism.

**Part D (rank spectrum):** Somewhat circular (it's basically replotting OCA convergence data). I include it because it produces a clean visualization, not because it's a new analysis. The bar chart is for the paper, not for discovery.

---

## Execution Plan

**All four parts in ONE script.** No forward passes needed. Required inputs:
1. `svd_cache.pt` — contains V matrices (write directions) for all (layer, head) pairs, with singular values
2. The list of 207 mask-identified (layer, head, sv_index) triples
3. The list of 13 CFR survivors (with their AccDrop, AUC, CFR values)
4. The list of 3 OCA survivors
5. The OCA convergence R² values for k=1,2,3
6. GPT-2 model weights (just `model.transformer.wte.weight` for the embedding table)
7. The name → gender mapping from the dataset

**Runtime estimate:** Under 5 minutes on any machine. No GPU needed (just matrix operations on 768-dim vectors).

**Outputs:** 7-8 plots, 1 summary table with ~10 key numbers.

---

## Pseudocode

Below is the complete pseudocode. Copy this to the other Claude window for Python conversion.

```
PHASE 1: CAUSAL SUBSPACE GEOMETRY
==================================
Inputs needed:
  - svd_cache.pt (saved from mask training / CFR experiment)
  - GPT-2 Small model (for embedding weights only)
  - Dataset CSV with name-gender mappings
  - The following lists (hardcoded from previous experiment results):
      * 207 mask-identified triples: [(layer, head, sv_index), ...]
      * 13 CFR survivors: [(layer, head, sv_index, AccDrop, AUC, CFR), ...]
      * 3 OCA survivors: R1=(10,9,0), R3=(9,7,1), Dark=(11,1,1)
      * OCA R² values: [R²(1), R²(2), R²(3)] = [?, ?, 0.9815]

SETUP:
  Load svd_cache.pt
  Load GPT-2 model (can be just the weights, or full model)
  
  # Extract all 207 direction vectors
  directions_207 = []
  for (layer, head, sv_idx) in mask_identified_triples:
      V = svd_cache[(layer, head)]['V']  # V matrix from OV-SVD
      v = V[sv_idx, :]  # write direction, shape (768,)
      v = v / norm(v)   # normalize
      directions_207.append(v)
  directions_207 = stack(directions_207)  # shape (207, 768)
  
  # Extract 13 CFR survivor direction vectors
  directions_13 = []
  for (layer, head, sv_idx, ...) in cfr_survivors:
      V = svd_cache[(layer, head)]['V']
      v = V[sv_idx, :] / norm(V[sv_idx, :])
      directions_13.append(v)
  directions_13 = stack(directions_13)  # shape (13, 768)
  
  # Extract 3 OCA survivor direction vectors
  v_R1  = normalized direction for (10, 9, 0)
  v_R3  = normalized direction for (9, 7, 1)
  v_dark = normalized direction for (11, 1, 1)
  directions_3 = stack([v_R1, v_R3, v_dark])  # shape (3, 768)


==============================
PART A: PAIRWISE COSINE STRUCTURE
==============================

For each set S in [directions_207, directions_13, directions_3]:
    K = S.shape[0]
    # Compute cosine matrix
    cos_matrix = S @ S.T  # since rows are normalized, this IS the cosine matrix
    
    # Extract upper-triangle (off-diagonal) entries
    mask = upper_triangular_mask(K, diagonal=1)
    off_diag = abs(cos_matrix[mask])
    
    # Statistics
    mean_abs_cos = mean(off_diag)
    max_abs_cos = max(off_diag)
    median_abs_cos = median(off_diag)
    
    PRINT: f"Set size {K}: mean|cos|={mean_abs_cos:.4f}, max|cos|={max_abs_cos:.4f}, median|cos|={median_abs_cos:.4f}"

# PLOT 1: Three histograms overlaid (207, 13, 3)
fig, ax = subplots(1,1)
for each set, plot histogram of off-diagonal |cos| values
Title: "Pairwise |cosine| distribution across pipeline stages"
X-label: "|cosine similarity|"
Y-label: "Count"
Legend: ["207 mask-identified", "13 CFR survivors", "3 OCA survivors"]
Save: phase1_plot1_cosine_histograms.png

# PLOT 2: 13×13 cosine heatmap (the 13 CFR survivors)
fig, ax = subplots(1,1)
imshow(abs(cos_matrix_13), cmap='RdBu_r', vmin=0, vmax=1)
Label rows/columns with (layer,head,sv) identifiers
Annotate cells with values
Title: "Pairwise |cosine| among 13 CFR survivors"
Save: phase1_plot2_cosine_heatmap_13.png

# PLOT 3: 3×3 cosine matrix (OCA survivors)
# Small enough to print as a table too
cos_3x3 = directions_3 @ directions_3.T
PRINT the 3×3 matrix with labels R1, R3, Dark
Save: phase1_plot3_cosine_matrix_3.png

# KEY NUMBERS for Part A:
PRINT: "=== PART A SUMMARY ==="
PRINT: f"Mean |cos| drop: 207→13: {mean207:.4f} → {mean13:.4f} (Δ = {mean207-mean13:.4f})"
PRINT: f"Mean |cos| drop: 13→3:  {mean13:.4f} → {mean3:.4f} (Δ = {mean13-mean3:.4f})"
PRINT: f"Total drop 207→3: {mean207:.4f} → {mean3:.4f}"


==============================
PART B: SINGULAR VALUE SPECTRUM
==============================

# SVD of 13 CFR survivors
R13 = directions_13  # (13, 768) - rows are normalized direction vectors
U13, S13, Vt13 = svd(R13, full_matrices=False)
# S13 has shape (13,), these are the singular values

# Cumulative variance explained
total_var_13 = sum(S13**2)
cum_var_13 = cumsum(S13**2) / total_var_13

# Effective rank (95% threshold)
eff_rank_95 = argmax(cum_var_13 >= 0.95) + 1  # +1 for 1-indexing
eff_rank_99 = argmax(cum_var_13 >= 0.99) + 1

# RSI
RSI = 1 - eff_rank_95 / 13

# Stable rank
stable_rank_13 = total_var_13 / (S13[0]**2)

PRINT: f"Singular values (13 survivors): {S13}"
PRINT: f"Cumulative variance: {cum_var_13}"
PRINT: f"Effective rank (95%): {eff_rank_95}"
PRINT: f"Effective rank (99%): {eff_rank_99}"
PRINT: f"RSI: {RSI:.4f}"
PRINT: f"Stable rank: {stable_rank_13:.2f}"

# Repeat for 207 directions
R207 = directions_207  # (207, 768)
U207, S207, Vt207 = svd(R207, full_matrices=False)
total_var_207 = sum(S207**2)
cum_var_207 = cumsum(S207**2) / total_var_207
eff_rank_207_95 = argmax(cum_var_207 >= 0.95) + 1
eff_rank_207_99 = argmax(cum_var_207 >= 0.99) + 1
stable_rank_207 = total_var_207 / (S207[0]**2)

PRINT: f"Effective rank 207 (95%): {eff_rank_207_95}"
PRINT: f"Effective rank 207 (99%): {eff_rank_207_99}"
PRINT: f"Stable rank 207: {stable_rank_207:.2f}"

# PLOT 4: Singular value spectrum (both sets)
fig, axes = subplots(1, 2, figsize=(14, 5))

# Left: singular values (log scale y-axis for 207)
axes[0].bar(range(1, 14), S13, color='steelblue', alpha=0.8)
axes[0].set_title("Singular values: 13 CFR survivors")
axes[0].set_xlabel("Component index")
axes[0].set_ylabel("Singular value")
axes[0].axvline(x=eff_rank_95-0.5, color='red', linestyle='--', label=f'95% rank = {eff_rank_95}')

axes[1].plot(range(1, min(21, len(S207)+1)), S207[:20], 'o-', color='steelblue')
axes[1].set_title("Singular values: 207 mask-identified (top 20)")
axes[1].set_xlabel("Component index")
axes[1].set_ylabel("Singular value")
Save: phase1_plot4_singular_values.png

# PLOT 5: Cumulative variance explained
fig, ax = subplots(1, 1)
ax.plot(range(1, 14), cum_var_13, 'o-', label='13 CFR survivors', color='steelblue')
ax.plot(range(1, min(14, len(cum_var_207)+1)), cum_var_207[:13], 's--', label='207 mask-identified', color='coral')
ax.axhline(y=0.95, color='red', linestyle=':', label='95% threshold')
ax.axhline(y=0.99, color='darkred', linestyle=':', label='99% threshold')
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative variance explained")
ax.set_title("Effective dimensionality of receptor direction sets")
ax.legend()
Save: phase1_plot5_cumulative_variance.png


==============================
PART C: EMBEDDING GEOMETRY AUDIT
==============================

# Load embedding matrix
W_E = model.transformer.wte.weight  # shape (50257, 768)

# Load name-gender data from dataset
# For each name in the dataset, get:
#   - token_id (from tokenizer)
#   - he_fraction (from dataset)
# Partition into male (he_frac > 0.7), female (he_frac < 0.3), ambiguous (rest)

# Get name embeddings
name_embeddings = []
he_fractions = []
for name in name_list:
    token_ids = tokenizer.encode(" " + name)  # add space prefix as GPT-2 does
    if len(token_ids) == 1:
        emb = W_E[token_ids[0]]
        name_embeddings.append(emb)
        he_fractions.append(he_frac_dict[name])

name_embeddings = stack(name_embeddings)  # (N_names, 768)
he_fractions = array(he_fractions)

# Compute gender direction
male_mask = he_fractions > 0.7
female_mask = he_fractions < 0.3
mean_male = mean(name_embeddings[male_mask], dim=0)
mean_female = mean(name_embeddings[female_mask], dim=0)
d_gender = mean_male - mean_female
d_gender = d_gender / norm(d_gender)

# Alignment measurements
cos_R1_gender = dot(v_R1, d_gender)
cos_R3_gender = dot(v_R3, d_gender)
cos_dark_gender = dot(v_dark, d_gender)

PRINT: f"cos(R1, d_gender) = {cos_R1_gender:.4f}"
PRINT: f"cos(R3, d_gender) = {cos_R3_gender:.4f}"
PRINT: f"cos(dark, d_gender) = {cos_dark_gender:.4f}"

# Project name embeddings onto each receptor direction
proj_R1 = name_embeddings @ v_R1    # (N_names,)
proj_R3 = name_embeddings @ v_R3    # (N_names,)
proj_dark = name_embeddings @ v_dark # (N_names,)

# Embedding AUC: how well does the embedding projection separate genders?
from sklearn.metrics import roc_auc_score
labels = (he_fractions > 0.5).astype(int)  # 1 = male-leaning, 0 = female-leaning
emb_auc_R1 = roc_auc_score(labels, proj_R1)
emb_auc_R3 = roc_auc_score(labels, proj_R3)
emb_auc_dark = roc_auc_score(labels, proj_dark)
# Handle polarity: if AUC < 0.5, flip
emb_auc_R1 = max(emb_auc_R1, 1 - emb_auc_R1)
emb_auc_R3 = max(emb_auc_R3, 1 - emb_auc_R3)
emb_auc_dark = max(emb_auc_dark, 1 - emb_auc_dark)

PRINT: f"Embedding AUC along R1 direction: {emb_auc_R1:.4f}"
PRINT: f"Embedding AUC along R3 direction: {emb_auc_R3:.4f}"
PRINT: f"Embedding AUC along dark direction: {emb_auc_dark:.4f}"

# Fraction of d_gender captured by receptor subspace
# Project d_gender onto span(v_R1, v_R3, v_dark)
B = directions_3.T  # (768, 3)
proj_coeff = lstsq(B, d_gender)  # or: B @ pinv(B) @ d_gender
d_gender_projected = B @ proj_coeff
captured_frac = norm(d_gender_projected)**2  # since d_gender is unit norm, this is the cos² of the angle
PRINT: f"Fraction of d_gender in receptor subspace: {captured_frac:.4f}"

# PLOT 6: Embedding projections (3 panels)
fig, axes = subplots(1, 3, figsize=(16, 5))

for ax, proj, title, auc_val in zip(axes,
    [proj_R1, proj_R3, proj_dark],
    ["R1 (L10H9sv0)", "R3 (L9H7sv1)", "Dark (L11H1sv1)"],
    [emb_auc_R1, emb_auc_R3, emb_auc_dark]):
    
    scatter(he_fractions, proj, c=he_fractions, cmap='RdBu_r', alpha=0.6)
    ax.set_xlabel("He-fraction")
    ax.set_ylabel("Embedding · receptor direction")
    ax.set_title(f"{title}\nEmb AUC = {auc_val:.3f}")
    
Save: phase1_plot6_embedding_projections.png

# PLOT 7: Bar chart of cosine alignments
fig, ax = subplots(1, 1)
bars = [abs(cos_R1_gender), abs(cos_R3_gender), abs(cos_dark_gender)]
ax.bar(["R1", "R3", "Dark"], bars, color=['steelblue', 'coral', 'gray'])
ax.set_ylabel("|cos(receptor, d_gender)|")
ax.set_title("Receptor alignment with embedding gender axis")
for i, v in enumerate(bars):
    ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
Save: phase1_plot7_alignment_bars.png


==============================
PART D: RECEPTOR RANK SPECTRUM
==============================

# OCA convergence R² values (from OCA experiment output)
# These are cumulative: R²(1) when adding 1st direction (dark, highest AccDrop),
#                        R²(2) when adding 2nd, R²(3) when adding 3rd = 0.9815
R2_values = [R2_1, R2_2, R2_3]  # fill from OCA output

# Marginal gain
rho = [R2_values[0],
       R2_values[1] - R2_values[0],
       R2_values[2] - R2_values[1]]

# Direction labels (ordered by AccDrop: dark first, then R1, then R3)
labels = ["Dark (L11H1sv1)\nAccDrop=8.17%",
          "R1 (L10H9sv0)\nAccDrop=3.59%",
          "R3 (L9H7sv1)\nAccDrop=1.31%"]

# ICR (Information Concentration Ratio)
ICR = rho[0] / sum(rho)

PRINT: f"Marginal R² gains: {rho}"
PRINT: f"ICR (top direction): {ICR:.4f}"

# PLOT 8: Bar chart of marginal R² gains
fig, ax = subplots(1, 1, figsize=(8, 5))
bars = ax.bar(range(3), rho, color=['gray', 'steelblue', 'coral'])
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_ylabel("Marginal R² gain")
ax.set_title("Receptor rank spectrum (marginal explanatory power)")
# Annotate cumulative on top
for i, (r, cumr) in enumerate(zip(rho, R2_values)):
    ax.text(i, r + 0.005, f"Δ={r:.4f}\ncum={cumr:.4f}", ha='center', fontsize=9)
Save: phase1_plot8_receptor_spectrum.png


==============================
FINAL SUMMARY TABLE
==============================

PRINT: "="*60
PRINT: "PHASE 1: CAUSAL SUBSPACE GEOMETRY — SUMMARY"
PRINT: "="*60
PRINT: f"Mean |cos| (207 raw):         {mean_cos_207:.4f}"
PRINT: f"Mean |cos| (13 CFR):          {mean_cos_13:.4f}"
PRINT: f"Mean |cos| (3 OCA):           {mean_cos_3:.4f}"
PRINT: f"Decorrelation ratio (207→3):  {mean_cos_207/mean_cos_3:.2f}x"
PRINT: f""
PRINT: f"Effective rank 13 (95%):      {eff_rank_95}"
PRINT: f"Effective rank 13 (99%):      {eff_rank_99}"
PRINT: f"Stable rank (13 survivors):   {stable_rank_13:.2f}"
PRINT: f"RSI (Superposition Index):    {RSI:.4f}"
PRINT: f""
PRINT: f"Effective rank 207 (95%):     {eff_rank_207_95}"
PRINT: f"Effective rank 207 (99%):     {eff_rank_207_99}"
PRINT: f"Stable rank (207 raw):        {stable_rank_207:.2f}"
PRINT: f""
PRINT: f"cos(R1, d_gender):            {cos_R1_gender:.4f}"
PRINT: f"cos(R3, d_gender):            {cos_R3_gender:.4f}"
PRINT: f"cos(dark, d_gender):          {cos_dark_gender:.4f}"
PRINT: f"Embedding AUC (R1 dir):       {emb_auc_R1:.4f}"
PRINT: f"Embedding AUC (R3 dir):       {emb_auc_R3:.4f}"
PRINT: f"Embedding AUC (dark dir):     {emb_auc_dark:.4f}"
PRINT: f"d_gender captured fraction:   {captured_frac:.4f}"
PRINT: f""
PRINT: f"ICR (top direction share):    {ICR:.4f}"
PRINT: f"Marginal R²: {rho}"
PRINT: "="*60
```

---

## What You Need To Do

1. **Check what's in `svd_cache.pt`** — You need to know the exact key structure. It might be keyed by `(layer, head)` with values containing the V matrix and singular values from the OV-SVD. Print `svd_cache.keys()` and `svd_cache[some_key].keys()` to see the structure.

2. **Provide the 207 mask-identified triples** — These should be saved from the mask training output. If they're in a file, load them. If not, you may need to re-extract them from the mask weights (nonzero entries).

3. **Provide the 13 CFR survivors** — These should be saved from the CFR experiment output.

4. **Provide the OCA convergence R² values** — R²(1), R²(2), R²(3) in order of AccDrop. R²(3) = 0.9815 we know. We need the intermediate values.

5. **Provide the name-gender mapping** — The dataset CSV with names and their he-fractions.

Give all of this to the coding Claude, along with the pseudocode above, and ask it to produce a clean Python script that generates all 8 plots and the summary table.

---

## Experiment Rating (Pre-Execution)

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Novelty | 7/10 | SVD of causal directions + superposition index + embedding alignment is new |
| Rigor | 9/10 | Every claim backed by a specific number; SVD/cosine/AUC are standard tools |
| Effort | 1-1.5 hrs | Single script, no forward passes, loads saved data |
| Feasibility | 10/10 | Pure linear algebra on saved tensors |
| Publishability impact | +1.5 points | Superposition connection + mechanistic embedding proof |
| Likelihood of clean results | 9/10 | SVD will almost certainly show 3 dominant singular values; embedding alignment is very likely given commitment results |
| Risk of trivial results | Low | Even if predictions are violated, violations are interesting |
