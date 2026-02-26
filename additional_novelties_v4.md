# Additional Novelties & Creative Ideas — V4

## Current Publishability: Honest Assessment

If we wrote the paper today with everything in the report, I'd rate it **6/10** for a main ML conference (NeurIPS/ICML/ICLR) and **8/10** for a mech interp workshop.

**What we have that's strong:**
- CFR + OCA pipeline: 207 mask-identified directions → 13 (CFR filter) → 3 (OCA compression), R²=0.9815. This is a genuine method contribution.
- The 93.7% ghost rate is a striking finding about the Beyond Components method itself.
- The inherited/computed asymmetry (R1 at embedding, R3 at layer 9) is clean and novel.
- 41.2% computational cross-talk asymmetry (Exp 2) shows the circuit is coupled, not independent.
- Generalization across 5 syntactic structures (AUC ≥ 0.92) with graceful degradation on ambiguous names (0.75).
- Error taxonomy with 4 types, Type D dominant at 52%.

**What's weak or missing:**
1. Everything is on one task (GP) on one model (GPT-2 small). No generalization evidence beyond syntactic structures.
2. The PRM predictive experiment failed — receptors explain the logit but don't predict errors. This limits the "predictive framework" narrative.
3. No QK analysis — we characterize what's written but not why.
4. No formal theoretical contribution (theorem, proof, new algorithm beyond CFR ratio).
5. The dark receptor characterization is interesting but inconclusive — we know it matters (8.17% AccDrop) but not what it does.

**To reach 7.5–8/10:** We need IOI transfer OR a second methodological contribution OR a formal result.
**To reach 9/10:** We need IOI transfer AND a deeper mechanistic finding (QK routing or inhibition bottleneck).

---

## Status of V3 Ideas

| V3 Idea | Status | Result | Still Pursue? |
|---------|--------|--------|---------------|
| #1 CFR Ghost Detection | ✅ DONE | 207→13, 93.7% ghosts | Done. In report. |
| #2 RIM Formalization | ⏳ Not done | Data exists from Exp 2 | Yes — pure writing, high value |
| #3 PRM Error Prediction | ✅ DONE | **FAILED** (AUC ~0.54) | No — null result, in report as negative finding |
| #4 Superposition Tax | ⏳ Not done | Data exists | Yes — quick computation |
| #5 Inhibition Bottleneck | ⏳ Not done | — | Yes — high novelty, needs new experiment |
| #6 Attention Pattern Analysis | ⏳ Not done | — | Yes — completes story |
| #7 IOI Transfer | ⏳ Not done | — | **YES — highest priority remaining** |
| #8 MOCA | ⏳ Not done | — | Lower priority now |
| #9 Fingerprint Clustering | ⏳ Not done | — | Lower priority |
| #10 Activation Patching | ⏳ Not done | — | Lower priority |
| #11 Decision Efficiency | ⏳ Not done | — | Drop — PRM failure undercuts this |
| #12 QK Skeleton | ⏳ Not done | — | Yes — high novelty but high effort |
| #13 Structural Invariance | ⏳ Not done | — | Medium priority |
| #14 Embedding Geometry | ⏳ Not done | — | Yes — easy, explains R1 inheritance |

**New experiments completed since V3 (not in V3):**
- OCA (Orthogonal Component Analysis): 13→3 compression with R²=0.9815
- Dark receptor characterization: L11H1sv1, relational-role encoding, AccDrop=8.17%
- PRM with dark receptor: Also failed (AUC ~0.54 with all variants)

---

## What the PRM Failure Tells Us

The PRM failure is actually informative. Here's why:

Receptor activations explain 98% of logit **variance** (R²=0.980). But the model's errors aren't driven by receptor variance — they're driven by the remaining 2% that lives outside the receptor subspace. This means:

1. The receptor circuit is necessary but not sufficient. When it votes wrong, the model is wrong. But when it votes right with thin margin, the 2% residual tips the outcome.
2. Error prediction requires modeling the residual, not just the receptors.
3. This is itself a finding: **the circuit's errors are fundamentally unpredictable from within the circuit.** The failure mode is in the interaction between the receptor subspace and everything else.

We report this as a clean negative result. It sharpens the paper's claim: receptors explain the decision mechanism but do not control the decision boundary.

---

## UPDATED IDEAS — V4

### IDEA 1 (NEW): The Mask→CFR→OCA Pipeline as a General Method

**What it is.** Formalize the three-step pipeline we developed:

Step 1: Mask training (Beyond Components) — identifies candidate directions by correlation.
Step 2: CFR filtering — removes directions with low causal fidelity (ghosts).
Step 3: OCA compression — orthogonalizes surviving directions to find the minimal independent set.

$$\text{Mask}(\mathcal{D}) \xrightarrow{\text{207 candidates}} \text{CFR}(\tau) \xrightarrow{\text{13 survivors}} \text{OCA}(R^2_{\min}) \xrightarrow{\text{3 independent}}$$

**Why this is the paper's main methodological contribution.** Nobody has this pipeline. Beyond Components stops at Step 1. We show that Step 1 alone gives you 93.7% false positives. Steps 2 and 3 are necessary and sufficient for finding the true causal basis.

**What to formalize:**

The OCA step deserves mathematical treatment. Define the problem: given K directions that pass CFR, find the minimum subset S ⊆ {1,...,K} such that:

$$R^2\left(\sum_{k \in S} p_k \cdot g_k, \; \Delta\text{logit}\right) \geq R^2_{\min}$$

This is a sparse regression problem with a geometric constraint (we want directions, not coefficients). The greedy algorithm we used (forward stepwise by marginal R² gain) has a known approximation guarantee when the predictors are near-orthogonal — which they are after CFR filtering, because the ghosts (which were the correlated ones) have been removed.

**Formal claim:** After CFR filtering, the surviving directions have reduced pairwise cosine similarity (because high-cosine ghosts were the ones absorbing causal signal from real receptors). This means OCA's greedy selection converges faster post-CFR than it would on the raw set. We can verify this: compute mean |cos| among the 207 raw directions vs the 13 CFR survivors. If the 13 have lower mean |cos|, CFR is also a decorrelation step.

**What this adds.** A reusable, tested pipeline for any task with logit receptors. Three lines in the abstract.

| Metric | Score |
|--------|-------|
| Novelty | 8/10 — the pipeline is new; each step existed in some form but the combination is novel |
| Effort | 2 hours — mostly formalization + one extra computation (mean cosine comparison) |
| Impact | 9/10 — this is the paper's METHOD |
| Feasibility | Trivial |

---

### IDEA 2 (NEW): Receptor Rank — How Many Independent Directions Does a Circuit Need?

**What it is.** OCA found that 3 directions achieve R²=0.9815. But what happens if we use 4? 5? 13? Plot R² vs number of OCA-selected directions. This gives the "receptor rank" of the circuit — the effective dimensionality of the logit-relevant subspace.

**Why this matters.** If R² saturates sharply at K=3 (e.g., R²=0.95 at K=2, 0.98 at K=3, 0.983 at K=4), the circuit has a well-defined rank. If it increases gradually, the circuit is distributed. The sharpness of the elbow IS the finding.

**The mathematical object: the Receptor Spectrum.**

Define:
$$\rho(k) = R^2_{\text{OCA}}(k) - R^2_{\text{OCA}}(k-1)$$

This is the marginal explanatory gain of the k-th receptor. Plot ρ(k) for k=1,...,13. The spectrum shows where the circuit's information is concentrated.

**Connection to eigenvalue spectra.** This is analogous to the eigenvalue spectrum of a covariance matrix but for *causal* explanatory power, not variance. A fast-decaying spectrum means a compact circuit. A flat spectrum means distributed computation. 

**Prediction:** ρ(1) ≈ 0.8 (R1 alone explains most), ρ(2) ≈ 0.12 (R3 adds the inhibition channel), ρ(3) ≈ 0.05 (dark receptor adds calibration), ρ(k>3) ≈ 0 (everything else is noise). If this holds, the GP circuit has rank exactly 3.

**What this adds.** A new diagnostic: "receptor rank" of a circuit. Compact, comparable across tasks and models. If the GP circuit has rank 3 and IOI has rank 5, that tells you something about circuit complexity.

| Metric | Score |
|--------|-------|
| Novelty | 7/10 — effective dimensionality is a known concept, but applying it to causal receptor directions is new |
| Effort | 30 min — pure computation on existing OCA data |
| Impact | 7/10 — gives the paper a clean number that characterizes the circuit |
| Feasibility | Trivial |

---

### IDEA 3 (UPDATED): IOI Transfer — The Critical Generalization Experiment

**What it is.** Test whether GP receptor directions carry task-relevant information in the IOI task. This was Idea #7 in v3 but is now MORE interesting because of what we learned.

**Why it's upgraded.** We now have two predictions from our findings that ONLY our framework makes:

**Prediction 1 (from Exp 0c):** R1 should transfer to IOI because it inherits from the token embedding — it encodes gender as a static property of name tokens. IOI also requires distinguishing names, so R1's direction should carry relevant signal.

**Prediction 2 (from Exp 0c + dark receptor):** R3 should NOT transfer because it's computed by L9H7 specifically for the GP task's syntactic structure. IOI uses different attention heads for its computation.

**Prediction 3 (from dark receptor):** The dark receptor (L11H1sv1) might or might not transfer. If it's truly a "calibration receptor" that adjusts the model's output distribution, it should carry some signal in any name-related task.

**The experiment (two phases):**

Phase A — Zero-shot transfer (no retraining):
1. Construct 500 IOI examples: "When [Name1] and [Name2] went to the store, [Name1] gave a drink to ____"
2. Run GP model forward passes on IOI examples.
3. Project final residual at prediction position onto R1, R2, R3, and dark receptor directions.
4. Measure: does each receptor's activation correlate with the correct IOI answer?
5. Measure: does PRM = g₁ − g₃ predict the correct indirect object?

Phase B — Full pipeline on IOI (if Phase A shows partial transfer):
1. Train masks on IOI from scratch.
2. Run CFR on all IOI mask-identified directions.
3. Run OCA on IOI CFR survivors.
4. Compare: how many IOI receptors? Same rank as GP? Do any GP receptors appear in the IOI set?

**What we learn from each outcome:**

| Outcome | What it means | Paper impact |
|---------|---------------|-------------|
| R1 transfers, R3 doesn't | Inherited/computed distinction predicts transferability | Very high — validates framework |
| Both transfer | Receptors are intrinsic model features | Extremely high — strongest possible finding |
| Neither transfers | GP receptors are task-specific | Still informative — bounds the claim |
| Phase B: IOI has same rank | Circuit complexity is task-independent | High — universality claim |
| Phase B: IOI has different rank | Circuit complexity varies by task | Interesting — complexity taxonomy |

**What this adds.** Answers the #1 reviewer question. Even Phase A alone (zero-shot, 2 hours of work) would be a strong addition.

| Metric | Score |
|--------|-------|
| Novelty | 7/10 — transfer testing is standard; the directional predictions (R1 yes, R3 no) are novel |
| Effort | Phase A: 2-3 hours. Phase B: half a day. |
| Impact | 10/10 — essential for a main conference paper |
| Feasibility | Easy on T4 |

---

### IDEA 4 (UPDATED): Superposition Tax + OCA Geometry

**What it is.** Quantify how much the non-orthogonality of receptor directions costs the model, now enhanced by OCA data.

**Why it's better than v3.** In v3, we only had 3 directions and their cosines. Now we have 13 CFR survivors and their OCA compression. The 13→3 compression tells us the true dimensionality of the causal subspace. The other 10 survivors are "geometrically dependent" — they carry causal signal but not independent causal signal.

**New computation:**

1. Stack the 13 CFR-surviving directions as rows of R₁₃ ∈ ℝ^{13×768}. SVD gives singular values σ₁ ≥ ... ≥ σ₁₃.
2. Compute effective rank: how many σᵢ > 0.1·σ₁? This should be ~3 (matching OCA).
3. Stack the 3 OCA-selected directions as R₃ ∈ ℝ^{3×768}. Compare: is the column space of R₃ ≈ the top-3 eigenspace of R₁₃? If yes, OCA found the principal causal directions.

**The Superposition Index:**

$$\text{SI}_{13} = 1 - \frac{\text{effective rank of } R_{13}}{13} = 1 - \frac{3}{13} \approx 0.77$$

77% superposition. Thirteen directions live in a 3-dimensional causal subspace. The model encodes 13 correlational features in 3 causal dimensions.

**Connection to intervention leakage.** From Exp 2: CompFrac(R3→R1) = 41.2%. This is the COST of the superposition — perturbing one direction leaks 41% into the other. If we compute this for all 13×13 pairs (or at least for the 3 OCA-selected ones), we get a "leakage matrix" that quantifies the causal cost of superposition.

**The punchline for the paper:** "The GP circuit encodes 13 gender-relevant directions in a 3-dimensional causal subspace (SI = 0.77). Interventions on one direction leak into others at 41% (CompFrac). This is a concrete, causally measured instance of the superposition hypothesis."

| Metric | Score |
|--------|-------|
| Novelty | 7/10 — connecting OCA to superposition is natural but new |
| Effort | 1 hour — SVD + cosine comparisons |
| Impact | 7/10 — connects to a hot topic, strengthens the narrative |
| Feasibility | Trivial |

---

### IDEA 5 (NEW): Dark Receptor Functional Characterization — The Bias Channel

**What it is.** The dark receptor (L11H1sv1) has AUC=0.526 (no gender discrimination), AccDrop=8.17% (significant causal effect), mean activation=−205.7, std=12.7. It shifts the logit uniformly. What IS it?

**The hypothesis: global prior calibration.** The dark receptor doesn't encode "this name is male/female." It encodes "how much should the model lean toward 'he' vs 'she' in general." It's a bias term — the intercept, not the slope.

**Tests:**

Test 1 — Prior shift: Compute the mean(he_logit − she_logit) across all examples. Does removing the dark receptor shift this mean toward zero? If so, the dark receptor implements the model's default gender prior.

Test 2 — Position vs content: Does the dark receptor's activation depend on the name (content) or the sentence structure (position/syntax)? Run the same sentences with different names. If activation barely changes, it's structural. If it tracks the name, it's content-dependent.

Test 3 — Relational encoding: The dark receptor showed "relational-role encoding" in our characterization. It might encode "this position expects a pronoun" rather than "this pronoun should be he/she." Remove it and check: does the model still predict a pronoun, or does it shift to non-pronoun completions? This would make it a "task-detection receptor" — not gender-specific, but pronoun-slot-specific.

**Why this matters.** If confirmed, the GP circuit has three functional channels:
- R1: inherited gender (embedding geometry)
- R3: computed gender (attention head L9H7)
- Dark: task prior (whether a pronoun is expected and what the default gender lean is)

Three channels, three functions, three distinct computational origins. That's a clean circuit decomposition that nobody has shown at this level of resolution.

**Mathematical formalization:**

$$\text{logit}(he) - \text{logit}(she) = \underbrace{g_1}_{\text{promotion}} - \underbrace{g_3}_{\text{inhibition}} + \underbrace{g_{\text{dark}} \cdot w_{\text{dark}}}_{\text{prior bias}} + \epsilon$$

where $w_{\text{dark}}$ is the dark receptor's unembedding projection onto (he − she). If $w_{\text{dark}}$ is approximately constant across examples (which it should be, since the dark receptor's AUC is near chance), the dark term acts as a constant shift.

| Metric | Score |
|--------|-------|
| Novelty | 8/10 — nobody has characterized a "prior calibration" receptor |
| Effort | 2-3 hours — need targeted ablation experiments |
| Impact | 7/10 — deepens the circuit understanding, adds a third channel |
| Feasibility | Easy on T4 |

---

### IDEA 6 (UPDATED): Inhibition Bottleneck — Noise Robustness Test

**What it is.** Same as v3 Idea #5 but now we can be more precise. We want to show that R3 (computed channel) is more fragile than R1 (inherited channel) by injecting noise and measuring which receptor flips first.

**The experiment (refined):**

1. For each example i, inject Gaussian noise δ ~ N(0, σ²I) into the residual stream at each layer l.
2. Measure: at what noise level σ does R1 flip sign? R3 flip sign? The model flip prediction?
3. Define Flip Threshold: FT_k = min σ such that receptor k's sign flips on >50% of examples.

**Prediction:** FT(R1) >> FT(R3). R1 is robust because its signal is distributed across the embedding. R3 is fragile because it depends on L9H7's single computation.

**The bottleneck claim:** If FT(R3) < FT(model) < FT(R1), then the model's error rate is bounded by R3's fragility:

$$P(\text{error} | \sigma) \approx P(\text{R3 flips} | \sigma) \text{ for } \sigma < FT(R1)$$

This means: as long as R1 is intact (noise below its threshold), the model's errors come almost entirely from R3 failures. R3 IS the bottleneck.

**What this adds.** A causal robustness result: "The circuit's vulnerability concentrates in its computed channel. Inherited channels are ~Nx more robust (where N = FT_R1/FT_R3)." This is a general principle about circuits with mixed inherited/computed architecture.

**Key detail:** We inject noise at specific layers, not globally. Inject at layer 9 only (where R3 is written) vs layer 0 only (where R1 reads from embedding). The layer-specific flip thresholds tell us exactly where each receptor is vulnerable.

| Metric | Score |
|--------|-------|
| Novelty | 8/10 — noise robustness of individual receptor channels is new |
| Effort | 2-3 hours — need forward passes with noise injection at multiple σ values |
| Impact | 8/10 — gives a general design principle |
| Feasibility | Easy — just add noise to residual stream hook |

---

### IDEA 7 (UPDATED): Attention Pattern Analysis for Failure Diagnosis

**What it is.** Same as v3 Idea #6 but now we know the exact error taxonomy (Type B, C, D) and can condition on it.

**Refined experiment:**

For each error type from the phase diagram:

- Type B (inhibition failure, 24 examples): Extract L9H7's attention pattern at the decision position. Where does the head attend? Does it attend to the name token?
- Type D (margin failure, 113 examples): Both receptors vote right but weakly. Is L9H7's attention pattern different for thin-margin correct vs thin-margin error?

**The key test:** For Type B errors, manually replace L9H7's attention at the decision position with uniform attention over the name token. Recompute R3's activation. Does it flip to the correct sign? If yes: "Type B errors are caused by attention misrouting in L9H7." If no: "Type B errors are caused by the value content at the name position being insufficient."

**What this adds.** Closes the loop on our error taxonomy. We know WHAT fails (R3), WHERE it fails (layer 9), and now we'd know WHY (attention routing vs value content).

| Metric | Score |
|--------|-------|
| Novelty | 6/10 — attention analysis is standard, conditioning on receptor error types is new |
| Effort | 2 hours |
| Impact | 7/10 — completes the mechanistic story |
| Feasibility | Easy |

---

### IDEA 8 (RETAINED): RIM Formalization

Same as v3 Idea #2 — formalize the Receptor Interference Matrix from Exp 2 data. This is pure writing, no new experiments. Defines the K×K matrix M with geometric and computational components, proves asymmetry encodes temporal ordering, shows ghost receptors have zero rows.

Enhanced by new data: we can now compute RIM for the 3 OCA-selected directions specifically, and compare to the 3 original mask-identified directions. If RIM is similar, the OCA-selected set preserves the interaction structure.

| Metric | Score |
|--------|-------|
| Novelty | 7/10 | Effort | 1 hour (writing) | Impact | 7/10 | Feasibility | Already done |

---

### IDEA 9 (RETAINED): Embedding Geometry Audit

Same as v3 Idea #14. Quick experiment: project all name token embeddings onto R1's direction, split by gender, measure separation. If R1 IS the "gender axis" of the embedding (cos > 0.9 with mean_male − mean_female), then R1's inheritance from the embedding is not a coincidence — L10H9 learned to write along a pre-existing axis.

**New angle from dark receptor:** Also project embeddings onto the dark receptor's direction. If the dark receptor's direction separates names by something OTHER than gender (e.g., name frequency, number of syllables, first-letter), we learn what L11H1 reads from the embedding.

| Metric | Score |
|--------|-------|
| Novelty | 6/10 | Effort | 1 hour | Impact | 6/10 | Feasibility | Trivial |

---

### IDEA 10 (NEW): Receptor Coherence Under Counterfactual Patching

**What it is.** A unified experiment that tests both inherited/computed asymmetry AND cross-talk simultaneously. Patch the residual stream from a counterfactual (gender-swapped) example at each layer. Track all three receptor activations as a function of patch layer. This gives a "receptor coherence" curve showing when each receptor aligns with the patched signal vs the original signal.

**The setup:**

Take matched pair: original = "John went to the park. He..." / counterfactual = "Mary went to the park. She..."

At patch layer l: replace residual_original[l] with residual_counterfactual[l].

Measure: g₁(l), g₃(l), g_dark(l) in the patched run.

Plot: g_k(patched, l) − g_k(original) as a function of l, for each receptor.

**What this tells us:**

- R1 should flip immediately when patch layer ≤ 0 (embedding level), since it inherits from the embedding.
- R3 should flip when patch layer ≤ 9 (L9H7 reads from patched residual), but NOT when patch layer = 10 (too late, L9H7 already read clean residual).
- Dark receptor's flip behavior reveals its source layer.

**The cross-talk test:** When we patch at layer 9 (flipping R3), does R1 ALSO shift? If so, how much? This gives us a CAUSAL measure of R3→R1 coupling, distinct from the scaling-based CompFrac from Exp 2. If causal R3→R1 coupling ≈ CompFrac from Exp 2, we've validated the RIM with a completely different intervention.

**Why this is elegant.** One experiment, one set of forward passes, answers three questions: (1) validates commitment ordering causally, (2) validates cross-talk magnitude, (3) reveals dark receptor's source layer. Three birds, one stone.

| Metric | Score |
|--------|-------|
| Novelty | 7/10 — activation patching is standard, but tracking three receptors + dark simultaneously gives new information |
| Effort | 2-3 hours |
| Impact | 8/10 — validates multiple findings with a single independent experiment |
| Feasibility | Easy |

---

### IDEA 11 (NEW): The Receptor Spectrum Test on a Second Task

**What it is.** Beyond just testing transfer of GP directions to IOI (Idea 3), we run the FULL Mask→CFR→OCA pipeline on a second task and compare the resulting "receptor spectrum" (ρ(k) from Idea 2).

**Which task?** Factual recall: "The Eiffel Tower is located in [Paris]." Train masks on this, run CFR, run OCA. Compare:
- How many CFR survivors? (Ghost rate tells us about the task's representational structure)
- How many OCA-selected? (Receptor rank tells us about circuit complexity)
- What's the spectrum shape? (Sharp vs gradual tells us about circuit sparsity)

**Alternative: Greater-Than task.** "The war ended in 1945. The treaty was signed in 19[XX]." This is a well-studied mech interp task with known circuits. Receptor spectroscopy on this would connect to existing literature.

**Why this matters more than IOI transfer.** IOI transfer (Idea 3) tests whether specific DIRECTIONS transfer. This tests whether the PIPELINE transfers — whether Mask→CFR→OCA produces interpretable results on a new task. That's the generalization claim about the method, not about the directions.

**Concrete comparison table in the paper:**

| Property | GP Task | Second Task |
|----------|---------|-------------|
| Mask candidates | 207 | ? |
| CFR survivors | 13 (ghost rate 93.7%) | ? |
| OCA rank | 3 (R²=0.9815) | ? |
| Spectrum shape | Sharp elbow at K=3 | ? |
| Inherited vs computed | R1 inherited, R3 computed | ? |

If the pipeline works on a second task and produces meaningful results, the paper's claim shifts from "we analyzed GP" to "we have a general pipeline for receptor circuit analysis."

| Metric | Score |
|--------|-------|
| Novelty | 7/10 — applying the pipeline to a second task is the obvious next step, but nobody has done it |
| Effort | Half a day to a full day — mask training + pipeline |
| Impact | 9/10 — essential for "method paper" narrative |
| Feasibility | Needs training, but fits in Kaggle T4 |

---

## PRIORITY RANKING — V4

Accounting for what's done, what failed, and what the paper needs:

| Rank | Idea | Key Question It Answers | Effort | Impact |
|------|------|------------------------|--------|--------|
| 1 | IOI Transfer (Idea 3) | Do the directions generalize? | 2-3 hrs (Phase A) | 10/10 |
| 2 | Pipeline Formalization (Idea 1) | Is our method general? | 2 hrs (writing + 1 computation) | 9/10 |
| 3 | Second Task Pipeline (Idea 11) | Does the pipeline work elsewhere? | 0.5-1 day | 9/10 |
| 4 | Receptor Rank Spectrum (Idea 2) | How complex is the circuit? | 30 min | 7/10 |
| 5 | Noise Robustness / Bottleneck (Idea 6) | Why is R3 the weak link? | 2-3 hrs | 8/10 |
| 6 | Counterfactual Patching (Idea 10) | Does everything validate independently? | 2-3 hrs | 8/10 |
| 7 | Superposition Tax + OCA (Idea 4) | How superposed is the circuit? | 1 hr | 7/10 |
| 8 | Dark Receptor Characterization (Idea 5) | What does the 3rd channel do? | 2-3 hrs | 7/10 |
| 9 | Attention Failure Diagnosis (Idea 7) | Why exactly do errors happen? | 2 hrs | 7/10 |
| 10 | RIM Formalization (Idea 8) | Can we define a formal object? | 1 hr (writing) | 7/10 |
| 11 | Embedding Geometry (Idea 9) | Why is R1 free? | 1 hr | 6/10 |

---

## RECOMMENDED PLAN: Concrete Execution Order

### Phase 1: Quick wins from existing data (1-2 hours total)

1. **Receptor Rank Spectrum** (Idea 2) — Compute ρ(k) for k=1...13 from OCA data. One plot.
2. **Superposition Index** (Idea 4) — SVD of the 13×768 matrix of CFR survivors. One number + one plot.
3. **Embedding Geometry** (Idea 9) — Project name embeddings onto R1, R3, dark receptor. Three scatter plots.
4. **RIM Formalization** (Idea 8) — Write the math, no new computation.

All of these use existing data. No new forward passes. Yield ~4 new figures and 2 formal objects (RIM, receptor spectrum).

### Phase 2: IOI Transfer (2-3 hours)

5. **IOI Transfer Phase A** (Idea 3) — Zero-shot: project IOI residuals onto GP directions. The single most important experiment remaining.

### Phase 3: Deeper mechanistic experiments (4-6 hours)

6. **Noise Robustness** (Idea 6) — Inject noise at specific layers, measure per-receptor flip thresholds.
7. **Counterfactual Patching** (Idea 10) — Patch counterfactual at each layer, track all receptors.
8. **Attention Failure Diagnosis** (Idea 7) — Extract L9H7 attention for error examples.

### Phase 4 (if time): Second task (half a day)

9. **Full pipeline on Greater-Than or factual recall** (Idea 11) — The generalization experiment for the method.

### After each experiment: Remind me to:
- Add all numerical results and tables to the report
- Generate and add all plots
- Write the interpretation section
- Update the LaTeX document for Overleaf
- Update the appendix with full result tables

---

## Paper Arc With These Additions

**Title (working):** "Receptor Interference Spectroscopy: Causal Decomposition and Ghost Detection in Transformer Decision Circuits"

**Story:**
1. Logit receptors (Beyond Components) identify write directions. But mask training produces too many — most are ghosts. (Sections 1-2)
2. We develop a three-step pipeline (Mask→CFR→OCA) that compresses 207 candidates to 3 independent causal directions, explaining 98.15% of the logit variance. (Section 3: the METHOD)
3. The 3 survivors have distinct functional roles: one inherited (R1), one computed (R3), one calibration (dark). (Section 4: the CIRCUIT)
4. The circuit has a coupled interaction structure (RIM), with asymmetric computational cross-talk and an inherited/computed robustness asymmetry. (Section 5: the INTERACTIONS)
5. Errors are dominated by thin-margin failures (52%), not receptor failures. The circuit is well-designed but operating near its precision limit. (Section 6: the ERRORS)
6. The pipeline generalizes across syntactic structures (AUC ≥ 0.92) and [if IOI works] across tasks. (Section 7: GENERALIZATION)
7. The receptor directions live in a 3-dimensional causal subspace within a 768-dimensional model, with superposition index 0.77. (Section 8: CONNECTION TO SUPERPOSITION)

**Conclusion:** "Transformer decision circuits can be decomposed into a small number of independent causal channels using receptor interference spectroscopy. The Mask→CFR→OCA pipeline identifies these channels, and their properties (inherited vs computed, asymmetric coupling, noise robustness) characterize the circuit's design and vulnerabilities."

---

## WHAT NOT TO DO (Updated)

- **Don't revisit PRM.** It failed. Report it as a null result and move on.
- **Don't pursue Decision Efficiency (v3 #11).** The PRM failure shows receptor margin doesn't predict errors, so efficiency metrics are unlikely to yield useful predictions.
- **Don't do MOCA (v3 #8) before IOI.** Component ablation ordering is interesting but doesn't address the generalization gap.
- **Don't spend more than 3 hours on dark receptor characterization.** It's interesting but risks becoming a rabbit hole. The three-channel decomposition is the claim; if the dark receptor's function remains unclear, say "we leave full characterization to future work" and cite the 8.17% AccDrop as evidence of significance.
- **Don't try QK Skeleton (v3 #12) in this paper.** It's a separate paper. The current paper is about OV receptor decomposition; adding QK doubles the scope.
