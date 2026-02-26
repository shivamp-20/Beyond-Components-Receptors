# Additional Novelties & Creative Ideas — V3 (Post All Experiments)

## What We Have (Grounding for Everything Below)

**Hard numbers from 7 completed experiments:**

- R1 (L10H9, sv0, pol +1): Inherits gender from token embedding. Sign agreement > 0.85 from layer −1. Shapley φ = 0.148.
- R3 (L9H7, sv1, pol −1): Computes gender at layer 9. Sign agreement crosses 0.85 at layer 9. Shapley φ = 0.085.
- R2 (L11H8, sv6, pol +1): Ghost. AUC = 0.952 but Shapley φ = −0.003. Correlational artifact.
- cos(R1, R3) = −0.775. Interaction index negative (redundant).
- R3→R1 computational fraction = 41.2%. R1→R3 = 19.3%. Asymmetry confirmed.
- Linearity of cross-talk: R² > 0.99 across all pairs at scales 1–20×.
- Phase diagram: 4 error types. Type B (inhibition failure) dominates errors.
- R² of receptor-vs-model logit: 0.980. Receptors explain 90.2% of logit variance.
- Exp 5: AUC ≥ 0.92 across 5 sentence structures. Exp 5b: AUC drops to 0.75 on ambiguous names.
- Delta peaks: R3 at layers 9→10, R1 at layers 10→11. Deliberation ratio explodes 10⁶× at late layers.

**The honest problem:** All of this is characterization of known objects (logit receptors) on a known task (gender pronouns, GPT-2 small). The experiments are solid but could be seen as "an extended appendix to Beyond Components." To make this a standalone paper, we need at least one of: (a) a new method that emerges from our findings, (b) a new theoretical object with formal properties, (c) a prediction that only our framework can make, or (d) a concrete connection to a bigger open problem.

The ideas below are ranked by how much they address this gap.

---

## IDEA 1: Causal Fidelity Ratio (CFR) — A Ghost Detection Method

**What it is.** A cheap, general-purpose diagnostic that identifies ghost receptors without running full Shapley.

**Why it comes from our results.** We discovered R2 is a ghost: AUC = 0.952 (looks great) but Shapley φ = −0.003 (causally dead). The mask training procedure in Beyond Components flagged R2 as important (mask weight ≥ 0.9). This means mask-based receptor selection systematically overestimates the number of real receptors. We need a causal filter.

**The method.**

For each receptor k, project out its direction from the final residual stream and measure accuracy drop:

$$\text{CFR}_k = \frac{\text{AccDrop}_k}{\max_j \text{AccDrop}_j}$$

From our data:
- CFR(R1) = 0.036 / 0.036 = 1.0
- CFR(R2) = 0.001 / 0.036 = 0.028 → ghost
- CFR(R3) = 0.013 / 0.036 = 0.361 → real but secondary

Threshold: CFR < 0.05 → ghost. Requires K forward passes, not 2^K.

**The extension that makes this a real contribution.** Don't just test on R1/R2/R3. Go back to the full set of all receptors identified by mask training (all heads × all sv_idx with mask weight ≥ 0.9 — there should be ~30-40 of them). Compute CFR for all of them. Plot the distribution. Prediction: most will be ghosts (CFR < 0.05), with only 2-4 being causally real. This would be a quantitative statement about the sparsity of true causal receptors vs. correlated ones, which is a finding about the Beyond Components method itself.

**What this adds to the paper.** A reusable methodological contribution. Anyone using logit receptors on any task can apply CFR to prune ghosts. This is not task-specific. It's a tool.

| Metric | Rating |
|--------|--------|
| Novelty | 8/10 — nobody has proposed a cheap ghost detection metric for logit receptors |
| Effort | 2-3 hours — one forward pass per receptor, pure inference |
| Impact on paper | 9/10 — turns an observation (R2 is ghost) into a method (here's how to detect ghosts in general) |
| Feasibility on Kaggle T4 | Easy |
| When to do it | Right after writing the report. This is the single highest-value addition. |

---

## IDEA 2: Receptor Interference Matrix (RIM) — A New Formal Object

**What it is.** A K×K matrix that encodes the full structure of receptor interactions, decomposed into geometric and computational components.

**Why it comes from our results.** Exp 2 gave us a 3×3 matrix of cross-talk values, and we decomposed each entry into geometric (cosine-predicted) and computational (residual) parts. The computational fraction matrix had clear structure: asymmetric (41.2% vs 19.3%), with R2's row near zero (confirming ghost status from a completely different angle). This matrix IS the object — we just haven't formalized it.

**The formalization.**

Define the Receptor Interference Matrix:

$$\text{RIM}_{jk}(\lambda) = \frac{1}{N} \sum_i \Delta g_j^{(k)}(i, \lambda)$$

where $\Delta g_j^{(k)}(i, \lambda)$ is the change in receptor j's activation when receptor k is scaled by $\lambda$.

Decompose:

$$\text{RIM}_{jk}(\lambda) = \underbrace{(\lambda - 1) \cdot \cos(r_j, r_k) \cdot \bar{g}_k}_{\text{geometric component}} + \underbrace{\text{residual}_{jk}(\lambda)}_{\text{computational component}}$$

From Exp 2, we know:
- The geometric component is exactly linear in (λ − 1). R² > 0.99.
- The computational component is ALSO linear in (λ − 1). This is non-obvious — it means the computational pathway is also linear in the perturbation strength.
- The full RIM is therefore a bilinear object: RIM ≈ (λ − 1) · M, where M is a constant K×K matrix.

**Properties of M:**
- Diagonal: self-coupling strength per receptor.
- Off-diagonal: directed coupling. M_jk ≠ M_kj (asymmetry = temporal ordering).
- The asymmetry of M encodes causal ordering: if receptor k writes before receptor j reads, M_jk > M_kj.
- Ghost receptors have near-zero rows in the computational component of M.

**What this adds.** RIM is a compact, formal object that summarizes the entire receptor interaction structure. You can compare RIMs across tasks, models, and layers. It's like a "circuit adjacency matrix" but at the receptor level. Nobody has defined this.

**The key claim to make in the paper:** "The RIM's asymmetry is a causal fingerprint of temporal ordering in the circuit. The entry M_jk − M_kj is nonzero if and only if there exists a computational pathway from receptor k's host layer to receptor j's host layer."

We can verify this: R3 (layer 9) → R1 (layer 10) has CompFrac 41.2%, while R1 (layer 10) → R3 (layer 9) has CompFrac 19.3%. The direction with the higher CompFrac is the one where the earlier receptor passes through the later receptor's host head. The 19.3% for R1→R3 exists because R1's perturbation at layer 10 still passes through layer 11 processing, which feeds back indirectly.

| Metric | Rating |
|--------|--------|
| Novelty | 7/10 — formalizing what we already computed, but the formalization itself is new |
| Effort | 1 hour — we already have all the numbers, just need to write the formalism |
| Impact on paper | 7/10 — gives the paper a defined theoretical object, not just experiments |
| Feasibility | Already done, just needs framing |
| When to do it | During paper writing. No new experiments needed. |

---

## IDEA 3: Predictive Receptor Margin (PRM) — Using Receptors to Predict Model Confidence

**What it is.** We already know receptors explain 98% of logit variance (R² = 0.980). But that's a post-hoc fit. Can we use the receptor activations to PREDICT, before seeing the model's output, which examples the model will get wrong?

**Why it comes from our results.** The phase diagram and error autopsy showed that errors cluster in specific regions of (R1, R3) space. Type B errors (inhibition failure) occur when R3 doesn't suppress enough. Type D errors (margin failure) occur when both receptors point right but the combined margin is thin. We have the exact boundaries.

**The method.**

Define the Predictive Receptor Margin:

$$\text{PRM}(i) = v_1 \cdot g_1^{(\text{final})}(i) + v_3 \cdot g_3^{(\text{final})}(i) = g_1(i) - g_3(i)$$

(ignoring R2 since it's a ghost)

Then:
- Threshold PRM at zero: if PRM > 0, predict "he"; if PRM < 0, predict "she."
- Measure the accuracy of this two-receptor predictor.
- Measure the calibration: does |PRM| predict the model's actual confidence (softmax probability gap)?

**What we expect.** Given R² = 0.980, the PRM predictor should be nearly as accurate as the full model. But the interesting cases are where PRM and the model disagree — these are examples where the 2% of variance NOT captured by receptors is decision-critical. These are the "beyond-receptor" examples where other components (MLP, non-host attention heads) override the receptor circuit.

**The real contribution: error prediction.** Rank examples by |PRM|. The bottom 10% (lowest margin) should be enriched for model errors. Measure the precision: what fraction of the bottom-10%-PRM examples are actual model errors? If this precision is high (say > 50%), then the receptor margin is a practical error predictor — you can identify fragile inputs without running the full model.

**What this adds.** Transforms the receptor framework from a post-hoc analysis tool into a predictive tool. "Given the receptor activations, we can predict which inputs the model will fail on" is a much stronger claim than "we can explain the model's outputs."

| Metric | Rating |
|--------|--------|
| Novelty | 7/10 — predictive use of receptors is new; the math is simple but the framing is powerful |
| Effort | 30 min — pure computation on cached tensors |
| Impact on paper | 8/10 — shifts the narrative from "analysis" to "prediction," which reviewers value |
| Feasibility | Trivial |
| When to do it | Immediately, use existing Exp 0c data |

---

## IDEA 4: The Superposition Tax — Quantifying the Cost of Non-Orthogonality

**What it is.** cos(R1, R3) = −0.775 is very high. The model is packing two functionally opposing directions (promotion and inhibition) into nearly anti-aligned directions. This IS superposition — but in a setting where we have causally validated directions, not SAE features.

**Why it comes from our results.** Exp 2 showed that scaling R3 leaks 41.2% computational + geometric coupling into R1. Exp 4 showed ablating R1 removes 60% of R3's geometric component (cos² ≈ 0.60). The non-orthogonality directly causes intervention impurity. This is a measurable cost.

**The method.**

Stack receptor directions as rows: R ∈ ℝ^{3×768}. SVD: R = UΣV^T.

Compute:
- Effective rank = number of singular values > 0.1 × σ₁.
- Receptor Superposition Index: RSI = 1 − (σ₃/σ₁).
- Geometric Tax = 1 − det(RR^T)^{1/3} / (σ₁σ₂σ₃)^{2/3}.

**The causal cost measurement.** From Exp 2, we have InterventionLeakage for all pairs. Plot InterventionLeakage_{k→j} against |cos(r_k, r_j)|. If they correlate, superposition directly causes intervention impurity.

**The punchline.** "The model encodes 3 functional directions in an effectively 2-dimensional subspace. The cost of this compression is a 41% intervention leakage from inhibition to promotion. This is a concrete, measured instance of the superposition hypothesis in a causally validated circuit."

**Connection to Elhage et al. (2022).** The toy models paper showed superposition exists in principle. We show it in a real circuit, with a real cost, measured causally. That's a meaningful empirical contribution to a hot topic.

| Metric | Rating |
|--------|--------|
| Novelty | 7/10 — connecting to superposition is obvious in hindsight, but nobody has done it with causally validated receptor directions |
| Effort | 30 min — SVD of a 3×768 matrix + plotting |
| Impact on paper | 7/10 — connects to a major open question in the field |
| Feasibility | Trivial |
| When to do it | During paper writing, use existing data |

---

## IDEA 5: The Inhibition Bottleneck Theorem — Formalizing Circuit Vulnerability

**What it is.** We found that R1 is inherited (robust, free, but shallow) and R3 is computed (powerful, fragile, task-specific). The errors are dominated by Type B (inhibition failure). This means the circuit's weakest link is always its computed channel.

**Why it comes from our results.** Exp 0c: R1 commits at embedding, R3 at layer 9. Phase diagram: Type B errors dominate. Exp 5b: ambiguous names primarily break R3 (inhibition) not R1 (promotion). Exp 2: R3 has higher computational fraction (29.3% self) than R1 (24.4% self), meaning R3's output is more heavily processed by downstream layers.

**The formalization.**

Define a receptor as "inherited" if its commitment layer is at or before the embedding (L* ≤ 0) and "computed" if L* > 0.

Claim: In any receptor system with mixed inherited/computed channels, the error rate is bounded by the computed channel's reliability:

$$P(\text{error}) \leq P(\text{computed receptor fails}) + \epsilon$$

where ε accounts for the rare cases where the inherited channel itself is wrong (ambiguous names).

**The experiment to validate this.** Inject Gaussian noise into the residual stream at each layer. Measure how much noise is needed to flip R1 vs R3. Prediction: R3 requires less noise to flip (it's computed by a single head at a single layer), R1 requires more (it's distributed across the entire embedding geometry).

More precisely: for each receptor k, compute:

$$\text{Robustness}_k = \min_{\|\delta\| = 1} \frac{|g_k^{\text{clean}}|}{|\nabla_\delta g_k|}$$

This is the ratio of clean activation to sensitivity. Higher = more robust.

**What this adds.** A general principle about circuit design: inherited channels are robust but uninformative (they carry whatever the embedding geometry gives for free), computed channels are informative but fragile (they depend on specific attention computations that can fail). This is a design principle, not just an observation about one circuit.

| Metric | Rating |
|--------|--------|
| Novelty | 8/10 — formalizing the inherited/computed distinction as a general principle is new |
| Effort | 2-3 hours — noise injection experiment needs gradient computation |
| Impact on paper | 8/10 — gives the paper a conceptual contribution beyond just numbers |
| Feasibility | Needs torch.autograd, ~1 hour on Kaggle |
| When to do it | After the report, before paper submission |

---

## IDEA 6: Receptor-Conditioned Attention Pattern Analysis

**What it is.** We know WHAT the receptors write (OV directions) and HOW MUCH they're entangled (Exp 2, 4). But we don't know WHY certain examples activate R3 strongly and others don't. The answer is in the attention patterns of the host heads.

**Why it comes from our results.** Exp 5b showed that ambiguous names cause AUC to drop from 0.92+ to 0.75. Where does this failure originate? It must be in the attention pattern of L9H7 (R3's host head) — the head isn't attending to the right source tokens for ambiguous names.

**The method.**

For each example, extract the attention pattern of L9H7 at the decision position. This gives a vector α ∈ ℝ^S (one weight per source position).

Split examples into:
- High-confidence correct (|PRM| > threshold, correct)
- Low-confidence correct (|PRM| < threshold, correct)
- Errors

For each group, average the attention patterns. Look at:
- Does L9H7 attend to the name token? How much?
- Does the attention distribution differ between correct and error cases?
- For ambiguous names specifically, where does L9H7 attend instead?

**The contribution.** This connects OV analysis (what is written) to QK analysis (what is read). We can say: "R3 fails on ambiguous names because L9H7 doesn't strongly attend to the name token, causing the value computation to miss the gender signal." This is the full mechanistic story — not just "it fails" but "here's exactly why."

**Going further: the Attention Sufficiency Test.** For the error cases, manually replace L9H7's attention pattern with a pattern that attends fully to the name token. If R3's activation flips to the correct sign, we've proven that the failure is in attention routing, not in the value computation.

| Metric | Rating |
|--------|--------|
| Novelty | 6/10 — attention pattern analysis exists, but conditioning it on receptor failure modes is new |
| Effort | 2 hours — need one forward pass with attention caching |
| Impact on paper | 7/10 — completes the mechanistic story |
| Feasibility | Easy on T4 |
| When to do it | After core additions are done |

---

## IDEA 7: Cross-Task Receptor Transfer (IOI)

**What it is.** Test whether the GP receptor directions R1, R2, R3 carry gender information in a completely different task.

**Why it comes from our results.** Exp 0c showed R1's direction is embedded in the token embedding geometry — it's not task-specific, it's a property of how GPT-2 represents names. If true, R1 should transfer to IOI (Indirect Object Identification: "When John and Mary went to the store, John gave a drink to ____"). R3 is computed by L9H7 specifically for the GP circuit, so it probably won't transfer. This asymmetry (R1 transfers, R3 doesn't) would be a strong validation of the inherited/computed distinction.

**The experiment.**

1. Construct 500 IOI examples programmatically.
2. Run forward passes (no retraining).
3. Project residual stream at decision position onto R1, R2, R3.
4. Measure: does PRM = g₁ − g₃ predict the correct IOI answer?

**Possible outcomes:**
- R1 transfers, R3 doesn't → validates inherited/computed distinction.
- Both transfer → receptors are intrinsic model features (very strong finding).
- Neither transfers → GP receptors are task-specific (still informative).

**What this adds.** Addresses the #1 reviewer objection: "is this just about one toy task?" Even partial transfer is publishable.

| Metric | Rating |
|--------|--------|
| Novelty | 6/10 — transfer testing is standard, but the inherited-transfers/computed-doesn't prediction is novel |
| Effort | Half a day — need to construct IOI examples, run forward passes |
| Impact on paper | 9/10 — generalization is essential for a main conference paper |
| Feasibility | Easy on T4 (no training, just inference) |
| When to do it | Before paper submission. This is the second highest-value addition after CFR. |

---

## IDEA 8: Margin-Ordered Component Ablation (MOCA)

**What it is.** Rank all model components (attention heads + MLPs) by their contribution to the receptor margin. Ablate them one by one in order of contribution. Measure how quickly accuracy degrades.

**Why it comes from our results.** We know R² = 0.980 — receptors explain almost everything. But we don't know how many components PRODUCE this signal. Is the receptor signal written by 3 heads (sparse circuit) or smeared across 50+ components (distributed)? The margin decomposition from Exp 0b identified the top contributors, but we didn't ablate to test causal necessity.

**The method.**

From Exp 0b data, we have per-component margin contributions μ_n for every component n.

1. Sort components by |μ_n| descending.
2. Ablate the top-1 component (zero its output), measure accuracy.
3. Ablate top-2, measure accuracy.
4. Continue until accuracy hits chance (50%).

Plot: number of components ablated vs. accuracy.

**What we expect.** If the circuit is sparse, accuracy should drop to chance after ablating ~5-10 components. If distributed, it degrades gradually. Based on the Beyond Components finding that a few heads dominate, expect sparse.

**The key number: the "circuit size."** How many components do you need to ablate to bring accuracy below 60%? This is a single number that characterizes the circuit's sparsity.

**Comparison to existing methods.** ACDC and other circuit discovery methods find minimal sufficient circuits. MOCA finds the minimal necessary components ordered by receptor margin. These might differ — a component can be sufficient without being the highest-margin contributor. The comparison itself is interesting.

| Metric | Rating |
|--------|--------|
| Novelty | 5/10 — ablation ordering exists, but doing it via receptor margin is new |
| Effort | 3-4 hours — need forward passes with component ablation |
| Impact on paper | 7/10 — gives a concrete "circuit size" number |
| Feasibility | Medium — need to hook into each component |
| When to do it | After CFR and IOI transfer |

---

## IDEA 9: Receptor Fingerprint Clustering

**What it is.** Every component (head or MLP) produces a "fingerprint" — its projection onto each receptor direction. Cluster all components by their fingerprints. See if clusters correspond to known functional roles.

**Why it comes from our results.** Exp 0b gave us per-component contributions to each receptor. We know R1 and R3 have opposite polarities. Components that contribute positively to R1 and negatively to R3 are "promoters." Components that do the opposite are "inhibitors." Components that contribute to both equally are "neutral."

**The method.**

For each component n, compute fingerprint f_n = [C[1,n], C[2,n], C[3,n]].

Since R2 is a ghost, project to 2D: f_n → (C[1,n], C[3,n]).

Cluster in this 2D space. Possible clusters:
- Promoters: high C[1,n], low C[3,n]
- Inhibitors: low C[1,n], high C[3,n]
- Neutral: near origin
- Conflicted: high on both (pushing in both directions)

**What this adds.** An automatic discovery of component roles without any prior knowledge of the circuit. Compare the discovered clusters to the known GP circuit (name movers, S-inhibition heads) from Wang et al. If they match, the fingerprint clustering is validated as a circuit discovery tool.

| Metric | Rating |
|--------|--------|
| Novelty | 6/10 — component clustering exists, but using receptor fingerprints as the basis is new |
| Effort | 1 hour — pure computation on existing Exp 0b data |
| Impact on paper | 6/10 — nice visualization, supports the framework |
| Feasibility | Trivial |
| When to do it | During paper writing |

---

## IDEA 10: Temporal Causal Validation via Activation Patching

**What it is.** Exp 0c showed commitment ordering (R1 at embedding, R3 at layer 9) via correlational metrics (sign agreement). Validate this causally by patching activations from a counterfactual example.

**The method.**

Take matched pairs: "John went... [he]" and "Mary went... [she]."

At each layer l, patch the residual from the Mary run into the John run. Measure:
- At what layer does the patched model flip from "he" to "she"?
- At what layer does each receptor flip sign?

**Prediction.** R1 should flip at layer 0 (or earlier — since it inherits from the embedding, patching the embedding should flip it). R3 should flip at layer 9 (when L9H7 reads the patched residual). If confirmed, this causally validates the commitment ordering.

**The additional insight: flip ordering.** If R1 flips before R3, the model briefly enters a state where both receptors agree (both say "she") — clean convergence. If R3 flips first, the model briefly enters a state where receptors disagree — a "conflict" state. The dynamics of this transition are only visible with receptor-level patching, not with standard logit lens.

| Metric | Rating |
|--------|--------|
| Novelty | 5/10 — activation patching is standard, but receptor-level analysis of the patching dynamics is new |
| Effort | 2-3 hours |
| Impact on paper | 6/10 — validates Exp 0c causally, nice but not essential |
| Feasibility | Easy |
| When to do it | If time permits |

---

## IDEA 11: The Receptor-Orthogonalized Basis and Decision Efficiency

**What it is.** R1, R2, R3 are non-orthogonal. Orthogonalize them via QR decomposition to get an orthonormal basis for the receptor subspace. Decompose each component's output into a "decision-aligned" component and a "decision-orthogonal" component.

**The method.**

Compute polarity-weighted decision direction: d = v₁r₁ + v₃r₃ = r₁ − r₃ (ignoring ghost R2).

For each component n with output o_n at the decision position:
- Decision-aligned: (o_n · d̂) · d̂
- Decision-orthogonal: o_n − (o_n · d̂) · d̂

Decision Efficiency:

$$\text{DE}_n = \frac{|o_n \cdot \hat{d}|}{\|o_n\|}$$

Components with DE ≈ 1 are perfectly aimed at the gender decision. Components with DE ≈ 0 but large ||o_n|| are writing heavily into the residual stream but NOT for the gender task. What ARE they doing?

**The interesting finding.** Identify the top-5 components with lowest DE but highest ||o_n||. These are components that write a lot but not for gender. They might be serving other tasks (number agreement, semantic role, etc.) — a concrete instance of polysemanticity at the component level.

| Metric | Rating |
|--------|--------|
| Novelty | 6/10 — decision efficiency metric is new, but the orthogonalization idea is standard |
| Effort | 1 hour — computation on existing data |
| Impact on paper | 5/10 — nice metric, interesting finding if the polysemantic components are interpretable |
| Feasibility | Trivial |
| When to do it | During paper writing |

---

## IDEA 12: Receptor-Grounded QK Skeleton (From the QK Ideas Chat)

**What it is.** The previous chat on QK ideas proposed using logit receptors as an anchor to interpret the QK circuit. Specifically: for each attention head, compute how its QK circuit controls receptor activation. This connects the "what is written" (OV/receptors) to "why it's written" (QK/routing).

**Why it's relevant now.** We have the full OV story: R1 inherits, R3 computes, R2 is ghost. The missing half is: what determines WHEN R3 activates strongly? The answer is L9H7's attention pattern, which is controlled by its QK circuit. If we can decompose L9H7's QK into a "receptor-relevant" component and a "receptor-irrelevant" component, we complete the mechanistic picture.

**The method (from pullback maps idea).**

For L9H7, compute:

$$\frac{\partial g_3}{\partial S_j} = \alpha_j(\beta_j - g_3)$$

where S_j is the attention score to position j, α_j is the attention weight, and β_j = u₃ · [1; x_j^{LN}] is the value-content alignment at position j.

Aggregate this gradient over the dataset (covariance matrix of ∂g₃/∂S). SVD of this covariance gives the "routing modes" — directions in score space that most control R3's activation.

**What this adds.** If the top 1-2 routing modes explain >80% of R3's activation variance, then the QK circuit's effect on the receptor is low-rank and interpretable. We'd have: "L9H7's gender computation is controlled by a 1-2 dimensional routing mode in its attention scores. When the attention is directed at a strongly gendered name, this mode activates, causing R3 to fire."

**Important caveat.** This is essentially the original "receptor-conditioned QK decomposition" idea that started the whole chat. The difference is that now we have the full OV characterization to anchor it. We know exactly what to condition on (R3's activation) and what the downstream effect is (modification of R1 via 41.2% computational coupling). The QK analysis completes the circuit.

| Metric | Rating |
|--------|--------|
| Novelty | 8/10 — QK conditioned on a downstream receptor scalar is genuinely new |
| Effort | 3-4 hours — need attention scores + gradient computation |
| Impact on paper | 8/10 — completes the OV-to-QK story, which is the "big gap" in mech interp |
| Feasibility | Medium — gradient computation on T4 is fine |
| When to do it | After CFR and IOI. This could be its own paper section. |

---

## IDEA 13: Receptor Stability Spectrum Across Sentence Structures

**What it is.** Exp 5 showed AUC ≥ 0.92 across 5 structures, but dropped to 0.75 on ambiguous names. Go deeper: for each sentence structure, compute the FULL set of metrics (not just AUC) — commitment layers, cross-talk, Shapley, phase diagram. Does the circuit's internal structure change across syntactic contexts, or just its reliability?

**Why it comes from our results.** We have the infrastructure to run all experiments. The question is whether the findings (R1 inherits, R3 computes, R2 is ghost, 41.2% asymmetric coupling) are invariants of the circuit or context-dependent properties.

**The method.**

For each of the 5 sentence structures from Exp 5:
1. Compute per-layer receptor activations (same as Exp 0c).
2. Check: does R1 still commit at layer −1? Does R3 still commit at layer 9?
3. Compute simplified cross-talk: scale R3 by 5×, measure R1 shift. Is the computational fraction still ~41%?

**What we expect.** Commitment layers should be stable (they're properties of the circuit, not the input). Cross-talk magnitudes might vary (different syntactic structures might route information differently through layers 9-11). The pattern of variation is the finding: which aspects of the circuit are universal vs. context-dependent?

**What this adds.** Moves from "we tested on 5 structures and AUC was high" to "we tested on 5 structures and the circuit's INTERNAL STRUCTURE is invariant." That's a much stronger generalization claim.

| Metric | Rating |
|--------|--------|
| Novelty | 6/10 — testing structural invariance is new, but the experiments are repeats on different data |
| Effort | Half a day — rerun a subset of experiments per structure |
| Impact on paper | 7/10 — strengthens generalization claims significantly |
| Feasibility | Straightforward |
| When to do it | After main additions |

---

## IDEA 14: Embedding Geometry Audit — Why Is R1 Free?

**What it is.** R1 inherits gender from the token embedding. But why? What property of GPT-2's embedding space makes R1's direction (the right singular vector of L10H9's OV) align with a gender axis that's already in the embeddings?

**The experiment.**

1. Take all name tokens in the vocabulary. Extract their embeddings.
2. Project onto R1's direction. Split by known gender of the name.
3. Measure separation: is R1's direction a clean gender separator in embedding space?
4. Compare to a random direction and to the first principal component of name embeddings.

**Going further:** Compute the principal components of the gender difference vector (mean male embedding − mean female embedding). Check if R1's direction is close to this difference vector. If cos(R1, gender_diff) > 0.9, then R1 is literally the "gender direction" in embedding space, and L10H9 learned to write along this pre-existing axis.

**What this adds.** Explains WHY R1 is inherited — it's not a coincidence, it's because the OV training of L10H9 aligned with a structural feature of the embedding. This connects circuit-level findings to representation-level findings.

| Metric | Rating |
|--------|--------|
| Novelty | 6/10 — embedding geometry analysis exists, but linking it to a specific receptor's inheritance is new |
| Effort | 1 hour |
| Impact on paper | 6/10 — explains a finding, nice but not essential |
| Feasibility | Trivial |
| When to do it | During paper writing |

---

## PRIORITY RANKING (Honest Assessment)

| Rank | Idea | Novelty | Effort | Paper Impact | Transforms Paper From "Appendix" To... |
|------|------|---------|--------|-------------|---------------------------------------|
| 1 | CFR Ghost Detection (#1) | 8/10 | 2-3 hrs | 9/10 | "A method paper" — gives a reusable tool |
| 2 | IOI Transfer (#7) | 6/10 | 0.5 day | 9/10 | "A generalizable finding" — answers reviewer #1 |
| 3 | PRM Error Prediction (#3) | 7/10 | 30 min | 8/10 | "A predictive framework" — not just post-hoc |
| 4 | Inhibition Bottleneck (#5) | 8/10 | 2-3 hrs | 8/10 | "A design principle" — general insight about circuits |
| 5 | QK Skeleton (#12) | 8/10 | 3-4 hrs | 8/10 | "A complete circuit story" — OV + QK |
| 6 | RIM Formalization (#2) | 7/10 | 1 hr | 7/10 | "A theoretical contribution" — defines new objects |
| 7 | Superposition Tax (#4) | 7/10 | 30 min | 7/10 | "Connected to big questions" — superposition link |
| 8 | Structural Invariance (#13) | 6/10 | 0.5 day | 7/10 | "A robust finding" — circuit is universal |
| 9 | MOCA (#8) | 5/10 | 3-4 hrs | 7/10 | "A circuit with a number" — sparsity claim |
| 10 | Attention Pattern Analysis (#6) | 6/10 | 2 hrs | 7/10 | "A complete mechanistic story" |
| 11 | Fingerprint Clustering (#9) | 6/10 | 1 hr | 6/10 | "An automatic discovery tool" |
| 12 | Embedding Geometry (#14) | 6/10 | 1 hr | 6/10 | "An explanatory finding" |
| 13 | Activation Patching (#10) | 5/10 | 2-3 hrs | 6/10 | "Causally validated" |
| 14 | Decision Efficiency (#11) | 6/10 | 1 hr | 5/10 | "A nice metric" |

---

## RECOMMENDED PLAN: What Transforms This Into a Real Paper

**Minimum viable paper (do these 4 things):**

1. **CFR on all ~38 mask-identified receptors** — shows that only 2-4 are causally real. This is your methodological contribution.
2. **PRM error prediction** — shows receptors are predictive, not just explanatory. 30 minutes of work.
3. **RIM formalization** — gives the paper a defined object. Pure writing.
4. **IOI transfer** — shows the inherited/computed distinction generalizes. Half a day.

With these 4 additions, the paper story becomes:

"Beyond Components identifies logit receptors by correlation. We show that (1) most correlation-identified receptors are causal ghosts, detectable by our CFR method; (2) the real receptors form a promotion/inhibition circuit with an inherited channel (embedding) and a computed channel (L9H7); (3) the interaction structure is captured by the Receptor Interference Matrix, which reveals asymmetric computational coupling; (4) the receptor margin predicts model errors before seeing the output; and (5) the inherited/computed distinction generalizes to IOI. Together, these establish Receptor Interference Spectroscopy as a general framework for analyzing how multiple logit receptors compete to determine transformer outputs."

That's a 7-8/10 publishability paper. Add the inhibition bottleneck (#5) and QK skeleton (#12) and it's 8-9/10.

---

## WHAT NOT TO DO

**Don't try to do all 14 ideas.** Pick the top 4-5, execute them cleanly, and write a tight paper. A paper with 5 strong experiments is better than one with 14 mediocre ones.

**Don't pursue Decision Efficiency (#11) or Activation Patching (#10) unless you have extra time.** They're nice-to-have metrics but don't change the paper's narrative.

**Don't start with QK analysis (#12).** It's high-novelty but also high-effort and could become a rabbit hole. Do it after the minimum viable paper is locked in.

**Don't worry about running on bigger models.** GPT-2 small is fine for a first paper. Reviewers will ask "does this scale?" — answer in the discussion: "the framework is model-agnostic; we leave scaling to future work." This is standard.
