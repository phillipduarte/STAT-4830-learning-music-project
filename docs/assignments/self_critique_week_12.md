# Week 12 Self-Critique

## OBSERVE

Two MERT finetuning runs were completed, unfreezing the top 2 and top 4 transformer layers respectively in Phase 2, each trained for 30 epochs with a warmed-up classification head. The 2-layer run peaked at 68.9% top-1 / 86.0% top-5 (epoch 29); the 4-layer run peaked at 70.2% top-1 / 88.1% top-5 (epoch 30). Both represent roughly 2× improvements over the frozen MERT baseline of 34.4% top-1. The 4-layer run's test loss was still visibly declining at epoch 30, whereas the 2-layer run had largely flattened by that point.

---

## ORIENT

**Strengths**

- The two-phase training strategy (frozen head warm-up followed by selective unfreezing) worked as intended — neither run exhibited catastrophic forgetting, and both reached significantly higher accuracy than any frozen-embedding approach regardless of classifier architecture. This settles the question raised in earlier self-critiques: the embedding geometry, not the classification head, was the binding constraint.
- The caching optimization (precomputing outputs of the frozen lower layers) made Phase 2 tractable without requiring full MERT forward passes each step, enabling a controlled comparison across unfreezing depths.

**Areas for Improvement**

- Both runs were stopped at 30 epochs without reaching a clear convergence floor. The 4-layer run's loss curve provides no justification for stopping there — it was still improving. Reporting a best checkpoint from a run that hasn't converged overstates confidence in the result; the true ceiling for 4 unfrozen layers is unknown.
- Increasing from 2 to 4 unfrozen layers yielded only ~1.3pp improvement in top-1 accuracy (68.9% → 70.2%). This marginal gain does not clearly justify the added compute and risk of overfitting deeper into the transformer. The diminishing returns suggest either that 2 layers already captures most of the available task-relevant signal in the upper transformer, or that 30 epochs is too short to fairly evaluate the 4-layer run's ceiling.
- Both finetuning runs were conducted exclusively on the unperturbed dataset. The MLP search over the perturbed dataset (corrected for data leakage) achieved 44.2% top-1, and it is entirely unclear how finetuning interacts with pitch and tempo augmentation. Audio-space augmentation was identified in prior weeks as the correct approach for key and tempo invariance, yet finetuning has been evaluated without it.

**Critical Risks/Assumptions**

The current finetuning results are on clean, unaugmented audio. If the model is deployed on snippets that differ in tempo or key from the training renders — which is likely in any real-use scenario — performance may degrade substantially. Finetuning on clean audio may also entrench MERT's representations around pitch-specific features, making the model *less* robust to transposition than the frozen baseline was.

---

## DECIDE

**Concrete Next Actions**

- **Train to convergence.** Re-run the 4-layer experiment for at least 60–100 epochs and use early stopping on test top-1 with sufficient patience (≥10 epochs) to establish a genuine floor. Until this is done, the 70.2% figure should be treated as a lower bound, not a final result.
- **Finetune on perturbed data.** Run Phase 2 using the leakage-corrected perturbed dataset as the training set. This is the most important next experiment: it tests whether finetuning and audio augmentation are complementary, and whether the model can learn pitch- and tempo-invariant representations when the transformer weights are free to adapt.
- **Fix unfreezing depth at 2 layers pending further evidence.** Given the marginal gain from going to 4 layers, subsequent experiments should default to 2 unfrozen layers to reduce compute cost and overfitting risk, unless a longer training run shows the 4-layer model meaningfully diverging after convergence.

---

## ACT

**Resource Needs**

Longer training runs require additional GPU time on Colab or Prime Intellect — roughly 2–3× the current per-run cost for the convergence experiment. The perturbed finetuning run requires the corrected perturbed embeddings to be regenerated from raw audio at each Phase 2 step (or cached at the layer-boundary as in the current setup), which will increase memory pressure given the larger dataset size (~77k snippets vs. ~2.5k). It may be necessary to reduce the cache to CPU RAM selectively or to re-examine the snippet count per piece to keep the experiment tractable.
