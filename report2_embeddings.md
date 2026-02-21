# Week 6 Report: Music Piece Classification via Logistic Regression on MERT Embeddings

## Problem Statement

**What are we optimizing?** We learn a weight matrix $W \in \mathbb{R}^{d \times C}$ (and bias $b \in \mathbb{R}^C$) that maps pre-computed audio embeddings of Bach chorale snippets to piece identity predictions. Concretely, given an embedding $x \in \mathbb{R}^d$ of a snippet, the model computes logits $z = Wx + b \in \mathbb{R}^C$ and predicts the piece via $\hat{y} = \arg\max_c z_c$. We minimize the cross-entropy loss over the training set:

$$\mathcal{L}(W, b) = -\frac{1}{n} \sum_{i=1}^{n} \log \frac{e^{z_{y_i}}}{\sum_{c=1}^{C} e^{z_c}}$$

**Why does this problem matter?** Piece identification from audio snippets is a prerequisite for music recommendation, plagiarism detection, and archival search. Rather than hand-designing features, we use MERT — a transformer pre-trained on general audio — to produce rich 768-dimensional representations, then ask: how much piece-identity information do these embeddings already carry, and can a simple linear classifier extract it?

**How will we measure success?** Top-1 and top-5 accuracy on a held-out test set, against a random baseline of 0.23% (1/430 pieces). Per-class accuracy breakdown reveals whether the model generalizes across pieces or concentrates on a few easy ones.

**Constraints:** All 430 pieces are from the Music21 Bach chorale corpus. The split is `by_snippet`: snippets from every piece appear in both train and test, so the classifier sees each piece's embedding distribution during training. Compute runs on a single GPU (CUDA). MERT embeddings are extracted once and cached to disk — no fine-tuning of the transformer.

**What could go wrong?** The embedding space was not trained for piece identity, so a linear classifier may be too weak to carve out 430 decision boundaries from it. L2 normalization of embeddings, a common preprocessing step, might discard magnitude information that MERT encodes.

---

## Technical Approach

**Data pipeline.** The pipeline runs in four modular stages: `data_prep.py` loads Bach chorales from Music21 and segments them into snippets; `render.py` converts each snippet to audio via FluidSynth; `embed.py` passes audio through MERT-v1-95M and caches mean-pooled hidden states (768-dim) to disk; and the classification notebook loads the cached embeddings and trains the linear classifier.

**Mathematical formulation.** The model is a single `nn.Linear(768, 430)` layer — multi-class logistic regression. With data matrix $X \in \mathbb{R}^{n \times 768}$ and integer labels $y \in \{0, \ldots, 429\}^n$, the forward pass computes logits $Z = XW^\top + \mathbf{1}b^\top \in \mathbb{R}^{n \times 430}$, and the loss is weighted cross-entropy. Since regularization experiments found no benefit from weight decay (see Results), the final objective is unregularized:

$$\min_{W, b} \; \mathcal{L}(W, b)$$

This is a smooth, unconstrained problem in 330,670 parameters ($768 \times 430 + 430$).

**Optimizer.** Adam (lr = 1e-3, weight_decay = 0, batch size 256) is used over 1,000 epochs. Adam is chosen over SGD because the loss landscape has poor conditioning — 430 classes each pulling gradients in different directions across a 768-dimensional input space — and Adam's per-parameter adaptive learning rates compensate for this. A comparison run with SGD + L2-normalized embeddings confirmed the choice: that configuration reached only 4.4% top-1 while Adam on raw embeddings reaches 34.4%. Regularization strength $\lambda$ was swept over $\{10^{-4}, 10^{-6}, 0\}$; test accuracy changed by less than 0.6% across all values, indicating that performance is determined by model capacity rather than overfitting.

**Class weighting.** Inverse-frequency weights $v_c = \frac{1/n_c}{\sum_j 1/n_j} \cdot C$ are computed from training snippet counts, ranging 0.18–2.39. Without this, the cross-entropy loss is dominated by high-snippet pieces and rare chorales are effectively never predicted.

**Validation.** Top-1 and top-5 accuracy are computed each epoch on both train and test sets. A per-class accuracy breakdown and confusion matrix identify which pieces are systematically missed or confused.

---

## Initial Results

The pipeline runs end-to-end successfully. From 430 Bach chorales, it produces 2,487 training snippets and 521 test snippets (768-dimensional MERT embeddings each). The linear classifier has 330,670 trainable parameters.

After 1,000 epochs with Adam, the model achieves **34.4% top-1 and 56.4% top-5 test accuracy** against a random baseline of 0.23% — a 150× improvement over chance. This confirms that MERT embeddings contain genuine piece-identity signal that a linear classifier can access.

The results reveal a clear capacity ceiling. Training accuracy reaches 99.8% while test accuracy plateaus at 34.4% with no upward trend across 1,000 epochs. Sweeping weight decay across three orders of magnitude ($10^{-4}$ to $0$) produced no meaningful change in test accuracy, ruling out overfitting as the primary bottleneck. The diagnosis is instead a model capacity problem: a linear decision boundary in 768-dimensional space is insufficient to separate 430 classes whose embeddings are not linearly arranged.

Per-class analysis confirms this. Median per-class accuracy is 0%, with only 8 of 430 pieces identified at 100% and 410 pieces at 0%. The aggregate 34.4% is carried almost entirely by those 8 easy pieces — likely ones with distinctive harmonic content that produces separable embeddings. The confusion matrix shows bwv248.64-6 and bwv79.3 acting as prediction attractors, absorbing misclassifications from dozens of other chorales.

Two key implementation lessons emerged from debugging. First, applying L2 normalization to MERT embeddings before the linear layer dropped top-1 accuracy from 34.4% to ~4.4% — MERT embedding magnitudes carry piece-identity information and normalizing to unit length discards it.

---

## Next Steps

**Nonlinear classifier (MLP).** The regularization sweep established that the performance ceiling is due to model capacity, not overfitting. A linear decision boundary simply cannot separate 430 classes in a space that was not organized around piece identity. A single hidden layer (768 → 512 → 430 with ReLU activation and dropout) introduces the nonlinearity needed to carve out more complex decision boundaries. Unlike the linear model — where the loss is convex and Adam finds the global optimum reliably — the MLP loss landscape is non-convex, with local minima and saddle points that make optimizer choice and initialization genuinely consequential. This is a more interesting optimization problem for this course context. Dropout also provides a principled regularization mechanism to control the wider train/test gap that a higher-capacity model will initially produce.

**Harder evaluation split.** The current `by_snippet` split lets the classifier see every piece's embedding distribution during training, making the test problem easier than the real-world use case. Switching to `by_piece` — holding out entire pieces — tests whether learned weights generalize to unseen chorales and is the setup required for eventual metric learning experiments.

**Understand the embedding space.** Before scaling model capacity further, computing pairwise distances between same-piece vs. different-piece snippets in the raw MERT space would clarify whether the bottleneck is classifier capacity or embedding geometry. If same-piece snippets do not cluster geometrically, no classifier trained on frozen embeddings will solve the problem — and the right response is fine-tuning MERT rather than deepening the classification head.

**What we've learned.** The MERT embedding pipeline works reliably and produces representations far more informative than hand-crafted features from the prior approach. The central challenge is now precisely diagnosed: the embedding space is not linearly organized around piece identity for most chorales, so the linear classifier saturates quickly on the few geometrically separable pieces and fails on the rest. This motivates the MLP as the immediate next step, and metric learning — which directly optimizes the embedding geometry rather than just the classification head — as the longer-term direction.
