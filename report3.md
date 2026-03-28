# Week 8 Report
In this report, beyond the initial linear/logistic regression models on sheet music-based features, we use MERT-generated embeddings in logistic regression for multi-class classification where we classify a piece of music specifically rather than only if 2 pages of music come from the same piece. Then, we also use these embeddings in a 1 hidden-layer MLP.

### Problem Statement
Given a large collection of sheet music, the goal is to classify a snippet of  music as coming from a piece, such as whether it matches the piece that another snippet is from, or to directly classify it as a specific piece. To allow this, we will extract features (harmonic, rhythmic, etc.) from the sheet music, use embeddings of audio based on pre-trained model, and later learn embeddings.

This problem as interesting as piece identification from audio snippets is a prerequisite for music recommendation, plagiarism detection, and archival search. We initially attempt to look at understanding of which features of a musical work are most characterizing of it and interactions between those features. Later, rather than hand-designing features, we use MERT — a transformer pre-trained on general audio — to produce rich 768-dimensional representations, then ask: how much piece-identity information do these embeddings already carry, and can a simple linear classifier extract it?

Success will be measured by classification performance on a held-out test set of page pairs. Specifically, we evaluate accuracy relative to a 0.5 baseline for random guessing, as well as precision, recall, and F1-score. We also track the training loss to ensure optimization convergence and compare weights across different methods. For the models based on embeddings where we are predicting the exact piece rather than classification of match vs. non-match, we use top-1 and top-5 accuracy on a held-out test set, against a random baseline of 0.23% (1/430 pieces). Per-class accuracy breakdown reveals whether the model generalizes across pieces or concentrates on a few easy ones.

Some constraints and things that may go wrong include that it is also unclear what linear/nonlinear models will be able to capture similarity well as well as understanding interactions between features. For the embedding appraoch, MERT embeddings are extracted once and cached to disk — no fine-tuning of the transformer. The embedding space was not trained for piece identity, so a linear classifier may be too weak to carve out 430 decision boundaries from it. L2 normalization of embeddings, a common preprocessing step, might discard magnitude information that MERT encodes.

### Technical Approach
We have attempted several problem formulations and types of models in working towards the goal of effective classification. All methods are summarized and the last is our most recent
1. **<u>Logistic Regression on Musical Features</u>**

*Processing:* Extract sheet music-based features using music21 and define similarity metrics for each of these. The 5 features chosen for the initial experiments were [key, time signature, average pitch, pitch range, note density] with heuristic-based similarity functions for each. Specify pages, requirements on number of measures, etc. and extract these features for each page.

*Mathematical Formulation*: For the initial experiments we used logistic regression after initially using linear regression (which was not ideal since this is a binary classification problem rather than prediction of some continuous value). In logistic regression, we predict match/non-match based on the sigmoid of the logits and whether it is $\geq 0.5$, or equivalently whether the logits are $\geq 0$. See further details in report1.md and report2.md.

**Features:** The similarity vectors previously contained values from 0-1, now they are re-scaled to be between -1 to 1 where 1 would mean they are more similar and -1 if they are less similar. Define $s = [s1, s2, s3, s4, s5]$

**Labels:** 1 for match (two pages from the same piece), 0 for non-match (two pages from different pieces, with also a further distinction of easy negative and hard negative referring to different pieces of different composers and the same composer, respectively)

**Loss (as optimization problem):**
Negative log-likelihood loss (Binary Cross-Entropy Loss). This is evidently also a convex loss. Taking out the constraint for the weights to have to sum to 1 as that considerably limited the performance.

![alt text](report_math2.png)

that is optimized for classification based on the above where $\mathbf{1}$ is the indicator function. The behavior of the sigmoid is that it converts values from $-\infty$ to $\infty$ into probabilities and is centered around 0 so this works.

*Optimization Methods:* Use various algorithms to learn weights for a weighted similarity function to distinguish matching vs. non-matching pages.
1. Projected Gradient Descent: This projected method was chosen because it will allow us to remain in the feasible set. The weight vector would be a learnable parameter which would be updated by going in the direction of the negative gradient, followed by projection to enforce the constraint of all weights being non-negative. This is implemented using PyTorch with a loop going through some fixed number of iterations, from which we will exit if the stopping condition is met (based on some tolerance for the norm of the gradient). torch.nn.functional.binary_cross_entropy_with_logits is chosen for loss calculation for better numerical stability.
2. Sequential Least Squares Programming (SLSQP): The function scipy.optimize.minimize was chosen because it can be used for constrained, non-convex optimization problems. This should yield the same solution as the convex solver.
3. Convex optimization (via CVXPY library): This method was chosen as the optimization problem is a quadratic program (previously misstated in report 1...) with a quadratic objective and linear constraint, and thus such a solver can give us the optimized weights.

*Validation:* We validated performance on a held-out testing set of matching and non-matching page pairs. For training, we monitor training loss to see how the optimization/convergence is happening.

**<u>2. Logistic Regression on Embeddings</u>**

*Processing:* The pipeline runs in four modular stages: `data_prep.py` loads Bach chorales from Music21 and segments them into snippets; `render.py` converts each snippet to audio via FluidSynth; `embed.py` passes audio through MERT-v1-95M and caches mean-pooled hidden states (768-dim) to disk; and the classification notebook loads the cached embeddings and trains the linear classifier.

*Mathematical Formulation:* Here, we learn a weight matrix $W \in \mathbb{R}^{d \times C}$ (and bias $b \in \mathbb{R}^C$) that maps pre-computed audio embeddings of Bach chorale snippets to piece identity predictions. Concretely, given an embedding $x \in \mathbb{R}^d$ of a snippet, the model computes logits $z = Wx + b \in \mathbb{R}^C$ and predicts the piece via $\hat{y} = \arg\max_c z_c$. We minimize the cross-entropy loss over the training set:

$$\mathcal{L}(W, b) = -\frac{1}{n} \sum_{i=1}^{n} \log \frac{e^{z_{y_i}}}{\sum_{c=1}^{C} e^{z_c}}$$

The model is a single `nn.Linear(768, 430)` layer — multi-class logistic regression. With data matrix $X \in \mathbb{R}^{n \times 768}$ and integer labels $y \in \{0, \ldots, 429\}^n$, the forward pass computes logits $Z = XW^\top + \mathbf{1}b^\top \in \mathbb{R}^{n \times 430}$, and the loss is weighted cross-entropy. Since regularization experiments found no benefit from weight decay (see Results), the final objective is unregularized:

$$\min_{W, b} \; \mathcal{L}(W, b)$$

This is a smooth, unconstrained problem in 330,670 parameters ($768 \times 430 + 430$).

*Optimization Methods:* Adam (lr = 1e-3, weight_decay = 0, batch size 256) is used over 1,000 epochs. Adam is chosen over SGD because the loss landscape has poor conditioning — 430 classes each pulling gradients in different directions across a 768-dimensional input space — and Adam's per-parameter adaptive learning rates compensate for this. A comparison run with SGD + L2-normalized embeddings confirmed the choice: that configuration reached only 4.4% top-1 while Adam on raw embeddings reaches 34.4%. Regularization strength $\lambda$ was swept over $\{10^{-4}, 10^{-6}, 0\}$; test accuracy changed by less than 0.6% across all values, indicating that performance is determined by model capacity rather than overfitting.

*Validation:* Top-1 and top-5 accuracy are computed each epoch on both train and test sets. A per-class accuracy breakdown and confusion matrix identify which pieces are systematically missed or confused.


**<u>3. MLP on Embeddings</u>**

*Processing:* The same pipeline as described in the above 2nd approach is used for generation of embeddings.

*Mathematical Formulation:* We use a one-hidden-layer MLP as an `nn.Module`:
$$x \in \mathbb{R}^{768} \xrightarrow{\text{Linear}} h \in \mathbb{R}^{H} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Dropout}} \xrightarrow{\text{Linear}} z \in \mathbb{R}^{430}$$
$h$ is a hidden representation with $H = 512$, $z$ are the logits. The model has approximately 395,000 trainable parameters. Kaiming uniform initialization is used for both linaer layers which accounts for the ReLU non-linearity — it scales weights by $\sqrt{2/\text{fanin}}$ so that the variance of activations is preserved through each layer at initialization. Dropout is placed after the ReLU activation, meaning that the activation decides which neurons fire and this step  stochastically zeros some of those active units, acting as regularization.

*Optimization Methods:* Adam (lr = 1e-2, weight_decay = 1e-3) is used with a cosine annealing learning rate schedule over 1000 epochs with batch size 256. The loss is cross-entropy loss with involves inverse-frequency class weighting and label smoothing to address the class imbalance. As the MLP loss landscape is non-convex, the cosine annealing schedule which decays at a predetermined rate is helpful.

*Validation:* Same validation as with the above 2nd approach.

### Results
**<u>1. Logistic Regression on Musical Features</u>**

For our initial approach using feature vectors that are 5 selected musical features, the results are based on a test set with works by Bach, Mozart, and Beethoven such that the training data contained 35 examples (14 matching, 21 non-matching) and the test data contained 15 examples (9 matching, 6 non-matching). Evidently there was some class imbalance and that affects results, some weighting of classes was also tested, but to no avail as the test set itself was not very large.

Results as follows:
| | Learned weights | Loss | Train Accuracy | Test Accuracy | Iters | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Projected Gradient Descent | [0.0582, 0, 0, 0.2407, 0] | 0.6902 | 0.5429 | 0.5333 | 1000* | 0.2327 
| Sequential Least Squares Programming | [0.0582, 0, 0, 0.2457, 0] | 0.6902 | 0.5429 | 0.5333 | 7 | 0.0489
| Convex Optimization | [0.0575, 0, 0, 0.2446, 0] | 0.6902 | 0.5429 | 0.5333 | N/A | 0.076

\* learning rate = 0.01, max_iterations = 1000, tolerance = 1e-4

As an analysis of these results, it is clear that it reaches a similar conclusion in terms of learned weights and feature importance as with the linear regression. However, whereas the linear regression identified the most importance features as (1) key and (2) average pitch, this logistic regression identified it as (1) pitch range and (2) key. So while key is common between these as well as factors relating to it in general, the weight vector learned to minimize the binary cross entropy loss compared to the squared error loss is slightly different (see report2.md for details).

**<u>2. Logistic Regression on Embeddings</u>**

The pipeline runs end-to-end successfully. From 430 Bach chorales, it produces 2,487 training snippets and 521 test snippets (768-dimensional MERT embeddings each). The linear classifier has 330,670 trainable parameters.

After 1,000 epochs with Adam, the model achieves **34.4% top-1 and 56.4% top-5 test accuracy** against a random baseline of 0.23% — a 150× improvement over chance. This confirms that MERT embeddings contain genuine piece-identity signal that a linear classifier can access.

The results reveal a clear capacity ceiling. Training accuracy reaches 99.8% while test accuracy plateaus at 34.4% with no upward trend across 1,000 epochs. Sweeping weight decay across three orders of magnitude ($10^{-4}$ to $0$) produced no meaningful change in test accuracy, ruling out overfitting as the primary bottleneck. The diagnosis is instead a model capacity problem: a linear decision boundary in 768-dimensional space is insufficient to separate 430 classes whose embeddings are not linearly arranged.

Per-class analysis confirms this. Median per-class accuracy is 0%, with only 8 of 430 pieces identified at 100% and 410 pieces at 0%. The aggregate 34.4% is carried almost entirely by those 8 easy pieces — likely ones with distinctive harmonic content that produces separable embeddings. The confusion matrix shows bwv248.64-6 and bwv79.3 acting as prediction attractors, absorbing misclassifications from dozens of other chorales.

Two key implementation lessons emerged from debugging. First, applying L2 normalization to MERT embeddings before the linear layer dropped top-1 accuracy from 34.4% to ~4.4% — MERT embedding magnitudes carry piece-identity information and normalizing to unit length discards it.

**<u>3. MLP on Embeddings</u>**

In comparison with attempt 2, after 1,000 epochs with Adam, this model achieves **27.4% top-1 and 51.1% top-5 test accuracy** against a random baseline of 0.23%. This is a similar top-5 test accuracy as the previous logistic regression model, but a slightly lower top-1 accuracy.

Similar capacity ceilings as the logistic regression also exist upon evaluating loss and accuracy curves. Training accuracy for top-5 reaches nearly 100% while test accuracy for the same plateaus at around 50% around 1/5 of the way into the 1000 epochs.

For the per-class analysis, we find that median per-class accuracy is still 0%, but with considerable improvement of **102 of 430 pieces identified at 100%** and 308 pieces at 0%. While the logistic regression model yielded more pieces which had 2+ snippets confused with another particular piece, this model had just 10 pieces which had 2 snippets confused with another particular piece (all other pairs of pieces confused was a one-off occurrence).

Upon closer manual analysis of the failure cases, meaningful patterns can be seen in the pieces confused for each other, whereas similar patterns could not be seen in the logistic regression. Fewer, specific misclassifications were noted, which can mainly be attributed to pieces being in the same key, and a secondary attribute is potentially similar rhythms (ex. lots of duplets, 3/4, etc.). One such 'misclassification' was actually correct due to the pieces being different harmonizations of the same chorale.

The following examples are given, numbers are BWV (Bach works catalog) numbers:

*Same chorale:* 149.7 + 340

*Same key (+ time signature/rhythms often):*
- 172 + 248.42-4 + 155.5, 325 + 281 (F)
- 137.5 + 19.7 (C)
- 140.7 + 245.40 (E flat)
- 187.7 + 341, 378 + 374 (g minor)
- 288 + 277 (d minor?)

Same-key misclassifications accounted for 172 of 379 total misclassifications. The fraction of misclassifications where the key was same was calculated to be 45.38% (via music21). Compare to the probability that two random pieces share the same key: 8.8% (although number not based on number of snippets), this is a significant result suggesting that key is a primary factor in classification (echoing some past results based on linear/logistic regression on musical features).

### Next Steps
**Further analysis of misclassifications** Currently pieces tend to be misclassfied based on key, but if we may be able to better learn the rhythms, etc. of pieces this may be helpful. Another consideration is whether the number of snippets of a piece (which may range from anywhere from around 3-40) increases its chances of misclassification.

**Understand the embedding space.** Before scaling model capacity further, computing pairwise distances between same-piece vs. different-piece snippets in the raw MERT space would clarify whether the bottleneck is classifier capacity or embedding geometry. If same-piece snippets do not cluster geometrically, no classifier trained on frozen embeddings will solve the problem — and the right response is fine-tuning MERT rather than deepening the classification head.

**Architectural Search** Identify Current Best: Gridsearch over MLP configurations to find most accurate hyperparameters. Explore Other Models: Replace mean-pooling with a Temporal Transformer Encoder or Bi-LSTM.

**Metric learning** This directly optimizes the embedding geometry rather than just the classification head. Use a Triplet Loss or Contrastive Learning (CLAP) approach. Instead of classifying 430 labels, train the model to push embeddings of the same piece together and pull different pieces apart. This way, embeddings can be learned to better support our classification purposes. 

**Continued Expansion of Dataset**: Going beyond the 430 Bach chorales.