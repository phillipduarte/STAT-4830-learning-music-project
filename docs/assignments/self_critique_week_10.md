# Week 10 Self-Critique

## OBSERVE

The pipeline has been successfully refactored from exploratory notebooks into a modular, reproducible Python package compatible with Prime Intellect instances. Embedding-based KNN retrieval was evaluated alongside the MLP classifier on two datasets. On the initial dataset, KNN retrieval achieves 20.35% Top-1 and 33.78% Top-5, well below the MLP's 34.40% / 56.40%. On the perturbed dataset, both methods improve dramatically — KNN reaches 80.85% / 93.25% and the MLP reaches 87.75% / 96.55%. The gap between methods narrows significantly on the perturbed set, suggesting that the embedding space captures intra-piece variation well but struggles with inter-piece discrimination on more diverse data.

---

## ORIENT

**Strengths**

- The pipeline refactor is a meaningful engineering contribution. The model-agnostic design with separated configs, model definitions, and training logic makes it straightforward to plug in new architectures (CNNs, transformers) without touching core pipeline code. This directly enables the metric learning and architecture expansion directions identified in earlier self-critiques.
- The side-by-side comparison of KNN retrieval and MLP classification across two datasets is analytically useful. Evaluating both on the same splits allows performance differences to be attributed to the method rather than the data, which is methodologically sound.
- The interpretation section correctly identifies that the perturbed dataset's high performance may reflect artificial proximity between training and test samples rather than true generalization — this is honest and important self-assessment.

**Areas for Improvement**

- The report does not analyze *which* pieces or classes the KNN retrieval succeeds or fails on. Just as the Week 6 critique identified the bimodal per-class distribution (117 pieces at 100%, 290 at 0%) as the key diagnostic question for the MLP, the same analysis should be applied here: are the retrieval failures concentrated on specific pieces, and do those failures overlap with the MLP's failures? That overlap — or lack of it — would reveal whether the two methods are failing for the same structural reason or different ones.
- The dramatic performance jump on the perturbed dataset (KNN: 20% → 81%) is noted but not fully interrogated. The report attributes it to intra-class clustering from perturbations, but doesn't ask how much of the test set is directly derived from training pieces. If the perturbed dataset has near-duplicate train/test pairs, both metrics may be largely measuring memorization rather than generalization, and the gap between the two datasets is the more important result to explain.
- The pipeline architecture section lists component responsibilities clearly, but the report doesn't discuss any design decisions or tradeoffs encountered during the refactor. What was hard to generalize? What assumptions are baked into `data.py` that might break when switching to a new dataset or modality? Surfacing these would make the engineering contribution more analytically rigorous.

**Critical Risks/Assumptions**

The perturbed dataset's high scores risk becoming the headline result if not framed carefully. Because perturbations of the same piece are likely distributed across train and test splits, the evaluation is closer to a robustness-to-augmentation test than a generalization test. If future experiments on more diverse, held-out data revert closer to the initial dataset's numbers, the Week 10 results will look like an artifact of data construction rather than a real improvement. The "Further Work" section acknowledges this, but the risk should be foregrounded in the results interpretation rather than deferred to a future section.

---

## DECIDE

**Concrete Next Actions**

- **Run per-class retrieval diagnostics.** For each piece, compute whether it appears in Top-1 and Top-5 retrieval results, and compare the failure set to the MLP's failure set from Week 6. If the failures are correlated, the bottleneck is in the embedding geometry and metric learning is the right next step. If they're uncorrelated, the two methods are making different errors and an ensemble or hybrid approach could be worth exploring.
- **Audit the perturbed dataset split.** Before reporting further results on the perturbed set, determine what fraction of test samples are perturbations of training pieces. If that fraction is high, establish a stricter evaluation protocol — either a by-piece split (all perturbations of a piece go exclusively to train or test) or a held-out set of entirely unseen compositions. This will give a more honest picture of where the model actually stands.
- **Begin metric learning implementation.** The pipeline is now ready to support triplet or contrastive loss training. The next concrete step is implementing a triplet sampling strategy over the existing embedding outputs and running a training loop that directly optimizes embedding distances. This was the direction identified in both the Week 4 and Week 6 self-critiques and the infrastructure is now in place to pursue it.

---

## ACT

**Resource Needs**

The per-class diagnostic requires no new infrastructure — it's a post-processing step on existing evaluation outputs. The dataset audit similarly requires only inspecting the split construction logic in `data.py`. The metric learning implementation will require adding a new loss module and a triplet sampler, but the training loop in `train.py` and the model abstraction in `models/` are already designed to accommodate this. The main open question is whether hard negative mining is needed from the start or whether random triplet sampling is sufficient as a baseline — this should be answered empirically with a short initial run before committing to a more complex sampling strategy. Compute on Prime Intellect instances is now accessible via the new pipeline, so scaling up once the loss is validated locally should be straightforward.