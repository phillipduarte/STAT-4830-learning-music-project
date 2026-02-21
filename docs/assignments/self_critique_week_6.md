# Week 6 Self-Critique

## OBSERVE

The pipeline runs end-to-end and produces real results: 34.4% top-1, 56.4% top-5 on 430 Bach chorales vs. 0.23% random. The report is clean and mathematically grounded. The per-class results (median 0%, only 117/430 pieces at 100% and 290 at 0%) reveal that most of the pieces are not being classified well. The classifier is doing real work on over a quarter of pieces, but completely fails on the majority. We still have to determine whether that failure is a capacity problem or an embedding geometry problem.

---

## ORIENT

**Strengths**

- We debugged different aspects of the optimization to determine where complexity was needed. Running controlled experiments (full-batch vs. mini-batch, with/without normalization, weight decay sweep) isolated the L2 normalization issue as a cause of decreased accuracy, and we found that weight decay wasn't necessary.
- The mathematical formulation is precise and consistent with the lecture notation, connecting `F.cross_entropy` to the loss we learned in class for logistic regression.
- The four-stage modular pipeline (data_prep → render → embed → classify) is clean and reproducible, with embeddings cached to disk so experiments are fast to iterate.

**Areas for Improvement**

- The 34.4% headline accuracy obscures a bimodal per-class distribution: 117 of 430 pieces are identified at 100% accuracy, while 290 are at 0% — with almost nothing in between. The report treats this as a single aggregate result rather than asking the more interesting question: what distinguishes the 117 solvable pieces from the 290 that the model completely fails on? That analysis would directly motivate the MLP and metric learning directions.
- The current architecture is fundamentally closed-set: it can only predict the 430 Bach chorales it was trained on, and adding a new piece requires full retraining. The report doesn't acknowledge this limitation or propose metric learning as the natural path toward an open-set system — where a new piece is added by embedding it, not by retraining.
- The "Next Steps" section lists three directions (MLP, harder split, embedding analysis) without prioritization or sequencing. We need to think about which steps make the most sense to complete first, or if it's better to split up and explore different approaches in parallel.

**Critical Risks/Assumptions**

The `by_snippet` split means test snippets come from pieces the model has already seen during training, making the evaluation easier than any real-world use case. The 34.4% figure may significantly overstate how well this approach would generalize to unseen pieces. Additionally, the entire pipeline assumes MERT's general-purpose audio representations contain enough piece-specific signal for classification — the per-class results suggest this assumption holds for a small subset of pieces but not most.

---

## DECIDE

**Concrete Next Actions**

- **Add a metric learning section to the report's Next Steps** that frames triplet loss as the progression after the MLP: instead of learning a fixed label mapping, train the embedding space directly so same-piece snippets cluster together. This sets up a by_piece split as the natural evaluation protocol and positions the project toward open-set identification.
- **Build the MLP classifier** (768 → 512 → 430, ReLU, dropout) as the immediate next model. This is the highest-priority direction because the regularization sweep already proved linear capacity is the bottleneck — and the MLP introduces non-convex optimization, which is more interesting for the course.
- **Reframe the results section** around the bimodal per-class distribution: 117 pieces at 100%, 290 at 0%, almost nothing in between. The key question becomes what separates the solvable pieces from the rest — distinctive harmonic content, longer pieces with more training snippets, or something in the embedding geometry. That analysis sharpens the motivation for both the MLP and metric learning.

---

## ACT

**Resource Needs**

The MLP implementation is straightforward in PyTorch — the main question is dropout rate and hidden size, which should be treated as hyperparameters to tune rather than fixed upfront. The key blocker is understanding whether the train/test gap widens significantly with more capacity, which requires running the MLP for a full training run before drawing conclusions. No new libraries or infrastructure needed — the existing embedding pipeline and DataLoader work unchanged.
