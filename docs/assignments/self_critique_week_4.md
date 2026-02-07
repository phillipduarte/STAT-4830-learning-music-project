# Week 4 Self-Critique

## 1. OBSERVE: Initial Reaction

Reading through my report and code, I realize I've built a working system but may have missed the mark on what an "optimization class project" should focus on. My approach feels more like a traditional ML feature engineering problem than a deep optimization problem. The professor's suggestion about learning embeddings is a significant pivot that better aligns with the course focus.

## 2. ORIENT: Analysis

### Strengths
- **Working end-to-end pipeline**: Successfully implemented multiple optimization algorithms (PGD, SLSQP, CVXPY) with proper constraints and validation
- **Clear mathematical formulation**: The constrained optimization on the simplex is well-defined and properly implemented
- **Methodical evaluation**: Used multiple metrics (accuracy, precision, recall, F1) and compared across methods

### Areas for Improvement
- **Wrong level of abstraction**: Optimizing 5 hand-crafted feature weights is too simple - this is undergraduate ML, not graduate optimization. The "optimization" is trivial once features are chosen.
- **Missing the core challenge**: The real optimization problem should be *learning the representations themselves*, not just weighting pre-computed features. Embeddings offer a much richer optimization landscape.
- **Limited scalability**: Current approach won't scale beyond toy datasets. Hand-crafting features for complex musical relationships (modulation, voice leading, harmonic progressions) is intractable.

### Critical Risks/Assumptions
- **Assumption**: Hand-crafted features can capture musical similarity. Reality: This ignores learned representations that have proven far more effective (transformers for MIDI/sheet music).
- **Risk**: Continuing down this path means building a project that doesn't demonstrate optimization depth expected in this course. The gradient descent issues mentioned in my report (divergence after iteration 1000) hint that the problem is too simple to be interesting.

## 3. DECIDE: Concrete Next Actions

### Immediate Pivots (Week 5)
1. **Switch to embedding learning**: Reformulate the problem as learning embeddings for sheet music pages where similar pages have close embeddings in learned space. Use pre-trained music transformers (skytnt/midi-model or MuPT-v1) as starting points.

2. **Implement metric learning framework**: Set up PyTorch pipeline for:
   - Triplet loss or contrastive loss optimization
   - Embedding projection layers on top of frozen/fine-tuned transformer
   - Hard negative mining strategies

3. **Redefine optimization problem**: Instead of learning 5 weights, optimize:
   - Embedding space geometry (via learned projection matrices)
   - Margin parameters in metric learning losses
   - Fine-tuning strategies (which layers, learning rates, regularization)

### Why This Pivot Matters
The optimization becomes non-trivial: multi-layer neural network optimization with non-convex loss landscapes, careful initialization, learning rate schedules, gradient clipping, batch composition strategies. This is graduate-level optimization, not just solving `min ||Ax - b||²`.

## 4. ACT: Resource Needs

### Technical Learning Required
- **Hugging Face Transformers**: Need to learn how to load and use the two models professor suggested. Start with their documentation and example notebooks.
- **Metric Learning in PyTorch**: Study PyTorch Metric Learning library or implement triplet/contrastive losses from scratch. Key concepts: anchor-positive-negative sampling, margin tuning, hard negative mining.
- **MIDI/symbolic music processing**: The transformers expect specific input formats. Need to convert my music21 corpus to MIDI or tokenized format compatible with these models.

### Immediate Blockers
- **Computational resources**: Transformer fine-tuning needs GPU. Need to confirm available compute (Colab Pro, university cluster, or local GPU).
- **Data format mismatch**: Current features are in CSV; transformers need sequential MIDI tokens or audio. Need preprocessing pipeline.
- **Conceptual gap**: Need to deeply understand what these music transformers have learned and how to extract/fine-tune embeddings for my similarity task.

### Week 5 Gameplan
1. Monday-Tuesday: Deep dive into the two transformer models - read papers, run inference examples, understand architecture
2. Wednesday: Implement basic embedding extraction from frozen model + simple projection layer
3. Thursday: Get triplet/contrastive loss training loop working on my Bach chorales
4. Friday: Compare learned embeddings vs. hand-crafted features, document in new notebook

This pivot transforms my project from "learn 5 weights" to "optimize high-dimensional embedding geometry with sophisticated loss functions" - much more aligned with course objectives.
