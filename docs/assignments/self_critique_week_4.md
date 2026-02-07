# Week 4 Self-Critique

## ORIENT: Analysis

### Strengths
- **Working end-to-end pipeline**: Successfully implemented multiple optimization algorithms (PGD, SLSQP, CVXPY) with proper constraints and validation
- **Clear mathematical formulation**: The constrained optimization on the simplex is well-defined and properly implemented
- **Methodical evaluation**: Used multiple metrics (accuracy, precision, recall, F1) and compared across methods

### Areas for Improvement
- **Too shallow**: The optimization problem (learning 5 weights under simplex constraints) is essentially solved once features are hand-crafted. This is a convex problem with known solutions.
- **Missing the core challenge**: A better optimization problem could be *learning the representations themselves*, not just weighting pre-computed features. Embeddings offer a much richer optimization landscape.

### Critical Risks/Assumptions
- **Assumption**: Hand-crafted features can capture musical similarity. Reality: This ignores learned representations that have proven far more effective (transformers for MIDI/sheet music).
- **Risk**: Continuing down this path means building a project that doesn't demonstrate optimization depth expected in this course. The gradient descent issues mentioned in my report (divergence after iteration 1000) hint that the problem is too simple to be interesting.

## DECIDE: Concrete Next Actions

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

## ACT: Resource Needs

### Technical Learning Required
- **Hugging Face Transformers**: Need to learn how to load and use the two models professor suggested. Start with their documentation and example notebooks.
- **Metric Learning in PyTorch**: Study PyTorch Metric Learning library or implement triplet/contrastive losses from scratch. Key concepts: anchor-positive-negative sampling, margin tuning, hard negative mining.
- **MIDI/symbolic music processing**: The transformers expect specific input formats. Need to convert my music21 corpus to MIDI or tokenized format compatible with these models.

### Immediate Blockers
- **Computational resources**: Transformer fine-tuning needs GPU. Need to confirm available compute (provided by Professor Davis).
- **Data format mismatch**: Current features are in CSV; transformers need sequential MIDI tokens or audio. Need preprocessing pipeline.
- **Conceptual gap**: Need to deeply understand what these music transformers have learned and how to extract/fine-tune embeddings for my similarity task.

### Week 5 Gameplan
1. Monday-Tuesday: Deep dive into the two transformer models - run inference examples, understand architecture
2. Wednesday: Implement basic embedding extraction from frozen model + simple projection layer
3. Thursday: Get triplet/contrastive loss training loop working on my Bach chorales
4. Friday: Compare learned embeddings vs. hand-crafted features, document in new notebook

This pivot transforms my project from "learn 5 weights" to "optimize high-dimensional embedding geometry with sophisticated loss functions,” which should lend to a more meaningful optimization question and improved results.


