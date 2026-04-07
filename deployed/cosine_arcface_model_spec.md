# Cosine / ArcFace-Style Classifier for Mean-Pooled MERT Embeddings

## Overview

This model is designed for **piece prediction from fixed-length, mean-pooled MERT embeddings**. The goal is to improve class separation over a standard MLP classifier without requiring access to the pre-mean-pool embedding sequences.

The key idea is to keep the current input representation pipeline unchanged while replacing the standard classification head with a **geometry-aware embedding classifier**. Instead of only learning logits for cross-entropy, the model learns a projected embedding space in which snippets from the same musical piece cluster more tightly and snippets from different pieces are pushed farther apart.

This document describes the model using a **small projection network** followed by either:

1. a **cosine classifier**, or  
2. an **ArcFace-style margin classifier**.

---

## Motivation

A standard MLP classifier typically uses:

- input pooled embedding
- one or more hidden layers
- final linear layer
- softmax + cross-entropy loss

This can work well, but it does not explicitly encourage a clean embedding geometry. It only requires the correct class logit to be larger than the others.

In this task, the input is already a meaningful embedding derived from MERT. That suggests a different approach:

- refine the pooled embedding with a small projection network
- normalize the projected representation
- classify using angular similarity rather than unconstrained logits
- optionally enforce an angular margin for the true class

This is useful when:

- k-NN on the embeddings is somewhat effective
- the embeddings contain class signal
- the standard MLP performs better than k-NN, suggesting that the raw embedding space is not ideally organized
- the goal is to improve discrimination without re-deriving the full per-frame MERT outputs

---

## Input and Output

### Input

Each training example is:

- a **single mean-pooled MERT embedding**
- shape: `(input_dim,)`

If batched:

- shape: `(batch_size, input_dim)`

### Output

The model produces:

1. a **projected embedding**
2. normalized class scores for multiclass piece prediction

Depending on the head, the classifier output is based on:

- cosine similarity between the projected embedding and class weights
- optionally modified by an angular margin for the target class

---

## Model Architecture

### High-Level Structure

```text
mean-pooled MERT embedding
    -> projection network
    -> normalized embedding
    -> cosine / ArcFace classifier head
    -> class logits
```

### Recommended Projection Network

A simple and strong default:

```text
Linear(input_dim -> hidden_dim)
LayerNorm or BatchNorm
GELU or ReLU
Dropout
Linear(hidden_dim -> embed_dim)
L2 normalization
```

### Suggested Dimensions

Good starting values:

- `input_dim`: depends on MERT output size
- `hidden_dim`: `256` or `512`
- `embed_dim`: `128` or `256`
- `dropout`: `0.1` to `0.3`

A deeper alternative is possible, but start simple:

```text
Linear(input_dim -> hidden_dim)
GELU
Dropout
Linear(hidden_dim -> hidden_dim)
GELU
Dropout
Linear(hidden_dim -> embed_dim)
L2 normalization
```

A residual MLP block can also be used, but the first experiment should prioritize clarity and stability.

---

## Embedding Normalization

The final projected embedding should be **L2-normalized** before classification.

Let the output of the projection network be:

```math
h \in \mathbb{R}^{d}
```

Normalize it as:

```math
\hat{h} = \frac{h}{\|h\|_2}
```

This ensures that classification depends primarily on **angular similarity** rather than vector magnitude.

---

## Cosine Classifier Head

### Definition

Let:

- `\hat{h}` be the normalized projected embedding
- `\hat{w}_c` be the normalized weight vector for class `c`

Then the class score is:

```math
z_c = s \cdot \hat{w}_c^T \hat{h}
```

where:

- `s` is a scaling factor
- `\hat{w}_c^T \hat{h}` is the cosine similarity between the embedding and the class prototype

These logits are passed into standard cross-entropy loss.

### Intuition

This head treats each class as a direction in embedding space. The model learns to align each snippet embedding with the correct class direction and separate it from others.

### Why Use It

Compared to a standard linear classifier, a cosine classifier:

- removes dependence on uncontrolled feature magnitude
- makes the classifier geometry easier to interpret
- often improves performance on embedding-based classification tasks

---

## ArcFace-Style Margin Head

### Definition

ArcFace modifies the target class score by adding an angular margin.

Let:

```math
\cos(\theta_y) = \hat{w}_y^T \hat{h}
```

For the correct class `y`, ArcFace uses:

```math
z_y = s \cdot \cos(\theta_y + m)
```

For all incorrect classes:

```math
z_c = s \cdot \cos(\theta_c), \quad c \neq y
```

where:

- `m` is the angular margin
- `s` is a scaling factor

These logits are then used with cross-entropy.

### Intuition

Standard cosine classification asks the correct class to be closer than the others.

ArcFace asks the correct class to be closer by a **meaningful angular margin**, which encourages:

- tighter within-class clusters
- larger inter-class separation
- better generalization when classes are similar

### Why Use It

ArcFace is appropriate when:

- the feature space already has some semantic structure
- the goal is to improve class separation rather than just fit a discriminative boundary
- a plain MLP has plateaued

---

## Recommended Training Variants

To isolate where the gain comes from, implement and compare the following variants.

### Variant A: Standard Projection + Softmax

```text
input -> projection network -> linear classifier -> cross-entropy
```

Use this as a baseline to test whether a bottleneck projection alone helps.

### Variant B: Projection + Cosine Classifier

```text
input -> projection network -> normalized embedding -> cosine classifier -> cross-entropy
```

This is the easiest geometry-aware upgrade.

### Variant C: Projection + ArcFace Head

```text
input -> projection network -> normalized embedding -> ArcFace classifier -> cross-entropy
```

This is the strongest version of the proposed model.

---

## Loss Functions

### Standard Cross-Entropy

Used for:

- standard softmax classifier
- cosine classifier

### ArcFace Loss

ArcFace still uses cross-entropy over logits, but the target class logit is replaced with the angular-margin version.

From an implementation perspective, it behaves like:

1. compute cosine similarities
2. modify the target class similarity
3. apply scaling
4. compute cross-entropy

---

## Suggested Hyperparameters

### Projection Network

- `hidden_dim = 256` or `512`
- `embed_dim = 128` or `256`
- `dropout = 0.1` to `0.3`
- activation = `GELU` or `ReLU`

### Optimization

- optimizer = `AdamW`
- learning rate = `1e-3` or `3e-4`
- weight decay = `1e-4`
- batch size = task-dependent
- early stopping on validation accuracy or macro-F1

### Cosine Head

- scale `s = 16` to `32`

### ArcFace Head

Typical starting values:

- scale `s = 16` to `32`
- margin `m = 0.2` to `0.35`

If training becomes unstable, reduce the margin first.

---

## Recommended Evaluation Metrics

Because this is multiclass piece prediction, evaluate at least:

- top-1 accuracy
- top-k accuracy (for example top-5 if useful in your setup)
- macro-F1 if class balance is uneven
- confusion matrix for commonly confused pieces

Also compare against:

- current best MLP
- k-NN baseline on raw pooled embeddings

---

## Expected Behavior

### If This Model Helps

You may observe:

- improved validation accuracy over the standard MLP
- cleaner class clustering in embedding space
- better nearest-neighbor behavior in the learned projection space
- lower confusion among similar pieces

### If This Model Does Not Help

Possible reasons:

- mean pooling has removed too much useful information
- the current embeddings are already near their limit
- there are too few examples per piece
- the bottleneck is representation quality, not classifier geometry

In that case, the next step would likely be to re-derive the pre-mean-pool MERT embeddings and test sequence-aware pooling or temporal models.

---

## Ablation Plan

A minimal experiment plan:

1. **Raw pooled embeddings + k-NN**
2. **Raw pooled embeddings + current MLP**
3. **Projection network + softmax**
4. **Projection network + cosine classifier**
5. **Projection network + ArcFace head**

This sequence helps answer:

- does the projection help?
- does normalized angular classification help?
- does margin-based separation help?

If possible, also evaluate k-NN on the **learned projected embeddings** after training. If that improves substantially, it is evidence that the model has learned a cleaner geometry.

---

## Practical Implementation Notes

### Class Weights

For cosine / ArcFace heads, the classifier weights should also be normalized before similarity is computed.

### Numerical Stability

For ArcFace:

- clamp cosine values before `arccos`-related computations if necessary
- take care with margin implementation to avoid NaNs
- verify that logits and labels are aligned correctly

### Regularization

Reasonable defaults:

- dropout in projection network
- weight decay
- early stopping
- optional label smoothing for the baseline softmax model

### Initialization

Use standard PyTorch initialization for linear layers unless you already have a preferred setup.

---

## Recommended Default Configuration

A sensible first implementation:

```yaml
model:
  type: projection_arcface
  input_dim: <mert_pooled_dim>
  hidden_dim: 512
  embed_dim: 128
  dropout: 0.2
  activation: gelu

head:
  type: arcface
  scale: 24.0
  margin: 0.25

optimizer:
  type: adamw
  lr: 0.0005
  weight_decay: 0.0001

training:
  batch_size: 64
  epochs: 100
  early_stopping_patience: 10
```

A matching cosine version would simply swap the head type and remove the margin.

---

## Summary

This model is a **drop-in upgrade for fixed mean-pooled MERT embeddings**. It does not require sequence-level MERT outputs and is intended to improve performance by learning a more discriminative embedding space.

The main recommendation is to implement:

1. a **projection network**
2. a **cosine classifier**
3. optionally an **ArcFace-style head**

This should be evaluated directly against the current MLP and k-NN baselines.

If it succeeds, the likely reason is that the pooled MERT embeddings contain useful class information but are not yet arranged in the most discriminative geometry for piece-level classification.
