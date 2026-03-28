# Week 10 Report

# Experiment Pipeline Development

## Motivation

Our initial experiments were conducted using exploratory notebooks, which made it difficult to run structured experiments or scale across multiple compute environments. In order to support reproducibility and enable larger model training runs, the notebook-based workflow was modularized into a Python-based experimentation pipeline compatible with Prime Intellect reservable instances.

The primary design goal was to create a **model-agnostic pipeline** capable of supporting different architectures and experiment configurations without requiring major code modifications.

## Pipeline Architecture

The pipeline was refactored into a modular structure using `uv` for dependency management and package loading. This allows experiments to be executed consistently across different compute instances while ensuring reproducible environments.

The repository structure is organized as follows:

```
project/
│
├── models/
│   └── mlp.py
│
├── configs/
│   └── base.py
│   └── mlp.py
│
├── data.py
├── train.py
├── evaluate.py
├── run.py
│
└── pyproject.toml
```

### Component Responsibilities

**run.py**

* Main entry point for experiments
* Loads configuration files
* Orchestrates training and evaluation workflow
* Enables easy swapping of model architectures

**data.py**

* Handles dataset loading and preprocessing
* Produces standardized dataset objects
* Designed to remain independent of model choice

**train.py**

* Implements model training logic
* Accepts model class and configuration parameters
* Supports flexible hyperparameter experimentation

**evaluate.py**

* Computes performance metrics including Top-k accuracy
* Designed to support both classification and retrieval evaluation

**models/**

* Contains model definitions
* Currently includes an MLP baseline
* Designed for easy extension to CNNs, transformers, or metric-learning architectures

**configs/**

* Stores experiment configuration scripts
* Enables reproducible experiments
* Separates experiment parameters from code

## Benefits of the Refactor

This structure enables:

* rapid iteration on new architectures
* reproducible experiments across compute environments
* easier hyperparameter tuning
* scalable execution on Prime Intellect reservable instances
* clean separation between model logic and experiment configuration

Most importantly, the pipeline is designed to support more complex experimentation in the future, including representation learning and retrieval-based evaluation methods.

---

# Embedding Retrieval Experiments

## Motivation

In addition to classification-based approaches, we explored embedding-based retrieval methods as a way to evaluate the quality of learned representations. Rather than predicting class labels directly, this approach assesses whether similar musical pieces cluster together in embedding space.

If embeddings are meaningful, similar musical segments should lie near each other, enabling accurate retrieval via distance-based methods such as k-nearest neighbors (KNN).

## Method

We generated embeddings for each musical segment and performed KNN retrieval over the embedding vectors.

Evaluation metrics:

* Top-1 Accuracy
* Top-5 Accuracy

These metrics measure whether the correct musical piece appears within the closest retrieved neighbors.

To summarize the perturbations:

The perturbations were created with the goal of introducing pitch shifting, time stretching to the FluidSynth renders to prevent the model from "memorizing" clean audio. We used "librosa" library's 'effects' before generating the embeddings, and in doing so made some alterations to the data pipeline, which can be viewed in `perturb.py`.

Some modifications to the pipeline now allow for the pipeline to be entered via music21, MIDI, or WAV files. Rather than creating snippets based on measure numbers from the music21 Score object, the snippets are now creates following perturbations of the audio and based on duration (8 seconds, with 4 seconds overlap to allow more data). These snippets are then embedded with MERT similarly.

The perturbations performed were (1) *tempo change* for which we used 0.9, 0.95, 1.0, 1.05, and 1.1 times the original speed, and (2) *pitch shift* for which we used 0, 1, 2 semitones up/down. Combinations of these changes yield 25 versions. Overall this increased the previous ~7 snippets/piece to now ~179 snippets/piece for a total of 77070 snippets (split across train/test).

---

## Results

### Initial Dataset

| Method            | Top-1 Accuracy | Top-5 Accuracy |
| ----------------- | -------------- | -------------- |
| KNN on embeddings | 0.2035         | 0.3378         |
| MLP classifier    | 0.3440         | 0.5640         |

Performance using KNN retrieval on the initial dataset was significantly lower than the MLP baseline. This suggests that while the embeddings contain some useful structure, they may not yet fully capture discriminative musical characteristics needed for robust classification.

---

### Perturbed Dataset

We then evaluated both methods on a dataset containing perturbations of existing musical pieces.

| Method            | Top-1 Accuracy | Top-5 Accuracy |
| ----------------- | -------------- | -------------- |
| KNN on embeddings | 0.8085         | 0.9325         |
| MLP classifier    | 0.8775         | 0.9655         |

Performance improved substantially for both approaches. While the MLP remains the stronger performer, the gap between the two methods narrowed considerably.

---

## Interpretation

The strong performance of KNN on the perturbed dataset suggests that embeddings for variations of the same musical piece are located close together in embedding space.

Because many samples in this dataset are perturbations of the same underlying piece, the retrieval task becomes easier, as similar samples naturally cluster together. This also benefits the MLP classifier, which can learn decision boundaries more easily when examples from the same class are closely related.

These results indicate that:

* the embedding space contains meaningful structure
* the dataset may not be sufficiently diverse to fully evaluate generalization
* retrieval-based approaches may perform well when samples are highly related
* classification performance may be inflated by similarity between perturbed examples

---

# Further Work

Several directions could improve both the quality of embeddings and the robustness of evaluation.

### 1. Increase Dataset Diversity

The current dataset contains many perturbations of the same musical pieces. While useful for testing robustness to transformations, this may artificially inflate performance metrics.

Future work should incorporate:

* more distinct musical compositions
* broader stylistic diversity
* variations in instrumentation and structure
* cross-dataset validation

A more diverse dataset will provide a stronger test of generalization ability.

---

### 2. Metric Learning Approaches

Given that embedding space structure appears promising, metric learning may improve representation quality.

Potential approaches include:

* triplet loss
* contrastive loss
* supervised contrastive learning
* Siamese networks

These methods explicitly optimize embedding distance relationships, encouraging samples from the same piece to cluster together while separating unrelated pieces.

Metric learning may allow embeddings to serve as a stronger foundation for both retrieval and downstream classification tasks.

---

### 3. Alternative Retrieval Methods

KNN is a simple baseline but may not fully leverage structure in embedding space.

Future retrieval experiments could explore:

* cosine similarity vs Euclidean distance
* approximate nearest neighbor search
* clustering-based retrieval
* hybrid classification-retrieval models

---

### 4. Expanded Model Architectures

The current pipeline structure allows straightforward integration of additional models, such as:

* convolutional neural networks
* transformer-based encoders
* sequence models
* pretrained music representation models

These architectures may better capture temporal and harmonic structure in musical data.

---

### 5. Scalable Experimentation on Prime Intellect

With the pipeline now modularized, future experiments can be scaled across reservable instances.

This enables:

* systematic hyperparameter sweeps
* evaluation across multiple model families
* larger dataset training runs
* reproducible benchmarking

---

### 6. Continued data perturbations

Consider creating more perturbations beyond the current small changes to create for more robustness, especially more perturbations based on pitch. Try to perturb the embeddings directly, rather than the audio, which may be less interpretable but still be valuable. At the same time, may need to reevaluate computing resources needed / generating those changes during training.