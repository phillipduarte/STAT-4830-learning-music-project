# Deployed Pipeline Summary

This summary documents the data format and end-to-end training pipeline in the deployed folder so it can be used as context for designing and testing new models.

## 1) Data Format

Primary loader: data.py

Expected input files (under embeddings_dir, default: `embeddings/`):
- `embeddings_train.npy`
- `embeddings_test.npy`
- `labels_train.npy`
- `labels_test.npy`

### Embedding arrays
- Loaded with NumPy from `.npy` files.
- Expected shape is 2D per split: `(N, d)`.
- `d` (input dimension) is inferred from `X_train.shape[1]`.
- Converted to torch float tensors (`float32`) before model use.

### Label arrays
- Labels are loaded as strings (piece identifiers / class names).
- A `LabelEncoder` is fitted on train labels and applied to train and test labels.
- Encoded labels are integer class indices (`int64`) for PyTorch cross-entropy.

### Dataset and loaders
- `EmbeddingDataset` returns `(X, y)` pairs.
- Train loader: shuffled.
- Test loader: not shuffled.
- Batch size comes from config (`batch_size`, default 256).

### Data metadata returned by loader
`load_data(...)` returns:
- `train_loader`
- `test_loader`
- `le` (fitted `LabelEncoder`)
- `d` (embedding dimension)
- `n_classes`
- `y_train_np` (encoded train labels, used to compute class weights)

## 2) Model Interface and Extensibility

Primary files: models/__init__.py, models/mlp.py, config/base.py, config/mlp.py

### Current architecture
The deployed model is a one-hidden-layer MLP:
- `Linear(d, hidden_dim)`
- `ReLU`
- `Dropout(dropout_p)`
- `Linear(hidden_dim, n_classes)`

Weights are initialized with Kaiming uniform for linear layers; biases are zero-initialized.

### Model registry pattern
- `models/__init__.py` keeps a registry mapping `model_name` to model class.
- `build_model(cfg, d, n_classes)` inspects the target model constructor and forwards only matching config fields.
- This makes the train loop architecture-agnostic and simplifies adding new model families.

### Config split
- `BaseConfig` contains shared settings (paths, device, optimizer, scheduler, loss, training, evaluation plotting options).
- `MLPConfig` extends `BaseConfig` with architecture fields (`hidden_dim`, `dropout_p`) and a descriptive `run_name`.

## 3) End-to-End Pipeline

Primary orchestrator: run.py

### Stage A: configuration + reproducibility
1. Parse CLI arguments.
2. Construct config from `CONFIG_REGISTRY` (currently `mlp`).
3. Apply non-None CLI overrides onto config fields.
4. Set Python/NumPy/Torch random seeds.

### Stage B: load data
1. Call `load_data(cfg.embeddings_dir, cfg.batch_size)`.
2. Receive dataloaders, label encoder, input dim, class count, and train labels.

### Stage C: build model
1. Call `build_model(cfg, d, n_classes)`.
2. Move model to configured device (`cuda` if available, else `cpu`).
3. Print parameter count.

### Stage D: build loss
1. Compute per-class sample counts from `y_train_np`.
2. Build inverse-frequency class weights, normalized to mean around 1.
3. Use `CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)`.

This setup explicitly addresses class imbalance while also applying label smoothing.

### Stage E: train loop
Primary file: train.py

For each epoch:
1. `train_one_epoch`: forward, weighted CE loss, backward, optimizer step.
2. Evaluate on both train and test sets via shared `evaluate(...)`.
3. Step scheduler:
	- Cosine scheduler: step every epoch.
	- Plateau scheduler: step based on test top-1.
4. Store history fields:
	- train/test loss
	- train/test top-1
	- train/test top-5
	- learning rate

Optimizer and schedule defaults:
- Optimizer: Adam
- `lr`: 1e-3
- `weight_decay`: 1e-3
- Scheduler default: cosine annealing
- Epoch default: 1000

### Stage F: checkpoint + final eval
1. Save checkpoint to `outputs/<run_name>.pt`.
2. Checkpoint includes:
	- `model_state`
	- serialized config dict
3. Run final test evaluation to get:
	- `loss`
	- `top1`
	- `top5`
	- predicted class indices
	- true class indices

### Stage G: diagnostics and plots
Primary file: evaluate.py

Generated outputs include:
- Training curves (loss, top-1/top-5, LR schedule)
- Per-class accuracy summary and chart (best/worst classes)
- Focused confusion matrix for most-confused class pairs
- Printed final summary with top-1, top-5, loss, median per-class accuracy, and random baseline

Saved figures go to `outputs/` with filename prefix `<run_name>_...`.

## 4) Additional Baseline Script

File: knn.py

- Independent KNN baseline sweep over `k=1..49` using cosine distance.
- Reports top-1 and top-5.
- Note: this script currently loads data from `../../old_data/` rather than the configurable embeddings path used by run.py.

## 5) Practical Design Implications for New Models

The current deployed system is already set up for easy architecture swaps if the new model follows the same constructor contract.

Recommended compatibility contract for a new model class:
- Constructor accepts at least `d` and `n_classes`.
- Any additional hyperparameters should be added to a config dataclass.
- Register the model in `models/__init__.py` and config in `run.py` config registry.

Because data is already represented as fixed-size embedding vectors, this pipeline is best viewed as a supervised classifier over embedding space. New models that should plug in cleanly include:
- Deeper residual MLPs
- Mixture-of-experts style MLP heads
- Metric-learning heads with classifier fine-tuning
- Lightweight attention/gating blocks on top of embedding features

## 6) Known Caveats

- `evaluate.py` contains a placeholder parameter-count line in `print_final_summary` that does not currently use the actual model parameter count.
- The KNN script uses a different data path convention than the main pipeline.
- Data contract assumes all splits are already precomputed and aligned (no raw feature extraction in deployed folder).

## 7) One-Line Mental Model

`run.py` orchestrates: load fixed embeddings -> encode labels -> build registered model -> train with imbalance-aware CE + scheduler -> save checkpoint -> produce class-level diagnostics for error analysis.
