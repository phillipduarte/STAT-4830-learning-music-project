Config — BaseConfig holds everything architecture-agnostic. MLPConfig inherits from it and adds only hidden_dim and dropout_p. When you add a transformer, you'd make TransformerConfig(BaseConfig) with its own fields. The run_name property auto-generates a unique string for output files, which also solves the drift issue from the original script.

models/__init__.py's build_model — uses inspect.signature to figure out which config fields the model's __init__ wants, and passes only those. This means the training loop never needs to know what architecture it's running — you just drop in a new model class and it wires itself.

train.py — train_one_epoch and evaluate know nothing about model architecture. The scheduler dispatch (cosine vs plateau) is handled cleanly since plateau needs the metric passed to .step() while cosine doesn't.

run.py — CLI overrides are applied on top of the config defaults, so you can do quick sweeps without editing any file. The checkpoint saves both the weights and the config dict together, so you always know what produced a given .pt file.