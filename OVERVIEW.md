This ia a project overview.
## Quantization (`src/quantization/`)
`core.py`: Main `QuantLinear` class.
`model_utils.py`: Replaces layers with quantized ones.
`config.py`: Loads YAML configs.

## LoRA (`src/lora/`)
`core.py`: `LoRA` and `LoRAWrapped` classes.
`model_utils.py`: Adds LoRA to model.
`activation.py`: Activates LoRA branches.

## Training (`src/training/`)
`data_utils.py`: Loads SQuAD dataset.
`trainer.py`: `SwitchableTrainer` for training.

## Evaluation (`src/evaluation/`)
`metrics.py`: Calculates SQuAD performance.
`config_analyzer.py`: Tests all configs.
`experiment_logger.py`: Saves results.

_Note: `__init__.py` files just import everything for easy use._
