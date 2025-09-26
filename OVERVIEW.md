This is a project overview.

## Quantization (`src/quantization/`) step 1
`core.py`: This contains our main QuantLinear class(like a normal nn.Linear), which allows us to convert float32 weights into 2–8 bit.
`model_utils.py`: Replaces layers with our QuantLinear class, find the linear(Conv1D) layers of GPT2 model. It also supports re-quantizing the model to new bit cfg.
`config.py`: Loads YAML configs from our configs folder.

## LoRA (`src/lora/`) step 2
`core.py`:Defines `LoRA` and `LoRAWrapped` classes, which provides per-layer LoRA adapters.
`model_utils.py`: Adds LoRA to to specified layers.
`activation.py`: Connects YAML configs with our LoRA wrappers, activating the correct branch for each layer.

## Training (`src/training/`) step 3/4/5
`data_utils.py`: Loads SQuAD dataset, prepares the SQuAD dataset to right format for future training.
`trainer.py`: `SwitchableTrainer` Makes the model able to switch bit-widths during training iterations.

## Evaluation (`src/evaluation/`) step 4/5
`metrics.py`: Calculates SQuAD performance (EM/F1),and adversarial robustness tests.
`config_analyzer.py`: Finds all configs files, then tests all configs in batch, returns results as DataFrame.
`eval_step4_squad.py`: Step 4 evaluation script,evaluates all quantization configs on SQuAD, save results to CSV.
(unused)`experiment_logger.py`: Saves Step 4–6 experiment results . (JSON logs)

## Adversarial (`src/adversarial/`) step 6
`attacks.py`: Implements Embedding-PGD and HotFlip attacks.
`defense.py`: Implements Random bitwidth switching defense.
`evaluator.py`: Robustness evaluation functions.(inefficient, ~50% code can be reused from previous steps in future)

## Scripts (`scripts/`)
`step3_switchable_training.py`: Step 3 training script, train our model in SQuAD with switchable quantization and LoRA branches.
`step5_cpt_slope.py`: Step 5 cyclic precision training script, compares step 3 and step 5 result, save results to CSV. (inefficient, ~50% code can be reused from previous steps in future)
`step6_adversarial_robustness.py`: Step 6 adversarial robustness evaluation script, runs attacks and defenses, evaluates, and saves results to CSV.(inefficient, ~30% code can be reused from previous steps in future)

## Configs (`configs/`)
YAML files for different quantization strategies. Format:
- `default_w_bits`: default bit width(maybe unused in this project cfgs.)
- `per_layer_bits`: bit width setting for each specific layer.

