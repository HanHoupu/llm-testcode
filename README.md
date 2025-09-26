
# GPT-2 with switchable quantization and LoRA adapters.

This is a small project for EIC Lab test code.

## Features

**Step 1: Base Quantization** Replace all GPT-2 linear layers with quantized versions.(float32 →2-8bit)

**Step 2: LoRA Integration** Add LoRA adapters to all quantized linear layers, make fine-tuning lighter.

**Step 3: Switchable Precision Training** Run the model on SQuAD with 4/6/8-bit configs, adding a mechanism to set different layers with different bits.

**Step 4: Evaluation on SQuAD** Use different configs to see the performance–bit-width trend, evaluate the model on SQuAD, and try to find a performance–cost balanced setting.

**Step 5: Cyclic Precision Training** Apply cyclic bit-width switching ( 8→4→8→4...) during training of the Step 3 model and evaluate on SQuAD, to see whether dynamic precision can enhance performance.

**Step 6: Adversarial Robustness** Apply random precision switching during deployment and attack the model with PGD and HotFlip, to see whether this improves robustness.

## Setup

please see requirements.txt file.

## Usage

### Step 3 python scripts/step3_switchable_training.py
Train GPT-2 with switchable quantization (4/6/8-bit) and LoRA adapters.

### Step 4 python -m src.evaluation.eval_step4_squad --n 1500 --device cuda
Evaluate the trained model on the SQuAD dataset(n=1500) with EM/F1 metrics.

### Step 5 python scripts/step5_cpt_slope.py
Run cyclic precision training (8→4→8→4...) on GPT-2.

### Step 6 python scripts/step6_adversarial_robustness.py
Apply random precision switching during deployment. Evaluate robustness against adversarial attacks (PGD, HotFlip).

For detailed arguments, see the file itself.

## Results

For full details, see the [Full Report (PDF)](https://hanhoupu.github.io/files/test_code.pdf)].

## Project Structure

The repository is organized as follows:

src/quantization/ # QuantLinear and utilities
src/lora/ # LoRA integration
src/training/ # DataLoader and Trainer
src/evaluation/ # Step 4-6 evaluation scripts
src/adversarial/ # Adversarial attacks and defense
configs/ # YAML configs for quantization
scripts/ # Main training scripts (Step 3, 5, 6)
outputs/ # Logs and checkpoints
notebooks/ # Jupyter notebooks for experiments(old)
utils/ # Utility functions


// TODO :References//