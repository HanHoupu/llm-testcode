#!/usr/bin/env python3
"""
Step 6: Adversarial Robustness Evaluation
"""
import os
import sys
import torch
import pandas as pd
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import load_dataset

def set_deterministic_mode(seed=42):
    """Set deterministic mode for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Basic reproducibility settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.quantization.model_utils import replace_with_quant
from src.lora.model_utils import attach_lora_to_model
from src.adversarial import EmbeddingPGD, HotFlip, RandomBitwidthDefense, evaluate_robustness
from src.adversarial.evaluator import filter_high_f1_samples

def build_model_with_lora(precision_bits=None, lora_ckpt=None, device="cuda", model_id="model"):
    """Build independent model instance"""
    # Create independent tokenizer and model (Step 3 compatible)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Setup tokenizer (Step 3 compatible)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Apply quantization if needed
    if precision_bits is not None:
        replace_with_quant(model, {"default_w_bits": precision_bits, "per_layer_bits": {}})
    
    # Load LoRA if available
    wrappers = None
    if lora_ckpt and os.path.exists(lora_ckpt):
        checkpoint = torch.load(lora_ckpt, map_location="cpu")
        wrappers = attach_lora_to_model(model, checkpoint["name2branches"])
        
        for name, wrapper in wrappers.items():
            if name in checkpoint["lora_state_dict"]:
                wrapper.bank.load_state_dict(checkpoint["lora_state_dict"][name], strict=False)
    
    model.to(device).eval()
    return model, tokenizer, wrappers

def main():
    # Setup
    seed = 42
    set_deterministic_mode(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "outputs/ckpt/step3_switchable_model/lora_checkpoint.pt"
    
    # Build independent model instances
    fp32_model, fp32_tokenizer, fp32_wrappers = build_model_with_lora(
        None, checkpoint_path, device, "fp32_lora"
    )
    static8_model, static_tokenizer, static_wrappers = build_model_with_lora(
        8, checkpoint_path, device, "static_8bit"
    )
    random_model, random_tokenizer, random_wrappers = build_model_with_lora(
        8, checkpoint_path, device, "random_switch"
    )
    
    # Load dataset and filter samples
    val_dataset = load_dataset("squad", split="validation")
    high_f1_samples = filter_high_f1_samples(
        fp32_model, fp32_tokenizer, val_dataset, threshold=0.4, max_samples=100
    )
    
    if len(high_f1_samples) < 10:
        print("Warning: Not enough high F1 samples found")
        return
    
    # Setup defense strategies
    static_defense = RandomBitwidthDefense(static8_model, static_wrappers, [8])
    random_defense = RandomBitwidthDefense(random_model, random_wrappers, [4, 6, 8])
    
    # Model configurations
    models = {
        "fp32_lora": {"model": fp32_model, "tokenizer": fp32_tokenizer, "defense": None, "wrappers": fp32_wrappers},
        "static_8bit": {"model": static8_model, "tokenizer": static_tokenizer, "defense": static_defense, "wrappers": static_wrappers},
        "random_switch": {"model": random_model, "tokenizer": random_tokenizer, "defense": random_defense, "wrappers": random_wrappers}
    }
    
    # Initialize white-box attackers
    pgd_whitebox = {}
    hotflip_whitebox = {}
    
    for name, info in models.items():
        torch.manual_seed(seed + hash(name) % 1000)
        pgd_whitebox[name] = EmbeddingPGD(info["model"], info["tokenizer"], epsilon=0.1, steps=8)
        hotflip_whitebox[name] = HotFlip(info["model"], info["tokenizer"], candidate_k=200)
    
    
    results = []
    
    # Evaluate attacks
    for attack_name, attackers in [("pgd", pgd_whitebox), ("hotflip", hotflip_whitebox)]:
        for model_name in models.keys():
            single_model = {model_name: models[model_name]}
            attack_results = evaluate_robustness(
                single_model, high_f1_samples, attackers[model_name].attack, device
            )
            
            for mn, metrics in attack_results.items():
                results.append({
                    "attack": attack_name,
                    "model": mn,
                    "clean_f1": metrics["clean_f1"],
                    "adv_f1": metrics["adv_f1"],
                    "delta_f1": metrics["delta_f1"],
                    "success_rate": metrics["success_rate"],
                    "clean_em": metrics.get("clean_em", 0),
                    "adv_em": metrics.get("adv_em", 0)
                })
    
    
    # Save results
    df = pd.DataFrame(results)
    
    # Add metadata
    df.insert(0, "timestamp", datetime.utcnow().isoformat())
    df.insert(1, "seed", seed)
    df.insert(2, "sample_count", len(high_f1_samples))
    df.insert(3, "sample_selector", "fp32_lora")
    df.insert(4, "deterministic_mode", True)
    
    # Save to CSV
    output_path = "outputs/logs/step6_robustness.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    write_header = not os.path.exists(output_path)
    df.to_csv(output_path, mode="a", index=False, header=write_header)
    
    print(f"Results saved to {output_path}")
    
    # Display results
    display_cols = ["attack", "model", "clean_f1", "adv_f1", "delta_f1", "success_rate"]
    print(df[display_cols].to_string(index=False, float_format='%.3f'))

if __name__ == "__main__":
    main()
