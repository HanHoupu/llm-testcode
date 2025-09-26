#!/usr/bin/env python3
import os
import sys
import yaml
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.quantization.model_utils import replace_with_quant
from src.lora.model_utils import attach_lora_to_model
from src.training.data_utils import create_squad_dataloader
from src.training.trainer import SwitchableTrainer

def main():
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Load configs (only 4/6/8 bit uniform configs)
    configs = []
    for name in ["C2_all4", "C0_all6", "C1_all8"]:
        with open(f"configs/{name}.yaml", 'r') as f:
            config = yaml.safe_load(f)
            config['name'] = name
            configs.append(config)
    
    print(f"📋 Loaded {len(configs)} training configs: {[c['name'] for c in configs]}")
    
    # Apply quantization
    replace_with_quant(model, configs[0])
    
    # Create LoRA branches (only 4/6/8 bit support, like notebook)
    branches = {}
    for i in range(12):
        for layer in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
            layer_name = f"transformer.h.{i}.{layer}"
            branches[layer_name] = {
                "bw4": (8, 16),   # 4-bit: rank=8, alpha=16
                "bw6": (12, 24),  # 6-bit: rank=12, alpha=24
                "bw8": (16, 32)   # 8-bit: rank=16, alpha=32
            }
    
    print(f"🔧 Created LoRA branches for {len(branches)} layers with [4,6,8]-bit support")
    
    # Setup training
    wrappers = attach_lora_to_model(model, branches)
    dataloader = create_squad_dataloader(tokenizer, batch_size=16, subset_size=87599, seed=42)
    trainer = SwitchableTrainer(model, wrappers, configs, lr=5e-5)
    
    # Train
    trainer.train(dataloader, iterations=1500)
    
    # Save only LoRA parameters for Step 4 evaluation
    import os
    os.makedirs("outputs/ckpt/step3_switchable_model", exist_ok=True)
    
    # Extract only LoRA parameters
    lora_state_dict = {}
    for name, wrapper in wrappers.items():
        lora_state_dict[name] = wrapper.bank.state_dict()
    
    checkpoint = {
        "lora_state_dict": lora_state_dict,
        "name2branches": branches
    }
    torch.save(checkpoint, "outputs/ckpt/step3_switchable_model/lora_checkpoint.pt")
    print("✅ LoRA parameters saved to outputs/ckpt/step3_switchable_model/lora_checkpoint.pt")

if __name__ == "__main__":
    main()
