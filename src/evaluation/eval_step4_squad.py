# -*- coding: utf-8 -*-
"""
Step 4 evaluator (moved under src/):
  python -m src.evaluation.eval_step4_squad --n 500 --device cuda
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .config_analyzer import analyze_all_configs
from ..quantization import replace_with_quant


def create_fresh_model(checkpoint_path=None):
    """Create a fresh GPT-2 model and LoRA wrappers."""
    # Create fresh GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.pad_token_id = 50256  # Use EOS token as pad token
    
    # Apply quantization
    replace_with_quant(model, {"default_w_bits": 8, "per_layer_bits": {}})
    
    # Setup LoRA if checkpoint exists
    wrappers = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        from ..lora.model_utils import attach_lora_to_model
        wrappers = attach_lora_to_model(model, checkpoint["name2branches"])
        
        # Load LoRA parameters
        for name, wrapper in wrappers.items():
            if name in checkpoint["lora_state_dict"]:
                wrapper.bank.load_state_dict(checkpoint["lora_state_dict"][name], strict=False)
    
    return model, wrappers

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: Evaluate quantization configs on SQuAD")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--output_csv", type=str, default="outputs/logs/quant_eval.csv")
    parser.add_argument("--n", dest="n_examples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Setup GPT-2 tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Setup device and half precision
    device = torch.device(args.device)
    use_half = args.device.startswith("cuda")
    
    # Create model factory that applies half precision if needed
    def model_factory(checkpoint_path):
        model, wrappers = create_fresh_model(checkpoint_path)
        if use_half:
            model = model.half()
        return model, wrappers

    # Load dataset
    val_ds = load_dataset("squad", split="validation")
    n_eval = args.n_examples if args.n_examples and args.n_examples > 0 else len(val_ds)

    # Check if checkpoint exists
    checkpoint_path = "outputs/ckpt/step3_switchable_model/lora_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print("✅ Found LoRA checkpoint, will load for each config")
    else:
        print("⚠️  No LoRA checkpoint found, using base GPT-2")
        checkpoint_path = None

    # Evaluate all configs with fresh model for each
    df = analyze_all_configs(model_factory, tok, val_ds, device, 
                           config_dir=args.config_dir, n=n_eval, 
                           checkpoint_path=checkpoint_path)
    if df is None or df.empty:
        return

    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    write_header = not os.path.exists(args.output_csv)
    df.insert(0, "timestamp", datetime.utcnow().isoformat())
    df.to_csv(args.output_csv, mode="a", index=False, header=write_header)
    
    print(f"\n✅ Results saved to {args.output_csv}")
    print(f"📊 Evaluated {len(df)} configurations")
    print(f"🏆 Best F1: {df.iloc[0]['config']} ({df.iloc[0]['F1']:.3f})")


if __name__ == "__main__":
    main()

