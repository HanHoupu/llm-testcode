#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch
from datasets import load_dataset

from src.quantization.model_utils import replace_with_quant, requantize_model_to_config
from src.quantization.core import QuantLinear
from src.training.data_utils import create_squad_dataloader
from src.lora.model_utils import attach_lora_to_model
from src.lora.activation import activate_lora_by_config
from src.evaluation.metrics import eval_config
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_all_bits(model, bits: int) -> None:
    """Requantize all quantized layers to the same bitwidth."""
    requantize_model_to_config(model, {"default_w_bits": bits, "per_layer_bits": {}})

def build_lora_wrappers_auto(model, bits: List[int]) -> Dict[str, object]:
    """Attach LoRA to all QuantLinear-like layers with branches per bit (same as Step3)."""
    # Use same branch config as Step3 for fair comparison
    branches = {}
    for b in bits:
        if b == 4:
            branches[f"bw{b}"] = (8, 16)   # 4-bit: rank=8, alpha=16
        elif b == 6:
            branches[f"bw{b}"] = (12, 24)  # 6-bit: rank=12, alpha=24
        elif b == 8:
            branches[f"bw{b}"] = (16, 32)  # 8-bit: rank=16, alpha=32
        else:
            branches[f"bw{b}"] = (8, 16)   # fallback
    
    name2branches: Dict[str, Dict[str, tuple]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinear) or (hasattr(mod, 'in_features') and hasattr(mod, 'out_features')):
            name2branches[name] = branches
    return attach_lora_to_model(model, name2branches)


# ---- Reused from Step4 (fresh model creation) ----
def create_fresh_model_step5(checkpoint_path=None):
    """Create a fresh model and LoRA wrappers (Step5 copy of Step4 logic)."""
    # Create fresh model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.config.pad_token_id = 50256  # eos_token_id
    
    # Apply quantization
    replace_with_quant(model, {"default_w_bits": 8, "per_layer_bits": {}})
    
    # Setup LoRA if checkpoint exists
    wrappers = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        wrappers = attach_lora_to_model(model, checkpoint["name2branches"])
        
        # Load LoRA parameters
        for name, wrapper in wrappers.items():
            if name in checkpoint["lora_state_dict"]:
                wrapper.bank.load_state_dict(checkpoint["lora_state_dict"][name], strict=False)
    
    return model, wrappers


# ---- GPT-2 training ----


def train_gpt2(
    model: AutoModelForCausalLM,
    dataloader,
    steps: int,
    bitwidths: List[int],
    segment_steps: int = 50,
    lr: float = 1e-4,
    dev: Union[str, torch.device] = "cuda",
) -> Tuple[List[Tuple[int, float]], Dict[str, object]]:
    """CPT training for GPT-2 on SQuAD (minimal)."""
    device = torch.device(dev)
    model.to(device)
    model.train()
    hist: List[Tuple[int, float]] = []

    data_iter = iter(dataloader)

    # freeze all params, then optimize LoRA only
    for p in model.parameters():
        p.requires_grad = False
    wrappers = build_lora_wrappers_auto(model, bitwidths)
    lora_params = [p for w in wrappers.values() for p in w.bank.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=lr)

    def batch_iter_reset():
        nonlocal data_iter
        data_iter = iter(dataloader)

    def step_once() -> float:
        try:
            batch = next(data_iter)
        except StopIteration:
            batch_iter_reset()
            batch = next(data_iter)
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)  # Use labels from dataloader (answer-only supervision)
        loss = outputs.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach().cpu())

    t = 0
    idx = 0
    while t < steps:
        nb = bitwidths[idx % len(bitwidths)]
        idx += 1
        set_all_bits(model, nb)
        activate_lora_by_config(wrappers, {"default_w_bits": nb, "per_layer_bits": {}}, 
                               verbose=False, config_name=f"cpt_{nb}bit")
        logging.info("choose bits=%d (cyclic)", nb)
        steps_to_run = min(segment_steps, steps - t)
        for _ in range(steps_to_run):
            loss_val = step_once()
            hist.append((nb, loss_val))
            t += 1
            if t % 100 == 0 or t == steps:
                logging.info("step=%d bits=%d loss=%.4f", t, nb, loss_val)
    return hist, wrappers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step5: GPT-2 + SQuAD CPT (cyclic)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--bitwidths", type=int, nargs="+", default=[8, 4])
    p.add_argument("--segment_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda"])  # GPU only
    p.add_argument("--model_path", type=str, default="gpt2")
    p.add_argument("--squad_subset", type=int, default=87599)  # Full SQuAD training set
    p.add_argument("--eval_n", type=int, default=10570)  # Full validation set
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---- Reused from Step4 (comparison evaluation) ----
def compare_step3_vs_step5(step5_model, step5_wrappers, tokenizer, bitwidths, device, n_eval=500):
    """Compare Step3 (random) vs Step5 (CPT) results using Step4 evaluation method."""
    val_ds = load_dataset("squad", split="validation")
    
    # Pre-sample fixed evaluation samples (same as Step4)
    import random
    random.seed(42)
    if n_eval < len(val_ds):
        fixed_indices = random.sample(range(len(val_ds)), n_eval)
        fixed_samples = val_ds.select(fixed_indices)
        logging.info(f"Using {len(fixed_samples)} fixed samples for Step3 vs Step5 comparison")
    else:
        fixed_samples = val_ds
        logging.info(f"Using full validation set: {len(val_ds)} samples")
    
    all_results = []
    
    # 1. Evaluate Step5 CPT results (current model)
    logging.info("=== Evaluating Step5 CPT results ===")
    step5_model.eval()
    for bits in bitwidths:
        cfg = {"default_w_bits": bits, "per_layer_bits": {}}
        config_name = f"step5_cpt_{bits}bit"
        result = eval_config(step5_model, tokenizer, fixed_samples, device, cfg, step5_wrappers, 
                           len(fixed_samples), config_name=config_name)
        all_results.append({
            "config": config_name,
            "method": "CPT",
            "EM": result["exact_match"],
            "F1": result["f1"],
            "default_bits": bits
        })
        logging.info(f"Step5 CPT {bits}-bit: EM={result['exact_match']:.3f}, F1={result['f1']:.3f}")
    
    # 2. Evaluate Step3 results (load checkpoint)
    step3_checkpoint = "outputs/ckpt/step3_switchable_model/lora_checkpoint.pt"
    if os.path.exists(step3_checkpoint):
        logging.info("=== Evaluating Step3 Random results ===")
        for bits in bitwidths:
            # Create fresh model for each evaluation (same as Step4)
            step3_model, step3_wrappers = create_fresh_model_step5(step3_checkpoint)
            step3_model.to(device)
            step3_model.eval()
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            cfg = {"default_w_bits": bits, "per_layer_bits": {}}
            config_name = f"step3_random_{bits}bit"
            result = eval_config(step3_model, tokenizer, fixed_samples, device, cfg, step3_wrappers,
                               len(fixed_samples), config_name=config_name)
            all_results.append({
                "config": config_name,
                "method": "Random", 
                "EM": result["exact_match"],
                "F1": result["f1"],
                "default_bits": bits
            })
            logging.info(f"Step3 Random {bits}-bit: EM={result['exact_match']:.3f}, F1={result['f1']:.3f}")
            
            # Clean up
            del step3_model, step3_wrappers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        logging.warning("Step3 checkpoint not found, skipping Step3 comparison")
    
    return pd.DataFrame(all_results)


def main() -> None:
    a = parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Step 5; no CUDA device found.")
    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_path)
    model.config.pad_token_id = tok.eos_token_id
    init_bits = a.bitwidths[0]
    ret = replace_with_quant(model, {"default_w_bits": init_bits, "per_layer_bits": {}})
    if ret is not None:
        model = ret
    logging.info("quantized GPT-2 with default bits=%d", init_bits)
    dl = create_squad_dataloader(tok, batch_size=a.batch_size, subset_size=a.squad_subset, seed=a.seed)
    
    # CPT training
    hist, wrappers = train_gpt2(
        model=model,
        dataloader=dl,
        steps=a.steps,
        bitwidths=a.bitwidths,
        segment_steps=a.segment_steps,
        lr=a.lr,
        dev=a.device,
    )
    
    # Training summary
    agg = {}
    for b, l in hist:
        agg.setdefault(b, []).append(l)
    for b, ls in sorted(agg.items()):
        logging.info("bits=%d avg_loss=%.4f n=%d", b, sum(ls) / len(ls), len(ls))
    
    # Post-training comparison: Step3 vs Step5 (reusing Step4 evaluation logic)
    logging.info("=== Step3 vs Step5 Comparison ===")
    device = torch.device(a.device)
    df = compare_step3_vs_step5(model, wrappers, tok, a.bitwidths, device, n_eval=a.eval_n)
    
    # Save comparison results
    output_csv = "outputs/logs/step5_cpt_comparison.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_header = not os.path.exists(output_csv)
    df.insert(0, "timestamp", datetime.utcnow().isoformat())
    df.to_csv(output_csv, mode="a", index=False, header=write_header)
    logging.info(f"Comparison results saved to {output_csv}")
    
    # Print summary
    print("\n" + "="*60)
    print("STEP3 vs STEP5 COMPARISON SUMMARY")
    print("="*60)
    for bits in a.bitwidths:
        step3_row = df[df['config'] == f'step3_random_{bits}bit']
        step5_row = df[df['config'] == f'step5_cpt_{bits}bit']
        
        if not step3_row.empty and not step5_row.empty:
            step3_f1 = step3_row.iloc[0]['F1']
            step5_f1 = step5_row.iloc[0]['F1']
            improvement = step5_f1 - step3_f1
            print(f"{bits}-bit: Step3={step3_f1:.3f} | Step5={step5_f1:.3f} | Δ={improvement:+.3f}")


if __name__ == "__main__":
    main()