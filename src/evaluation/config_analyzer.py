import glob
import os
import yaml
import pandas as pd
import torch
import random
from .metrics import eval_config

# Enhanced batch evaluation with fresh model for each config
def analyze_all_configs(model_factory, tokenizer, val_ds, device, config_dir="../configs", n=100, checkpoint_path=None):
    """
    Analyze all configs with fresh model state for each config.
    
    Args:
        model_factory: Function that returns (model, wrappers) tuple
        tokenizer: Tokenizer instance
        val_ds: Validation dataset
        device: Device to use
        config_dir: Directory containing config files
        n: Number of samples to evaluate
        checkpoint_path: Path to LoRA checkpoint
    """
    configs = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    rows = []
    
    # Pre-sample fixed evaluation samples for all configs (consistency)
    print(f"📊 Pre-sampling {n} fixed evaluation samples for all configs...")
    random.seed(42)  # Fixed seed for reproducibility
    if n < len(val_ds):
        fixed_indices = random.sample(range(len(val_ds)), n)
        fixed_samples = val_ds.select(fixed_indices)
        print(f"📊 Using {len(fixed_samples)} fixed samples from {len(val_ds)} total")
    else:
        fixed_samples = val_ds
        print(f"📊 Using full dataset: {len(val_ds)} samples")
    
    for i, p in enumerate(configs):
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
        name = cfg.get("name", os.path.basename(p).replace(".yaml",""))
        
        print(f"\n=== Evaluating config {i+1}/{len(configs)}: {name} ===")
        
        # Create fresh model and wrappers for each config
        model, wrappers = model_factory(checkpoint_path)
        model.to(device)
        model.eval()
        
        # Clear any cached states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        res = eval_config(model, tokenizer, fixed_samples, device, cfg, wrappers, len(fixed_samples), config_name=name)
        rows.append({
            "config": name,
            "EM": res["exact_match"],
            "F1": res["f1"],
            "default_bits": cfg.get("default_w_bits","-")
        })
        
        # Clean up model to free memory
        del model, wrappers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return df

# Find optimal config for accuracy-efficiency trade-off
def find_optimal_config(results, criteria="f1"):
    """Find best config based on criteria."""
    if not results:
        return None
    
    if criteria == "f1":
        best = max(results, key=lambda x: x["F1"])
    elif criteria == "em":
        best = max(results, key=lambda x: x["EM"])
    else:
        best = results[0]
    
    return best

# Generate insights for report writing
def generate_insights(results):
    """Generate insights for Step 4 report questions."""
    if not results:
        return "No results to analyze."
    
    df = pd.DataFrame(results)
    best_f1 = df.loc[df["F1"].idxmax()]
    
    insights = []
    insights.append(f"Best F1: {best_f1['config']} ({best_f1['F1']:.3f})")
    
    # Bit-width analysis for report
    if "default_bits" in df.columns:
        bit_groups = df.groupby("default_bits")["F1"].mean()
        insights.append(f"Average F1 by bits: {dict(bit_groups)}")
    
    return "\n".join(insights)
