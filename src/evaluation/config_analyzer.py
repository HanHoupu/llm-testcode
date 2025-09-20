import glob
import os
import yaml
import pandas as pd
from .metrics import eval_config

# Direct copy from notebook - batch evaluation logic
def analyze_all_configs(model, tokenizer, val_ds, device, config_dir="../configs", n=100):
    configs = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    rows = []
    
    for p in configs:
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
        name = cfg.get("name", os.path.basename(p).replace(".yaml",""))
        res = eval_config(model, tokenizer, val_ds, device, cfg, n)
        rows.append({
            "config": name,
            "EM": res["exact_match"],
            "F1": res["f1"],
            "default_bits": cfg.get("default_w_bits","-")
        })
    
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
