import torch
import evaluate
from tqdm import tqdm

# Direct copy from notebook - eval_config function
def eval_config(model, tokenizer, val_ds, device, cfg, n=100):
    from ..quantization import requantize_model_to_config
    from ..lora import activate_lora_by_bits
    
    requantize_model_to_config(model, cfg)
    # activate_lora_by_bits if wrappers exist
    
    metric = evaluate.load("squad")
    preds, refs = [], []
    for ex in tqdm(val_ds.select(range(min(n, len(val_ds)))), leave=False):
        prompt = f"question: {ex['question']} context: {ex['context']} answer:"
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
        gen_ids = out[0, inp['input_ids'].size(1):]
        ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        preds.append({"id": ex["id"], "prediction_text": ans})
        refs.append({"id": ex["id"], "answers": ex["answers"]})
    return metric.compute(predictions=preds, references=refs)

# Monitor training metrics for Step 5
def monitor_training_metrics(model, loss, iteration, config_name):
    """Record training metrics for cyclic training analysis."""
    return {
        "iteration": iteration,
        "loss": loss.item(),
        "config": config_name,
        "memory_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    }

# Adversarial robustness evaluation for Step 6
def evaluate_adversarial_robustness(model, tokenizer, attack_samples, device):
    """Evaluate model robustness against adversarial attacks."""
    correct = 0
    total = len(attack_samples)
    
    for sample in tqdm(attack_samples, desc="Testing robustness"):
        inp = tokenizer(sample["input"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10)
        pred = tokenizer.decode(out[0, inp['input_ids'].size(1):], skip_special_tokens=True)
        
        if sample["expected"] in pred.lower():
            correct += 1
    
    return {"accuracy": correct / total, "total_samples": total}
