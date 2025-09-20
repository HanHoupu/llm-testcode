import json

# Simple result saver
def save_results(results, filename="results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

# Log cyclic training metrics for Step 5
def log_cyclic_training(iteration, loss, config_name, cycle_num):
    """Log cyclic training progress for CPT comparison."""
    return {
        "iteration": iteration,
        "loss": loss,
        "config": config_name,
        "cycle": cycle_num
    }

# Log adversarial results for Step 6  
def log_adversarial_results(attack_name, accuracy, config_name):
    """Log adversarial robustness results for Double-Win Quant comparison."""
    return {
        "attack": attack_name,
        "accuracy": accuracy,
        "config": config_name
    }

# Export report data
def export_report_data(all_results, filename="report_data.json"):
    """Export structured data for report writing."""
    report_data = {
        "step4_configs": all_results.get("step4", []),
        "step5_cyclic": all_results.get("step5", []),
        "step6_adversarial": all_results.get("step6", [])
    }
    
    with open(filename, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Report data exported to {filename}")
