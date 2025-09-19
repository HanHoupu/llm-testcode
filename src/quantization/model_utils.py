# Model quantization utilities
import re
import torch
import torch.nn as nn
from .core import QuantLinear

# Check if layer should be quantized
def want_quant(name, mod, cfg):
    # skip embedding / norm / lm_head
    if name == "lm_head": 
        return False
    if isinstance(mod, nn.Linear) or mod.__class__.__name__ == "Conv1D":
        return True
    return False

# Change all quantized layers to new bit config
def requantize_model_to_config(model, cfg):
    default_bits = cfg.get('default_w_bits', 8)
    per_layer_bits = cfg.get('per_layer_bits', {})
    
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            target_bits = per_layer_bits.get(name, default_bits)
            module.requantize_to_bits(target_bits)

# Replace model layers with quantized versions
def replace_with_quant(model, cfg):
    name_to_module = dict(model.named_modules())
    for name, mod in list(name_to_module.items()):
        print(name, mod.__class__.__name__)
        if not want_quant(name, mod, cfg):
            continue

        # find parent module to replace child
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent = name_to_module[parent_name]
        else:
            parent, child_name = model, name

        # GPT2 uses Conv1D, convert to Linear first
        if mod.__class__.__name__ == "Conv1D":
                in_f, out_f = mod.weight.shape
                base = nn.Linear(in_f, out_f, bias=(mod.bias is not None))
                base.to(mod.weight.device, dtype=mod.weight.dtype)
                with torch.no_grad():
                    base.weight.copy_(mod.weight.T)  # transpose for Conv1D
                    if mod.bias is not None:
                        base.bias.copy_(mod.bias)
        else:
            base = mod

        qcfg = cfg.copy()
        qmod = QuantLinear.from_linear(base, name, cfg=qcfg)
        setattr(parent, child_name, qmod)
