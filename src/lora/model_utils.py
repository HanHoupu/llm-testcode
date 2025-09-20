import torch
import torch.nn as nn
from .core import LoRAWrapped

# Attach LoRA to quantized model (original method)
def attach_lora_to_quant(model, name2branches, quant_cfg):
    """LoRA and quant at the same time"""
    # import here to avoid circular bugs
    from ..quantization import replace_with_quant
    
    # quant first
    replace_with_quant(model, quant_cfg)
    
    # then add LoRA
    return attach_lora_to_model(model, name2branches)

# Attach LoRA to any model (decoupled version)
def attach_lora_to_model(model, name2branches):
    """Add LoRA wrappers to specified layers."""
    wrappers = {}
    for name, mod in list(model.named_modules()):
        if name in name2branches and hasattr(mod, 'in_features'):
            # handle naming alignment for .base suffix
            actual_name = name
            if hasattr(mod, 'base'):  # already wrapped (e.g. QuantLinear)
                actual_name = name + '.base'
            
            parent = model.get_submodule(name.rsplit('.',1)[0]) if '.' in name else model
            attr = name.split('.')[-1]
            w = LoRAWrapped(mod, name2branches[name], layer_name=actual_name)
            setattr(parent, attr, w)
            wrappers[name] = w
    return wrappers
