import torch
import torch.nn as nn

class LoRA(nn.Module):
    """Basic LoRA implementation with A and B matrices."""
    def __init__(self, in_f, out_f, r=4, alpha=None):
        super().__init__()
        self.scale = (alpha or r) / r
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, r))
    
    def forward(self, x):
        return (x @ self.A.t()) @ self.B.t() * self.scale

class LoRAWrapped(nn.Module):
    """Wrapper that manages multiple LoRA branches for different bit-widths."""
    def __init__(self, base, branches, layer_name=None):
        super().__init__()
        self.base = base
        # freeze base parameters
        for p in self.base.parameters():
            p.requires_grad = False

        # keep device and precision
        dev = next(self.base.parameters()).device
        dtype = next(self.base.parameters()).dtype

        # shape validation for Conv1D compatibility
        if hasattr(base, 'in_features') and hasattr(base, 'out_features'):
            in_f = self.base.in_features
            out_f = self.base.out_features
        else:
            raise ValueError(f"Base module {base.__class__.__name__} must have in_features/out_features")

        self.bank = nn.ModuleDict({
            k: LoRA(in_f, out_f, r, a).to(device=dev, dtype=dtype)
            for k, (r, a) in branches.items()
        })
        self.active = None  # only one branch active
        self.layer_name = layer_name

    def set_active(self, name_or_none):
        # safe activation with fallback
        if name_or_none is not None and name_or_none not in self.bank:
            print(f"Warning: LoRA branch '{name_or_none}' not found in {self.layer_name}, using None")
            self.active = None
        else:
            self.active = name_or_none

    def forward(self, x):
        y = self.base(x)
        if self.active in self.bank:
            y = y + self.bank[self.active](x)
        return y
