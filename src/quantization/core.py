import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter 

class QuantLinear(nn.Module):
    """Like nn.Linear but weights are int8 to save memory.
    Each output channel has its own scale factor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight",
            torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("w_scale",
            torch.ones(out_features, dtype=torch.float32, device=device))
        self.register_buffer("w_zp",
            torch.zeros(out_features, dtype=torch.int32, device=device))
        self.register_buffer("fp32_weight", None)  # orignal
        self.current_bits = None  # current 
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    # Save original weights for requantization
    def store_fp32_weight(self, weight: torch.Tensor):
        self.fp32_weight = weight.clone()

    # Change to different bit precision
    def requantize_to_bits(self, bits: int):
        if self.current_bits == bits:
            return
        
        self.quantize_from_float(self.fp32_weight, bits=bits)
        self.current_bits = bits

    def forward(self, input: Tensor) -> Tensor:
        if torch.any(self.w_zp != 0):
            # with zero points: (q - zp) * scale
            W = (self.qweight.int() - self.w_zp.view(-1, 1)).float() * self.w_scale.view(-1, 1)
        else:
            # simple case: q * scale
            W = self.qweight.float() * self.w_scale.view(-1, 1)
        return F.linear(input, W, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, dtype=int8, per_channel=True")
                
    # Get quantization bits for specific layer
    @staticmethod
    def get_bits_for_layer(name: str, cfg: dict) -> int:
        return cfg["per_layer_bits"].get(name, cfg.get("default_w_bits", 8))

    # Main quantization function: float32 -> int8
    def quantize_from_float(self, weight: torch.Tensor, bits: int = 8):
        # support 2-8 bits quantization
        qmin, qmax = -(2**(bits-1)), 2**(bits-1) - 1 
        # per-channel scaling
        w_max_abs = weight.abs().max(dim=1, keepdim=True)[0]
        w_max_abs = torch.clamp(w_max_abs, min=1e-8)
        scale = w_max_abs / qmax
        qweight = torch.clamp(torch.round(weight / scale), qmin, qmax).to(torch.int8)
        zero_point = torch.zeros(weight.size(0), dtype=torch.int32, device=weight.device)
        self.qweight.copy_(qweight)
        self.w_scale.copy_(scale.squeeze())
        self.w_zp.copy_(zero_point)

    # Create QuantLinear from regular Linear
    @classmethod
    def from_linear(cls, base: nn.Linear, name: str, cfg: dict):
        bits = cls.get_bits_for_layer(name, cfg)
        q = cls(base.in_features, base.out_features,
                bias=(base.bias is not None),
                device=base.weight.device, dtype=base.weight.dtype)
        with torch.no_grad():
            q.store_fp32_weight(base.weight)
            bits = cls.get_bits_for_layer(name, cfg)
            q.quantize_from_float(base.weight, bits=bits)
            q.current_bits = bits
            if base.bias is not None:
                q.bias.copy_(base.bias)
        return q
