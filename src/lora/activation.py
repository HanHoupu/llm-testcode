def create_bit_mapping(bits_list):
    """Auto-generate bit-width to branch name mapping."""
    return {b: f"bw{b}" for b in bits_list}

# Original activation method (backward compatible)
def activate_lora_by_bits(wrappers, bit_cfg, default_bits=None):
    """Activate LoRA branches based on bit configuration."""
    m = {4: "bw4", 8: "bw8"}  # original mapping
    for n, w in wrappers.items():
        bw = bit_cfg.get(n, default_bits)
        branch_name = m.get(bw)
        w.set_active(branch_name)

# Enhanced activation with auto bit mapping
def activate_lora_by_config(wrappers, cfg, supported_bits=[2, 4, 6, 8]):
    """Enhanced activation supporting 2-8 bits with safe fallback."""
    bit_map = create_bit_mapping(supported_bits)
    default_bits = cfg.get('default_w_bits', 8)
    per_layer_bits = cfg.get('per_layer_bits', {})
    
    for name, wrapper in wrappers.items():
        target_bits = per_layer_bits.get(name, default_bits)
        branch_name = bit_map.get(target_bits)
        
        # safe activation with fallback
        if branch_name not in wrapper.bank:
            # fallback to closest available branch
            available_bits = [int(b.replace('bw', '')) for b in wrapper.bank.keys() if b.startswith('bw')]
            if available_bits:
                closest_bits = min(available_bits, key=lambda x: abs(x - target_bits))
                fallback_branch = f"bw{closest_bits}"
                print(f"Warning: {name} fallback from {target_bits}bit to {closest_bits}bit")
                wrapper.set_active(fallback_branch)
            else:
                wrapper.set_active(None)
        else:
            wrapper.set_active(branch_name)
