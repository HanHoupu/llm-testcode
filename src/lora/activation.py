def create_bit_mapping(bits_list):
    """Auto-generate bit-width to branch name mapping."""
    return {b: f"bw{b}" for b in bits_list}

# Legacy activation method for 4/8-bit configurations
def activate_lora_by_bits(wrappers, bit_cfg, default_bits=None):
    """Activate LoRA branches based on bit configuration."""
    m = {4: "bw4", 8: "bw8"}
    for n, w in wrappers.items():
        bw = bit_cfg.get(n, default_bits)
        branch_name = m.get(bw)
        w.set_active(branch_name)

# Strict activation supporting multiple bit-widths
def activate_lora_by_config(wrappers, cfg, supported_bits=[4, 6, 8], verbose=True, config_name=None):
    """Strict activation supporting 4/6/8 bits - no fallback, fail fast."""
    bit_map = create_bit_mapping(supported_bits)
    default_bits = cfg.get('default_w_bits', 8)
    per_layer_bits = cfg.get('per_layer_bits', {})
    
    # Count activations for summary
    activation_count = {}
    
    for name, wrapper in wrappers.items():
        target_bits = per_layer_bits.get(name, default_bits)
        branch_name = bit_map.get(target_bits)
        
        # Strict activation - no fallback
        if branch_name not in wrapper.bank:
            raise ValueError(f"LoRA branch '{branch_name}' not found for layer '{name}'. "
                           f"Available branches: {list(wrapper.bank.keys())}")
        
        wrapper.set_active(branch_name)
        activation_count[branch_name] = activation_count.get(branch_name, 0) + 1
    
    # Print summary with config name if verbose
    if verbose:
        config_info = f" [{config_name}]" if config_name else ""
        print(f" LoRA activated{config_info}: {dict(activation_count)}")
