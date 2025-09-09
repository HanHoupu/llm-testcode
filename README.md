We use a YAML file to set quantization bit-widths.

Example:

model_name: gpt2
default_w_bits: 8
per_layer_bits:
  transformer.h.0.attn.c_attn: 4
  transformer.h.0.mlp.c_fc:    4


default_w_bits: default bit-width for all layers.

per_layer_bits: override for specific layers (layer names from model.named_modules()).

Run:

#to-do after final arrangement.