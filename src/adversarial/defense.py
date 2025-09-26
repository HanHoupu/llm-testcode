import random
import torch

class RandomBitwidthDefense:
    def __init__(self, model, wrappers, bitwidths=[4, 6, 8]):
        self.model = model
        self.wrappers = wrappers  
        self.bitwidths = bitwidths
    
    def random_inference(self, inputs, tokenizer, num_runs=5, gen_args=None):
        """Random bitwidth, support custom generation arguments"""
        if gen_args is None:
            gen_args = {"max_new_tokens": 16, "do_sample": False, "num_beams": 1}
        
        results = []
        
        for run in range(num_runs):
            # Random bitwidth config
            chosen_bits = random.choice(self.bitwidths)
            config = {"default_w_bits": chosen_bits, "per_layer_bits": {}}
            
            # Apply config
            from ..quantization import requantize_model_to_config
            from ..lora import activate_lora_by_config
            requantize_model_to_config(self.model, config)
            if self.wrappers:
                activate_lora_by_config(self.wrappers, config)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    pad_token_id=tokenizer.eos_token_id,
                    **gen_args
                )
            
            # Decode answer
            gen_ids = outputs[0, inputs['input_ids'].size(1):]
            answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            # Clean up answer
            for sep in ["\n", ".", "?", "!"]:
                if sep in answer:
                    answer = answer.split(sep, 1)[0].strip()
                    break
                    
            results.append(answer)
            
        return results
    
    def static_inference(self, inputs, tokenizer, bits=8):
        # Static quantization inference
        config = {"default_w_bits": bits, "per_layer_bits": {}}
        
        from ..quantization import requantize_model_to_config
        from ..lora import activate_lora_by_config
        requantize_model_to_config(self.model, config)
        if self.wrappers:
            activate_lora_by_config(self.wrappers, config)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=16, 
                do_sample=False, 
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        gen_ids = outputs[0, inputs['input_ids'].size(1):]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        for sep in ["\n", ".", "?", "!"]:
            if sep in answer:
                answer = answer.split(sep, 1)[0].strip()
                break
                
        return answer
