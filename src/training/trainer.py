import torch
import random
from tqdm import tqdm

class SwitchableTrainer:
    """Trainer with switchable precision during training."""
    
    def __init__(self, model, wrappers, precision_configs, lr=1e-4):
        self.model = model
        self.wrappers = wrappers
        self.precision_configs = precision_configs
        
        # Collect all LoRA parameters
        lora_params = []
        for w in wrappers.values():
            for p in w.parameters():
                if p.requires_grad:
                    lora_params.append(p)
        self.optimizer = torch.optim.AdamW(lora_params, lr=lr)
        
        # Setup device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training but was not found.")
        self.device = torch.device("cuda")
        self.model.to(self.device)

    def train(self, dataloader, iterations=1000):
        """Train with switchable precision for specified iterations."""
        self.model.train()
        
        progress_bar = tqdm(range(iterations))
        data_iter = iter(dataloader)

        for i in progress_bar:
            # Refresh data iterator if exhausted
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Randomly select a precision configuration
            chosen_config = random.choice(self.precision_configs)
            config_name = chosen_config.get('name', f"default_{chosen_config.get('default_w_bits', '?')}")

            # Switch quantization bits and activate corresponding LoRA branch
            from ..quantization import requantize_model_to_config
            from ..lora import activate_lora_by_config
            
            # Only print detailed info every 100 iterations to reduce spam
            verbose = (i + 1) % 100 == 0 or i == 0
            if verbose:
                print(f"\n--- Iteration {i+1}: Switching to {config_name} ---")
            
            requantize_model_to_config(self.model, chosen_config, verbose=verbose)
            activate_lora_by_config(self.wrappers, chosen_config, verbose=verbose)

            # Forward pass
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**inputs)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_description(f"Iteration {i+1} | Loss: {loss.item():.3f} | Config: {config_name}")

        print("\nTraining completed!")

