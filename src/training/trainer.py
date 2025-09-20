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
        lora_params = [
            p for w in wrappers.values() for p in w.bank.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(lora_params, lr=lr)
        
        # Setup device (CUDA-only)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training but was not found.")
        self.device = torch.device("cuda")
        self.model.to(self.device)

    def train(self, dataloader, iterations=1000):
        """Train with switchable precision for specified iterations."""
        self.model.train()
        
        # Use tqdm to create a progress bar
        progress_bar = tqdm(range(iterations))
        
        # Get data from dataloader
        data_iter = iter(dataloader)

        for i in progress_bar:
            # If data is used up, create a new iterator
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Randomly select a precision configuration
            chosen_config = random.choice(self.precision_configs)

            # Activate the corresponding LoRA branch
            self._activate_lora_for_config(chosen_config)

            # Training process
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_description(f"Iteration {i+1} | Loss: {loss.item():.3f} | Config: {chosen_config['default_w_bits']}-bit")

        print("\nTraining completed!")

    def _activate_lora_for_config(self, config):
        """Activate LoRA branches based on config."""
        # import here to avoid circular dependency
        from ..lora import activate_lora_by_bits
        
        per_layer_config = config.get('per_layer_bits', {})
        default_bits = config.get('default_w_bits')
        activate_lora_by_bits(self.wrappers, per_layer_config, default_bits)
