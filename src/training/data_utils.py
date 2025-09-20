from datasets import load_dataset
from torch.utils.data import DataLoader
import random

# Format SQuAD sample to prompt
def format_squad_prompt(sample):
    return f"question: {sample['question']} context: {sample['context']} answer: {sample['answers']['text'][0]}"

# Create SQuAD dataloader
def create_squad_dataloader(tokenizer, batch_size=4, subset_size=1000):
    # Load dataset
    squad_dataset = load_dataset("squad", split="train")
    
    # Create a small subset
    subset_indices = random.sample(range(len(squad_dataset)), subset_size)
    squad_subset = squad_dataset.select(subset_indices)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # DataLoader
    def collate_fn(batch):
        prompts = [format_squad_prompt(s) for s in batch]
        return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    return DataLoader(squad_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
