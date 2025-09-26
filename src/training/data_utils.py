from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import torch

# Create SQuAD dataloader with answer-only supervision
def create_squad_dataloader(tokenizer, batch_size=16, subset_size=87599, seed=42):
    squad_dataset = load_dataset("squad", split="train")
    
    if subset_size >= len(squad_dataset):
        squad_subset = squad_dataset
    else:
        random.seed(seed)
        subset_indices = random.sample(range(len(squad_dataset)), subset_size)
        squad_subset = squad_dataset.select(subset_indices)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def collate_fn(batch):
        input_ids_list = []
        labels_list = []
        
        for sample in batch:
            prompt = f"question: {sample['question']} context: {sample['context']} answer: "
            answer = sample['answers']['text'][0]
            
            # Encode prompt and answer separately
            prompt_enc = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=100)
            answer_enc = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=20)
            
            # Combine input_ids
            input_ids = prompt_enc["input_ids"] + answer_enc["input_ids"]
            if tokenizer.eos_token_id:
                input_ids.append(tokenizer.eos_token_id)
            
            # Labels: -100 for prompt, actual tokens for answer
            labels = [-100] * len(prompt_enc["input_ids"]) + answer_enc["input_ids"]
            if tokenizer.eos_token_id:
                labels.append(tokenizer.eos_token_id)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        
        # Pad sequences
        max_len = max(len(seq) for seq in input_ids_list)
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            attention_masks.append([1] * len(input_ids) + [0] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "labels": torch.tensor(padded_labels)
        }

    return DataLoader(squad_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
