import torch
import torch.nn.functional as F
import random
from tqdm import tqdm

class EmbeddingPGD:
    def __init__(self, model, tokenizer, epsilon=0.1, steps=8):
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = epsilon / steps
        
    def attack(self, question, context, answer):
        # Clear GPU cache before attack
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        prompt = f"question: {question} context: {context} answer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        embeddings = embeddings.clone().detach().requires_grad_(True)
        
        # PGD iterations
        for step in range(self.steps):
            embeddings.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(inputs_embeds=embeddings)
            logits = outputs.logits
            
            # Compute loss (maximize prediction uncertainty)
            loss = -F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                   inputs['input_ids'].view(-1), 
                                   ignore_index=self.tokenizer.pad_token_id)
            
            # Backward pass
            loss.backward()
            
            # Update embeddings
            grad = embeddings.grad.data
            embeddings.data += self.alpha * grad.sign()
            
            # Project to L-inf ball
            embeddings.data = torch.clamp(embeddings.data, 
                                        embeddings.data - self.epsilon,
                                        embeddings.data + self.epsilon)
            
            if embeddings.grad is not None:
                embeddings.grad.zero_()
        
        # Map back to tokens with safety check
        adv_tokens = self._embed_to_tokens(embeddings)
        
        # Ensure all tokens are within valid range
        vocab_size = self.tokenizer.vocab_size
        adv_tokens = torch.clamp(adv_tokens, 0, vocab_size - 1)
        
        adv_prompt = self.tokenizer.decode(adv_tokens[0], skip_special_tokens=True)
        
        # Extract question part
        if "answer:" in adv_prompt:
            adv_question = adv_prompt.split("answer:")[0].replace("question:", "").replace("context:", "").strip()
        else:
            adv_question = question  # fallback
            
        return adv_question
    
    def _embed_to_tokens(self, embeddings):
        vocab_embeddings = self.model.get_input_embeddings().weight
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        # Memory-efficient token mapping - process one position at a time
        best_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long, device=embeddings.device)
        
        for b in range(batch_size):
            for s in range(seq_len):
                # Process one embedding at a time to save memory
                emb = embeddings[b, s]  # [hidden_dim]
                
                # Compute similarities in chunks to avoid OOM
                chunk_size = 15000  # Conservative setting for stability
                best_sim = -1
                best_token = 0
                
                for start_idx in range(0, vocab_embeddings.size(0), chunk_size):
                    end_idx = min(start_idx + chunk_size, vocab_embeddings.size(0))
                    vocab_chunk = vocab_embeddings[start_idx:end_idx]  # [chunk_size, hidden_dim]
                    
                    # Compute similarity for this chunk
                    sims = F.cosine_similarity(emb.unsqueeze(0), vocab_chunk, dim=1)
                    chunk_max_sim, chunk_max_idx = sims.max(0)
                    
                    if chunk_max_sim > best_sim:
                        best_sim = chunk_max_sim
                        best_token = start_idx + chunk_max_idx.item()
                
                # Ensure token is within valid range
                best_token = min(best_token, vocab_embeddings.size(0) - 1)
                best_token = max(best_token, 0)
                best_tokens[b, s] = best_token
        
        return best_tokens

class HotFlip:
    def __init__(self, model, tokenizer, candidate_k=200):
        self.model = model
        self.tokenizer = tokenizer
        self.candidate_k = candidate_k
        
    def attack(self, question, context, answer, budget=1):
        # Clear GPU cache before attack
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        prompt = f"question: {question} context: {context} answer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get token gradients
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        embeddings = embeddings.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs_embeds=embeddings)
        logits = outputs.logits
        
        # Compute loss
        loss = -F.cross_entropy(logits.view(-1, logits.size(-1)), 
                               inputs['input_ids'].view(-1), 
                               ignore_index=self.tokenizer.pad_token_id)
        loss.backward()
        
        # Get gradients with safety check
        if embeddings.grad is not None:
            grad = embeddings.grad.data
            grad_norm = grad.norm(dim=-1)
        else:
            print("Warning: No gradients found, using random positions")
            grad_norm = torch.randn(embeddings.shape[:2], device=embeddings.device)
        
        # Find top positions to modify
        question_tokens = self.tokenizer(question, return_tensors="pt")['input_ids'][0]
        question_len = len(question_tokens)
        
        # Focus on question part only
        question_grad = grad_norm[0, :question_len]
        top_positions = question_grad.topk(min(budget, question_len)).indices
        
        # Generate candidates for each position
        adv_question = question
        for pos in top_positions:
            candidates = self._get_candidates(question_tokens[pos].item())
            best_token = self._select_best_candidate(
                inputs, embeddings, pos, candidates
            )
            
            # Replace token
            new_tokens = question_tokens.clone()
            new_tokens[pos] = best_token
            adv_question = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        return adv_question
    
    def _get_candidates(self, original_token):
        # Simple candidate selection - nearby vocab indices
        vocab_size = self.tokenizer.vocab_size
        candidates = []
        
        # Add nearby tokens with bounds checking
        for offset in range(-50, 51):
            candidate = original_token + offset
            # Ensure candidate is within valid range
            if 0 <= candidate < vocab_size:
                candidates.append(candidate)
        
        # Add some random candidates
        for _ in range(min(100, self.candidate_k - len(candidates))):
            candidates.append(random.randint(0, vocab_size - 1))
            
        return candidates[:self.candidate_k]
    
    def _select_best_candidate(self, inputs, embeddings, position, candidates):
        best_token = inputs['input_ids'][0, position].item()
        best_loss = float('-inf')
        device = embeddings.device
        
        for candidate in candidates:
            # Try this candidate
            test_embeddings = embeddings.clone()
            candidate_embed = self.model.get_input_embeddings()(torch.tensor([candidate]).to(device))
            test_embeddings[0, position] = candidate_embed
            
            # Compute loss
            with torch.no_grad():
                outputs = self.model(inputs_embeds=test_embeddings)
                logits = outputs.logits
                loss = -F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                       inputs['input_ids'].view(-1), 
                                       ignore_index=self.tokenizer.pad_token_id)
                
                if loss > best_loss:
                    best_loss = loss
                    best_token = candidate
                    
        return best_token
