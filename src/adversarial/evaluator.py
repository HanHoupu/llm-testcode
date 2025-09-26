import evaluate
import torch
from tqdm import tqdm

def evaluate_robustness(models, samples, attack_fn, device):
    metric = evaluate.load("squad")
    results = {}
    
    print(f"Evaluating {len(samples)} samples...")
    
    for model_name, model_info in models.items():
        print(f"\n=== {model_name.upper()} MODEL ===")
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        defense = model_info.get('defense', None)
        
        clean_preds, clean_refs = [], []
        adv_preds, adv_refs = [], []
        
        success_count = 0
        
        for i, sample in enumerate(tqdm(samples, desc=f"{model_name}")):
            question = sample['question']
            context = sample['context']
            answers = sample['answers']
            sample_id = sample['id']
            
            # Clean evaluation
            clean_pred = get_prediction(model, tokenizer, question, context, defense)
            clean_preds.append({"id": sample_id, "prediction_text": clean_pred})
            clean_refs.append({"id": sample_id, "answers": answers})
            
            # Generate adversarial sample
            try:
                adv_question = attack_fn(question, context, answers['text'][0])
                adv_pred = get_prediction(model, tokenizer, adv_question, context, defense)
                adv_preds.append({"id": sample_id, "prediction_text": adv_pred})
                adv_refs.append({"id": sample_id, "answers": answers})
                
                # Check attack success
                clean_f1 = compute_f1(clean_pred, answers['text'][0])
                adv_f1 = compute_f1(adv_pred, answers['text'][0])
                
                if clean_f1 > 0.4 and adv_f1 < 0.1:
                    success_count += 1
                    
            except Exception as e:
                print(f"Attack failed for sample {i}: {e}")
                # Clear CUDA cache on error to prevent state pollution
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                adv_preds.append({"id": sample_id, "prediction_text": clean_pred})
                adv_refs.append({"id": sample_id, "answers": answers})
        
        # Compute metrics
        clean_result = metric.compute(predictions=clean_preds, references=clean_refs)
        adv_result = metric.compute(predictions=adv_preds, references=adv_refs)
        
        success_rate = success_count / len(samples)
        
        results[model_name] = {
            "clean_f1": clean_result["f1"],
            "clean_em": clean_result["exact_match"],
            "adv_f1": adv_result["f1"],
            "adv_em": adv_result["exact_match"],
            "delta_f1": adv_result["f1"] - clean_result["f1"],
            "success_rate": success_rate
        }
        
        print(f"Clean F1: {clean_result['f1']:.3f}")
        print(f"Adv F1: {adv_result['f1']:.3f}")
        print(f"Success Rate: {success_rate:.3f}")
    
    return results

def get_prediction(model, tokenizer, question, context, defense=None):
    prompt = f"question: {question} context: {context} answer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    if defense:
        # Random defense - take majority vote
        predictions = defense.random_inference(inputs, tokenizer, num_runs=5)
        # Simple majority vote
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        prediction = max(pred_counts.keys(), key=lambda x: pred_counts[x])
    else:
        # Standard inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=16, 
                do_sample=False, 
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        gen_ids = outputs[0, inputs['input_ids'].size(1):]
        prediction = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        for sep in ["\n", ".", "?", "!"]:
            if sep in prediction:
                prediction = prediction.split(sep, 1)[0].strip()
                break
    
    return prediction

def compute_f1(pred, target):
    pred_tokens = pred.lower().split()
    target_tokens = target.lower().split()
    
    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        return 0.0
    
    common = set(pred_tokens) & set(target_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    
    return 2 * precision * recall / (precision + recall)

def filter_high_f1_samples(model, tokenizer, samples, threshold=0.3, max_samples=250):
    print(f"Filtering samples with F1 > {threshold}...")
    
    filtered = []
    sample_list = list(samples)
    
    for sample in tqdm(sample_list):
        question = sample['question']
        context = sample['context']
        answer = sample['answers']['text'][0]
        
        pred = get_prediction(model, tokenizer, question, context)
        f1 = compute_f1(pred, answer)
        
        if f1 > threshold:
            filtered.append(sample)
            
        if len(filtered) >= max_samples:
            break
    
    print(f"Found {len(filtered)} samples with F1 > {threshold}")
    return filtered
