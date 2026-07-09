import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ==========================================
# 0. Environment & Architecture Setup
# ==========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ==========================================
# 1. Dataset Loading & Complexity Filtering
# ==========================================
def load_and_filter_tldr(min_length=1500, max_samples=500):
    print("Loading CarperAI/openai_summarize_tldr dataset...")
    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
    
    filtered_data = [
        item for item in dataset 
        if len(item['prompt']) >= min_length
    ]
    
    np.random.shuffle(filtered_data)
    selected_data = filtered_data[:max_samples]
    
    print(f"Filtered down to {len(selected_data)} complex prompts.")
    return selected_data

# ==========================================
# 2. Treatment Generation (Chunked for OOM Prevention)
# ==========================================
def generate_candidate_treatments(prompts, model_name, num_candidates=50, max_concurrent=10, device="cuda"):
    """
    Uses the reference policy pi_0 to sample candidate answers.
    Chunks the generation (max_concurrent) to prevent VRAM explosion for large k.
    """
    print(f"Loading reference policy (pi_0): {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    all_candidates = []
    
    print(f"Sampling {num_candidates} candidate treatments per prompt (chunked by {max_concurrent})...")
    for p in tqdm(prompts):
        prompt_text = p['prompt']
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        prompt_len = len(prompt_text)
        
        candidates = []
        remaining = num_candidates
        
        while remaining > 0:
            current_k = min(remaining, max_concurrent)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=current_k,
                    pad_token_id=tokenizer.pad_token_id
                )
                
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for text in decoded_outputs:
                # Extract just the generated summary
                candidates.append(text[prompt_len:].strip())
                
            remaining -= current_k
            
        all_candidates.append(candidates)
            
    # Free up VRAM before embedding
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return all_candidates

# ==========================================
# 3. Continuous Mapping (Embedding into R^d)
# ==========================================
def embed_treatments(all_candidates, embedding_model_name, device="cuda"):
    print(f"Loading continuous mapping model: {embedding_model_name}...")
    encoder = SentenceTransformer(embedding_model_name, device=device)
    
    all_embeddings = []
    print("Projecting discrete treatments into continuous space...")
    
    for candidates in tqdm(all_candidates):
        embeddings = encoder.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(embeddings)
        
    return all_embeddings

# ==========================================
# 4. Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Data Prep and Continuous Mapping")
    parser.add_argument("--out_dir", type=str, default="./llm_pref_data", help="Output directory")
    parser.add_argument("--samples", type=int, default=200, help="Number of prompts to process")
    parser.add_argument("--k_candidates", type=int, default=10, help="Treatments per prompt")
    parser.add_argument("--pi_0_model", type=str, default="HuggingFaceH4/mistral-7b-sft-beta", help="Base reference model")
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model for mapping")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    
    np.random.seed(42)
    torch.manual_seed(42)

    prompts = load_and_filter_tldr(max_samples=args.samples)
    
    # max_concurrent=10 restricts the GPU to processing 10 sequences at a time, protecting VRAM
    discrete_treatments = generate_candidate_treatments(
        prompts, 
        model_name=args.pi_0_model, 
        num_candidates=args.k_candidates, 
        max_concurrent=10, 
        device=device
    )
    
    continuous_treatments = embed_treatments(
        discrete_treatments, 
        embedding_model_name=args.embed_model, 
        device=device
    )
    
    print("Structuring and saving dataset...")
    structured_data = []
    for i, p in enumerate(prompts):
        structured_data.append({
            "prompt_id": i,
            "prompt_text": p['prompt'],
            "reference_summary": p['label'],
            "candidate_texts": discrete_treatments[i],
            "candidate_embeddings": continuous_treatments[i].tolist() 
        })
        
    df = pd.DataFrame(structured_data)
    out_path = os.path.join(args.out_dir, "step1_treatments.json")
    df.to_json(out_path, orient="records", indent=4)
    
    print(f"Step 1 Complete! Continuous treatment data saved to {out_path}.")