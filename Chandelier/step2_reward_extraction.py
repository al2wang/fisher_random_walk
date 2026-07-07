import argparse
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 0. Environment Setup
# ==========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ==========================================
# 1. Log-Probability Computation
# ==========================================
def compute_sequence_log_probs(model, tokenizer, prompt, candidate_text, device):
    """
    Computes the log-probability of the candidate_text conditioned on the prompt.
    """
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    full_text = prompt + candidate_text
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    prompt_len = prompt_inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits
        
    # Shift logits and labels for next-token prediction
    shift_logits = logits[0, :-1, :].contiguous()
    shift_labels = full_inputs["input_ids"][0, 1:].contiguous()
    
    # Calculate token-wise log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Isolate the log-prob of the generated candidate (shift index by 1)
    candidate_log_probs = token_log_probs[prompt_len - 1:]
    
    # Sum token log-probs to get the full sequence log-prob
    sequence_log_prob = candidate_log_probs.sum().item()
    return sequence_log_prob

# ==========================================
# 2. Model Loading & Scoring
# ==========================================
def score_candidates_with_model(data_df, model_name, device):
    """
    Loads a causal LM and computes sequence log-probs for all candidates in the dataset.
    """
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    
    all_log_probs = []
    
    print(f"Computing log-probabilities under {model_name}...")
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
        prompt = row["prompt_text"]
        candidates = row["candidate_texts"]
        
        row_log_probs = []
        for candidate in candidates:
            # Mistral chat template formatting
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            lp = compute_sequence_log_probs(model, tokenizer, formatted_prompt, candidate, device)
            row_log_probs.append(lp)
            
        all_log_probs.append(row_log_probs)
        
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return all_log_probs

# ==========================================
# 3. Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Implicit Reward Extraction")
    parser.add_argument("--in_file", type=str, default="./llm_pref_data/step1_treatments.json", help="Input JSON from Step 1")
    parser.add_argument("--out_file", type=str, default="./llm_pref_data/step2_rewards.json", help="Output JSON with rewards")
    parser.add_argument("--pi_0_model", type=str, default="HuggingFaceH4/mistral-7b-sft-beta", help="Base reference model")
    parser.add_argument("--pi_star_model", type=str, default="yale-nlp/mistral-7b-dpo-beta-0.01", help="DPO fine-tuned optimal model")
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty parameter from DPO training")
    args = parser.parse_args()

    device = get_device()
    
    print(f"Loading data from {args.in_file}...")
    with open(args.in_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # 1. Compute log-probabilities under the reference policy (pi_0)
    pi_0_log_probs = score_candidates_with_model(df, args.pi_0_model, device)
    
    # 2. Compute log-probabilities under the optimal policy (pi_star)
    pi_star_log_probs = score_candidates_with_model(df, args.pi_star_model, device)
    
    # 3. Calculate Implicit DPO Rewards
    print("\nCalculating implicit rewards r*(x,y) = beta * (log pi* - log pi_0)...")
    all_rewards = []
    
    for i in range(len(df)):
        row_rewards = []
        for j in range(len(pi_0_log_probs[i])):
            lp_0 = pi_0_log_probs[i][j]
            lp_star = pi_star_log_probs[i][j]
            
            # The exact closed-form DPO reward mapping
            r_star = args.beta * (lp_star - lp_0)
            row_rewards.append(r_star)
            
        all_rewards.append(row_rewards)
        
    df["pi_0_log_probs"] = pi_0_log_probs
    df["pi_star_log_probs"] = pi_star_log_probs
    df["implicit_rewards"] = all_rewards
    
    df.to_json(args.out_file, orient="records", indent=4)
    print(f"\nStep 2 Complete! Dataset with implicit rewards saved to {args.out_file}.")