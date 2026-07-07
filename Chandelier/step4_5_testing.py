import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from sentence_transformers import SentenceTransformer

# ==========================================
# 0. Environment Setup
# ==========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def sigmoid(t):
    return 1.0 / (1.0 + torch.exp(-t))

def mu_prime(t):
    s = sigmoid(t)
    return s * (1 - s)

# ==========================================
# 1. Networks Definitions
# ==========================================
class SimpleNet(nn.Module):
    """Used for both Green Density g_0 and Riesz Representer alpha."""
    def __init__(self, input_dim, hidden_dim=256):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ==========================================
# 2. Empirical Riesz Representer Solver
# ==========================================
def solve_riesz_representer(X_tensor, Y_tensor, target_x, target_y1, target_y2, d, epochs=250, lr=1e-3, weight_decay=1e-2, device="cuda"):
    """
    Stabilized Empirical Riesz problem solver with strict L2 regularization.
    """
    alpha_net = SimpleNet(input_dim=2*d).to(device)
    # Increased weight decay for stabilization
    optimizer = optim.Adam(alpha_net.parameters(), lr=lr, weight_decay=weight_decay)
    
    n = X_tensor.shape[0]
    m = Y_tensor.shape[1]
    
    X_flat = X_tensor.unsqueeze(1).expand(-1, m, -1).reshape(-1, d).to(device)
    Y_flat = Y_tensor.reshape(-1, d).to(device)
    
    target_x = target_x.to(device)
    target_y1 = target_y1.to(device)
    target_y2 = target_y2.to(device)
    
    alpha_net.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        alpha_flat = alpha_net(torch.cat([X_flat, Y_flat], dim=-1))
        design_norm = torch.mean(alpha_flat ** 2)
        
        s1 = torch.cat([target_x, target_y1]).unsqueeze(0)
        s2 = torch.cat([target_x, target_y2]).unsqueeze(0)
        
        eval_diff = alpha_net(s1) - alpha_net(s2)
        
        # Added an explicit ridge penalty on the evaluation difference to prevent exploding gradients
        ridge_penalty = 1e-3 * (eval_diff ** 2).mean()
        
        loss = design_norm - 2.0 * eval_diff + ridge_penalty
        
        loss.backward()
        # Gradient clipping to prevent sudden spikes
        torch.nn.utils.clip_grad_norm_(alpha_net.parameters(), max_norm=1.0)
        optimizer.step()
        
    alpha_net.eval()
    return alpha_net

# ==========================================
# 3. Debiased Score & Hypothesis Testing
# ==========================================
def run_pairwise_z_test(X_tensor, Y_tensor, r_tensor, g_net, test_idx, d, m, beta, device):
    """
    Computes the debiased test statistic Delta_hat and its preference variance.
    """
    # 1. Identify top answer and runner-up for the chosen prompt
    rewards_test = r_tensor[test_idx]
    top_two_idx = torch.topk(rewards_test, 2).indices
    a1_idx, a2_idx = top_two_idx[0].item(), top_two_idx[1].item()
    
    target_x = X_tensor[test_idx]
    target_y1 = Y_tensor[test_idx, a1_idx]
    target_y2 = Y_tensor[test_idx, a2_idx]
    
    # Raw DPO delta
    raw_delta = (rewards_test[a1_idx] - rewards_test[a2_idx]).item()
    
    # 2. Learn the optimal localized Riesz Representer alpha_Delta
    alpha_net = solve_riesz_representer(
        X_tensor, Y_tensor, target_x, target_y1, target_y2, d, device=device
    )
    
    # 3. Compute score components uniformly over the dataset samples
    n = X_tensor.shape[0]
    influence_values = []
    
    with torch.no_grad():
        for i in range(n):
            X_i = X_tensor[i].to(device)
            Y_all_i = Y_tensor[i].to(device)
            r_all_i = r_tensor[i].to(device)
            
            # Select 2 random candidate treatments as our active observations W_i
            idx1, idx2 = np.random.choice(m, 2, replace=False)
            Y1, Y2 = Y_all_i[idx1], Y_all_i[idx2]
            r1, r2 = r_all_i[idx1], r_all_i[idx2]
            
            # Simulated observed preference from implicit reward
            Z = torch.bernoulli(sigmoid(r1 - r2))
            
            # Evaluate alpha_Delta across candidates
            X_i_exp = X_i.unsqueeze(0).expand(m, -1)
            alpha_vals = alpha_net(torch.cat([X_i_exp, Y_all_i], dim=-1)) # shape: (m,)
            
            # Calculate exit rates
            # Clamp the derivative of the sigmoid to ensure stability in the denominator
            d_tilde = torch.mean(mu_prime(r_all_i.unsqueeze(1) - r_all_i), dim=1)
            d_tilde = torch.clamp(d_tilde, min=1e-2)
            
            # Assemble target Green density g_{alpha} (Eq 4.10)
            # Since rho = pi_0, density ratio is 1[cite: 1]
            g0_y1 = torch.mean(g_net(torch.cat([X_i_exp, Y1.unsqueeze(0).expand(m, -1), Y_all_i], dim=-1)) * alpha_vals)
            g0_y2 = torch.mean(g_net(torch.cat([X_i_exp, Y2.unsqueeze(0).expand(m, -1), Y_all_i], dim=-1)) * alpha_vals)
            
            g_alpha_y1 = (alpha_vals[idx1] / d_tilde[idx1]) + g0_y1
            g_alpha_y2 = (alpha_vals[idx2] / d_tilde[idx2]) + g0_y2
            
            # First linear part: \int alpha(X, z) r(X, z) \rho(z|X) dz[cite: 1]
            linear_part = torch.mean(alpha_vals * r_all_i).item()
            
            # Bias correction term[cite: 1]
            bias_correction = -0.5 * (g_alpha_y2 - g_alpha_y1) * (Z - sigmoid(r1 - r2))
            
            psi_f = linear_part - bias_correction.item()
            influence_values.append(psi_f)
            
        # Compute final debiased test statistic and variance[cite: 1]
        influence_values = np.array(influence_values)
        debiased_delta = np.mean(influence_values)
        
        sigma_sq = np.var(influence_values, ddof=1)
        sigma = np.sqrt(sigma_sq)
        
        # Calculate standard Z-test metric[cite: 1]
        z_score = debiased_delta / (sigma / np.sqrt(n))
        p_value = 1.0 - norm.cdf(z_score)
        
    return raw_delta, debiased_delta, z_score, p_value

# ==========================================
# 4. Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4 & 5: Pointwise Inference & Hypothesis Testing")
    parser.add_argument("--in_file", type=str, default="./llm_pref_data/step2_rewards.json", help="Input JSON from Step 2")
    parser.add_argument("--weights_file", type=str, default="./llm_pref_data/g_net_weights.pt", help="Weights from Step 3")
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model for mapping")
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty parameter")
    parser.add_argument("--test_prompts", type=int, default=5, help="Number of random prompts to test")
    args = parser.parse_args()

    device = get_device()
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading augmented preference dataset...")
    with open(args.in_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Re-encode prompts to match continuous dimensions[cite: 1]
    print(f"Re-mapping prompt vectors using {args.embed_model}...")
    encoder = SentenceTransformer(args.embed_model, device=device)
    prompt_embeddings = encoder.encode(df["prompt_text"].tolist(), convert_to_numpy=True, normalize_embeddings=True)
    
    X_tensor = torch.tensor(prompt_embeddings, dtype=torch.float32)
    Y_tensor = torch.tensor(df["candidate_embeddings"].tolist(), dtype=torch.float32)
    r_tensor = torch.tensor(df["implicit_rewards"].tolist(), dtype=torch.float32)
    
    d = X_tensor.shape[1]
    m = Y_tensor.shape[1]
    
    # Load trained Green density network
    g_net = SimpleNet(input_dim=3*d).to(device)
    g_net.load_state_dict(torch.load(args.weights_file, map_location=device))
    g_net.eval()
    
    # Select sample prompts to display top answer inference
    test_indices = np.random.choice(len(df), args.test_prompts, replace=False)
    
    print("\n" + "="*50)
    print("RUNNING pairwise TESTING FOR TOP LLM GENERATIONS")
    print("="*50)
    
    results = []
    for t_idx in test_indices:
        prompt_txt = df.iloc[t_idx]["prompt_text"][:60] + "..."
        
        raw_d, debiased_d, z_val, p_val = run_pairwise_z_test(
            X_tensor, Y_tensor, r_tensor, g_net, t_idx, d, m, args.beta, device
        )
        
        print(f"\nPrompt ID {t_idx}: {prompt_txt}")
        print(f"  -> Raw Reward Margin (a1 - a2): {raw_d:.5f}")
        print(f"  -> Debiased Margin (Delta_hat):  {debiased_d:.5f}")
        print(f"  -> Calculated Z-Score:          {z_val:.4f}")
        print(f"  -> One-Sided P-Value:           {p_val:.5f}")
        
        if p_val < 0.05:
            print("  STATUS: [REJECT NULL] Top generation is statistically superior.")
        else:
            print("  STATUS: [ACCEPT NULL] Reward gap is not statistically significant.")