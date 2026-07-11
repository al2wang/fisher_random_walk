import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sentence_transformers import SentenceTransformer

# ==========================================
# 0. Setup & Networks
# ==========================================
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def sigmoid(t): return 1.0 / (1.0 + torch.exp(-t))
def mu_prime(t): s = sigmoid(t); return s * (1 - s)

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def setup_neurips_style():
    try:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    except AttributeError:
        sns.set(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({
        "font.family": "serif", "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight"
    })

# ==========================================
# 1. Riesz Representer (Re-used for Target)
# ==========================================
def solve_riesz_representer(X_tensor, Y_tensor, target_x, target_y1, target_y2, d, epochs=250, lr=1e-3, weight_decay=1e-2, device="cuda"):
    alpha_net = SimpleNet(input_dim=2*d).to(device)
    optimizer = optim.Adam(alpha_net.parameters(), lr=lr, weight_decay=weight_decay)
    
    n, m = X_tensor.shape[0], Y_tensor.shape[1]
    X_flat = X_tensor.unsqueeze(1).expand(-1, m, -1).reshape(-1, d).to(device)
    Y_flat = Y_tensor.reshape(-1, d).to(device)
    target_x, target_y1, target_y2 = target_x.to(device), target_y1.to(device), target_y2.to(device)
    
    alpha_net.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        alpha_flat = alpha_net(torch.cat([X_flat, Y_flat], dim=-1))
        design_norm = torch.mean(alpha_flat ** 2)
        
        s1 = torch.cat([target_x, target_y1]).unsqueeze(0)
        s2 = torch.cat([target_x, target_y2]).unsqueeze(0)
        eval_diff = alpha_net(s1) - alpha_net(s2)
        
        ridge_penalty = 1e-3 * (eval_diff ** 2).mean()
        loss = design_norm - 2.0 * eval_diff + ridge_penalty
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(alpha_net.parameters(), max_norm=1.0)
        optimizer.step()
        
    alpha_net.eval()
    return alpha_net

# ==========================================
# 2. Extracting Influence Functions
# ==========================================
def extract_influence_functions(X_tensor, Y_tensor, r_tensor, g_net, target_idx, d, m, beta, device):
    rewards_test = r_tensor[target_idx]
    top_two_idx = torch.topk(rewards_test, 2).indices
    a1_idx, a2_idx = top_two_idx[0].item(), top_two_idx[1].item()
    
    target_x = X_tensor[target_idx]
    target_y1 = Y_tensor[target_idx, a1_idx]
    target_y2 = Y_tensor[target_idx, a2_idx]
    
    print(f"Solving Riesz representer for Prompt ID {target_idx}...")
    alpha_net = solve_riesz_representer(X_tensor, Y_tensor, target_x, target_y1, target_y2, d, device=device)
    
    n = X_tensor.shape[0]
    influence_values = np.zeros(n)
    
    print("Computing doubly robust influence array...")
    with torch.no_grad():
        for i in range(n):
            X_i = X_tensor[i].to(device)
            Y_all_i = Y_tensor[i].to(device)
            r_all_i = r_tensor[i].to(device)
            
            idx1, idx2 = np.random.choice(m, 2, replace=False)
            Y1, Y2 = Y_all_i[idx1], Y_all_i[idx2]
            r1, r2 = r_all_i[idx1], r_all_i[idx2]
            
            Z = torch.bernoulli(sigmoid(r1 - r2))
            
            X_i_exp = X_i.unsqueeze(0).expand(m, -1)
            alpha_vals = alpha_net(torch.cat([X_i_exp, Y_all_i], dim=-1))
            
            d_tilde = torch.mean(mu_prime(r_all_i.unsqueeze(1) - r_all_i), dim=1)
            d_tilde = torch.clamp(d_tilde, min=1e-2)
            
            g0_y1 = torch.mean(g_net(torch.cat([X_i_exp, Y1.unsqueeze(0).expand(m, -1), Y_all_i], dim=-1)) * alpha_vals)
            g0_y2 = torch.mean(g_net(torch.cat([X_i_exp, Y2.unsqueeze(0).expand(m, -1), Y_all_i], dim=-1)) * alpha_vals)
            
            g_alpha_y1 = (alpha_vals[idx1] / d_tilde[idx1]) + g0_y1
            g_alpha_y2 = (alpha_vals[idx2] / d_tilde[idx2]) + g0_y2
            
            linear_part = torch.mean(alpha_vals * r_all_i).item()
            bias_correction = -0.5 * (g_alpha_y2 - g_alpha_y1) * (Z - sigmoid(r1 - r2))
            
            influence_values[i] = linear_part - bias_correction.item()
            
    return influence_values

# ==========================================
# 3. Bootstrap Uncertainty Assessment
# ==========================================
def assess_uncertainty(influence_values, B=1000, out_dir="./llm_pref_data/plots"):
    n = len(influence_values)
    delta_true = np.mean(influence_values)  # Pseudo-population truth
    
    boot_deltas = np.zeros(B)
    boot_sigmas = np.zeros(B)
    boot_z_scores = np.zeros(B)
    
    print(f"Running {B} bootstrap iterations for coverage and QQ assessment...")
    for b in range(B):
        # Resample influence functions with replacement
        sample_idx = np.random.choice(n, size=n, replace=True)
        boot_sample = influence_values[sample_idx]
        
        delta_b = np.mean(boot_sample)
        sigma_b = np.std(boot_sample, ddof=1)
        
        boot_deltas[b] = delta_b
        boot_sigmas[b] = sigma_b
        # Studentized test statistic around the "true" parameter
        boot_z_scores[b] = (delta_b - delta_true) / (sigma_b / np.sqrt(n))
        
    # --- 1. Compute Coverage Probabilities ---
    alphas = [0.10, 0.05, 0.01]  # 90%, 95%, 99% nominal coverage
    print("\n--- Empirical Coverage Probabilities ---")
    print(f"True Full-Sample Delta: {delta_true:.4f}")
    
    coverage_results = []
    for alpha in alphas:
        z_crit = stats.norm.ppf(1 - alpha/2)
        lower_bounds = boot_deltas - z_crit * (boot_sigmas / np.sqrt(n))
        upper_bounds = boot_deltas + z_crit * (boot_sigmas / np.sqrt(n))
        
        coverage = np.mean((delta_true >= lower_bounds) & (delta_true <= upper_bounds))
        print(f"Nominal {100*(1-alpha):.0f}% CI Coverage: {coverage*100:.2f}%")
        coverage_results.append({"Nominal": f"{100*(1-alpha):.0f}%", "Empirical Coverage": coverage})
        
    pd.DataFrame(coverage_results).to_csv(os.path.join(out_dir, "coverage_probabilities.csv"), index=False)

    # --- 2. Generate Q-Q Plot ---
    setup_neurips_style()
    plt.figure(figsize=(7, 7))
    stats.probplot(boot_z_scores, dist="norm", plot=plt)
    
    plt.title('Q-Q Plot of the Debiased Estimator')
    plt.xlabel('Theoretical Standard Normal Quantiles')
    plt.ylabel('Empirical Studentized Quantiles')
    
    # Beautify the Q-Q plot line elements
    ax = plt.gca()
    ax.get_lines()[0].set_marker('o')
    ax.get_lines()[0].set_markerfacecolor('purple')
    ax.get_lines()[0].set_markeredgecolor('white')
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('black')
    ax.get_lines()[1].set_linewidth(2)
    ax.get_lines()[1].set_linestyle('--')
    
    out_path = os.path.join(out_dir, "qq_plot_estimator.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved Q-Q Plot: {out_path}")

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="./results_chandelier/samples1000_k20/step2_rewards.json")
    parser.add_argument("--weights_file", type=str, default="./results_chandelier/samples1000_k20/g_net_weights.pt")
    parser.add_argument("--out_dir", type=str, default="./results_chandelier/samples1000_k20/plots")
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--target_prompt_idx", type=int, default=15, help="Index of prompt to run uncertainty assessment on")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading datasets and networks for Uncertainty Assessment...")
    with open(args.in_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    encoder = SentenceTransformer(args.embed_model, device=device)
    prompt_embeddings = encoder.encode(df["prompt_text"].tolist(), convert_to_numpy=True, normalize_embeddings=True)
    
    X_tensor = torch.tensor(prompt_embeddings, dtype=torch.float32)
    Y_tensor = torch.tensor(df["candidate_embeddings"].tolist(), dtype=torch.float32)
    r_tensor = torch.tensor(df["implicit_rewards"].tolist(), dtype=torch.float32)
    d, m = X_tensor.shape[1], Y_tensor.shape[1]
    
    g_net = SimpleNet(input_dim=3*d).to(device)
    g_net.load_state_dict(torch.load(args.weights_file, map_location=device))
    g_net.eval()
    
    influence_values = extract_influence_functions(
        X_tensor, Y_tensor, r_tensor, g_net, args.target_prompt_idx, d, m, 0.01, device
    )
    
    assess_uncertainty(influence_values, B=1000, out_dir=args.out_dir)