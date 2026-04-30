import argparse
import csv
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


d = 50 
D_bound = math.sqrt(3) 
pi_e_bound = 1.0 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta_vec = (torch.ones(d) / math.sqrt(d)).to(device) 

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
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

def omega(x):
    return (torch.matmul(x, beta_vec) > -0.5).float()

def sigmoid(t):
    return 1.0 / (1.0 + torch.exp(-t))

def mu_prime(t):
    s = sigmoid(t)
    return s * (1 - s)

def true_reward(x, y):
    s_bar = torch.mean(torch.sin(math.pi * y / 8.0), dim=-1)
    beta_tx = torch.matmul(x, beta_vec)
    return (1.0 / 0.628) * s_bar * torch.tanh(beta_tx)



# data generation + training from 6.1
def generate_data(n):
    X = (torch.rand(n, d, device=device) * 2 * D_bound) - D_bound
    Y1 = (torch.rand(n, d, device=device) * 2 * D_bound) - D_bound
    Y2 = (torch.rand(n, d, device=device) * 2 * D_bound) - D_bound
    
    r_diff = true_reward(X, Y1) - true_reward(X, Y2)
    Z = torch.bernoulli(sigmoid(r_diff))
    return X, Y1, Y2, Z

def train_reward_estimator(X, Y1, Y2, Z, epochs=500, lr=1e-3):
    r_net = SimpleNet(input_dim=2*d).to(device)
    optimizer = optim.Adam(r_net.parameters(), lr=lr)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        r1 = r_net(torch.cat([X, Y1], dim=-1))
        r2 = r_net(torch.cat([X, Y2], dim=-1))
        
        loss = -torch.mean(Z * torch.log(sigmoid(r1 - r2) + 1e-8) + 
                           (1 - Z) * torch.log(1 - sigmoid(r1 - r2) + 1e-8))
        loss.backward()
        optimizer.step()
        
    def centered_reward(x_eval, y_eval, num_mc=100):
        with torch.no_grad():
            raw_r = r_net(torch.cat([x_eval, y_eval], dim=-1))
            
            # use original batch dimensions for MC sampling
            batch_dims = list(x_eval.shape[:-1])
            mc_shape = batch_dims + [num_mc, d]
            
            Y_mc = (torch.rand(*mc_shape, device=device) * 2 * D_bound) - D_bound
            
            x_expanded = x_eval.unsqueeze(-2).expand(*mc_shape) # expand x to match Y_mc
            
            raw_r_mc = r_net(torch.cat([x_expanded, Y_mc], dim=-1))
            mean_r = torch.mean(raw_r_mc, dim=-1)
            return raw_r - mean_r
            
    return centered_reward

def train_green_density(X, Y1, Y2, r_hat_fn, m=10, epochs=300, lr=1e-3):
    n = X.shape[0]
    g_net = SimpleNet(input_dim=3*d).to(device)
    optimizer = optim.Adam(g_net.parameters(), lr=lr)
    
    with torch.no_grad():
        r1 = r_hat_fn(X, Y1)
        r2 = r_hat_fn(X, Y2)
        K_hat_i = mu_prime(r1 - r2)
        
    for _ in range(epochs):
        optimizer.zero_grad()
        Y_tilde = (torch.rand(n, m, d, device=device) * 2 * D_bound) - D_bound
        
        with torch.no_grad():
            r_tilde = r_hat_fn(X.unsqueeze(1).expand(-1, m, -1), Y_tilde)
            d_tilde_y1 = torch.mean(mu_prime(r1.unsqueeze(1) - r_tilde), dim=1) + 1e-6
            d_tilde_y2 = torch.mean(mu_prime(r2.unsqueeze(1) - r_tilde), dim=1) + 1e-6

        X_exp = X.unsqueeze(1).expand(-1, m, -1)
        Y1_exp = Y1.unsqueeze(1).expand(-1, m, -1)
        Y2_exp = Y2.unsqueeze(1).expand(-1, m, -1)
        
        g_y1 = g_net(torch.cat([X_exp, Y_tilde, Y1_exp], dim=-1))
        g_y2 = g_net(torch.cat([X_exp, Y_tilde, Y2_exp], dim=-1))
        
        E_i = torch.mean((g_y1 - g_y2)**2, dim=1) * K_hat_i
        
        g_y1y2 = g_net(torch.cat([X, Y1, Y2], dim=-1))
        part1 = 0.5 * g_y1y2 * (1.0/d_tilde_y1 + 1.0/d_tilde_y2) * K_hat_i
        
        Y_tilde_j = Y_tilde.unsqueeze(2).expand(-1, m, m, -1)
        Y_tilde_k = Y_tilde.unsqueeze(1).expand(-1, m, m, -1)
        X_jk = X.unsqueeze(1).unsqueeze(2).expand(-1, m, m, -1)
        
        mask = ~torch.eye(m, dtype=torch.bool, device=device).unsqueeze(0)
        g_jk = g_net(torch.cat([X_jk, Y_tilde_j, Y_tilde_k], dim=-1))
        part2 = torch.sum(g_jk * mask, dim=(1,2)) / (m * (m-1))
        
        S_i = part1 - part2
        
        loss = torch.mean(0.5 * E_i - S_i)
        loss.backward()
        optimizer.step()
        
    return g_net


# 6.2 fenchel duality policy optimization
def solve_c_bisection(r_eval, lam, div_type, q=None, iters=30):
    batch_size = r_eval.shape[0]
    left = torch.min(lam * r_eval, dim=1)[0] - 10.0
    right = torch.max(lam * r_eval, dim=1)[0] + 1.0
    
    for _ in range(iters):
        mid = (left + right) / 2.0
        mid_exp = mid.unsqueeze(1)
        
        if div_type == "chi2":
            ratio = torch.clamp(lam * r_eval - mid_exp, min=0.0)
        elif div_type == "tsallis":
            base = torch.clamp(lam * r_eval - mid_exp, min=0.0)
            ratio = ((q - 1) * base + 1.0) ** (1.0 / (q - 1))
            
        integral = torch.mean(ratio, dim=1)
        
        left = torch.where(integral > 1.0, mid, left)
        right = torch.where(integral <= 1.0, mid, right)
        
    return (left + right) / 2.0

def get_policy_ratios(r_eval, lam, div_type, q=None):
    if div_type == "kl":
        exp_term = torch.exp(lam * r_eval)
        Z = torch.mean(exp_term, dim=1, keepdim=True)
        pi_ratio = exp_term / Z
        expected_r = torch.mean(r_eval * pi_ratio, dim=1, keepdim=True)
        pi_tilde_ratio = pi_ratio * (1.0 + lam * (r_eval - expected_r))
        
    elif div_type == "chi2":
        c_x = solve_c_bisection(r_eval, lam, "chi2").unsqueeze(1)
        pi_ratio = torch.clamp(lam * r_eval - c_x, min=0.0)
        support_mask = (pi_ratio > 0).float()
        r_supp_mean = torch.sum(r_eval * support_mask, dim=1, keepdim=True) / (torch.sum(support_mask, dim=1, keepdim=True) + 1e-8)
        pi_tilde_ratio = pi_ratio + lam * support_mask * (r_eval - r_supp_mean)
        
    elif div_type == "tsallis":
        c_x = solve_c_bisection(r_eval, lam, "tsallis", q=q).unsqueeze(1)
        base = torch.clamp(lam * r_eval - c_x, min=0.0)
        pi_ratio = ((q - 1) * base + 1.0) ** (1.0 / (q - 1))
        support_mask = (pi_ratio > 0).float()
        W = 1.0 / (pi_ratio ** (q - 2) + 1e-8)
        W_supp = W * support_mask
        r_w_mean = torch.sum(r_eval * W_supp, dim=1, keepdim=True) / (torch.sum(W_supp, dim=1, keepdim=True) + 1e-8)
        pi_tilde_ratio = pi_ratio + lam * W_supp * (r_eval - r_w_mean)
        
    return pi_ratio, pi_tilde_ratio

def compute_influence_opt(X, Y1, Y2, Z, r_hat_fn, g_net, lam, div_type, q=None, m_mc=200):
    n = X.shape[0]
    with torch.no_grad():
        w_X = omega(X)
        r1, r2 = r_hat_fn(X, Y1), r_hat_fn(X, Y2)
        
        Y_mc = (torch.rand(n, m_mc, d, device=device) * 2 * D_bound) - D_bound
        X_mc = X.unsqueeze(1).expand(-1, m_mc, -1)
        r_mc = r_hat_fn(X_mc, Y_mc)
        
        pi_ratio, pi_tilde_ratio = get_policy_ratios(r_mc, lam, div_type, q)
        r_int = torch.mean(r_mc * pi_ratio, dim=1)
        
        g0_y1_e = torch.mean(g_net(torch.cat([X_mc, Y1.unsqueeze(1).expand(-1, m_mc, -1), Y_mc], dim=-1)) * pi_tilde_ratio, dim=1)
        g0_y2_e = torch.mean(g_net(torch.cat([X_mc, Y2.unsqueeze(1).expand(-1, m_mc, -1), Y_mc], dim=-1)) * pi_tilde_ratio, dim=1)
        
        Y_tilde_d = (torch.rand(n, 20, d, device=device) * 2 * D_bound) - D_bound
        r_tilde_d = r_hat_fn(X.unsqueeze(1).expand(-1, 20, -1), Y_tilde_d)
        d_tilde_y1 = torch.mean(mu_prime(r1.unsqueeze(1) - r_tilde_d), dim=1) + 1e-6
        d_tilde_y2 = torch.mean(mu_prime(r2.unsqueeze(1) - r_tilde_d), dim=1) + 1e-6
        
        _, tilde_ratio_y1 = get_policy_ratios(r1.unsqueeze(1), lam, div_type, q)
        _, tilde_ratio_y2 = get_policy_ratios(r2.unsqueeze(1), lam, div_type, q)
        
        g_pi_e_y1 = (1.0 / d_tilde_y1) * tilde_ratio_y1.squeeze(-1) + g0_y1_e
        g_pi_e_y2 = (1.0 / d_tilde_y2) * tilde_ratio_y2.squeeze(-1) + g0_y2_e
        
        bias_correction = 0.5 * (g_pi_e_y2 - g_pi_e_y1) * (Z - sigmoid(r1 - r2))
        psi_pi = r_int - bias_correction
        IF_i = w_X * psi_pi
        
        r_hat_d = torch.mean(IF_i).item()
        sigma_sq = torch.var(IF_i, unbiased=True).item()
        ci_half = 1.96 * math.sqrt(sigma_sq) / math.sqrt(n)
        
    return r_hat_d, r_hat_d - ci_half, r_hat_d + ci_half, 2 * ci_half

def compute_oracle(lam, div_type, q=None, N_oracle=10000):
    X_o = (torch.rand(N_oracle, d, device=device) * 2 * D_bound) - D_bound
    Y_o = (torch.rand(N_oracle, 100, d, device=device) * 2 * D_bound) - D_bound
    
    with torch.no_grad():
        X_o_exp = X_o.unsqueeze(1).expand(-1, 100, -1)
        r_star_mc = true_reward(X_o_exp, Y_o)
        
        pi_ratio, _ = get_policy_ratios(r_star_mc, lam, div_type, q)
        J_lam = torch.mean(r_star_mc * pi_ratio, dim=1)
        r_star_lam = torch.mean(omega(X_o) * J_lam).item()
        
    return r_star_lam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="./results_sec_62")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== Starting Section 6.2 | n={args.n}, seed={args.seed} ===")

    # 1. Base Fit (Shared across all lambda/f-divergences for this seed)
    print("Generating data...")
    X, Y1, Y2, Z = generate_data(args.n)
    
    print("Training Reward Estimator...")
    r_hat_fn = train_reward_estimator(X, Y1, Y2, Z, epochs=300)
    
    print("Training Green Density Estimator...")
    g_net = train_green_density(X, Y1, Y2, r_hat_fn, epochs=200)

    divergences = [
        ("kl", None),
        ("chi2", None),
        ("tsallis", 5),
        ("tsallis", 10)
    ]
    r_grid = np.arange(0.1, 1.1, 0.1)

    out_file = os.path.join(args.out_dir, f"results_62_n{args.n}_s{args.seed}.csv")
    file_exists = os.path.isfile(out_file)
    
    print(f"Evaluating Fenchel Duality Grid. Saving to {out_file}...")
    with open(out_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["seed", "n", "r_param", "divergence", "q", "lam", "r_star", "r_hat_d", "error", "covered", "ci_length"])
            
        for r_val in r_grid:
            lam_n = float(args.n ** r_val)
            
            for div_type, q_val in divergences:
                # oracle truth
                r_star_val = compute_oracle(lam_n, div_type, q=q_val)
                # debiased inference
                r_hat_d, ci_lo, ci_hi, ci_len = compute_influence_opt(
                    X, Y1, Y2, Z, r_hat_fn, g_net, lam_n, div_type, q=q_val
                )
                
                error = abs(r_hat_d - r_star_val)
                covered = 1 if (ci_lo <= r_star_val <= ci_hi) else 0
                
                div_label = f"tsallis-{q_val}" if div_type == "tsallis" else div_type
                writer.writerow([args.seed, args.n, r_val, div_label, q_val, lam_n, r_star_val, r_hat_d, error, covered, ci_len])
                print(f"  r={r_val:.1f}, {div_label:<10} | Err: {error:.4f} | Cov: {covered}")