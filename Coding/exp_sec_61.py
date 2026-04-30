import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm


d = 50
D_bound = np.sqrt(3) # D = (-\sqrt{3}, \sqrt{3})^d
pi_e_bound = 1.0 # \pi_e domain = [-1, 1]^d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta = (torch.ones(d) / math.sqrt(d)).to(device)
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
    beta_tx = torch.matmul(x, beta)
    return (beta_tx > -0.5).float()

def sigmoid(t):
    return 1.0 / (1.0 + torch.exp(-t))

def mu_prime(t):
    s = sigmoid(t)
    return s * (1 - s)

def true_reward(x, y):
    """Computes the true latent reward function r^*(x,y)"""
    # \bar{s}(y) = d^{-1} \sum_{j=1}^d \sin(\pi y_j / 8)
    s_bar = torch.mean(torch.sin(math.pi * y / 8.0), dim=-1)
    beta_tx = torch.matmul(x, beta)
    return (1.0 / 0.628) * s_bar * torch.tanh(beta_tx)

# step 1; data generation
def generate_data(n):
    # X ~ Unif(D), Y1, Y2 ~ \pi_0(\cdot|X) = Unif(D)
    X = (torch.rand(n, d, device=device) * 2 * D_bound) - D_bound
    Y1 = (torch.rand(n, d, device=device) * 2 * D_bound) - D_bound
    Y2 = (torch.rand(n, d, device=device) * 2 * D_bound) - D_bound
    
    r_diff = true_reward(X, Y1) - true_reward(X, Y2)
    prob = sigmoid(r_diff)
    
    # Z ~ Bernoulli(\mu(r^*(X, Y_1) - r^*(X, Y_2)))
    Z = torch.bernoulli(prob)
    return X, Y1, Y2, Z


# step 2; reward estimation
def train_reward_estimator(X, Y1, Y2, Z, epochs=500, lr=1e-3):
    r_net = SimpleNet(input_dim=2*d).to(device)
    optimizer = optim.Adam(r_net.parameters(), lr=lr)
    
    # train the pairwise logistic loss (eq4.1)
    for _ in range(epochs):
        optimizer.zero_grad()
        r1 = r_net(torch.cat([X, Y1], dim=-1))
        r2 = r_net(torch.cat([X, Y2], dim=-1))
        
        loss = -torch.mean(Z * torch.log(sigmoid(r1 - r2) + 1e-8) + 
                           (1 - Z) * torch.log(1 - sigmoid(r1 - r2) + 1e-8))
        loss.backward()
        optimizer.step()
        
    # wrapper to enforce the centering constraint
    def centered_reward(x_eval, y_eval, num_mc=100):
        with torch.no_grad():
            raw_r = r_net(torch.cat([x_eval, y_eval], dim=-1))
            # estimate \Pi_x \hat{r} using MC draws from \pi_0 (Unif(D))
            Y_mc = (torch.rand(x_eval.shape[0], num_mc, d, device=device) * 2 * D_bound) - D_bound
            X_mc_expanded = x_eval.unsqueeze(1).expand(-1, num_mc, -1)
            raw_r_mc = r_net(torch.cat([X_mc_expanded, Y_mc], dim=-1))
            mean_r = torch.mean(raw_r_mc, dim=1)
            return raw_r - mean_r
            
    return centered_reward

# step 3; green density estimation
def train_green_density(X, Y1, Y2, r_hat_fn, m=10, epochs=300, lr=1e-3):
    n = X.shape[0]
    g_net = SimpleNet(input_dim=3*d).to(device)
    optimizer = optim.Adam(g_net.parameters(), lr=lr)
    
    # pre-compute empirical \hat{K}_i terms
    with torch.no_grad():
        r1 = r_hat_fn(X, Y1)
        r2 = r_hat_fn(X, Y2)
        K_hat_i = mu_prime(r1 - r2) # \hat{K}_i
        
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # draw m auxiliary samples \tilde{Y}_{i,j} ~ \pi_0(\cdot | X_i)
        Y_tilde = (torch.rand(n, m, d, device=device) * 2 * D_bound) - D_bound
        
        # we need \tilde{d}_{X_i}(Y_{1i}) and \tilde{d}_{X_i}(Y_{2i})
        # \tilde{d}_x(y) = 1/m \sum_k \mu'(\hat{r}(x, y) - \hat{r}(x, \tilde{Y}_{i,k}))
        with torch.no_grad():
            r_tilde = r_hat_fn(X.unsqueeze(1).expand(-1, m, -1).reshape(n*m, d), Y_tilde.reshape(n*m, d)).reshape(n, m)
            d_tilde_y1 = torch.mean(mu_prime(r1.unsqueeze(1) - r_tilde), dim=1) + 1e-6
            d_tilde_y2 = torch.mean(mu_prime(r2.unsqueeze(1) - r_tilde), dim=1) + 1e-6

        # compute \hat{J}_n(g) components
        # 1. \hat{E}_i(g,g)
        X_exp = X.unsqueeze(1).expand(-1, m, -1)
        Y1_exp = Y1.unsqueeze(1).expand(-1, m, -1)
        Y2_exp = Y2.unsqueeze(1).expand(-1, m, -1)
        
        g_y1 = g_net(torch.cat([X_exp, Y_tilde, Y1_exp], dim=-1))
        g_y2 = g_net(torch.cat([X_exp, Y_tilde, Y2_exp], dim=-1))
        
        E_i = torch.mean((g_y1 - g_y2)**2, dim=1) * K_hat_i
        
        # 2. \hat{S}_i
        g_y1y2 = g_net(torch.cat([X, Y1, Y2], dim=-1))
        part1 = 0.5 * g_y1y2 * (1.0/d_tilde_y1 + 1.0/d_tilde_y2) * K_hat_i
        
        # for the U-statistic part of \hat{S}_i (sum over j != k)
        Y_tilde_j = Y_tilde.unsqueeze(2).expand(-1, m, m, -1)
        Y_tilde_k = Y_tilde.unsqueeze(1).expand(-1, m, m, -1)
        X_jk = X.unsqueeze(1).unsqueeze(2).expand(-1, m, m, -1)
        
        # mask out the diagonal where j == k
        mask = ~torch.eye(m, dtype=torch.bool, device=device).unsqueeze(0)
        g_jk = g_net(torch.cat([X_jk, Y_tilde_j, Y_tilde_k], dim=-1))
        part2 = torch.sum(g_jk * mask, dim=(1,2)) / (m * (m-1))
        
        S_i = part1 - part2
        
        # minimize the regularized variational risk
        loss = torch.mean(0.5 * E_i - S_i)
        loss.backward()
        optimizer.step()
        
    return g_net

# steps 4,5; debiased estimator & variance
def compute_influence_and_ci(X, Y1, Y2, Z, r_hat_fn, g_net, n, m_pi_e=50):
    with torch.no_grad():
        w_X = omega(X)
        r1 = r_hat_fn(X, Y1)
        r2 = r_hat_fn(X, Y2)
        
        # first term of influence function: \int r(X, z)\pi_e(z|X)dz
        Z_e = (torch.rand(n, m_pi_e, d, device=device) * 2 * pi_e_bound) - pi_e_bound
        X_e = X.unsqueeze(1).expand(-1, m_pi_e, -1)
        r_int = torch.mean(r_hat_fn(X_e.reshape(n*m_pi_e, d), Z_e.reshape(n*m_pi_e, d)).reshape(n, m_pi_e), dim=1)
        

        # compute \hat{g}_{\pi_e}(X, Y1) and \hat{g}_{\pi_e}(X, Y2) via Eq 2.7

        # NOTE: pi_e/pi_0 density ratios must be factored in

        # \pi_0 is Unif(D) = 1/(2\sqrt{3})^d. \pi_e is Unif([-1,1]^d) = 1/2^d
        # the density ratio \pi_e/\pi_0 is (sqrt(3))^d inside [-1, 1]^d, 0 outside
        density_ratio = (math.sqrt(3))**d 
        
        # we need \tilde{d}_X(Y) again
        Y_tilde = (torch.rand(n, 20, d, device=device) * 2 * D_bound) - D_bound
        r_tilde = r_hat_fn(X.unsqueeze(1).expand(-1, 20, -1).reshape(n*20, d), Y_tilde.reshape(n*20, d)).reshape(n, 20)
        d_tilde_y1 = torch.mean(mu_prime(r1.unsqueeze(1) - r_tilde), dim=1) + 1e-6
        d_tilde_y2 = torch.mean(mu_prime(r2.unsqueeze(1) - r_tilde), dim=1) + 1e-6

        # integral of \hat{g}_0 over \pi_e
        g0_y1_e = torch.mean(g_net(torch.cat([X_e, Y1.unsqueeze(1).expand(-1, m_pi_e, -1), Z_e], dim=-1)), dim=1)
        g0_y2_e = torch.mean(g_net(torch.cat([X_e, Y2.unsqueeze(1).expand(-1, m_pi_e, -1), Z_e], dim=-1)), dim=1)
        
        # check if Y1/Y2 are in [-1, 1]^d for the Dirac component
        in_pi_e_y1 = (torch.max(torch.abs(Y1), dim=1)[0] <= pi_e_bound).float()
        in_pi_e_y2 = (torch.max(torch.abs(Y2), dim=1)[0] <= pi_e_bound).float()

        g_pi_e_y1 = (1.0 / d_tilde_y1) * density_ratio * in_pi_e_y1 + g0_y1_e
        g_pi_e_y2 = (1.0 / d_tilde_y2) * density_ratio * in_pi_e_y2 + g0_y2_e
        
        # assemble the full influence function
        bias_correction = 0.5 * (g_pi_e_y2 - g_pi_e_y1) * (Z - sigmoid(r1 - r2))
        psi_pi_e = r_int - bias_correction
        
        IF_i = w_X * psi_pi_e
        
        # compute debiased estimator
        r_hat_d = torch.mean(IF_i).item()
        # compute variance and CI
        sigma_sq = torch.var(IF_i, unbiased=True).item()
        sigma = math.sqrt(sigma_sq)
        ci_lower = r_hat_d - 1.96 * sigma / math.sqrt(n)
        ci_upper = r_hat_d + 1.96 * sigma / math.sqrt(n)
        
    return r_hat_d, ci_lower, ci_upper, (ci_upper - ci_lower)


# main simulation loop
def run_simulation():
    sample_sizes = [500, 1000, 2000, 5000]
    replications = 5
    
    # approximate true expected reward via oracle

    print("Approximating Oracle r^*_pi_e ...")
    N_oracle = 100000
    X_oracle = (torch.rand(N_oracle, d, device=device) * 2 * D_bound) - D_bound
    Y_oracle = (torch.rand(N_oracle, d, device=device) * 2 * pi_e_bound) - pi_e_bound
    r_star_oracle = torch.mean(omega(X_oracle) * true_reward(X_oracle, Y_oracle)).item()
    print(f"True Expected Reward (r^*_pi_e): {r_star_oracle:.4f}\n")
    
    for n in sample_sizes:
        print(f"=== Running Sample Size n={n} ===")
        coverage_count = 0
        errors = []
        lengths = []
        
        for rep in range(replications):
            X, Y1, Y2, Z = generate_data(n) # step 1
            r_hat_fn = train_reward_estimator(X, Y1, Y2, Z, epochs=300) # step 2
            g_net = train_green_density(X, Y1, Y2, r_hat_fn, m=10, epochs=200) # step 3
            r_hat_d, ci_lower, ci_upper, ci_length = compute_influence_and_ci(X, Y1, Y2, Z, r_hat_fn, g_net, n) # step 4,5
            
            error = abs(r_hat_d - r_star_oracle)
            covered = 1 if (ci_lower <= r_star_oracle <= ci_upper) else 0
            
            errors.append(error)
            coverage_count += covered
            lengths.append(ci_length)
            
        print(f"Mean Estimation Error: {np.mean(errors):.4f}")
        print(f"Empirical Coverage: {coverage_count / replications * 100:.1f}%")
        print(f"Average CI Length: {np.mean(lengths):.4f}\n")

import argparse
import csv
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="Sample size")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for replication")
    parser.add_argument("--out_dir", type=str, default="./", help="Output directory")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    N_oracle = 100000
    X_oracle = (torch.rand(N_oracle, d, device=device) * 2 * D_bound) - D_bound
    Y_oracle = (torch.rand(N_oracle, d, device=device) * 2 * pi_e_bound) - pi_e_bound
    r_star_oracle = torch.mean(omega(X_oracle) * true_reward(X_oracle, Y_oracle)).item()

    # run single experiment
    X, Y1, Y2, Z = generate_data(args.n)
    r_hat_fn = train_reward_estimator(X, Y1, Y2, Z, epochs=300)
    g_net = train_green_density(X, Y1, Y2, r_hat_fn, m=10, epochs=200)
    r_hat_d, ci_lower, ci_upper, ci_length = compute_influence_and_ci(X, Y1, Y2, Z, r_hat_fn, g_net, args.n)
    
    error = abs(r_hat_d - r_star_oracle)
    covered = 1 if (ci_lower <= r_star_oracle <= ci_upper) else 0

    out_file = os.path.join(args.out_dir, f"results_n{args.n}.csv")
    file_exists = os.path.isfile(out_file)
    with open(out_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["seed", "n", "r_star", "r_hat_d", "error", "covered", "ci_length"])
        writer.writerow([args.seed, args.n, r_star_oracle, r_hat_d, error, covered, ci_length])
        
    print(f"Finished n={args.n}, seed={args.seed}. Covered: {bool(covered)}")