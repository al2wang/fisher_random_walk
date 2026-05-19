import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import copy
from tqdm import tqdm

# ==========================================
# 0. Environment Setup & Primitives
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = ['gpt3', 'alpaca', 'gpt4o', 'llama1', 'llama2']
DOMAINS = ['anatomy', 'clinical_knowledge', 'college_biology', 'college_medicine', 'medical_genetics']
n = len(MODELS)
D = len(DOMAINS)
S = 3          # 3-fold cross-fitting
L = 100 * D    # 100 questions per domain
input_dim = 256

class SimpleNet(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, output_size=1):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ==========================================
# 1. Data Ingestion (Embeddings & GPT-4 Labels)
# ==========================================
def load_real_data():
    X_list = []
    for ss in DOMAINS:
        # Load 256D truncated OpenAI embeddings
        df = pd.read_csv(f"../Fisher_Random_Walk/llm_experiment/data/reduced_questions_{ss}.csv")
        array = np.vstack(df["embedding"].apply(lambda x: np.array(eval(x))))
        X_list.append(array)
    X = np.vstack(X_list)
    
    # Fully connected comparison graph (p=1.0)
    A = np.ones((n, n)) - np.eye(n)
    
    X_ijl = np.zeros((n, n, L, input_dim))
    Y_ijl = np.zeros((n, n, L))
    domain_labels = np.repeat(np.arange(D), 100)
    
    splitted_dense_data = [[] for _ in range(S)]
    
    for i in range(n):
        for j in range(i):
            if A[i, j] == 1:
                # Load GPT-4 pairwise judgments
                Y_list = []
                for ss in DOMAINS:
                    pairs = json.load(open(f"../Fisher_Random_Walk/llm_experiment/eval/{ss}_num=100.json", "r"))
                    for pair in pairs:
                        if pair["variable_name1"] == MODELS[j] and pair["variable_name2"] == MODELS[i]:
                            Y_list.append(1 - np.array(pair["GPT4_output"]))
                
                Y = np.concatenate(Y_list)
                
                # Shuffle to distribute domains evenly across folds
                indices = np.random.permutation(L)
                X_ijl[i, j, :, :] = X[indices]
                X_ijl[j, i, :, :] = X_ijl[i, j, :, :]
                Y_ijl[i, j, :] = Y[indices]
                Y_ijl[j, i, :] = Y_ijl[i, j, :]
                shuffled_domain = domain_labels[indices]
                
                for s in range(S):
                    start, end = s * (L // S), (s + 1) * (L // S)
                    splitted_dense_data[s].append((
                        i, j, 
                        X_ijl[i, j, start:end, :], 
                        Y_ijl[i, j, start:end], 
                        shuffled_domain[start:end]
                    ))
                    
    return A, splitted_dense_data

# ==========================================
# 2. Contextual Reward Estimation (\hat{\theta})
# ==========================================
def train_theta(splitted_dense_data, epochs=4500, lr=0.1):
    networks = [[SimpleNet().to(device) for _ in range(n)] for _ in range(S)]
    
    # Construct Leave-One-Out sets for cross-fitting
    loo_data = [list(itertools.chain(*splitted_dense_data[:s] + splitted_dense_data[s+1:])) for s in range(S)]
    
    for s in range(S):
        optimizer = optim.SGD([p for net in networks[s] for p in net.parameters()], lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Mini-batching
            batch = [loo_data[s][idx] for idx in np.random.choice(len(loo_data[s]), 16, replace=False)]
            
            total_loss = 0.0
            for item in batch:
                i, j, x_val, y_val, _ = item
                x_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
                y_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
                
                out_i = networks[s][i](x_tensor)
                out_j = networks[s][j](x_tensor)
                
                diff = out_i - out_j
                prob = torch.sigmoid(diff)
                
                loss = -(y_tensor * torch.log(prob + 1e-8) + (1 - y_tensor) * torch.log(1 - prob + 1e-8))
                total_loss += loss.mean()
            
            (total_loss / len(batch)).backward()
            optimizer.step()
            
    return networks

# ==========================================
# 3. Nuisance Parameter Matrix (\Pi)
# ==========================================
def estimate_pi(networks, A, splitted_dense_data):
    pis = np.zeros((S, n, n, D, n, n, L // S, n))
    
    with torch.no_grad(): # Suspend gradient tracking
        for s in range(S):
            for data in splitted_dense_data[s]:
                i, j, x_val, y_val, domains = data
                x_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
                
                thetas = torch.stack([networks[s][k](x_tensor) for k in range(n)])
                
                diffs = thetas.unsqueeze(1) - thetas.unsqueeze(0)
                M = torch.exp(diffs) / (torch.exp(diffs) + 1)**2
                
                M_permuted = M.permute(2, 0, 1)
                lower = torch.tril(M_permuted)
                M_sym = lower + lower.transpose(1, 2) - torch.diag_embed(lower.diagonal(dim1=1, dim2=2))
                
                repeated_A = torch.tensor(A, dtype=torch.float32, device=device).unsqueeze(0).expand(M_sym.shape[0], -1, -1)
                M_sym = repeated_A * M_sym
                
                row_sums = torch.sum(M_sym, dim=2)
                diag_matrices = torch.diag_embed(row_sums)
                input_M = (diag_matrices - M_sym).detach().cpu().numpy() # Added .detach()
                
                for d_idx in range(D):
                    in_domain = (domains == d_idx).astype(float)
                    svdsinv = np.array([
                        np.linalg.pinv(input_M[idx], rcond=1e-3, hermitian=True) if in_domain[idx] == 1 else np.zeros((n, n)) 
                        for idx in range(L // S)
                    ])
                    
                    for j_0, i_0 in itertools.combinations(range(n), 2):
                        basis = np.eye(n)[i_0] - np.eye(n)[j_0]
                        pi_val = len(splitted_dense_data[0]) * in_domain[:, None] * np.matmul(svdsinv, basis)
                        pis[s, i, j, d_idx, i_0, j_0] = pi_val
    return pis

# ==========================================
# 4. Debiased Estimator & Variance Computation
# ==========================================
def compute_Q_and_V(networks, pis, splitted_dense_data):
    Q_matrix = np.zeros((D, n, n))
    V_matrix = np.zeros((D, n, n))
    
    with torch.no_grad(): # Suspend gradient tracking
        for d_idx in range(D):
            for j_0, i_0 in itertools.combinations(range(n), 2):
                Q_s_list = []
                V_s_list = []
                
                sigma_s_list = []
                for s in range(S):
                    sum_pi = sum((pis[s, data[0], data[1], d_idx, i_0, j_0, :, i_0] - 
                                  pis[s, data[0], data[1], d_idx, i_0, j_0, :, j_0]).sum() 
                                 for data in splitted_dense_data[s])
                    sigma_s_list.append(sum_pi / (len(splitted_dense_data[s])**2) / L * S)
                
                for s in range(S):
                    sum_q = 0
                    sum_v = 0
                    for data in splitted_dense_data[s]:
                        i, j, x_val, y_val, domains = data
                        in_domain = (domains == d_idx).astype(float)
                        
                        x_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
                        t_i0 = networks[s][i_0](x_tensor).detach().cpu().numpy() # Added .detach()
                        t_j0 = networks[s][j_0](x_tensor).detach().cpu().numpy() # Added .detach()
                        t_i = networks[s][i](x_tensor).detach().cpu().numpy() # Added .detach()
                        t_j = networks[s][j](x_tensor).detach().cpu().numpy() # Added .detach()
                        
                        pi_i = pis[s, i, j, d_idx, i_0, j_0, :, i]
                        pi_j = pis[s, i, j, d_idx, i_0, j_0, :, j]
                        
                        q_val = in_domain * (t_i0 - t_j0) + (pi_i - pi_j) * (y_val - np.exp(t_i - t_j) / (np.exp(t_i - t_j) + 1))
                        sum_q += q_val
                        
                    Q_s = sum_q.sum() / len(splitted_dense_data[s]) / L * S
                    Q_s_list.append(Q_s)
                    
                Q_final = np.mean(Q_s_list)
                Q_matrix[d_idx, i_0, j_0] = Q_final
                Q_matrix[d_idx, j_0, i_0] = -Q_final 
                
                for s in range(S):
                    sum_v = 0
                    for data in splitted_dense_data[s]:
                        i, j, x_val, _, domains = data
                        in_domain = (domains == d_idx).astype(float)
                        x_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
                        diff_i0_j0 = networks[s][i_0](x_tensor).detach().cpu().numpy() - networks[s][j_0](x_tensor).detach().cpu().numpy()
                        
                        v_val = (in_domain * diff_i0_j0 - Q_final)**2
                        sum_v += v_val
                        
                    V_s = sum_v.sum() / (len(splitted_dense_data[s])**2) / (L**2) * (S**2) + sigma_s_list[s] / L
                    V_s_list.append(V_s)
                    
                V_matrix[d_idx, i_0, j_0] = np.mean(V_s_list)
                V_matrix[d_idx, j_0, i_0] = np.mean(V_s_list) 
                
    return Q_matrix, V_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./results_llm_eval")
    parser.add_argument("--epochs", type=int, default=4500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading MMLU Embeddings and GPT-4 Judgments...")
    A, splitted_dense_data = load_real_data()
    
    print("Training Contextual Reward Networks...")
    networks = train_theta(splitted_dense_data, epochs=args.epochs)
    
    print("Computing Nuisance Parameter Matrix (Pi)...")
    pis = estimate_pi(networks, A, splitted_dense_data)
    
    print("Calculating Debiased Inference Matrices...")
    Q_mat, V_mat = compute_Q_and_V(networks, pis, splitted_dense_data)
    
    # Save final matrices per domain
    for d_idx, domain in enumerate(DOMAINS):
        q_df = pd.DataFrame(Q_mat[d_idx], columns=MODELS, index=MODELS)
        v_df = pd.DataFrame(V_mat[d_idx], columns=MODELS, index=MODELS)
        
        q_df.to_csv(os.path.join(args.out_dir, f"Q_matrix_{domain}.csv"))
        v_df.to_csv(os.path.join(args.out_dir, f"V_matrix_{domain}.csv"))
        
    print(f"Evaluation complete. Results saved to {args.out_dir}")