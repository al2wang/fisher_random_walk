import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

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
    """Derivative of the Bradley-Terry logistic link function[cite: 3]."""
    s = sigmoid(t)
    return s * (1 - s)

# ==========================================
# 1. Network Architecture
# ==========================================
class SimpleNet(nn.Module):
    """
    MLP architecture mapping the concatenated embeddings (X, Y1, Y2) 
    to a scalar representing the regular Green density g_0[cite: 3].
    """
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
# 2. Dataset and Prompt Embedding
# ==========================================
class LLMPreferenceDataset(Dataset):
    def __init__(self, prompts, candidates, rewards):
        self.prompts = prompts          # shape: (n, d)
        self.candidates = candidates    # shape: (n, m, d)
        self.rewards = rewards          # shape: (n, m)
        self.n = prompts.shape[0]
        self.m = candidates.shape[1]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.prompts[idx], self.candidates[idx], self.rewards[idx]

def prepare_dataset(data_df, embedding_model_name, device):
    print(f"Embedding prompts into continuous space using {embedding_model_name}...")
    encoder = SentenceTransformer(embedding_model_name, device=device)
    
    prompt_texts = data_df["prompt_text"].tolist()
    # Encode prompts to shape (n, d)
    prompt_embeddings = encoder.encode(prompt_texts, convert_to_numpy=True, normalize_embeddings=True)
    
    # Extract candidate embeddings and implicit rewards
    candidate_embeddings = np.array(data_df["candidate_embeddings"].tolist()) # shape: (n, m, d)
    implicit_rewards = np.array(data_df["implicit_rewards"].tolist())         # shape: (n, m)
    
    return torch.tensor(prompt_embeddings, dtype=torch.float32), \
           torch.tensor(candidate_embeddings, dtype=torch.float32), \
           torch.tensor(implicit_rewards, dtype=torch.float32)

# ==========================================
# 3. Variational Risk Minimization
# ==========================================
def train_green_density(dataloader, d, m, epochs=300, lr=1e-3, device="cuda"):
    """
    Minimizes the empirical variational risk \hat{J}_n(g) to estimate \hat{g}_0[cite: 3].
    """
    # input_dim is 3*d because it takes concat(X, Y1, Y2)[cite: 3]
    g_net = SimpleNet(input_dim=3*d).to(device)
    optimizer = optim.Adam(g_net.parameters(), lr=lr)
    
    print("Training the Green density estimator \hat{g}_0...")
    g_net.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_X, batch_Y_all, batch_r_all in dataloader:
            batch_X = batch_X.to(device)
            batch_Y_all = batch_Y_all.to(device)
            batch_r_all = batch_r_all.to(device)
            
            batch_size = batch_X.shape[0]
            
            # To simulate Y1, Y2 ~ \pi_0(\cdot|X), randomly sample 2 distinct indices 
            # from the m generated candidates for each item in the batch.
            idx1 = torch.randint(0, m, (batch_size,), device=device)
            idx2 = torch.randint(0, m, (batch_size,), device=device)
            idx2 = torch.where(idx1 == idx2, (idx2 + 1) % m, idx2) # Ensure idx1 != idx2
            
            b_idx = torch.arange(batch_size, device=device)
            Y1 = batch_Y_all[b_idx, idx1]
            Y2 = batch_Y_all[b_idx, idx2]
            r1 = batch_r_all[b_idx, idx1]
            r2 = batch_r_all[b_idx, idx2]
            
            # The full set of m generated candidates acts as our auxiliary samples \tilde{Y}[cite: 3]
            Y_tilde = batch_Y_all
            r_tilde = batch_r_all
            
            # Pre-compute empirical \hat{K}_i terms[cite: 3]
            K_hat_i = mu_prime(r1 - r2) 
            
            # Compute \tilde{d}_{X_i}(Y_1) and \tilde{d}_{X_i}(Y_2)[cite: 3]
            d_tilde_y1 = torch.mean(mu_prime(r1.unsqueeze(1) - r_tilde), dim=1) + 1e-6
            d_tilde_y2 = torch.mean(mu_prime(r2.unsqueeze(1) - r_tilde), dim=1) + 1e-6

            # Compute \hat{E}_i(g, g) component[cite: 3]
            X_exp = batch_X.unsqueeze(1).expand(-1, m, -1)
            Y1_exp = Y1.unsqueeze(1).expand(-1, m, -1)
            Y2_exp = Y2.unsqueeze(1).expand(-1, m, -1)
            
            g_y1 = g_net(torch.cat([X_exp, Y_tilde, Y1_exp], dim=-1))
            g_y2 = g_net(torch.cat([X_exp, Y_tilde, Y2_exp], dim=-1))
            
            E_i = torch.mean((g_y1 - g_y2)**2, dim=1) * K_hat_i
            
            # Compute \hat{S}_i component[cite: 3]
            g_y1y2 = g_net(torch.cat([batch_X, Y1, Y2], dim=-1))
            part1 = 0.5 * g_y1y2 * (1.0/d_tilde_y1 + 1.0/d_tilde_y2) * K_hat_i
            
            # U-statistic part of \hat{S}_i (sum over j != k)[cite: 3]
            Y_tilde_j = Y_tilde.unsqueeze(2).expand(-1, m, m, -1)
            Y_tilde_k = Y_tilde.unsqueeze(1).expand(-1, m, m, -1)
            X_jk = batch_X.unsqueeze(1).unsqueeze(2).expand(-1, m, m, -1)
            
            # Mask out the diagonal where j == k[cite: 3]
            mask = ~torch.eye(m, dtype=torch.bool, device=device).unsqueeze(0)
            g_jk = g_net(torch.cat([X_jk, Y_tilde_j, Y_tilde_k], dim=-1))
            part2 = torch.sum(g_jk * mask, dim=(1,2)) / (m * (m-1))
            
            S_i = part1 - part2
            
            # Minimize the regularized variational risk[cite: 3]
            loss = torch.mean(0.5 * E_i - S_i)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
            
    return g_net

# ==========================================
# 4. Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Green Density Estimation")
    parser.add_argument("--in_file", type=str, default="./llm_pref_data/step2_rewards.json", help="Input JSON from Step 2")
    parser.add_argument("--out_dir", type=str, default="./llm_pref_data", help="Output directory for model weights")
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model for mapping prompts")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs for the Green density network")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Loading data from {args.in_file}...")
    with open(args.in_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # 1. Prepare continuous tensors
    X_tensor, Y_tensor, r_tensor = prepare_dataset(df, args.embed_model, device)
    d = X_tensor.shape[1]  # Embedding dimension
    m = Y_tensor.shape[1]  # Number of candidate treatments
    
    # 2. Build DataLoader
    dataset = LLMPreferenceDataset(X_tensor, Y_tensor, r_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. Train the Green density estimator
    g_net = train_green_density(dataloader, d, m, epochs=args.epochs, lr=args.lr, device=device)
    
    # 4. Save the optimized network weights
    out_path = os.path.join(args.out_dir, "g_net_weights.pt")
    torch.save(g_net.state_dict(), out_path)
    print(f"\nStep 3 Complete! Green density network weights saved to {out_path}.")