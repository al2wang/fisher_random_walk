#!/usr/bin/env python
# coding: utf-8

# ### Import Packages

# In[74]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import itertools
import copy
import csv
import scipy
import json
np.random.seed(1234)
random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ### Parameter Setting

# In[118]:


n = 5
p = 1.0
S = 3
L = 500
D = 5
num_epochs = 4500
batch_size = 16
T = 100
lr = 0.1
device = "cuda:1"


# ### Data Generation

# Generate A

# In[76]:


def generate_erdos_renyi_graph(n: int, p: float):
    """
    Generates an Erdos-Renyi graph G(n, p).

    Parameters:
    n (int): Number of nodes.
    p (float): Probability of edge creation.

    Returns:
    np.ndarray: Adjacency matrix of the generated graph.
    """
    # Initialize an n x n matrix with all entries set to 0
    A = np.zeros((n, n))

    # Iterate over all pairs (i, j) with i > j
    for i in range(n):
        for j in range(i):
            if np.random.rand() < p:
                A[i, j] = 1
                A[j, i] = 1

    return A


# In[77]:


# A = generate_erdos_renyi_graph(n, p)
# A


# In[78]:

def get_X(ss0):
# ss0 = ['anatomy','clinical_knowledge','college_biology', 'college_medicine','medical_genetics']
    X = []  # Initialize an empty list to store the 2D arrays

    for ss in ss0:
        # Read the CSV file and convert the embeddings into 2D arrays
        array = np.vstack(pd.read_csv(f"data/reduced_questions_{ss}.csv")["embedding"].apply(lambda x: np.array(eval(x))))
        X.append(array)  # Append each 2D array to the list

    # Stack all 2D arrays vertically into a single 2D array
    X = np.vstack(X)
    return X


# In[79]:





# In[80]:


# ss0 = ['anatomy', 'clinical_knowledge', 'college_biology', 'college_medicine', 'medical_genetics']
# vnames = ['gpt3', 'alpaca', 'gpt4o', 'llama1', 'llama2']

def get_Y(i, j, ss0, vnames): 
    assert i > j
    Y_list = []
    
    for ss in ss0:
        pairs = json.load(open(f"eval/{ss}_num=100.json","rb"))
        for pair in pairs:
            if pair["variable_name1"] == vnames[j] and pair["variable_name2"] == vnames[i]:
                Y_list.append(1 - np.array(pair["GPT4_output"]))
    return np.concat(Y_list)


# Generate X_ijl and Y_ijl

# In[87]:


def generate_X_Y(A, L, ss0, vnames, X):
    """
    Generates (X_ijl, Y_ijl) pairs for given adjacency matrix A.

    Parameters:
    A (np.ndarray): Adjacency matrix of the generated Erdos-Renyi graph.
    L (int): Number of samples for any (i, j) pair.

    Returns:
    np.ndarray: Tensor X_ijl and Y_ijl with shape (n, n, L).
    """
    X_ijl = np.zeros((n, n, L, 256))
    Y_ijl = np.zeros((n, n, L))
    domain = np.repeat(np.arange(D), 100)
    # dense_data = []
    splitted_dense_data = [[] for _ in range(S)]

    for i in range(n):
        for j in range(i):
            if A[i, j] == 1:
                # Shuffle X, Y and domain with random indices
                indices = np.random.permutation(L)

                X_ijl[i, j, :, :] = X[indices]
                X_ijl[j, i, :, :] = X_ijl[i, j, :, :]

                Y = get_Y(i, j, ss0, vnames)
                Y_ijl[i, j, :] = Y[indices]
                Y_ijl[j, i, :] = Y_ijl[i, j, :]

                shuffled_domain = domain[indices]

                for s in range(S):
                    splitted_dense_data[s].append((i, j, X_ijl[i, j, s * L//S:(s + 1) * L//S, :][:L//S,:], Y_ijl[i, j, s * L//S:(s + 1) * L//S][:L//S], shuffled_domain[s * L//S:(s + 1) * L//S][:L//S]))

    return X_ijl, Y_ijl, splitted_dense_data


# In[88]:


# X_ijl, Y_ijl, splitted_dense_data = generate_X_Y(A, L)


# Generate Leave-One-Out data

# In[89]:


def generate_LOO(splitted_dense_data):
    """
    Generates Leave-One-Out data for given splitted_dense_data.

    Parameters:
    splitted_dense_data (list): List of splitted dense data.

    Returns:
    list: List of Leave-One-Out dense data.
    """
    leave_one_out_dense_data = []

    for s in range(S):
        new_list = list(itertools.chain(*splitted_dense_data[:s] + splitted_dense_data[s+1:])) 
        leave_one_out_dense_data.append(new_list)
    return leave_one_out_dense_data


# In[90]:


# leave_one_out_dense_data = generate_LOO(splitted_dense_data)


# ### Estimate Theta

# Define the class of ReLU functions

# In[100]:


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# Initialize k Independent Networks

# In[101]:


def initialize_networks(k, input_size = 256, hidden_size = 128, output_size = 1):
    """
    Initialize networks for theta estimation.

    Parameters:
    k (int): Number of networks to be initialized given s.
    input_size (int): Dimension of input to theta.
    hidden_size (int): Dimension of hidden layer.
    output_size (int): Dimension of output of theta.

    Returns:
    list: List of networks.
    """
    networks = [[] for _ in range(S)]
    for s in range(S):
        for _ in range(k):
            net = SimpleNet(input_size, hidden_size, output_size)
            net.to(device)
            networks[s].append(copy.deepcopy(net))
    return networks


# In[102]:


# networks = initialize_networks(n)


# Batch Generator

# In[103]:


def sample_batch(dense_data, batch_size):
    return random.sample(dense_data, batch_size)


# Define Loss Function

# In[104]:


def compute_loss(networks, batch, s):
    total_loss = 0.0
    for index in range(len(batch)):
        output_i = networks[s][batch[index][0]](torch.tensor(batch[index][2],dtype=torch.float32, device=device))
        output_j = networks[s][batch[index][1]](torch.tensor(batch[index][2],dtype=torch.float32, device=device))
        Y = torch.tensor(batch[index][3],dtype=torch.float32, device=device).reshape(-1,1)
        loss = -(Y * torch.log(torch.exp(output_i - output_j) / (torch.exp(output_i - output_j) + 1)) \
        + (1 - Y) * torch.log(1 - torch.exp(output_i - output_j) / (torch.exp(output_i - output_j) + 1)))
        total_loss += loss.mean()
    return total_loss / len(batch)


# Define Optimizer

# In[105]:


def initialize_optimizer(networks):
    all_parameters = []
    for s in range(S):
        for net in networks[s]:
            all_parameters.extend(list(net.parameters()))

    optimizer = optim.SGD(all_parameters, lr=lr)
    return optimizer


# In[106]:


# optimizer = initialize_optimizer(networks)


# Estimate $\hat{\theta}_i(\cdot)$

# In[63]:


def estimate_theta(networks, optimizer, leave_one_out_dense_data, splitted_dense_data):
    for s in range(S):
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            batch = sample_batch(leave_one_out_dense_data[s], batch_size)
            loss = compute_loss(networks, batch, s)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                valid_loss = compute_loss(networks, splitted_dense_data[s], s)
                print(f"Epoch: {epoch + 1}/{num_epochs}, Training Loss: {loss}, Valid Loss: {valid_loss}")
                


# In[64]:


# estimate_theta(networks, optimizer, leave_one_out_dense_data, splitted_dense_data)


# ### Estimate Pi

# In[21]:


def make_symmetric_from_lower_3d(matrix):
    # Check if the last two dimensions are square
    if matrix.shape[1] != matrix.shape[2]:
        raise ValueError("The last two dimensions of the matrix must be square to make it symmetric")
    
    # Iterate over the first dimension
    for i in range(matrix.shape[0]):
        lower_triangle = torch.tril(matrix[i])
        symmetric_matrix = lower_triangle + lower_triangle.T - torch.diag(lower_triangle.diag())
        matrix[i] = symmetric_matrix

    return matrix


# In[22]:


def estimate_pi(networks, A, splitted_dense_data):
    pis = np.zeros((S, n, n, D, n, n, L // S, n))

    for s in range(S):
        for data in splitted_dense_data[s]:
            thetas = torch.stack([networks[s][i](torch.tensor(data[2],dtype=torch.float32, device=device)) for i in range(n)])

            M = (thetas[:,None] - thetas[None, :]).squeeze()
            M = torch.exp(M) / (torch.exp(M) + 1) ** 2
            M_permuted = M.permute(2, 0, 1)
            M_permuted = make_symmetric_from_lower_3d(M_permuted)

            expanded_A = np.expand_dims(A, axis=0)
            expanded_A = expanded_A.astype(np.float32)

            # Tile the array to repeat it L // S times along the first dimension
            repeated_A = np.tile(expanded_A, (M.shape[2], 1, 1))

            M_permuted = torch.tensor(repeated_A, dtype=torch.float32, device=device) * M_permuted

            # Initialize the matrix of ones with shape (n, 1)
            ones = torch.ones((n, 1), dtype=torch.float32, device=M_permuted.device)

            # Perform batch matrix multiplication to get the result of shape (L // S, n, 1)
            multiplied = torch.matmul(M_permuted, ones)

            # Remove the last dimension to get a shape of (L // S, n)
            multiplied = multiplied.squeeze(-1)

            # Create an identity matrix of shape (n, n) on the same device as x
            identity_matrix = torch.eye(n, dtype=torch.float32, device=M_permuted.device)

            # Use broadcasting to create diagonal matrices from the multiplied vectors
            diag_matrices = multiplied[:, :, None] * identity_matrix[None, :, :]

            # Convert PyTorch tensor to NumPy array
            diag_matrices_np = diag_matrices.detach().cpu().numpy()
            M_permuted_np = M_permuted.detach().cpu().numpy()
            input_M = diag_matrices_np - M_permuted_np
            # input_M = input_M.astype(np.float16)
            # input_M = input_M.astype(np.float32)

            # Compute the pseudoinverse using NumPy
            # svdsinv = np.linalg.pinv(input_M, rcond=1e-3, hermitian=True)

            # Compute pi and store pi in pis
            for domain in range(D):
                within_bounds = (data[4] == domain)
                within_bounds = within_bounds.astype(float)
                svdsinv = np.array([np.linalg.pinv(input_M[i], rcond=1e-3, hermitian=True) if within_bounds[i] == 1 else np.zeros((n, n)) for i in range(L // S)])

                for j_0, i_0 in itertools.combinations(range(n), 2):
                    pi = len(splitted_dense_data[0]) * within_bounds[:,None] * np.matmul(svdsinv, np.eye(n)[i_0] - np.eye(n)[j_0])
                    pis[s][data[0]][data[1]][domain][i_0][j_0] = pi
    return pis


# Sparse SVD

# In[23]:


# svdsinv_matrices = []
# for i in range(input_M.shape[0]):
#     input_M_i = scipy.sparse.csr_array(input_M[i])
#     solssvd_i = scipy.sparse.linalg.svds(input_M_i,k = n-1)
#     svdsinv_i = np.matmul(np.matmul(solssvd_i[0], np.diag(1/solssvd_i[1])),solssvd_i[2]) 
#     svdsinv_matrices.append(svdsinv_i)

# svdsinv = np.stack(svdsinv_matrices)
# svdsinv


# Sparse Linear System

# In[24]:


# svdsinv_matrices = []
# for i in range(input_M.shape[0]):
#     input_M_i = input_M[i]
#     input_M_i = np.vstack([input_M_i,[np.repeat(1,n)]])
#     input_M_i = scipy.sparse.csr_array(input_M_i)
#     sol2 = scipy.sparse.linalg.lsqr(input_M_i, np.eye(n + 1)[i_0 - 1] - np.eye(n + 1)[j_0 - 1])
#     svdsinv_matrices.append(svdsinv_i)

# svdsinv = np.stack(svdsinv_matrices)
# svdsinv


# Truncated SVD

# In[25]:


# solsvd = np.linalg.svd(input_M)
# singular_values = np.concatenate(((1/solsvd[1])[:,range(n-1)], np.zeros((L//S,1))), axis=1)

# singular_values_diag = np.eye(n)[None, :, :] * singular_values[:, None, :]

# # Perform batch matrix multiplication
# result = np.einsum('ijk,ikl->ijl', solsvd[0], singular_values_diag)  # Using einsum
# result = np.einsum('ijk,ikl->ijl', result, solsvd[2])  # Using einsum
# result


# In[26]:


# results = estimate_pi(networks, A, splitted_dense_data)


# ### Estimate $Q_{i_0 j_0}(\Omega)$ and its variance

# Estimate $\hat{Q}_{i_0j_0}^{(s)}(\Omega)$

# In[27]:


def estimate_Q(networks, results, splitted_dense_data):
    estimated_Q_domain_ij = np.zeros((D,n,n))
    trivial_estimated_Q_domain_ij = np.zeros((D,n,n))
    # Iterate over domain, i_0 and j_0
    for domain in range(D):
        for j_0, i_0 in itertools.combinations(range(n), 2):
            estimated_Q_list = []
            trivial_estimated_Q_list = []
            for s in range(S):
                sum_estimated_q = 0
                sum_trivial_estimated_q = 0
                for data in splitted_dense_data[s]:
                    within_bounds = (data[4] == domain)
                    within_bounds = within_bounds.astype(float)

                    theta_i0 = networks[s][i_0](torch.tensor(data[2], dtype=torch.float, device=device)) # change i_0
                    theta_j0 = networks[s][j_0](torch.tensor(data[2], dtype=torch.float, device=device))
                    theta_i0 = theta_i0.detach().cpu().numpy().squeeze()
                    theta_j0 = theta_j0.detach().cpu().numpy().squeeze()

                    pi_i = results[s,data[0],data[1],domain,i_0,j_0,:,data[0]]
                    pi_j = results[s,data[0],data[1],domain,i_0,j_0,:,data[1]]

                    theta_i = networks[s][data[0]](torch.tensor(data[2], dtype=torch.float, device=device))
                    theta_j = networks[s][data[1]](torch.tensor(data[2], dtype=torch.float, device=device))
                    theta_i = theta_i.detach().cpu().numpy().squeeze()
                    theta_j = theta_j.detach().cpu().numpy().squeeze()

                    # Compute estimated_q 
                    estimated_q = within_bounds * (theta_i0 - theta_j0) + (pi_i - pi_j) * (data[3] - np.exp(theta_i - theta_j) / (np.exp(theta_i - theta_j) + 1))
                    trivial_estimated_q = within_bounds * (theta_i0 - theta_j0)
                    sum_estimated_q += estimated_q
                    sum_trivial_estimated_q += trivial_estimated_q
                estimated_Q_s = sum_estimated_q.sum() / len(splitted_dense_data[s]) / L * S
                trivial_estimated_Q_s = sum_trivial_estimated_q.sum() / len(splitted_dense_data[s]) / L * S
                estimated_Q_list.append(estimated_Q_s)
                trivial_estimated_Q_list.append(trivial_estimated_Q_s)
            estimated_Q = np.mean(estimated_Q_list)
            trivial_estimated_Q = np.mean(trivial_estimated_Q_list)
            print(estimated_Q)
            print(trivial_estimated_Q)

            estimated_Q_domain_ij[domain][i_0][j_0] = estimated_Q
            trivial_estimated_Q_domain_ij[domain][i_0][j_0] = trivial_estimated_Q
    return estimated_Q_domain_ij, trivial_estimated_Q_domain_ij


# In[28]:

# estimated_Q = estimate_Q(networks, results, splitted_dense_data)


# Estimate $\hat{\sigma}^{(s)}(A)$

# In[29]:


# def estimate_Sigma(networks, A, splitted_dense_data):
#     estimated_Sigma_list = []
#     for s in range(S):
#         sum_estimated_sigma = 0
#         for data in splitted_dense_data[s]:
#             within_bounds = (data[2].sum(axis=1) / np.sqrt(50) > domain_lower_bound)
#             within_bounds = within_bounds.astype(float)

#             sum_adjacent_i0 = 0
#             sum_adjacent_j0 = 0

#             # Iterate over nodes adjacent to i_0
#             for j in range(n):
#                 if A[i_0 - 1, j] == 1:
#                     sub_theta_i0_j = networks[s][i_0 - 1](torch.tensor(data[2], dtype=torch.float, device=device)) - networks[s][j](torch.tensor(data[2], dtype=torch.float, device=device))
#                     sum_adjacent_i0 += torch.exp(sub_theta_i0_j) / (torch.exp(sub_theta_i0_j) + 1) ** 2
            
#             # Iterate over nodes adjacent to j_0
#             for i in range(n):
#                 if A[j_0 - 1, i] == 1:
#                     sub_theta_j0_i = networks[s][j_0 - 1](torch.tensor(data[2], dtype=torch.float, device=device)) - networks[s][i](torch.tensor(data[2], dtype=torch.float, device=device))
#                     sum_adjacent_j0 += torch.exp(sub_theta_j0_i) / (torch.exp(sub_theta_j0_i) + 1) ** 2  

#             sum_adjacent_i0 = sum_adjacent_i0.detach().cpu().numpy().squeeze()
#             sum_adjacent_j0 = sum_adjacent_j0.detach().cpu().numpy().squeeze()

#             # Compute estimated_sigma
#             estimated_sigma = within_bounds * (1.0 / sum_adjacent_i0 + 1.0 / sum_adjacent_j0)
#             sum_estimated_sigma += estimated_sigma
#         estimated_Sigma_s = sum_estimated_sigma.sum() / len(splitted_dense_data[s]) / L * S
#         estimated_Sigma_list.append(estimated_Sigma_s)
#     return estimated_Sigma_list


# In[30]:


# def old_estimate_Sigma(networks, A, splitted_dense_data):
#     estimated_Sigma_list = []
#     for s in range(S):
#         sum_estimated_sigma = 0
#         for data in splitted_dense_data[s]:
#             within_bounds = (data[2].sum(axis=1) / np.sqrt(50) > domain_lower_bound)
#             within_bounds = within_bounds.astype(float)

#             sum_adjacent_i0 = 0
#             sum_adjacent_j0 = 0

#             # Iterate over nodes adjacent to i_0
#             for j in range(n):
#                 if A[i_0 - 1, j] == 1:
#                     sub_theta_i0_j = networks[s][i_0 - 1](torch.tensor(data[2], dtype=torch.float, device=device)) - networks[s][j](torch.tensor(data[2], dtype=torch.float, device=device))
#                     sum_adjacent_i0 += torch.exp(sub_theta_i0_j) / (torch.exp(sub_theta_i0_j) + 1) ** 2
            
#             # Iterate over nodes adjacent to j_0
#             for i in range(n):
#                 if A[j_0 - 1, i] == 1:
#                     sub_theta_j0_i = networks[s][j_0 - 1](torch.tensor(data[2], dtype=torch.float, device=device)) - networks[s][i](torch.tensor(data[2], dtype=torch.float, device=device))
#                     sum_adjacent_j0 += torch.exp(sub_theta_j0_i) / (torch.exp(sub_theta_j0_i) + 1) ** 2  

#             sum_adjacent_i0 = sum_adjacent_i0.detach().cpu().numpy().squeeze()
#             sum_adjacent_j0 = sum_adjacent_j0.detach().cpu().numpy().squeeze()

#             # Compute estimated_sigma
#             estimated_sigma = within_bounds * (1.0 / sum_adjacent_i0 + 1.0 / sum_adjacent_j0)
#             sum_estimated_sigma += estimated_sigma
#         estimated_Sigma_s = sum_estimated_sigma.sum() / len(splitted_dense_data[s]) / L * S
#         estimated_Sigma_list.append(estimated_Sigma_s)
#     return estimated_Sigma_list


# In[ ]:


def new_estimate_Sigma(splitted_dense_data, pis):
    estimated_Sigma_ij = np.zeros((n,n,S))
    for j_0, i_0 in itertools.combinations(range(n), 2):
        estimated_Sigma_list = []
        for s in range(S):
            sum_pi = 0
            for data in splitted_dense_data[s]:
                # Iterate over nodes adjacent to i_0
                sum_pi_i_j = (pis[s,data[0],data[1],:,i_0,j_0,:,i_0] - pis[s,data[0],data[1],:,i_0,j_0,:,j_0]).sum() # change i_0 as well
                sum_pi += sum_pi_i_j

            # Compute estimated_sigma
            estimated_Sigma_s = sum_pi / len(splitted_dense_data[s]) ** 2 / L * S
            estimated_Sigma_list.append(estimated_Sigma_s)
        estimated_Sigma_ij[i_0][j_0] = estimated_Sigma_list
    return estimated_Sigma_ij


# In[31]:


# estimated_Sigma_ij = new_estimate_Sigma(splitted_dense_data, results)


# Estimate $\hat{V}_{i_0j_0}^{(s)}(\Omega)$

# In[32]:


def estimate_V(networks, splitted_dense_data, estimated_Q, estimated_Sigma_ij):
    estimated_V_domain_ij = np.zeros((D,n,n))
    for domain in range(D):
        for j_0, i_0 in itertools.combinations(range(n), 2):
            estimated_V_list = []
            for s in range(S):
                sum_estimated_v = 0
                for data in splitted_dense_data[s]:
                    within_bounds = (data[4] == domain)
                    within_bounds = within_bounds.astype(float)

                    sub_theta_i0_j0 = networks[s][i_0](torch.tensor(data[2], dtype=torch.float, device=device)) - networks[s][j_0](torch.tensor(data[2], dtype=torch.float, device=device))
                    sub_theta_i0_j0 = sub_theta_i0_j0.detach().cpu().numpy().squeeze()
                    
                    # Compute estimated_v
                    estimated_v = ((within_bounds * sub_theta_i0_j0 - estimated_Q[domain][i_0][j_0]) ** 2)
                    sum_estimated_v += estimated_v
                estimated_V_s = sum_estimated_v.sum() / len(splitted_dense_data[s]) ** 2 / L ** 2 * S ** 2 + estimated_Sigma_ij[i_0][j_0][s] / L
                estimated_V_list.append(estimated_V_s)
            estimated_V = np.mean(estimated_V_list)
            print(estimated_V)
            estimated_V_domain_ij[domain][i_0][j_0] = estimated_V
    return estimated_V_domain_ij


# In[33]:


# estimated_V = estimate_V(networks, splitted_dense_data, estimated_Q[0], estimated_Sigma_ij)


# ### Confidence Compuation

# In[34]:


# is_in_CI = (Q < estimated_Q[0] + 1.96 * np.sqrt(estimated_V)) & (Q > estimated_Q[0] - 1.96 * np.sqrt(estimated_V))

# # Print CI
# print("Estimate of Q: ", estimated_Q[0])
# print("Estimate of Variance: ", estimated_V)
# print(f"Confidence Interval of Q: ({estimated_Q[0] - 1.96 * np.sqrt(estimated_V)}, {estimated_Q[0] + 1.96 * np.sqrt(estimated_V)})")
# print("Value of Q by MC: ", Q)
# print("Whether Q is in the 95% CI: ", is_in_CI)


# In[35]:


def experiment():
    # Generate Data
    ss0 = ['anatomy', 'clinical_knowledge', 'college_biology', 'college_medicine', 'medical_genetics']
    vnames = ['gpt3', 'alpaca', 'gpt4o', 'llama1', 'llama2']

    A = generate_erdos_renyi_graph(n, p)
    X = get_X(ss0)
    X_ijl, Y_ijl, splitted_dense_data = generate_X_Y(A, L, ss0, vnames, X)
    leave_one_out_dense_data = generate_LOO(splitted_dense_data)

    # Estimate Theta
    networks = initialize_networks(n)
    optimizer = initialize_optimizer(networks)
    estimate_theta(networks, optimizer, leave_one_out_dense_data, splitted_dense_data)

    # Estimate pi
    pis = estimate_pi(networks, A, splitted_dense_data)

    # Estimate Q and V
    estimated_Q, trivial_estimated_Q = estimate_Q(networks, pis, splitted_dense_data)
    new_estimated_Sigma_list = new_estimate_Sigma(splitted_dense_data, pis)
    new_estimated_V = estimate_V(networks, splitted_dense_data, estimated_Q, new_estimated_Sigma_list)

    # Store results
    for domain in range(D):
        Q_df = pd.DataFrame(estimated_Q[domain])
        V_df = pd.DataFrame(new_estimated_V[domain])
        Q_df.to_csv(f"Q_llm_domain_{ss0[domain]}_n_{n}_p_{p}_L_{L}_lr_{lr}_steps_{num_epochs}.csv", index=False, header=vnames)
        V_df.to_csv(f"V_llm_domain_{ss0[domain]}_n_{n}_p_{p}_L_{L}_lr_{lr}_steps_{num_epochs}.csv", index=False, header=vnames)


# In[36]:


# with open(f"results_2_n_{n}_p_{p}_L_{L}_lr_{lr}_steps_{num_epochs}.csv", 'a', newline='') as file:
#         writer = csv.writer(file)
#         columns = ["Estimation error", "Trivial estimation error", "Old CI length", "New CI length", "# within CI", "Empirical coverage rate"]
#         writer.writerow(columns)


experiment()


# In[ ]:




