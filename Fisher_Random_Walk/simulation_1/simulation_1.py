#!/usr/bin/env python
# coding: utf-8

# ### Import Packages

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import itertools
import copy
import csv
import scipy
np.random.seed(1234)
random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ### Parameter Setting

# In[2]:


n = 50
p = 0.1
domain_lower_bound = 0.3
domain_upper_bound = 0.8
S = 3
L = 1500
i_0 = 1
j_0 = 4
num_epochs = 5000
batch_size = 16
T = 100
lr = 0.1
device = "cuda:2"


# ### Monte Carlo Evaluation

# In[3]:


def theta_star(i: int, X: np.ndarray):
    """
    Output latent preference function of item i given prompt X

    Parameters:
    i (int): Index of items.
    X (float): Prompt.

    Returns:
    np.ndarray: Output of preference function 
    """
    return X * np.sin(i * np.pi / 8.0)


# In[4]:


# Generate 100000 random samples following uniform distribution on (0,1)
x = np.random.rand(100000)
within_bounds = (x > domain_lower_bound) & (x < domain_upper_bound)
Q = np.mean(within_bounds.astype(float) * (theta_star(i_0, x) - theta_star(j_0, x)))


# In[5]:


Q


# ### Data Generation

# Generate A

# In[6]:


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


# In[7]:


# A = generate_erdos_renyi_graph(n, p)
# A


# Generate X_ijl and Y_ijl

# In[8]:


def generate_X_Y(A, L):
    """
    Generates (X_ijl, Y_ijl) pairs for given adjacency matrix A.

    Parameters:
    A (np.ndarray): Adjacency matrix of the generated Erdos-Renyi graph.
    L (int): Number of samples for any (i, j) pair.

    Returns:
    np.ndarray: Tensor X_ijl and Y_ijl with shape (n, n, L).
    """
    X_ijl = np.zeros((n, n, L))
    Y_ijl = np.zeros((n, n, L))
    # dense_data = []
    splitted_dense_data = [[] for _ in range(S)]

    for i in range(n):
        for j in range(i):
            if A[i, j] == 1:
                X_ijl[i, j, :] = np.random.rand(L)
                X_ijl[j, i, :] = X_ijl[i, j, :]
                # prob_y[l] is the probability of Y_ijl = 1
                prob_y = np.exp(theta_star(i + 1, X_ijl[i, j, :])) / (np.exp(theta_star(i + 1, X_ijl[i, j, :])) + np.exp(theta_star(j + 1, X_ijl[i, j, :])))
                Y_ijl[i, j, :] = (np.random.rand(L) < prob_y).astype(float)
                Y_ijl[j, i, :] = Y_ijl[i, j, :]

                for s in range(S):
                    splitted_dense_data[s].append((i, j, X_ijl[i, j, s * L//S:(s + 1) * L//S], Y_ijl[i, j, s * L//S:(s + 1) * L//S]))

    return X_ijl, Y_ijl, splitted_dense_data


# In[9]:


# X_ijl, Y_ijl, splitted_dense_data = generate_X_Y(A, L)


# Generate Leave-One-Out data

# In[10]:


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


# In[11]:


# leave_one_out_dense_data = generate_LOO(splitted_dense_data)


# ### Estimate Theta

# Define the class of ReLU functions

# In[12]:


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

# In[13]:


def initialize_networks(k, input_size = 1, hidden_size = 128, output_size = 1):
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


# In[14]:


# networks = initialize_networks(n)


# Batch Generator

# In[15]:


def sample_batch(dense_data, batch_size):
    return random.sample(dense_data, batch_size)


# Define Loss Function

# In[16]:


def compute_loss(networks, batch, s):
    total_loss = 0.0
    for index in range(len(batch)):
        output_i = networks[s][batch[index][0]](torch.tensor(batch[index][2],dtype=torch.float32, device=device).reshape(-1,1))
        output_j = networks[s][batch[index][1]](torch.tensor(batch[index][2],dtype=torch.float32, device=device).reshape(-1,1))
        Y = torch.tensor(batch[index][3],dtype=torch.float32, device=device).reshape(-1,1)
        loss = -(Y * torch.log(torch.exp(output_i - output_j) / (torch.exp(output_i - output_j) + 1)) \
        + (1 - Y) * torch.log(1 - torch.exp(output_i - output_j) / (torch.exp(output_i - output_j) + 1)))
        total_loss += loss.mean()
    return total_loss / len(batch)


# Define Optimizer

# In[17]:


def initialize_optimizer(networks):
    all_parameters = []
    for s in range(S):
        for net in networks[s]:
            all_parameters.extend(list(net.parameters()))

    optimizer = optim.SGD(all_parameters, lr=lr)
    
    return optimizer


# In[18]:


# optimizer = initialize_optimizer(networks)


# Estimate $\hat{\theta}_i(\cdot)$

# In[19]:


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


# In[20]:


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
    pis = np.zeros((S, n, n, L // S, n))

    for s in range(S):
        for data in splitted_dense_data[s]:
            thetas = torch.stack([networks[s][i](torch.tensor(data[2],dtype=torch.float32, device=device).reshape(-1,1)) for i in range(n)])

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
            within_bounds = (data[2] > domain_lower_bound) & (data[2] < domain_upper_bound)
            within_bounds = within_bounds.astype(float)
            svdsinv = np.array([np.linalg.pinv(input_M[i], rcond=1e-3, hermitian=True) if within_bounds[i] == 1 else np.zeros((n, n)) for i in range(L // S)])

            pi = len(splitted_dense_data[0]) * within_bounds[:,None] * np.matmul(svdsinv, np.eye(n)[i_0 - 1] - np.eye(n)[j_0 - 1])
            pis[s][data[0]][data[1]] = pi
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
    estimated_Q_list = []
    trivial_estimated_Q_list = []
    for s in range(S):
        sum_estimated_q = 0
        sum_trivial_estimated_q = 0
        for data in splitted_dense_data[s]:
            within_bounds = (data[2] > domain_lower_bound) & (data[2] < domain_upper_bound)
            within_bounds = within_bounds.astype(float)

            theta_i0 = networks[s][i_0 - 1](torch.tensor(data[2], dtype=torch.float, device=device).reshape(-1,1))
            theta_j0 = networks[s][j_0 - 1](torch.tensor(data[2], dtype=torch.float, device=device).reshape(-1,1))
            theta_i0 = theta_i0.detach().cpu().numpy().squeeze()
            theta_j0 = theta_j0.detach().cpu().numpy().squeeze()

            pi_i = results[s,data[0],data[1],:,data[0]]
            pi_j = results[s,data[0],data[1],:,data[1]]

            theta_i = networks[s][data[0]](torch.tensor(data[2], dtype=torch.float, device=device).reshape(-1,1))
            theta_j = networks[s][data[1]](torch.tensor(data[2], dtype=torch.float, device=device).reshape(-1,1))
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
    return estimated_Q, trivial_estimated_Q


# In[28]:


# estimated_Q = estimate_Q(networks, results, splitted_dense_data)


# Estimate $\hat{\sigma}^{(s)}(A)$

# In[29]:


def old_estimate_Sigma(networks, A, splitted_dense_data):
    estimated_Sigma_list = []
    for s in range(S):
        sum_estimated_sigma = 0
        for data in splitted_dense_data[s]:
            within_bounds = (data[2] > domain_lower_bound) & (data[2] < domain_upper_bound)
            within_bounds = within_bounds.astype(float)

            sum_adjacent_i0 = 0
            sum_adjacent_j0 = 0

            # Iterate over nodes adjacent to i_0
            for j in range(n):
                if A[i_0 - 1, j] == 1:
                    sub_theta_i0_j = networks[s][i_0 - 1](torch.tensor(data[2], dtype=torch.float, device=device).reshape(-1,1)) - networks[s][j](torch.tensor(data[2], dtype=torch.float, device=device).reshape(-1,1))
                    sum_adjacent_i0 += torch.exp(sub_theta_i0_j) / (torch.exp(sub_theta_i0_j) + 1) ** 2
            
            # Iterate over nodes adjacent to j_0
            for i in range(n):
                if A[j_0 - 1, i] == 1:
                    sub_theta_j0_i = networks[s][j_0 - 1](torch.tensor(data[2], dtype=torch.float, device=device).reshape(-1,1)) - networks[s][i](torch.tensor(data[2], dtype=torch.float, device=device).reshape(-1,1))
                    sum_adjacent_j0 += torch.exp(sub_theta_j0_i) / (torch.exp(sub_theta_j0_i) + 1) ** 2  

            sum_adjacent_i0 = sum_adjacent_i0.detach().cpu().numpy().squeeze()
            sum_adjacent_j0 = sum_adjacent_j0.detach().cpu().numpy().squeeze()

            # Compute estimated_sigma
            estimated_sigma = within_bounds * (1.0 / sum_adjacent_i0 + 1.0 / sum_adjacent_j0)
            sum_estimated_sigma += estimated_sigma
        estimated_Sigma_s = sum_estimated_sigma.sum() / len(splitted_dense_data[s]) / L * S
        estimated_Sigma_list.append(estimated_Sigma_s)
    return estimated_Sigma_list


def new_estimate_Sigma(splitted_dense_data, pis):
    estimated_Sigma_list = []
    for s in range(S):
        sum_pi = 0
        for data in splitted_dense_data[s]:
            within_bounds = (data[2] > domain_lower_bound) & (data[2] < domain_upper_bound)
            within_bounds = within_bounds.astype(float)

            # Iterate over nodes adjacent to i_0
            sum_pi_i_j = (pis[s,data[0],data[1],:,i_0 - 1] - pis[s,data[0],data[1],:,j_0 - 1]).sum()
            sum_pi += sum_pi_i_j

        # Compute estimated_sigma
        estimated_Sigma_s = sum_pi / len(splitted_dense_data[s]) ** 2 / L * S
        estimated_Sigma_list.append(estimated_Sigma_s)
    return estimated_Sigma_list


# In[30]:


# estimated_Sigma_list = estimate_Sigma(networks, splitted_dense_data)


# Estimate $\hat{V}_{i_0j_0}^{(s)}(\Omega)$

# In[31]:


def estimate_V(networks, splitted_dense_data, estimated_Q, estimated_Sigma_list):
    estimated_V_list = []
    for s in range(S):
        sum_estimated_v = 0
        for data in splitted_dense_data[s]:
            within_bounds = (data[2] > domain_lower_bound) & (data[2] < domain_upper_bound)
            within_bounds = within_bounds.astype(float)

            sub_theta_i0_j0 = networks[s][i_0 - 1](torch.tensor(data[2], dtype=torch.float, device=device).reshape(-1,1)) - networks[s][j_0 - 1](torch.tensor(data[2], dtype=torch.float, device=device).reshape(-1,1))
            sub_theta_i0_j0 = sub_theta_i0_j0.detach().cpu().numpy().squeeze()
            
            # Compute estimated_v
            estimated_v = ((within_bounds * sub_theta_i0_j0 - estimated_Q) ** 2)
            sum_estimated_v += estimated_v
        estimated_V_s = sum_estimated_v.sum() / len(splitted_dense_data[s]) ** 2 / L ** 2 * S **2 + estimated_Sigma_list[s] / L
        estimated_V_list.append(estimated_V_s)
    estimated_V = np.mean(estimated_V_list)
    print(estimated_V)
    return estimated_V


# In[32]:


# estimated_V = estimate_V(networks, splitted_dense_data, estimated_Q, estimated_Sigma_list)


# ### Confidence Compuation

# In[33]:


# is_in_CI = (Q < estimated_Q + 1.96 * np.sqrt(estimated_V)) & (Q > estimated_Q - 1.96 * np.sqrt(estimated_V))

# # Print CI
# print("Estimate of Q: ", estimated_Q)
# print("Estimate of Variance: ", estimated_V)
# print(f"Confidence Interval of Q: ({estimated_Q - 1.96 * np.sqrt(estimated_V)}, {estimated_Q + 1.96 * np.sqrt(estimated_V)})")
# print("Value of Q by MC: ", Q)
# print("Whether Q is in the 95% CI: ", is_in_CI)


# In[34]:


def experiment(num_in_CI, t):
    # Generate Data
    A = generate_erdos_renyi_graph(n, p)
    while (np.abs(min(A.sum(axis=0))) < 1e-4):
        A = generate_erdos_renyi_graph(n, p)
        
    X_ijl, Y_ijl, splitted_dense_data = generate_X_Y(A, L)
    leave_one_out_dense_data = generate_LOO(splitted_dense_data)

    # Estimate Theta
    networks = initialize_networks(n)
    optimizer = initialize_optimizer(networks)
    
    estimate_theta(networks, optimizer, leave_one_out_dense_data, splitted_dense_data)

    # Estimate pi
    pis = estimate_pi(networks, A, splitted_dense_data)

    # Estimate Q and V
    estimated_Q, trivial_estimated_Q = estimate_Q(networks, pis, splitted_dense_data)
    old_estimated_Sigma_list = old_estimate_Sigma(networks, A, splitted_dense_data)
    new_estimated_Sigma_list = new_estimate_Sigma(splitted_dense_data, pis)
    old_estimated_V = estimate_V(networks, splitted_dense_data, estimated_Q, old_estimated_Sigma_list)
    new_estimated_V = estimate_V(networks, splitted_dense_data, estimated_Q, new_estimated_Sigma_list)
    is_in_CI = (Q < estimated_Q + 1.96 * np.sqrt(new_estimated_V)) & (Q > estimated_Q - 1.96 * np.sqrt(new_estimated_V))

    # Print CI
    print("Estimate of Q: ", estimated_Q)
    print("Estimate of Variance: ", new_estimated_V)
    print(f"Confidence Interval of Q: ({estimated_Q - 1.96 * np.sqrt(new_estimated_V)}, {estimated_Q + 1.96 * np.sqrt(new_estimated_V)})")
    print("Value of Q by MC: ", Q)
    print("Whether Q is in the 95% CI: ", is_in_CI)

    # Update num_in_CI
    if is_in_CI:
        new_num_in_CI = num_in_CI + 1
    else:
        new_num_in_CI =  num_in_CI

    # Store results
    with open(f"results_1_n_{n}_p_{p}_L_{L}_lr_{lr}_steps_{num_epochs}.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        row = [estimated_Q - Q, trivial_estimated_Q - Q, 2 * 1.96 * np.sqrt(old_estimated_V), 2 * 1.96 * np.sqrt(new_estimated_V), new_num_in_CI, new_num_in_CI / t]
        writer.writerow(row)

    # Return num_in_CI
    return new_num_in_CI



# In[35]:


with open(f"results_1_n_{n}_p_{p}_L_{L}_lr_{lr}_steps_{num_epochs}.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        columns = ["Estimation error", "Trivial estimation error", "Old CI length", "New CI length", "# within CI", "Empirical coverage rate"]
        writer.writerow(columns)

num_in_CI = 0
for t in range(1, T + 1):
    num_in_CI = experiment(num_in_CI, t)
    
print("Number of times that the CI contains Q: ", num_in_CI)
print("Number of experiments: ", T)


# In[ ]:




