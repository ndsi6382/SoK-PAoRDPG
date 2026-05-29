import numpy as np
from numpy.linalg import qr, pinv
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
import networkx as nx
import pickle

def PrivCom(A, epsilon=1, seed=0):
    np.random.seed(seed)
    n = A.shape[0]
    k = n // 8 #min(256, n // 4)
    delta = 1e-5
    alpha = 5 * 10e-6
    gamma = min(1000, n * k)
    eta = 1 / np.sqrt(gamma)
    #print(gamma, eta)
    
    # LINE 1 Approximate Katz index
    beta = 0.01
    h = 2
    H_tilde = np.zeros_like(A)
    for l in range(1, 2*h + 2):
        H_tilde += (beta**l) * np.linalg.matrix_power(A, l) 
    
    try:
        eigvals, eigvecs = eigsh(H_tilde, k=k, which="LA", tol=1e-12)
    except ArpackNoConvergence as e:
        eigvecs = e.eigenvectors
        eigvals = e.eigenvalues
        k = eigvecs.shape[1]
    print(f"K: {k}")
    # LINE 3
    lambda_hat = eigvals + np.random.normal(0, np.sqrt(2*np.log(1.25/delta)) / (epsilon/2), size=k) 
    
    # Corollary 1, dynamic budgets)
    epsilon_vec = np.zeros(k)
    err = (1 / (epsilon/2)) * (1+epsilon) + 1/(epsilon/2)
    sqrt_lambdas = np.sqrt(np.maximum(lambda_hat, 0) + err)
    denom = np.sum(sqrt_lambdas)
    for i in range(k):
        epsilon_vec[i] = (epsilon/(2*gamma)) * (sqrt_lambdas[i] / denom if denom > 0 else 1/k)
    
    # LINE 2 Initialize eigenvector estimates
    V_hat = np.random.normal(0, 1, (n, k)) 
    spec_norm = np.linalg.norm(V_hat, 2)
    V_hat = V_hat / max(spec_norm, 1) # normalize

    # LINES 4 - 7 Oja method
    for t in tqdm(range(gamma), desc="Oja's Method"):
        # Noise scale for this iteration
        for i in range(k):
            sigma_vec = np.sqrt(2*np.log(1.25/delta)) / epsilon_vec[i]
            noise = np.random.normal(0, sigma_vec, size=(n,))
            V_hat[:, i] += eta * (H_tilde @ V_hat[:, i] + noise)
        Q, _ = qr(V_hat)
        V_hat = Q
    
    # Reconstruction
    H_noisy = V_hat @ np.diag(lambda_hat) @ V_hat.T
    H_noisy = (H_noisy + H_noisy.T) / 2

    I = np.eye(n)
    J = np.ones((n,n))
    L_tilde = -2 * pinv((I - J/n) @ H_noisy @ (I - J/n) + alpha * I)
    W_tilde = -np.minimum(L_tilde, 0)
    np.fill_diagonal(W_tilde, 0)
    return W_tilde

import sys

#datasets = ["Facebook", "LastFM"]
#epsilons = [0.5, 0.75, 1, 1.5, 2, 3, 4.5, 6.5, 9, 12, 16, 20]
#epsilons = [1, 2, 3]

data = sys.argv[1]
eps = float(sys.argv[2])
trial = int(sys.argv[3])

with open(f"../../Data/{data}/nx_adj.pkl", 'rb') as f:
    g = pickle.load(f)
h = PrivCom(nx.to_numpy_array(g), epsilon=eps, seed=trial)
o = nx.Graph(h)
with open(f"./result/{data}/nx_adj_eps{eps}_i{trial}.pkl", 'wb') as f:
    pickle.dump(o, f)

"""
for data in datasets:
    for eps in epsilons:
        for trial in range(10):
            with open(f"../../Data/{data}/nx_adj.pkl", 'rb') as f:
                g = pickle.load(f)
            h = PrivCom(nx.to_numpy_array(g), epsilon=eps, seed=trial)
            o = nx.Graph(h)
            with open(f"./result/{data}/nx_adj_eps{eps}_i{trial}.pkl", 'wb') as f:
                pickle.dump(o, f)
"""

