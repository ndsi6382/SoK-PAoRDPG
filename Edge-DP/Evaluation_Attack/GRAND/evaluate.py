import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import scipy as sp
import networkx as nx
import pickle
import copy

dataset = "Facebook"
EPS = [1, 3, 9]
methods = {
    "DER": {
        "directory": "Chen14 - DER/result/",
        "num_trials": 10
    },
    "TmF": {
        "directory": "Nguyen15 - TmF/result/",
        "num_trials": 10
    },
    "TriCycLe": {
        "directory": "CAGMDP, TriCycLe/TriCycLe_result/",
        "num_trials": 10
    },
    "LDPGen": {
        "directory": "Qin17 - LDPGen/result/",
        "num_trials": 10
    },
    "CAGMDP": {
        "directory": "CAGMDP, TriCycLe/CAGMDP_result/",
        "num_trials": 10
    },
    "DPGVAE": {
        "directory": "Yang21 - DPGGAN, DPGVAE/DPGVAE_result/",
        "num_trials": 10
    },
    "DPGGAN": {
        "directory": "Yang21 - DPGGAN, DPGVAE/DPGGAN_result/",
        "num_trials": 10
    },
    "PrivGraph": {
        "directory": "Yuan23 - PrivGraph/result/",
        "num_trials": 10
    },
}
measurements_reconstruction = {method: {epsilon: [] for epsilon in EPS} for method in methods.keys()}
measurements_non_reconstruction = {method: {epsilon: [] for epsilon in EPS} for method in methods.keys()}
    
for method in methods.keys():
    for epsilon in EPS:
        for i in range(10):
            print(f"Evaluating {method}, epsilon {epsilon}, trial {i}")
            ## LOAD ORIG
            with open(f"../../Data/{dataset}/nx_adj.pkl", 'rb') as f:
                G1 = pickle.load(f)
            # LOAD AND ADJUST RECONSTRUCTED
            with open(f"./result/nx_adj_{method}_eps{epsilon}_i{i}_0.0_DDH_0.0_0.0_0_1.pkl", 'rb') as f:
                G2 = pickle.load(f)
                if G2.number_of_nodes() < G1.number_of_nodes():
                    G2.add_nodes_from([x for x in range(G2.number_of_nodes(), G1.number_of_nodes())])
                elif G2.number_of_nodes() > G1.number_of_nodes():
                    G1.add_nodes_from([x for x in range(G1.number_of_nodes(), G2.number_of_nodes())])
            A1 = nx.to_scipy_sparse_array(G1, format='csr')
            F1 = np.sqrt(A1.multiply(A1).sum())
            A2 = nx.to_scipy_sparse_array(G2, format='csr')
            numerator = A1 - A2
            F2 = np.sqrt(numerator.multiply(numerator).sum())
            RAE_reconstruction = 100 * F2/F1

            # LOAD ORIG
            with open(f"../../Data/{dataset}/nx_adj.pkl", 'rb') as f:
                G1 = pickle.load(f)
            # LOAD AND ADJUST PRIVATE
            with open(f"../../{methods[method]['directory']}/{dataset}/nx_adj_eps{epsilon}_i{i}.pkl", 'rb') as f:
                G3 = pickle.load(f)
                G3 = nx.convert_node_labels_to_integers(G3)
                if G3.number_of_nodes() < G1.number_of_nodes():
                    G3.add_nodes_from([x for x in range(G3.number_of_nodes(), G1.number_of_nodes())])
                elif G3.number_of_nodes() > G1.number_of_nodes():
                    G1.add_nodes_from([x for x in range(G1.number_of_nodes(), G3.number_of_nodes())])
            A1 = nx.to_scipy_sparse_array(G1, format='csr')
            F1 = np.sqrt(A1.multiply(A1).sum())
            A3 = nx.to_scipy_sparse_array(G3, format='csr')
            numerator = A1 - A3
            F3= np.sqrt(numerator.multiply(numerator).sum())
            RAE_non_reconstruction = 100 * F3/F1
            
            measurements_reconstruction[method][epsilon].append(RAE_reconstruction)
            measurements_non_reconstruction[method][epsilon].append(RAE_non_reconstruction)

results_reconstruction = copy.deepcopy(measurements_reconstruction)
results_non_reconstruction = copy.deepcopy(measurements_non_reconstruction)
for method in methods.keys():
    for epsilon in EPS:
        results_reconstruction[method][epsilon] = sum(measurements_reconstruction[method][epsilon]) / len(measurements_reconstruction[method][epsilon])
        results_non_reconstruction[method][epsilon] = sum(measurements_non_reconstruction[method][epsilon]) / len(measurements_non_reconstruction[method][epsilon])

with open("./measurements_reconstruction.pkl", 'wb') as f:
    pickle.dump(measurements_reconstruction, f)
with open("./results_reconstruction.pkl", 'wb') as f:
    pickle.dump(results_reconstruction, f)

with open("./measurements_non_reconstruction.pkl", 'wb') as f:
    pickle.dump(measurements_non_reconstruction, f)
with open("./results_non_reconstruction.pkl", 'wb') as f:
    pickle.dump(results_non_reconstruction, f)
