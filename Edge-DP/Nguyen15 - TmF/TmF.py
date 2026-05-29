import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics
import os
import time
import pickle


def get_mat(data_path):
    data = np.loadtxt(data_path)
    # initial statistics
    dat = (np.append(data[:, 0], data[:, 1])).astype(int)
    dat_c = np.bincount(dat)
    d = {}
    node = 0
    mid = []
    for i in range(len(dat_c)):
        if dat_c[i] > 0:
            d[i] = node
            mid.append(i)
            node = node + 1
    mid = np.array(mid, dtype=np.int32)

    # initial statistics
    Edge_num = data.shape[0]
    c = len(d)
    # genarated adjancent matrix
    mat0 = np.zeros([c, c], dtype=np.uint8)
    for i in range(Edge_num):
        mat0[d[int(data[i, 0])], d[int(data[i, 1])]] = 1
    # transfer direct to undirect
    mat0 = mat0 + np.transpose(mat0)
    mat0 = np.triu(mat0, 1)
    mat0 = mat0 + np.transpose(mat0)
    mat0[mat0 > 0] = 1
    return mat0, mid


def laplace_mechanism(data, epsilon):
    scale = 1.0 / epsilon
    noise = np.random.laplace(0, scale, size=data.shape)
    privacy_data = data + noise
    privacy_data = np.maximum(privacy_data, 0)
    return privacy_data


def top_m_filter(adjacency_matrix, epsilon):
    epsilon1 = epsilon/2
    epsilon2 = epsilon/2
    n = adjacency_matrix.shape[0]
    m = np.sum(adjacency_matrix) // 2  
    sanitized_matrix = np.zeros((n, n))
    m_tilde = laplace_mechanism(m, epsilon2)
    epsilon_t = np.log(n * (n - 1) / (2 * m_tilde) - 1)

    if epsilon1 < epsilon_t:
        theta = (1 / (2 * epsilon1)) * np.log(n * (n - 1) / (2 * m_tilde) - 1)
    else:
        theta = (1 / epsilon1) * np.log(n * (n - 1) / (4 * m_tilde) + 0.5 * (np.exp(epsilon1) - 1))
    n1 = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] == 1:
                A_ij = adjacency_matrix[i, j]
                A_tilde_ij = laplace_mechanism(A_ij, epsilon1)
                if A_tilde_ij > theta:
                    sanitized_matrix[i, j] = 1
                    sanitized_matrix[j, i] = 1
                    n1 += 1
    n0 = m_tilde - n1
    while n0 > 0:
        i, j = np.random.choice(n, 2, replace=False)
        if sanitized_matrix[i, j] == 0:
            sanitized_matrix[i, j] = 1
            sanitized_matrix[j, i] = 1
            n0 -= 1
    return sanitized_matrix


def main_function(data_path, eps=[1], trials=10, name="Facebook"):
    t_begin = time.time()
    mat0, mid = get_mat(data_path)
    # original graph
    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)
    print(f'Dataset: {name}')
    print(f'{mat0_graph.number_of_nodes()} nodes, {mat0_graph.number_of_edges()} edges.')

    for epsilon in eps[::-1]:
        for trial in range(trials):
            ti = time.time()
            np.random.seed(trial)
            print(f'Epsilon {epsilon:.2f}, Trial {trial+1}/{trials}... ', end="")
            mat2 = top_m_filter(mat0, epsilon)
            mat2_graph = nx.from_numpy_array(mat2, create_using=nx.Graph)
            with open(f"./result/{name}/nx_adj_eps{epsilon}_i{trial}.pkl", "wb") as f:
                pickle.dump(mat2_graph, f)
            print(f"Complete! (Took {(time.time()-ti):.3f} seconds). Generated graph has {mat2_graph.number_of_nodes()} nodes and {mat2_graph.number_of_edges()} edges.")


if __name__ == '__main__':
    data_path = "../Data/Brightkite/edge_list.txt"#"../Data/LastFM/edge_list.txt"#'../Data/Facebook/facebook_combined.txt'
    data_name = "Brightkite"#"LastFM"#"Facebook"
    main_function(data_path=data_path, eps=[0.75, 1, 1.5, 2, 3, 4.5, 6.5, 9, 12, 16, 20], trials=10, name=data_name)
