EPS = [0.5, 0.75, 1, 1.5, 2, 3, 4.5, 6.5, 9, 12, 16, 20]
DATAS = {
    "Facebook": {
        "feature_file": "feature_matrix_50_densest_binary-ish.txt",
        "target_file": "",
    },
    "LastFM": {
        "feature_file": "feature_matrix_50_highest_entropy.pkl",
        "target_file": "targets_reduced_5.txt"
    },
    "GitHub": {
        "feature_file": "feature_matrix_50_highest_entropy.pkl",
        "target_file": "targets.txt"
    },
    "Brightkite": {
        "feature_file": "feature_matrix_top_50.txt",
        "target_file": ""
    }
}
METHODS = {
    "PrivCom": {
        "directory": "Zhang20 - PrivCom/result/",
        "num_trials": 10,
        "order": 0,
    },
    "pi_v": {
        "directory": "Jian23 - pi_v, pi_e/node_result/",
        "num_trials": 10,
        "order": 1,
    },
    "pi_e": {
        "directory": "Jian23 - pi_v, pi_e/edge_result/",
        "num_trials": 10,
        "order": 2,
    },
}

import os, sys, math, random, gc, pickle
worker = int(sys.argv[1])
tot_worker = int(sys.argv[2])
gpu_id = ""
if len(sys.argv) >= 4:
    gpu_id = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
if gpu_id:
    os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"
else:
    os.environ["NX_CUGRAPH_AUTOCONFIG"] = "False" 
#    os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = "parallel"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count()) # // tot_worker)
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count()) # // tot_worker)
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count()) # // tot_worker)
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count()) # // tot_worker)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(os.cpu_count()) # // tot_worker)
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import networkit as nk
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, ndcg_score
from scipy.stats import wasserstein_distance
from collections import Counter
from datetime import datetime
from threading import Thread
from glob import glob
from collections.abc import Mapping
#from gcn_tasks import unprivate_link_predict, unprivate_node_classify, private_link_predict, private_node_classify

nk.engineering.setNumberOfThreads(os.cpu_count()) #// tot_worker)
nx.config.backends.parallel.active = True
nx.config.backends.parallel.n_jobs = os.cpu_count() #// tot_worker
nx.config.warnings_to_ignore.add("cache")

def merge_dicts_to_lists(dicts):
    def merge(*objs):
        if all(isinstance(obj, Mapping) for obj in objs):
            keys = set().union(*objs)
            return {
                key: merge(*(obj.get(key) for obj in objs))
                for key in keys
            }
        else: # Base case: lowest-level value → collect into list
            return list(objs)
    return merge(*dicts)

def exact_harmonic_diameter(G):
    n = G.numberOfNodes()
    sum_reciprocal_distances = 0
    count_pairs = 0
    def bfs(u):
        nonlocal sum_reciprocal_distances, count_pairs
        x = nk.distance.BFS(G, u, storePaths=False)
        x.run()
        dists = x.getDistances()
        for v in range(n):
            if u != v:
                d = dists[v]
                if d < float('inf'):
                    sum_reciprocal_distances += 1 / d
                    count_pairs += 1
    for i in tqdm(range(0,n,os.cpu_count()), desc="Harmonic Diameter. Running BFS"):
        threads = []
        for j in range(os.cpu_count()):
            if i+j >= n:
                break
            threads.append(Thread(target=bfs, args=(i+j,)))
            threads[j].start()
        for j in range(len(threads)):
            threads[j].join()
    if count_pairs == 0:
        return float('inf')  # completely disconnected
    harmonic_diameter = n * (n - 1) / sum_reciprocal_distances
    return harmonic_diameter
    

def single_metrics(G):
    Gk = nk.nxadapter.nx2nk(G)
    degseq = [d for _, d in G.degree()]
    degcnt = Counter(degseq)
    if len(G.nodes()) > 10000: # approximations
        x = nk.centrality.ApproxBetweenness(Gk, epsilon=0.01, delta=0.05) # https://dl.acm.org/doi/abs/10.1145/2556195.2556224
        x.run()
        betweenness = x.scores()
        x = nk.centrality.ApproxCloseness(Gk, nSamples=1000, epsilon=0.01, normalized=True) #https://arxiv.org/pdf/1409.0035
        x.run()
        closeness = x.scores()
    else:
        betweenness = [y for x,y in sorted((v,c) for v,c in nx.betweenness_centrality(G).items())]
        closeness = [y for x,y in sorted((v,c) for v,c in nx.closeness_centrality(G).items())]
    h_diameter = exact_harmonic_diameter(Gk)
    try:
        communities = nx.community.louvain_communities(G)
        mod = nx.community.modularity(G, communities)
    except:
        mod = 0
    del Gk; gc.collect()
    
    return {
        'local': { # node-wise
            'Degree Distribution': [degcnt.get(i, 0) for i in range(max(degcnt)+1)],
            'Clustering Coefficient Distribution': [y for x,y in sorted((v,c) for v,c in nx.clustering(G).items())],
            'Betweenness Centrality Distribution': betweenness,
            'Closeness Centrality Distribution': closeness,
        },
        'global': {
            'Density': nx.density(G),
            'Harmonic Diameter': h_diameter,
            'Assortativity': nx.degree_assortativity_coefficient(G),
            'Modularity': mod,
            'Transitivity': nx.transitivity(G),
            'Number of Nodes': G.number_of_nodes(),
            'Number of Edges': G.number_of_edges(),
        },
    }

def private_evaluation(dataset, method, epsilon):
    results_list = list()
    trials = glob(f"../{METHODS[method]['directory']}/{dataset}/nx_adj_eps{epsilon}_i*.pkl")
    for i, path in enumerate(trials):
        print(f"Evaluating {dataset}, {method}, Epsilon {epsilon}, Trial {i}.")
        with open(path, 'rb') as f:
            G = pickle.load(f)
        results = dict()
        results = single_metrics(G)
        results_list.append(results)
    measurements = merge_dicts_to_lists(results_list)
    with open(f"./Measurements/{dataset}/{method}_eps{epsilon}.pkl", 'wb') as f:
        pickle.dump(measurements, f)
    print('\n\n')

# Assuming backgrounded from the CLI. (GPU set in the command line)
# Must be launched like: CUDA_VISIBLE_DEVICES=0 python3 script.py 0 0 0
# first number is worker_id, second is total_workers, third is gpu to use.
print(f"Worker {worker} out of {tot_worker}. Using GPU {gpu_id if gpu_id else 'None'}.")
methods = [method for i, method in enumerate(METHODS.keys())]
dataset = "GitHub"
#methods = [method for method in METHODS.keys() if method not in ["DPGGAN", "DPGVAE", "ALJ"]]
methods = ["PrivCom"]
#jobs = [(method, eps) for method in methods for eps in EPS]
jobs = [(method, eps) for method in methods for eps in EPS] #[4.5, 6.5, 9, 16, 20]]
#jobs = [job for i, job in enumerate(jobs) if i % tot_worker == worker and not os.path.exists(f"./Measurements/{dataset}/{job[0]}_eps{job[1]}.pkl")]
#random.shuffle(jobs)
print(jobs)
for method, epsilon in jobs:
    if os.path.exists(f"./Measurements/{dataset}/{method}_eps{epsilon}.pkl"):
        print(f"Skipping {dataset}, {method}, Epsilon {epsilon}. File exists.")
        continue
    else:
        private_evaluation(dataset, method, epsilon)
    gc.collect()
