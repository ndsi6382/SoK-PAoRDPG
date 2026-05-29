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
    "ALJ": {
        "directory": "Ahmed16 - ALJ/result/",
        "num_trials": 10
    },
    "CAGMDP": {
        "directory": "CAGMDP, TriCycLe/CAGMDP_result/",
        "num_trials": 10
    },
    "TriCycLe": {
        "directory": "CAGMDP, TriCycLe/TriCycLe_result/",
        "num_trials": 10
    },
    "DER": {
        "directory": "Chen14 - DER/result/",
        "num_trials": 10
    },
    "TmF": {
        "directory": "Nguyen15 - TmF/result/",
        "num_trials": 10
    },
    "LDPGen": {
        "directory": "Qin17 - LDPGen/result/",
        "num_trials": 10
    },
    "DPGGAN": {
        "directory": "Yang21 - DPGGAN, DPGVAE/DPGGAN_result/",
        "num_trials": 10
    },
    "DPGVAE": {
        "directory": "Yang21 - DPGGAN, DPGVAE/DPGVAE_result/",
        "num_trials": 10
    },
    "PrivGraph": {
        "directory": "Yuan23 - PrivGraph/result/",
        "num_trials": 10
    },
    "PrivDPR": {
        "directory": "Zhang25 - PrivDPR/result/",
        "num_trials": 10
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
    os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = "parallel"
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
import netmax.influence_maximization as im
from collections import Counter
from datetime import datetime
from threading import Thread
from glob import glob
from collections.abc import Mapping
from gcn_tasks import unprivate_link_predict, unprivate_node_classify, private_link_predict, private_node_classify

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
        # leskovec, 2006, Sampling from Large Graphs, KDD'06
        # https://www.sciencedirect.com/science/article/abs/pii/S0378437100003113
        #x = nk.community.detectCommunities(Gk) # Louvain in parallel
        #communities = community_vector2partition(x.getVector())
    else:
        betweenness = [y for x,y in sorted((v,c) for v,c in nx.betweenness_centrality(G).items())]
        closeness = [y for x,y in sorted((v,c) for v,c in nx.closeness_centrality(G).items())]
    h_diameter = exact_harmonic_diameter(Gk)
    communities = nx.community.louvain_communities(G)
    community_vector = [None for _ in range(max(G.nodes())+1)]
    for com_id, com_set in enumerate(communities):
        for v in com_set:
            community_vector[v] = com_id
    x = nk.centrality.PageRank(Gk, damp=0.85, tol=1e-6)
    x.run()
    pageranks = x.scores()
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
            'Modularity': nx.community.modularity(G, communities),
            'Transitivity': nx.transitivity(G),
        },
        'meta': {
            'community_vector': community_vector, # For ARI, NMI
            'pagerank_vector': pageranks # For NDCG
        },
    }

def community_vector2partition(community_vector):
    partition = [set() for _ in range(max(community_vector)+1)]
    for v, com in enumerate(community_vector):
        partition[com].add(v)
    return partition


def comparative_metrics(G1_dict, G2_dict):
    G1_community_vector = G1_dict['meta']['community_vector']
    G2_community_vector = G2_dict['meta']['community_vector']
    next_id = max(G1_community_vector)+1
    while len(G1_community_vector) < len(G2_community_vector):
        G1_community_vector.append(next_id)
        next_id += 1
    while len(G1_community_vector) > len(G2_community_vector):
        G2_community_vector.append(next_id)
        next_id += 1
    assert len(G1_community_vector) == len(G2_community_vector)
    return {
        'ARI': adjusted_rand_score(G1_community_vector, G2_community_vector),
        'NMI': normalized_mutual_info_score(G1_community_vector, G2_community_vector), 
        'NDCG over PageRank Scores': ndcg_score(G1_dict['meta']['pagerank_vector'], G2_dict['meta']['pagerank_vector']),
    }


def influence_maximisation(G): # DO RELATIVE ERROR FOR THIS. IT's LIKE A GLOBAL STATISTIC.
    # TIM+ method (NOT degree-discount)
    # Independent Cascade.
    # see the GitHub NetMax repo for references.
    D = G.to_directed()
    im_instance = im.InfluenceMaximization(input_graph=D, agents={"0":int(D.number_of_nodes() * 0.01)}, alg='tim_p', 
                                            diff_model='ic', inf_prob='uniform', r=1000,
                                            insert_opinion=False, endorsement_policy='random', verbose=False)
    seed, spread, execution_time = im_instance.run()
    return spread

def private_evaluation(dataset, method, epsilon, with_influence_maximisation=True):
    results_list = list()
    trials = glob(f"../{METHODS[method]['directory']}/{dataset}/nx_adj_eps{epsilon}_i*.pkl")
    for i, path in enumerate(trials):
        print(f"Evaluating {dataset}, {method}, Epsilon {epsilon}, Trial {i}.")
        with open(path, 'rb') as f:
            G = pickle.load(f)
        results = dict()
        results = single_metrics(G)
        #G.__networkx_cache__.clear()
        results["utility"] = dict()
        if with_influence_maximisation:
            results["utility"]["Influence Maximisation"] = influence_maximisation(G)
        else:
            results["utility"]["Influence Maximisation"] = list()
        with open(f"./Unprivate/{dataset}.pkl", 'rb') as f:
            unprivate = pickle.load(f)
            val_pos = unprivate["meta"]["Link Prediction"]["val_pos"]
            val_neg = unprivate["meta"]["Link Prediction"]["val_neg"]
            test_pos = unprivate["meta"]["Link Prediction"]["test_pos"]
            test_neg = unprivate["meta"]["Link Prediction"]["test_neg"]
        if method in ["CAGMDP", "TriCycLe"]:
            feature_file = f"../CAGMDP, TriCycLe/{method}_result/{dataset}/np_att_eps{epsilon}_i{i}.txt"
        else:
            feature_file = f"../Data/{dataset}/{DATAS[dataset]['feature_file']}"
        scores = private_link_predict.run(f"../Data/{dataset}/nx_adj.pkl", feature_file, val_pos, val_neg, test_pos, test_neg)
        results["utility"]["Link Prediction"] = scores
        if dataset in ["LastFM", "GitHub"]:
            with open(f"./Unprivate/{dataset}.pkl", 'rb') as f:
                unprivate = pickle.load(f)
                val_mask = unprivate["meta"]["Node Classification"]["val_mask"]
                test_mask = unprivate["meta"]["Node Classification"]["test_mask"]
            scores = private_node_classify.run(f"../Data/{dataset}/nx_adj.pkl", feature_file, f"../Data/{dataset}/{DATAS[dataset]['target_file']}", val_mask, test_mask, use_node2vec=True)
            results["utility"]["Node Classification"] = scores
        results_list.append(results)
    measurements = merge_dicts_to_lists(results_list)
    with open(f"./Measurements/{dataset}/{method}_eps{epsilon}.pkl", 'wb') as f:
        pickle.dump(measurements, f)
    print('\n\n')

# Assuming backgrounded from the CLI. (GPU set in the command line)
# Must be launched like: CUDA_VISIBLE_DEVICES=0 python3 script.py 0 0 0
# first number is worker_id, second is total_workers, third is gpu to use.
print(f"Worker {worker} out of {tot_worker}. Using GPU {gpu_id if gpu_id else 'None'}.")
#dataset = "GitHub"
methods = [method for i, method in enumerate(METHODS.keys())]
#methods = []
dataset = "Brightkite"
#methods = [method for method in METHODS.keys() if method not in ["DPGGAN", "DPGVAE", "ALJ"]]
methods = ["DPGVAE"]
#jobs = [(method, eps) for method in methods for eps in EPS]
jobs = [(method, eps) for method in methods for eps in [12, 16, 1, 2, 3]]
jobs = [job for i, job in enumerate(jobs) if i % tot_worker == worker and not os.path.exists(f"./Measurements/{dataset}/{job[0]}_eps{job[1]}.pkl")]
random.shuffle(jobs)
#jobs = [("ALJ", 20)]
print(jobs)
for method, epsilon in jobs:
    if os.path.exists(f"./Measurements/{dataset}/{method}_eps{epsilon}.pkl"):
        print(f"Skipping {dataset}, {method}, Epsilon {epsilon}. File exists.")
        continue
    else:
        private_evaluation(dataset, method, epsilon, with_influence_maximisation=False)
    gc.collect()
