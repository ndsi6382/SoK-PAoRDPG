import numpy as np
import networkx as nx
import pickle
import sys
import os
from decimal import Decimal, getcontext
from scipy.optimize import minimize

getcontext().prec = 512  # set precision

def bu(c, b, k, eps, n):
    beta = lambda n: 1 - c/n
    t1 = Decimal(str(np.exp(1/b)))
    t2 = Decimal(str(beta(n))) ** (Decimal(str(-n)))
    t3 = Decimal(str(beta(n+1))) / Decimal(str(beta(n)))
    exp = (n*(n-1))/2 + (k*(k-1)/2) + k*(n+1)
    return t1*t2*(t3**Decimal(str(exp)))

def jwc_e(data_path, epsilon=1, delta=0.00001, seed=0):
    np.random.seed(seed)
    with open(data_path, 'rb') as f:
        g = pickle.load(f)
    print(g)
    b = 2/epsilon
    mu = - np.log(delta) * b
    lo=0; hi=g.number_of_nodes(); c = float('inf')
    while lo<=hi:
        c = (lo+hi)/2
        k = 2*mu
        x = float(bu(c, b, k, epsilon, g.number_of_nodes()).ln())
        if abs(x - epsilon) <= 1e-6:
            break
        elif x < epsilon:
            lo = c
        else:
            hi = c
    beta = 1 - c/g.number_of_nodes()
    edges_to_remove = [(u,w) for u,w in g.edges() if np.random.random() <= beta]
    g.remove_edges_from(edges_to_remove)
    #g.remove_nodes_from(list(nx.isolates(g)))
    print(f"Removed {len(edges_to_remove)} edges.")
    k = max(0, round(np.random.laplace(loc=mu, scale=b)))
    print(f"g:{g}, c:{c}, mu:{mu}, b:{b}, k:{k}, beta:{beta}")
    for _ in range(k):
        u = g.number_of_nodes()
        g.add_node(u)
        for w in g.nodes():
            if u != w and np.random.random() <= 1 - beta:
                g.add_edge(u,w)
    print(g)
    print()
    return g

def jwc_n(data_path, epsilon=1, seed=0):
    np.random.seed(seed)
    with open(data_path, 'rb') as f:
        g = pickle.load(f)
    p = Decimal(1) - Decimal(np.exp(epsilon))/Decimal(2)**Decimal(g.number_of_nodes())
    q = 0.01
    #print(p, q) # https://www.wolframalpha.com/input?i=solve+for+p%3A+max%28-ln%28p%29%2C+ln%28p+%2B+%28%281-p%292%5E4039%29%2F0.9%29%29+%3D+2780
    print(g)
    next_id = g.number_of_nodes()
    node_list = list(g.nodes())
    for u in node_list:
        if np.random.random() <= p:
            g.remove_node(u)
    k = np.random.geometric(q)
    for _ in range(k):
        u = next_id
        g.add_node(u)
        for w in g.nodes():
            if u != w and np.random.random() <= 0.5:
                g.add_edge(u, w)
        next_id += 1
    print(g)
    print()
    return g
        

if __name__=="__main__":
    for alg in ["node", "edge"]:
        for dataset in ["Facebook", "LastFM", "GitHub", "Brightkite"]:
            for epsilon in [0.5, 0.75, 1, 1.5, 2, 3, 4.5, 6.5, 9, 12, 16, 20]:
                for seed in range(10):
                    if alg=="node":
                        g = jwc_n(f"../../Data/{dataset}/nx_adj.pkl", epsilon, seed)
                    else:
                        g = jwc_e(f"../../Data/{dataset}/nx_adj.pkl", epsilon, 0.00001, seed)
                    with open(f"./{alg}_result/{dataset}/nx_adj_eps{epsilon}_i{seed}.pkl", 'wb') as f:
                        pickle.dump(g, f)
    