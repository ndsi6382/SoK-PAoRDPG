import numpy as np
import networkx as nx
import random
from sklearn.cluster import KMeans
from tqdm import tqdm
import time
import pickle
import gc

class LDPGen():
    def __init__(self, seed: int, graph: list[list], k_groups: int = 10):
        self.seed = seed
        self.graph = graph
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.n = len(self.graph)
        self.k = k_groups
        z = [x for x in range(self.n)]
        random.shuffle(z)
        self.groups = {vertex: group % k_groups for group, vertex in enumerate(z)} # vertex: group
        self.grouprec = [] # [timestep][group_no] = {set_of_vertices_in_this_group} 
        
    def local(self, eps):
        self.degvec = [[0]*self.k for _ in range(self.n)]
        for v, nbrs in enumerate(self.graph):
            for w, weight in enumerate(nbrs):
                if weight > 0: # edge exists|
                    self.degvec[v][self.groups[w]] += 1
            self.degvec[v] += np.random.laplace(0, 1/eps, self.k)
            for i in range(self.k): # clamp degrees to min of 0, round to integers.
                self.degvec[v][i] = max(round(self.degvec[v][i]), 0)

    def server_get_k1(self, eps):
        # find k1
        degrees = [int(sum(x)) for x in self.degvec]
        degseq = [0 for _ in range(max(degrees)+1)]
        for d in degrees:
            degseq[d] += 1
        self.old_k = self.k
        self.k = 0
        for deg, cnt in enumerate(degseq):
            if cnt == 0 or deg > 50:
                continue
            p = cnt/self.n
            a = 0.5*deg
            num = a**2 - (2 * (1 + np.sqrt(5)) * a) + 1
            self.k += p * (a + num/eps)
        self.k = min(max(int(np.ceil(self.k)), 1), self.n) # 1 <= k <= n
        #print("New K: ", self.k)
    
    def server(self):
        kmeans = KMeans(n_clusters=self.k).fit(self.degvec)
        d = [set() for _ in range(self.k)]
        for vertex, group in enumerate(kmeans.labels_):
            self.groups[vertex] = int(group)
            d[int(group)].add(vertex)
        self.grouprec.append(d)


    def __estimate_degrees__(self, start, end):
        chunk = np.zeros((end-start, self.k))
        for u in range(start, end):
            x = np.zeros(self.k)
            for i in range(self.k): # group
                for j in range(self.k): # group
                    denom = len(self.grouprec[-2][j])
                    if denom == 0:
                        continue
                    else:
                        x[i] += (len(self.grouprec[-2][j].intersection(self.grouprec[-1][i])) / denom) * self.degvec[u][j]
            chunk[u-start] = x
        return (start, chunk)

    def phase_iii(self, undirected=True, parallel=True, num_workers=8):
        del self.graph; gc.collect()
        est_deg = np.zeros((self.n, self.k))
        if parallel:        
            from concurrent.futures import ProcessPoolExecutor, as_completed
            chunk_size = self.n // num_workers
            tasks = []
            with ProcessPoolExecutor(max_workers=num_workers) as exe:
                for i in range(num_workers):
                    start = i*chunk_size
                    end = (i+1) * chunk_size if i < num_workers-1 else self.n
                    tasks.append(exe.submit(self.__estimate_degrees__, start, end))
                for task in tqdm(as_completed(tasks), total=len(tasks), desc="Estimating degrees for node"):
                    start, chunk = task.result()
                    est_deg[start:start+chunk.shape[0], :] = chunk
        else:
            for u in tqdm(range(self.n), desc="Estimating degrees for node"): # user
                x = np.zeros(self.k)
                for i in range(self.k): # group
                    for j in range(self.k): # group
                        denom = len(self.grouprec[-2][j])
                        if denom == 0:
                            continue
                        else:
                            x[i] += (len(self.grouprec[-2][j].intersection(self.grouprec[-1][i])) / denom) * self.degvec[u][j]
                est_deg[u] = x

        # try to do one loop only
        del self.degvec; gc.collect()
        vertical_est_deg_sums = [sum([est_deg[w][j] for w in range(self.n)]) for j in range(self.k)] 
        cross_cache = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                cross_cache[i][j] = sum([est_deg[u][j] for u in self.grouprec[-1][i]])

        num_denoms = [len(self.grouprec[-1][j]) for j in range(self.k)]
        del self.grouprec; gc.collect()
        
        def prob(u, v):
            i = self.groups[u] # get group_no
            j = self.groups[v] # get group_no
            #numerator_denom = len(self.grouprec[-1][j])
            numerator_denom = num_denoms[j]
            if numerator_denom == 0:
                return 1
            else:
                numerator = est_deg[u][j] * vertical_est_deg_sums[j] / numerator_denom
                denominator = cross_cache[i][j] + vertical_est_deg_sums[j]
                return numerator/denominator

        #G = np.zeros((self.n, self.n)) # Assume undirected.
        G = np.random.uniform(size=(self.n,self.n))
        for x in tqdm(range(self.n), desc="Generating edges for node"):
            for y in range(self.n):
                if x == y or G[x][y] == 1 or G[x][y] == 0:
                    continue
                if G[x][y] < prob(x,y):
                    G[x][y] = 1
                    if undirected:
                        G[y][x] = 1
                else:
                    G[x][y] = 0
                    if undirected:
                        G[y][x] = 0
        return G.astype(int)
    
    def run(self, epsilon, k=10, undirected=True, parallel=True, num_workers=8):  
        self.local(epsilon/2)
        self.server_get_k1(epsilon/2)
        self.server()
        self.local(epsilon/2)
        self.server()
        a = self.phase_iii(undirected, parallel, num_workers)
        gc.collect()
        return np.maximum(a, a.T) if undirected else a
        
import pickle
import sys
import os
print(os.cpu_count(), "cores.") 

# Parameters
#eps = [0.5, 0.75, 1, 1.5, 2, 3, 4.5, 6.5, 9, 12, 16, 20]
# Note here, epsilon will be split in half for the stages, and K is kept at 10.
#trials = 10
#name="GitHub"#"Facebook"
name = sys.argv[1]
epsilon = float(sys.argv[2])
trial = int(sys.argv[3])

data_path = f"../Data/{name}/nx_adj.pkl"#"../Data/Facebook/facebook_combined.txt"
with open(data_path, 'rb') as f:
    G = pickle.load(f)
print(f"Num nodes: {len(G.nodes())}, Num edges: {len(G.edges())}")
#for epsilon in eps:
#    for trial in range(trials):#for trial in range(5,trials):
ti = time.time()
print(f'Epsilon {epsilon:.2f}, Trial {trial+1}... ')
M = LDPGen(trial, nx.adjacency_matrix(G).toarray())
del G; gc.collect()
G2 = nx.from_numpy_array(M.run(epsilon, num_workers=os.cpu_count()))
with open(f'result/{name}/nx_adj_eps{epsilon}_i{trial}.pkl', 'wb') as f: 
    pickle.dump(G2, f)
print(f"Complete! (Took {(time.time()-ti):.3f} seconds). Generated graph has {G2.number_of_nodes()} nodes and {G2.number_of_edges()} edges.")
