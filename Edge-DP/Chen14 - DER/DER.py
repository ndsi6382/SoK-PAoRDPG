import numpy as np
import random
import math
import itertools
import sys, resource
import pickle
import networkx as nx
import gc
import scipy.sparse as sp
from tqdm import tqdm
MAX_RECURSION = 100
sys.setrecursionlimit(MAX_RECURSION)

def compute_centrality(G, chunk_size=4096):
    V = G.shape[0]
    #assert A.shape[1] == V
    center = (V + 1) // 2
    norm = V - 2
    total = 0
    j_coords = np.abs(np.arange(V) - center)  # shape (V,)
    for i_start in tqdm(range(0, V, chunk_size)):
        i_end = min(i_start + chunk_size, V)
        A_block = G[i_start:i_end, :]  # shape (chunk, V)
        i_coords = np.abs(np.arange(i_start, i_end).reshape(-1, 1) - center)  # shape (chunk, 1)
        dist = i_coords + j_coords  # shape (chunk, V)
        total += np.sum(A_block * dist)
    return total / norm

#def compute_centrality(G):
#    cost = 0; n = len(G)
#    center = math.ceil(n/2)
#    m = np.abs(np.arange(n).reshape(-1, 1) - center) + np.abs(np.arange(n).reshape(1, -1) - center)   
#    M = sp.csr_matrix(m)
#    del m; gc.collect()
#    return sp.csr_matrix(G).multiply(m).sum()

def swap_set(graph):
    swappings = []
    for i in range(0,len(graph)):
        swappings.append(i)
    # random.shuffle(swappings)
    return [(swappings[i], swappings[i+1]) for i in range(0, len(swappings)-1, 2)]

def minimize_centrality(graph,minimum_centrality,pairs,epsilont):
    t = 1
    epsilon = 1/epsilont
    for i in range(t):
        j = 0
        for labels in tqdm(pairs):
            label1 = labels[0]
            label2 = labels[1]
            j = j+1
            affected = set()
            r1 = graph[label1].copy()
            r2 = graph[label2].copy()
            c1 = graph[:,label1].copy()
            c2 = graph[:,label2].copy()
            graph[[label1, label2]] = graph[[label2, label1]]
            graph[:, [label1, label2]] = graph[:, [label2, label1]]
            for i in range(len(graph)):
                if(graph[label1][i]!=r1[i]):
                    affected.add((label1,i))
                if(graph[label2][i]!=r2[i]):
                    affected.add((label2,i))
                if(graph[i][label1]!=c1[i]):
                    affected.add((i,label1))
                if(graph[i][label2]!=c2[i]):
                    affected.add((i,label2))
            del_centrality = 0
            for items in affected:
                if(graph[items[0]][items[1]]==1):
                    del_centrality = del_centrality+1*(abs(items[0]+1-len(graph)/2)+abs(items[1]+1-len(graph)/2))/(len(graph)-2)
                else:
                    del_centrality = del_centrality-1*(abs(items[0]+1-len(graph)/2)+abs(items[1]+1-len(graph)/2))/(len(graph)-2)
            if(del_centrality<0 and epsilon>=abs(del_centrality) and epsilon!=0):
                epsilon = epsilon-abs(del_centrality)
                minimum_centrality = minimum_centrality+del_centrality
            else:
                graph[:, [label1, label2]] = graph[:, [label2, label1]]
                graph[[label1, label2]] = graph[[label2, label1]]

def summary_matrix(graph):
    return graph.cumsum(axis=0).cumsum(axis=1)

def generate_permutations(matrix_size):
    permutations = list(itertools.product([0, 1], repeat=matrix_size[0]*matrix_size[1]))
    return [np.array(perm).reshape(matrix_size) for perm in permutations]

from decimal import Decimal, getcontext
getcontext().prec = 64

class QuadTreeNodeOpt:
    def __init__(self, matrix, x, y, size, threshold, cnt_summary, privacy_budget):
        self.x = x  # x-coordinate of the top-left corner of the node's region
        self.y = y  # y-coordinate of the top-left corner of the node's region
        self.size = size  # size of the node's region
        self.threshold = threshold  # threshold value for considering a leaf node
        self.privacy_budget = privacy_budget  # privacy budget
        self.value = self.compute_value(matrix, x, y, size, cnt_summary)
        self.children = None  # children nodes (if any)
        self.height = None

    def compute_value(self, matrix, x, y, size, cnt_summary):
        temp_noise = cnt_summary[x + size - 1][y + size - 1]
        if (x - 1) >= 0:
            temp_noise -= cnt_summary[x - 1][y + size - 1]
        if (y - 1) >= 0:
            temp_noise -= cnt_summary[x + size - 1][y - 1]
        if (x - 1) >= 0 and (y - 1) >= 0:
            temp_noise += cnt_summary[x - 1][y - 1]

        # Adding Laplace noise for differential privacy
        sensitivity = 1  # Sensitivity of the computation
        if(self.privacy_budget==0):
            return temp_noise
        scale = sensitivity / self.privacy_budget
        noise = np.random.laplace(0, scale)    #INTODUCING LAPLACE NOISE
        temp_noise += noise
        return round(temp_noise)

def build_quadtree_opt(matrix, x, y, size, threshold, cnt_summary, privacy_budget, parent_height=None):
    node = QuadTreeNodeOpt(matrix, x, y, size, threshold, cnt_summary, privacy_budget)
    if parent_height is None:
        node.height = 0
    else:
        node.height = parent_height + 1

    if threshold == 0:
        threshold = 0.1 * len(matrix) * len(matrix) / pow(4, node.height)
        #a = Decimal(0.9 * len(matrix) * len(matrix))
        #b = Decimal(pow(4, node.height))
        #threshold = a/b

    if node.value <= threshold or (parent_height and parent_height > 16-1): # Base case: leaf node
        #print(parent_height)
        return node

    if parent_height and parent_height > 16-1:
        print(parent_height)

    node.children = [ # Recursive case: non-leaf node
        build_quadtree_opt(matrix, x, y, size // 2, threshold, cnt_summary, privacy_budget, node.height),  # NW quadrant
        build_quadtree_opt(matrix, x, y + size // 2, size - size // 2, threshold, cnt_summary, privacy_budget, node.height),  # NE quadrant
        build_quadtree_opt(matrix, x + size // 2, y, size // 2, threshold, cnt_summary, privacy_budget, node.height),  # SW quadrant
        build_quadtree_opt(matrix, x + size // 2, y + size // 2, size - size // 2, threshold, cnt_summary, privacy_budget, node.height)  # SE quadrant
    ]
    return node


def get_leaf_regions(node):
    if node.children is None:# Base case: leaf node
        return [(node.x, node.y, node.size)]
    leaf_regions = []
    for child in node.children:
        leaf_regions.extend(get_leaf_regions(child))
    return leaf_regions


def edge_rearrangement(epsilon_A,graph,leaf_regions,edge_rearr1):
    cnt = 0
    for region in tqdm(leaf_regions):
        x1 = region[:2][0]
        y1 = region[:2][1]
        size = region[2]
        groups = np.zeros(size*size+1)
        cnt = cnt+1
        matrix_size = (size,size)
        if(size<=3):          
            permutations = generate_permutations(matrix_size)
            val_prop = np.zeros((size*size+1,4))
            R1 = []
            R0 = []
            
            for permute in permutations:
                x = 0
                y = 0
                z = 0
                w = 0
                for i in range(size):
                    for j in range(size):
                        if(graph[i+x1][j+y1]==1 and permute[i][j]==1):
                            w = w+1
                        elif(graph[i+x1][j+y1]==1 and permute[i][j]==0):
                            z = z+1
                        elif(graph[i+x1][j+y1]==0 and permute[i][j]==1):
                            y = y+1
                        elif(graph[i+x1][j+y1]==0 and permute[i][j]==0):
                            x = x+1
                        if(graph[i+x1][j+y1]==1):
                            R1.append([i+x1,j+y1])
                        else:
                            R0.append([i+x1,j+y1])
                score = x+w
                val_prop[score] = [x,y,z,w]
                groups[score] = groups[score]+1
            total = 0
    
            for i in range(size*size+1):
                groups[i] = groups[i]*math.exp(i*epsilon_A/2)       #USING EXPONENTIAL MECHAAANISM
    
            for items in groups:
                total = total+items
            for i in range(size*size+1):
                groups[i] = groups[i]/total
    
            scores = []
            for i in range(size*size+1):
                scores.append(i);
            score_selected = random.choices(scores,weights = groups,k = 1)[0] #SELECTING A SCORE BASED ON PROB. BY EXPONENTIAL MECHANISM 
            random.shuffle(R1)
            selected_indices1 = R1[:int(val_prop[score_selected][3])]
            random.shuffle(R0)
            selected_indices0 = R0[:int(val_prop[score_selected][1])]
            for x,y in selected_indices1:    #FIRSTLY ASSIGNING 1S TO ENSURE SCORE SELECTED IS ACHIEVED
                edge_rearr1[x][y] = 1
            for x,y in selected_indices0:    #IF SIMPLY NUMBER OF 1s IS INSUFFICIENT THEN ASSIGN 1S IF MORE NEED TO BE ALLOCATED TO 0s.
                edge_rearr1[x][y] = 1
        else:                    #SAMPLING FROM UNIFORM DISTRIBUTION
            selected_values = []
            region_min_x = x1
            region_max_x = x1+size
            region_min_y = y1
            region_max_y = y1+size
            number_ones = 0
            for i in range(size):
                for j in range(size):
                    if(graph[x1+i][y1+j]==1):
                        number_ones+=1
            for _ in range(number_ones):
                x2 = random.uniform(region_min_x, region_max_x)
                y2 = random.uniform(region_min_y, region_max_y)
                selected_values.append((x2, y2))
            for value in selected_values:
                x2 = value[0]
                y2 = value[1]
                edge_rearr1[int(x2)][int(y2)] = 1

def DER(graph, epsilon, adaptive=False):
    EpsilonI = epsilon * 0.01
    EpsilonE = epsilon * 0.395
    EpsilonA = epsilon * 0.595
    n = graph.shape[0]
    global cnt_summary_global, leaf_regions_global
    pairs = swap_set(graph)
    centrality = compute_centrality(graph) # first progress bar
    minimize_centrality(graph,centrality,pairs,EpsilonI) # second progress bar
    cnt_summary_global = summary_matrix(graph)
    #cnt_summary_global = cnt_summary
    if adaptive:
        threshold = 0
    else:
        threshold = 4
    quadtree_root_opt1 = build_quadtree_opt(graph, 0, 0, len(graph), threshold, cnt_summary_global, EpsilonE)
    del cnt_summary_global; gc.collect()
    leaf_regions_global = get_leaf_regions(quadtree_root_opt1)
    del quadtree_root_opt1; gc.collect()
    #leaf_regions_global = leaf_regions 
    #del quadtree_root_opt1, cnt_summary_global; gc.collect() #cnt_summary leaf_regions
    edge_rearr1 = np.zeros((n,n))#new_zero_matrix(graph)
    edge_rearrangement(EpsilonA, graph,leaf_regions_global, edge_rearr1) # last progress bar
    del graph, leaf_regions_global; gc.collect()
    return edge_rearr1


import time, sys
name = sys.argv[1]
epsilon = float(sys.argv[2])
trial = int(sys.argv[3])
G = nx.Graph()
with open(f'../Data/{name}/nx_adj.pkl', 'rb') as f:
    G = pickle.load(f)
#A = nx.to_numpy_array(G)
#for epsilon in [0.75, 1, 1.5, 2, 3, 4.5, 6.5, 9, 12, 16, 20]:#0.75 # 0.5 was removed for now. Max depth is 16
#    for trial in range(2,10):
ti = time.time()
np.random.seed(trial)
print(f'Epsilon {epsilon:.2f}, Trial {trial+1}/10... ')#, end="")
A2 = DER(nx.to_numpy_array(G), epsilon)
del G; gc.collect()
G2 = nx.from_numpy_array(np.maximum(A2, A2.T), create_using=nx.Graph)
del A2; gc.collect()
with open(f"./result/{name}/nx_adj_eps{epsilon}_i{trial}.pkl", "wb") as f:
    pickle.dump(G2, f)
print(f"Complete! (Took {(time.time()-ti):.3f} seconds) ")#, end="")
print(G2)
del G2; gc.collect()
