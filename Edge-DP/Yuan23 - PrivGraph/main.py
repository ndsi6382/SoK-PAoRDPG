import community
import networkx as nx
import time
import numpy as np
from numpy.random import laplace
from sklearn import metrics
from utils import *
import os
import pickle


def main_func(data_path='../Data/Facebook/facebook_combined.txt',eps=[0.5,1,1.5,2,2.5,3,3.5], e1_r=1/3, e2_r=1/3, N=20, t=1.0, trials=10,name="Facebook"):
    t_begin = time.time()
    mat0,mid = get_mat(data_path)
    # original graph
    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)
    mat0_graph = nx.from_numpy_array(mat0,create_using=nx.Graph)
    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    print(f'Dataset: {name}')
    print(f'{mat0_graph.number_of_nodes()} nodes, {mat0_graph.number_of_edges()} edges.')
    mat0_par = community.best_partition(mat0_graph)

    for epsilon in eps:
        ti = time.time()
        e1 = e1_r * epsilon
        e2 = e2_r * epsilon
        e3_r = 1 - e1_r - e2_r
        e3 = e3_r * epsilon
        ed = e3
        ev = e3
        ev_lambda = 1/ed
        dd_lam = 2/ev
        
        for trial in range(trials):
            t1 = time.time()
            np.random.seed(trial)
            print(f'Epsilon {epsilon:.2f}, Trial {trial+1}/{trials}... ', end="")
            
            # Community Initialization
            mat1_pvarr1 = community_init(mat0,mat0_graph,epsilon=e1,nr=N,t=t)
            part1 = {}
            for i in range(len(mat1_pvarr1)):
                part1[i] = mat1_pvarr1[i]

            # Community Adjustment
            mat1_par1 = comm.best_partition(mat0_graph,part1,epsilon_EM=e2)
            mat1_pvarr = np.array(list(mat1_par1.values()))

            # Information Extraction
            mat1_pvs = []
            for i in range(max(mat1_pvarr)+1):
                pv1 = np.where(mat1_pvarr==i)[0]
                pvs = list(pv1)
                mat1_pvs.append(pvs)

            comm_n = max(mat1_pvarr) + 1
            ev_mat = np.zeros([comm_n,comm_n],dtype=np.int64)
        
            # edge vector
            for i in range(comm_n):
                pi = mat1_pvs[i]
                ev_mat[i,i] = np.sum(mat0[np.ix_(pi,pi)])
                for j in range(i+1,comm_n):
                    pj = mat1_pvs[j]
                    ev_mat[i,j] = int(np.sum(mat0[np.ix_(pi,pj)]))
                    ev_mat[j,i] = ev_mat[i,j]

            ga = get_uptri_arr(ev_mat,ind=1)
            ga_noise = ga + laplace(0,ev_lambda,len(ga))
        
            ga_noise_pp = FO_pp(ga_noise)
            ev_mat = get_upmat(ga_noise_pp,comm_n,ind=1)

            # degree sequence
            dd_s = []
            for i in range(comm_n):
                dd1 = mat0[np.ix_(mat1_pvs[i],mat1_pvs[i])]
                dd1 = np.sum(dd1,1) 
        
                dd1 = (dd1 + laplace(0,dd_lam,len(dd1))).astype(int)
                dd1 = FO_pp(dd1)
                dd1[dd1<0] = 0
                dd1[dd1>=len(dd1)] = len(dd1)-1

                dd1 = list(dd1)
                dd_s.append(dd1)

            # Graph Reconstruction
            mat2 = np.zeros([mat0_node,mat0_node],dtype=np.int8)
            for i in range(comm_n):
                # Intra-community
                dd_ind = mat1_pvs[i]
                dd1 = dd_s[i]
                mat2[np.ix_(dd_ind,dd_ind)] = generate_intra_edge(dd1)
                    
                # Inter-community
                for j in range(i+1,comm_n):
                    ev1 = ev_mat[i,j]
                    pj = mat1_pvs[j]
                    if ev1 > 0:
                        c1 = np.random.choice(pi,ev1)
                        c2 = np.random.choice(pj,ev1)
                        for ind in range(ev1):
                            mat2[c1[ind],c2[ind]] = 1
                            mat2[c2[ind],c1[ind]] = 1
                            
            mat2 = mat2 + np.transpose(mat2)
            mat2 = np.triu(mat2,1)
            mat2 = mat2 + np.transpose(mat2)
            mat2[mat2>0] = 1

            mat2_graph = nx.from_numpy_array(mat2,create_using=nx.Graph)

            # save the graph
            #file_name = './result/' +  f'{dataset_name}_eps{epsilon}_i{exper}.txt'
            write_edge_txt(mat2,mid,'tmp.txt')#filename
            G = nx.Graph()
            with open("tmp.txt", "r") as f:
                for line in f.readlines():
                    if not line[0].isdigit():
                        continue
                    v, w = line.split("\t")
                    G.add_edge(int(v), int(w))
            with open(f"./result/{name}/nx_adj_eps{epsilon}_i{trial}.pkl", "wb") as f:
                pickle.dump(G, f)
            os.remove('tmp.txt')
            print(f"Complete! (Took {(time.time()-ti):.3f} seconds). Generated graph has {mat2_graph.number_of_nodes()} nodes and {mat2_graph.number_of_edges()} edges.")


if __name__ == '__main__':
    eps = [0.5, 0.75, 1, 1.5, 2, 3, 4.5, 6.5, 9, 12, 16, 20]
    # set the ratio of the privacy budget
    e1_r = 1/3
    e2_r = 1/3
    # set the number of nodes for community initialization
    n1 = 20
    # set the resolution parameter
    t = 1.0
    data_path = '../Data/Brightkite/edge_list.txt'#'../Data/LastFM/edge_list.txt'#'../Data/Facebook/facebook_combined.txt'
    data_name = "Brightkite"
    main_func(data_path=data_path, eps=[0.5, 0.75, 1, 1.5, 2, 3, 4.5, 6.5, 9, 12, 16, 20], e1_r=e1_r, e2_r=e2_r, N=n1, t=t, trials=10, name=data_name)