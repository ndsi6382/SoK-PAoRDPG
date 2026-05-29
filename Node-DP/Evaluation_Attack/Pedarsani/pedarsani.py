import os, sys
import networkx as nx
import pickle
import subprocess

#method = sys.argv[1]
dataset = "Facebook"
epsilon = sys.argv[1]
iteration = sys.argv[2]
##graph1 = sys.argv[4]
#graph2 = sys.argv[4]

print(epsilon, iteration)

# Open and write true graph (shouldn't need to redo this)
#with open("../../Data/Facebook/nx_adj.pkl", 'rb') as f:
#    G = pickle.load(f)
#nx.write_edgelist(G, "graph_a.txt", data=False)

# Open and write generated graph
directories = {
    "PrivDPR": f"Zhang25 - PrivDPR/result/{dataset}",
    "PrivCom": f"Node-DP/Zhang20 - PrivCom/result/{dataset}",
}

for k, v in directories.items():
    with open(f"../../{v}/nx_adj_eps{epsilon}_i{iteration}.pkl", 'rb') as f:
        G = pickle.load(f)
    filename = f"{k}_eps{epsilon}_i{iteration}.txt"
    filename_b = filename + "_b"
    print(filename)
    nx.write_edgelist(G, filename_b, data=False)
    subprocess.run([f"java -jar SecGraph/secGraphCLI.jar -m d -a Bayesian -gA graph_a.txt -gB {filename_b} -gO result/{filename}"], shell=True)
    os.remove(filename_b)
