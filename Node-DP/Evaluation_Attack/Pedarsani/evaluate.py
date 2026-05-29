import networkx as nx
import pickle
import copy

dataset = "Facebook"
EPS = [1, 3, 9]
methods = {
    "PrivDPR": f"Node-DP/Zhang25 - PrivDPR/result/{dataset}",
    "PrivCom": f"Node-DP/Zhang20 - PrivCom/result/{dataset}",
}
measurements = {method: {epsilon: {metric: [] for metric in ["Edge Correctness", "Symmetric Substructure Score"]} for epsilon in EPS} for method in methods.keys()}

with open("../../Data/Facebook/nx_adj.pkl", 'rb') as f:
    G1 = pickle.load(f)

for method, directory in methods.items():
    for epsilon in EPS:
        for i in range(10):
            with open(f"../../{directory}/nx_adj_eps{epsilon}_i{i}.pkl", 'rb') as f:
                G2 = pickle.load(f)
            mapping = dict()
            with open(f"./result/{method}_eps{epsilon}_i{i}.txt") as f:
                for line in f.readlines():
                    x = line.strip().split()
                    if len(x) > 2: continue
                    mapping[int(x[0])] = int(x[1])

            # Edge Correctness
            numerator = 0
            for u1, v1 in G1.edges():
                u2 = mapping[u1]
                v2 = mapping[u2]
                if G2.has_edge(u2, v2):
                    numerator += 1
            edge_correctness = 100 * numerator / G1.number_of_edges()
            print("Edge Correctness:", edge_correctness)

            # Symmetric Substructure Score (Jaccard form)
            denominator = G1.number_of_edges() + nx.induced_subgraph(G2, [*mapping.values()]).number_of_edges() - numerator
            symmetric_substructure_score = 100 * numerator/denominator
            print("Symmetric Substructure Score:", symmetric_substructure_score)
            measurements[method][epsilon]["Edge Correctness"].append(edge_correctness)
            measurements[method][epsilon]["Symmetric Substructure Score"].append(symmetric_substructure_score)

results = copy.deepcopy(measurements)
for method in methods.keys():
    for epsilon in EPS:
        for metric in ["Edge Correctness", "Symmetric Substructure Score"]:
            results[method][epsilon][metric] = sum(measurements[method][epsilon][metric]) / len(measurements[method][epsilon][metric])

print(measurements, results)
with open("./measurements.pkl", 'wb') as f:
    pickle.dump(measurements, f)
with open("./results.pkl", 'wb') as f:
    pickle.dump(results, f)
