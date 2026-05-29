from helpers import *
from Graph import Graph
from itertools import product, combinations
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
import time
from joblib import Parallel, delayed
from multiprocessing import Pool
import networkx as nx

class DeterministicAttack:
    """
    Class to perform deterministic attacks on a graph
    """

    def __init__(self, Ga, A, graph1_prop=0.0, dataset_name=None, log=False, expe_number=None):
        """
        Constructor of the class
        :param graph1: Graph object
        :param A: Adjacency matrix of the graph
        :param graph1_prop: Proportion of the graph1 in the dataset
        :param dataset_name: Name of the dataset
        :param log: Boolean to log the results
        :param expe_number: Number of the experiment
        """

        self.Gstar = Ga.copy()
        self.A = A
        self.modifications = 0
        self.dataset_name = dataset_name
        self.log = log
        self.graph1_prop = graph1_prop
        self.expe_number = expe_number

        if self.log:
            if self.dataset_name == None or self.graph1_prop == None:
                print("Error : dataset_name and/or size of known graph is not defined")
                exit()
            else:
                print("#### Deterministic attacks ####")

                self.log_file = open(f"logs/deterministics/{self.dataset_name}.csv", "a")



    def matching_attacks(self):
        modifs = 0
        A = self.A
        neighbors = [self.Gstar.neighbors(node) for node in self.Gstar.nodes]


        for i in tqdm(range(len(self.Gstar.nodes)), desc="Matching attacks"):
            neighbors_i = neighbors[i]
            for j in range(i+1, len(self.Gstar.nodes)):
                neighbors_j = neighbors[j]
                common_neighbors = neighbors_i & neighbors_j

                if A[i, j] == len(common_neighbors):
                    for node in neighbors_i.copy():
                        if self.Gstar.does_not_know_edge((j, node)):
                            self.Gstar.remove_edge((j, node))
                            modifs += 2
                    for node in neighbors_j.copy():
                        if self.Gstar.does_not_know_edge((i, node)):
                            self.Gstar.remove_edge((i, node))
                            modifs += 2


        print(f"Matching attack: {modifs} modifications")

        if self.log:
            self.log_file.write(f"neighbor_matching,{self.expe_number},{self.graph1_prop},{modifs}\n")

        # self.Gstar = reconstructed_graph
        self.modifications+=modifs



    def degree_matching_attack(self):
        modifs  = 0
        A = self.A
        nodes = [node for node in self.Gstar.nodes if A[node, node] == self.Gstar.degree(node)]
        for node in nodes:
            for other in self.Gstar.unknown_list[node].copy():
                self.Gstar.remove_edge((node, other))
                modifs += 2

        print(f"Degree matching attack: {modifs} modifications")
        self.modifications += modifs

        if self.log:
            self.log_file.write(f"degree_matching,{self.expe_number},{self.graph1_prop},{modifs}\n")




    def completion_attacks(self):
        modifs = 0
        # reconstructed_graph = self.Gstar.copy()
        A = self.A

        for i in tqdm(range(len(self.Gstar.nodes)), desc="Completion attacks"):
            neighbors_i = self.Gstar.neighbors(i)
            for j in range(i+1, len(self.Gstar.nodes)):
                neighbors_j = self.Gstar.neighbors(j)
                unknowed_edges_i = self.Gstar.unknown_list[i]
                unknowed_edges_j = self.Gstar.unknown_list[j]

                if len(neighbors_i) == A[i, j] - len(unknowed_edges_i):
                    for k in unknowed_edges_i.copy():
                        self.Gstar.add_edge((i, k))
                        self.Gstar.add_edge((j, k))
                        modifs += 4 if i != j else 2


                if len(neighbors_j) == A[i, j] - len(unknowed_edges_j):
                    for k in unknowed_edges_j.copy():
                        self.Gstar.add_edge((j, k))
                        self.Gstar.add_edge((i, k))
                        modifs += 4 if i != j else 2

        print(f"Neighbor Completion attack: {modifs} modifications")
        if self.log:
            self.log_file.write(f"neighbor_completion,{self.expe_number},{self.graph1_prop},{modifs}\n")

        # self.Gstar = reconstructed_graph
        self.modifications += modifs


    def degree_completion_attack(self):
        modifs = 0
        A = self.A

        for i in tqdm(range(len(self.Gstar.nodes)), desc="Degree completion attack"):
            if len(self.Gstar.neighbors(i)) == A[i, i] - len(self.Gstar.unknown_list[i]):
                for j in self.Gstar.unknown_list[i].copy():
                    self.Gstar.add_edge((i, j))
                    modifs += 2

        print(f"Degree completion attack: {modifs} modifications")
        self.modifications += modifs

        if self.log:
            self.log_file.write(f"degree_completion,{self.expe_number},{self.graph1_prop},{modifs}\n")




    def degree_attack(self):
        modifs = 0
        # reconstructed_graph = self.Gstar.copy()
        A = self.A
        degree_sequence = np.diag(A)
        degrees = np.unique(degree_sequence)
        count = 0


        for i in tqdm(range(A.shape[0]), desc="Degree attack"):
            
            degree = np.sum(A[i])
            candidate = None
            possibilities = np.where((degree_sequence <= degree - A[i, i] + 1))[0]   
            non_possibilities = np.where((degree_sequence > degree - A[i, i] + 1))[0]

            if A[i, i] <= 2:   # optmise for small computation
                count += 1
                for comb in combinations(possibilities, A[i, i]):
                    sum_of_degrees = 0

                    for k in comb:
                        sum_of_degrees += A[k, k]

                    if sum_of_degrees == degree:
                        if candidate == None:
                            candidate = comb
                        else:
                            candidate = None
                            break

                if candidate != None:
                    for j in candidate:
                        if self.Gstar.does_not_know_edge((i, j)):
                            self.Gstar.add_edge((i, j))
                            modifs += 2
            
            for j in non_possibilities:
                if self.Gstar.does_not_know_edge((i, j)):
                    self.Gstar.remove_edge((i, j))
                    modifs += 2


        print(f"Degree attack: {modifs} modifications")
        if self.log:
            self.log_file.write(f"degree,{self.expe_number},{self.graph1_prop},{modifs}\n")

        self.modifications += modifs



    def triangle_attack(self):
        modifs = 0
        A = self.A
        edges = self.Gstar.edges_no_repeat()
        for [u, v] in tqdm(edges, desc="Triangle attack"):
            if A[u, v] == 0 or A[u, u] < 2 or A[v, v] < 2:
                continue

            g2_u = np.where(A[u] > 0)[0]
            g2_u = np.setdiff1d(g2_u, u)
            g2_v = np.where(A[v] > 0)[0]
            g2_v = np.setdiff1d(g2_v, v)

            candidates = np.intersect1d(g2_u, g2_v)
            candidates = np.setdiff1d(candidates, self.Gstar.common_neighbors((u, v)))

            if len(candidates) == A[u, v] - len(self.Gstar.common_neighbors((u, v))):
                for w in candidates:
                    if self.Gstar.does_not_know_edge((u, w)):
                        self.Gstar.add_edge((u, w))
                        modifs += 2
                    if self.Gstar.does_not_know_edge((v, w)):
                        self.Gstar.add_edge((v, w))
                        modifs += 2

        print(f"Triangle attack: {modifs} modifications")
        if self.log:
            self.log_file.write(f"triangle,{self.expe_number},{self.graph1_prop},{modifs}\n")

        # self.Gstar = reconstructed_graph
        self.modifications += modifs

    
        
    def rectangle_attack(self, degrees=[1, 2, 3, 4, 5]):
        modifs = 0
        # reconstructed_graph = self.Gstar.copy()
        A = self.A

        common_neighbors_matrix = self.Gstar.common_neighbors_matrix()
        neighbors = [self.Gstar.neighbors(node) for node in self.Gstar.nodes]
        candidates = [node for node in self.Gstar.nodes if self.Gstar.unknown_list[node] == set()]
        others = self.Gstar.nodes - set(candidates)

        for node in tqdm(candidates, desc="Rectangle attack"):
            node_neighbors = neighbors[node]
            for graph_node in others:
                if A[node, graph_node] == A[node, node]:
                    for neighbor in node_neighbors:
                        if self.Gstar.does_not_know_edge((neighbor, graph_node)):
                            self.Gstar.add_edge((neighbor, graph_node))
                            modifs += 2
                else:
                    graph_node_neighbors = neighbors[graph_node]
                    number_missing_edges = self.A[graph_node, graph_node] - len(graph_node_neighbors)
                    number_missing_common_neighbors = self.A[node, graph_node] - len(node_neighbors & graph_node_neighbors)

                    if number_missing_edges <= number_missing_common_neighbors:
                        adds = list(self.Gstar.nodes - node_neighbors)
                        for i in adds:
                            if self.Gstar.does_not_know_edge((i, graph_node)):
                                self.Gstar.remove_edge((i, graph_node))
                                modifs += 2


        print(f"Rectangle attack: {modifs} modifications")

        if self.log:
            self.log_file.write(f"rectangle,{self.expe_number},{self.graph1_prop},{modifs}\n")

        # self.Gstar = reconstructed_graph
        self.modifications += modifs




    def run(self, run_matching=True, run_completion=True, run_degree=True, run_rectangle=True, run_triangle=True, run_degree2=True):

        iteration_deterministic = 0
        iteration_probabilistic = 0
        continue_deterministic = True

        degrees = np.diag(self.A)

        if run_degree:
            self.degree_attack()

        while continue_deterministic:
            old_modfications = self.modifications
            
            start = time.time()
            if run_matching:
                self.matching_attacks()
                self.degree_matching_attack()

            if run_completion:
                self.completion_attacks()
                self.degree_completion_attack()

            if run_rectangle:
                self.rectangle_attack(degrees)

            if run_triangle:
                self.triangle_attack()

            stats = self.Gstar.stats()
            print("0s : ", stats[0], "1s : ", stats[1], "?s : ", stats[2])

            end = time.time()

            if self.modifications == old_modfications:
                continue_deterministic = False

            iteration_deterministic += 1

            

    def complete_graph(self):
        """
        Complete the graph by adding edges if homomorphism components are identified
        """
        # Identifying the components of the graph that are not reconstructed

        print("Instanciating isomorphism")

        def equivalent(n1, n2, array):
            for i in range(array.shape[1]):
                if array[n1, i] != array[n2, i] and i != n1 and i != n2:
                    return False

            if array[n1, n1] != array[n2, n2]:
                return False

            return True


        def is_in_equivalence_classes(equivalence_class, equivalence_classes):
            for ec in equivalence_classes:
                if set(equivalence_class) == set(ec):
                    return True

            return False


        Gstar_fixed = self.Gstar.copy()
        Gsquare = self.A

        choices = {}
        unknown_nodes = []
        unknowned_edges = Gstar_fixed.unknown_edges()
        for edge in unknowned_edges:
            if edge[0] not in choices:
                choices[edge[0]] = set()
            choices[edge[0]].add(edge[1])


        equivalence_classes_per_node = {}

        for node, candidates in choices.items():
            equivalence_classes = []
            for candidate in candidates:
                equivalent_nodes = []
                equivalent_nodes.append(candidate)
                for other_candidate in candidates - {candidate}:
                    if equivalent(candidate, other_candidate, Gsquare):
                        equivalent_nodes.append(other_candidate)

                if not is_in_equivalence_classes(equivalent_nodes, equivalence_classes):
                    equivalence_classes.append(equivalent_nodes)
                
                np.random.shuffle(equivalence_classes)

            equivalence_classes_per_node[node] = equivalence_classes

        selected_nodes =equivalence_classes_per_node.keys()

        for node, equivalence_classes in equivalence_classes_per_node.items():
            if Gstar_fixed.degree(node) == Gsquare[node, node]:
                continue
            for candidates in equivalence_classes:
                for node_candidate in candidates:
                    if Gstar_fixed.degree(node_candidate) == Gsquare[node_candidate, node_candidate] or Gstar_fixed.degree(node) == Gsquare[node, node]:
                        continue
                    Gstar_fixed.add_edge((node, node_candidate))
                    print(f"Adding edge between {node} and {node_candidate}")
                    break


        self.Gstar = Gstar_fixed.copy()



    def get_Gstar(self):
        return self.Gstar
