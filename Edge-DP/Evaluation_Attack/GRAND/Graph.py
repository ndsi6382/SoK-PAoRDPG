import numpy as np
import math
import pandas as pd

class Graph(object):

    def __init__(self, size, with_fixed_edges=False):
        self.size = size
        self.nodes = set(range(size))

        self.adj_list = [set() for _ in range(size)]
        self.non_adj_list = [set() for _ in range(size)]
        self.unknown_list = [set() for _ in range(size)]

        if with_fixed_edges:
            for i in range(size):
                for j in range(i, size):
                    self.write_value((i, j), 0)
          
        else:
            for i in range(size):
                for j in range(i+1, size):
                    self.write_value((i, j), 2)
                self.write_value((i, i), 0)
            

    def write_value(self, edge, value):
        n1, n2 = edge
    
        if value == 1:
            self.adj_list[n1].add(n2)
            self.adj_list[n2].add(n1)
            self.non_adj_list[n1].discard(n2)
            self.non_adj_list[n2].discard(n1)
            self.unknown_list[n1].discard(n2)
            self.unknown_list[n2].discard(n1)

        elif value == 0:
            self.adj_list[n1].discard(n2)
            self.adj_list[n2].discard(n1)
            self.non_adj_list[n1].add(n2)
            self.non_adj_list[n2].add(n1)
            self.unknown_list[n1].discard(n2)
            self.unknown_list[n2].discard(n1)
            
        elif value == 2:
            self.adj_list[n1].discard(n2)
            self.adj_list[n2].discard(n1)
            self.non_adj_list[n1].discard(n2)
            self.non_adj_list[n2].discard(n1)
            self.unknown_list[n1].add(n2)
            self.unknown_list[n2].add(n1)



    def add_edge(self, edge):
        n1, n2 = edge
        self.write_value(edge, 1)

    def get_edge_label(self, edge):
        n1, n2 = edge
        return 0 if n2 in self.non_adj_list[n1] else 2 if n2 in self.unknown_list[n1] else 1

    def __len__(self):
        return len(self.graph.edges())

    def add_edges_from(self, edges):
        for edge in edges:
            n1, n2 = edge
            self.add_edge((n1, n2))


    def remove_edge(self, edge):
        n1, n2 = edge
        self.write_value(edge, 0)

    def neighbors(self, n1):
        return self.adj_list[n1]

    def has_edge(self, edge):
        n1, n2 = edge
        return n2 in self.adj_list[n1]

    def does_not_know_edge(self, edge):
        n1, n2 = edge
        return n2 in self.unknown_list[n1]

    def does_not_have_edge(self, edge):
        n1, n2 = edge
        return n2 in self.non_adj_list[n1]

    def edges(self):
        return [[i, j] for i in range(self.size) for j in range(self.size) if i in self.adj_list[j]]

    def edges_no_repeat(self):
        return [[i, j] for i in range(self.size) for j in range(i, self.size) if i in self.adj_list[j]]

    def non_edges(self):
        return [[i, j] for i in range(self.size) for j in range(self.size) if i in self.non_adj_list[j]]

    def unknown_edges(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if i in self.unknown_list[j]]

    def unknown_edges_no_repeat(self):
        return [(i, j) for i in range(self.size) for j in range(i, self.size) if i in self.unknown_list[j]]

    def degree(self, node):
        return len(self.neighbors(node))

    def common_neighbors(self, edge):
        n1, n2 = edge
        return self.neighbors(n1).intersection(self.neighbors(n2))

    def adjacency_matrix(self):
        adj_matrix = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                if i in self.adj_list[j]:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
                elif i in self.non_adj_list[j]:
                    adj_matrix[i][j] = 0
                    adj_matrix[j][i] = 0
                else:
                    adj_matrix[i][j] = 2
                    adj_matrix[j][i] = 2

        return adj_matrix


    def stats(self):
        num_absent = 0
        num_present = 0
        num_unknown = 0

        for i in range(self.size):
            for j in range(i, self.size):

                increment = 1 if i == j else 2 # self loops count for 1 in the adjacency matrix, others for 2

                if i in self.adj_list[j]:
                    num_present += increment
                elif i in self.non_adj_list[j]:
                    num_absent += increment
                else:
                    num_unknown += increment
                

        return num_absent, num_present, num_unknown

    def copy(self):
        copy =  Graph(self.size)

        copy.adj_list = [self.adj_list[i].copy() for i in range(self.size)]
        copy.non_adj_list = [self.non_adj_list[i].copy() for i in range(self.size)]
        copy.unknown_list = [self.unknown_list[i].copy() for i in range(self.size)]

        return copy


    @staticmethod
    def from_nodes_and_edges(nodes, edges):
        graph = Graph(nodes, with_fixed_edges=True)
        
        for edge in edges:
            graph.add_edge(edge)

        return graph

    ##### MY METHOD
    @staticmethod
    def from_nx(G, with_fixed_edges=False):
        nodes = list(range(G.number_of_nodes()))
        graph = Graph(G.number_of_nodes(), with_fixed_edges=with_fixed_edges)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                graph.write_value((i, j), int(G.has_edge(i,j)))
        return graph
    #####

    @staticmethod
    def from_adj_matrix(adj_matrix, with_fixed_edges=False):
        nodes = list(range(len(adj_matrix)))
        graph = Graph(len(nodes), with_fixed_edges=with_fixed_edges)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                graph.write_value((i, j), adj_matrix[i][j])

        return graph

    @staticmethod
    def from_npy(filepath):
        adj_matrix = np.load(filepath, allow_pickle=True)
        return Graph.from_adj_matrix(adj_matrix, with_fixed_edges=True)


    @staticmethod
    def from_txt(filepath, num_nodes=None):
        edges = pd.read_csv(filepath, sep='\t', header=None)
        if num_nodes is None:
            nodes = list(range(0, max(edges[0].tolist() + edges[1].tolist() ) ))
        else:
            nodes = list(range(0, num_nodes))
        graph = Graph(len(nodes), with_fixed_edges=True)
        for i in range(len(edges)):
            graph.add_edge((edges[0][i]-1, edges[1][i]-1))

        return graph

    @staticmethod
    def block_from_txt(filepath):
        edges = pd.read_csv(filepath, sep='\t', header=None)
        max_partition_1 = max(edges[0].tolist())
        nodes_partition_1 = list(range(0, max_partition_1 ) )
        nodes_partition_2 = list(range(max_partition_1, max(edges[1].tolist()) + max_partition_1 ))
        nodes = nodes_partition_1 + nodes_partition_2
        graph = Graph(nodes, with_fixed_edges=True)
        for i in range(len(edges)):
            n1 = edges[0][i]-1
            n2 = edges[1][i]-1 + max_partition_1
            graph.add_edge((n1, n2))

        graph.first_partition_size = max_partition_1
        return graph

    @staticmethod
    def random_graph(nodes, edge_prob=0.5):
        graph = Graph(nodes)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if np.random.random() < edge_prob:
                    graph.add_edge((i, j))
        return graph

    @staticmethod
    def barabasi_albert_graph(size, m=1, m0=1):
        graph = Graph(size, with_fixed_edges=True)
        for i in range(m0, size):
            probs = [graph.degree(j) for j in range(i)]
            probs = [p / sum(probs) for p in probs] if sum(probs) > 0 else [1 / i for _ in range(i)]
            targets = np.random.choice(range(i), size=m, p=probs, replace=False)
            for target in targets:
                graph.add_edge((i, target))

        return graph
        

    def split_dataset(self, common_prop=0, graph1_prop=0.5):
        # sample edges and non edges to create two graphs
        graph1 = self.copy()
        graph2 = self.copy()

        graph1_threshold = common_prop + graph1_prop

        for i in range(self.size):
            for j in range(i+1, self.size):
                prob = np.random.random()
                value = 0 if i in self.non_adj_list[j] else 1
                if prob < common_prop:
                    graph1.write_value((i, j), value)
                    graph2.write_value((j, i), value)

                elif common_prop < prob < graph1_threshold:
                    graph1.write_value((i, j), value)
                    graph2.write_value((j, i), 2)
                else:
                    graph1.write_value((i, j), 2)
                    graph2.write_value((j, i), value)
                    
        return graph1, graph2



    def gradual_sample(self, start=0, stop=1, increment=0.1):
        base_sample = Graph(self.size, with_fixed_edges=False)

        # Initialize the graph with `start` proportion of edges

        samples = []
        print("Generating samples...")

        # Compute total number of edges in a complete graph
        total_edges = self.size * (self.size - 1) // 2
        current_edges = (len(base_sample.edges() + base_sample.non_edges()) - self.size) // 2  # Edges already in the graph

        # Incrementally add edges to reach the target proportions
        for prop in np.linspace(start, stop, int((stop - start) / increment) + 1, endpoint=True):
            target_edges = int(prop * total_edges)
            edges_to_add = target_edges - current_edges

            sample = base_sample.copy()
            unknown_edges = list(sample.unknown_edges_no_repeat())
            np.random.shuffle(unknown_edges)

            for edge in unknown_edges[:edges_to_add]:
                sample.write_value(edge, self.get_edge_label(edge))

            samples.append((prop, sample.copy()))
            base_sample = sample
            current_edges += edges_to_add

        return samples


    def gradual_sample_edges(self, start=0, stop=1, increment=0.1):
        base_sample = Graph(self.size, with_fixed_edges=False)

        # Initialize the graph with `start` proportion of edges

        samples = []
        print("Generating samples...")

        # Compute total number of edges in a complete graph
        total_edges = len(self.edges()) // 2
        current_edges = len(base_sample.edges()) // 2

        # Incrementally add edges to reach the target proportions
        for prop in np.linspace(start, stop, int((stop - start) / increment) + 1, endpoint=True):
            target_edges = int(prop * total_edges)
            edges_to_add = target_edges - current_edges

            sample = base_sample.copy()
            unknown_edges = list(sample.unknown_edges_no_repeat())
            np.random.shuffle(unknown_edges)

            count = 0

            while count < edges_to_add:
                edge = unknown_edges.pop(0)
                if self.get_edge_label(edge) == 1:
                    sample.write_value(edge, 1)
                    count += 1

    
            samples.append((prop, sample.copy()))
            base_sample = sample
            current_edges += edges_to_add

        return samples
            

    def fix_edges(self, value=0):
        for i in range(self.size):
            for j in range(i+1, self.size):
                if i in self.unknown_list[j]:
                    self.write_value((i, j), 0)


    def common_neighbors_matrix(self):
        common_neighbors_matrix = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(i, self.size):
                common_neighbors_matrix[i][j] = len(self.common_neighbors((i, j)))
                common_neighbors_matrix[j][i] = common_neighbors_matrix[i][j]
        return common_neighbors_matrix


    def to_networkx(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        for i in range(self.size):
            for j in range(i+1, self.size):
                if self.adj_matrix[i][j] == 1:
                    G.add_edge(i, j)
        return G

    def to_txt(self, filepath):
        with open(filepath, 'w') as f:
            for (n1, n2) in self.edges():
                f.write(f"{n1+1}\t{n2+1 - self.first_partition_size}\t1\n")


    def __str__(self):
        return f"Graph (#nodes = {self.size}, #0s = {self.stats()[0]}, #1s = {self.stats()[1]}, #?s = {self.stats()[2]})"
        