from Graph import Graph
import numpy as np
from tqdm import tqdm


class RevisitedSpectral:

    def __init__(self, reconstructed_graph, A):
        self.last_G = reconstructed_graph
        self.G = reconstructed_graph
        self.G_adj_matrix = self.G.adjacency_matrix()
        self.A = A



    def run(self, alpha=0.0, beta=0.0, gamma=0.0, threshold=0.5):

        Gstar_edges = np.argwhere(self.G_adj_matrix == 1)
        Gstar_non_edges = np.argwhere(self.G_adj_matrix == 0)
        Gstar_unknowns = np.argwhere(self.G_adj_matrix == 2)

        if alpha + beta + gamma == 0:
            beta = (len(Gstar_non_edges) + len(Gstar_edges)) / (self.A.shape[0] ** 2)
            alpha = 1 - beta

        M_star = np.zeros((self.A.shape[0], self.A.shape[1]))
        U_A, S_A, V_A = np.linalg.svd(self.A, compute_uv=True)

        n = self.A.shape[0]

        distances = []
        distances_not_chosen = []

        for i in tqdm(range(S_A.shape[0]), desc=f"Revisited spectral attack : alpha={alpha}, beta={beta}, gamma={gamma}"):

            S_Gi_plus = np.sqrt(S_A[i])
            S_Gi_minus =  - np.sqrt(S_A[i])

            M_star_plus = M_star + U_A[:, i].reshape(-1, 1) * S_Gi_plus * V_A[i, :].reshape(1, -1)
            M_star_minus = M_star + U_A[:, i].reshape(-1, 1) * S_Gi_minus * V_A[i, :].reshape(1, -1)


            binary_M_star_plus = np.where(M_star_plus >= threshold, 1, 0)
            binary_M_star_minus = np.where(M_star_minus >= threshold, 1, 0)

            distance_plus_M_G = np.linalg.norm(M_star_plus[Gstar_edges[:, 0], Gstar_edges[:, 1]] - self.G_adj_matrix[Gstar_edges[:, 0], Gstar_edges[:, 1]])
            distance_minus_M_G = np.linalg.norm(M_star_minus[Gstar_edges[:, 0], Gstar_edges[:, 1]] - self.G_adj_matrix[Gstar_edges[:, 0], Gstar_edges[:, 1]])

            distance_plus_M_G += np.linalg.norm(M_star_plus[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]])
            distance_minus_M_G += np.linalg.norm(M_star_minus[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]])

            distance_plus_M_binary = np.linalg.norm(M_star_plus[Gstar_unknowns[:, 0], Gstar_unknowns[:, 1]] - binary_M_star_plus[Gstar_unknowns[:, 0], Gstar_unknowns[:, 1]])
            distance_minus_M_binary = np.linalg.norm(M_star_minus[Gstar_unknowns[:, 0], Gstar_unknowns[:, 1]] - binary_M_star_minus[Gstar_unknowns[:, 0], Gstar_unknowns[:, 1]])

            distance_squares_plus = np.linalg.norm(np.dot(binary_M_star_plus, binary_M_star_plus.T) - self.A, ord='fro') if gamma != 0 else 0
            distance_squares_minus = np.linalg.norm(np.dot(binary_M_star_minus, binary_M_star_minus.T) - self.A, ord='fro') if gamma != 0 else 0

            distance_plus = beta  * distance_plus_M_G + alpha * distance_plus_M_binary + gamma /n * distance_squares_plus
            distance_minus = beta * distance_minus_M_G + alpha * distance_minus_M_binary + gamma /n * distance_squares_minus


            if distance_plus <= distance_minus:
                M_star = M_star_plus
                distances.append(distance_plus)
                distances_not_chosen.append(distance_minus)
            else :
                M_star = M_star_minus
                distances.append(distance_minus)
                distances_not_chosen.append(distance_plus)

        M_star = np.where(M_star >= threshold, 1, 0)

        for edge in Gstar_edges:
            M_star[edge[0], edge[1]] = 1
            M_star[edge[1], edge[0]] = 1

        for non_edge in Gstar_non_edges:
            M_star[non_edge[0], non_edge[1]] = 0
            M_star[non_edge[1], non_edge[0]] = 0

        self.Gstar = Graph.from_adj_matrix(M_star)

    def run_bis(self, alpha=0.0, beta=0.0, gamma=0.0, threshold=0.5):

        Gstar_edges = np.argwhere(self.G.adj_matrix == 1)
        Gstar_non_edges = np.argwhere(self.G.adj_matrix == 0)
        Gstar_unknowns = np.argwhere(self.G.adj_matrix == 2)

        n = self.A.shape[0]

        if alpha + beta + gamma == 0:
            beta = (len(Gstar_non_edges) + len(Gstar_edges)) / (self.A.shape[0] ** 2)
            alpha = 1 - beta

        M_star = np.zeros_like(self.G.adj_matrix)
        U_A, S_A, V_A = np.linalg.svd(self.A, compute_uv=True)

        distances = []

        for i in tqdm(range(S_A.shape[0]), desc=f"Revisited spectral attack : alpha={alpha}, beta={beta}, gamma={gamma}"):

            S_Gi_plus = np.sqrt(S_A[i])
            S_Gi_minus =  -np.sqrt(S_A[i])

            M_star_plus = M_star + U_A[:, i].reshape(-1, 1) * S_Gi_plus * V_A[i, :].reshape(1, -1)
            M_star_minus = M_star + U_A[:, i].reshape(-1, 1) * S_Gi_minus * V_A[i, :].reshape(1, -1)

            binary_M_star_plus = np.where(M_star_plus >= threshold, 1, 0)
            binary_M_star_minus = np.where(M_star_minus >= threshold, 1, 0)

            distance_plus_M_G = np.linalg.norm(M_star_plus[Gstar_edges[:, 0], Gstar_edges[:, 1]] - self.G.adj_matrix[Gstar_edges[:, 0], Gstar_edges[:, 1]])
            distance_minus_M_G = np.linalg.norm(M_star_minus[Gstar_edges[:, 0], Gstar_edges[:, 1]] - self.G.adj_matrix[Gstar_edges[:, 0], Gstar_edges[:, 1]])

            distance_plus_M_G += np.linalg.norm(binary_M_star_plus[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]] - self.G.adj_matrix[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]])
            distance_minus_M_G += np.linalg.norm(binary_M_star_minus[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]] - self.G.adj_matrix[Gstar_non_edges[:, 0], Gstar_non_edges[:, 1]])

            distance_plus_M_binary = np.linalg.norm(M_star_plus[Gstar_unknowns[:, 0], Gstar_unknowns[:, 1]] - binary_M_star_plus[Gstar_unknowns[:, 0], Gstar_unknowns[:, 1]])
            distance_minus_M_binary = np.linalg.norm(M_star_minus[Gstar_unknowns[:, 0], Gstar_unknowns[:, 1]] - binary_M_star_minus[Gstar_unknowns[:, 0], Gstar_unknowns[:, 1]])

            distance_squares_plus = np.linalg.norm(np.dot(binary_M_star_plus, binary_M_star_plus.T) - self.A, ord='fro') if gamma != 0 else 0
            distance_squares_minus = np.linalg.norm(np.dot(binary_M_star_minus, binary_M_star_minus.T) - self.A, ord='fro') if gamma != 0 else 0

            distance_plus = beta  * distance_plus_M_G + alpha * distance_plus_M_binary + gamma /n * distance_squares_plus
            distance_minus = beta * distance_minus_M_G + alpha * distance_minus_M_binary + gamma /n * distance_squares_minus


            if distance_plus <= distance_minus:
                M_star = M_star_plus
                distances.append(distance_plus)
            else:
                M_star = M_star_minus
                distances.append(distance_minus)


        M_star = np.where(M_star >= threshold, 1, 0)

        for edge in Gstar_edges:
            M_star[edge[0], edge[1]] = 1
            M_star[edge[1], edge[0]] = 1

        for non_edge in Gstar_non_edges:
            M_star[non_edge[0], non_edge[1]] = 0
            M_star[non_edge[1], non_edge[0]] = 0

        self.Gstar = Graph.from_adj_matrix(M_star)


    def sanity_check(self):
        modifs = 0
        slots_to_forget = set()
        A_prime = np.dot(self.Gstar.adj_matrix, self.Gstar.adj_matrix.T)
        row_correctness = self.A == A_prime

        for i in range(self.A.shape[0]):
            for j in range(i, self.A.shape[1]):

                if self.A[i, j] != A_prime[i, j]:
                    if A_prime[i, i] == self.A[i, i]:
                        for k in range(self.A.shape[0]):
                            if self.last_G.adj_matrix[j, k] == 2:
                                self.Gstar.adj_matrix[j, k] = 2
                                self.Gstar.adj_matrix[k, j] = 2
                                modifs += 1
                                slots_to_forget.add(j)
                    elif A_prime[j, j] == self.A[j, j]:
                        for k in range(self.A.shape[0]):
                            if self.last_G.adj_matrix[i, k] == 2:
                                self.Gstar.adj_matrix[i, k] = 2
                                self.Gstar.adj_matrix[k, i] = 2
                                modifs += 1
                                slots_to_forget.add(i)
                    else:
                        for k in range(self.A.shape[0]):
                            if self.last_G.adj_matrix[i, k] == 2:
                                self.Gstar.adj_matrix[i, k] = 2
                                self.Gstar.adj_matrix[k, i] = 2
                                modifs += 1
                                slots_to_forget.add(i)
                            if self.last_G.adj_matrix[j, k] == 2:
                                self.Gstar.adj_matrix[j, k] = 2
                                self.Gstar.adj_matrix[k, j] = 2
                                modifs += 1
                                slots_to_forget.add(j)


    def optmized_sanity_check(self):
        modifs = 0
        A_prime = np.dot(self.Gstar.adj_matrix, self.Gstar.adj_matrix.T)

        for i in range(self.A.shape[0]):
            for j in range(i+1, self.A.shape[1]):


                if self.A[i, j] < A_prime[i, j]:
                    if A_prime[i, i] == self.A[i, i]:
                        for k in range(self.A.shape[0]):
                            if self.last_G.adj_matrix[j, k] == 2 and self.Gstar.adj_matrix[j, k] == 1:
                                self.Gstar.adj_matrix[j, k] = 2
                                self.Gstar.adj_matrix[k, j] = 2
                                modifs += 1
                    elif A_prime[j, j] == self.A[j, j]:
                        for k in range(self.A.shape[0]):
                            if self.last_G.adj_matrix[i, k] == 2 and self.Gstar.adj_matrix[i, k] == 1:
                                self.Gstar.adj_matrix[i, k] = 2
                                self.Gstar.adj_matrix[k, i] = 2
                                modifs += 1
                    elif self.A[i, j] > A_prime[i, j]:
                        for k in range(self.A.shape[0]):
                            if self.last_G.adj_matrix[i, k] == 2 and self.Gstar.adj_matrix[i, k] == 0:
                                self.Gstar.adj_matrix[i, k] = 2
                                self.Gstar.adj_matrix[k, i] = 2
                                modifs += 1
                            if self.last_G.adj_matrix[j, k] == 2 and self.Gstar.adj_matrix[j, k] == 0:
                                self.Gstar.adj_matrix[j, k] = 2
                                self.Gstar.adj_matrix[k, j] = 2
                                modifs += 1
                    else:
                        for k in range(self.A.shape[0]):
                            if self.last_G.adj_matrix[i, k] == 2 and self.Gstar.adj_matrix[i, k] == 1:
                                self.Gstar.adj_matrix[i, k] = 2
                                self.Gstar.adj_matrix[k, i] = 2
                                modifs += 1
                            if self.last_G.adj_matrix[j, k] == 2 and self.Gstar.adj_matrix[j, k] == 1:
                                self.Gstar.adj_matrix[j, k] = 2
                                self.Gstar.adj_matrix[k, j] = 2
                                modifs += 1




    def sanity_check_with_high_loss(self):
        modifs = 0
        adj_matrix = self.Gstar.adjacency_matrix()
        #A_prime = np.dot(adj_matrix, adj_matrix.T)
        A_prime = adj_matrix @ adj_matrix.T

        differences = np.argwhere(self.A != A_prime)

        # ensure symmetry
        for i in range(self.A.shape[0]):
            for j in range(i, self.A.shape[1]):
                if self.Gstar.get_edge_label((i, j)) != self.Gstar.get_edge_label((j, i)):
                    self.Gstar.write_value((i, j), 1)
                    self.Gstar.write_value((j, i), 1)
                    modifs += 1

        # detect and clean false inferences
        for i, j in differences:
            for k in range(self.A.shape[0]):
                if self.last_G.does_not_know_edge((i, k)):
                    self.Gstar.write_value((i, k), 2)
                    modifs += 1
                if self.last_G.does_not_know_edge((j, k)):
                    self.Gstar.write_value((j, k), 2)
                    modifs += 1

        print(f"Sanity check with high loss modifs: {modifs}")

    def sanity_check_with_early_stop(self, threshold=0.05):

        A_prime = np.dot(self.Gstar.adj_matrix, self.Gstar.adj_matrix.T)
        incorrectness = np.sum(self.A != A_prime) / (self.A.shape[0] ** 2)
        modifs = 0
        slots_to_forget = set()

        print(f"Incorrectness: {incorrectness}")
        if incorrectness > threshold:
            return True

        for i in range(self.A.shape[0]):
            for j in range(i, self.A.shape[1]):
                if self.Gstar.adj_matrix[i, j] != self.Gstar.adj_matrix[j, i] and (self.Gstar.adj_matrix[i, j] == 1 or self.Gstar.adj_matrix[j, i] == 1):
                    self.Gstar.adj_matrix[i, j] = 1
                    self.Gstar.adj_matrix[j, i] = 1
                if self.A[i, j] != A_prime[i, j]:
                    for k in range(self.A.shape[0]):
                        if self.last_G.adj_matrix[i, k] == 2:
                            self.Gstar.adj_matrix[i, k] = 2
                            self.Gstar.adj_matrix[k, i] = 2
                            modifs += 1
                            slots_to_forget.add(i)
                        if self.last_G.adj_matrix[j, k] == 2:
                            self.Gstar.adj_matrix[j, k] = 2
                            self.Gstar.adj_matrix[k, j] = 2
                            modifs += 1
                            slots_to_forget.add(j)


        return False



    def sanity_check_ultimate(self):
        modifs = 0
        addded_edges = set()
        A_prime = np.dot(self.Gstar.adj_matrix, self.Gstar.adj_matrix.T)

        for node in range(self.A.shape[0]):
            sum_row = np.sum(self.A[node])
            sum_degrees_neighbors = np.sum([A_prime[k, k] for k in self.Gstar.neighbors(node)])

            if sum_row != sum_degrees_neighbors:
                for n in self.Gstar.nodes:
                    if self.last_G.does_not_know_edge((node, n)):
                        self.Gstar.write_value((node, n), 2)

                        modifs += 1



    # def sanity_check_ultimate2(self):
    #     modifs = 0

    #     A_prime = np.dot(self.Gstar.adj_matrix, self.Gstar.adj_matrix.T)

    #     for i in range(self.A.shape[0]):
    #         A_i = self.A[i] > 0
    #         A_prime_i = A_prime[i] > 0
    #         for j in range(i+1, self.A.shape[1]):
    #             A_j = self.A[j] > 0
    #             A_prime_j = A_prime[j] > 0
    #             for k in range(self.A.shape[0]):
    #                 if (A_i[k] != A_prime_i[k] or A_j[k] != A_prime_j[k]):
    #                     self.Gstar.adj_matrix[i, j] = 2
    #                     self.Gstar.adj_matrix[j, i] = 2

    #     modifs = len(np.argwhere(self.Gstar.adj_matrix == 2))
    #     print(f"modifs: {modifs}")


    def sanity_check_ultimate2(self):
        modifs = 0
        adj_matrix = self.Gstar.adjacency_matrix()
        A_prime = np.dot(adj_matrix, adj_matrix.T)

        for i in range(self.A.shape[0]):
            sum_row = np.sum(self.A[i])
            sum_row_prime = np.sum(A_prime[i])
            if sum_row != sum_row_prime:
                for j in range(self.A.shape[0]):
                    if self.last_G.does_not_know_edge((i, j)):
                        self.Gstar.write_value((i, j), 2)
                        modifs += 1

        print(f"Sanity check ultimate 2 modifs: {modifs}")



    def get_Gstar(self):
        return self.Gstar


def all_true(v):
    for i in v:
        if not i:
            return False
    return True
