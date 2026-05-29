import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Node2Vec
from torch_geometric.utils import from_networkx, negative_sampling, remove_self_loops, to_undirected
import networkx as nx
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.sparse as sp
import os
from datetime import datetime

torch.set_num_threads(os.cpu_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.25):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def decode(z, edge_index):
    z_src = z[edge_index[0]]
    z_dst = z[edge_index[1]]
    return torch.sum(z_src * z_dst, dim=1)

def get_labels(pos_edge_index, neg_edge_index):
    pos_labels = torch.ones(pos_edge_index.size(1))
    neg_labels = torch.zeros(neg_edge_index.size(1))
    return torch.cat([pos_labels, neg_labels])

def evaluate(z, pos_edges, neg_edges):
    edge_combined = torch.cat([pos_edges, neg_edges], dim=1)
    labels = get_labels(pos_edges, neg_edges)
    logits = decode(z, edge_combined).sigmoid()
    preds = (logits > 0.5).float()
    acc = (preds == labels).sum().item() / labels.size(0)
    auc = roc_auc_score(labels.numpy(), logits.detach().numpy())
    ap = average_precision_score(labels.numpy(), logits.detach().numpy())
    return acc, auc, ap


def run(nx_graph_path, feature_matrix_path, use_node2vec=False): # filetype = ['text', 'sparse', 'pickle']
    model_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # ==== Load graph ====
    with open(nx_graph_path, 'rb') as f:
        nx_graph = pickle.load(f)
    # ==== Load features ====
    if feature_matrix_path[-4:] == '.npz':
        features = sp.load_npz(feature_matrix_path)
        x = torch.tensor(features.toarray(), dtype=torch.float)
    elif feature_matrix_path[-4:] == '.txt':
        features = np.loadtxt(feature_matrix_path)
        x = torch.tensor(features, dtype=torch.float)
    elif feature_matrix_path[-4:] == '.pkl':
        with open(feature_matrix_path, 'rb') as f:
            features = pickle.load(f)
        x = torch.tensor(features, dtype=torch.float)
    # ==== Convert to PyG format ====
    data = from_networkx(nx_graph)
    data.x = x
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index = to_undirected(data.edge_index) # should already be undirected
    num_nodes = data.num_nodes
    all_edges = data.edge_index.t().tolist()
    data = data.to(device)
    
    # ==== Create Train/Val/Test Split ====
    np.random.shuffle(all_edges) # Shuffle edges
    num_edges = len(all_edges)
    num_val = int(0.1 * num_edges)
    num_test = int(0.1 * num_edges)
    num_train = num_edges - num_val - num_test
    edges_train = torch.tensor(all_edges[:num_train], dtype=torch.long).t().to(device)
    edges_val   = torch.tensor(all_edges[num_train:num_train+num_val], dtype=torch.long).t().to(device)
    edges_test  = torch.tensor(all_edges[num_train+num_val:], dtype=torch.long).t().to(device)
    # ==== Negative samples ====
    neg_val = negative_sampling(edge_index=edges_train, num_nodes=num_nodes, num_neg_samples=edges_val.size(1))
    neg_test = negative_sampling(edge_index=edges_train, num_nodes=num_nodes, num_neg_samples=edges_test.size(1))

    # ==== Node2Vec Feature Augmentation ====
    if use_node2vec:
        model = Node2Vec(edges_train, embedding_dim=128, walk_length=80, context_size=10, walks_per_node=10, num_nodes=data.num_nodes, sparse=True).to(device)
        optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=0.01)
        for _ in tqdm(range(1), desc="Node2Vec, Epoch"):
            for pos_rw, neg_rw in model.loader(batch_size=128, shuffle=True):
                optimizer.zero_grad()
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            embeddings = model.embedding.weight.clone()
        if data.x.size(1) < 128:
            pca = PCA(n_components=data.x.size(1))
            embeddings = pca.fit_transform(embeddings) # project down the dimensionality
        scaler = StandardScaler()
        embeddings = torch.tensor(scaler.fit_transform(embeddings), dtype=torch.float).to(device)
        data.x = torch.cat([data.x, embeddings], dim=1)
        del embeddings
    
    # ==== Training ====
    model = GCNEncoder(in_dim=data.x.size(1), hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 500
    early_stopping_counter = 0; best_val_auc = 0
    for epoch in tqdm(range(n_epochs), desc="GCN, Epoch"):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, edges_train)  # Train on train edges only
        # Generate fresh negative samples each epoch for training
        neg_train = negative_sampling(edge_index=edges_train, num_nodes=num_nodes, num_neg_samples=edges_train.size(1))
        edge_combined = torch.cat([edges_train, neg_train], dim=1)
        labels = get_labels(edges_train, neg_train)
        logits = decode(z, edge_combined)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        #loss = F.binary_cross_entropy_with_logits(decode(z, torch.cat([edges_train, neg_train], dim=1)), get_labels(edges_train, neg_train))
        loss.backward()
        optimizer.step()
    # ==== Validation ====
        model.eval()
        with torch.no_grad():
            z = model(data.x, edges_train)  # embedding on train graph
            _, val_auc, _ = evaluate(z, edges_val, neg_val)
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stopping_counter = 0
            torch.save(model.state_dict(), f"{model_name}.pt")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= n_epochs * 0.05:
                print("Stopping early.")
                break
    
    # ==== Evaluation ====
    model = GCNEncoder(in_dim=data.x.size(1), hidden_dim=256)
    model.load_state_dict(torch.load(f"{model_name}.pt"))
    model.eval()
    with torch.no_grad():
        z = model(data.x, edges_train) # Given the TRAINING edges, which links can be predicted to exist?
        test_acc, test_auc, test_ap = evaluate(z, edges_test, neg_test)
        result = {
            "Accuracy": test_acc,
            "ROC AUC": test_auc,
            "Average Precision": test_ap,
        }
        print(f"Test Accuracy: {test_acc:.4f}, ROC AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
    os.remove(f"{model_name}.pt")
    return result, edges_val, neg_val, edges_test, neg_test


if __name__ == "__main__":
    run('./Data/LastFM/nx_adj.pkl', './Data/LastFM/np_feature_matrix_full.pkl', "LastFM")
