import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Node2Vec
from torch_geometric.utils import from_networkx
import networkx as nx
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
from datetime import datetime

torch.set_num_threads(os.cpu_count())

# ==== Model ====
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.25):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def run(nx_graph_path, feature_matrix_path, labels_path, use_node2vec=False): # filetype = ['text', 'sparse', 'pickle']
    model_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # ==== Load graph ====
    with open(nx_graph_path, 'rb') as f:
        nx_graph = pickle.load(f)
    
    # ==== Load features and labels ====
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
    y = torch.tensor(np.loadtxt(labels_path).astype(int).flatten(), dtype=torch.long)
    
    # ==== Convert to PyG Data ====
    data = from_networkx(nx_graph)
    data.x = x
    data.y = y
    data.num_nodes = x.size(0)

    # ==== Reduce Classes ====
    num_classes = len(torch.unique(y))
    if num_classes > 5:
        mean_vecs = list()
        for i in range(num_classes):
            class_mask = (data.y == i)
            mean_vecs.append(data.x[class_mask].float().mean(dim=0))
        mean_vecs = torch.stack(mean_vecs)
        kmeans = KMeans(n_clusters=5, random_state=0)
        clusters = kmeans.fit_predict(mean_vecs)
        data.y = torch.tensor([clusters[label] for label in data.y], dtype=torch.long)

    # ==== Train/Val/Test split ====
    idx = np.arange(data.num_nodes)
    idx_train, idx_temp, y_train, y_temp = train_test_split(idx, data.y, stratify=data.y, test_size=0.2)
    idx_val, idx_test, _, _ = train_test_split(idx_temp, y_temp, stratify=y_temp, test_size=0.5)
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # ==== Node2Vec ====
    if use_node2vec:
        node2vec = Node2Vec(data.edge_index, embedding_dim=128, walk_length=80, context_size=10, walks_per_node=10, num_nodes=data.num_nodes, sparse=True)
        opt = torch.optim.SparseAdam(node2vec.parameters(), lr=0.01)
        for _ in tqdm(range(1), desc="Node2Vec, Epoch"):
            for pos_rw, neg_rw in node2vec.loader(batch_size=128, shuffle=True):
                opt.zero_grad()
                loss = node2vec.loss(pos_rw, neg_rw)
                loss.backward()
                opt.step()
        with torch.no_grad():
            embeddings = node2vec.embedding.weight.clone()
    
        # ==== Combine features ===
        if data.x.size(1) < 128:
            pca = PCA(n_components=data.x.size(1))
            embeddings = pca.fit_transform(embeddings) # project down the dimensionality
        scaler = StandardScaler()
        embeddings = torch.tensor(scaler.fit_transform(embeddings), dtype=torch.float)
        data.x = torch.cat([data.x, embeddings], dim=1)

    class_counts = torch.bincount(data.y[data.train_mask])
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    
    model = GCN(in_dim=data.x.size(1), hidden_dim=256, out_dim=int(data.y.max())+1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    n_epochs = 500
    # ==== Training loop ====
    early_stopping_counter = 0; best_val_f1 = 0
    for epoch in tqdm(range(n_epochs), desc="GCN, Epoch"):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_f1 = f1_score(data.y[data.val_mask], out[data.val_mask].argmax(dim=1), average='macro')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0
            torch.save(model.state_dict(), f"{model_name}.pt")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= n_epochs * 0.05:
                print("Stopping early.")
                break
    
    # ==== Final test evaluation ====
    model = GCN(in_dim=data.x.size(1), hidden_dim=256, out_dim=int(data.y.max())+1)
    model.load_state_dict(torch.load(f"{model_name}.pt"))
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_preds = out[data.test_mask].argmax(dim=1)
        test_acc = accuracy_score(data.y[data.test_mask].cpu(), test_preds.cpu())
        test_f1 = f1_score(data.y[data.test_mask], test_preds, average='macro')
        print(f"Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        result = {
            "Accuracy": test_acc,
            "F1 Score": test_f1,
        }
    os.remove(f"{model_name}.pt")
    return result, val_mask, test_mask


if __name__ == "__main__":
    run('./Data/LastFM/nx_adj.pkl', './Data/LastFM/np_feature_matrix_full.pkl', './Data/LastFM/targets.txt', "LastFM")
