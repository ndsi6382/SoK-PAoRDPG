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
from tqdm import tqdm
import os, gc
from datetime import datetime

torch.set_num_threads(os.cpu_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def run(nx_graph_path, feature_matrix_path, labels_path, val_mask, test_mask, use_node2vec=False): # filetype = ['text', 'sparse', 'pickle']
    model_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
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
    
    labels = np.loadtxt(labels_path).astype(int).flatten()
    y = torch.tensor(labels, dtype=torch.long)
    
    # ==== Convert to PyG Data ====
    data = from_networkx(nx_graph)
    data.x = x
    data.y = y
    data.num_nodes = x.size(0)
    data = data.to(device)

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
        data.y = torch.tensor([clusters[label] for label in data.y], dtype=torch.long, device=device)

    # ==== Train/Val/Test split ====
    data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
    data.train_mask[val_mask | test_mask] = False
    data.val_mask = val_mask
    data.test_mask = test_mask

    # ==== Node2Vec ====
    if use_node2vec:
        model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=80, context_size=10, walks_per_node=10, num_nodes=data.num_nodes, sparse=True).to(device)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
        for _ in tqdm(range(1), desc="Node2Vec, Epoch"):
            for pos_rw, neg_rw in model.loader(batch_size=128, shuffle=True):
                pos_rw = pos_rw.to(device)
                neg_rw = neg_rw.to(device)
                optimizer.zero_grad()
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            embeddings = model.embedding.weight.clone()
    
        # ==== Combine features ===
        if data.x.size(1) < 128:
            pca = PCA(n_components=data.x.size(1))
            embeddings = pca.fit_transform(embeddings.detach().cpu().numpy()) # project down the dimensionality
        scaler = StandardScaler()
        embeddings = torch.tensor(scaler.fit_transform(embeddings), dtype=torch.float, device=device)
        data.x = torch.cat([data.x, embeddings], dim=1)
        del embeddings

    class_counts = torch.bincount(data.y[data.train_mask])
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() 
    
    model = GCN(in_dim=data.x.size(1), hidden_dim=256, out_dim=int(data.y.max())+1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
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
            val_f1 = f1_score(data.y[data.val_mask].detach().cpu().numpy(), out[data.val_mask].argmax(dim=1).detach().cpu().numpy(), average='macro')
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
    model = GCN(in_dim=data.x.size(1), hidden_dim=256, out_dim=int(data.y.max())+1).to(device)
    model.load_state_dict(torch.load(f"{model_name}.pt"))
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_preds = out[data.test_mask].argmax(dim=1).detach().cpu().numpy()
        labels = data.y[data.test_mask].detach().cpu().numpy()
        test_acc = accuracy_score(labels, test_preds)
        test_f1 = f1_score(labels, test_preds, average='macro')
        print(f"Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        result = {
            "Accuracy": test_acc,
            "F1 Score": test_f1,
        }
    os.remove(f"{model_name}.pt")
    if device != 'cpu':
        del model, optimizer, data, out
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
    return result


if __name__ == "__main__":
    run('./Data/LastFM/nx_adj.pkl', './Data/LastFM/np_feature_matrix_full.pkl', './Data/LastFM/targets.txt', "LastFM")
