import math
import numpy as np
import scipy.sparse as sp
import torch
import random

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(sorted(classes))}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(graph: str, path: str, test_frac: float = 0.20, val_frac: float = 0.10):
    print("Loading {} dataset...".format(graph))

    idx_features_labels = np.genfromtxt(f"{path}/{graph}/{graph}.content", dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    labels = torch.LongTensor(np.where(labels)[1])
    

    # build graph
    idx = np.array(idx_features_labels[:, 0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}/{graph}/{graph}.cites", dtype=str)
    edges = []
    for (src, tgt) in edges_unordered:
        if src not in idx_map or tgt not in idx_map:
            continue
        else:
            edges.append([idx_map[src], idx_map[tgt]])

    edges = np.array(edges, dtype=np.int32)
    print(f"Found {len(edges)} edges")
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
    #     edges_unordered.shape
    # )
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    # build symmetric adjacency matrix

    
    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    num_nodes = features.shape[0]
    train_frac = 1 - val_frac - test_frac
    idx_train = torch.LongTensor(range(int(num_nodes * train_frac)))
    idx_val = torch.LongTensor(
        range(int(num_nodes * train_frac), int(num_nodes * train_frac) + int(num_nodes * val_frac))
    )
    idx_test = torch.LongTensor(
        range(int(num_nodes * train_frac) + int(num_nodes * val_frac), num_nodes)
    )
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def batchify_edges(
    edges: torch.Tensor, edge_labels: torch.Tensor, batch_size: int, shuffle: bool = True
):
    num_edges = edges.shape[1]
    batch_num = math.ceil(num_edges / batch_size)
    index_array = list(range(num_edges))

    if shuffle:
        np.random.shuffle(index_array)
    for i in range(batch_num):
        indices = index_array[i * batch_size : (i + 1) * batch_size]
        batch_edges = edges[:, indices]
        batch_labels = edge_labels[indices]
        batch_nodes = batch_edges.flatten().unique()
        yield batch_nodes, batch_edges, batch_labels

def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)