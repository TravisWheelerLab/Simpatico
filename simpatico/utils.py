import torch
from torch_geometric.utils import k_hop_subgraph


def to_onehot(value, vocabulary):
    onehot_list = [0 for _ in vocabulary]
    if value in vocabulary:
        onehot_list[vocabulary.index(value)] = 1
    return onehot_list


def get_k_hop_neighborhoods(atom_index, max_k, ei):
    met_neighbors = torch.tensor([atom_index])
    k_hop_neighborhoods = []

    for k_val in range(1, max_k + 1):
        k_neighbors = k_hop_subgraph(atom_index, k_val, ei)[0]
        k_neighbors = k_neighbors[
            ~torch.any(k_neighbors.unsqueeze(1) == met_neighbors.unsqueeze(0), 1)
        ]
        met_neighbors = torch.unique(torch.hstack((met_neighbors, k_neighbors)))

        k_hop_neighborhoods.append(k_neighbors)

    return k_hop_neighborhoods
