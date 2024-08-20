import torch
from torch_geometric.utils import k_hop_subgraph, to_undirected


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


def get_k_hop_edges(edge_index, k=3):
    k_hop_vocab = [x + 1 for x in range(k)]
    edge_index = to_undirected(edge_index.clone())

    final_edge_index = torch.tensor([[], []])
    edge_attr = []

    for atom_idx in edge_index.flatten().unique():
        for k_i, k_neighborhood in enumerate(
            get_k_hop_neighborhoods(atom_idx.item(), k, edge_index)
        ):
            edge_attr += [
                to_onehot(k_i + 1, k_hop_vocab) for _ in range(k_neighborhood.size(0))
            ]
            final_edge_index = torch.hstack(
                (
                    final_edge_index,
                    torch.vstack(
                        (atom_idx.repeat(k_neighborhood.size(0)), k_neighborhood)
                    ),
                )
            )
    return final_edge_index.int(), torch.tensor(edge_attr)
