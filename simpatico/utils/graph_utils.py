from typing import List, Optional
from torch_geometric.nn import radius
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import torch


def get_proximal_atom_masks(
    protein_pos: torch.Tensor, ligand_pos: torch.Tensor, r: float = 4
) -> List[torch.Tensor]:
    """
    Generate per-atom masks for protein and ligand graphs indicating which are involved in interactions.
    Args:
        protein_pos (torch.Tensor): position values of protein graph.
        ligand_pos (torch.Tensor): position values of ligand graph.
        r (float, optional): threshold distance for interactivity (default = 4).
    Returns:
        (torch.Tensor, torch.Tensor): boolean masks for protein and ligand graphs.
    """
    protein_mask = torch.zeros(protein_pos.size(0)).bool()
    ligand_mask = torch.zeros(ligand_pos.size(0)).bool()

    interaction_index = radius(protein_pos, ligand_pos, r)

    protein_mask[interaction_index[1].unique()] = True
    ligand_mask[interaction_index[0].unique()] = True

    return protein_mask, ligand_mask


def drop_nodes(
    graph: Data,
    p: float,
    min_nodes: int = 1,
    require: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Data:
    """
    Randomly remove a fraction `p` of a graph's nodes, returning a new graph.

    For PROTEINS. Removal is safe here because a protein graph stores no edges -- the
    encoder rebuilds them from coordinates on every forward pass -- so a deleted atom
    simply ceases to exist and its neighbours re-link around it.

    Do NOT use this on ligands: deleting atoms severs bonds and, because `edge_attr` is a
    one-hot hop distance rather than a bond type, shifts hop distances across the whole
    molecule. Use `zero_node_features` there.

    Every per-node attribute is carried through the same index, so `pos`, `proximal`,
    `residue` and `chain` stay aligned with `x`. `edge_index`/`edge_attr` are rewritten via
    `subgraph` if present, so the function is still correct on a graph that has them.

    Args:
        graph (Data): PyG graph.
        p (float): probability of dropping each node. `p <= 0` returns `graph` unchanged.
        min_nodes (int, optional): floor on surviving node count. If a draw falls below it,
            randomly chosen dropped nodes are restored until the floor is met.
        require (torch.Tensor, optional): boolean per-node mask of nodes that must not be
            wiped out entirely. If every flagged node is dropped, one is restored at random.
            Pass the `proximal` mask: `ProteinEncoder.forward` voxelises the pocket from
            protein atoms near the ligand, and a pair whose contact atoms are all removed
            has no pocket left to encode. It binds only on pathologically small pockets --
            `get_batch` already skips pairs with no proximal atom at all.
        generator (torch.Generator, optional): RNG, for reproducible masks.
    Returns:
        (Data): a new graph with the surviving nodes, or `graph` itself if nothing was dropped.
    """
    if p <= 0:
        return graph

    n = graph.x.size(0)
    if n <= min_nodes:
        return graph

    keep = torch.rand(n, generator=generator) >= p

    n_keep = int(keep.sum())
    if n_keep < min_nodes:
        dropped = torch.where(~keep)[0]
        restore = dropped[torch.randperm(dropped.size(0), generator=generator)][
            : min_nodes - n_keep
        ]
        keep[restore] = True

    if require is not None and bool(require.any()) and not bool((keep & require).any()):
        flagged = torch.where(require)[0]
        keep[flagged[torch.randint(flagged.size(0), (1,), generator=generator)]] = True

    if bool(keep.all()):
        return graph

    keep_index = torch.where(keep)[0]
    masked = Data()

    for key, value in graph:
        if key in ("edge_index", "edge_attr"):
            continue
        if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == n:
            masked[key] = value[keep_index]
        else:
            masked[key] = value

    if "edge_index" in graph:
        edge_attr = graph.edge_attr if "edge_attr" in graph else None
        new_edge_index, new_edge_attr = subgraph(
            keep_index, graph.edge_index, edge_attr, relabel_nodes=True, num_nodes=n
        )
        masked.edge_index = new_edge_index
        if new_edge_attr is not None:
            masked.edge_attr = new_edge_attr

    return masked


def zero_node_features(
    graph: Data,
    p: float,
    min_visible: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Data:
    """
    Randomly zero the feature vectors of a fraction `p` of a graph's nodes, in place of
    removing them.

    For LIGANDS. The node, its coordinates and every bond it participates in survive; only
    its identity is hidden. The molecule therefore stays the molecule -- same topology,
    same geometry, same atom count -- and a masked atom must be characterised from its
    neighbourhood. An all-zero feature row acts as a single shared mask token:
    `MolEncoder.input_projection_layer` maps it to that layer's bias, the same vector for
    every masked atom. `edge_attr` is a one-hot hop distance (`get_k_hop_edges`, k=3), so it
    carries topology alone and leaks nothing about the atom it attaches to; it is untouched.

    Masked atoms stay in the graph, so they remain live in the 4 A contact set that
    `contrastive_loss` builds and are still asked to match their protein partner. That is
    the intended signal -- recover an atom's interaction from context -- not an oversight.

    Args:
        graph (Data): PyG graph.
        p (float): probability of zeroing each node's features. `p <= 0` returns `graph`
            unchanged.
        min_visible (int, optional): floor on nodes left unmasked, so a molecule is never
            reduced to pure topology.
        generator (torch.Generator, optional): RNG, for reproducible masks.
    Returns:
        (Data): a new graph with `x` zeroed at the masked rows and a per-node boolean
            `node_masked` attribute recording which they were.
    """
    if p <= 0:
        return graph

    n = graph.x.size(0)

    # `node_masked` is attached on every path where p > 0, even when nothing ends up
    # masked. PyG's Batch.from_data_list collates by key and raises KeyError if one graph
    # in the batch lacks an attribute the others have -- and a small ligand escapes masking
    # often enough to hit that (a 5-atom ligand survives p=0.5 intact 3% of the time).
    out = graph.clone()

    if n <= min_visible:
        out.node_masked = torch.zeros(n, dtype=torch.bool)
        return out

    masked = torch.rand(n, generator=generator) < p

    n_visible = int((~masked).sum())
    if n_visible < min_visible:
        hidden = torch.where(masked)[0]
        reveal = hidden[torch.randperm(hidden.size(0), generator=generator)][
            : min_visible - n_visible
        ]
        masked[reveal] = False

    out.node_masked = masked

    if bool(masked.any()):
        out.x = out.x.clone()
        out.x[masked] = 0

    return out
