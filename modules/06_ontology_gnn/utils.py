"""
Utility functions for Ontology-Aware GNN module.

Provides helper functions for:
- Graph construction from knowledge graph
- Feature preparation
- Biological prior normalization
- Evaluation metrics
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GraphData:
    """
    Container for graph data ready for GNN input.

    Attributes:
        node_features: Dict mapping node type to feature tensor
        edge_index: Edge indices [2, num_edges]
        edge_type: Edge type indices [num_edges]
        edge_type_names: List of edge type names
        node_type_indices: Dict mapping node type to node indices
        bio_priors: Dict of biological prior values
        hierarchy_edges: Hierarchy edges for aggregation
        labels: Optional node labels
    """
    node_features: Dict[str, Any]
    edge_index: Any  # torch.Tensor or np.ndarray
    edge_type: Any
    edge_type_names: List[str]
    node_type_indices: Dict[str, Any]
    bio_priors: Optional[Dict[str, Any]] = None
    hierarchy_edges: Optional[Any] = None
    labels: Optional[Any] = None


def prepare_graph_data(
    knowledge_graph: Any,
    node_embeddings: Optional[Dict[str, Any]] = None,
    bio_priors: Optional[Dict[str, Dict[str, float]]] = None,
    use_torch: bool = True,
) -> GraphData:
    """
    Prepare graph data from knowledge graph for GNN input.

    Args:
        knowledge_graph: Knowledge graph from Module 03
        node_embeddings: Optional pre-computed embeddings {node_id: embedding}
        bio_priors: Dict of biological priors {prior_type: {node_id: value}}
        use_torch: Whether to return PyTorch tensors

    Returns:
        GraphData ready for GNN forward pass
    """
    # Get nodes by type
    node_type_to_ids = defaultdict(list)
    id_to_idx = {}
    current_idx = 0

    # Build node mapping
    for node_id, node_data in knowledge_graph.nodes(data=True):
        node_type = node_data.get("type", "gene")
        node_type_to_ids[node_type].append(node_id)
        id_to_idx[node_id] = current_idx
        current_idx += 1

    num_nodes = current_idx

    # Build node type indices
    node_type_indices = {}
    for node_type, node_ids in node_type_to_ids.items():
        indices = [id_to_idx[nid] for nid in node_ids]
        if use_torch and TORCH_AVAILABLE:
            node_type_indices[node_type] = torch.tensor(indices, dtype=torch.long)
        else:
            node_type_indices[node_type] = np.array(indices, dtype=np.int64)

    # Build node features
    default_dim = 256
    if node_embeddings:
        # Get dimension from first embedding
        first_emb = next(iter(node_embeddings.values()))
        if hasattr(first_emb, 'shape'):
            default_dim = first_emb.shape[-1]

    node_features = {}
    for node_type, node_ids in node_type_to_ids.items():
        features = []
        for nid in node_ids:
            if node_embeddings and nid in node_embeddings:
                emb = node_embeddings[nid]
                if isinstance(emb, np.ndarray):
                    features.append(emb)
                elif TORCH_AVAILABLE and isinstance(emb, torch.Tensor):
                    features.append(emb.numpy())
                else:
                    features.append(np.array(emb))
            else:
                # Random initialization if no embedding
                features.append(np.random.randn(default_dim).astype(np.float32))

        features = np.stack(features)
        if use_torch and TORCH_AVAILABLE:
            node_features[node_type] = torch.tensor(features, dtype=torch.float32)
        else:
            node_features[node_type] = features

    # Build edges
    edge_sources = []
    edge_targets = []
    edge_types = []
    edge_type_to_idx = {}

    for src, dst, edge_data in knowledge_graph.edges(data=True):
        edge_type = edge_data.get("type", "interacts")
        if edge_type not in edge_type_to_idx:
            edge_type_to_idx[edge_type] = len(edge_type_to_idx)

        if src in id_to_idx and dst in id_to_idx:
            edge_sources.append(id_to_idx[src])
            edge_targets.append(id_to_idx[dst])
            edge_types.append(edge_type_to_idx[edge_type])

    edge_type_names = list(edge_type_to_idx.keys())

    if use_torch and TORCH_AVAILABLE:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        edge_type_tensor = np.array(edge_types, dtype=np.int64)

    # Build biological priors
    bio_prior_tensors = None
    if bio_priors:
        bio_prior_tensors = {}
        for prior_type, prior_values in bio_priors.items():
            values = []
            for nid in id_to_idx.keys():
                values.append(prior_values.get(nid, 0.5))
            if use_torch and TORCH_AVAILABLE:
                bio_prior_tensors[prior_type] = torch.tensor(values, dtype=torch.float32)
            else:
                bio_prior_tensors[prior_type] = np.array(values, dtype=np.float32)

    # Build hierarchy edges (GO is_a relationships)
    hierarchy_sources = []
    hierarchy_targets = []

    for src, dst, edge_data in knowledge_graph.edges(data=True):
        edge_type = edge_data.get("type", "")
        if edge_type in ["is_a", "part_of", "go_is_a"]:
            if src in id_to_idx and dst in id_to_idx:
                hierarchy_sources.append(id_to_idx[src])
                hierarchy_targets.append(id_to_idx[dst])

    hierarchy_edges = None
    if hierarchy_sources:
        if use_torch and TORCH_AVAILABLE:
            hierarchy_edges = torch.tensor(
                [hierarchy_sources, hierarchy_targets], dtype=torch.long
            )
        else:
            hierarchy_edges = np.array(
                [hierarchy_sources, hierarchy_targets], dtype=np.int64
            )

    return GraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_type=edge_type_tensor,
        edge_type_names=edge_type_names,
        node_type_indices=node_type_indices,
        bio_priors=bio_prior_tensors,
        hierarchy_edges=hierarchy_edges,
    )


def normalize_priors(
    priors: Dict[str, Any],
    method: str = "minmax",
) -> Dict[str, Any]:
    """
    Normalize biological prior values.

    Args:
        priors: Dict of prior values
        method: Normalization method ("minmax", "zscore", "rank")

    Returns:
        Normalized priors
    """
    normalized = {}

    for prior_type, values in priors.items():
        if TORCH_AVAILABLE and isinstance(values, torch.Tensor):
            values_np = values.numpy()
        else:
            values_np = np.array(values)

        if method == "minmax":
            vmin, vmax = values_np.min(), values_np.max()
            if vmax - vmin > 0:
                values_np = (values_np - vmin) / (vmax - vmin)
            else:
                values_np = np.full_like(values_np, 0.5)

        elif method == "zscore":
            mean, std = values_np.mean(), values_np.std()
            if std > 0:
                values_np = (values_np - mean) / std
            else:
                values_np = np.zeros_like(values_np)

        elif method == "rank":
            ranks = np.argsort(np.argsort(values_np))
            values_np = ranks / (len(ranks) - 1) if len(ranks) > 1 else np.array([0.5])

        if TORCH_AVAILABLE and isinstance(priors[prior_type], torch.Tensor):
            normalized[prior_type] = torch.tensor(values_np, dtype=torch.float32)
        else:
            normalized[prior_type] = values_np

    return normalized


def compute_metrics(
    predictions: Any,
    labels: Any,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels
        threshold: Classification threshold

    Returns:
        Dict of metrics
    """
    if TORCH_AVAILABLE and isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = np.array(predictions)
    labels = np.array(labels)

    # Handle logits vs probabilities
    if predictions.ndim == 2:
        # Multi-class: take argmax
        pred_labels = predictions.argmax(axis=-1)
        probs = predictions[:, 1] if predictions.shape[1] == 2 else predictions.max(axis=-1)
    else:
        # Binary: threshold
        probs = predictions
        pred_labels = (predictions > threshold).astype(int)

    # Basic metrics
    accuracy = (pred_labels == labels).mean()

    # Per-class metrics for binary classification
    if len(np.unique(labels)) == 2:
        tp = ((pred_labels == 1) & (labels == 1)).sum()
        fp = ((pred_labels == 1) & (labels == 0)).sum()
        tn = ((pred_labels == 0) & (labels == 0)).sum()
        fn = ((pred_labels == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    return {"accuracy": float(accuracy)}


def compute_link_prediction_metrics(
    scores: Any,
    labels: Any,
) -> Dict[str, float]:
    """
    Compute link prediction metrics.

    Args:
        scores: Predicted scores for edges
        labels: Ground truth (1 for positive, 0 for negative)

    Returns:
        Dict of metrics
    """
    if TORCH_AVAILABLE and isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    scores = np.array(scores)
    labels = np.array(labels)

    # Sort by score descending
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    # Hits@K
    hits_at_10 = sorted_labels[:10].mean() if len(sorted_labels) >= 10 else sorted_labels.mean()
    hits_at_50 = sorted_labels[:50].mean() if len(sorted_labels) >= 50 else sorted_labels.mean()

    # Mean Reciprocal Rank
    positive_idx = np.where(sorted_labels == 1)[0]
    if len(positive_idx) > 0:
        mrr = np.mean(1.0 / (positive_idx + 1))
    else:
        mrr = 0.0

    # AUC (simple approximation)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        auc = ((pos_scores[:, None] > neg_scores[None, :]).sum() / (n_pos * n_neg))
    else:
        auc = 0.5

    return {
        "hits@10": float(hits_at_10),
        "hits@50": float(hits_at_50),
        "mrr": float(mrr),
        "auc": float(auc),
    }


def create_negative_samples(
    edge_index: Any,
    num_nodes: int,
    num_negative: int,
) -> Tuple[Any, Any]:
    """
    Create negative samples for link prediction.

    Args:
        edge_index: Positive edges [2, num_edges]
        num_nodes: Total number of nodes
        num_negative: Number of negative samples per positive

    Returns:
        Tuple of (negative_edge_index, negative_labels)
    """
    if TORCH_AVAILABLE and isinstance(edge_index, torch.Tensor):
        edge_index_np = edge_index.numpy()
        use_torch = True
    else:
        edge_index_np = np.array(edge_index)
        use_torch = False

    num_positive = edge_index_np.shape[1]
    total_negative = num_positive * num_negative

    # Create set of existing edges for fast lookup
    existing_edges = set(zip(edge_index_np[0], edge_index_np[1]))

    # Sample negative edges
    neg_sources = []
    neg_targets = []

    while len(neg_sources) < total_negative:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst and (src, dst) not in existing_edges:
            neg_sources.append(src)
            neg_targets.append(dst)

    neg_edge_index = np.array([neg_sources, neg_targets], dtype=np.int64)

    if use_torch:
        return torch.tensor(neg_edge_index), torch.zeros(total_negative)
    return neg_edge_index, np.zeros(total_negative)


def split_edges(
    edge_index: Any,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Any, Any, Any]:
    """
    Split edges into train/val/test sets.

    Args:
        edge_index: Edge indices [2, num_edges]
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_edges, val_edges, test_edges)
    """
    np.random.seed(seed)

    if TORCH_AVAILABLE and isinstance(edge_index, torch.Tensor):
        edge_index_np = edge_index.numpy()
        use_torch = True
    else:
        edge_index_np = np.array(edge_index)
        use_torch = False

    num_edges = edge_index_np.shape[1]
    perm = np.random.permutation(num_edges)

    train_size = int(num_edges * train_ratio)
    val_size = int(num_edges * val_ratio)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    train_edges = edge_index_np[:, train_idx]
    val_edges = edge_index_np[:, val_idx]
    test_edges = edge_index_np[:, test_idx]

    if use_torch:
        return (
            torch.tensor(train_edges),
            torch.tensor(val_edges),
            torch.tensor(test_edges),
        )
    return train_edges, val_edges, test_edges


def get_subgraph(
    edge_index: Any,
    node_indices: Any,
    relabel: bool = True,
) -> Tuple[Any, Optional[Dict[int, int]]]:
    """
    Extract subgraph containing specified nodes.

    Args:
        edge_index: Edge indices [2, num_edges]
        node_indices: Indices of nodes to keep
        relabel: Whether to relabel nodes to 0...N-1

    Returns:
        Tuple of (subgraph edges, relabel mapping if relabeled)
    """
    if TORCH_AVAILABLE and isinstance(edge_index, torch.Tensor):
        edge_index_np = edge_index.numpy()
        node_indices_np = node_indices.numpy() if isinstance(node_indices, torch.Tensor) else np.array(node_indices)
        use_torch = True
    else:
        edge_index_np = np.array(edge_index)
        node_indices_np = np.array(node_indices)
        use_torch = False

    node_set = set(node_indices_np)

    # Filter edges
    mask = np.array([
        (src in node_set) and (dst in node_set)
        for src, dst in zip(edge_index_np[0], edge_index_np[1])
    ])

    sub_edges = edge_index_np[:, mask]

    relabel_map = None
    if relabel:
        # Create mapping from old to new indices
        relabel_map = {old: new for new, old in enumerate(sorted(node_set))}
        sub_edges = np.array([
            [relabel_map[src] for src in sub_edges[0]],
            [relabel_map[dst] for dst in sub_edges[1]],
        ])

    if use_torch:
        return torch.tensor(sub_edges), relabel_map
    return sub_edges, relabel_map
