"""
Network Propagation

Implements diffusion-based signal refinement on biological networks.
Spreads gene-level signals through protein-protein interaction and
pathway networks to improve signal detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

logger = logging.getLogger(__name__)


class PropagationMethod(Enum):
    """Network propagation methods."""

    RANDOM_WALK = "random_walk"  # Random walk with restart
    HEAT_DIFFUSION = "heat_diffusion"  # Heat diffusion kernel
    INSULATED = "insulated"  # Insulated heat diffusion (preserves boundaries)
    PAGERANK = "pagerank"  # Personalized PageRank


@dataclass
class PropagationConfig:
    """Configuration for network propagation."""

    method: PropagationMethod = PropagationMethod.RANDOM_WALK

    # Random walk parameters
    restart_prob: float = 0.5  # Probability of returning to seed (alpha)
    n_iterations: int = 100  # Max iterations for iterative methods
    convergence_threshold: float = 1e-6  # Convergence threshold

    # Heat diffusion parameters
    diffusion_time: float = 0.1  # Diffusion time parameter

    # Edge weight handling
    normalize_edges: bool = True  # Normalize edge weights to probabilities
    use_edge_weights: bool = True  # Use edge weights (vs binary)

    # Node filtering
    min_degree: int = 1  # Minimum node degree to include
    max_degree: int = 1000  # Maximum node degree (filter hubs)

    # Output processing
    normalize_output: bool = True  # Normalize output scores
    preserve_zeros: bool = False  # Keep zero scores as zero


@dataclass
class PropagationResult:
    """Result of network propagation."""

    gene_scores: Dict[str, float]  # Propagated gene scores
    n_iterations: int  # Iterations used (for iterative methods)
    converged: bool  # Whether method converged
    method: str  # Method used
    metadata: Dict[str, Any] = field(default_factory=dict)


class NetworkPropagator:
    """
    Propagates gene-level signals through biological networks.

    Uses random walk with restart (RWR) or heat diffusion to spread
    signals from seed genes to their network neighbors.
    """

    def __init__(self, config: Optional[PropagationConfig] = None):
        """
        Initialize network propagator.

        Args:
            config: Propagation configuration
        """
        self.config = config or PropagationConfig()
        self._adj_matrix: Optional[sparse.csr_matrix] = None
        self._node_to_idx: Dict[str, int] = {}
        self._idx_to_node: Dict[int, str] = {}

    def build_network(
        self,
        knowledge_graph: Any,  # KnowledgeGraph from Module 03
        edge_types: Optional[List[str]] = None,
    ) -> None:
        """
        Build propagation network from knowledge graph.

        Args:
            knowledge_graph: KnowledgeGraph instance
            edge_types: Edge types to include (default: gene_interacts_gene)
        """
        if edge_types is None:
            edge_types = ["gene_interacts_gene"]

        # Get gene nodes
        gene_nodes = knowledge_graph.get_nodes_by_type(
            knowledge_graph.schema.node_types.__class__.GENE
            if hasattr(knowledge_graph.schema.node_types, '__class__')
            else "gene"
        )

        if not gene_nodes:
            # Fallback: get all nodes that look like genes
            gene_nodes = [
                n for n in knowledge_graph.graph.nodes()
                if knowledge_graph.graph.nodes[n].get("node_type") == "gene"
            ]

        # Build node index
        self._node_to_idx = {node: idx for idx, node in enumerate(gene_nodes)}
        self._idx_to_node = {idx: node for node, idx in self._node_to_idx.items()}
        n_nodes = len(gene_nodes)

        logger.info(f"Building propagation network with {n_nodes} gene nodes")

        # Build adjacency matrix
        row_indices = []
        col_indices = []
        weights = []

        for source, target, data in knowledge_graph.graph.edges(data=True):
            edge_type = data.get("edge_type", "")
            if edge_type not in edge_types:
                continue

            if source not in self._node_to_idx or target not in self._node_to_idx:
                continue

            src_idx = self._node_to_idx[source]
            tgt_idx = self._node_to_idx[target]
            weight = data.get("weight", 1.0) if self.config.use_edge_weights else 1.0

            # Add edge in both directions (symmetric)
            row_indices.extend([src_idx, tgt_idx])
            col_indices.extend([tgt_idx, src_idx])
            weights.extend([weight, weight])

        # Create sparse adjacency matrix
        self._adj_matrix = sparse.csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes),
        )

        # Remove self-loops
        self._adj_matrix.setdiag(0)
        self._adj_matrix.eliminate_zeros()

        # Apply degree filtering
        degrees = np.array(self._adj_matrix.sum(axis=1)).flatten()
        valid_nodes = (degrees >= self.config.min_degree) & (degrees <= self.config.max_degree)

        if not np.all(valid_nodes):
            n_filtered = np.sum(~valid_nodes)
            logger.info(f"Filtered {n_filtered} nodes by degree constraints")

        logger.info(
            f"Built network: {n_nodes} nodes, {self._adj_matrix.nnz // 2} edges"
        )

    def build_network_from_edges(
        self,
        edges: List[Tuple[str, str, float]],
        nodes: Optional[List[str]] = None,
    ) -> None:
        """
        Build propagation network from edge list.

        Args:
            edges: List of (source, target, weight) tuples
            nodes: Optional list of nodes (inferred from edges if not provided)
        """
        # Get node set
        if nodes is None:
            node_set: Set[str] = set()
            for src, tgt, _ in edges:
                node_set.add(src)
                node_set.add(tgt)
            nodes = sorted(node_set)

        # Build node index
        self._node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self._idx_to_node = {idx: node for node, idx in self._node_to_idx.items()}
        n_nodes = len(nodes)

        # Build adjacency matrix
        row_indices = []
        col_indices = []
        weights = []

        for src, tgt, weight in edges:
            if src not in self._node_to_idx or tgt not in self._node_to_idx:
                continue

            src_idx = self._node_to_idx[src]
            tgt_idx = self._node_to_idx[tgt]

            row_indices.extend([src_idx, tgt_idx])
            col_indices.extend([tgt_idx, src_idx])
            weights.extend([weight, weight])

        self._adj_matrix = sparse.csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes),
        )

        self._adj_matrix.setdiag(0)
        self._adj_matrix.eliminate_zeros()

        logger.info(f"Built network: {n_nodes} nodes, {len(edges)} edges")

    def propagate(
        self,
        gene_scores: Dict[str, float],
    ) -> PropagationResult:
        """
        Propagate gene scores through the network.

        Args:
            gene_scores: Dictionary mapping gene ID to score

        Returns:
            PropagationResult with propagated scores
        """
        if self._adj_matrix is None:
            raise RuntimeError("Network not built. Call build_network() first.")

        method = self.config.method

        if method == PropagationMethod.RANDOM_WALK:
            return self._random_walk_with_restart(gene_scores)
        elif method == PropagationMethod.HEAT_DIFFUSION:
            return self._heat_diffusion(gene_scores)
        elif method == PropagationMethod.INSULATED:
            return self._insulated_diffusion(gene_scores)
        elif method == PropagationMethod.PAGERANK:
            return self._personalized_pagerank(gene_scores)
        else:
            raise ValueError(f"Unknown propagation method: {method}")

    def _random_walk_with_restart(
        self,
        gene_scores: Dict[str, float],
    ) -> PropagationResult:
        """
        Random walk with restart propagation.

        p_t+1 = (1 - alpha) * W * p_t + alpha * p_0

        Args:
            gene_scores: Seed gene scores

        Returns:
            PropagationResult
        """
        n_nodes = self._adj_matrix.shape[0]
        alpha = self.config.restart_prob

        # Build transition matrix (row-normalized adjacency)
        W = self._normalize_adjacency(self._adj_matrix)

        # Initialize seed vector
        p0 = np.zeros(n_nodes)
        for gene, score in gene_scores.items():
            if gene in self._node_to_idx:
                p0[self._node_to_idx[gene]] = score

        # Normalize seed
        if np.sum(p0) > 0:
            p0 = p0 / np.sum(p0)

        # Iterate until convergence
        p = p0.copy()
        converged = False

        for iteration in range(self.config.n_iterations):
            p_new = (1 - alpha) * W.dot(p) + alpha * p0

            # Check convergence
            diff = np.max(np.abs(p_new - p))
            if diff < self.config.convergence_threshold:
                converged = True
                p = p_new
                break

            p = p_new

        # Convert back to gene scores
        propagated = self._vector_to_scores(p, gene_scores)

        return PropagationResult(
            gene_scores=propagated,
            n_iterations=iteration + 1,
            converged=converged,
            method="random_walk",
            metadata={"alpha": alpha},
        )

    def _heat_diffusion(
        self,
        gene_scores: Dict[str, float],
    ) -> PropagationResult:
        """
        Heat diffusion kernel propagation.

        p = exp(-t * L) * p_0

        where L is the Laplacian matrix.

        Args:
            gene_scores: Seed gene scores

        Returns:
            PropagationResult
        """
        n_nodes = self._adj_matrix.shape[0]
        t = self.config.diffusion_time

        # Compute Laplacian: L = D - A
        degrees = np.array(self._adj_matrix.sum(axis=1)).flatten()
        D = sparse.diags(degrees)
        L = D - self._adj_matrix

        # Initialize seed vector
        p0 = np.zeros(n_nodes)
        for gene, score in gene_scores.items():
            if gene in self._node_to_idx:
                p0[self._node_to_idx[gene]] = score

        # Apply heat kernel: exp(-t * L) * p0
        # Using scipy's expm_multiply for efficiency
        p = expm_multiply(-t * L, p0)

        # Convert back to gene scores
        propagated = self._vector_to_scores(p, gene_scores)

        return PropagationResult(
            gene_scores=propagated,
            n_iterations=1,
            converged=True,
            method="heat_diffusion",
            metadata={"diffusion_time": t},
        )

    def _insulated_diffusion(
        self,
        gene_scores: Dict[str, float],
    ) -> PropagationResult:
        """
        Insulated heat diffusion (preserves signal at boundaries).

        Args:
            gene_scores: Seed gene scores

        Returns:
            PropagationResult
        """
        n_nodes = self._adj_matrix.shape[0]
        t = self.config.diffusion_time

        # Normalized Laplacian: L_norm = I - D^(-1/2) * A * D^(-1/2)
        degrees = np.array(self._adj_matrix.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero

        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
        L_norm = sparse.eye(n_nodes) - D_inv_sqrt @ self._adj_matrix @ D_inv_sqrt

        # Initialize seed vector
        p0 = np.zeros(n_nodes)
        for gene, score in gene_scores.items():
            if gene in self._node_to_idx:
                p0[self._node_to_idx[gene]] = score

        # Apply normalized heat kernel
        p = expm_multiply(-t * L_norm, p0)

        # Convert back to gene scores
        propagated = self._vector_to_scores(p, gene_scores)

        return PropagationResult(
            gene_scores=propagated,
            n_iterations=1,
            converged=True,
            method="insulated_diffusion",
            metadata={"diffusion_time": t},
        )

    def _personalized_pagerank(
        self,
        gene_scores: Dict[str, float],
    ) -> PropagationResult:
        """
        Personalized PageRank propagation.

        Similar to RWR but with specific damping formulation.

        Args:
            gene_scores: Seed gene scores

        Returns:
            PropagationResult
        """
        # PPR is essentially the same as RWR with alpha as damping
        return self._random_walk_with_restart(gene_scores)

    def _normalize_adjacency(
        self,
        adj: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """
        Row-normalize adjacency matrix to create transition matrix.

        Args:
            adj: Adjacency matrix

        Returns:
            Row-normalized matrix
        """
        row_sums = np.array(adj.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        if self.config.normalize_edges:
            # Create diagonal matrix of inverse row sums
            D_inv = sparse.diags(1.0 / row_sums)
            return D_inv @ adj
        else:
            return adj

    def _vector_to_scores(
        self,
        p: np.ndarray,
        original_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Convert propagated vector back to gene score dictionary.

        Args:
            p: Propagated score vector
            original_scores: Original input scores

        Returns:
            Dictionary mapping gene ID to propagated score
        """
        # Normalize if configured
        if self.config.normalize_output and np.max(p) > 0:
            p = p / np.max(p)

        result = {}
        for idx, score in enumerate(p):
            gene = self._idx_to_node.get(idx)
            if gene is None:
                continue

            # Optionally preserve zeros
            if self.config.preserve_zeros:
                if gene not in original_scores or original_scores[gene] == 0:
                    if score < 1e-10:
                        continue

            if score > 1e-10:  # Filter very small values
                result[gene] = float(score)

        return result

    def propagate_gene_burdens(
        self,
        gene_burdens: Any,  # GeneBurdenMatrix from Module 02
    ) -> Any:  # Returns GeneBurdenMatrix
        """
        Propagate all samples' gene burdens through the network.

        Args:
            gene_burdens: GeneBurdenMatrix with gene-level scores

        Returns:
            New GeneBurdenMatrix with propagated scores
        """
        if self._adj_matrix is None:
            raise RuntimeError("Network not built. Call build_network() first.")

        logger.info(
            f"Propagating gene burdens for {gene_burdens.n_samples} samples"
        )

        # Get all genes in network
        network_genes = list(self._node_to_idx.keys())

        # Build new burden matrix with propagated scores
        new_scores = np.zeros((gene_burdens.n_samples, len(network_genes)))

        for s_idx, sample_id in enumerate(gene_burdens.samples):
            # Get this sample's gene scores
            sample_scores = gene_burdens.get_sample(sample_id)

            if not sample_scores:
                continue

            # Propagate
            result = self.propagate(sample_scores)

            # Fill in propagated scores
            for gene, score in result.gene_scores.items():
                if gene in self._node_to_idx:
                    g_idx = network_genes.index(gene)
                    new_scores[s_idx, g_idx] = score

        # Import dynamically to avoid circular dependency and handle numeric module prefix
        import importlib
        gene_burden_module = importlib.import_module(
            "gene_burden",
            package="modules.02_variant_processing"
        )
        # Fallback: try direct path import
        import sys
        from pathlib import Path
        module_02_path = Path(__file__).parent.parent / "02_variant_processing"
        if str(module_02_path) not in sys.path:
            sys.path.insert(0, str(module_02_path))
        try:
            from gene_burden import GeneBurdenMatrix
        except ImportError:
            # Create a simple dataclass if module not available
            from dataclasses import dataclass, field
            @dataclass
            class GeneBurdenMatrix:
                samples: list
                genes: list
                scores: np.ndarray
                metadata: dict = field(default_factory=dict)

        return GeneBurdenMatrix(
            samples=gene_burdens.samples.copy(),
            genes=network_genes,
            scores=new_scores,
            metadata={
                **gene_burdens.metadata,
                "propagated": True,
                "propagation_method": self.config.method.value,
            },
        )

    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the propagation network."""
        if self._adj_matrix is None:
            return {"error": "Network not built"}

        degrees = np.array(self._adj_matrix.sum(axis=1)).flatten()

        return {
            "n_nodes": self._adj_matrix.shape[0],
            "n_edges": self._adj_matrix.nnz // 2,
            "avg_degree": float(np.mean(degrees)),
            "max_degree": int(np.max(degrees)),
            "min_degree": int(np.min(degrees[degrees > 0])) if np.any(degrees > 0) else 0,
            "density": self._adj_matrix.nnz / (self._adj_matrix.shape[0] ** 2),
        }
