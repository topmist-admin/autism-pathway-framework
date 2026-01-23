"""
Structural Causal Model implementation for ASD genetics.

Encodes the causal chain:
Genetic Variants -> Gene Function Disruption -> Pathway Perturbation
                -> Circuit-Level Effects -> Behavioral Phenotype

With explicit confounders:
- Ancestry (population stratification)
- Batch effects (technical confounders)
- Ascertainment bias (diagnostic confounders)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CausalNodeType(Enum):
    """Types of nodes in the causal graph."""
    VARIANT = "variant"
    GENE_FUNCTION = "gene_function"
    PATHWAY = "pathway"
    CIRCUIT = "circuit"
    PHENOTYPE = "phenotype"
    CONFOUNDER = "confounder"


class CausalEdgeType(Enum):
    """Types of causal relationships."""
    CAUSES = "causes"
    MEDIATES = "mediates"
    CONFOUNDS = "confounds"
    MODIFIES = "modifies"


@dataclass
class CausalNode:
    """A node in the structural causal model."""
    id: str
    node_type: CausalNodeType
    observed: bool
    value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raise ValueError("Node id cannot be empty")
        if self.value is not None and not (0 <= self.value <= 1):
            # Allow values outside 0-1 for some use cases, just log
            logger.debug(f"Node {self.id} has value {self.value} outside [0,1]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "observed": self.observed,
            "value": self.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalNode":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            node_type=CausalNodeType(data["node_type"]),
            observed=data["observed"],
            value=data.get("value"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CausalEdge:
    """A directed edge in the causal graph."""
    source: str
    target: str
    edge_type: CausalEdgeType
    strength: float  # Estimated causal effect strength [0, 1]
    mechanism: str  # Biological mechanism description

    def __post_init__(self):
        if not self.source or not self.target:
            raise ValueError("Edge source and target cannot be empty")
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Edge strength must be between 0 and 1, got {self.strength}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "strength": self.strength,
            "mechanism": self.mechanism,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalEdge":
        """Deserialize from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            edge_type=CausalEdgeType(data["edge_type"]),
            strength=data["strength"],
            mechanism=data["mechanism"],
        )


@dataclass
class CausalQuery:
    """Structured representation of a causal query."""
    query_type: str  # "intervention", "counterfactual", "effect"
    treatment: str
    outcome: str
    intervention_value: Optional[float] = None
    conditioning: Optional[Dict[str, float]] = None
    mediator: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query_type": self.query_type,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "intervention_value": self.intervention_value,
            "conditioning": self.conditioning,
            "mediator": self.mediator,
        }


class CausalQueryBuilder:
    """Fluent interface for building causal queries."""

    def __init__(self):
        self._treatment: Optional[str] = None
        self._outcome: Optional[str] = None
        self._conditioning: Optional[Dict[str, float]] = None
        self._intervention: Optional[Dict[str, float]] = None
        self._mediator: Optional[str] = None
        self._query_type: str = "intervention"

    def treatment(self, var: str) -> "CausalQueryBuilder":
        """Set the treatment variable."""
        self._treatment = var
        return self

    def outcome(self, var: str) -> "CausalQueryBuilder":
        """Set the outcome variable."""
        self._outcome = var
        return self

    def given(self, evidence: Dict[str, float]) -> "CausalQueryBuilder":
        """Set conditioning evidence."""
        self._conditioning = evidence
        return self

    def do(self, intervention: Dict[str, float]) -> "CausalQueryBuilder":
        """Set do-intervention."""
        self._intervention = intervention
        self._query_type = "intervention"
        return self

    def mediated_by(self, mediator: str) -> "CausalQueryBuilder":
        """Set mediator for effect decomposition."""
        self._mediator = mediator
        self._query_type = "effect"
        return self

    def counterfactual(self) -> "CausalQueryBuilder":
        """Mark as counterfactual query."""
        self._query_type = "counterfactual"
        return self

    def build(self) -> CausalQuery:
        """Build the causal query."""
        if not self._treatment:
            raise ValueError("Treatment variable must be specified")
        if not self._outcome:
            raise ValueError("Outcome variable must be specified")

        intervention_value = None
        if self._intervention and self._treatment in self._intervention:
            intervention_value = self._intervention[self._treatment]

        return CausalQuery(
            query_type=self._query_type,
            treatment=self._treatment,
            outcome=self._outcome,
            intervention_value=intervention_value,
            conditioning=self._conditioning,
            mediator=self._mediator,
        )


class StructuralCausalModel:
    """
    Structural Causal Model for ASD genetics.

    Encodes the causal chain:
    Genetic Variants -> Gene Function Disruption -> Pathway Perturbation
                    -> Circuit-Level Effects -> Behavioral Phenotype

    With explicit confounders:
    - Ancestry
    - Batch effects
    - Ascertainment bias
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.structural_equations: Dict[str, Callable] = {}
        self._parent_cache: Dict[str, List[str]] = {}
        self._child_cache: Dict[str, List[str]] = {}
        self._cache_valid = False

    def add_node(self, node: CausalNode) -> None:
        """Add a node to the causal model."""
        if node.id in self.nodes:
            logger.warning(f"Node {node.id} already exists, overwriting")
        self.nodes[node.id] = node
        self._invalidate_cache()

    def add_edge(self, edge: CausalEdge) -> None:
        """Add a causal edge to the model."""
        if edge.source not in self.nodes:
            raise ValueError(f"Source node {edge.source} not found")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node {edge.target} not found")

        # Check for duplicate edge
        for existing in self.edges:
            if existing.source == edge.source and existing.target == edge.target:
                logger.warning(f"Edge {edge.source} -> {edge.target} already exists, overwriting")
                self.edges.remove(existing)
                break

        self.edges.append(edge)
        self._invalidate_cache()

    def remove_edge(self, source: str, target: str) -> bool:
        """Remove an edge from the model."""
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                self.edges.remove(edge)
                self._invalidate_cache()
                return True
        return False

    def set_structural_equation(self, node_id: str, equation: Callable) -> None:
        """
        Set the structural equation for a node.

        The equation should be a function that takes parent values as kwargs
        and returns the node value.
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        self.structural_equations[node_id] = equation

    def _invalidate_cache(self) -> None:
        """Invalidate cached relationships."""
        self._cache_valid = False
        self._parent_cache.clear()
        self._child_cache.clear()

    def _build_cache(self) -> None:
        """Build parent/child relationship caches."""
        if self._cache_valid:
            return

        self._parent_cache = {node_id: [] for node_id in self.nodes}
        self._child_cache = {node_id: [] for node_id in self.nodes}

        for edge in self.edges:
            self._parent_cache[edge.target].append(edge.source)
            self._child_cache[edge.source].append(edge.target)

        self._cache_valid = True

    def get_parents(self, node_id: str) -> List[str]:
        """Get direct parents of a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        self._build_cache()
        return self._parent_cache.get(node_id, []).copy()

    def get_children(self, node_id: str) -> List[str]:
        """Get direct children of a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        self._build_cache()
        return self._child_cache.get(node_id, []).copy()

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Get all ancestors of a node (transitive parents)."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        ancestors = set()
        queue = deque(self.get_parents(node_id))

        while queue:
            parent = queue.popleft()
            if parent not in ancestors:
                ancestors.add(parent)
                queue.extend(self.get_parents(parent))

        return ancestors

    def get_descendants(self, node_id: str) -> Set[str]:
        """Get all descendants of a node (transitive children)."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        descendants = set()
        queue = deque(self.get_children(node_id))

        while queue:
            child = queue.popleft()
            if child not in descendants:
                descendants.add(child)
                queue.extend(self.get_children(child))

        return descendants

    def get_edge(self, source: str, target: str) -> Optional[CausalEdge]:
        """Get the edge between two nodes if it exists."""
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return edge
        return None

    def get_edge_strength(self, source: str, target: str) -> float:
        """Get the causal strength of an edge."""
        edge = self.get_edge(source, target)
        return edge.strength if edge else 0.0

    def is_d_separated(
        self,
        x: str,
        y: str,
        conditioning: Set[str]
    ) -> bool:
        """
        Test if X and Y are d-separated given conditioning set Z.

        Uses a path-based approach to determine d-separation.
        X and Y are d-separated given Z if all paths between them are blocked.
        """
        if x not in self.nodes or y not in self.nodes:
            raise ValueError(f"Nodes {x} or {y} not found")

        # Special case: if x == y, they are not d-separated
        if x == y:
            return False

        # Find all undirected paths from x to y
        all_paths = self._find_all_undirected_paths(x, y)

        # If no paths exist, they are d-separated
        if not all_paths:
            return True

        # Check if ALL paths are blocked
        for path in all_paths:
            if not self._is_path_blocked(path, conditioning):
                return False  # Found an active (unblocked) path

        return True  # All paths are blocked

    def _find_all_undirected_paths(self, start: str, end: str) -> List[List[str]]:
        """Find all undirected paths between two nodes."""
        all_paths = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if current == end:
                all_paths.append(path.copy())
                return

            # Get all neighbors (undirected)
            neighbors = set()
            for edge in self.edges:
                if edge.source == current:
                    neighbors.add(edge.target)
                elif edge.target == current:
                    neighbors.add(edge.source)

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        visited = {start}
        dfs(start, [start], visited)
        return all_paths

    def _is_path_blocked(self, path: List[str], conditioning: Set[str]) -> bool:
        """
        Check if a path is blocked by the conditioning set.

        A path is blocked if ANY of its intermediate nodes satisfies:
        1. It's a non-collider (chain or fork) AND is conditioned on
        2. It's a collider AND neither it nor any of its descendants is conditioned on
        """
        if len(path) < 3:
            return False  # No intermediate nodes, path is active

        # Check each intermediate node (not first or last)
        for i in range(1, len(path) - 1):
            node = path[i]
            prev_node = path[i - 1]
            next_node = path[i + 1]

            # Determine if this node is a collider on this path
            # A collider has both adjacent edges pointing INTO it
            # prev -> node <- next
            edge_from_prev = self.get_edge(prev_node, node)
            edge_from_next = self.get_edge(next_node, node)

            is_collider = (edge_from_prev is not None) and (edge_from_next is not None)

            if is_collider:
                # Collider: path is blocked UNLESS we condition on the collider
                # or any of its descendants
                descendants_and_self = self.get_descendants(node) | {node}
                if not (conditioning & descendants_and_self):
                    return True  # Blocked at this collider
            else:
                # Non-collider (chain or fork): path is blocked IF we condition on it
                if node in conditioning:
                    return True  # Blocked at this non-collider

        return False  # No blocking found, path is active

    def _is_collider(self, node_id: str) -> bool:
        """Check if a node is a collider (has multiple parents)."""
        return len(self.get_parents(node_id)) >= 2

    def _find_all_paths(
        self,
        start: str,
        end: str,
        directed: bool = False
    ) -> List[List[str]]:
        """
        Find all paths between two nodes.

        If directed=True, only follow edges in their causal direction.
        If directed=False, treat graph as undirected (for backdoor paths).
        """
        if start not in self.nodes or end not in self.nodes:
            return []

        all_paths = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if current == end:
                all_paths.append(path.copy())
                return

            for edge in self.edges:
                # Check if this edge connects to current
                if directed:
                    # Only follow forward direction
                    if edge.source == current and edge.target not in visited:
                        visited.add(edge.target)
                        path.append(edge.target)
                        dfs(edge.target, path, visited)
                        path.pop()
                        visited.remove(edge.target)
                else:
                    # Treat as undirected
                    next_node = None
                    if edge.source == current and edge.target not in visited:
                        next_node = edge.target
                    elif edge.target == current and edge.source not in visited:
                        next_node = edge.source

                    if next_node:
                        visited.add(next_node)
                        path.append(next_node)
                        dfs(next_node, path, visited)
                        path.pop()
                        visited.remove(next_node)

        visited = {start}
        dfs(start, [start], visited)

        return all_paths

    def get_backdoor_paths(
        self,
        treatment: str,
        outcome: str
    ) -> List[List[str]]:
        """
        Find all backdoor paths from treatment to outcome.

        A backdoor path is a path that starts with an arrow into the treatment.
        These are non-causal paths that create confounding.
        """
        if treatment not in self.nodes or outcome not in self.nodes:
            raise ValueError(f"Nodes {treatment} or {outcome} not found")

        backdoor_paths = []

        # Find all undirected paths from treatment to outcome
        all_paths = self._find_all_paths(treatment, outcome, directed=False)

        for path in all_paths:
            if len(path) < 2:
                continue

            # A backdoor path starts with an arrow INTO the treatment
            # i.e., the first edge on the path goes: path[1] -> treatment
            first_step = path[1]
            edge = self.get_edge(first_step, treatment)

            if edge is not None:
                # This is a backdoor path (arrow into treatment)
                backdoor_paths.append(path)

        return backdoor_paths

    def get_valid_adjustment_sets(
        self,
        treatment: str,
        outcome: str
    ) -> List[Set[str]]:
        """
        Find valid adjustment sets for estimating causal effect.

        A valid adjustment set blocks all backdoor paths without blocking
        any causal paths (front-door paths).
        """
        if treatment not in self.nodes or outcome not in self.nodes:
            raise ValueError(f"Nodes {treatment} or {outcome} not found")

        backdoor_paths = self.get_backdoor_paths(treatment, outcome)

        if not backdoor_paths:
            # No backdoor paths, empty set is valid
            return [set()]

        # Find all non-treatment, non-outcome nodes that could be adjusted
        adjustable_nodes = set(self.nodes.keys()) - {treatment, outcome}

        # Remove descendants of treatment (can't adjust for them)
        descendants = self.get_descendants(treatment)
        adjustable_nodes -= descendants

        # Find minimal adjustment sets using a simple approach
        # Try single nodes first, then pairs
        valid_sets = []

        # Check empty set
        all_blocked = all(
            self.is_d_separated(treatment, outcome, set())
            for _ in [None]  # Just to run once
        )
        # Actually need to check each backdoor path
        if not backdoor_paths:
            valid_sets.append(set())

        # Try single nodes
        for node in adjustable_nodes:
            adj_set = {node}
            if self._blocks_all_backdoors(treatment, outcome, adj_set, backdoor_paths):
                valid_sets.append(adj_set)

        # If we found single-node sets, those are minimal
        if valid_sets:
            return valid_sets

        # Try pairs
        adjustable_list = list(adjustable_nodes)
        for i, node1 in enumerate(adjustable_list):
            for node2 in adjustable_list[i + 1:]:
                adj_set = {node1, node2}
                if self._blocks_all_backdoors(treatment, outcome, adj_set, backdoor_paths):
                    valid_sets.append(adj_set)

        # If still no valid sets found, try all adjustable nodes
        if not valid_sets and adjustable_nodes:
            if self._blocks_all_backdoors(treatment, outcome, adjustable_nodes, backdoor_paths):
                valid_sets.append(adjustable_nodes)

        return valid_sets if valid_sets else [set()]

    def _blocks_all_backdoors(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str],
        backdoor_paths: List[List[str]]
    ) -> bool:
        """Check if adjustment set blocks all backdoor paths."""
        for path in backdoor_paths:
            if not self._path_blocked(path, adjustment_set):
                return False
        return True

    def _path_blocked(self, path: List[str], conditioning: Set[str]) -> bool:
        """
        Check if a path is blocked by conditioning set.

        A path is blocked if:
        1. Any non-collider on the path is conditioned on
        2. No collider on the path (or its descendant) is conditioned on
        """
        if len(path) < 3:
            # Path too short to have intermediate nodes
            return False

        for i in range(1, len(path) - 1):
            node = path[i]
            prev_node = path[i - 1]
            next_node = path[i + 1]

            # Check if this is a collider
            # A collider has arrows pointing into it from both sides
            edge_from_prev = self.get_edge(prev_node, node)
            edge_from_next = self.get_edge(next_node, node)

            is_collider = edge_from_prev is not None and edge_from_next is not None

            if is_collider:
                # Collider: blocked unless conditioned on it or its descendant
                descendants = self.get_descendants(node) | {node}
                if not (conditioning & descendants):
                    return True  # Blocked at this collider
            else:
                # Non-collider: blocked if conditioned on
                if node in conditioning:
                    return True  # Blocked at this non-collider

        return False

    def copy(self) -> "StructuralCausalModel":
        """Create a deep copy of the model."""
        new_model = StructuralCausalModel()

        for node_id, node in self.nodes.items():
            new_node = CausalNode(
                id=node.id,
                node_type=node.node_type,
                observed=node.observed,
                value=node.value,
                metadata=node.metadata.copy(),
            )
            new_model.nodes[node_id] = new_node

        for edge in self.edges:
            new_edge = CausalEdge(
                source=edge.source,
                target=edge.target,
                edge_type=edge.edge_type,
                strength=edge.strength,
                mechanism=edge.mechanism,
            )
            new_model.edges.append(new_edge)

        for node_id, eq in self.structural_equations.items():
            new_model.structural_equations[node_id] = eq

        return new_model

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the model to dictionary."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuralCausalModel":
        """Deserialize model from dictionary."""
        model = cls()
        for node_data in data["nodes"]:
            model.add_node(CausalNode.from_dict(node_data))
        for edge_data in data["edges"]:
            model.add_edge(CausalEdge.from_dict(edge_data))
        return model

    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        node_types = {}
        for node in self.nodes.values():
            node_type = node.node_type.value
            node_types[node_type] = node_types.get(node_type, 0) + 1

        edge_types = {}
        for edge in self.edges:
            edge_type = edge.edge_type.value
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "observed_nodes": sum(1 for n in self.nodes.values() if n.observed),
            "latent_nodes": sum(1 for n in self.nodes.values() if not n.observed),
        }


def create_sample_asd_scm() -> StructuralCausalModel:
    """
    Create a sample SCM for ASD genetics.

    Models the causal chain:
    SHANK3_variant -> SHANK3_function -> synaptic_pathway -> asd_phenotype
    CHD8_variant -> CHD8_function -> chromatin_pathway -> asd_phenotype

    With confounders:
    ancestry -> all nodes
    """
    scm = StructuralCausalModel()

    # Add variant nodes
    scm.add_node(CausalNode("SHANK3_variant", CausalNodeType.VARIANT, observed=True))
    scm.add_node(CausalNode("CHD8_variant", CausalNodeType.VARIANT, observed=True))
    scm.add_node(CausalNode("SCN2A_variant", CausalNodeType.VARIANT, observed=True))

    # Add gene function nodes
    scm.add_node(CausalNode("SHANK3_function", CausalNodeType.GENE_FUNCTION, observed=True))
    scm.add_node(CausalNode("CHD8_function", CausalNodeType.GENE_FUNCTION, observed=True))
    scm.add_node(CausalNode("SCN2A_function", CausalNodeType.GENE_FUNCTION, observed=True))

    # Add pathway nodes
    scm.add_node(CausalNode("synaptic_pathway", CausalNodeType.PATHWAY, observed=True))
    scm.add_node(CausalNode("chromatin_pathway", CausalNodeType.PATHWAY, observed=True))
    scm.add_node(CausalNode("ion_channel_pathway", CausalNodeType.PATHWAY, observed=True))

    # Add circuit node
    scm.add_node(CausalNode("cortical_circuit", CausalNodeType.CIRCUIT, observed=False))

    # Add phenotype node
    scm.add_node(CausalNode("asd_phenotype", CausalNodeType.PHENOTYPE, observed=True))

    # Add confounder
    scm.add_node(CausalNode("ancestry", CausalNodeType.CONFOUNDER, observed=True))

    # Add causal edges: variant -> function
    scm.add_edge(CausalEdge(
        "SHANK3_variant", "SHANK3_function",
        CausalEdgeType.CAUSES, 0.9, "Loss of function variant"
    ))
    scm.add_edge(CausalEdge(
        "CHD8_variant", "CHD8_function",
        CausalEdgeType.CAUSES, 0.85, "Haploinsufficiency"
    ))
    scm.add_edge(CausalEdge(
        "SCN2A_variant", "SCN2A_function",
        CausalEdgeType.CAUSES, 0.8, "Channel dysfunction"
    ))

    # Add causal edges: function -> pathway
    scm.add_edge(CausalEdge(
        "SHANK3_function", "synaptic_pathway",
        CausalEdgeType.CAUSES, 0.85, "Scaffold protein disruption"
    ))
    scm.add_edge(CausalEdge(
        "CHD8_function", "chromatin_pathway",
        CausalEdgeType.CAUSES, 0.8, "Chromatin remodeling disruption"
    ))
    scm.add_edge(CausalEdge(
        "SCN2A_function", "ion_channel_pathway",
        CausalEdgeType.CAUSES, 0.75, "Sodium channel dysfunction"
    ))

    # Cross-pathway effects
    scm.add_edge(CausalEdge(
        "CHD8_function", "synaptic_pathway",
        CausalEdgeType.MODIFIES, 0.3, "Transcriptional regulation of synaptic genes"
    ))

    # Add causal edges: pathway -> circuit
    scm.add_edge(CausalEdge(
        "synaptic_pathway", "cortical_circuit",
        CausalEdgeType.CAUSES, 0.7, "Synaptic transmission disruption"
    ))
    scm.add_edge(CausalEdge(
        "chromatin_pathway", "cortical_circuit",
        CausalEdgeType.CAUSES, 0.5, "Developmental timing disruption"
    ))
    scm.add_edge(CausalEdge(
        "ion_channel_pathway", "cortical_circuit",
        CausalEdgeType.CAUSES, 0.6, "Excitability imbalance"
    ))

    # Add causal edges: circuit -> phenotype
    scm.add_edge(CausalEdge(
        "cortical_circuit", "asd_phenotype",
        CausalEdgeType.CAUSES, 0.8, "Circuit dysfunction"
    ))

    # Add direct pathway -> phenotype edges (for some pathways)
    scm.add_edge(CausalEdge(
        "synaptic_pathway", "asd_phenotype",
        CausalEdgeType.CAUSES, 0.3, "Direct synaptic effects"
    ))

    # Add confounder edges
    scm.add_edge(CausalEdge(
        "ancestry", "SHANK3_variant",
        CausalEdgeType.CONFOUNDS, 0.2, "Population stratification"
    ))
    scm.add_edge(CausalEdge(
        "ancestry", "asd_phenotype",
        CausalEdgeType.CONFOUNDS, 0.1, "Diagnostic bias"
    ))

    return scm
