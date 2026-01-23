"""
Neurosymbolic integration combining GNN with rule-based inference.

This module provides the NeuroSymbolicModel that unifies neural network
predictions with symbolic rule-based inferences for autism genetics analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime

import numpy as np

import sys
from pathlib import Path

# Add module to path for imports
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from combiner import (
    LearnedCombiner,
    CombinerConfig,
    CombinationMethod,
    combine_gene_scores,
    create_symbolic_score_vector,
)

# Optional PyTorch support
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None

logger = logging.getLogger(__name__)


@dataclass
class NeuroSymbolicConfig:
    """Configuration for neurosymbolic model."""

    # Combination settings
    combination_method: str = "weighted_sum"  # weighted_sum, attention, gating, learned
    neural_weight: float = 0.6
    symbolic_weight: float = 0.4

    # Neural settings
    use_neural_embeddings: bool = True
    neural_output_dim: int = 128

    # Symbolic settings
    use_rule_confidence: bool = True
    min_rule_confidence: float = 0.5

    # Output settings
    normalize_outputs: bool = True
    temperature: float = 1.0

    # Training settings
    learnable_combination: bool = False
    dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration."""
        valid_methods = ["weighted_sum", "attention", "gating", "learned", "rule_guided", "max"]
        if self.combination_method not in valid_methods:
            raise ValueError(f"combination_method must be one of {valid_methods}")


@dataclass
class NeuroSymbolicOutput:
    """Output from neurosymbolic model."""

    # Combined predictions
    predictions: Dict[str, float] = field(default_factory=dict)

    # Contribution breakdown
    neural_contribution: Dict[str, float] = field(default_factory=dict)
    symbolic_contribution: Dict[str, float] = field(default_factory=dict)

    # Symbolic component details
    fired_rules: List[Any] = field(default_factory=list)

    # Explanation
    explanation: str = ""

    # Optional neural embeddings
    neural_embeddings: Optional[Dict[str, Any]] = None

    # Metadata
    confidence: float = 0.0
    combination_weights: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "predictions": self.predictions,
            "neural_contribution": self.neural_contribution,
            "symbolic_contribution": self.symbolic_contribution,
            "fired_rules": [
                fr.to_dict() if hasattr(fr, "to_dict") else str(fr)
                for fr in self.fired_rules
            ],
            "explanation": self.explanation,
            "confidence": self.confidence,
            "combination_weights": self.combination_weights,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @property
    def top_genes(self) -> List[Tuple[str, float]]:
        """Get top scoring genes."""
        return sorted(
            self.predictions.items(),
            key=lambda x: -x[1]
        )[:20]

    @property
    def rules_fired_count(self) -> int:
        """Number of rules that fired."""
        return len(self.fired_rules)


class NeuroSymbolicModel:
    """
    Unified neurosymbolic model combining GNN with rule-based inference.

    This model integrates:
    1. Neural pathway: GNN for learning gene representations
    2. Symbolic pathway: Rule engine for biological inference
    3. Combination: Learned or fixed strategies for combining predictions
    """

    def __init__(
        self,
        neural_model: Optional[Any] = None,
        rule_engine: Optional[Any] = None,
        config: Optional[NeuroSymbolicConfig] = None,
    ):
        """
        Initialize neurosymbolic model.

        Args:
            neural_model: GNN model (OntologyAwareGNN from Module 06)
            rule_engine: Rule engine (RuleEngine from Module 09)
            config: Model configuration
        """
        self.neural_model = neural_model
        self.rule_engine = rule_engine
        self.config = config or NeuroSymbolicConfig()

        # Initialize combiner
        method = CombinationMethod(self.config.combination_method)
        combiner_config = CombinerConfig(
            method=method,
            neural_weight=self.config.neural_weight,
            symbolic_weight=self.config.symbolic_weight,
            temperature=self.config.temperature,
            dropout=self.config.dropout,
            use_rule_confidence=self.config.use_rule_confidence,
            normalize_outputs=self.config.normalize_outputs,
        )
        self.combiner = LearnedCombiner(
            config=combiner_config,
            neural_dim=self.config.neural_output_dim,
            symbolic_dim=64,
            output_dim=64,
        )

        self._training = False

    def forward(
        self,
        individual_data: Any,
        graph_data: Optional[Any] = None,
        node_features: Optional[Dict[str, Any]] = None,
        edge_index: Optional[Any] = None,
        edge_type: Optional[Any] = None,
        bio_priors: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> NeuroSymbolicOutput:
        """
        Forward pass combining neural and symbolic predictions.

        Args:
            individual_data: IndividualData from Module 09
            graph_data: Optional GraphData from Module 06
            node_features: Node features for GNN (alternative to graph_data)
            edge_index: Edge indices for GNN
            edge_type: Edge types for GNN
            bio_priors: Biological priors for GNN
            **kwargs: Additional arguments for GNN

        Returns:
            NeuroSymbolicOutput with combined predictions
        """
        # Get neural predictions
        neural_output = self._neural_forward(
            graph_data=graph_data,
            node_features=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            bio_priors=bio_priors,
            **kwargs,
        )

        # Get symbolic predictions
        symbolic_output = self._symbolic_forward(individual_data)

        # Combine predictions
        combined_output = self._combine_outputs(
            neural_output=neural_output,
            symbolic_output=symbolic_output,
            individual_data=individual_data,
        )

        return combined_output

    def _neural_forward(
        self,
        graph_data: Optional[Any] = None,
        node_features: Optional[Dict[str, Any]] = None,
        edge_index: Optional[Any] = None,
        edge_type: Optional[Any] = None,
        bio_priors: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run neural (GNN) forward pass.

        Returns dict with:
        - gene_scores: Dict[str, float]
        - embeddings: Optional embeddings
        """
        if self.neural_model is None:
            logger.warning("No neural model provided, using dummy scores")
            return {"gene_scores": {}, "embeddings": None}

        try:
            # Prepare inputs
            if graph_data is not None:
                # Use graph_data container
                output = self.neural_model(
                    node_features=graph_data.node_features,
                    edge_index=graph_data.edge_index,
                    edge_type=graph_data.edge_type,
                    bio_priors=graph_data.bio_priors if hasattr(graph_data, "bio_priors") else None,
                    node_type_indices=graph_data.node_type_indices if hasattr(graph_data, "node_type_indices") else None,
                    **kwargs,
                )
            elif node_features is not None:
                # Use direct inputs
                output = self.neural_model(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    bio_priors=bio_priors,
                    **kwargs,
                )
            else:
                logger.warning("No graph data provided for neural forward")
                return {"gene_scores": {}, "embeddings": None}

            # Extract gene scores
            gene_scores = {}
            embeddings = None

            if hasattr(output, "node_embeddings"):
                embeddings = output.node_embeddings
                if "gene" in output.node_embeddings:
                    gene_emb = output.node_embeddings["gene"]
                    # Convert embeddings to scores (e.g., mean or norm)
                    if HAS_TORCH and isinstance(gene_emb, torch.Tensor):
                        scores = gene_emb.mean(dim=-1).detach().cpu().numpy()
                    else:
                        scores = np.mean(gene_emb, axis=-1)

                    # Map to gene IDs if available
                    if hasattr(graph_data, "gene_ids"):
                        for i, gene_id in enumerate(graph_data.gene_ids):
                            if i < len(scores):
                                gene_scores[gene_id] = float(scores[i])

            if hasattr(output, "gene_logits") and output.gene_logits is not None:
                # Use logits as scores
                logits = output.gene_logits
                if HAS_TORCH and isinstance(logits, torch.Tensor):
                    logits = logits.detach().cpu().numpy()
                if logits.ndim > 1:
                    logits = logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]

                if hasattr(graph_data, "gene_ids"):
                    for i, gene_id in enumerate(graph_data.gene_ids):
                        if i < len(logits):
                            gene_scores[gene_id] = float(logits[i])

            return {"gene_scores": gene_scores, "embeddings": embeddings}

        except Exception as e:
            logger.error(f"Neural forward failed: {e}")
            return {"gene_scores": {}, "embeddings": None}

    def _symbolic_forward(
        self,
        individual_data: Any,
    ) -> Dict[str, Any]:
        """
        Run symbolic (rule engine) forward pass.

        Returns dict with:
        - gene_scores: Dict[str, float]
        - fired_rules: List[FiredRule]
        """
        if self.rule_engine is None:
            logger.warning("No rule engine provided, using dummy scores")
            return {"gene_scores": {}, "fired_rules": []}

        try:
            # Evaluate rules
            fired_rules = self.rule_engine.evaluate(individual_data)

            # Convert fired rules to gene scores
            gene_scores = {}
            for fired_rule in fired_rules:
                gene = (
                    fired_rule.bindings.get("G") or
                    fired_rule.bindings.get("gene") or
                    fired_rule.evidence.get("gene")
                )
                if gene:
                    confidence = fired_rule.confidence
                    if self.config.min_rule_confidence <= confidence:
                        # Accumulate scores for genes with multiple rules
                        current = gene_scores.get(gene, 0.0)
                        if self.config.use_rule_confidence:
                            gene_scores[gene] = current + confidence
                        else:
                            gene_scores[gene] = current + 1.0

            return {"gene_scores": gene_scores, "fired_rules": fired_rules}

        except Exception as e:
            logger.error(f"Symbolic forward failed: {e}")
            return {"gene_scores": {}, "fired_rules": []}

    def _combine_outputs(
        self,
        neural_output: Dict[str, Any],
        symbolic_output: Dict[str, Any],
        individual_data: Any,
    ) -> NeuroSymbolicOutput:
        """
        Combine neural and symbolic outputs.
        """
        neural_scores = neural_output.get("gene_scores", {})
        symbolic_scores = symbolic_output.get("gene_scores", {})
        fired_rules = symbolic_output.get("fired_rules", [])

        # Combine gene scores
        combined_predictions = combine_gene_scores(
            neural_gene_scores=neural_scores,
            symbolic_gene_scores=symbolic_scores,
            method=self.config.combination_method,
            neural_weight=self.config.neural_weight,
        )

        # Normalize if requested
        if self.config.normalize_outputs and combined_predictions:
            max_score = max(combined_predictions.values())
            if max_score > 0:
                combined_predictions = {
                    g: s / max_score for g, s in combined_predictions.items()
                }

        # Calculate confidence
        confidence = self._calculate_confidence(
            neural_scores=neural_scores,
            symbolic_scores=symbolic_scores,
            fired_rules=fired_rules,
        )

        # Generate explanation
        explanation = self._generate_explanation(
            neural_scores=neural_scores,
            symbolic_scores=symbolic_scores,
            combined=combined_predictions,
            fired_rules=fired_rules,
            individual_data=individual_data,
        )

        return NeuroSymbolicOutput(
            predictions=combined_predictions,
            neural_contribution=neural_scores,
            symbolic_contribution=symbolic_scores,
            fired_rules=fired_rules,
            explanation=explanation,
            neural_embeddings=neural_output.get("embeddings"),
            confidence=confidence,
            combination_weights={
                "neural": self.config.neural_weight,
                "symbolic": self.config.symbolic_weight,
            },
            metadata={
                "combination_method": self.config.combination_method,
                "individual_id": getattr(individual_data, "sample_id", "unknown"),
            },
        )

    def _calculate_confidence(
        self,
        neural_scores: Dict[str, float],
        symbolic_scores: Dict[str, float],
        fired_rules: List[Any],
    ) -> float:
        """Calculate overall confidence of predictions."""
        confidences = []

        # Neural confidence (based on score magnitude)
        if neural_scores:
            neural_conf = np.mean(list(neural_scores.values()))
            confidences.append(neural_conf)

        # Symbolic confidence (based on rule confidences)
        if fired_rules:
            rule_confs = [fr.confidence for fr in fired_rules]
            symbolic_conf = np.mean(rule_confs)
            confidences.append(symbolic_conf)

        if confidences:
            return float(np.mean(confidences))
        return 0.0

    def _generate_explanation(
        self,
        neural_scores: Dict[str, float],
        symbolic_scores: Dict[str, float],
        combined: Dict[str, float],
        fired_rules: List[Any],
        individual_data: Any,
    ) -> str:
        """Generate human-readable explanation of predictions."""
        parts = []

        sample_id = getattr(individual_data, "sample_id", "Individual")
        parts.append(f"=== Neurosymbolic Analysis for {sample_id} ===\n")

        # Summary
        parts.append(f"\nCombination method: {self.config.combination_method}")
        parts.append(f"Neural weight: {self.config.neural_weight:.1%}")
        parts.append(f"Symbolic weight: {self.config.symbolic_weight:.1%}")

        # Top genes from combined predictions
        if combined:
            top_genes = sorted(combined.items(), key=lambda x: -x[1])[:10]
            parts.append(f"\n\nTop genes (combined):")
            for gene, score in top_genes:
                neural = neural_scores.get(gene, 0)
                symbolic = symbolic_scores.get(gene, 0)
                parts.append(
                    f"\n  {gene}: {score:.3f} "
                    f"(neural={neural:.3f}, symbolic={symbolic:.3f})"
                )

        # Fired rules summary
        if fired_rules:
            parts.append(f"\n\nRules fired: {len(fired_rules)}")

            # Group by rule type
            rule_types = {}
            for fr in fired_rules:
                rule_type = fr.rule.conclusion.type
                if rule_type not in rule_types:
                    rule_types[rule_type] = []
                rule_types[rule_type].append(fr)

            for rule_type, rules in rule_types.items():
                parts.append(f"\n  {rule_type}: {len(rules)} rules")

        return "".join(parts)

    def train(self, mode: bool = True):
        """Set training mode."""
        self._training = mode
        if self.neural_model is not None and hasattr(self.neural_model, "train"):
            self.neural_model.train(mode)
        self.combiner.train(mode)

    def eval(self):
        """Set evaluation mode."""
        self._training = False
        if self.neural_model is not None and hasattr(self.neural_model, "eval"):
            self.neural_model.eval()
        self.combiner.eval()

    def parameters(self):
        """Get all learnable parameters."""
        params = []
        if self.neural_model is not None and hasattr(self.neural_model, "parameters"):
            params.extend(self.neural_model.parameters())
        params.extend(self.combiner.parameters())
        return params

    def get_gene_embedding(
        self,
        gene_id: str,
        graph_data: Any,
    ) -> Optional[np.ndarray]:
        """Get embedding for a specific gene."""
        if self.neural_model is None:
            return None

        output = self._neural_forward(graph_data=graph_data)
        embeddings = output.get("embeddings")

        if embeddings is None or "gene" not in embeddings:
            return None

        gene_emb = embeddings["gene"]
        if hasattr(graph_data, "gene_ids"):
            try:
                idx = graph_data.gene_ids.index(gene_id)
                if HAS_TORCH and isinstance(gene_emb, torch.Tensor):
                    return gene_emb[idx].detach().cpu().numpy()
                return gene_emb[idx]
            except (ValueError, IndexError):
                pass

        return None


def create_neurosymbolic_model(
    neural_model: Optional[Any] = None,
    rule_engine: Optional[Any] = None,
    combination_method: str = "weighted_sum",
    neural_weight: float = 0.6,
    **kwargs,
) -> NeuroSymbolicModel:
    """
    Factory function to create a neurosymbolic model.

    Args:
        neural_model: GNN model (optional)
        rule_engine: Rule engine (optional)
        combination_method: How to combine predictions
        neural_weight: Weight for neural component
        **kwargs: Additional config options

    Returns:
        Configured NeuroSymbolicModel
    """
    config = NeuroSymbolicConfig(
        combination_method=combination_method,
        neural_weight=neural_weight,
        symbolic_weight=1 - neural_weight,
        **kwargs,
    )
    return NeuroSymbolicModel(
        neural_model=neural_model,
        rule_engine=rule_engine,
        config=config,
    )
