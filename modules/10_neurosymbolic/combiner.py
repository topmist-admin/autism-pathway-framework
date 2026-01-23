"""
Learned combination strategies for neural and symbolic predictions.

This module provides various methods for combining neural network outputs
with symbolic rule-based inferences.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import logging
import math

import numpy as np

# Optional PyTorch support
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None

logger = logging.getLogger(__name__)


class CombinationMethod(Enum):
    """Available methods for combining neural and symbolic predictions."""

    WEIGHTED_SUM = "weighted_sum"
    ATTENTION = "attention"
    GATING = "gating"
    LEARNED = "learned"
    RULE_GUIDED = "rule_guided"
    MAX = "max"
    PRODUCT = "product"


@dataclass
class CombinerConfig:
    """Configuration for prediction combiner."""

    method: CombinationMethod = CombinationMethod.WEIGHTED_SUM
    neural_weight: float = 0.6
    symbolic_weight: float = 0.4
    temperature: float = 1.0
    hidden_dim: int = 64
    dropout: float = 0.1
    use_rule_confidence: bool = True
    normalize_outputs: bool = True
    learnable_weights: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.neural_weight <= 1:
            raise ValueError("neural_weight must be between 0 and 1")
        if not 0 <= self.symbolic_weight <= 1:
            raise ValueError("symbolic_weight must be between 0 and 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


class LearnedCombiner:
    """
    Combines neural and symbolic predictions using various strategies.

    This is the main combiner class that supports both PyTorch-based
    learned combinations and numpy-based fixed combinations.
    """

    def __init__(
        self,
        config: Optional[CombinerConfig] = None,
        neural_dim: int = 128,
        symbolic_dim: int = 64,
        output_dim: int = 64,
    ):
        """
        Initialize the combiner.

        Args:
            config: Combiner configuration
            neural_dim: Dimension of neural features
            symbolic_dim: Dimension of symbolic features
            output_dim: Dimension of output features
        """
        self.config = config or CombinerConfig()
        self.neural_dim = neural_dim
        self.symbolic_dim = symbolic_dim
        self.output_dim = output_dim

        # Initialize combination layers if using learned methods
        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize neural network layers for learned combination."""
        if not HAS_TORCH:
            self._layers = None
            return

        method = self.config.method

        if method == CombinationMethod.ATTENTION:
            self._layers = AttentionCombiner(
                neural_dim=self.neural_dim,
                symbolic_dim=self.symbolic_dim,
                output_dim=self.output_dim,
                num_heads=4,
                dropout=self.config.dropout,
            )

        elif method == CombinationMethod.GATING:
            self._layers = GatingCombiner(
                neural_dim=self.neural_dim,
                symbolic_dim=self.symbolic_dim,
                output_dim=self.output_dim,
                dropout=self.config.dropout,
            )

        elif method == CombinationMethod.LEARNED:
            self._layers = MLPCombiner(
                neural_dim=self.neural_dim,
                symbolic_dim=self.symbolic_dim,
                output_dim=self.output_dim,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
            )

        elif method == CombinationMethod.RULE_GUIDED:
            self._layers = RuleGuidedCombiner(
                neural_dim=self.neural_dim,
                symbolic_dim=self.symbolic_dim,
                output_dim=self.output_dim,
                dropout=self.config.dropout,
            )

        else:
            # Fixed combination methods don't need layers
            self._layers = None

    def combine(
        self,
        neural_scores: Union[np.ndarray, "torch.Tensor"],
        symbolic_scores: Union[np.ndarray, "torch.Tensor"],
        features: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        rule_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    ) -> Tuple[Union[np.ndarray, "torch.Tensor"], Dict[str, float]]:
        """
        Combine neural and symbolic predictions.

        Args:
            neural_scores: Neural network predictions [batch, dim]
            symbolic_scores: Symbolic rule scores [batch, dim]
            features: Optional features for attention [batch, feat_dim]
            rule_mask: Optional mask indicating which rules fired [batch, num_rules]

        Returns:
            Tuple of (combined scores, metadata dict)
        """
        method = self.config.method

        if method == CombinationMethod.WEIGHTED_SUM:
            return self._weighted_sum(neural_scores, symbolic_scores)

        elif method == CombinationMethod.MAX:
            return self._max_combination(neural_scores, symbolic_scores)

        elif method == CombinationMethod.PRODUCT:
            return self._product_combination(neural_scores, symbolic_scores)

        elif method in (
            CombinationMethod.ATTENTION,
            CombinationMethod.GATING,
            CombinationMethod.LEARNED,
            CombinationMethod.RULE_GUIDED,
        ):
            if self._layers is None:
                # Fallback to weighted sum if PyTorch not available
                logger.warning(
                    f"PyTorch not available for {method.value}, "
                    "falling back to weighted_sum"
                )
                return self._weighted_sum(neural_scores, symbolic_scores)

            return self._learned_combination(
                neural_scores, symbolic_scores, features, rule_mask
            )

        else:
            raise ValueError(f"Unknown combination method: {method}")

    def _weighted_sum(
        self,
        neural_scores: Union[np.ndarray, "torch.Tensor"],
        symbolic_scores: Union[np.ndarray, "torch.Tensor"],
    ) -> Tuple[Union[np.ndarray, "torch.Tensor"], Dict[str, float]]:
        """Fixed-weight linear combination."""
        alpha = self.config.neural_weight
        beta = self.config.symbolic_weight

        # Normalize weights
        total = alpha + beta
        alpha = alpha / total
        beta = beta / total

        combined = alpha * neural_scores + beta * symbolic_scores

        metadata = {
            "neural_weight": alpha,
            "symbolic_weight": beta,
            "method": "weighted_sum",
        }

        return combined, metadata

    def _max_combination(
        self,
        neural_scores: Union[np.ndarray, "torch.Tensor"],
        symbolic_scores: Union[np.ndarray, "torch.Tensor"],
    ) -> Tuple[Union[np.ndarray, "torch.Tensor"], Dict[str, float]]:
        """Element-wise maximum."""
        if HAS_TORCH and isinstance(neural_scores, torch.Tensor):
            combined = torch.maximum(neural_scores, symbolic_scores)
        else:
            combined = np.maximum(neural_scores, symbolic_scores)

        metadata = {"method": "max"}
        return combined, metadata

    def _product_combination(
        self,
        neural_scores: Union[np.ndarray, "torch.Tensor"],
        symbolic_scores: Union[np.ndarray, "torch.Tensor"],
    ) -> Tuple[Union[np.ndarray, "torch.Tensor"], Dict[str, float]]:
        """Geometric mean combination."""
        eps = 1e-8

        if HAS_TORCH and isinstance(neural_scores, torch.Tensor):
            combined = torch.sqrt(
                (neural_scores + eps) * (symbolic_scores + eps)
            )
        else:
            combined = np.sqrt(
                (neural_scores + eps) * (symbolic_scores + eps)
            )

        metadata = {"method": "product"}
        return combined, metadata

    def _learned_combination(
        self,
        neural_scores: Union[np.ndarray, "torch.Tensor"],
        symbolic_scores: Union[np.ndarray, "torch.Tensor"],
        features: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        rule_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        """Use learned layers for combination."""
        # Convert to tensors if needed
        if not isinstance(neural_scores, torch.Tensor):
            neural_scores = torch.tensor(neural_scores, dtype=torch.float32)
        if not isinstance(symbolic_scores, torch.Tensor):
            symbolic_scores = torch.tensor(symbolic_scores, dtype=torch.float32)

        if features is not None and not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        if rule_mask is not None and not isinstance(rule_mask, torch.Tensor):
            rule_mask = torch.tensor(rule_mask, dtype=torch.float32)

        # Forward through learned layers
        combined, attention_weights = self._layers(
            neural_scores, symbolic_scores, features, rule_mask
        )

        metadata = {
            "method": self.config.method.value,
            "attention_weights": attention_weights,
        }

        return combined, metadata

    def parameters(self):
        """Get learnable parameters (for training)."""
        if self._layers is not None and HAS_TORCH:
            return self._layers.parameters()
        return []

    def train(self, mode: bool = True):
        """Set training mode."""
        if self._layers is not None and HAS_TORCH:
            self._layers.train(mode)

    def eval(self):
        """Set evaluation mode."""
        if self._layers is not None and HAS_TORCH:
            self._layers.eval()


# ============== PyTorch Combiner Modules ==============

if HAS_TORCH:

    class AttentionCombiner(nn.Module):
        """Attention-based combination of neural and symbolic predictions."""

        def __init__(
            self,
            neural_dim: int,
            symbolic_dim: int,
            output_dim: int,
            num_heads: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.neural_dim = neural_dim
            self.symbolic_dim = symbolic_dim
            self.output_dim = output_dim
            self.num_heads = num_heads

            # Project both to common dimension
            self.neural_proj = nn.Linear(neural_dim, output_dim)
            self.symbolic_proj = nn.Linear(symbolic_dim, output_dim)

            # Multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

            # Output projection
            self.output_proj = nn.Linear(output_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(output_dim)

        def forward(
            self,
            neural_scores: torch.Tensor,
            symbolic_scores: torch.Tensor,
            features: Optional[torch.Tensor] = None,
            rule_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # Project to common dimension
            neural = self.neural_proj(neural_scores)
            symbolic = self.symbolic_proj(symbolic_scores)

            # Stack as sequence [batch, 2, dim]
            if neural.dim() == 1:
                neural = neural.unsqueeze(0)
            if symbolic.dim() == 1:
                symbolic = symbolic.unsqueeze(0)

            # Create sequence: neural as query, symbolic as key/value
            query = neural.unsqueeze(1)  # [batch, 1, dim]
            key_value = symbolic.unsqueeze(1)  # [batch, 1, dim]

            # Attention
            attn_output, attn_weights = self.attention(
                query, key_value, key_value
            )

            # Combine with residual
            combined = neural + attn_output.squeeze(1)
            combined = self.layer_norm(combined)
            combined = self.output_proj(combined)

            return combined, attn_weights

    class GatingCombiner(nn.Module):
        """Gating-based combination with learned gate."""

        def __init__(
            self,
            neural_dim: int,
            symbolic_dim: int,
            output_dim: int,
            dropout: float = 0.1,
        ):
            super().__init__()

            # Project to common dimension
            self.neural_proj = nn.Linear(neural_dim, output_dim)
            self.symbolic_proj = nn.Linear(symbolic_dim, output_dim)

            # Gate network
            self.gate = nn.Sequential(
                nn.Linear(neural_dim + symbolic_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid(),
            )

            self.output_proj = nn.Linear(output_dim, output_dim)

        def forward(
            self,
            neural_scores: torch.Tensor,
            symbolic_scores: torch.Tensor,
            features: Optional[torch.Tensor] = None,
            rule_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # Project to common dimension
            neural = self.neural_proj(neural_scores)
            symbolic = self.symbolic_proj(symbolic_scores)

            # Compute gate
            gate_input = torch.cat([neural_scores, symbolic_scores], dim=-1)
            gate = self.gate(gate_input)

            # Gated combination
            combined = gate * neural + (1 - gate) * symbolic
            combined = self.output_proj(combined)

            return combined, gate

    class MLPCombiner(nn.Module):
        """MLP-based learned combination."""

        def __init__(
            self,
            neural_dim: int,
            symbolic_dim: int,
            output_dim: int,
            hidden_dim: int = 64,
            dropout: float = 0.1,
        ):
            super().__init__()

            input_dim = neural_dim + symbolic_dim

            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(
            self,
            neural_scores: torch.Tensor,
            symbolic_scores: torch.Tensor,
            features: Optional[torch.Tensor] = None,
            rule_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, None]:
            # Concatenate inputs
            combined_input = torch.cat([neural_scores, symbolic_scores], dim=-1)

            # MLP forward
            output = self.mlp(combined_input)

            return output, None

    class RuleGuidedCombiner(nn.Module):
        """Rule-guided combination using symbolic rules to guide neural attention."""

        def __init__(
            self,
            neural_dim: int,
            symbolic_dim: int,
            output_dim: int,
            dropout: float = 0.1,
        ):
            super().__init__()

            # Project to common dimension
            self.neural_proj = nn.Linear(neural_dim, output_dim)
            self.symbolic_proj = nn.Linear(symbolic_dim, output_dim)

            # Rule attention
            self.rule_attention = nn.Sequential(
                nn.Linear(symbolic_dim, output_dim),
                nn.Tanh(),
                nn.Linear(output_dim, output_dim),
            )

            # Output combination
            self.combine = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
            )

        def forward(
            self,
            neural_scores: torch.Tensor,
            symbolic_scores: torch.Tensor,
            features: Optional[torch.Tensor] = None,
            rule_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # Project inputs
            neural = self.neural_proj(neural_scores)
            symbolic = self.symbolic_proj(symbolic_scores)

            # Compute rule-based attention weights
            rule_attn = self.rule_attention(symbolic_scores)
            rule_attn = F.softmax(rule_attn, dim=-1)

            # Apply rule attention to neural features
            neural_attended = neural * rule_attn

            # Combine
            combined = torch.cat([neural_attended, symbolic], dim=-1)
            output = self.combine(combined)

            return output, rule_attn


# ============== NumPy-based Combiners (fallback) ==============

class NumpyAttentionCombiner:
    """NumPy-based attention combination (no learning)."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def combine(
        self,
        neural_scores: np.ndarray,
        symbolic_scores: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Simple attention-like weighting based on score magnitudes."""
        # Compute attention weights from score magnitudes
        neural_mag = np.abs(neural_scores).mean()
        symbolic_mag = np.abs(symbolic_scores).mean()

        total = neural_mag + symbolic_mag + 1e-8
        neural_weight = neural_mag / total
        symbolic_weight = symbolic_mag / total

        combined = neural_weight * neural_scores + symbolic_weight * symbolic_scores

        metadata = {
            "neural_weight": float(neural_weight),
            "symbolic_weight": float(symbolic_weight),
        }

        return combined, metadata


def combine_gene_scores(
    neural_gene_scores: Dict[str, float],
    symbolic_gene_scores: Dict[str, float],
    method: str = "weighted_sum",
    neural_weight: float = 0.6,
) -> Dict[str, float]:
    """
    Combine gene-level scores from neural and symbolic sources.

    Args:
        neural_gene_scores: Gene -> score from GNN
        symbolic_gene_scores: Gene -> score from rules
        method: Combination method
        neural_weight: Weight for neural scores

    Returns:
        Combined gene scores
    """
    all_genes = set(neural_gene_scores.keys()) | set(symbolic_gene_scores.keys())
    symbolic_weight = 1 - neural_weight

    combined = {}
    for gene in all_genes:
        neural_score = neural_gene_scores.get(gene, 0.0)
        symbolic_score = symbolic_gene_scores.get(gene, 0.0)

        if method == "weighted_sum":
            combined[gene] = neural_weight * neural_score + symbolic_weight * symbolic_score
        elif method == "max":
            combined[gene] = max(neural_score, symbolic_score)
        elif method == "product":
            combined[gene] = math.sqrt((neural_score + 0.01) * (symbolic_score + 0.01))
        else:
            combined[gene] = (neural_score + symbolic_score) / 2

    return combined


def create_symbolic_score_vector(
    fired_rules: List[Any],
    gene_list: List[str],
    use_confidence: bool = True,
) -> np.ndarray:
    """
    Create a score vector from fired rules.

    Args:
        fired_rules: List of FiredRule objects
        gene_list: Ordered list of genes for vector alignment
        use_confidence: Whether to weight by rule confidence

    Returns:
        Score vector aligned with gene_list
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    scores = np.zeros(len(gene_list))

    for fired_rule in fired_rules:
        # Get gene from bindings
        gene = (
            fired_rule.bindings.get("G") or
            fired_rule.bindings.get("gene") or
            fired_rule.evidence.get("gene")
        )

        if gene and gene in gene_to_idx:
            idx = gene_to_idx[gene]
            if use_confidence:
                scores[idx] += fired_rule.confidence
            else:
                scores[idx] += 1.0

    return scores
