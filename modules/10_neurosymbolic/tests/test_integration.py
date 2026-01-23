"""Tests for integration module - NeuroSymbolicModel."""

import sys
from pathlib import Path

# Add module to path
_module_dir = Path(__file__).parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from integration import (
    NeuroSymbolicConfig,
    NeuroSymbolicOutput,
    NeuroSymbolicModel,
    create_neurosymbolic_model,
)

# Check for PyTorch availability
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ================= Mock Classes =================

@dataclass
class MockIndividualData:
    """Mock individual data for testing."""
    sample_id: str = "TEST_001"
    variants: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MockGraphData:
    """Mock graph data for testing."""
    node_features: Dict[str, Any] = field(default_factory=dict)
    edge_index: Any = None
    edge_type: Any = None
    gene_ids: List[str] = field(default_factory=list)
    bio_priors: Optional[Dict[str, Any]] = None


class MockFiredRule:
    """Mock fired rule for testing."""
    def __init__(self, gene: str, confidence: float, rule_type: str = "gene_highlight"):
        self.bindings = {"G": gene}
        self.evidence = {"gene": gene}
        self.confidence = confidence

        # Mock rule with conclusion
        self.rule = MockRule(rule_type)

    def to_dict(self):
        return {
            "gene": self.bindings.get("G"),
            "confidence": self.confidence,
        }


class MockRule:
    """Mock rule for testing."""
    def __init__(self, rule_type: str):
        self.conclusion = MockConclusion(rule_type)


class MockConclusion:
    """Mock conclusion for testing."""
    def __init__(self, rule_type: str):
        self.type = rule_type


class MockRuleEngine:
    """Mock rule engine for testing."""
    def __init__(self, fired_rules: Optional[List[MockFiredRule]] = None):
        self._fired_rules = fired_rules or []

    def evaluate(self, individual_data: Any) -> List[MockFiredRule]:
        return self._fired_rules


class MockGNNOutput:
    """Mock GNN output for testing."""
    def __init__(self, gene_scores: Dict[str, float], embeddings: Optional[Dict] = None):
        self.node_embeddings = embeddings or {}
        self.gene_logits = None
        self._gene_scores = gene_scores


class MockGNNModel:
    """Mock GNN model for testing."""
    def __init__(self, gene_scores: Dict[str, float]):
        self._gene_scores = gene_scores
        self._training = False

    def __call__(self, **kwargs):
        # Return mock output with embeddings
        gene_ids = kwargs.get("gene_ids", list(self._gene_scores.keys()))
        num_genes = len(gene_ids)

        if HAS_TORCH:
            embeddings = {"gene": torch.randn(num_genes, 64)}
        else:
            embeddings = {"gene": np.random.randn(num_genes, 64)}

        return MockGNNOutput(self._gene_scores, embeddings)

    def train(self, mode: bool = True):
        self._training = mode

    def eval(self):
        self._training = False


# ================= Tests =================

class TestNeuroSymbolicConfig:
    """Tests for NeuroSymbolicConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NeuroSymbolicConfig()

        assert config.combination_method == "weighted_sum"
        assert config.neural_weight == 0.6
        assert config.symbolic_weight == 0.4
        assert config.use_neural_embeddings is True
        assert config.neural_output_dim == 128
        assert config.use_rule_confidence is True
        assert config.min_rule_confidence == 0.5
        assert config.normalize_outputs is True
        assert config.temperature == 1.0
        assert config.learnable_combination is False
        assert config.dropout == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = NeuroSymbolicConfig(
            combination_method="attention",
            neural_weight=0.7,
            symbolic_weight=0.3,
            min_rule_confidence=0.6,
        )

        assert config.combination_method == "attention"
        assert config.neural_weight == 0.7
        assert config.symbolic_weight == 0.3
        assert config.min_rule_confidence == 0.6

    def test_invalid_combination_method(self):
        """Test validation rejects invalid combination method."""
        with pytest.raises(ValueError, match="combination_method must be one of"):
            NeuroSymbolicConfig(combination_method="invalid")

    def test_valid_combination_methods(self):
        """Test all valid combination methods are accepted."""
        valid_methods = ["weighted_sum", "attention", "gating", "learned", "rule_guided", "max"]
        for method in valid_methods:
            config = NeuroSymbolicConfig(combination_method=method)
            assert config.combination_method == method


class TestNeuroSymbolicOutput:
    """Tests for NeuroSymbolicOutput dataclass."""

    def test_default_output(self):
        """Test default output values."""
        output = NeuroSymbolicOutput()

        assert output.predictions == {}
        assert output.neural_contribution == {}
        assert output.symbolic_contribution == {}
        assert output.fired_rules == []
        assert output.explanation == ""
        assert output.neural_embeddings is None
        assert output.confidence == 0.0
        assert output.combination_weights == {}
        assert output.metadata == {}

    def test_custom_output(self):
        """Test custom output values."""
        predictions = {"GENE1": 0.8, "GENE2": 0.6}
        neural = {"GENE1": 0.9, "GENE2": 0.5}
        symbolic = {"GENE1": 0.7, "GENE2": 0.7}
        fired = [MockFiredRule("GENE1", 0.7)]

        output = NeuroSymbolicOutput(
            predictions=predictions,
            neural_contribution=neural,
            symbolic_contribution=symbolic,
            fired_rules=fired,
            explanation="Test explanation",
            confidence=0.75,
        )

        assert output.predictions == predictions
        assert output.neural_contribution == neural
        assert output.symbolic_contribution == symbolic
        assert len(output.fired_rules) == 1
        assert output.explanation == "Test explanation"
        assert output.confidence == 0.75

    def test_to_dict(self):
        """Test serialization to dictionary."""
        output = NeuroSymbolicOutput(
            predictions={"GENE1": 0.8},
            neural_contribution={"GENE1": 0.9},
            symbolic_contribution={"GENE1": 0.7},
            fired_rules=[MockFiredRule("GENE1", 0.7)],
            confidence=0.8,
        )

        result = output.to_dict()

        assert "predictions" in result
        assert "neural_contribution" in result
        assert "symbolic_contribution" in result
        assert "fired_rules" in result
        assert "confidence" in result
        assert "timestamp" in result

    def test_top_genes(self):
        """Test top_genes property."""
        predictions = {
            "GENE1": 0.5,
            "GENE2": 0.9,
            "GENE3": 0.7,
        }
        output = NeuroSymbolicOutput(predictions=predictions)

        top = output.top_genes

        assert top[0] == ("GENE2", 0.9)
        assert top[1] == ("GENE3", 0.7)
        assert top[2] == ("GENE1", 0.5)

    def test_rules_fired_count(self):
        """Test rules_fired_count property."""
        fired = [
            MockFiredRule("GENE1", 0.7),
            MockFiredRule("GENE2", 0.8),
            MockFiredRule("GENE3", 0.6),
        ]
        output = NeuroSymbolicOutput(fired_rules=fired)

        assert output.rules_fired_count == 3


class TestNeuroSymbolicModel:
    """Tests for NeuroSymbolicModel class."""

    def test_initialization_no_components(self):
        """Test initialization without neural or symbolic components."""
        model = NeuroSymbolicModel()

        assert model.neural_model is None
        assert model.rule_engine is None
        assert model.config is not None
        assert model.combiner is not None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = NeuroSymbolicConfig(
            combination_method="attention",
            neural_weight=0.7,
        )
        model = NeuroSymbolicModel(config=config)

        assert model.config.combination_method == "attention"
        assert model.config.neural_weight == 0.7

    def test_initialization_with_components(self):
        """Test initialization with neural and symbolic components."""
        rule_engine = MockRuleEngine()

        model = NeuroSymbolicModel(
            rule_engine=rule_engine,
        )

        assert model.rule_engine is rule_engine

    def test_forward_no_components(self):
        """Test forward pass with no components."""
        model = NeuroSymbolicModel()
        individual = MockIndividualData()

        output = model.forward(individual)

        assert isinstance(output, NeuroSymbolicOutput)
        assert output.predictions == {}
        assert output.neural_contribution == {}
        assert output.symbolic_contribution == {}

    def test_forward_with_rule_engine(self):
        """Test forward pass with rule engine."""
        fired_rules = [
            MockFiredRule("GENE1", 0.8, "constrained_lof"),
            MockFiredRule("GENE2", 0.7, "pathway_convergence"),
        ]
        rule_engine = MockRuleEngine(fired_rules)

        model = NeuroSymbolicModel(rule_engine=rule_engine)
        individual = MockIndividualData()

        output = model.forward(individual)

        assert len(output.fired_rules) == 2
        assert "GENE1" in output.symbolic_contribution
        assert "GENE2" in output.symbolic_contribution
        assert output.symbolic_contribution["GENE1"] == 0.8
        assert output.symbolic_contribution["GENE2"] == 0.7

    def test_forward_rule_confidence_filtering(self):
        """Test that low confidence rules are filtered."""
        fired_rules = [
            MockFiredRule("GENE1", 0.8),
            MockFiredRule("GENE2", 0.3),  # Below default threshold of 0.5
        ]
        rule_engine = MockRuleEngine(fired_rules)

        model = NeuroSymbolicModel(rule_engine=rule_engine)
        individual = MockIndividualData()

        output = model.forward(individual)

        # GENE2 should be filtered due to low confidence
        assert "GENE1" in output.symbolic_contribution
        assert "GENE2" not in output.symbolic_contribution

    def test_forward_normalization(self):
        """Test output normalization."""
        fired_rules = [
            MockFiredRule("GENE1", 0.8),
            MockFiredRule("GENE2", 0.6),
        ]
        rule_engine = MockRuleEngine(fired_rules)

        config = NeuroSymbolicConfig(normalize_outputs=True)
        model = NeuroSymbolicModel(rule_engine=rule_engine, config=config)
        individual = MockIndividualData()

        output = model.forward(individual)

        # Max score should be 1.0 after normalization
        if output.predictions:
            max_score = max(output.predictions.values())
            assert max_score <= 1.0

    def test_explanation_generation(self):
        """Test explanation is generated."""
        fired_rules = [MockFiredRule("GENE1", 0.8)]
        rule_engine = MockRuleEngine(fired_rules)

        model = NeuroSymbolicModel(rule_engine=rule_engine)
        individual = MockIndividualData(sample_id="SAMPLE_001")

        output = model.forward(individual)

        assert "SAMPLE_001" in output.explanation
        assert "Neurosymbolic Analysis" in output.explanation
        assert "weighted_sum" in output.explanation

    def test_train_mode(self):
        """Test train mode setting."""
        model = NeuroSymbolicModel()

        model.train(True)
        assert model._training is True

        model.eval()
        assert model._training is False

    def test_parameters(self):
        """Test parameters method."""
        model = NeuroSymbolicModel()
        params = model.parameters()

        # Should return empty or combiner params
        assert isinstance(params, list)


class TestNeuroSymbolicModelCombination:
    """Tests for NeuroSymbolicModel combination methods."""

    def test_weighted_sum_combination(self):
        """Test weighted sum combination method."""
        config = NeuroSymbolicConfig(
            combination_method="weighted_sum",
            neural_weight=0.6,
            symbolic_weight=0.4,
            normalize_outputs=False,
        )

        # Only symbolic (no neural model provided)
        fired_rules = [MockFiredRule("GENE1", 1.0)]
        rule_engine = MockRuleEngine(fired_rules)

        model = NeuroSymbolicModel(rule_engine=rule_engine, config=config)
        individual = MockIndividualData()

        output = model.forward(individual)

        # With only symbolic, contribution should be scaled by symbolic weight
        # But since neural is 0, result is 0.4 * 1.0 = 0.4
        assert "GENE1" in output.predictions
        # The combined score for GENE1 = 0.6 * 0 + 0.4 * 1.0 = 0.4
        assert output.predictions["GENE1"] == pytest.approx(0.4)

    def test_max_combination(self):
        """Test max combination method."""
        config = NeuroSymbolicConfig(
            combination_method="max",
            normalize_outputs=False,
        )

        fired_rules = [MockFiredRule("GENE1", 0.9)]
        rule_engine = MockRuleEngine(fired_rules)

        model = NeuroSymbolicModel(rule_engine=rule_engine, config=config)
        individual = MockIndividualData()

        output = model.forward(individual)

        # Max of (0, 0.9) = 0.9
        assert output.predictions["GENE1"] == 0.9

    def test_multiple_rules_same_gene(self):
        """Test accumulation of scores for same gene."""
        fired_rules = [
            MockFiredRule("GENE1", 0.6),  # Above min_rule_confidence
            MockFiredRule("GENE1", 0.5),  # Same gene, also above threshold
        ]
        rule_engine = MockRuleEngine(fired_rules)

        config = NeuroSymbolicConfig(normalize_outputs=False, min_rule_confidence=0.5)
        model = NeuroSymbolicModel(rule_engine=rule_engine, config=config)
        individual = MockIndividualData()

        output = model.forward(individual)

        # Scores should accumulate: 0.6 + 0.5 = 1.1
        assert output.symbolic_contribution["GENE1"] == pytest.approx(1.1)


class TestCreateNeuroSymbolicModel:
    """Tests for create_neurosymbolic_model factory function."""

    def test_basic_creation(self):
        """Test basic model creation."""
        model = create_neurosymbolic_model()

        assert isinstance(model, NeuroSymbolicModel)
        assert model.config.combination_method == "weighted_sum"
        assert model.config.neural_weight == 0.6

    def test_custom_parameters(self):
        """Test creation with custom parameters."""
        model = create_neurosymbolic_model(
            combination_method="attention",
            neural_weight=0.8,
        )

        assert model.config.combination_method == "attention"
        assert model.config.neural_weight == 0.8
        assert model.config.symbolic_weight == pytest.approx(0.2)

    def test_with_components(self):
        """Test creation with components."""
        rule_engine = MockRuleEngine()

        model = create_neurosymbolic_model(
            rule_engine=rule_engine,
        )

        assert model.rule_engine is rule_engine


class TestNeuroSymbolicModelConfidence:
    """Tests for confidence calculation."""

    def test_confidence_with_rules(self):
        """Test confidence calculation with rules."""
        fired_rules = [
            MockFiredRule("GENE1", 0.8),
            MockFiredRule("GENE2", 0.6),
        ]
        rule_engine = MockRuleEngine(fired_rules)

        model = NeuroSymbolicModel(rule_engine=rule_engine)
        individual = MockIndividualData()

        output = model.forward(individual)

        # Confidence should be mean of rule confidences
        expected_conf = (0.8 + 0.6) / 2
        assert output.confidence == pytest.approx(expected_conf, rel=0.1)

    def test_confidence_no_rules(self):
        """Test confidence with no rules."""
        model = NeuroSymbolicModel()
        individual = MockIndividualData()

        output = model.forward(individual)

        assert output.confidence == 0.0


class TestNeuroSymbolicModelMetadata:
    """Tests for metadata handling."""

    def test_metadata_includes_method(self):
        """Test metadata includes combination method."""
        fired_rules = [MockFiredRule("GENE1", 0.8)]
        rule_engine = MockRuleEngine(fired_rules)

        model = NeuroSymbolicModel(rule_engine=rule_engine)
        individual = MockIndividualData(sample_id="TEST_SAMPLE")

        output = model.forward(individual)

        assert output.metadata["combination_method"] == "weighted_sum"
        assert output.metadata["individual_id"] == "TEST_SAMPLE"

    def test_combination_weights_recorded(self):
        """Test combination weights are recorded."""
        config = NeuroSymbolicConfig(
            neural_weight=0.7,
            symbolic_weight=0.3,
        )
        model = NeuroSymbolicModel(config=config)
        individual = MockIndividualData()

        output = model.forward(individual)

        assert output.combination_weights["neural"] == 0.7
        assert output.combination_weights["symbolic"] == 0.3


class TestNeuroSymbolicModelEdgeCases:
    """Tests for edge cases."""

    def test_empty_variants(self):
        """Test with empty variants."""
        model = NeuroSymbolicModel()
        individual = MockIndividualData(variants=[])

        output = model.forward(individual)

        assert isinstance(output, NeuroSymbolicOutput)

    def test_rule_engine_exception_handling(self):
        """Test handling of rule engine exceptions."""
        class FailingRuleEngine:
            def evaluate(self, individual):
                raise RuntimeError("Evaluation failed")

        model = NeuroSymbolicModel(rule_engine=FailingRuleEngine())
        individual = MockIndividualData()

        # Should not raise, but log error
        output = model.forward(individual)

        assert output.fired_rules == []
        assert output.symbolic_contribution == {}


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestNeuroSymbolicModelTorch:
    """Tests requiring PyTorch."""

    def test_with_mock_gnn(self):
        """Test with mock GNN model."""
        gene_scores = {"GENE1": 0.8, "GENE2": 0.6}
        gnn_model = MockGNNModel(gene_scores)

        model = NeuroSymbolicModel(neural_model=gnn_model)
        individual = MockIndividualData()

        # Need graph data for GNN
        graph_data = MockGraphData(
            gene_ids=["GENE1", "GENE2"],
        )

        output = model.forward(individual, graph_data=graph_data)

        assert isinstance(output, NeuroSymbolicOutput)

    def test_train_eval_with_gnn(self):
        """Test train/eval mode propagates to GNN."""
        gnn_model = MockGNNModel({})

        model = NeuroSymbolicModel(neural_model=gnn_model)

        model.train(True)
        assert gnn_model._training is True

        model.eval()
        assert gnn_model._training is False


class TestNeuroSymbolicIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_symbolic_only(self):
        """Test full pipeline with symbolic component only."""
        fired_rules = [
            MockFiredRule("CHD8", 0.9, "chd8_cascade"),
            MockFiredRule("SYNGAP1", 0.8, "synaptic_disruption"),
            MockFiredRule("SCN2A", 0.7, "constrained_lof"),
        ]
        rule_engine = MockRuleEngine(fired_rules)

        config = NeuroSymbolicConfig(
            combination_method="weighted_sum",
            neural_weight=0.5,
            symbolic_weight=0.5,
            normalize_outputs=True,
        )

        model = NeuroSymbolicModel(rule_engine=rule_engine, config=config)
        individual = MockIndividualData(sample_id="AUTISM_PATIENT_001")

        output = model.forward(individual)

        # Check outputs
        assert len(output.fired_rules) == 3
        assert "CHD8" in output.predictions
        assert "SYNGAP1" in output.predictions
        assert "SCN2A" in output.predictions

        # Check normalization
        if output.predictions:
            max_score = max(output.predictions.values())
            assert max_score <= 1.0

        # Check explanation
        assert "AUTISM_PATIENT_001" in output.explanation
        assert "Rules fired: 3" in output.explanation

    def test_top_genes_ordering(self):
        """Test that top genes are properly ordered."""
        fired_rules = [
            MockFiredRule("GENE_A", 0.9),
            MockFiredRule("GENE_B", 0.5),
            MockFiredRule("GENE_C", 0.7),
        ]
        rule_engine = MockRuleEngine(fired_rules)

        config = NeuroSymbolicConfig(normalize_outputs=False)
        model = NeuroSymbolicModel(rule_engine=rule_engine, config=config)
        individual = MockIndividualData()

        output = model.forward(individual)
        top = output.top_genes

        # Should be sorted by score descending
        scores = [score for _, score in top]
        assert scores == sorted(scores, reverse=True)
