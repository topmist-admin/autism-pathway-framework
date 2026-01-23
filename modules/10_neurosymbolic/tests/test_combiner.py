"""Tests for combiner module - learned combination strategies."""

import sys
from pathlib import Path

# Add module to path
_module_dir = Path(__file__).parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

import pytest
import numpy as np
from combiner import (
    CombinationMethod,
    CombinerConfig,
    LearnedCombiner,
    NumpyAttentionCombiner,
    combine_gene_scores,
    create_symbolic_score_vector,
)

# Check for PyTorch availability
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestCombinationMethod:
    """Tests for CombinationMethod enum."""

    def test_all_methods_defined(self):
        """Test all combination methods are defined."""
        methods = [
            CombinationMethod.WEIGHTED_SUM,
            CombinationMethod.ATTENTION,
            CombinationMethod.GATING,
            CombinationMethod.LEARNED,
            CombinationMethod.RULE_GUIDED,
            CombinationMethod.MAX,
            CombinationMethod.PRODUCT,
        ]
        assert len(methods) == 7

    def test_method_values(self):
        """Test method string values."""
        assert CombinationMethod.WEIGHTED_SUM.value == "weighted_sum"
        assert CombinationMethod.ATTENTION.value == "attention"
        assert CombinationMethod.GATING.value == "gating"
        assert CombinationMethod.LEARNED.value == "learned"
        assert CombinationMethod.RULE_GUIDED.value == "rule_guided"
        assert CombinationMethod.MAX.value == "max"
        assert CombinationMethod.PRODUCT.value == "product"


class TestCombinerConfig:
    """Tests for CombinerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CombinerConfig()
        assert config.method == CombinationMethod.WEIGHTED_SUM
        assert config.neural_weight == 0.6
        assert config.symbolic_weight == 0.4
        assert config.temperature == 1.0
        assert config.hidden_dim == 64
        assert config.dropout == 0.1
        assert config.use_rule_confidence is True
        assert config.normalize_outputs is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CombinerConfig(
            method=CombinationMethod.ATTENTION,
            neural_weight=0.7,
            symbolic_weight=0.3,
            temperature=0.5,
            hidden_dim=128,
        )
        assert config.method == CombinationMethod.ATTENTION
        assert config.neural_weight == 0.7
        assert config.symbolic_weight == 0.3
        assert config.temperature == 0.5
        assert config.hidden_dim == 128

    def test_invalid_neural_weight(self):
        """Test validation rejects invalid neural weight."""
        with pytest.raises(ValueError, match="neural_weight must be between"):
            CombinerConfig(neural_weight=1.5)

    def test_invalid_symbolic_weight(self):
        """Test validation rejects invalid symbolic weight."""
        with pytest.raises(ValueError, match="symbolic_weight must be between"):
            CombinerConfig(symbolic_weight=-0.1)

    def test_invalid_temperature(self):
        """Test validation rejects invalid temperature."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            CombinerConfig(temperature=0.0)


class TestLearnedCombinerWeightedSum:
    """Tests for LearnedCombiner with weighted_sum method."""

    def test_initialization(self):
        """Test combiner initialization."""
        combiner = LearnedCombiner()
        assert combiner.config.method == CombinationMethod.WEIGHTED_SUM
        assert combiner.neural_dim == 128
        assert combiner.symbolic_dim == 64
        assert combiner.output_dim == 64

    def test_custom_dimensions(self):
        """Test combiner with custom dimensions."""
        combiner = LearnedCombiner(
            neural_dim=256,
            symbolic_dim=128,
            output_dim=128,
        )
        assert combiner.neural_dim == 256
        assert combiner.symbolic_dim == 128
        assert combiner.output_dim == 128

    def test_weighted_sum_numpy(self):
        """Test weighted sum combination with numpy arrays."""
        config = CombinerConfig(
            method=CombinationMethod.WEIGHTED_SUM,
            neural_weight=0.6,
            symbolic_weight=0.4,
        )
        combiner = LearnedCombiner(config=config)

        neural = np.array([1.0, 2.0, 3.0])
        symbolic = np.array([0.5, 1.0, 1.5])

        combined, metadata = combiner.combine(neural, symbolic)

        # Expected: 0.6 * neural + 0.4 * symbolic
        expected = 0.6 * neural + 0.4 * symbolic
        np.testing.assert_array_almost_equal(combined, expected)
        assert metadata["method"] == "weighted_sum"
        assert metadata["neural_weight"] == 0.6
        assert metadata["symbolic_weight"] == 0.4

    def test_weighted_sum_2d(self):
        """Test weighted sum with 2D arrays."""
        combiner = LearnedCombiner()

        neural = np.array([[1.0, 2.0], [3.0, 4.0]])
        symbolic = np.array([[0.5, 1.0], [1.5, 2.0]])

        combined, metadata = combiner.combine(neural, symbolic)

        assert combined.shape == (2, 2)
        assert metadata["method"] == "weighted_sum"


class TestLearnedCombinerMax:
    """Tests for LearnedCombiner with max method."""

    def test_max_combination(self):
        """Test max combination."""
        config = CombinerConfig(method=CombinationMethod.MAX)
        combiner = LearnedCombiner(config=config)

        neural = np.array([1.0, 3.0, 2.0])
        symbolic = np.array([2.0, 1.0, 4.0])

        combined, metadata = combiner.combine(neural, symbolic)

        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_equal(combined, expected)
        assert metadata["method"] == "max"


class TestLearnedCombinerProduct:
    """Tests for LearnedCombiner with product method."""

    def test_product_combination(self):
        """Test product (geometric mean) combination."""
        config = CombinerConfig(method=CombinationMethod.PRODUCT)
        combiner = LearnedCombiner(config=config)

        neural = np.array([1.0, 4.0, 9.0])
        symbolic = np.array([1.0, 4.0, 9.0])

        combined, metadata = combiner.combine(neural, symbolic)

        # sqrt((n + eps) * (s + eps)) ~ sqrt(n * s) for large values
        expected = np.sqrt((neural + 1e-8) * (symbolic + 1e-8))
        np.testing.assert_array_almost_equal(combined, expected, decimal=5)
        assert metadata["method"] == "product"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestLearnedCombinerTorch:
    """Tests for LearnedCombiner with PyTorch tensors."""

    def test_weighted_sum_torch(self):
        """Test weighted sum with PyTorch tensors."""
        combiner = LearnedCombiner()

        neural = torch.tensor([1.0, 2.0, 3.0])
        symbolic = torch.tensor([0.5, 1.0, 1.5])

        combined, metadata = combiner.combine(neural, symbolic)

        assert isinstance(combined, torch.Tensor)
        expected = 0.6 * neural + 0.4 * symbolic
        torch.testing.assert_close(combined, expected)

    def test_attention_combiner_init(self):
        """Test attention combiner initialization."""
        config = CombinerConfig(method=CombinationMethod.ATTENTION)
        combiner = LearnedCombiner(
            config=config,
            neural_dim=64,
            symbolic_dim=64,
            output_dim=64,
        )

        assert combiner._layers is not None

    def test_gating_combiner_init(self):
        """Test gating combiner initialization."""
        config = CombinerConfig(method=CombinationMethod.GATING)
        combiner = LearnedCombiner(
            config=config,
            neural_dim=64,
            symbolic_dim=64,
            output_dim=64,
        )

        assert combiner._layers is not None

    def test_learned_combiner_init(self):
        """Test learned (MLP) combiner initialization."""
        config = CombinerConfig(method=CombinationMethod.LEARNED)
        combiner = LearnedCombiner(
            config=config,
            neural_dim=64,
            symbolic_dim=64,
            output_dim=64,
        )

        assert combiner._layers is not None

    def test_attention_forward(self):
        """Test attention combiner forward pass."""
        config = CombinerConfig(method=CombinationMethod.ATTENTION)
        combiner = LearnedCombiner(
            config=config,
            neural_dim=64,
            symbolic_dim=64,
            output_dim=64,
        )

        neural = torch.randn(2, 64)
        symbolic = torch.randn(2, 64)

        combined, metadata = combiner.combine(neural, symbolic)

        assert combined.shape == (2, 64)
        assert "attention_weights" in metadata

    def test_gating_forward(self):
        """Test gating combiner forward pass."""
        config = CombinerConfig(method=CombinationMethod.GATING)
        combiner = LearnedCombiner(
            config=config,
            neural_dim=64,
            symbolic_dim=64,
            output_dim=64,
        )

        neural = torch.randn(2, 64)
        symbolic = torch.randn(2, 64)

        combined, metadata = combiner.combine(neural, symbolic)

        assert combined.shape == (2, 64)
        assert "attention_weights" in metadata

    def test_mlp_forward(self):
        """Test MLP combiner forward pass."""
        config = CombinerConfig(method=CombinationMethod.LEARNED)
        combiner = LearnedCombiner(
            config=config,
            neural_dim=64,
            symbolic_dim=64,
            output_dim=64,
        )

        neural = torch.randn(2, 64)
        symbolic = torch.randn(2, 64)

        combined, metadata = combiner.combine(neural, symbolic)

        assert combined.shape == (2, 64)

    def test_train_eval_mode(self):
        """Test train/eval mode switching."""
        config = CombinerConfig(method=CombinationMethod.ATTENTION)
        combiner = LearnedCombiner(
            config=config,
            neural_dim=64,
            symbolic_dim=64,
            output_dim=64,
        )

        combiner.train(True)
        assert combiner._layers.training

        combiner.eval()
        assert not combiner._layers.training

    def test_parameters(self):
        """Test parameter retrieval."""
        config = CombinerConfig(method=CombinationMethod.ATTENTION)
        combiner = LearnedCombiner(
            config=config,
            neural_dim=64,
            symbolic_dim=64,
            output_dim=64,
        )

        params = list(combiner.parameters())
        assert len(params) > 0


class TestNumpyAttentionCombiner:
    """Tests for NumpyAttentionCombiner."""

    def test_initialization(self):
        """Test initialization."""
        combiner = NumpyAttentionCombiner(temperature=0.5)
        assert combiner.temperature == 0.5

    def test_combine(self):
        """Test numpy attention combination."""
        combiner = NumpyAttentionCombiner()

        neural = np.array([1.0, 2.0, 3.0])
        symbolic = np.array([0.5, 1.0, 1.5])

        combined, metadata = combiner.combine(neural, symbolic)

        assert combined.shape == (3,)
        assert "neural_weight" in metadata
        assert "symbolic_weight" in metadata
        assert metadata["neural_weight"] + metadata["symbolic_weight"] == pytest.approx(1.0, rel=1e-6)

    def test_magnitude_weighting(self):
        """Test that larger magnitude scores get higher weights."""
        combiner = NumpyAttentionCombiner()

        # Neural has higher magnitude
        neural = np.array([10.0, 20.0, 30.0])
        symbolic = np.array([1.0, 2.0, 3.0])

        combined, metadata = combiner.combine(neural, symbolic)

        # Neural should have higher weight
        assert metadata["neural_weight"] > metadata["symbolic_weight"]


class TestCombineGeneScores:
    """Tests for combine_gene_scores function."""

    def test_weighted_sum(self):
        """Test weighted sum combination of gene scores."""
        neural_scores = {"GENE1": 0.8, "GENE2": 0.6}
        symbolic_scores = {"GENE1": 0.5, "GENE3": 0.7}

        combined = combine_gene_scores(
            neural_scores,
            symbolic_scores,
            method="weighted_sum",
            neural_weight=0.6,
        )

        # GENE1: 0.6 * 0.8 + 0.4 * 0.5 = 0.68
        assert combined["GENE1"] == pytest.approx(0.68)
        # GENE2: 0.6 * 0.6 + 0.4 * 0 = 0.36
        assert combined["GENE2"] == pytest.approx(0.36)
        # GENE3: 0.6 * 0 + 0.4 * 0.7 = 0.28
        assert combined["GENE3"] == pytest.approx(0.28)

    def test_max_combination(self):
        """Test max combination of gene scores."""
        neural_scores = {"GENE1": 0.8, "GENE2": 0.3}
        symbolic_scores = {"GENE1": 0.5, "GENE2": 0.7}

        combined = combine_gene_scores(
            neural_scores,
            symbolic_scores,
            method="max",
        )

        assert combined["GENE1"] == 0.8
        assert combined["GENE2"] == 0.7

    def test_product_combination(self):
        """Test product combination of gene scores."""
        neural_scores = {"GENE1": 1.0, "GENE2": 0.5}
        symbolic_scores = {"GENE1": 1.0, "GENE2": 0.5}

        combined = combine_gene_scores(
            neural_scores,
            symbolic_scores,
            method="product",
        )

        # sqrt((1.0 + 0.01) * (1.0 + 0.01)) ~ 1.01
        assert combined["GENE1"] == pytest.approx(1.01, rel=1e-2)

    def test_empty_scores(self):
        """Test handling of empty score dictionaries."""
        combined = combine_gene_scores({}, {})
        assert combined == {}

    def test_all_genes_included(self):
        """Test that all genes from both sources are included."""
        neural_scores = {"A": 1.0, "B": 2.0}
        symbolic_scores = {"C": 3.0, "D": 4.0}

        combined = combine_gene_scores(neural_scores, symbolic_scores)

        assert set(combined.keys()) == {"A", "B", "C", "D"}


class TestCreateSymbolicScoreVector:
    """Tests for create_symbolic_score_vector function."""

    def test_basic_vector_creation(self):
        """Test basic score vector creation."""
        # Create mock fired rules
        class MockFiredRule:
            def __init__(self, gene, confidence):
                self.bindings = {"G": gene}
                self.evidence = {}
                self.confidence = confidence

        fired_rules = [
            MockFiredRule("GENE1", 0.8),
            MockFiredRule("GENE2", 0.6),
        ]
        gene_list = ["GENE1", "GENE2", "GENE3"]

        scores = create_symbolic_score_vector(
            fired_rules,
            gene_list,
            use_confidence=True,
        )

        assert scores.shape == (3,)
        assert scores[0] == 0.8  # GENE1
        assert scores[1] == 0.6  # GENE2
        assert scores[2] == 0.0  # GENE3

    def test_without_confidence(self):
        """Test score vector without confidence weighting."""
        class MockFiredRule:
            def __init__(self, gene, confidence):
                self.bindings = {"G": gene}
                self.evidence = {}
                self.confidence = confidence

        fired_rules = [
            MockFiredRule("GENE1", 0.8),
            MockFiredRule("GENE1", 0.6),  # Same gene, should add up
        ]
        gene_list = ["GENE1", "GENE2"]

        scores = create_symbolic_score_vector(
            fired_rules,
            gene_list,
            use_confidence=False,
        )

        assert scores[0] == 2.0  # Two rules for GENE1
        assert scores[1] == 0.0

    def test_accumulation(self):
        """Test score accumulation for same gene."""
        class MockFiredRule:
            def __init__(self, gene, confidence):
                self.bindings = {"G": gene}
                self.evidence = {}
                self.confidence = confidence

        fired_rules = [
            MockFiredRule("GENE1", 0.5),
            MockFiredRule("GENE1", 0.3),
        ]
        gene_list = ["GENE1"]

        scores = create_symbolic_score_vector(
            fired_rules,
            gene_list,
            use_confidence=True,
        )

        assert scores[0] == pytest.approx(0.8)

    def test_gene_binding_alternatives(self):
        """Test different gene binding formats."""
        class MockFiredRuleGene:
            def __init__(self, gene, confidence):
                self.bindings = {"gene": gene}
                self.evidence = {}
                self.confidence = confidence

        class MockFiredRuleEvidence:
            def __init__(self, gene, confidence):
                self.bindings = {}
                self.evidence = {"gene": gene}
                self.confidence = confidence

        fired_rules = [
            MockFiredRuleGene("GENE1", 0.5),
            MockFiredRuleEvidence("GENE2", 0.7),
        ]
        gene_list = ["GENE1", "GENE2"]

        scores = create_symbolic_score_vector(fired_rules, gene_list)

        assert scores[0] == 0.5
        assert scores[1] == 0.7

    def test_missing_gene_ignored(self):
        """Test that genes not in list are ignored."""
        class MockFiredRule:
            def __init__(self, gene, confidence):
                self.bindings = {"G": gene}
                self.evidence = {}
                self.confidence = confidence

        fired_rules = [
            MockFiredRule("UNKNOWN", 0.8),
        ]
        gene_list = ["GENE1", "GENE2"]

        scores = create_symbolic_score_vector(fired_rules, gene_list)

        assert scores[0] == 0.0
        assert scores[1] == 0.0

    def test_empty_fired_rules(self):
        """Test handling of empty fired rules."""
        gene_list = ["GENE1", "GENE2"]
        scores = create_symbolic_score_vector([], gene_list)

        assert scores.shape == (2,)
        np.testing.assert_array_equal(scores, [0.0, 0.0])


class TestCombinerEdgeCases:
    """Tests for combiner edge cases."""

    def test_zero_scores(self):
        """Test handling of zero scores."""
        combiner = LearnedCombiner()

        neural = np.zeros(10)
        symbolic = np.zeros(10)

        combined, metadata = combiner.combine(neural, symbolic)

        np.testing.assert_array_equal(combined, np.zeros(10))

    def test_negative_scores(self):
        """Test handling of negative scores."""
        combiner = LearnedCombiner()

        neural = np.array([-1.0, -2.0, 3.0])
        symbolic = np.array([0.5, -1.0, 1.5])

        combined, metadata = combiner.combine(neural, symbolic)

        expected = 0.6 * neural + 0.4 * symbolic
        np.testing.assert_array_almost_equal(combined, expected)

    def test_single_element(self):
        """Test handling of single element arrays."""
        combiner = LearnedCombiner()

        neural = np.array([1.0])
        symbolic = np.array([2.0])

        combined, metadata = combiner.combine(neural, symbolic)

        assert combined.shape == (1,)
        expected = 0.6 * 1.0 + 0.4 * 2.0
        assert combined[0] == pytest.approx(expected)

    def test_large_arrays(self):
        """Test handling of large arrays."""
        combiner = LearnedCombiner()

        neural = np.random.randn(10000)
        symbolic = np.random.randn(10000)

        combined, metadata = combiner.combine(neural, symbolic)

        assert combined.shape == (10000,)

    def test_equal_weights(self):
        """Test equal weight combination."""
        config = CombinerConfig(
            neural_weight=0.5,
            symbolic_weight=0.5,
        )
        combiner = LearnedCombiner(config=config)

        neural = np.array([1.0, 2.0])
        symbolic = np.array([3.0, 4.0])

        combined, metadata = combiner.combine(neural, symbolic)

        expected = np.array([2.0, 3.0])  # (1+3)/2, (2+4)/2
        np.testing.assert_array_almost_equal(combined, expected)
