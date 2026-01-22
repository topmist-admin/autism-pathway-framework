"""
Tests for Module 06: Ontology-Aware GNN

Tests cover:
- GNN layers (EdgeTypeTransform, MessagePassingLayer)
- Attention mechanisms (BiologicalAttention, EdgeTypeAttention)
- Main model (OntologyAwareGNN)
- Configuration management
- Utility functions
"""

import pytest
import numpy as np
import importlib
from dataclasses import asdict

# Import module using importlib (numeric prefix)
ontology_gnn = importlib.import_module("modules.06_ontology_gnn")

# Extract classes
EdgeTypeTransform = ontology_gnn.EdgeTypeTransform
MessagePassingLayer = ontology_gnn.MessagePassingLayer
HierarchicalAggregator = ontology_gnn.HierarchicalAggregator
BioPriorWeighting = ontology_gnn.BioPriorWeighting
BiologicalAttention = ontology_gnn.BiologicalAttention
EdgeTypeAttention = ontology_gnn.EdgeTypeAttention
GOSemanticAttention = ontology_gnn.GOSemanticAttention
PathwayCoAttention = ontology_gnn.PathwayCoAttention
OntologyAwareGNN = ontology_gnn.OntologyAwareGNN
GNNConfig = ontology_gnn.GNNConfig
GNNOutput = ontology_gnn.GNNOutput
GNNTrainer = ontology_gnn.GNNTrainer
OntologyGNNConfig = ontology_gnn.OntologyGNNConfig
ModelConfig = ontology_gnn.ModelConfig
GraphConfig = ontology_gnn.GraphConfig
BioPriorConfig = ontology_gnn.BioPriorConfig
create_default_config = ontology_gnn.create_default_config
create_autism_config = ontology_gnn.create_autism_config
create_lightweight_config = ontology_gnn.create_lightweight_config
GraphData = ontology_gnn.GraphData
prepare_graph_data = ontology_gnn.prepare_graph_data
normalize_priors = ontology_gnn.normalize_priors
compute_metrics = ontology_gnn.compute_metrics
compute_link_prediction_metrics = ontology_gnn.compute_link_prediction_metrics
create_negative_samples = ontology_gnn.create_negative_samples
split_edges = ontology_gnn.split_edges
get_subgraph = ontology_gnn.get_subgraph
TORCH_AVAILABLE = ontology_gnn.TORCH_AVAILABLE

# Try to import torch for tensor tests
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestConfiguration:
    """Tests for configuration classes."""

    def test_model_config_defaults(self):
        """Test ModelConfig has sensible defaults."""
        config = ModelConfig()
        assert config.input_dim == 256
        assert config.hidden_dim == 256
        assert config.output_dim == 128
        assert config.num_layers == 3
        assert config.num_heads == 8
        assert 0 <= config.dropout <= 1

    def test_gnn_config_defaults(self):
        """Test GNNConfig has sensible defaults."""
        config = GNNConfig()
        assert config.input_dim == 256
        assert config.hidden_dim == 256
        assert len(config.edge_types) > 0
        assert len(config.node_types) > 0
        assert len(config.prior_types) > 0

    def test_ontology_gnn_config(self):
        """Test OntologyGNNConfig combines all sub-configs."""
        config = OntologyGNNConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.graph, GraphConfig)
        assert isinstance(config.bio_priors, BioPriorConfig)

    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = OntologyGNNConfig()
        d = config.to_dict()
        assert "model" in d
        assert "graph" in d
        assert "training" in d

    def test_config_from_dict(self):
        """Test config deserialization from dict."""
        original = OntologyGNNConfig(
            model=ModelConfig(hidden_dim=512)
        )
        d = original.to_dict()
        loaded = OntologyGNNConfig.from_dict(d)
        assert loaded.model.hidden_dim == 512

    def test_create_default_config(self):
        """Test default config factory."""
        config = create_default_config()
        assert isinstance(config, OntologyGNNConfig)

    def test_create_autism_config(self):
        """Test autism-optimized config factory."""
        config = create_autism_config()
        assert config.model.num_layers >= 3
        assert "pli" in config.bio_priors.prior_types
        assert "sfari_score" in config.bio_priors.prior_types

    def test_create_lightweight_config(self):
        """Test lightweight config factory."""
        config = create_lightweight_config()
        assert config.model.hidden_dim < 128
        assert config.model.num_layers <= 2


class TestLayers:
    """Tests for GNN layers."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_edge_type_transform_init(self):
        """Test EdgeTypeTransform initialization."""
        edge_types = ["ppi", "pathway", "go"]
        transform = EdgeTypeTransform(
            in_dim=64, out_dim=32, edge_types=edge_types
        )
        assert len(transform.transforms) == len(edge_types)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_edge_type_transform_forward(self):
        """Test EdgeTypeTransform forward pass."""
        edge_types = ["ppi", "pathway"]
        transform = EdgeTypeTransform(
            in_dim=64, out_dim=32, edge_types=edge_types
        )
        x = torch.randn(10, 64)
        out = transform(x, "ppi")
        assert out.shape == (10, 32)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_edge_type_transform_unknown_type(self):
        """Test EdgeTypeTransform handles unknown edge types."""
        transform = EdgeTypeTransform(
            in_dim=64, out_dim=32, edge_types=["ppi"]
        )
        x = torch.randn(10, 64)
        out = transform(x, "unknown_type")
        assert out.shape == (10, 32)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_message_passing_layer(self):
        """Test MessagePassingLayer forward pass."""
        edge_types = ["ppi", "pathway"]
        layer = MessagePassingLayer(
            in_dim=64, out_dim=64, edge_types=edge_types
        )
        x = torch.randn(20, 64)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_type = torch.randint(0, 2, (50,))

        out = layer(x, edge_index, edge_type, edge_types)
        assert out.shape == (20, 64)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_hierarchical_aggregator(self):
        """Test HierarchicalAggregator."""
        agg = HierarchicalAggregator(hidden_dim=64, aggregation="mean")
        x = torch.randn(30, 64)
        hierarchy_edges = torch.tensor([
            [0, 1, 2, 3],  # children
            [10, 10, 11, 11],  # parents
        ])
        out = agg(x, hierarchy_edges, num_levels=2)
        assert out.shape == x.shape

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_bio_prior_weighting(self):
        """Test BioPriorWeighting."""
        prior_types = ["pli", "expression"]
        weighting = BioPriorWeighting(
            hidden_dim=64,
            prior_types=prior_types,
            combination="multiplicative",
        )
        bio_priors = {
            "pli": torch.rand(20),
            "expression": torch.rand(20),
        }
        weights = weighting(bio_priors)
        assert weights.shape == (20, 1)
        assert (weights >= 0).all()
        assert (weights <= 1).all()


class TestAttention:
    """Tests for attention mechanisms."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_biological_attention_init(self):
        """Test BiologicalAttention initialization."""
        attn = BiologicalAttention(
            hidden_dim=64, num_heads=8, prior_types=["pli"]
        )
        assert attn.num_heads == 8
        assert attn.head_dim == 8

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_biological_attention_forward(self):
        """Test BiologicalAttention forward pass."""
        attn = BiologicalAttention(
            hidden_dim=64, num_heads=8, prior_types=["pli"]
        )
        x = torch.randn(20, 64)
        bio_priors = {"pli": torch.rand(20)}
        out, weights = attn(x, x, x, bio_priors=bio_priors)
        assert out.shape == (20, 64)
        assert weights.shape[-1] == 20  # seq_len

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_biological_attention_batched(self):
        """Test BiologicalAttention with batched input."""
        attn = BiologicalAttention(hidden_dim=64, num_heads=8)
        x = torch.randn(4, 20, 64)  # batch=4
        out, weights = attn(x, x, x)
        assert out.shape == (4, 20, 64)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_edge_type_attention(self):
        """Test EdgeTypeAttention."""
        edge_types = ["ppi", "pathway"]
        attn = EdgeTypeAttention(
            hidden_dim=64, edge_types=edge_types, heads_per_type=2
        )
        x = torch.randn(20, 64)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_type = torch.randint(0, 2, (50,))
        out = attn(x, edge_index, edge_type, edge_types)
        assert out.shape == (20, 64)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_go_semantic_attention(self):
        """Test GOSemanticAttention."""
        attn = GOSemanticAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(20, 64)
        go_sim = torch.rand(20, 20)
        out, weights = attn(x, go_similarity=go_sim)
        assert out.shape == (20, 64)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pathway_co_attention(self):
        """Test PathwayCoAttention."""
        attn = PathwayCoAttention(gene_dim=64, pathway_dim=32, num_heads=4)
        gene_feat = torch.randn(100, 64)
        pathway_feat = torch.randn(20, 32)
        membership = torch.randint(0, 2, (100, 20)).float()
        gene_out, pathway_out = attn(gene_feat, pathway_feat, membership)
        assert gene_out.shape == gene_feat.shape
        assert pathway_out.shape == pathway_feat.shape


class TestModel:
    """Tests for OntologyAwareGNN model."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_model_init(self):
        """Test OntologyAwareGNN initialization."""
        config = GNNConfig(hidden_dim=64, num_layers=2, num_heads=4)
        model = OntologyAwareGNN(config)
        assert model.config.hidden_dim == 64

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_model_forward(self):
        """Test OntologyAwareGNN forward pass."""
        config = GNNConfig(
            input_dim=64, hidden_dim=64, output_dim=32,
            num_layers=2, num_heads=4
        )
        model = OntologyAwareGNN(config)

        node_features = {"gene": torch.randn(50, 64)}
        edge_index = torch.randint(0, 50, (2, 100))
        edge_type = torch.randint(0, 3, (100,))
        node_type_indices = {"gene": torch.arange(50)}

        output = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            node_type_indices=node_type_indices,
        )

        assert isinstance(output, GNNOutput)
        assert "gene" in output.node_embeddings

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_model_with_bio_priors(self):
        """Test model with biological priors."""
        config = GNNConfig(
            input_dim=64, hidden_dim=64, output_dim=32,
            num_layers=2, num_heads=4,
            prior_types=["pli", "expression"],
        )
        model = OntologyAwareGNN(config)

        node_features = {"gene": torch.randn(50, 64)}
        edge_index = torch.randint(0, 50, (2, 100))
        edge_type = torch.randint(0, 3, (100,))
        node_type_indices = {"gene": torch.arange(50)}
        bio_priors = {
            "pli": torch.rand(50),
            "expression": torch.rand(50),
        }

        output = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            node_type_indices=node_type_indices,
            bio_priors=bio_priors,
        )

        assert output.node_embeddings["gene"].shape == (50, 32)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_model_encode(self):
        """Test model encode method."""
        config = GNNConfig(
            input_dim=64, hidden_dim=64, output_dim=32,
            num_layers=2, num_heads=4
        )
        model = OntologyAwareGNN(config)

        node_features = {"gene": torch.randn(50, 64)}
        edge_index = torch.randint(0, 50, (2, 100))
        edge_type = torch.randint(0, 3, (100,))

        embeddings = model.encode(node_features, edge_index, edge_type)
        assert embeddings.shape[0] == 50
        assert embeddings.shape[1] == 32


class TestTrainer:
    """Tests for GNNTrainer."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_trainer_init(self):
        """Test GNNTrainer initialization."""
        config = GNNConfig(
            input_dim=64, hidden_dim=64, output_dim=32,
            num_layers=2, num_heads=4
        )
        model = OntologyAwareGNN(config)
        trainer = GNNTrainer(model, learning_rate=1e-3)
        assert trainer.model is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_trainer_evaluate(self):
        """Test trainer evaluation."""
        config = GNNConfig(
            input_dim=64, hidden_dim=64, output_dim=32,
            num_layers=2, num_heads=4
        )
        model = OntologyAwareGNN(config)
        trainer = GNNTrainer(model)

        node_features = {"gene": torch.randn(50, 64)}
        edge_index = torch.randint(0, 50, (2, 100))
        edge_type = torch.randint(0, 3, (100,))
        labels = torch.randint(0, 2, (50,))
        node_type_indices = {"gene": torch.arange(50)}

        metrics = trainer.evaluate(
            node_features, edge_index, edge_type,
            labels=labels, node_type_indices=node_type_indices,
        )
        assert "loss" in metrics


class TestUtils:
    """Tests for utility functions."""

    def test_normalize_priors_minmax(self):
        """Test minmax normalization."""
        priors = {"pli": np.array([0.1, 0.5, 0.9])}
        normalized = normalize_priors(priors, method="minmax")
        assert normalized["pli"].min() >= 0
        assert normalized["pli"].max() <= 1

    def test_normalize_priors_zscore(self):
        """Test zscore normalization."""
        priors = {"pli": np.array([0.1, 0.5, 0.9, 0.3, 0.7])}
        normalized = normalize_priors(priors, method="zscore")
        assert abs(normalized["pli"].mean()) < 0.1

    def test_normalize_priors_rank(self):
        """Test rank normalization."""
        priors = {"pli": np.array([0.1, 0.5, 0.9])}
        normalized = normalize_priors(priors, method="rank")
        assert normalized["pli"].min() == 0
        assert normalized["pli"].max() == 1

    def test_compute_metrics_binary(self):
        """Test binary classification metrics."""
        predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        labels = np.array([1, 0, 1])
        metrics = compute_metrics(predictions, labels)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_compute_metrics_accuracy(self):
        """Test accuracy computation."""
        predictions = np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
        labels = np.array([0, 1, 1])
        metrics = compute_metrics(predictions, labels)
        assert metrics["accuracy"] == 1.0

    def test_compute_link_prediction_metrics(self):
        """Test link prediction metrics."""
        scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0, 0])
        metrics = compute_link_prediction_metrics(scores, labels)
        assert "hits@10" in metrics
        assert "mrr" in metrics
        assert "auc" in metrics

    def test_create_negative_samples(self):
        """Test negative sampling."""
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        neg_edges, neg_labels = create_negative_samples(
            edge_index, num_nodes=10, num_negative=2
        )
        assert neg_edges.shape[1] == 6  # 3 edges * 2 negatives each
        assert (neg_labels == 0).all()

    def test_split_edges(self):
        """Test edge splitting."""
        edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        train, val, test = split_edges(edge_index, train_ratio=0.6, val_ratio=0.2)
        total = train.shape[1] + val.shape[1] + test.shape[1]
        assert total == 5

    def test_get_subgraph(self):
        """Test subgraph extraction."""
        edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        node_indices = np.array([0, 1, 2])
        sub_edges, relabel_map = get_subgraph(edge_index, node_indices, relabel=True)
        # Only edges within {0, 1, 2}
        assert sub_edges.shape[1] <= edge_index.shape[1]
        assert relabel_map is not None


class TestGraphData:
    """Tests for GraphData container."""

    def test_graph_data_creation(self):
        """Test GraphData creation."""
        data = GraphData(
            node_features={"gene": np.random.randn(50, 64)},
            edge_index=np.array([[0, 1], [1, 2]]),
            edge_type=np.array([0, 1]),
            edge_type_names=["ppi", "pathway"],
            node_type_indices={"gene": np.arange(50)},
        )
        assert "gene" in data.node_features
        assert len(data.edge_type_names) == 2


class TestNumpyFallback:
    """Tests for numpy fallback implementations."""

    def test_fallback_edge_transform(self):
        """Test EdgeTypeTransform fallback."""
        transform = EdgeTypeTransform(
            in_dim=64, out_dim=32, edge_types=["ppi"]
        )
        x = np.random.randn(10, 64)
        out = transform.forward(x, "ppi")
        assert out.shape == (10, 32)

    def test_fallback_message_passing(self):
        """Test MessagePassingLayer fallback."""
        layer = MessagePassingLayer(
            in_dim=64, out_dim=32, edge_types=["ppi"]
        )
        x = np.random.randn(20, 64)
        edge_index = np.random.randint(0, 20, (2, 50))
        edge_type = np.random.randint(0, 1, (50,))
        out = layer.forward(x, edge_index, edge_type, ["ppi"])
        assert out.shape[0] == 20

    def test_fallback_biological_attention(self):
        """Test BiologicalAttention fallback."""
        attn = BiologicalAttention(hidden_dim=64, num_heads=8)
        x = np.random.randn(20, 64)
        out, weights = attn.forward(x, x, x)
        assert out.shape == x.shape

    def test_fallback_ontology_gnn(self):
        """Test OntologyAwareGNN fallback."""
        config = GNNConfig(input_dim=64, hidden_dim=64, output_dim=32)
        model = OntologyAwareGNN(config)
        node_features = {"gene": np.random.randn(50, 64)}
        edge_index = np.random.randint(0, 50, (2, 100))
        edge_type = np.random.randint(0, 3, (100,))
        output = model.forward(node_features, edge_index, edge_type)
        assert isinstance(output, GNNOutput)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
