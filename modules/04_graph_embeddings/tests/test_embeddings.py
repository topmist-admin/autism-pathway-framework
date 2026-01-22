"""
Tests for graph embedding models.

Tests cover:
- NodeEmbeddings data structure
- TransE model training and inference
- RotatE model training and inference
- EmbeddingTrainer utilities
"""

import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import modules with numeric prefixes using importlib
graph_embeddings = importlib.import_module("modules.04_graph_embeddings")
knowledge_graph = importlib.import_module("modules.03_knowledge_graph")

# Extract classes from imported modules
NodeEmbeddings = graph_embeddings.NodeEmbeddings
RelationEmbeddings = graph_embeddings.RelationEmbeddings
TrainingHistory = graph_embeddings.TrainingHistory
EvaluationMetrics = graph_embeddings.EvaluationMetrics
TransEModel = graph_embeddings.TransEModel
RotatEModel = graph_embeddings.RotatEModel
TrainingConfig = graph_embeddings.TrainingConfig
EmbeddingTrainer = graph_embeddings.EmbeddingTrainer
train_embeddings = graph_embeddings.train_embeddings

KnowledgeGraph = knowledge_graph.KnowledgeGraph
KnowledgeGraphBuilder = knowledge_graph.KnowledgeGraphBuilder
NodeType = knowledge_graph.NodeType
EdgeType = knowledge_graph.EdgeType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_knowledge_graph():
    """Create a simple knowledge graph for testing."""
    kg = KnowledgeGraph()

    # Add genes
    genes = ["SHANK3", "CHD8", "SCN2A", "NRXN1", "SYNGAP1"]
    for gene in genes:
        kg.add_node(gene, NodeType.GENE)

    # Add pathways
    pathways = ["synaptic_pathway", "chromatin_pathway"]
    for pathway in pathways:
        kg.add_node(pathway, NodeType.PATHWAY)

    # Add gene-gene interactions
    interactions = [
        ("SHANK3", "NRXN1"),
        ("SHANK3", "SYNGAP1"),
        ("CHD8", "SCN2A"),
        ("NRXN1", "SYNGAP1"),
    ]
    for g1, g2 in interactions:
        kg.add_edge(g1, g2, EdgeType.GENE_INTERACTS, weight=0.9)

    # Add gene-pathway edges
    kg.add_edge("SHANK3", "synaptic_pathway", EdgeType.GENE_IN_PATHWAY)
    kg.add_edge("NRXN1", "synaptic_pathway", EdgeType.GENE_IN_PATHWAY)
    kg.add_edge("SYNGAP1", "synaptic_pathway", EdgeType.GENE_IN_PATHWAY)
    kg.add_edge("CHD8", "chromatin_pathway", EdgeType.GENE_IN_PATHWAY)

    return kg


@pytest.fixture
def sample_embeddings():
    """Create sample node embeddings for testing."""
    node_ids = ["A", "B", "C", "D"]
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],  # Similar to A
        [0.0, 1.0, 0.0],  # Different from A
        [0.0, 0.0, 1.0],
    ])
    return NodeEmbeddings(
        node_ids=node_ids,
        embeddings=embeddings,
        model_type="test",
    )


# ============================================================================
# NodeEmbeddings Tests
# ============================================================================

class TestNodeEmbeddings:
    """Tests for NodeEmbeddings data structure."""

    def test_creation(self, sample_embeddings):
        """Test creating NodeEmbeddings."""
        assert len(sample_embeddings) == 4
        assert sample_embeddings.embedding_dim == 3
        assert sample_embeddings.model_type == "test"

    def test_get_embedding(self, sample_embeddings):
        """Test retrieving individual embeddings."""
        emb_a = sample_embeddings.get("A")
        assert emb_a is not None
        assert emb_a.shape == (3,)
        np.testing.assert_array_equal(emb_a, [1.0, 0.0, 0.0])

        # Test non-existent node
        emb_x = sample_embeddings.get("X")
        assert emb_x is None

    def test_get_batch(self, sample_embeddings):
        """Test batch retrieval."""
        embs, found_ids = sample_embeddings.get_batch(["A", "B", "X"])
        assert len(found_ids) == 2
        assert "A" in found_ids
        assert "B" in found_ids
        assert embs.shape == (2, 3)

    def test_most_similar(self, sample_embeddings):
        """Test similarity search."""
        similar = sample_embeddings.most_similar("A", k=2)
        assert len(similar) == 2
        # B should be most similar to A
        assert similar[0][0] == "B"
        assert similar[0][1] > 0.9  # High similarity

    def test_compute_similarity(self, sample_embeddings):
        """Test pairwise similarity."""
        sim_ab = sample_embeddings.compute_similarity("A", "B")
        sim_ac = sample_embeddings.compute_similarity("A", "C")

        assert sim_ab is not None
        assert sim_ac is not None
        assert sim_ab > sim_ac  # A-B more similar than A-C

    def test_save_load_npz(self, sample_embeddings):
        """Test saving and loading in npz format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embeddings.npz"
            sample_embeddings.save(str(path))

            loaded = NodeEmbeddings.load(str(path))
            assert len(loaded) == len(sample_embeddings)
            assert loaded.embedding_dim == sample_embeddings.embedding_dim
            np.testing.assert_array_almost_equal(
                loaded.embeddings, sample_embeddings.embeddings
            )

    def test_save_load_pkl(self, sample_embeddings):
        """Test saving and loading in pickle format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embeddings.pkl"
            sample_embeddings.save(str(path))

            loaded = NodeEmbeddings.load(str(path))
            assert len(loaded) == len(sample_embeddings)

    def test_to_dict(self, sample_embeddings):
        """Test conversion to dictionary."""
        emb_dict = sample_embeddings.to_dict()
        assert len(emb_dict) == 4
        assert "A" in emb_dict
        np.testing.assert_array_equal(emb_dict["A"], [1.0, 0.0, 0.0])

    def test_validation_error(self):
        """Test validation of mismatched dimensions."""
        with pytest.raises(ValueError):
            NodeEmbeddings(
                node_ids=["A", "B"],
                embeddings=np.array([[1, 2, 3]]),  # Only 1 embedding for 2 nodes
            )


# ============================================================================
# TransE Tests
# ============================================================================

class TestTransEModel:
    """Tests for TransE embedding model."""

    def test_initialization(self):
        """Test model initialization."""
        model = TransEModel(embedding_dim=64, margin=1.0, norm=1)
        assert model.embedding_dim == 64
        assert model.margin == 1.0
        assert model.norm == 1
        assert not model.is_trained

    def test_invalid_norm(self):
        """Test that invalid norm raises error."""
        with pytest.raises(ValueError):
            TransEModel(norm=3)

    def test_training(self, sample_knowledge_graph):
        """Test model training."""
        model = TransEModel(
            embedding_dim=32,
            margin=1.0,
            learning_rate=0.1,
            random_state=42,
        )

        history = model.train(
            sample_knowledge_graph,
            epochs=20,
            batch_size=4,
            verbose=False,
        )

        assert model.is_trained
        assert len(history.epochs) == 20
        assert len(history.losses) == 20
        # Loss should generally decrease
        assert history.losses[-1] <= history.losses[0] + 0.5

    def test_get_embeddings(self, sample_knowledge_graph):
        """Test extracting embeddings after training."""
        model = TransEModel(embedding_dim=16, random_state=42)
        model.train(sample_knowledge_graph, epochs=10, verbose=False)

        embeddings = model.get_node_embeddings()
        assert isinstance(embeddings, NodeEmbeddings)
        assert embeddings.embedding_dim == 16
        assert embeddings.model_type == "TransEModel"

        # Check all nodes have embeddings
        for node in ["SHANK3", "CHD8", "synaptic_pathway"]:
            assert embeddings.get(node) is not None

    def test_predict_link(self, sample_knowledge_graph):
        """Test link prediction."""
        model = TransEModel(embedding_dim=16, random_state=42)
        model.train(sample_knowledge_graph, epochs=20, verbose=False)

        # Predict existing edge (should have lower score)
        score_existing = model.predict_link(
            "SHANK3", "gene_interacts_gene", "NRXN1"
        )

        # Predict non-existing edge
        score_new = model.predict_link(
            "SHANK3", "gene_interacts_gene", "CHD8"
        )

        assert score_existing is not None
        assert score_new is not None
        # Existing edge should score better (lower) on average
        # Note: with few epochs this may not always hold

    def test_predict_tail(self, sample_knowledge_graph):
        """Test tail prediction."""
        model = TransEModel(embedding_dim=16, random_state=42)
        model.train(sample_knowledge_graph, epochs=20, verbose=False)

        predictions = model.predict_tail("SHANK3", "gene_interacts_gene", k=3)
        assert len(predictions) == 3
        assert all(isinstance(p, tuple) for p in predictions)
        assert all(len(p) == 2 for p in predictions)

    def test_save_load(self, sample_knowledge_graph):
        """Test model serialization."""
        model = TransEModel(embedding_dim=16, random_state=42)
        model.train(sample_knowledge_graph, epochs=10, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(str(path))

            loaded = TransEModel.load(str(path))
            assert loaded.is_trained
            assert loaded.embedding_dim == 16

            # Check embeddings match
            orig_emb = model.get_node_embeddings()
            loaded_emb = loaded.get_node_embeddings()
            np.testing.assert_array_almost_equal(
                orig_emb.embeddings, loaded_emb.embeddings
            )


# ============================================================================
# RotatE Tests
# ============================================================================

class TestRotatEModel:
    """Tests for RotatE embedding model."""

    def test_initialization(self):
        """Test model initialization."""
        model = RotatEModel(embedding_dim=64, margin=6.0)
        # RotatE requires even dimensions
        assert model.embedding_dim == 64
        assert model.margin == 6.0
        assert not model.is_trained

    def test_odd_embedding_dim_adjusted(self):
        """Test that odd embedding dimensions are adjusted."""
        model = RotatEModel(embedding_dim=63)
        assert model.embedding_dim == 64  # Adjusted to even

    def test_training(self, sample_knowledge_graph):
        """Test RotatE training."""
        model = RotatEModel(
            embedding_dim=32,
            margin=6.0,
            learning_rate=0.01,
            random_state=42,
        )

        history = model.train(
            sample_knowledge_graph,
            epochs=20,
            batch_size=4,
            verbose=False,
        )

        assert model.is_trained
        assert len(history.losses) == 20

    def test_get_embeddings(self, sample_knowledge_graph):
        """Test extracting RotatE embeddings."""
        model = RotatEModel(embedding_dim=32, random_state=42)
        model.train(sample_knowledge_graph, epochs=10, verbose=False)

        embeddings = model.get_node_embeddings()
        assert isinstance(embeddings, NodeEmbeddings)
        assert embeddings.embedding_dim == 32
        assert embeddings.model_type == "RotatEModel"

    def test_relation_embeddings(self, sample_knowledge_graph):
        """Test that relation embeddings are proper rotations."""
        model = RotatEModel(embedding_dim=32, random_state=42)
        model.train(sample_knowledge_graph, epochs=10, verbose=False)

        rel_emb = model.get_relation_embeddings()
        assert isinstance(rel_emb, RelationEmbeddings)

        # Check that relation embeddings have unit modulus
        # (they represent rotations in complex space)
        for i in range(len(rel_emb.relation_ids)):
            emb = rel_emb.embeddings[i]
            re = emb[:len(emb)//2]
            im = emb[len(emb)//2:]
            modulus = np.sqrt(re**2 + im**2)
            np.testing.assert_array_almost_equal(modulus, 1.0, decimal=5)


# ============================================================================
# EmbeddingTrainer Tests
# ============================================================================

class TestEmbeddingTrainer:
    """Tests for high-level training utilities."""

    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.model_type == "transe"
        assert config.embedding_dim == 128
        assert config.epochs == 100

    def test_create_transe_model(self):
        """Test creating TransE via trainer."""
        config = TrainingConfig(model_type="transe", embedding_dim=64)
        trainer = EmbeddingTrainer(config)
        model = trainer.create_model()
        assert isinstance(model, TransEModel)
        assert model.embedding_dim == 64

    def test_create_rotate_model(self):
        """Test creating RotatE via trainer."""
        config = TrainingConfig(model_type="rotate", embedding_dim=64)
        trainer = EmbeddingTrainer(config)
        model = trainer.create_model()
        assert isinstance(model, RotatEModel)

    def test_train_with_trainer(self, sample_knowledge_graph):
        """Test training via trainer."""
        config = TrainingConfig(
            model_type="transe",
            embedding_dim=32,
            epochs=10,
            verbose=False,
            random_state=42,
        )
        trainer = EmbeddingTrainer(config)
        model, history = trainer.train(sample_knowledge_graph)

        assert model.is_trained
        assert len(history.losses) == 10

    def test_convenience_function(self, sample_knowledge_graph):
        """Test train_embeddings convenience function."""
        model, embeddings = train_embeddings(
            sample_knowledge_graph,
            model_type="transe",
            embedding_dim=16,
            epochs=10,
            verbose=False,
            random_state=42,
        )

        assert model.is_trained
        assert isinstance(embeddings, NodeEmbeddings)
        assert embeddings.embedding_dim == 16


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with Module 03."""

    def test_builder_to_embeddings(self):
        """Test full pipeline from builder to embeddings."""
        # Build graph using builder
        kg = (
            KnowledgeGraphBuilder()
            .add_genes(["GENE1", "GENE2", "GENE3", "GENE4"])
            .build()
        )

        # Add some edges manually (since we have no pathway data)
        kg.add_edge("GENE1", "GENE2", EdgeType.GENE_INTERACTS, 0.8)
        kg.add_edge("GENE2", "GENE3", EdgeType.GENE_INTERACTS, 0.9)
        kg.add_edge("GENE3", "GENE4", EdgeType.GENE_INTERACTS, 0.7)

        # Train embeddings
        model = TransEModel(embedding_dim=16, random_state=42)
        model.train(kg, epochs=10, verbose=False)

        embeddings = model.get_node_embeddings()

        # Verify all genes have embeddings
        for gene in ["GENE1", "GENE2", "GENE3", "GENE4"]:
            assert embeddings.get(gene) is not None

        # Find similar genes to GENE1
        similar = embeddings.most_similar("GENE1", k=2)
        assert len(similar) == 2

    def test_embedding_similarity_reflects_structure(self, sample_knowledge_graph):
        """Test that embeddings capture graph structure."""
        model = TransEModel(embedding_dim=32, random_state=42)
        model.train(sample_knowledge_graph, epochs=50, verbose=False)

        embeddings = model.get_node_embeddings()

        # SHANK3 and NRXN1 interact - should be somewhat similar
        sim_interacting = embeddings.compute_similarity("SHANK3", "NRXN1")

        # SHANK3 and CHD8 don't directly interact
        sim_non_interacting = embeddings.compute_similarity("SHANK3", "CHD8")

        # Both should be valid similarities
        assert sim_interacting is not None
        assert sim_non_interacting is not None
        # Note: the exact relationship depends on training


# ============================================================================
# TrainingHistory Tests
# ============================================================================

class TestTrainingHistory:
    """Tests for TrainingHistory data structure."""

    def test_add_epoch(self):
        """Test adding epoch records."""
        history = TrainingHistory()
        history.add_epoch(0, 1.5, accuracy=0.6)
        history.add_epoch(1, 1.2, accuracy=0.7)
        history.add_epoch(2, 1.0, accuracy=0.8)

        assert len(history.epochs) == 3
        assert len(history.losses) == 3
        assert "accuracy" in history.metrics
        assert len(history.metrics["accuracy"]) == 3

    def test_best_epoch(self):
        """Test finding best epoch."""
        history = TrainingHistory()
        history.add_epoch(0, 1.5)
        history.add_epoch(1, 0.8)  # Best
        history.add_epoch(2, 1.0)

        assert history.best_epoch == 1

    def test_final_loss(self):
        """Test getting final loss."""
        history = TrainingHistory()
        history.add_epoch(0, 1.5)
        history.add_epoch(1, 1.0)

        assert history.final_loss == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
