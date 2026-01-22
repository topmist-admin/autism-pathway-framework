"""
Tests for pretrained embedding extractors.

Tests cover:
- NodeEmbeddings data structure
- GeneformerExtractor (fallback mode)
- ESM2Extractor (fallback mode)
- LiteratureEmbedder (fallback mode)
- EmbeddingFusion
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
pretrained = importlib.import_module("modules.05_pretrained_embeddings")

# Extract classes from imported module
NodeEmbeddings = pretrained.NodeEmbeddings
VariantEffect = pretrained.VariantEffect
ExtractionMode = pretrained.ExtractionMode
EmbeddingSource = pretrained.EmbeddingSource
GeneformerExtractor = pretrained.GeneformerExtractor
ESM2Extractor = pretrained.ESM2Extractor
LiteratureEmbedder = pretrained.LiteratureEmbedder
EmbeddingFusion = pretrained.EmbeddingFusion
FusionMethod = pretrained.FusionMethod
FusionConfig = pretrained.FusionConfig
fuse_embeddings = pretrained.fuse_embeddings
align_embeddings = pretrained.align_embeddings
normalize_gene_id = pretrained.normalize_gene_id


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_gene_ids():
    """Sample gene IDs for testing."""
    return ["SHANK3", "CHD8", "SCN2A", "NRXN1", "SYNGAP1"]


@pytest.fixture
def sample_protein_sequences():
    """Sample protein sequences for testing."""
    return {
        "SHANK3": "MAEQQPVPSLPRLGRKIKSGTARVPGPGDP" + "A" * 50,
        "CHD8": "MEPSNQQSVDLQPVRQKIKSGTARPGPGDP" + "G" * 50,
        "SCN2A": "MSSSVDKPTTQHLSPGKLLSTCISCCQPV" + "L" * 50,
    }


@pytest.fixture
def sample_gene_descriptions():
    """Sample gene descriptions for testing."""
    return {
        "SHANK3": "SHANK3 is a scaffold protein in the postsynaptic density that plays a role in synapse formation and dendritic spine maturation.",
        "CHD8": "CHD8 is a chromatin remodeling factor that regulates gene expression during neurodevelopment and is associated with autism.",
        "SCN2A": "SCN2A encodes a voltage-gated sodium channel subunit essential for action potential generation in neurons.",
    }


@pytest.fixture
def sample_embeddings():
    """Create sample node embeddings for testing."""
    node_ids = ["A", "B", "C", "D"]
    embeddings = np.random.randn(4, 64).astype(np.float32)
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return NodeEmbeddings(
        node_ids=node_ids,
        embeddings=embeddings,
        source=EmbeddingSource.GENEFORMER,
        model_name="test_model",
    )


# ============================================================================
# NodeEmbeddings Tests
# ============================================================================

class TestNodeEmbeddings:
    """Tests for NodeEmbeddings data structure."""

    def test_creation(self, sample_embeddings):
        """Test creating NodeEmbeddings."""
        assert len(sample_embeddings) == 4
        assert sample_embeddings.embedding_dim == 64
        assert sample_embeddings.source == EmbeddingSource.GENEFORMER

    def test_get_embedding(self, sample_embeddings):
        """Test retrieving individual embeddings."""
        emb_a = sample_embeddings.get("A")
        assert emb_a is not None
        assert emb_a.shape == (64,)

        # Test non-existent node
        emb_x = sample_embeddings.get("X")
        assert emb_x is None

    def test_get_batch(self, sample_embeddings):
        """Test batch retrieval."""
        embs, found_ids = sample_embeddings.get_batch(["A", "B", "X"])
        assert len(found_ids) == 2
        assert "A" in found_ids
        assert "B" in found_ids
        assert embs.shape == (2, 64)

    def test_most_similar(self, sample_embeddings):
        """Test similarity search."""
        similar = sample_embeddings.most_similar("A", k=2)
        assert len(similar) == 2
        assert all(isinstance(s, tuple) for s in similar)
        assert all(len(s) == 2 for s in similar)

    def test_subset(self, sample_embeddings):
        """Test creating subset."""
        subset = sample_embeddings.subset(["A", "C"])
        assert len(subset) == 2
        assert "A" in subset.node_ids
        assert "C" in subset.node_ids

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

    def test_validation_error(self):
        """Test validation of mismatched dimensions."""
        with pytest.raises(ValueError):
            NodeEmbeddings(
                node_ids=["A", "B"],
                embeddings=np.array([[1, 2, 3]]),
            )


# ============================================================================
# GeneformerExtractor Tests
# ============================================================================

class TestGeneformerExtractor:
    """Tests for Geneformer extractor (fallback mode)."""

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = GeneformerExtractor(use_fallback=True)
        assert extractor.embedding_dim == 256
        assert extractor.source == EmbeddingSource.GENEFORMER

    def test_extract_genes(self, sample_gene_ids):
        """Test extracting gene embeddings."""
        extractor = GeneformerExtractor(use_fallback=True)
        embeddings = extractor.extract(sample_gene_ids)

        assert len(embeddings) == len(sample_gene_ids)
        assert embeddings.embedding_dim == 256
        assert all(g in embeddings.node_ids for g in sample_gene_ids)

    def test_deterministic_fallback(self, sample_gene_ids):
        """Test that fallback embeddings are deterministic."""
        extractor = GeneformerExtractor(use_fallback=True)

        emb1 = extractor.extract(sample_gene_ids)
        emb2 = extractor.extract(sample_gene_ids)

        np.testing.assert_array_almost_equal(emb1.embeddings, emb2.embeddings)

    def test_normalized_embeddings(self, sample_gene_ids):
        """Test that embeddings are normalized."""
        extractor = GeneformerExtractor(use_fallback=True)
        embeddings = extractor.extract(sample_gene_ids, normalize=True)

        norms = np.linalg.norm(embeddings.embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, 1.0, decimal=5)

    def test_gene_id_normalization(self):
        """Test that gene IDs are normalized."""
        assert normalize_gene_id("shank3") == "SHANK3"
        assert normalize_gene_id("Shank3") == "SHANK3"
        assert normalize_gene_id("ENSG00000148498") == "ENSG00000148498"


# ============================================================================
# ESM2Extractor Tests
# ============================================================================

class TestESM2Extractor:
    """Tests for ESM-2 extractor (fallback mode)."""

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = ESM2Extractor(use_fallback=True)
        assert extractor.embedding_dim == 1280  # Default model
        assert extractor.source == EmbeddingSource.ESM2

    def test_extract_proteins(self, sample_protein_sequences):
        """Test extracting protein embeddings."""
        extractor = ESM2Extractor(use_fallback=True)
        embeddings = extractor.extract(sample_protein_sequences)

        assert len(embeddings) == len(sample_protein_sequences)
        assert embeddings.embedding_dim == 1280

    def test_deterministic_fallback(self, sample_protein_sequences):
        """Test that fallback embeddings are deterministic."""
        extractor = ESM2Extractor(use_fallback=True)

        emb1 = extractor.extract(sample_protein_sequences)
        emb2 = extractor.extract(sample_protein_sequences)

        np.testing.assert_array_almost_equal(emb1.embeddings, emb2.embeddings)

    def test_predict_variant_effect(self, sample_protein_sequences):
        """Test variant effect prediction."""
        extractor = ESM2Extractor(use_fallback=True)

        effect = extractor.predict_variant_effect(
            gene_id="SHANK3",
            wt_sequence=sample_protein_sequences["SHANK3"],
            position=10,
            ref_aa="K",
            alt_aa="E",
        )

        assert isinstance(effect, VariantEffect)
        assert effect.gene_id == "SHANK3"
        assert effect.position == 10
        assert 0 <= effect.pathogenicity_score <= 1

    def test_different_model_variants(self):
        """Test different ESM-2 model sizes."""
        # Smaller model
        extractor_small = ESM2Extractor(
            model_variant="esm2_t6_8M", use_fallback=True
        )
        assert extractor_small.embedding_dim == 320

        # Larger model
        extractor_large = ESM2Extractor(
            model_variant="esm2_t33_650M", use_fallback=True
        )
        assert extractor_large.embedding_dim == 1280


# ============================================================================
# LiteratureEmbedder Tests
# ============================================================================

class TestLiteratureEmbedder:
    """Tests for literature embedder (fallback mode)."""

    def test_initialization(self):
        """Test embedder initialization."""
        embedder = LiteratureEmbedder(use_fallback=True)
        assert embedder.embedding_dim == 768  # PubMedBERT default
        assert embedder.source == EmbeddingSource.PUBMEDBERT

    def test_extract_descriptions(self, sample_gene_descriptions):
        """Test extracting embeddings from descriptions."""
        embedder = LiteratureEmbedder(use_fallback=True)
        embeddings = embedder.extract_from_descriptions(sample_gene_descriptions)

        assert len(embeddings) == len(sample_gene_descriptions)
        assert embeddings.embedding_dim == 768

    def test_deterministic_fallback(self, sample_gene_descriptions):
        """Test that fallback embeddings are deterministic."""
        embedder = LiteratureEmbedder(use_fallback=True)

        emb1 = embedder.extract_from_descriptions(sample_gene_descriptions)
        emb2 = embedder.extract_from_descriptions(sample_gene_descriptions)

        np.testing.assert_array_almost_equal(emb1.embeddings, emb2.embeddings)

    def test_extract_from_abstracts(self, sample_gene_descriptions):
        """Test extracting from multiple abstracts per gene."""
        embedder = LiteratureEmbedder(use_fallback=True)

        # Create multiple abstracts per gene
        gene_abstracts = {
            gene: [desc, desc + " Additional information."]
            for gene, desc in sample_gene_descriptions.items()
        }

        embeddings = embedder.extract_from_abstracts(gene_abstracts)
        assert len(embeddings) == len(gene_abstracts)


# ============================================================================
# EmbeddingFusion Tests
# ============================================================================

class TestEmbeddingFusion:
    """Tests for embedding fusion."""

    def test_concat_fusion(self):
        """Test concatenation fusion."""
        # Create embeddings with same nodes
        emb1 = NodeEmbeddings(
            node_ids=["A", "B", "C"],
            embeddings=np.random.randn(3, 32).astype(np.float32),
            source=EmbeddingSource.GENEFORMER,
        )
        emb2 = NodeEmbeddings(
            node_ids=["A", "B", "C"],
            embeddings=np.random.randn(3, 64).astype(np.float32),
            source=EmbeddingSource.ESM2,
        )

        fused = fuse_embeddings(
            {"geneformer": emb1, "esm2": emb2},
            method="concat",
        )

        assert len(fused) == 3
        assert fused.embedding_dim == 32 + 64  # Concatenated

    def test_average_fusion(self):
        """Test average fusion (same dimensions required)."""
        emb1 = NodeEmbeddings(
            node_ids=["A", "B", "C"],
            embeddings=np.ones((3, 32)).astype(np.float32),
            source=EmbeddingSource.GENEFORMER,
        )
        emb2 = NodeEmbeddings(
            node_ids=["A", "B", "C"],
            embeddings=np.ones((3, 32)).astype(np.float32) * 3,
            source=EmbeddingSource.ESM2,
        )

        fused = fuse_embeddings(
            {"source1": emb1, "source2": emb2},
            method="average",
        )

        # Average of 1 and 3 = 2, then normalized
        assert len(fused) == 3
        assert fused.embedding_dim == 32

    def test_weighted_sum_fusion(self):
        """Test weighted sum fusion."""
        emb1 = NodeEmbeddings(
            node_ids=["A", "B"],
            embeddings=np.ones((2, 32)).astype(np.float32),
            source=EmbeddingSource.GENEFORMER,
        )
        emb2 = NodeEmbeddings(
            node_ids=["A", "B"],
            embeddings=np.ones((2, 32)).astype(np.float32) * 2,
            source=EmbeddingSource.ESM2,
        )

        fused = fuse_embeddings(
            {"source1": emb1, "source2": emb2},
            method="weighted_sum",
            weights={"source1": 0.3, "source2": 0.7},
        )

        assert len(fused) == 2

    def test_pca_fusion(self):
        """Test PCA fusion."""
        # Need enough samples for PCA: n_components <= min(n_samples, n_features)
        node_ids = [f"gene_{i}" for i in range(50)]
        emb1 = NodeEmbeddings(
            node_ids=node_ids,
            embeddings=np.random.randn(50, 64).astype(np.float32),
            source=EmbeddingSource.GENEFORMER,
        )
        emb2 = NodeEmbeddings(
            node_ids=node_ids,
            embeddings=np.random.randn(50, 64).astype(np.float32),
            source=EmbeddingSource.ESM2,
        )

        config = FusionConfig(
            method=FusionMethod.PCA,
            output_dim=32,
        )
        fusion = EmbeddingFusion(config)
        fused = fusion.fuse({"source1": emb1, "source2": emb2})

        assert fused.embedding_dim == 32

    def test_partial_overlap(self):
        """Test fusion with partial node overlap."""
        emb1 = NodeEmbeddings(
            node_ids=["A", "B", "C"],
            embeddings=np.random.randn(3, 32).astype(np.float32),
            source=EmbeddingSource.GENEFORMER,
        )
        emb2 = NodeEmbeddings(
            node_ids=["B", "C", "D"],
            embeddings=np.random.randn(3, 32).astype(np.float32),
            source=EmbeddingSource.ESM2,
        )

        fused = fuse_embeddings(
            {"source1": emb1, "source2": emb2},
            method="average",
        )

        # Only B and C are common
        assert len(fused) == 2
        assert "B" in fused.node_ids
        assert "C" in fused.node_ids

    def test_align_embeddings(self):
        """Test embedding alignment utility."""
        emb1 = NodeEmbeddings(
            node_ids=["A", "B", "C"],
            embeddings=np.random.randn(3, 32).astype(np.float32),
        )
        emb2 = NodeEmbeddings(
            node_ids=["B", "C", "D"],
            embeddings=np.random.randn(3, 64).astype(np.float32),
        )

        common_ids, aligned = align_embeddings({"s1": emb1, "s2": emb2})

        assert set(common_ids) == {"B", "C"}
        assert aligned["s1"].shape == (2, 32)
        assert aligned["s2"].shape == (2, 64)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(
        self,
        sample_gene_ids,
        sample_protein_sequences,
        sample_gene_descriptions,
    ):
        """Test full extraction and fusion pipeline."""
        # Extract from each source (all use fallback mode)
        geneformer = GeneformerExtractor(use_fallback=True)
        gene_emb = geneformer.extract(sample_gene_ids)

        esm2 = ESM2Extractor(use_fallback=True)
        protein_emb = esm2.extract(sample_protein_sequences)

        literature = LiteratureEmbedder(use_fallback=True)
        lit_emb = literature.extract_from_descriptions(sample_gene_descriptions)

        # Fuse all sources
        fused = fuse_embeddings(
            {
                "geneformer": gene_emb,
                "esm2": protein_emb,
                "literature": lit_emb,
            },
            method="concat",
        )

        # Should have only genes common to all sources
        common_genes = set(sample_gene_ids) & set(sample_protein_sequences.keys()) & set(sample_gene_descriptions.keys())
        assert len(fused) == len(common_genes)

        # Total dimension = 256 + 1280 + 768
        expected_dim = 256 + 1280 + 768
        assert fused.embedding_dim == expected_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
