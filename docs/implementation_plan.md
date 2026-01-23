# Implementation Plan: Domain-Aware Autism Genetics Platform

> **Implementation Status**: Phase 5 Complete - ALL MODULES IMPLEMENTED | Last Updated: January 2026

## Module Status Overview

| Module | Name | Phase | Status | Completion |
|--------|------|-------|--------|------------|
| 01 | Data Loaders | 1A | ✅ Complete | 100% |
| 02 | Variant Processing | 1B | ✅ Complete | 100% |
| 03 | Knowledge Graph | 2A | ✅ Complete | 100% |
| 04 | Graph Embeddings | 2B | ✅ Complete | 100% |
| 05 | Pretrained Embeddings | 2C | ✅ Complete | 100% |
| 06 | Ontology GNN | 3A | ✅ Complete | 100% |
| 07 | Pathway Scoring | 3B | ✅ Complete | 100% |
| 08 | Subtype Clustering | 3C | ✅ Complete | 100% |
| 09 | Symbolic Rules | 4A | ✅ Complete | 100% |
| 10 | Neurosymbolic | 4B | ✅ Complete | 100% |
| 11 | Therapeutic Hypotheses | 4C | ✅ Complete | 100% |
| 12 | Causal Inference | 5 | ✅ Complete | 100% |

**Legend**: ✅ Complete | 🔄 In Progress | 🔲 Not Started

**All core modules are now complete!** Next steps: Cross-module integration pipelines (Sessions 17-18)

---

## Design Principles for Context-Manageable Implementation

### Problem: LLM Context Limitations

When implementing with AI assistance, large codebases and complex systems can exceed context windows, leading to:
- Loss of important details
- Inconsistent implementations across sessions
- Difficulty maintaining coherence

### Solution: Modular Architecture with Clear Boundaries

Each module should be:
1. **Self-contained**: Implementable in a single session
2. **Well-documented**: README + interface contracts
3. **Independently testable**: No dependencies on unbuilt modules
4. **Small enough**: <1000 lines per module core logic

---

## Project Structure

```
autism-genetics-platform/
│
├── README.md                           # Project overview
├── ARCHITECTURE.md                     # System design document
├── pyproject.toml                      # Project dependencies
│
├── modules/                            # Independent modules
│   │
│   ├── 01_data_loaders/               # Phase 1A
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── vcf_loader.py              # VCF parsing
│   │   ├── annotation_loader.py       # External annotations
│   │   ├── pathway_loader.py          # GO, Reactome, KEGG
│   │   ├── expression_loader.py       # BrainSpan developmental expression
│   │   ├── single_cell_loader.py      # Single-cell atlas (Allen Brain)
│   │   ├── constraint_loader.py       # gnomAD pLI/LOEUF scores
│   │   └── tests/
│   │
│   ├── 02_variant_processing/         # Phase 1B
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── qc_filters.py              # Quality control
│   │   ├── annotation.py              # Functional annotation
│   │   ├── gene_burden.py             # Variant → gene aggregation
│   │   └── tests/
│   │
│   ├── 03_knowledge_graph/            # Phase 2A
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── schema.py                  # Node/edge type definitions
│   │   ├── builder.py                 # Graph construction
│   │   ├── exporters.py               # Neo4j, DGL, PyG formats
│   │   └── tests/
│   │
│   ├── 04_graph_embeddings/           # Phase 2B
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── transe.py                  # TransE implementation
│   │   ├── rotate.py                  # RotatE implementation
│   │   ├── trainer.py                 # Embedding training
│   │   └── tests/
│   │
│   ├── 05_pretrained_embeddings/      # Phase 2C
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── geneformer_extractor.py    # Geneformer embeddings
│   │   ├── esm2_extractor.py          # Protein embeddings
│   │   ├── fusion.py                  # Multi-modal fusion
│   │   └── tests/
│   │
│   ├── 06_ontology_gnn/               # Phase 3A
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── layers.py                  # GNN layer implementations
│   │   ├── attention.py               # Biological attention
│   │   ├── model.py                   # Full model architecture
│   │   └── tests/
│   │
│   ├── 07_pathway_scoring/            # Phase 3B
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── aggregation.py             # Gene → pathway scoring
│   │   ├── network_propagation.py     # Signal refinement
│   │   ├── normalization.py           # Score normalization
│   │   └── tests/
│   │
│   ├── 08_subtype_clustering/         # Phase 3C
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── clustering.py              # GMM, spectral, hierarchical
│   │   ├── stability.py               # Bootstrap stability
│   │   ├── characterization.py        # Subtype profiles
│   │   ├── validation.py              # Research integrity (confounds, negative controls, provenance)
│   │   └── tests/
│   │
│   ├── 09_symbolic_rules/             # Phase 4A
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── rule_engine.py             # Rule evaluation
│   │   ├── biological_rules.py        # Curated rules
│   │   ├── explanation.py             # Reasoning chains
│   │   └── tests/
│   │
│   ├── 10_neurosymbolic/              # Phase 4B
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── integration.py             # Neural + symbolic
│   │   ├── combiner.py                # Learned combination
│   │   └── tests/
│   │
│   ├── 11_therapeutic_hypotheses/     # Phase 4C
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── pathway_drug_mapping.py    # Pathway → drug links
│   │   ├── ranking.py                 # Hypothesis ranking
│   │   ├── evidence.py                # Evidence scoring
│   │   └── tests/
│   │
│   └── 12_causal_inference/           # Phase 5 (Advanced)
│       ├── README.md
│       ├── __init__.py
│       ├── causal_graph.py            # Structural causal model
│       ├── do_calculus.py             # Intervention reasoning
│       ├── counterfactuals.py         # Counterfactual queries
│       ├── effect_estimation.py       # Direct/indirect effects
│       └── tests/
│
├── pipelines/                          # End-to-end workflows
│   ├── README.md
│   ├── subtype_discovery.py           # Full subtype pipeline
│   ├── therapeutic_hypothesis.py      # Full hypothesis pipeline
│   └── validation.py                  # Cross-cohort validation
│
├── configs/                            # Configuration files
│   ├── default.yaml
│   ├── small_test.yaml
│   └── production.yaml
│
├── data/                               # Data directory (gitignored)
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── notebooks/                          # Exploration notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_validation.ipynb
│   └── 03_results_analysis.ipynb
│
└── scripts/                            # Utility scripts
    ├── download_databases.sh
    ├── setup_environment.sh
    └── run_tests.sh
```

---

## Implementation Phases

### Phase 1: Data Foundation (Modules 01-02)

**Goal**: Load and process genetic data independently of ML components.

#### Module 01: Data Loaders (Session 1)

**Scope**: ~400 lines
**Dependencies**: None (external data only)
**Interface Contract**:

```python
# Input: File paths
# Output: Standardized data structures

class VCFLoader:
    def load(self, vcf_path: str) -> VariantDataset
    def validate(self, dataset: VariantDataset) -> ValidationReport

class PathwayLoader:
    def load_go(self, obo_path: str) -> PathwayDatabase
    def load_reactome(self, gmt_path: str) -> PathwayDatabase
    def merge(self, databases: List[PathwayDatabase]) -> PathwayDatabase

class ExpressionLoader:
    """Load developmental expression data (BrainSpan)."""
    def load_brainspan(self, data_dir: str) -> DevelopmentalExpression
    def get_expression_by_stage(self, gene_id: str, stage: str) -> float
    def get_prenatal_expressed_genes(self, threshold: float = 1.0) -> List[str]

class SingleCellLoader:
    """Load single-cell atlas data (Allen Brain, other cortical atlases)."""
    def load_allen_brain(self, h5ad_path: str) -> SingleCellAtlas
    def get_cell_type_markers(self, cell_type: str) -> List[str]
    def get_expression_by_cell_type(self, gene_id: str) -> Dict[str, float]

class ConstraintLoader:
    """Load gene constraint scores (gnomAD pLI/LOEUF, SFARI)."""
    def load_gnomad_constraints(self, tsv_path: str) -> GeneConstraints
    def load_sfari_genes(self, csv_path: str) -> SFARIGenes
    def get_constrained_genes(self, pli_threshold: float = 0.9) -> List[str]

# Data structures defined in this module
@dataclass
class Variant:
    chrom: str
    pos: int
    ref: str
    alt: str
    sample_id: str
    genotype: str
    quality: float

@dataclass
class VariantDataset:
    variants: List[Variant]
    samples: List[str]
    metadata: Dict

@dataclass
class DevelopmentalExpression:
    """BrainSpan developmental expression matrix."""
    genes: List[str]
    stages: List[str]  # e.g., ["8pcw", "12pcw", ..., "40y"]
    regions: List[str]  # Brain regions
    expression: np.ndarray  # shape: (n_genes, n_stages, n_regions)

    def get_prenatal_expression(self, gene_id: str) -> np.ndarray
    def get_cortical_expression(self, gene_id: str, stage: str) -> float

@dataclass
class SingleCellAtlas:
    """Single-cell expression atlas."""
    genes: List[str]
    cell_types: List[str]
    expression: np.ndarray  # shape: (n_genes, n_cell_types)
    cell_type_hierarchy: Dict[str, List[str]]  # e.g., {"neuron": ["excitatory", "inhibitory"]}

    def get_cell_type_specific_genes(self, cell_type: str, fold_change: float = 2.0) -> List[str]

@dataclass
class GeneConstraints:
    """Gene constraint scores from gnomAD."""
    gene_ids: List[str]
    pli_scores: Dict[str, float]  # Probability of LoF intolerance
    loeuf_scores: Dict[str, float]  # Loss-of-function observed/expected upper bound
    mis_z_scores: Dict[str, float]  # Missense Z-scores

    def is_constrained(self, gene_id: str, pli_threshold: float = 0.9) -> bool

@dataclass
class SFARIGenes:
    """SFARI autism gene database."""
    gene_ids: List[str]
    scores: Dict[str, int]  # 1 = high confidence, 2 = strong candidate, 3 = suggestive
    syndromic: Dict[str, bool]
    evidence: Dict[str, List[str]]
```

**Standalone Test**:
```bash
python -m modules.01_data_loaders.tests.test_vcf_loader
# Should work with sample VCF, no other modules needed
```

---

#### Module 02: Variant Processing (Session 2)

**Scope**: ~500 lines
**Dependencies**: Module 01 (data structures only)
**Interface Contract**:

```python
# Input: VariantDataset from Module 01
# Output: Gene burden scores

class QCFilter:
    def filter_variants(self, dataset: VariantDataset, config: QCConfig) -> VariantDataset
    def filter_samples(self, dataset: VariantDataset, config: QCConfig) -> VariantDataset

class GeneBurdenCalculator:
    def compute(self, dataset: VariantDataset, weights: WeightConfig) -> GeneBurdenMatrix

@dataclass
class GeneBurdenMatrix:
    samples: List[str]
    genes: List[str]
    scores: np.ndarray  # shape: (n_samples, n_genes)

    def to_sparse(self) -> scipy.sparse.csr_matrix
    def get_sample(self, sample_id: str) -> Dict[str, float]
```

---

### Phase 2: Knowledge Representation (Modules 03-05)

**Goal**: Build knowledge graph and gene embeddings independently.

#### Module 03: Knowledge Graph (Session 3)

**Scope**: ~600 lines
**Dependencies**: Module 01 (PathwayLoader only)
**Interface Contract**:

```python
# Input: Pathway databases, PPI networks
# Output: Heterogeneous knowledge graph

@dataclass
class NodeType(Enum):
    GENE = "gene"
    PATHWAY = "pathway"
    GO_TERM = "go_term"
    CELL_TYPE = "cell_type"
    DRUG = "drug"

@dataclass
class EdgeType(Enum):
    GENE_INTERACTS = "gene_interacts_gene"
    GENE_IN_PATHWAY = "gene_in_pathway"
    GENE_HAS_GO = "gene_has_go"
    # ... etc

class KnowledgeGraphBuilder:
    def add_genes(self, gene_list: List[str]) -> None
    def add_pathways(self, pathway_db: PathwayDatabase) -> None
    def add_ppi(self, ppi_network: PPINetwork) -> None
    def build(self) -> KnowledgeGraph

class KnowledgeGraph:
    def get_neighbors(self, node_id: str, edge_type: EdgeType) -> List[str]
    def to_dgl(self) -> dgl.DGLGraph
    def to_pyg(self) -> torch_geometric.data.HeteroData
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

**Standalone Test**:
```bash
python -m modules.03_knowledge_graph.tests.test_builder
# Build small test graph, verify structure
```

---

#### Module 04: Graph Embeddings (Session 4)

**Scope**: ~500 lines
**Dependencies**: Module 03 (KnowledgeGraph only)
**Interface Contract**:

```python
# Input: KnowledgeGraph
# Output: Node embeddings

class TransEModel:
    def __init__(self, embedding_dim: int, margin: float)
    def train(self, graph: KnowledgeGraph, epochs: int) -> TrainingHistory
    def get_embeddings(self) -> Dict[str, np.ndarray]

class RotatEModel:
    # Same interface as TransE

class EmbeddingTrainer:
    def train(self, model: BaseEmbeddingModel, graph: KnowledgeGraph) -> None
    def evaluate(self, model: BaseEmbeddingModel, test_edges: List) -> Metrics

@dataclass
class NodeEmbeddings:
    node_ids: List[str]
    embeddings: np.ndarray  # shape: (n_nodes, embedding_dim)

    def get(self, node_id: str) -> np.ndarray
    def most_similar(self, node_id: str, k: int) -> List[Tuple[str, float]]
    def save(self, path: str) -> None
```

---

#### Module 05: Pretrained Embeddings (Sessions 5-5B)

**Scope**: ~600 lines (split into 2 sessions if fine-tuning implemented)
**Dependencies**: Module 01 (expression data for fine-tuning context)
**Interface Contract**:

```python
# Input: Gene list + optional fine-tuning data
# Output: Pretrained or fine-tuned embeddings

from enum import Enum

class ExtractionMode(Enum):
    FROZEN = "frozen"      # Use pretrained weights as-is
    FINE_TUNED = "fine_tuned"  # Fine-tune on autism-specific data

class GeneformerExtractor:
    """
    Extract gene embeddings from Geneformer.

    Geneformer learns gene representations from single-cell expression data.
    Can be fine-tuned on brain-specific or autism-specific data.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = self.load_model(model_path)
        self.device = device

    def extract(self, gene_ids: List[str], mode: ExtractionMode = ExtractionMode.FROZEN) -> NodeEmbeddings:
        """Extract embeddings for given genes."""

    def fine_tune(self,
                  training_data: SingleCellDataset,
                  config: FineTuneConfig) -> 'GeneformerExtractor':
        """
        Fine-tune Geneformer on autism-specific single-cell data.

        Recommended datasets for fine-tuning:
        - Brain organoid single-cell data
        - Postmortem ASD brain tissue
        - Developing cortex (fetal) data
        """

    def extract_with_context(self,
                              gene_ids: List[str],
                              cell_type_context: str) -> NodeEmbeddings:
        """
        Extract cell-type-contextualized embeddings.

        Different embeddings for same gene in different cell types.
        """

class ESM2Extractor:
    """
    Extract protein embeddings from ESM-2.

    ESM-2 learns protein representations from evolutionary sequences.
    Captures structural and functional information.
    """

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        self.model = self.load_model(model_name)

    def extract(self, protein_sequences: Dict[str, str]) -> NodeEmbeddings:
        """Extract embeddings via mean pooling over sequence."""

    def extract_with_variant(self,
                              protein_sequences: Dict[str, str],
                              variants: Dict[str, List[Variant]]) -> NodeEmbeddings:
        """
        Extract embeddings considering variant effects.

        For missense variants, compute embedding difference between
        wild-type and mutant sequence to capture variant impact.
        """

    def predict_variant_effect(self,
                                wt_sequence: str,
                                variant: Variant) -> VariantEffect:
        """
        Use ESM-2 log-likelihood ratio to predict variant pathogenicity.
        """

class LiteratureEmbedder:
    """
    Extract gene embeddings from biomedical literature.

    Uses PubMedBERT/BioGPT to embed gene descriptions and associations.
    """

    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        self.model = self.load_model(model_name)

    def extract_from_descriptions(self, gene_descriptions: Dict[str, str]) -> NodeEmbeddings:
        """Embed gene functional descriptions."""

    def extract_from_abstracts(self, gene_pmids: Dict[str, List[str]]) -> NodeEmbeddings:
        """Embed aggregated literature about each gene."""

class EmbeddingFusion:
    """
    Fuse multiple embedding sources into unified gene representation.
    """

    def __init__(self, fusion_method: str = "learned"):
        self.fusion_method = fusion_method

    def fuse(self,
             kg_embeddings: NodeEmbeddings,
             geneformer_embeddings: NodeEmbeddings,
             esm2_embeddings: NodeEmbeddings,
             literature_embeddings: Optional[NodeEmbeddings] = None) -> NodeEmbeddings:
        """
        Fuse embeddings from multiple sources.

        Methods:
        - "concat": Simple concatenation
        - "weighted_sum": Learned weights per source
        - "attention": Cross-attention fusion
        - "learned": MLP-based learned fusion
        """

    def fuse_with_learned_weights(self,
                                   embeddings: List[NodeEmbeddings],
                                   labels: Optional[np.ndarray] = None) -> NodeEmbeddings:
        """
        Learn optimal fusion weights from downstream task.

        If labels provided (e.g., SFARI gene labels), optimize weights
        to maximize predictive performance.
        """

@dataclass
class FineTuneConfig:
    """Configuration for foundation model fine-tuning."""
    learning_rate: float = 1e-5
    epochs: int = 10
    batch_size: int = 32
    warmup_steps: int = 500
    weight_decay: float = 0.01
    # Multi-task fine-tuning
    tasks: List[str] = None  # e.g., ["pathway_prediction", "phenotype_prediction"]
    task_weights: Dict[str, float] = None
    # Adapter-based fine-tuning (parameter efficient)
    use_adapters: bool = True
    adapter_dim: int = 64

@dataclass
class VariantEffect:
    """Predicted effect of a genetic variant."""
    gene_id: str
    variant: Variant
    pathogenicity_score: float  # 0-1
    embedding_shift: np.ndarray  # Change in embedding space
    confidence: float

class MultiTaskFineTuner:
    """
    Fine-tune foundation models on multiple autism-relevant tasks.

    Tasks:
    1. Pathway membership prediction
    2. SFARI gene classification
    3. Phenotype association prediction
    4. Gene-gene interaction prediction
    """

    def __init__(self, base_model: nn.Module, tasks: List[str]):
        self.base_model = base_model
        self.task_heads = self.create_task_heads(tasks)

    def fine_tune(self,
                  datasets: Dict[str, Dataset],
                  config: FineTuneConfig) -> nn.Module:
        """Multi-task fine-tuning with task-specific heads."""

    def extract_fine_tuned_embeddings(self, gene_ids: List[str]) -> NodeEmbeddings:
        """Extract embeddings from fine-tuned model."""
```

**Session 5A: Extraction (Frozen)**
- Implement GeneformerExtractor, ESM2Extractor, LiteratureEmbedder
- Basic fusion methods

**Session 5B: Fine-tuning (Optional, Advanced)**
- Implement FineTuneConfig, MultiTaskFineTuner
- Adapter-based fine-tuning for parameter efficiency
- Variant-aware ESM-2 extraction

**Note**: This module can be implemented in parallel with Module 04. Session 5B (fine-tuning) is optional for v1.

---

### Phase 3: Neural Models (Modules 06-08)

**Goal**: Implement GNN and clustering, building on embeddings.

#### Module 06: Ontology-Aware GNN (Sessions 6-7)

**Scope**: ~700 lines (split into 2 sessions)
**Dependencies**: Module 03 (graph), Module 04/05 (embeddings)

**Session 6A: Layers and Attention**
```python
# layers.py + attention.py (~350 lines)

class EdgeTypeTransform(nn.Module):
    """Separate transformation per edge type."""
    def forward(self, x: Tensor, edge_type: str) -> Tensor

class BiologicalAttention(nn.Module):
    """Attention weighted by biological priors."""
    def __init__(self, hidden_dim: int, prior_types: List[str])
    def forward(self,
                query: Tensor,
                key: Tensor,
                bio_priors: Dict[str, Tensor]) -> Tensor

class HierarchicalAggregator(nn.Module):
    """Aggregate following ontology hierarchy."""
    def forward(self, messages: Tensor, hierarchy: Dict) -> Tensor
```

**Session 6B: Full Model**
```python
# model.py (~350 lines)

class OntologyAwareGNN(nn.Module):
    def __init__(self, config: GNNConfig)
    def forward(self,
                graph: DGLGraph,
                node_features: Tensor,
                bio_priors: Dict[str, Tensor]) -> Tensor

class GNNConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    edge_types: List[str]
    attention_heads: int = 4
    dropout: float = 0.1
```

---

#### Module 07: Pathway Scoring (Session 8)

**Scope**: ~450 lines
**Dependencies**: Module 02 (gene burdens), Module 06 (GNN, optional)
**Interface Contract**:

```python
# Input: Gene burdens + pathway definitions
# Output: Pathway scores per individual

class PathwayAggregator:
    def aggregate(self,
                  gene_burdens: GeneBurdenMatrix,
                  pathway_db: PathwayDatabase,
                  method: str = "weighted_sum") -> PathwayScoreMatrix

class NetworkPropagator:
    def __init__(self, graph: KnowledgeGraph, restart_prob: float = 0.5)
    def propagate(self, gene_scores: Dict[str, float]) -> Dict[str, float]

class PathwayScoreNormalizer:
    def normalize(self,
                  scores: PathwayScoreMatrix,
                  method: str = "zscore") -> PathwayScoreMatrix

@dataclass
class PathwayScoreMatrix:
    samples: List[str]
    pathways: List[str]
    scores: np.ndarray
    contributing_genes: Dict[str, Dict[str, List[str]]]  # sample -> pathway -> genes
```

---

#### Module 08: Subtype Clustering (Session 9)

**Scope**: ~500 lines
**Dependencies**: Module 07 (pathway scores)
**Interface Contract**:

```python
# Input: Pathway scores
# Output: Subtype assignments

class SubtypeClusterer:
    def fit(self,
            pathway_scores: PathwayScoreMatrix,
            method: str = "gmm",
            n_clusters: int = None) -> ClusteringResult

    def predict(self, new_scores: PathwayScoreMatrix) -> np.ndarray

class StabilityAnalyzer:
    def bootstrap_stability(self,
                            pathway_scores: PathwayScoreMatrix,
                            n_bootstrap: int = 100) -> StabilityReport

class SubtypeCharacterizer:
    def characterize(self,
                     result: ClusteringResult,
                     pathway_scores: PathwayScoreMatrix) -> List[SubtypeProfile]

@dataclass
class ClusteringResult:
    labels: np.ndarray
    probabilities: np.ndarray
    n_clusters: int
    model: Any  # Fitted model for prediction

@dataclass
class SubtypeProfile:
    subtype_id: int
    size: int
    characteristic_pathways: List[Tuple[str, float]]
    mean_profile: Dict[str, float]

# Research Integrity Validation (validation.py)
class ConfoundAnalyzer:
    def test_cluster_confound_alignment(
        self, cluster_labels: np.ndarray,
        confounds: Dict[str, np.ndarray]) -> ConfoundReport
    def compute_confound_association(
        self, cluster_labels: np.ndarray,
        confound_values: np.ndarray) -> Tuple[float, float, float]

class NegativeControlRunner:
    def permutation_test(
        self, data: np.ndarray, clusterer: SubtypeClusterer) -> PermutationResult
    def random_geneset_baseline(
        self, data: np.ndarray, clusterer: SubtypeClusterer) -> Dict[str, Any]
    def run_full_negative_control(
        self, data: np.ndarray, clusterer: SubtypeClusterer) -> NegativeControlReport

@dataclass
class ProvenanceRecord:
    reference_genome: str
    annotation_version: str
    pathway_db_versions: Dict[str, str]
    pipeline_version: str
    timestamp: datetime
    def validate_compatibility(self, other: 'ProvenanceRecord') -> Tuple[bool, List[str]]
```

---

### Phase 4: Symbolic Reasoning (Modules 09-11)

**Goal**: Add explainability and therapeutic hypotheses.

#### Module 09: Symbolic Rules (Session 10)

**Scope**: ~600 lines
**Dependencies**: Module 01 (constraint/expression data), Module 02 (gene data), Module 07 (pathway scores)
**Interface Contract**:

```python
# Input: Gene/pathway data + biological context
# Output: Rule-based inferences with explanations

@dataclass
class Condition:
    """A single condition in a rule."""
    predicate: str  # e.g., "has_lof_variant", "is_constrained", "expressed_in"
    arguments: Dict[str, Any]
    negated: bool = False

@dataclass
class Conclusion:
    """The conclusion of a fired rule."""
    type: str  # e.g., "pathway_disruption", "subtype_indicator", "therapeutic_candidate"
    attributes: Dict[str, Any]
    confidence_modifier: float = 1.0

@dataclass
class Rule:
    id: str
    name: str
    description: str
    conditions: List[Condition]
    conclusion: Conclusion
    base_confidence: float
    evidence_sources: List[str]  # Literature references

class RuleEngine:
    def __init__(self, rules: List[Rule], biological_context: BiologicalContext)
    def evaluate(self, individual_data: IndividualData) -> List[FiredRule]
    def evaluate_batch(self, cohort_data: List[IndividualData]) -> Dict[str, List[FiredRule]]
    def explain(self, fired_rule: FiredRule) -> str
    def get_reasoning_chain(self, fired_rules: List[FiredRule]) -> ReasoningChain

class BiologicalRules:
    """Curated autism-specific biological rules (R1-R6 from domain analysis)."""

    @staticmethod
    def R1_constrained_lof_developing_cortex() -> Rule:
        """
        R1: LoF in constrained gene expressed in developing cortex
        → High-confidence pathway disruption

        Conditions:
        - Individual has loss-of-function variant in gene G
        - Gene G has pLI > 0.9 (highly constrained)
        - Gene G is expressed in developing cortex (BrainSpan prenatal)

        Conclusion: High-confidence pathway disruption for pathways containing G
        """
        return Rule(
            id="R1",
            name="Constrained LoF in Developing Cortex",
            description="Loss-of-function variant in constrained gene with prenatal cortical expression",
            conditions=[
                Condition("has_variant", {"gene": "G", "variant_type": "loss_of_function"}),
                Condition("gene_constraint", {"gene": "G", "pli_threshold": 0.9}),
                Condition("expressed_in", {"gene": "G", "tissue": "cortex", "stage": "prenatal"})
            ],
            conclusion=Conclusion("pathway_disruption", {"confidence": "high", "mechanism": "haploinsufficiency"}),
            base_confidence=0.9,
            evidence_sources=["gnomAD constraint", "BrainSpan expression"]
        )

    @staticmethod
    def R2_pathway_convergence() -> Rule:
        """
        R2: Multiple hits in same pathway (≥2 genes)
        → Pathway-level convergence signal

        Conditions:
        - Individual has damaging variants in genes G1, G2, ...
        - G1 and G2 are both members of pathway P
        - Hits are in distinct genes (not compound het in same gene)

        Conclusion: Strong pathway convergence evidence
        """
        return Rule(
            id="R2",
            name="Pathway Convergence",
            description="Multiple independent hits converging on same biological pathway",
            conditions=[
                Condition("has_multiple_hits", {"min_genes": 2, "pathway": "P"}),
                Condition("hits_are_independent", {"pathway": "P"})
            ],
            conclusion=Conclusion("pathway_convergence", {"strength": "strong"}),
            base_confidence=0.85,
            evidence_sources=["Pathway membership", "Variant independence"]
        )

    @staticmethod
    def R3_chd8_cascade() -> Rule:
        """
        R3: Disruption in CHD8 or its regulatory targets
        → Chromatin regulation cascade

        Conditions:
        - Individual has damaging variant in CHD8 OR
        - Individual has damaging variant in known CHD8 target gene

        Conclusion: Chromatin regulation cascade disruption (known ASD mechanism)
        """
        return Rule(
            id="R3",
            name="CHD8 Chromatin Cascade",
            description="Disruption in CHD8 regulatory network affecting chromatin remodeling",
            conditions=[
                Condition("has_variant", {"gene": "CHD8", "variant_type": "damaging"}, negated=False),
                # OR condition handled by rule engine
                Condition("is_chd8_target", {"gene": "G"})
            ],
            conclusion=Conclusion("pathway_disruption", {
                "pathway": "chromatin_regulation",
                "mechanism": "CHD8_cascade",
                "subtype_indicator": "chromatin_remodeling"
            }),
            base_confidence=0.88,
            evidence_sources=["Cotney et al. 2015", "Sugathan et al. 2014"]
        )

    @staticmethod
    def R4_synaptic_excitatory() -> Rule:
        """
        R4: Synaptic gene hit + expression in excitatory neurons
        → Synaptic subtype indicator

        Conditions:
        - Individual has damaging variant in synaptic gene (SynGO annotated)
        - Gene is preferentially expressed in excitatory neurons

        Conclusion: Synaptic dysfunction subtype indicator
        """
        return Rule(
            id="R4",
            name="Synaptic Excitatory Disruption",
            description="Synaptic gene disruption with excitatory neuron expression pattern",
            conditions=[
                Condition("has_variant", {"gene": "G", "variant_type": "damaging"}),
                Condition("is_synaptic_gene", {"gene": "G", "ontology": "SynGO"}),
                Condition("cell_type_expression", {"gene": "G", "cell_type": "excitatory_neuron", "enriched": True})
            ],
            conclusion=Conclusion("subtype_indicator", {
                "subtype": "synaptic_dysfunction",
                "cell_type": "excitatory",
                "mechanism": "synaptic_transmission"
            }),
            base_confidence=0.82,
            evidence_sources=["SynGO", "Single-cell expression atlas"]
        )

    @staticmethod
    def R5_compensatory_paralog() -> Rule:
        """
        R5: Paralog intact + expressed
        → Potential compensation (reduced penetrance)

        Conditions:
        - Individual has loss-of-function variant in gene G
        - Gene G has a paralog P
        - Paralog P is NOT disrupted in this individual
        - Paralog P is highly expressed in relevant tissue

        Conclusion: Potential functional compensation (modifier of effect)
        """
        return Rule(
            id="R5",
            name="Paralog Compensation",
            description="Intact expressed paralog may compensate for disrupted gene",
            conditions=[
                Condition("has_variant", {"gene": "G", "variant_type": "loss_of_function"}),
                Condition("has_paralog", {"gene": "G", "paralog": "P"}),
                Condition("has_variant", {"gene": "P", "variant_type": "damaging"}, negated=True),
                Condition("expressed_in", {"gene": "P", "tissue": "brain", "level": "high"})
            ],
            conclusion=Conclusion("effect_modifier", {
                "modifier_type": "compensation",
                "confidence_reduction": 0.3,
                "mechanism": "paralog_redundancy"
            }),
            base_confidence=0.7,
            evidence_sources=["Paralog databases", "Expression data"]
        )

    @staticmethod
    def R6_drug_pathway_target() -> Rule:
        """
        R6: Drug targets disrupted pathway
        → Therapeutic hypothesis candidate

        Conditions:
        - Pathway P is disrupted in individual (from pathway scoring)
        - Drug D has known target gene T
        - Gene T is a member of pathway P
        - Drug D mechanism aligns with pathway biology

        Conclusion: Therapeutic hypothesis candidate
        """
        return Rule(
            id="R6",
            name="Therapeutic Pathway Target",
            description="Drug targets gene within disrupted pathway",
            conditions=[
                Condition("pathway_disrupted", {"pathway": "P", "score_threshold": 2.0}),
                Condition("drug_targets", {"drug": "D", "target": "T"}),
                Condition("gene_in_pathway", {"gene": "T", "pathway": "P"}),
                Condition("mechanism_alignment", {"drug": "D", "pathway": "P"})
            ],
            conclusion=Conclusion("therapeutic_hypothesis", {
                "drug": "D",
                "target_pathway": "P",
                "requires_validation": True
            }),
            base_confidence=0.6,
            evidence_sources=["DrugBank", "Pathway databases", "Mechanism annotations"]
        )

    @staticmethod
    def get_all_rules() -> List[Rule]:
        """Return all curated biological rules."""
        return [
            BiologicalRules.R1_constrained_lof_developing_cortex(),
            BiologicalRules.R2_pathway_convergence(),
            BiologicalRules.R3_chd8_cascade(),
            BiologicalRules.R4_synaptic_excitatory(),
            BiologicalRules.R5_compensatory_paralog(),
            BiologicalRules.R6_drug_pathway_target()
        ]

@dataclass
class FiredRule:
    rule: Rule
    bindings: Dict[str, Any]  # Variable assignments that satisfied conditions
    confidence: float  # Final confidence after modifiers
    explanation: str
    evidence: Dict[str, Any]  # Supporting data for this firing

@dataclass
class ReasoningChain:
    """Chain of reasoning from variants to conclusions."""
    individual_id: str
    fired_rules: List[FiredRule]
    pathway_conclusions: Dict[str, float]
    subtype_indicators: List[str]
    therapeutic_hypotheses: List[Dict]
    explanation_text: str

@dataclass
class BiologicalContext:
    """Biological reference data needed for rule evaluation."""
    gene_constraints: GeneConstraints
    developmental_expression: DevelopmentalExpression
    single_cell_atlas: SingleCellAtlas
    sfari_genes: SFARIGenes
    paralog_map: Dict[str, List[str]]
    chd8_targets: List[str]
    syngo_genes: List[str]
    drug_targets: Dict[str, List[str]]
```

---

#### Module 10: Neuro-Symbolic Integration (Session 11)

**Scope**: ~400 lines
**Dependencies**: Module 06 (GNN), Module 09 (rules)
**Interface Contract**:

```python
# Input: Neural predictions + symbolic inferences
# Output: Combined predictions with explanations

class NeuroSymbolicModel:
    def __init__(self,
                 neural_model: OntologyAwareGNN,
                 rule_engine: RuleEngine)

    def forward(self, individual_data: IndividualData) -> NeuroSymbolicOutput

class LearnedCombiner(nn.Module):
    """Learn to weight neural vs symbolic predictions."""
    def forward(self,
                neural_scores: Tensor,
                symbolic_scores: Tensor) -> Tensor

@dataclass
class NeuroSymbolicOutput:
    predictions: Dict[str, float]
    neural_contribution: Dict[str, float]
    symbolic_contribution: Dict[str, float]
    fired_rules: List[FiredRule]
    explanation: str
```

---

#### Module 11: Therapeutic Hypotheses (Session 12)

**Scope**: ~450 lines
**Dependencies**: Module 07 (pathway scores), Module 09 (rules)
**Interface Contract**:

```python
# Input: Disrupted pathways
# Output: Ranked therapeutic hypotheses

class PathwayDrugMapper:
    def __init__(self, drug_target_db: DrugTargetDatabase)
    def map(self, pathway_id: str) -> List[DrugCandidate]

class HypothesisRanker:
    def rank(self,
             pathway_scores: Dict[str, float],
             drug_candidates: List[DrugCandidate]) -> List[TherapeuticHypothesis]

class EvidenceScorer:
    def score(self, hypothesis: TherapeuticHypothesis) -> EvidenceScore

@dataclass
class TherapeuticHypothesis:
    drug_id: str
    drug_name: str
    target_pathway: str
    mechanism: str
    score: float
    evidence: EvidenceScore
    explanation: str

@dataclass
class EvidenceScore:
    biological_plausibility: float
    mechanistic_alignment: float
    literature_support: float
    safety_flags: List[str]
    overall: float
```

---

## Session-by-Session Implementation Guide

### How to Use This Guide

For each session:
1. **Read only the relevant module README**
2. **Implement against the interface contract**
3. **Write tests that work standalone**
4. **Document any deviations**

### Session Checklist Template

```markdown
## Session N: Module XX

### Pre-Session
- [ ] Read module README
- [ ] Review interface contract
- [ ] Check dependencies are available

### Implementation
- [ ] Create module directory structure
- [ ] Implement core classes
- [ ] Write unit tests
- [ ] Verify standalone execution

### Post-Session
- [ ] Update module README with any changes
- [ ] Document API in docstrings
- [ ] Commit with descriptive message
```

---

## Detailed Session Plans

### Session 1: Data Loaders

**Context Needed**:
- This README section
- VCF format specification (brief)
- GO OBO format specification (brief)

**Files to Create**:
```
modules/01_data_loaders/
├── README.md              # Module documentation
├── __init__.py            # Exports
├── vcf_loader.py          # ~150 lines
├── annotation_loader.py   # ~100 lines
├── pathway_loader.py      # ~150 lines
├── expression_loader.py   # ~150 lines (BrainSpan)
├── single_cell_loader.py  # ~150 lines (Allen Brain)
├── constraint_loader.py   # ~100 lines (gnomAD, SFARI)
└── tests/
    ├── test_vcf_loader.py
    ├── test_pathway_loader.py
    ├── test_expression_loader.py
    └── test_constraint_loader.py
```

**Success Criteria**:
```bash
# These should pass without any other modules
python -m pytest modules/01_data_loaders/tests/ -v
```

---

### Session 2: Variant Processing

**Context Needed**:
- Module 01 data structures (copy interface only)
- QC filter criteria (brief list)
- Variant weighting schemes (brief)

**Files to Create**:
```
modules/02_variant_processing/
├── README.md
├── __init__.py
├── qc_filters.py       # ~150 lines
├── annotation.py       # ~150 lines
├── gene_burden.py      # ~200 lines
└── tests/
    └── test_gene_burden.py
```

**Success Criteria**:
```bash
python -m pytest modules/02_variant_processing/tests/ -v
```

---

### Session 3: Knowledge Graph

**Context Needed**:
- Module 01 PathwayDatabase structure
- Node/edge type definitions (from this doc)
- PPI network format

**Files to Create**:
```
modules/03_knowledge_graph/
├── README.md
├── __init__.py
├── schema.py           # ~100 lines
├── builder.py          # ~300 lines
├── exporters.py        # ~200 lines
└── tests/
    └── test_builder.py
```

---

### Session 4: Graph Embeddings

**Context Needed**:
- Module 03 KnowledgeGraph interface
- TransE/RotatE algorithm descriptions (brief)

**Files to Create**:
```
modules/04_graph_embeddings/
├── README.md
├── __init__.py
├── base.py             # ~100 lines (base class)
├── transe.py           # ~150 lines
├── rotate.py           # ~150 lines
├── trainer.py          # ~100 lines
└── tests/
    └── test_embeddings.py
```

---

### Sessions 5-5B: Pretrained Embeddings

**Session 5A Context Needed**:
- Geneformer API documentation
- ESM-2 API documentation
- Embedding fusion approach

**Session 5B Context Needed** (Optional):
- Adapter-based fine-tuning (LoRA, etc.)
- Multi-task learning setup
- Autism-specific datasets (SFARI, brain single-cell)

**Files to Create**:
```
modules/05_pretrained_embeddings/
├── README.md
├── __init__.py
├── geneformer_extractor.py  # ~200 lines
├── esm2_extractor.py        # ~200 lines
├── literature_embedder.py   # ~100 lines
├── fusion.py                # ~150 lines
├── fine_tuning.py           # ~150 lines (Session 5B, optional)
└── tests/
    ├── test_extractors.py
    ├── test_fusion.py
    └── test_fine_tuning.py
```

---

### Sessions 6-7: Ontology-Aware GNN

**Session 6 Context**:
- PyTorch basics
- Attention mechanism description
- Biological prior definitions

**Session 7 Context**:
- Session 6 outputs (layers.py, attention.py)
- Full model architecture diagram

**Files to Create**:
```
modules/06_ontology_gnn/
├── README.md
├── __init__.py
├── layers.py           # Session 6: ~200 lines
├── attention.py        # Session 6: ~150 lines
├── model.py            # Session 7: ~350 lines
└── tests/
    ├── test_layers.py
    └── test_model.py
```

---

### Session 8: Pathway Scoring

**Context Needed**:
- Module 02 GeneBurdenMatrix interface
- Aggregation methods (weighted sum, max, etc.)
- Network propagation algorithm

**Files to Create**:
```
modules/07_pathway_scoring/
├── README.md
├── __init__.py
├── aggregation.py          # ~150 lines
├── network_propagation.py  # ~200 lines
├── normalization.py        # ~100 lines
└── tests/
    └── test_scoring.py
```

---

### Session 9: Subtype Clustering

**Context Needed**:
- Module 07 PathwayScoreMatrix interface
- Clustering algorithms (GMM, spectral)
- Stability analysis approach

**Files to Create**:
```
modules/08_subtype_clustering/
├── README.md
├── __init__.py
├── clustering.py           # ~200 lines
├── stability.py            # ~150 lines
├── characterization.py     # ~150 lines
└── tests/
    └── test_clustering.py
```

---

### Session 10: Symbolic Rules

**Context Needed**:
- Rule structure definition
- Curated biological rules (from analysis doc)
- Explanation generation approach

**Files to Create**:
```
modules/09_symbolic_rules/
├── README.md
├── __init__.py
├── rule_engine.py          # ~250 lines
├── biological_rules.py     # ~300 lines (R1-R6 implementations)
├── conditions.py           # ~100 lines (condition evaluators)
├── explanation.py          # ~150 lines (reasoning chain generation)
└── tests/
    ├── test_rules.py
    ├── test_rule_engine.py
    └── test_explanations.py
```

---

### Session 11: Neuro-Symbolic Integration

**Context Needed**:
- Module 06 GNN interface
- Module 09 RuleEngine interface
- Combination strategies

**Files to Create**:
```
modules/10_neurosymbolic/
├── README.md
├── __init__.py
├── integration.py          # ~200 lines
├── combiner.py             # ~200 lines
└── tests/
    └── test_integration.py
```

---

### Session 12: Therapeutic Hypotheses

**Context Needed**:
- Module 07 pathway scores interface
- Drug-target database format
- Ranking algorithm

**Files to Create**:
```
modules/11_therapeutic_hypotheses/
├── README.md
├── __init__.py
├── pathway_drug_mapping.py # ~150 lines
├── ranking.py              # ~200 lines
├── evidence.py             # ~100 lines
└── tests/
    └── test_hypotheses.py
```

---

### Phase 5: Causal Inference (Module 12)

**Goal**: Enable causal reasoning about genetic mechanisms and intervention effects.

#### Module 12: Causal Inference (Sessions 13-14)

**Scope**: ~700 lines (split into 2 sessions)
**Dependencies**: Module 03 (knowledge graph), Module 07 (pathway scores), Module 09 (rules)
**Interface Contract**:

```python
# Input: Knowledge graph + pathway scores + variant data
# Output: Causal queries, intervention reasoning, counterfactuals

from enum import Enum
from typing import Optional

class CausalNodeType(Enum):
    VARIANT = "variant"
    GENE_FUNCTION = "gene_function"
    PATHWAY = "pathway"
    CIRCUIT = "circuit"
    PHENOTYPE = "phenotype"
    CONFOUNDER = "confounder"

class CausalEdgeType(Enum):
    CAUSES = "causes"
    MEDIATES = "mediates"
    CONFOUNDS = "confounds"
    MODIFIES = "modifies"

@dataclass
class CausalNode:
    id: str
    node_type: CausalNodeType
    observed: bool
    value: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class CausalEdge:
    source: str
    target: str
    edge_type: CausalEdgeType
    strength: float  # Estimated causal effect strength
    mechanism: str  # Biological mechanism description

class StructuralCausalModel:
    """
    Structural Causal Model for ASD genetics.

    Encodes the causal chain:
    Genetic Variants → Gene Function Disruption → Pathway Perturbation
                    → Circuit-Level Effects → Behavioral Phenotype

    With explicit confounders:
    - Ancestry
    - Batch effects
    - Ascertainment bias
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.structural_equations: Dict[str, Callable] = {}

    def add_node(self, node: CausalNode) -> None
    def add_edge(self, edge: CausalEdge) -> None
    def set_structural_equation(self, node_id: str, equation: Callable) -> None

    def get_parents(self, node_id: str) -> List[str]
    def get_children(self, node_id: str) -> List[str]
    def get_ancestors(self, node_id: str) -> Set[str]
    def get_descendants(self, node_id: str) -> Set[str]

    def is_d_separated(self, x: str, y: str, conditioning: Set[str]) -> bool
    def get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]
    def get_valid_adjustment_sets(self, treatment: str, outcome: str) -> List[Set[str]]

    @classmethod
    def from_knowledge_graph(cls, kg: KnowledgeGraph) -> 'StructuralCausalModel':
        """Construct SCM from biological knowledge graph."""

class DoCalculusEngine:
    """
    Implements Pearl's do-calculus for intervention reasoning.

    Enables queries like:
    - P(phenotype | do(gene_disrupted))
    - P(phenotype | do(pathway_targeted))
    """

    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm

    def do(self, intervention: Dict[str, float]) -> 'IntervenedModel':
        """
        Apply do-operator: set node values and remove incoming edges.

        Example:
            engine.do({"SHANK3_function": 0})  # Simulate SHANK3 knockout
        """

    def query(self,
              outcome: str,
              intervention: Dict[str, float],
              evidence: Optional[Dict[str, float]] = None) -> Distribution:
        """
        Compute P(outcome | do(intervention), evidence).

        Example:
            # What's the probability of ASD phenotype if we disrupt synaptic pathway?
            engine.query(
                outcome="asd_phenotype",
                intervention={"synaptic_pathway": "disrupted"}
            )
        """

    def average_treatment_effect(self,
                                  treatment: str,
                                  outcome: str,
                                  treatment_values: Tuple[float, float] = (0, 1)) -> float:
        """
        Compute ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
        """

    def conditional_average_treatment_effect(self,
                                              treatment: str,
                                              outcome: str,
                                              subgroup: Dict[str, float]) -> float:
        """
        Compute CATE for a specific subgroup.
        """

class CounterfactualEngine:
    """
    Enables counterfactual reasoning.

    Queries like:
    - "Would phenotype differ if pathway X were intact?"
    - "What if this individual had a different variant?"
    """

    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm

    def counterfactual(self,
                       factual_evidence: Dict[str, float],
                       counterfactual_intervention: Dict[str, float],
                       query_variable: str) -> Distribution:
        """
        Three-step counterfactual computation:
        1. Abduction: Infer exogenous variables from factual evidence
        2. Action: Apply counterfactual intervention
        3. Prediction: Compute query variable under modified model

        Example:
            # For an individual with SHANK3 mutation and ASD diagnosis,
            # what would phenotype be if SHANK3 were intact?
            engine.counterfactual(
                factual_evidence={"SHANK3_function": 0, "asd_diagnosis": 1},
                counterfactual_intervention={"SHANK3_function": 1},
                query_variable="asd_phenotype"
            )
        """

    def probability_of_necessity(self,
                                  treatment: str,
                                  outcome: str,
                                  factual: Dict[str, float]) -> float:
        """
        P(Y_0 = 0 | T = 1, Y = 1)
        "Given that treatment happened and outcome occurred,
         would outcome not have occurred without treatment?"
        """

    def probability_of_sufficiency(self,
                                    treatment: str,
                                    outcome: str,
                                    factual: Dict[str, float]) -> float:
        """
        P(Y_1 = 1 | T = 0, Y = 0)
        "Given that treatment didn't happen and outcome didn't occur,
         would outcome have occurred with treatment?"
        """

class CausalEffectEstimator:
    """
    Estimate direct, indirect, and total causal effects.
    """

    def __init__(self, scm: StructuralCausalModel, do_engine: DoCalculusEngine):
        self.scm = scm
        self.do_engine = do_engine

    def total_effect(self, treatment: str, outcome: str) -> float:
        """Total causal effect of treatment on outcome."""

    def direct_effect(self, treatment: str, outcome: str, mediator: str) -> float:
        """
        Natural Direct Effect: Effect not through mediator.

        Example:
            # Direct effect of gene on phenotype, not through pathway
            estimator.direct_effect("SHANK3", "asd_phenotype", "synaptic_pathway")
        """

    def indirect_effect(self, treatment: str, outcome: str, mediator: str) -> float:
        """
        Natural Indirect Effect: Effect through mediator.

        Example:
            # How much of SHANK3's effect on phenotype is mediated by synaptic pathway?
            estimator.indirect_effect("SHANK3", "asd_phenotype", "synaptic_pathway")
        """

    def mediation_analysis(self,
                           treatment: str,
                           outcome: str,
                           mediator: str) -> MediationResult:
        """
        Full mediation analysis with proportion mediated.
        """

@dataclass
class MediationResult:
    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float
    confidence_interval: Tuple[float, float]

@dataclass
class CausalQuery:
    """Structured representation of a causal query."""
    query_type: str  # "intervention", "counterfactual", "effect"
    treatment: str
    outcome: str
    intervention_value: Optional[float]
    conditioning: Optional[Dict[str, float]]
    mediator: Optional[str]

class CausalQueryBuilder:
    """Fluent interface for building causal queries."""

    def treatment(self, var: str) -> 'CausalQueryBuilder'
    def outcome(self, var: str) -> 'CausalQueryBuilder'
    def given(self, evidence: Dict[str, float]) -> 'CausalQueryBuilder'
    def do(self, intervention: Dict[str, float]) -> 'CausalQueryBuilder'
    def mediated_by(self, mediator: str) -> 'CausalQueryBuilder'
    def build(self) -> CausalQuery

# Example usage:
# query = CausalQueryBuilder()
#     .treatment("CHD8_function")
#     .outcome("asd_phenotype")
#     .mediated_by("chromatin_pathway")
#     .do({"CHD8_function": 0})
#     .build()
```

**Session 13A: Structural Causal Model + Do-Calculus**
```python
# causal_graph.py + do_calculus.py (~350 lines)

# Build SCM from knowledge graph
# Implement do-operator
# D-separation and adjustment sets
# Basic intervention queries
```

**Session 13B: Counterfactuals + Effect Estimation**
```python
# counterfactuals.py + effect_estimation.py (~350 lines)

# Three-step counterfactual algorithm
# Probability of necessity/sufficiency
# Direct/indirect effect estimation
# Mediation analysis
```

**Files to Create**:
```
modules/12_causal_inference/
├── README.md
├── __init__.py
├── causal_graph.py         # ~200 lines (SCM implementation)
├── do_calculus.py          # ~200 lines (intervention reasoning)
├── counterfactuals.py      # ~150 lines (counterfactual queries)
├── effect_estimation.py    # ~150 lines (mediation analysis)
└── tests/
    ├── test_scm.py
    ├── test_do_calculus.py
    └── test_counterfactuals.py
```

**Success Criteria**:
```bash
python -m pytest modules/12_causal_inference/tests/ -v

# Example validation: Known causal structure should yield correct effects
# SHANK3 → synaptic_pathway → asd_phenotype
# Direct effect should be small, indirect effect through pathway should be large
```

---

## Cross-Module Integration (Sessions 17-18)

After all modules are built, integration sessions combine them.

### Session 17: Subtype Discovery Pipeline

**Context Needed**:
- Module interfaces only (not implementations)
- Pipeline configuration schema

**File to Create**:
```python
# pipelines/subtype_discovery.py (~200 lines)

class SubtypeDiscoveryPipeline:
    def __init__(self, config: PipelineConfig):
        self.loader = VCFLoader()
        self.processor = VariantProcessor()
        self.kg = KnowledgeGraph.load(config.kg_path)
        self.gnn = OntologyAwareGNN.load(config.model_path)
        self.clusterer = SubtypeClusterer()

    def run(self, vcf_path: str) -> SubtypeResult:
        # Step 1: Load data
        variants = self.loader.load(vcf_path)

        # Step 2: Process variants
        gene_burdens = self.processor.process(variants)

        # Step 3: Compute pathway scores
        pathway_scores = self.score_pathways(gene_burdens)

        # Step 4: Cluster
        result = self.clusterer.fit(pathway_scores)

        return result
```

---

### Session 18: Full Pipeline with Hypotheses and Causal Reasoning

**Context Needed**:
- Session 17 pipeline
- Therapeutic hypothesis module interface
- Causal inference module interface

**File to Create**:
```python
# pipelines/therapeutic_hypothesis.py (~300 lines)

class TherapeuticHypothesisPipeline:
    def __init__(self, config: PipelineConfig):
        self.subtype_pipeline = SubtypeDiscoveryPipeline(config)
        self.rule_engine = RuleEngine(BiologicalRules.get_all_rules(), config.bio_context)
        self.hypothesis_ranker = HypothesisRanker()
        self.causal_model = StructuralCausalModel.from_knowledge_graph(config.kg)
        self.do_engine = DoCalculusEngine(self.causal_model)
        self.counterfactual_engine = CounterfactualEngine(self.causal_model)

    def run(self, vcf_path: str) -> TherapeuticResult:
        # Step 1: Discover subtypes
        subtype_result = self.subtype_pipeline.run(vcf_path)

        # Step 2: Apply symbolic rules (R1-R6)
        rule_results = self.apply_rules(subtype_result)

        # Step 3: Generate hypotheses
        hypotheses = self.generate_hypotheses(subtype_result, rule_results)

        # Step 4: Causal validation of hypotheses
        causal_results = self.validate_with_causal_reasoning(hypotheses)

        return TherapeuticResult(
            subtypes=subtype_result,
            rules=rule_results,
            hypotheses=hypotheses,
            causal_analysis=causal_results
        )

    def validate_with_causal_reasoning(self,
                                        hypotheses: List[TherapeuticHypothesis]) -> CausalValidation:
        """
        Use causal inference to validate and rank therapeutic hypotheses.

        For each hypothesis:
        1. Estimate direct effect of drug target on outcome
        2. Estimate indirect effect through disrupted pathway
        3. Compute counterfactual: "What if pathway were targeted?"
        """
        validated = []
        for hyp in hypotheses:
            # Intervention query: What's effect of targeting this pathway?
            intervention_effect = self.do_engine.query(
                outcome="asd_phenotype",
                intervention={hyp.target_pathway: "restored"}
            )

            # Mediation: How much effect goes through this pathway?
            mediation = CausalEffectEstimator(self.causal_model, self.do_engine).mediation_analysis(
                treatment=hyp.disrupted_gene,
                outcome="asd_phenotype",
                mediator=hyp.target_pathway
            )

            validated.append(CausallyValidatedHypothesis(
                hypothesis=hyp,
                intervention_effect=intervention_effect,
                mediation_result=mediation,
                causal_confidence=self.compute_causal_confidence(mediation)
            ))

        return CausalValidation(validated_hypotheses=validated)

# pipelines/causal_analysis.py (~200 lines)

class CausalAnalysisPipeline:
    """
    Standalone pipeline for causal queries on individual cases.
    """

    def __init__(self, config: PipelineConfig):
        self.scm = StructuralCausalModel.from_knowledge_graph(config.kg)
        self.do_engine = DoCalculusEngine(self.scm)
        self.cf_engine = CounterfactualEngine(self.scm)
        self.effect_estimator = CausalEffectEstimator(self.scm, self.do_engine)

    def analyze_individual(self, individual_data: IndividualData) -> CausalReport:
        """
        Generate causal analysis report for an individual.

        Includes:
        - Key disrupted pathways with causal effect estimates
        - Counterfactual analysis: "What if key genes were intact?"
        - Mediation analysis: Which pathways mediate variant effects?
        """

    def compare_subtypes_causally(self,
                                   subtype_profiles: List[SubtypeProfile]) -> SubtypeCausalComparison:
        """
        Compare subtypes using causal metrics.

        - Which pathways have strongest causal effects per subtype?
        - Do different subtypes have different causal mechanisms?
        """
```

---

## Context Management Best Practices

### For Each Session

1. **Start Fresh**: Don't assume context from previous sessions
2. **Provide Interface Only**: Share data structure definitions, not implementations
3. **Use Type Hints**: Makes interfaces self-documenting
4. **Write Standalone Tests**: Each module testable in isolation

### What to Include in Context

| Include | Exclude |
|---------|---------|
| Module README | Other module implementations |
| Interface contracts (dataclasses, method signatures) | Full source code of dependencies |
| Test data samples | Large datasets |
| Configuration schemas | Runtime configurations |
| Error handling patterns | Debugging logs |

### Context Budget Per Session

| Component | Approximate Tokens |
|-----------|-------------------|
| This session's README | 500 |
| Interface contracts | 300 |
| Implementation guidance | 200 |
| Test examples | 200 |
| **Total** | **~1,200 tokens** |

This leaves ample room for implementation discussion and code generation.

---

## Dependency Graph

```
Module 01 (Data Loaders)
    │
    ├──→ Module 02 (Variant Processing)
    │         │
    │         └──→ Module 07 (Pathway Scoring) ──→ Module 08 (Clustering)
    │                     │
    │                     └──────────────────────────────┐
    │                                                    │
    └──→ Module 03 (Knowledge Graph)                     │
              │                                          │
              ├──→ Module 04 (Graph Embeddings)          │
              │         │                                │
              │         └──→ Module 06 (Ontology GNN) ──→ Module 10 (Neuro-Symbolic)
              │                                                   │
              │                                                   │
              └──→ Module 05 (Pretrained Embeddings) ────────────┘
                        │                                         │
                        │ (fine-tuning uses Module 01 expr data)  │
                        │                                         │
                        │                           Module 09 (Symbolic Rules)
                        │                                  │      │
                        │                                  │      │
                        │                                  ▼      │
                        │                     Module 11 (Therapeutic Hypotheses)
                        │                                         │
                        │                                         │
                        └────────────────────────────────────────►│
                                                                  ▼
                                              Module 12 (Causal Inference)
                                              ────────────────────────────
                                              • Structural Causal Model
                                              • do-calculus queries
                                              • Counterfactual reasoning
                                              • Mediation analysis
```

### Parallel Implementation Opportunities

These module pairs can be implemented simultaneously:
- Module 04 + Module 05 (both produce embeddings)
- Module 07 + Module 09 (independent of each other)
- Module 08 + Module 11 (both consume pathway scores)
- Module 10 + Module 12 (both are advanced reasoning modules)

---

## Testing Strategy

### Unit Tests (Per Module)

Each module has standalone tests:
```bash
python -m pytest modules/XX_module_name/tests/ -v
```

### Integration Tests (After Phase Completion)

After each phase:
```bash
# Phase 1 integration
python -m pytest tests/integration/test_phase1.py -v

# Phase 2 integration
python -m pytest tests/integration/test_phase2.py -v
```

### End-to-End Tests (After All Modules)

```bash
python -m pytest tests/e2e/ -v
```

---

## Versioning and Checkpoints

### After Each Session

```bash
git add modules/XX_module_name/
git commit -m "feat(XX_module_name): implement core functionality"
git tag session-XX-complete
```

### After Each Phase

```bash
git tag phase-X-complete
```

This allows rollback if integration issues arise.

---

## Research Integrity: Limitations and Failure Mode Mitigation

This section documents how the framework addresses known limitations and failure modes in AI-driven autism genetics research, based on analysis of common pitfalls across four research approaches.

### 1) AI-Driven Genetic Subtyping Mitigations

| Limitation | Framework Mitigation | Module | Status |
|------------|---------------------|--------|--------|
| Clusters reflect cohort/ascertainment bias | `ConfoundAnalyzer` tests cluster-confound alignment (batch, site, ancestry, etc.) | 08 | ✅ Strong |
| Subtypes unstable under resampling | `StabilityAnalyzer` computes ARI/NMI, identifies unstable samples, rates stability | 08 | ✅ Strong |
| Pathway hits skewed by database bias | `PathwayLoader` supports multiple databases (GO, Reactome, KEGG) for cross-validation | 01 | ⚠️ Partial |
| Signals are associational, not causal | `StructuralCausalModel`, `DoCalculusEngine`, `CounterfactualEngine` | 12 | 🔲 Planned |
| Limited developmental context | `ExpressionLoader` (BrainSpan), `SingleCellLoader` (cell-type context) | 01 | ✅ Strong |

**Implemented Guardrails:**
- `StabilityResult.stability_rating` → qualitative assessment ("excellent" to "unstable")
- `StabilityResult.get_unstable_samples()` → identifies samples to exclude from interpretation
- `StabilityResult.is_stable` → programmatic gate (ARI > 0.8)
- `ConfoundAnalyzer.test_cluster_confound_alignment()` → tests for batch/site/ancestry effects
- `ConfoundReport.overall_risk` → risk level ("low"/"moderate"/"high")
- `NegativeControlRunner.permutation_test()` → validates structure is significant
- `ProvenanceRecord.validate_compatibility()` → ensures version consistency

### 2) Systems Biology Pathway Mapping Mitigations

| Limitation | Framework Mitigation | Module | Status |
|------------|---------------------|--------|--------|
| Incomplete/biased pathway databases | Multiple database support; `KnowledgeGraph` integrates multiple sources | 01, 03 | ⚠️ Partial |
| Batch effects, tissue mismatch | `QCFilter` for variant QC; developmental expression context | 02, 01 | ⚠️ Partial |
| Correlational, not causal mechanisms | `BiologicalRules` (R1-R6) for mechanistic reasoning; causal inference | 09, 12 | 🔲 Planned |
| Small/heterogeneous datasets | No explicit sample size warnings | — | 🔲 Gap |
| Functional impact is hypothesis-level | `FiredRule` includes `explanation`, `evidence`, `base_confidence` | 09 | 🔲 Planned |

**Implemented Guardrails:**
- `DevelopmentalExpression.get_prenatal_expression()` → prenatal cortex context
- `SingleCellAtlas.get_cell_type_specific_genes()` → cell-type specificity
- `PathwayScoreMatrix.contributing_genes` → transparency in pathway scoring

### 3) Phenotype-Genotype Correlation Mitigations

| Limitation | Framework Mitigation | Module | Status |
|------------|---------------------|--------|--------|
| Broad, noisy phenotypes | Framework is genetics-focused; phenotype handling minimal | — | 🔲 Gap |
| Confounding (ancestry, site, age) | `GeneConstraints` provides population context (gnomAD) | 01 | ⚠️ Partial |
| VUS uncertainty | Variant classification in annotation; no uncertainty propagation | 02 | ⚠️ Partial |
| Cross-cohort replication | No built-in replication framework | — | 🔲 Gap |

**Implemented Guardrails:**
- `SubtypeCharacterizer` enables "mechanism-first" approach (subtype by pathway, then evaluate)
- `SFARIGenes.scores` provides curated evidence levels (1-3)

### 4) Data-Driven Hypothesis Generation Mitigations

| Limitation | Framework Mitigation | Module | Status |
|------------|---------------------|--------|--------|
| Artifacts of bias, missingness, batch | `QCFilter` + `NegativeControlRunner` (permutation tests, null detection) | 02, 08 | ✅ Strong |
| Multiple testing inflation | FDR correction in `SubtypeCharacterizer._fdr_correction()` | 08 | ✅ Implemented |
| Lacks temporal/causal context | Developmental expression; causal inference planned | 01, 12 | ⚠️ Partial |
| Non-replicable across cohorts | Stability analysis; no cross-cohort framework | 08 | ⚠️ Partial |
| Outputs are hypotheses, not conclusions | `FiredRule` with explanations; `TherapeuticHypothesis.requires_validation` | 09, 11 | 🔲 Planned |

**Implemented Guardrails:**
- `PathwaySignature` includes `p_value`, `effect_size`, `fold_change`
- `CharacterizationConfig.use_fdr_correction` → automatic multiple testing correction

### Cross-Cutting Guardrails

| Guardrail | Implementation | Status |
|-----------|---------------|--------|
| **Pin versions and provenance** | `ProvenanceRecord` tracks reference genome, annotation, pathway DB versions, dependencies | ✅ Implemented |
| **Separate Signal/Hypothesis/Validation** | `ReasoningChain` separates observations from conclusions | 🔲 Planned |
| **Simple models that survive harsh tests** | Multiple clustering methods for comparison | ✅ Supported |
| **Never imply treatment readiness** | `TherapeuticHypothesis.requires_validation: True` | 🔲 Planned |

### Identified Gaps and Recommended Enhancements

#### High Priority Enhancements ✅ IMPLEMENTED

All three high-priority research integrity enhancements have been implemented in `modules/08_subtype_clustering/validation.py`:

**1. Confound Testing Module** ✅ Implemented
- `ConfoundAnalyzer` tests cluster-confound alignment for batch, site, ancestry, sex, age
- Chi-squared tests for categorical confounds, Kruskal-Wallis for continuous
- Effect size calculation (Cramer's V, eta-squared)
- Automatic Bonferroni correction for multiple testing
- Risk assessment and recommendations

**2. Negative Control Framework** ✅ Implemented
- `NegativeControlRunner` validates pipeline doesn't find structure in null data
- Permutation tests with configurable iterations
- Random gene set baseline comparisons
- Label shuffle tests
- Full negative control reports with p-values and recommendations

**3. Version Provenance Tracking** ✅ Implemented
- `ProvenanceRecord` tracks reference genome, annotation, pathway DB versions
- Dependency version capture
- Cohort and pipeline metadata
- Compatibility validation between records
- JSON serialization for reproducibility

#### Medium Priority Enhancements

**4. Multi-Representation Agreement** (enhance Module 08)
```python
def require_multi_representation_agreement(
    results: List[ClusteringResult],
    min_agreement_ari: float = 0.7,
) -> AgreementReport:
    """Require subtypes to be consistent across feature representations."""
```

**5. Cross-Cohort Validation Framework** (new utility)
```python
class ReplicationValidator:
    """Validate findings across independent cohorts."""
    def validate_subtype_geometry(
        self,
        discovery: ClusteringResult,
        replication: ClusteringResult,
    ) -> GeometryValidation

    def validate_pathway_themes(
        self,
        discovery: List[SubtypeProfile],
        replication: List[SubtypeProfile],
    ) -> ThemeValidation
```

### Coverage Summary by Research Approach

| Research Approach | Framework Coverage | Primary Gaps |
|------------------|-------------------|--------------|
| AI-Driven Genetic Subtyping | **Strong** | Confound testing, multi-representation |
| Systems Biology Pathway Mapping | **Moderate** | Batch/tissue mismatch warnings |
| Phenotype-Genotype Correlation | **Weak** | Phenotype structures, confound modeling |
| Data-Driven Hypothesis Generation | **Moderate** | Permutation/negative control framework |

### Implementation Priority for Remaining Modules

Given the gaps analysis, Module 09 (Symbolic Rules) and Module 12 (Causal Inference) are critical for addressing:
- "Correlation vs causation" concerns
- "Hypothesis not conclusion" discipline
- Mechanistic reasoning with explicit evidence

---

## Summary

| Phase | Modules | Sessions | Dependencies |
|-------|---------|----------|--------------|
| 1: Data Foundation | 01-02 | 2 | External data only |
| 2: Knowledge Representation | 03-05 | 3-4 | Phase 1 |
| 3: Neural Models | 06-08 | 4 | Phase 2 |
| 4: Symbolic Reasoning | 09-11 | 3 | Phase 3 |
| 5: Causal Inference | 12 | 2 | Phase 3, Phase 4 |
| Integration | Pipelines | 2 | All modules |
| **Total** | **12 modules** | **16-17 sessions** | |

Each session is designed to be completable in 1-2 hours with focused context, producing a tested, documented module ready for integration.

### New Capabilities Summary

| Enhancement | Module | Capability Added |
|-------------|--------|------------------|
| Developmental context | 01 | BrainSpan expression, single-cell atlas loading |
| Gene constraints | 01 | gnomAD pLI/LOEUF, SFARI gene scores |
| Foundation model fine-tuning | 05 | Autism-specific fine-tuning, variant-aware embeddings |
| Autism-specific rules | 09 | R1-R6 biological rules with full explanations |
| Causal reasoning | 12 | do-calculus, counterfactuals, mediation analysis |

---

**Document Status**: Implementation plan complete. Modules 01-11 implemented (Phase 4C complete).

**Next Step**: Begin Session 13 (Module 12: Causal Inference) - structural causal models, do-calculus, and counterfactual reasoning
