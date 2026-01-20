# Implementation Plan: Domain-Aware Autism Genetics Platform

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
│   └── 11_therapeutic_hypotheses/     # Phase 4C
│       ├── README.md
│       ├── __init__.py
│       ├── pathway_drug_mapping.py    # Pathway → drug links
│       ├── ranking.py                 # Hypothesis ranking
│       ├── evidence.py                # Evidence scoring
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

#### Module 05: Pretrained Embeddings (Session 5)

**Scope**: ~400 lines
**Dependencies**: None (uses external models)
**Interface Contract**:

```python
# Input: Gene list
# Output: Pretrained embeddings

class GeneformerExtractor:
    def __init__(self, model_path: str)
    def extract(self, gene_ids: List[str]) -> NodeEmbeddings

class ESM2Extractor:
    def __init__(self, model_name: str)
    def extract(self, protein_sequences: Dict[str, str]) -> NodeEmbeddings

class EmbeddingFusion:
    def fuse(self,
             kg_embeddings: NodeEmbeddings,
             geneformer_embeddings: NodeEmbeddings,
             esm2_embeddings: NodeEmbeddings) -> NodeEmbeddings
```

**Note**: This module can be implemented in parallel with Module 04.

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
```

---

### Phase 4: Symbolic Reasoning (Modules 09-11)

**Goal**: Add explainability and therapeutic hypotheses.

#### Module 09: Symbolic Rules (Session 10)

**Scope**: ~500 lines
**Dependencies**: Module 02 (gene data), Module 07 (pathway scores)
**Interface Contract**:

```python
# Input: Gene/pathway data
# Output: Rule-based inferences

@dataclass
class Rule:
    id: str
    name: str
    conditions: List[Condition]
    conclusion: Conclusion
    confidence: float

class RuleEngine:
    def __init__(self, rules: List[Rule])
    def evaluate(self,
                 individual_data: IndividualData) -> List[FiredRule]
    def explain(self, fired_rule: FiredRule) -> str

class BiologicalRules:
    """Curated biological rules."""

    @staticmethod
    def developmental_timing_rule() -> Rule

    @staticmethod
    def cell_type_specificity_rule() -> Rule

    @staticmethod
    def compensatory_mechanism_rule() -> Rule

    @staticmethod
    def get_all_rules() -> List[Rule]

@dataclass
class FiredRule:
    rule: Rule
    bindings: Dict[str, Any]
    confidence: float
    explanation: str
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
├── README.md           # Module documentation
├── __init__.py         # Exports
├── vcf_loader.py       # ~150 lines
├── annotation_loader.py # ~100 lines
├── pathway_loader.py   # ~150 lines
└── tests/
    ├── test_vcf_loader.py
    └── test_pathway_loader.py
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

### Session 5: Pretrained Embeddings

**Context Needed**:
- Geneformer API documentation
- ESM-2 API documentation
- Embedding fusion approach

**Files to Create**:
```
modules/05_pretrained_embeddings/
├── README.md
├── __init__.py
├── geneformer_extractor.py  # ~150 lines
├── esm2_extractor.py        # ~150 lines
├── fusion.py                # ~100 lines
└── tests/
    └── test_extractors.py
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
├── rule_engine.py          # ~200 lines
├── biological_rules.py     # ~200 lines
├── explanation.py          # ~100 lines
└── tests/
    └── test_rules.py
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

## Cross-Module Integration (Sessions 13-14)

After all modules are built, integration sessions combine them.

### Session 13: Subtype Discovery Pipeline

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

### Session 14: Full Pipeline with Hypotheses

**Context Needed**:
- Session 13 pipeline
- Therapeutic hypothesis module interface

**File to Create**:
```python
# pipelines/therapeutic_hypothesis.py (~200 lines)

class TherapeuticHypothesisPipeline:
    def __init__(self, config: PipelineConfig):
        self.subtype_pipeline = SubtypeDiscoveryPipeline(config)
        self.rule_engine = RuleEngine(BiologicalRules.get_all_rules())
        self.hypothesis_ranker = HypothesisRanker()

    def run(self, vcf_path: str) -> TherapeuticResult:
        # Step 1: Discover subtypes
        subtype_result = self.subtype_pipeline.run(vcf_path)

        # Step 2: Apply symbolic rules
        rule_results = self.apply_rules(subtype_result)

        # Step 3: Generate hypotheses
        hypotheses = self.generate_hypotheses(subtype_result, rule_results)

        return TherapeuticResult(
            subtypes=subtype_result,
            rules=rule_results,
            hypotheses=hypotheses
        )
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
    │
    └──→ Module 03 (Knowledge Graph)
              │
              ├──→ Module 04 (Graph Embeddings)
              │         │
              │         └──→ Module 06 (Ontology GNN) ──→ Module 10 (Neuro-Symbolic)
              │                                                   │
              │                                                   │
              └──→ Module 05 (Pretrained Embeddings) ────────────┘
                                                                  │
                                                    Module 09 (Symbolic Rules)
                                                                  │
                                                                  ▼
                                                    Module 11 (Therapeutic Hypotheses)
```

### Parallel Implementation Opportunities

These module pairs can be implemented simultaneously:
- Module 04 + Module 05 (both produce embeddings)
- Module 07 + Module 09 (independent of each other)
- Module 08 + Module 11 (both consume pathway scores)

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

## Summary

| Phase | Modules | Sessions | Dependencies |
|-------|---------|----------|--------------|
| 1: Data Foundation | 01-02 | 2 | External data only |
| 2: Knowledge Representation | 03-05 | 3 | Phase 1 |
| 3: Neural Models | 06-08 | 4 | Phase 2 |
| 4: Symbolic Reasoning | 09-11 | 3 | Phase 3 |
| Integration | Pipelines | 2 | All modules |
| **Total** | **11 modules** | **14 sessions** | |

Each session is designed to be completable in 1-2 hours with focused context, producing a tested, documented module ready for integration.

---

**Document Status**: Implementation plan complete. Ready for execution.

**Next Step**: Begin Session 1 (Module 01: Data Loaders)
