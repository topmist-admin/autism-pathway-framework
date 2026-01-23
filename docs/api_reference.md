# API Reference

> Complete API reference for the Autism Pathway Framework modules.

## Module 01: Data Loaders

### VCFLoader

```python
class VCFLoader:
    """Load and parse VCF (Variant Call Format) files."""

    def load(
        self,
        vcf_path: str,
        samples: Optional[List[str]] = None,
        region: Optional[str] = None
    ) -> VariantDataset:
        """
        Load variants from a VCF file.

        Args:
            vcf_path: Path to VCF file (.vcf or .vcf.gz)
            samples: Optional list of sample IDs to include (None = all)
            region: Optional genomic region (e.g., "chr1:1000-2000")

        Returns:
            VariantDataset containing all parsed variants

        Raises:
            FileNotFoundError: If VCF file doesn't exist
            ValueError: If VCF format is invalid

        Example:
            loader = VCFLoader()
            dataset = loader.load("variants.vcf.gz")
            print(f"Loaded {len(dataset.variants)} variants")
        """

    def validate(self, dataset: VariantDataset) -> ValidationReport:
        """
        Validate a loaded variant dataset.

        Args:
            dataset: VariantDataset to validate

        Returns:
            ValidationReport with statistics and warnings

        Example:
            report = loader.validate(dataset)
            if report.warnings:
                print(f"Warnings: {report.warnings}")
        """

    def load_region(
        self,
        vcf_path: str,
        chrom: str,
        start: int,
        end: int
    ) -> VariantDataset:
        """
        Load variants from a specific genomic region.

        Args:
            vcf_path: Path to indexed VCF file
            chrom: Chromosome name
            start: Start position (1-based)
            end: End position (inclusive)

        Returns:
            VariantDataset with variants in region
        """
```

### PathwayLoader

```python
class PathwayLoader:
    """Load pathway databases from various formats."""

    def load_gmt(self, gmt_path: str) -> PathwayDatabase:
        """
        Load pathways from GMT (Gene Matrix Transposed) format.

        Args:
            gmt_path: Path to GMT file

        Returns:
            PathwayDatabase with pathways and gene mappings

        Example:
            loader = PathwayLoader()
            db = loader.load_gmt("reactome.gmt")
            print(f"Loaded {len(db.pathways)} pathways")
        """

    def load_go(
        self,
        obo_path: str,
        gaf_path: str,
        namespace: str = "biological_process"
    ) -> PathwayDatabase:
        """
        Load Gene Ontology pathways.

        Args:
            obo_path: Path to GO OBO file
            gaf_path: Path to gene association file
            namespace: GO namespace to use

        Returns:
            PathwayDatabase with GO terms as pathways
        """

    def filter_by_size(
        self,
        database: PathwayDatabase,
        min_size: int = 5,
        max_size: int = 500
    ) -> PathwayDatabase:
        """
        Filter pathways by gene count.

        Args:
            database: Input PathwayDatabase
            min_size: Minimum genes per pathway
            max_size: Maximum genes per pathway

        Returns:
            Filtered PathwayDatabase
        """

    def merge(self, databases: List[PathwayDatabase]) -> PathwayDatabase:
        """
        Merge multiple pathway databases.

        Args:
            databases: List of PathwayDatabase objects

        Returns:
            Combined PathwayDatabase (with source prefixes)
        """
```

### ExpressionLoader

```python
class ExpressionLoader:
    """Load gene expression data (BrainSpan, GTEx, etc.)."""

    def load_brainspan(self, data_dir: str) -> DevelopmentalExpression:
        """
        Load BrainSpan developmental expression atlas.

        Args:
            data_dir: Directory containing BrainSpan files

        Returns:
            DevelopmentalExpression with expression matrix

        Expected files in data_dir:
            - expression_matrix.csv
            - rows_metadata.csv (genes)
            - columns_metadata.csv (samples)
        """

    def get_expression_by_stage(
        self,
        expression: DevelopmentalExpression,
        gene_id: str,
        stage: str
    ) -> float:
        """
        Get expression level for gene at developmental stage.

        Args:
            expression: DevelopmentalExpression object
            gene_id: Gene identifier
            stage: Developmental stage name

        Returns:
            Mean expression value (log2 TPM)
        """

    def get_prenatal_expressed_genes(
        self,
        expression: DevelopmentalExpression,
        threshold: float = 1.0
    ) -> List[str]:
        """
        Get genes expressed prenatally above threshold.

        Args:
            expression: DevelopmentalExpression object
            threshold: Minimum log2(TPM) value

        Returns:
            List of gene identifiers
        """
```

### ConstraintLoader

```python
class ConstraintLoader:
    """Load gene constraint scores from gnomAD and SFARI."""

    def load_gnomad_constraints(self, tsv_path: str) -> GeneConstraints:
        """
        Load gnomAD gene constraint scores.

        Args:
            tsv_path: Path to gnomAD constraint file

        Returns:
            GeneConstraints with pLI, LOEUF, mis_z scores

        Example:
            loader = ConstraintLoader()
            constraints = loader.load_gnomad_constraints("gnomad.tsv")
            pli = constraints.get_pli("SHANK3")  # Returns 1.0
        """

    def load_sfari_genes(self, csv_path: str) -> SFARIGenes:
        """
        Load SFARI autism gene annotations.

        Args:
            csv_path: Path to SFARI gene CSV

        Returns:
            SFARIGenes with scores and evidence
        """

    def get_constrained_genes(
        self,
        constraints: GeneConstraints,
        pli_threshold: float = 0.9,
        loeuf_threshold: Optional[float] = None
    ) -> List[str]:
        """
        Get genes meeting constraint thresholds.

        Args:
            constraints: GeneConstraints object
            pli_threshold: Minimum pLI score
            loeuf_threshold: Maximum LOEUF score (optional)

        Returns:
            List of constrained gene identifiers
        """
```

---

## Module 02: Variant Processing

### QCFilter

```python
class QCFilter:
    """Quality control filtering for variants and samples."""

    def __init__(self):
        """Initialize QC filter with default settings."""

    def filter_variants(
        self,
        dataset: VariantDataset,
        config: QCConfig
    ) -> VariantDataset:
        """
        Filter variants based on quality criteria.

        Args:
            dataset: Input VariantDataset
            config: QCConfig with filter thresholds

        Returns:
            Filtered VariantDataset

        Filters applied:
            - Quality score (QUAL)
            - Read depth (DP)
            - Genotype quality (GQ)
            - Filter status (PASS only)
            - Allele frequency
            - Chromosome exclusion
        """

    def filter_samples(
        self,
        dataset: VariantDataset,
        config: QCConfig
    ) -> VariantDataset:
        """
        Filter samples based on QC criteria.

        Args:
            dataset: Input VariantDataset
            config: QCConfig with sample thresholds

        Returns:
            VariantDataset with passing samples only

        Filters applied:
            - Minimum variants per sample
            - Maximum variants per sample
            - Sample call rate
        """

    def run_full_qc(
        self,
        dataset: VariantDataset,
        config: QCConfig
    ) -> Tuple[VariantDataset, QCReport]:
        """
        Run complete QC pipeline.

        Args:
            dataset: Input VariantDataset
            config: QCConfig with all thresholds

        Returns:
            Tuple of (filtered_dataset, qc_report)

        Example:
            qc = QCFilter()
            config = QCConfig(min_quality=20, max_allele_freq=0.01)
            filtered, report = qc.run_full_qc(dataset, config)
            print(report)
        """
```

### QCConfig

```python
@dataclass
class QCConfig:
    """Configuration for QC filtering."""

    # Variant filters
    min_quality: float = 20.0
    min_depth: Optional[int] = 10
    min_genotype_quality: Optional[int] = 20
    filter_pass_only: bool = True

    # Allele frequency filters
    min_allele_freq: float = 0.0
    max_allele_freq: float = 0.01

    # Sample filters
    max_missing_rate: float = 0.1
    min_variants_per_sample: int = 0
    max_variants_per_sample: Optional[int] = None

    # Chromosome filters
    exclude_chromosomes: Set[str] = field(
        default_factory=lambda: {"chrM", "M", "MT"}
    )
```

### VariantAnnotator

```python
class VariantAnnotator:
    """Annotate variants with functional consequences."""

    def __init__(self):
        """Initialize annotator with empty caches."""

    def annotate(self, variant: Variant) -> AnnotatedVariant:
        """
        Annotate a single variant.

        Args:
            variant: Variant to annotate

        Returns:
            AnnotatedVariant with consequence, impact, scores
        """

    def annotate_batch(
        self,
        variants: List[Variant]
    ) -> List[AnnotatedVariant]:
        """
        Annotate multiple variants efficiently.

        Args:
            variants: List of variants

        Returns:
            List of AnnotatedVariants
        """

    def load_cadd_scores(self, cadd_path: str) -> None:
        """
        Load CADD scores from file.

        Args:
            cadd_path: Path to CADD score file (TSV)
        """

    def load_gnomad_frequencies(self, gnomad_path: str) -> None:
        """
        Load gnomAD allele frequencies.

        Args:
            gnomad_path: Path to gnomAD frequency file
        """

    def get_lof_variants(
        self,
        annotated: List[AnnotatedVariant]
    ) -> List[AnnotatedVariant]:
        """
        Filter to loss-of-function variants only.

        Args:
            annotated: List of annotated variants

        Returns:
            List of LoF variants
        """

    def get_damaging_missense(
        self,
        annotated: List[AnnotatedVariant],
        cadd_threshold: float = 20.0,
        revel_threshold: float = 0.5
    ) -> List[AnnotatedVariant]:
        """
        Get damaging missense variants.

        Args:
            annotated: List of annotated variants
            cadd_threshold: Minimum CADD phred score
            revel_threshold: Minimum REVEL score

        Returns:
            List of damaging missense variants
        """
```

### GeneBurdenCalculator

```python
class GeneBurdenCalculator:
    """Calculate gene-level burden scores from variants."""

    def __init__(self, config: Optional[WeightConfig] = None):
        """
        Initialize calculator with weighting configuration.

        Args:
            config: WeightConfig (uses defaults if None)
        """

    def compute(
        self,
        variants: List[AnnotatedVariant],
        samples: Optional[List[str]] = None
    ) -> GeneBurdenMatrix:
        """
        Compute gene burden matrix from annotated variants.

        Args:
            variants: List of annotated variants
            samples: Optional sample list (inferred if None)

        Returns:
            GeneBurdenMatrix with scores for all samples/genes

        Example:
            calc = GeneBurdenCalculator()
            matrix = calc.compute(annotated_variants)
            burden = matrix.get_score("SAMPLE_001", "SHANK3")
        """

    def compute_lof_burden(
        self,
        variants: List[AnnotatedVariant]
    ) -> GeneBurdenMatrix:
        """
        Compute burden using only LoF variants.

        Args:
            variants: List of annotated variants

        Returns:
            GeneBurdenMatrix with LoF-only scores
        """

    def compute_missense_burden(
        self,
        variants: List[AnnotatedVariant],
        cadd_threshold: float = 20.0,
        revel_threshold: float = 0.5
    ) -> GeneBurdenMatrix:
        """
        Compute burden using only damaging missense variants.

        Args:
            variants: Annotated variants
            cadd_threshold: Min CADD score
            revel_threshold: Min REVEL score

        Returns:
            GeneBurdenMatrix with missense-only scores
        """

    @staticmethod
    def combine_burdens(
        burdens: List[GeneBurdenMatrix],
        weights: Optional[List[float]] = None
    ) -> GeneBurdenMatrix:
        """
        Combine multiple burden matrices.

        Args:
            burdens: List of GeneBurdenMatrix objects
            weights: Optional weights for each matrix

        Returns:
            Combined GeneBurdenMatrix
        """
```

### WeightConfig

```python
@dataclass
class WeightConfig:
    """Configuration for variant weighting in burden calculation."""

    # Consequence weights
    consequence_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "frameshift_variant": 1.0,
            "stop_gained": 1.0,
            "splice_acceptor_variant": 1.0,
            "splice_donor_variant": 1.0,
            "start_lost": 1.0,
            "missense_variant": 0.5,
            "inframe_insertion": 0.3,
            "inframe_deletion": 0.3,
            # ... more consequences
        }
    )

    # CADD weighting
    use_cadd_weighting: bool = True
    cadd_threshold: float = 20.0
    cadd_weight_scale: float = 0.05

    # REVEL weighting
    use_revel_weighting: bool = True
    revel_threshold: float = 0.5

    # AF weighting
    use_af_weighting: bool = False
    af_weight_beta: float = 1.0

    # Filter settings
    include_synonymous: bool = False
    min_impact: str = "MODERATE"

    # Aggregation
    aggregation: str = "weighted_sum"  # weighted_sum, max, count
```

### GeneBurdenMatrix

```python
@dataclass
class GeneBurdenMatrix:
    """Gene burden scores for all samples."""

    samples: List[str]
    genes: List[str]
    scores: np.ndarray  # Shape: (n_samples, n_genes)
    sample_index: Dict[str, int]
    gene_index: Dict[str, int]
    contributing_variants: Dict[Tuple[str, str], List[str]]

    @property
    def n_samples(self) -> int:
        """Number of samples."""

    @property
    def n_genes(self) -> int:
        """Number of genes."""

    def get_sample(self, sample_id: str) -> Dict[str, float]:
        """
        Get burden scores for a sample.

        Args:
            sample_id: Sample identifier

        Returns:
            Dict mapping gene -> score (non-zero only)
        """

    def get_gene(self, gene_id: str) -> np.ndarray:
        """
        Get scores for a gene across all samples.

        Args:
            gene_id: Gene identifier

        Returns:
            Array of scores (length n_samples)
        """

    def get_score(self, sample_id: str, gene_id: str) -> float:
        """
        Get score for specific sample and gene.

        Args:
            sample_id: Sample identifier
            gene_id: Gene identifier

        Returns:
            Burden score (0.0 if not found)
        """

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.

        Returns:
            DataFrame with samples as rows, genes as columns
        """

    def to_sparse(self) -> scipy.sparse.csr_matrix:
        """
        Convert to sparse matrix format.

        Returns:
            Sparse CSR matrix
        """

    def filter_genes(self, genes: Set[str]) -> "GeneBurdenMatrix":
        """
        Filter to subset of genes.

        Args:
            genes: Set of genes to keep

        Returns:
            New filtered GeneBurdenMatrix
        """

    def normalize(self, method: str = "zscore") -> "GeneBurdenMatrix":
        """
        Normalize burden scores.

        Args:
            method: "zscore", "minmax", or "rank"

        Returns:
            New normalized GeneBurdenMatrix
        """
```

---

## Data Classes

### Variant

```python
@dataclass
class Variant:
    """Single genetic variant for one sample."""

    chrom: str              # Chromosome
    pos: int                # Position (1-based)
    ref: str                # Reference allele
    alt: str                # Alternate allele
    sample_id: str          # Sample identifier
    genotype: str           # Genotype (e.g., "0/1")
    quality: float          # QUAL score
    filter_status: str      # Filter status
    info: Dict[str, Any]    # INFO fields
    variant_id: Optional[str] = None

    @property
    def variant_type(self) -> str:
        """Returns: 'SNV', 'insertion', 'deletion', or 'MNV'"""

    @property
    def is_snv(self) -> bool:
        """Check if single nucleotide variant."""
```

### AnnotatedVariant

```python
@dataclass
class AnnotatedVariant:
    """Variant with functional annotations."""

    variant: Variant
    gene_id: Optional[str] = None
    gene_symbol: Optional[str] = None
    transcript_id: Optional[str] = None
    consequence: VariantConsequence = VariantConsequence.UNKNOWN
    impact: ImpactLevel = ImpactLevel.MODIFIER
    cadd_phred: Optional[float] = None
    revel_score: Optional[float] = None
    gnomad_af: Optional[float] = None

    @property
    def is_lof(self) -> bool:
        """Check if loss-of-function."""

    @property
    def is_missense(self) -> bool:
        """Check if missense variant."""

    @property
    def is_coding(self) -> bool:
        """Check if in coding region."""

    @property
    def is_rare(self, threshold: float = 0.01) -> bool:
        """Check if rare (AF < threshold)."""
```

---

## Enums

### VariantConsequence

```python
class VariantConsequence(Enum):
    """VEP consequence types."""

    FRAMESHIFT = "frameshift_variant"
    STOP_GAINED = "stop_gained"
    SPLICE_ACCEPTOR = "splice_acceptor_variant"
    SPLICE_DONOR = "splice_donor_variant"
    START_LOST = "start_lost"
    STOP_LOST = "stop_lost"
    MISSENSE = "missense_variant"
    INFRAME_INSERTION = "inframe_insertion"
    INFRAME_DELETION = "inframe_deletion"
    SYNONYMOUS = "synonymous_variant"
    SPLICE_REGION = "splice_region_variant"
    INTRON = "intron_variant"
    UTR_5_PRIME = "5_prime_UTR_variant"
    UTR_3_PRIME = "3_prime_UTR_variant"
    INTERGENIC = "intergenic_variant"
    UNKNOWN = "unknown"
```

### ImpactLevel

```python
class ImpactLevel(Enum):
    """VEP impact levels."""

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    MODIFIER = "MODIFIER"
```

---

## Module 03: Knowledge Graph

### KnowledgeGraph

```python
class KnowledgeGraph:
    """Biological knowledge graph for gene-pathway-phenotype relationships."""

    def add_node(self, node_id: str, node_type: str, **attributes) -> None:
        """Add a node to the graph."""

    def add_edge(self, source: str, target: str, relation: str, **attributes) -> None:
        """Add a directed edge between nodes."""

    def load_pathways(self, gmt_path: str) -> int:
        """Load pathways from GMT file. Returns count of pathways loaded."""

    def load_interactions(self, interactions_path: str) -> int:
        """Load gene-gene interactions. Returns count of edges added."""

    def get_subgraph(self, node_ids: List[str], max_hops: int = 1) -> 'KnowledgeGraph':
        """Extract subgraph around specified nodes."""

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph."""

    def to_pyg(self) -> torch_geometric.data.HeteroData:
        """Convert to PyTorch Geometric heterogeneous graph."""
```

---

## Module 04: Graph Embeddings

### TransEModel

```python
class TransEModel:
    """TransE knowledge graph embedding model."""

    def __init__(self, kg: KnowledgeGraph, embedding_dim: int = 128):
        """Initialize TransE model."""

    def train(self, epochs: int = 100, lr: float = 0.01) -> np.ndarray:
        """Train embeddings. Returns entity embedding matrix."""

    def get_embedding(self, entity_id: str) -> np.ndarray:
        """Get embedding vector for an entity."""

    def predict_link(self, head: str, relation: str, tail: str) -> float:
        """Predict link probability."""
```

### RotatEModel

```python
class RotatEModel:
    """RotatE knowledge graph embedding model (complex-valued)."""

    def __init__(self, kg: KnowledgeGraph, embedding_dim: int = 128):
        """Initialize RotatE model."""

    def train(self, epochs: int = 100) -> np.ndarray:
        """Train embeddings. Returns entity embedding matrix."""
```

---

## Module 05: Pretrained Embeddings

### GeneformerEmbedder

```python
class GeneformerEmbedder:
    """Geneformer foundation model embeddings for genes."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize with optional pretrained model path."""

    def embed_genes(self, gene_ids: List[str]) -> np.ndarray:
        """Get embeddings for list of genes."""

    def embed_expression_profile(self, expression: Dict[str, float]) -> np.ndarray:
        """Embed an expression profile."""
```

### ESM2Embedder

```python
class ESM2Embedder:
    """ESM-2 protein language model embeddings."""

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Get embedding for protein sequence."""

    def embed_genes(self, gene_ids: List[str]) -> np.ndarray:
        """Get embeddings via canonical protein sequences."""
```

---

## Module 06: Ontology GNN

### OntologyAwareGNN

```python
class OntologyAwareGNN(torch.nn.Module):
    """Graph Neural Network respecting ontology hierarchy."""

    def __init__(self, config: GNNConfig):
        """Initialize with configuration."""

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Forward pass through GNN layers."""

    def predict_pathway_scores(self, gene_burdens: GeneBurdenMatrix) -> PathwayScores:
        """Predict pathway disruption scores from gene burdens."""
```

### GNNConfig

```python
@dataclass
class GNNConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    aggregation: str = "mean"  # mean, sum, max
    use_ontology_attention: bool = True
```

---

## Module 07: Pathway Scoring

### PathwayScorer

```python
class PathwayScorer:
    """Multi-evidence pathway disruption scoring."""

    def __init__(self, config: ScoringConfig):
        """Initialize with scoring configuration."""

    def score(
        self,
        gene_burdens: GeneBurdenMatrix,
        context: Optional[BiologicalContext] = None
    ) -> PathwayScores:
        """Compute pathway disruption scores."""

    def score_with_uncertainty(
        self,
        gene_burdens: GeneBurdenMatrix,
        n_bootstrap: int = 100
    ) -> Tuple[PathwayScores, np.ndarray]:
        """Score with bootstrap confidence intervals."""
```

### PathwayScores

```python
@dataclass
class PathwayScores:
    samples: List[str]
    pathways: List[str]
    scores: np.ndarray  # (n_samples, n_pathways)
    z_scores: np.ndarray  # Normalized scores

    def get_top_pathways(self, sample: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get top disrupted pathways for a sample."""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
```

---

## Module 08: Subtype Clustering

### SubtypeClusterer

```python
class SubtypeClusterer:
    """GMM-based clustering with validation."""

    def __init__(self, config: ClusterConfig):
        """Initialize clusterer."""

    def fit(self, pathway_scores: PathwayScores) -> ClusterResult:
        """Fit clustering model."""

    def predict(self, pathway_scores: PathwayScores) -> np.ndarray:
        """Predict cluster assignments."""

    def fit_predict(self, pathway_scores: PathwayScores) -> ClusterResult:
        """Fit and predict in one step."""
```

### ClusterResult

```python
@dataclass
class ClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray  # Soft assignments
    n_clusters: int
    silhouette_score: float
    stability_score: float
    cluster_profiles: Dict[int, PathwayProfile]
```

---

## Module 09: Symbolic Rules

### RuleEngine

```python
class RuleEngine:
    """Biological rule inference engine."""

    def __init__(self, rules: List[BiologicalRule], context: BiologicalContext):
        """Initialize with rules and biological context."""

    def evaluate(self, gene_data: GeneData) -> List[RuleResult]:
        """Evaluate all rules on gene data."""

    def evaluate_sample(self, sample_id: str, burdens: GeneBurdenMatrix) -> List[RuleResult]:
        """Evaluate rules for a specific sample."""
```

### BiologicalRule

```python
@dataclass
class BiologicalRule:
    rule_id: str  # R1, R2, etc.
    name: str
    condition: Callable[[GeneData, BiologicalContext], bool]
    action: Callable[[GeneData], RuleOutput]
    confidence: float
    explanation_template: str
```

---

## Module 10: Neurosymbolic

### NeuroSymbolicModel

```python
class NeuroSymbolicModel:
    """Combined GNN + symbolic rule model."""

    def __init__(self, gnn: OntologyAwareGNN, rule_engine: RuleEngine, config: NSConfig):
        """Initialize neurosymbolic model."""

    def predict(self, data: HeteroData) -> NeuroSymbolicOutput:
        """Combined prediction with neural and symbolic components."""

    def explain(self, sample_id: str) -> Explanation:
        """Generate explanation for prediction."""
```

### NeuroSymbolicOutput

```python
@dataclass
class NeuroSymbolicOutput:
    neural_scores: np.ndarray
    symbolic_adjustments: np.ndarray
    combined_scores: np.ndarray
    fired_rules: List[RuleResult]
    explanations: List[str]
```

---

## Module 11: Therapeutic Hypotheses

### PathwayDrugMapper

```python
class PathwayDrugMapper:
    """Map disrupted pathways to drug candidates."""

    def __init__(self, drug_db: DrugTargetDatabase, config: MapperConfig):
        """Initialize mapper with drug database."""

    def map(
        self,
        pathway_id: str,
        pathway_genes: Optional[List[str]] = None,
        disrupted_genes: Optional[List[str]] = None
    ) -> List[DrugCandidate]:
        """Map pathway to candidate drugs."""
```

### HypothesisRanker

```python
class HypothesisRanker:
    """Rank therapeutic hypotheses with evidence scoring."""

    def rank(
        self,
        hypotheses: List[TherapeuticHypothesis],
        evidence_scorer: EvidenceScorer
    ) -> RankingResult:
        """Rank hypotheses by evidence and diversity."""
```

### TherapeuticHypothesis

```python
@dataclass
class TherapeuticHypothesis:
    hypothesis_id: str
    drug: DrugCandidate
    target_pathway: str
    mechanism: str
    evidence_score: EvidenceScore
    requires_validation: bool = True  # Always True (safety)
```

---

## Module 12: Causal Inference

### StructuralCausalModel

```python
class StructuralCausalModel:
    """Structural Causal Model for ASD genetics."""

    def add_node(self, node: CausalNode) -> None:
        """Add a causal node."""

    def add_edge(self, edge: CausalEdge) -> None:
        """Add a causal edge."""

    def is_d_separated(self, x: str, y: str, conditioning: Set[str]) -> bool:
        """Test d-separation."""

    def get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """Find backdoor paths."""

    def get_valid_adjustment_sets(self, treatment: str, outcome: str) -> List[Set[str]]:
        """Find valid adjustment sets."""
```

### DoCalculusEngine

```python
class DoCalculusEngine:
    """Pearl's do-calculus for intervention reasoning."""

    def __init__(self, scm: StructuralCausalModel):
        """Initialize with SCM."""

    def do(self, intervention: Dict[str, float]) -> IntervenedModel:
        """Apply do-operator."""

    def query(
        self,
        outcome: str,
        intervention: Dict[str, float],
        evidence: Optional[Dict[str, float]] = None
    ) -> Distribution:
        """Compute P(outcome | do(intervention), evidence)."""

    def average_treatment_effect(
        self,
        treatment: str,
        outcome: str,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """Compute ATE = E[Y|do(T=1)] - E[Y|do(T=0)]."""
```

### CounterfactualEngine

```python
class CounterfactualEngine:
    """Counterfactual reasoning engine."""

    def counterfactual(
        self,
        factual_evidence: Dict[str, float],
        counterfactual_intervention: Dict[str, float],
        query_variable: str
    ) -> CounterfactualResult:
        """Three-step counterfactual: abduction, action, prediction."""

    def probability_of_necessity(
        self,
        treatment: str,
        outcome: str,
        factual: Dict[str, float]
    ) -> float:
        """P(Y_0=0 | T=1, Y=1) - was treatment necessary?"""

    def probability_of_sufficiency(
        self,
        treatment: str,
        outcome: str,
        factual: Dict[str, float]
    ) -> float:
        """P(Y_1=1 | T=0, Y=0) - would treatment be sufficient?"""
```

### CausalEffectEstimator

```python
class CausalEffectEstimator:
    """Estimate direct, indirect, and total causal effects."""

    def total_effect(self, treatment: str, outcome: str) -> float:
        """Total causal effect."""

    def direct_effect(self, treatment: str, outcome: str, mediator: str) -> float:
        """Natural Direct Effect (not through mediator)."""

    def indirect_effect(self, treatment: str, outcome: str, mediator: str) -> float:
        """Natural Indirect Effect (through mediator)."""

    def mediation_analysis(
        self,
        treatment: str,
        outcome: str,
        mediator: str
    ) -> MediationResult:
        """Full mediation analysis with proportion mediated."""
```

### MediationResult

```python
@dataclass
class MediationResult:
    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float
    confidence_interval: Tuple[float, float]
    treatment: str
    outcome: str
    mediator: str
    explanation: str
```

---

## See Also

- [Configuration Guide](configuration.md) - Parameter configuration
- [Data Formats](data_formats.md) - File format specifications
- [Testing Guide](testing.md) - How to test modules
- [Implementation Plan](implementation_plan.md) - Module development roadmap
