"""
Subtype Discovery Pipeline (Session 17)

End-to-end pipeline from VCF data to autism subtype identification.

This pipeline integrates:
- Module 01: Data loaders (VCF, pathways, constraints)
- Module 02: Variant processing (QC, annotation, gene burden)
- Module 03: Knowledge graph (optional, for network propagation)
- Module 07: Pathway scoring (aggregation, normalization)
- Module 08: Subtype clustering (clustering, stability, characterization)

Example Usage:
    from pipelines import SubtypeDiscoveryPipeline, PipelineConfig

    # Configure pipeline
    config = PipelineConfig(
        data=DataConfig(
            vcf_path="cohort.vcf.gz",
            pathway_gmt_path="reactome.gmt",
        ),
        processing=ProcessingConfig(
            min_quality=30.0,
            max_allele_freq=0.01,
        ),
        clustering=ClusteringPipelineConfig(
            n_clusters=5,
            run_stability=True,
        ),
    )

    # Run pipeline
    pipeline = SubtypeDiscoveryPipeline(config)
    result = pipeline.run()

    # Access results
    print(f"Identified {result.n_subtypes} subtypes")
    for profile in result.subtype_profiles:
        print(profile.summary)
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Dynamic Module Imports (handles numeric prefixes in module names)
# =============================================================================

_project_root = Path(__file__).parent.parent
_modules_dir = _project_root / "modules"


def _import_module_from_path(module_name: str, dir_name: str):
    """Import a module from a directory with numeric prefix."""
    module_path = _modules_dir / dir_name / "__init__.py"
    if not module_path.exists():
        raise ImportError(f"Module not found: {module_path}")

    # Add module directory to path for internal imports
    module_dir = str(_modules_dir / dir_name)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Module 01: Data Loaders
_mod_01 = _import_module_from_path("data_loaders", "01_data_loaders")
VCFLoader = _mod_01.VCFLoader
VariantDataset = _mod_01.VariantDataset
ValidationReport = _mod_01.ValidationReport
PathwayLoader = _mod_01.PathwayLoader
PathwayDatabase = _mod_01.PathwayDatabase
ConstraintLoader = _mod_01.ConstraintLoader
GeneConstraints = _mod_01.GeneConstraints

# Module 02: Variant Processing
_mod_02 = _import_module_from_path("variant_processing", "02_variant_processing")
QCFilter = _mod_02.QCFilter
QCConfig = _mod_02.QCConfig
QCReport = _mod_02.QCReport
VariantAnnotator = _mod_02.VariantAnnotator
AnnotatedVariant = _mod_02.AnnotatedVariant
GeneBurdenCalculator = _mod_02.GeneBurdenCalculator
GeneBurdenMatrix = _mod_02.GeneBurdenMatrix
WeightConfig = _mod_02.WeightConfig

# Module 03: Knowledge Graph
_mod_03 = _import_module_from_path("knowledge_graph", "03_knowledge_graph")
KnowledgeGraph = _mod_03.KnowledgeGraph
KnowledgeGraphBuilder = _mod_03.KnowledgeGraphBuilder
load_ppi_from_file = _mod_03.load_ppi_from_file

# Module 07: Pathway Scoring
_mod_07 = _import_module_from_path("pathway_scoring", "07_pathway_scoring")
PathwayAggregator = _mod_07.PathwayAggregator
AggregationConfig = _mod_07.AggregationConfig
AggregationMethod = _mod_07.AggregationMethod
PathwayScoreMatrix = _mod_07.PathwayScoreMatrix
NetworkPropagator = _mod_07.NetworkPropagator
PropagationConfig = _mod_07.PropagationConfig
PropagationMethod = _mod_07.PropagationMethod
PathwayScoreNormalizer = _mod_07.PathwayScoreNormalizer
NormalizationConfig = _mod_07.NormalizationConfig
NormalizationMethod = _mod_07.NormalizationMethod

# Module 08: Subtype Clustering
_mod_08 = _import_module_from_path("subtype_clustering", "08_subtype_clustering")
SubtypeClusterer = _mod_08.SubtypeClusterer
ClusteringConfig = _mod_08.ClusteringConfig
ClusteringMethod = _mod_08.ClusteringMethod
ClusteringResult = _mod_08.ClusteringResult
StabilityAnalyzer = _mod_08.StabilityAnalyzer
StabilityConfig = _mod_08.StabilityConfig
StabilityResult = _mod_08.StabilityResult
SubtypeCharacterizer = _mod_08.SubtypeCharacterizer
CharacterizationConfig = _mod_08.CharacterizationConfig
SubtypeProfile = _mod_08.SubtypeProfile
ConfoundAnalyzer = _mod_08.ConfoundAnalyzer
ConfoundAnalyzerConfig = _mod_08.ConfoundAnalyzerConfig
ConfoundReport = _mod_08.ConfoundReport
NegativeControlRunner = _mod_08.NegativeControlRunner
NegativeControlConfig = _mod_08.NegativeControlConfig
NegativeControlReport = _mod_08.NegativeControlReport
ProvenanceRecord = _mod_08.ProvenanceRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading."""

    # Required: VCF input
    vcf_path: str

    # Required: Pathway database
    pathway_gmt_path: Optional[str] = None
    pathway_go_obo_path: Optional[str] = None
    pathway_go_gaf_path: Optional[str] = None

    # Optional: Pre-loaded pathway database
    pathway_database: Optional[PathwayDatabase] = None

    # Optional: Constraint scores for weighting
    gnomad_constraints_path: Optional[str] = None
    sfari_genes_path: Optional[str] = None

    # Optional: Pre-built knowledge graph for network propagation
    knowledge_graph_path: Optional[str] = None
    knowledge_graph: Optional[KnowledgeGraph] = None

    # Optional: PPI network for knowledge graph building
    ppi_network_path: Optional[str] = None
    ppi_min_score: float = 700.0

    # Gene ID type used in data
    gene_id_type: str = "symbol"


@dataclass
class ProcessingConfig:
    """Configuration for variant processing."""

    # QC parameters
    min_quality: float = 30.0
    min_depth: int = 10
    filter_pass_only: bool = True
    max_allele_freq: float = 0.01  # Rare variants only

    # Variant weighting
    use_cadd_weighting: bool = False
    cadd_threshold: float = 20.0
    use_constraint_weighting: bool = True

    # Burden aggregation
    burden_aggregation: str = "weighted_sum"  # weighted_sum, max, count


@dataclass
class ClusteringPipelineConfig:
    """Configuration for clustering and subtype analysis."""

    # Clustering method
    method: ClusteringMethod = ClusteringMethod.GMM
    n_clusters: Optional[int] = None  # Auto-detect if None
    min_clusters: int = 2
    max_clusters: int = 10

    # Stability analysis
    run_stability: bool = True
    n_bootstrap: int = 100

    # Characterization
    n_top_pathways: int = 10
    min_fold_change: float = 1.5
    use_fdr_correction: bool = True

    # Validation
    run_confound_analysis: bool = False
    run_negative_controls: bool = False
    n_permutations: int = 100


@dataclass
class PathwayScoringConfig:
    """Configuration for pathway scoring."""

    # Aggregation
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_SUM
    min_pathway_size: int = 5
    max_pathway_size: int = 500
    normalize_by_pathway_size: bool = True

    # Network propagation (optional)
    use_network_propagation: bool = False
    propagation_method: PropagationMethod = PropagationMethod.RANDOM_WALK
    restart_prob: float = 0.5

    # Normalization
    normalization_method: NormalizationMethod = NormalizationMethod.ZSCORE


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    data: DataConfig
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    pathway_scoring: PathwayScoringConfig = field(default_factory=PathwayScoringConfig)
    clustering: ClusteringPipelineConfig = field(default_factory=ClusteringPipelineConfig)

    # Pipeline behavior
    verbose: bool = True
    random_state: int = 42

    # Output
    output_dir: Optional[str] = None
    save_intermediate: bool = False


# =============================================================================
# Pipeline Result
# =============================================================================

@dataclass
class SubtypeDiscoveryResult:
    """Complete results from subtype discovery pipeline."""

    # Core results
    clustering_result: ClusteringResult
    subtype_profiles: List[SubtypeProfile]
    n_subtypes: int

    # Intermediate data
    pathway_scores: PathwayScoreMatrix
    gene_burdens: GeneBurdenMatrix

    # Validation results (optional)
    stability_result: Optional[StabilityResult] = None
    confound_report: Optional[ConfoundReport] = None
    negative_control_report: Optional[NegativeControlReport] = None

    # Processing reports
    qc_report: Optional[QCReport] = None
    validation_report: Optional[ValidationReport] = None

    # Metadata
    provenance: Optional[ProvenanceRecord] = None
    config: Optional[PipelineConfig] = None
    runtime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def summary(self) -> str:
        """Generate a summary of the pipeline results."""
        lines = [
            "=" * 60,
            "SUBTYPE DISCOVERY PIPELINE RESULTS",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Runtime: {self.runtime_seconds:.1f} seconds",
            "",
            "DATA SUMMARY:",
            f"  Samples: {len(self.pathway_scores.samples)}",
            f"  Genes analyzed: {len(self.gene_burdens.genes)}",
            f"  Pathways scored: {len(self.pathway_scores.pathways)}",
            "",
            "CLUSTERING RESULTS:",
            f"  Number of subtypes: {self.n_subtypes}",
            f"  Method: {self.clustering_result.method}",
            f"  Silhouette score: {self.clustering_result.metrics.get('silhouette_score', 'N/A'):.3f}"
            if 'silhouette_score' in self.clustering_result.metrics else "",
        ]

        # Add cluster sizes
        lines.append("  Cluster sizes:")
        for cluster_id, size in sorted(self.clustering_result.cluster_sizes.items()):
            lines.append(f"    Subtype {cluster_id}: {size} samples")

        # Add stability info if available
        if self.stability_result:
            lines.extend([
                "",
                "STABILITY ANALYSIS:",
                f"  Mean ARI: {self.stability_result.mean_ari:.3f}"
                if hasattr(self.stability_result, 'mean_ari') else "",
                f"  Stability rating: {self.stability_result.stability_rating}"
                if hasattr(self.stability_result, 'stability_rating') else "",
            ])

        # Add validation warnings
        if self.confound_report and self.confound_report.overall_risk != "low":
            lines.extend([
                "",
                f"WARNING: Confound risk level: {self.confound_report.overall_risk}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_subtype_assignment(self, sample_id: str) -> Optional[int]:
        """Get the subtype assignment for a specific sample."""
        if sample_id not in self.pathway_scores.sample_index:
            return None
        idx = self.pathway_scores.sample_index[sample_id]
        return int(self.clustering_result.labels[idx])

    def get_samples_by_subtype(self, subtype_id: int) -> List[str]:
        """Get all sample IDs assigned to a specific subtype."""
        mask = self.clustering_result.labels == subtype_id
        return [
            self.pathway_scores.samples[i]
            for i in range(len(mask))
            if mask[i]
        ]


# =============================================================================
# Main Pipeline
# =============================================================================

class SubtypeDiscoveryPipeline:
    """
    End-to-end pipeline for autism subtype discovery.

    Takes VCF data through variant processing, pathway scoring, and clustering
    to identify pathway-based subtypes with stability validation.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: Complete pipeline configuration
        """
        self.config = config
        self._setup_logging()

        # Initialize components (lazy loading)
        self._vcf_loader: Optional[VCFLoader] = None
        self._pathway_db: Optional[PathwayDatabase] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._gene_constraints: Optional[GeneConstraints] = None

    def _setup_logging(self) -> None:
        """Configure logging based on verbosity setting."""
        level = logging.INFO if self.config.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def run(
        self,
        vcf_path: Optional[str] = None,
        confounds: Optional[Dict[str, np.ndarray]] = None,
    ) -> SubtypeDiscoveryResult:
        """
        Execute the complete subtype discovery pipeline.

        Args:
            vcf_path: Optional override for VCF path from config
            confounds: Optional confound variables for validation
                       Dict mapping confound name to array of values per sample

        Returns:
            SubtypeDiscoveryResult with all pipeline outputs
        """
        start_time = datetime.now()
        logger.info("Starting subtype discovery pipeline")

        # Use provided path or config path
        vcf_path = vcf_path or self.config.data.vcf_path

        # Step 1: Load data
        logger.info("Step 1: Loading data")
        variant_dataset, validation_report = self._load_vcf(vcf_path)
        pathway_db = self._load_pathways()

        # Step 2: Process variants
        logger.info("Step 2: Processing variants")
        filtered_dataset, qc_report = self._apply_qc(variant_dataset)
        annotated_variants = self._annotate_variants(filtered_dataset)
        gene_burdens = self._compute_gene_burdens(annotated_variants, filtered_dataset.samples)

        # Step 3: Score pathways
        logger.info("Step 3: Computing pathway scores")
        pathway_scores = self._compute_pathway_scores(gene_burdens, pathway_db)

        # Step 4: Cluster into subtypes
        logger.info("Step 4: Clustering into subtypes")
        clustering_result = self._cluster_samples(pathway_scores)

        # Step 5: Analyze stability (optional)
        stability_result = None
        if self.config.clustering.run_stability:
            logger.info("Step 5: Analyzing cluster stability")
            stability_result = self._analyze_stability(pathway_scores)

        # Step 6: Characterize subtypes
        logger.info("Step 6: Characterizing subtypes")
        subtype_profiles = self._characterize_subtypes(
            clustering_result, pathway_scores
        )

        # Step 7: Validation (optional)
        confound_report = None
        negative_control_report = None

        if self.config.clustering.run_confound_analysis and confounds:
            logger.info("Step 7a: Running confound analysis")
            confound_report = self._analyze_confounds(
                clustering_result.labels, confounds
            )

        if self.config.clustering.run_negative_controls:
            logger.info("Step 7b: Running negative controls")
            negative_control_report = self._run_negative_controls(pathway_scores)

        # Create provenance record
        provenance = self._create_provenance_record()

        # Calculate runtime
        runtime = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pipeline completed in {runtime:.1f} seconds")

        # Assemble results
        result = SubtypeDiscoveryResult(
            clustering_result=clustering_result,
            subtype_profiles=subtype_profiles,
            n_subtypes=clustering_result.n_clusters,
            pathway_scores=pathway_scores,
            gene_burdens=gene_burdens,
            stability_result=stability_result,
            confound_report=confound_report,
            negative_control_report=negative_control_report,
            qc_report=qc_report,
            validation_report=validation_report,
            provenance=provenance,
            config=self.config,
            runtime_seconds=runtime,
        )

        # Save intermediate results if configured
        if self.config.save_intermediate and self.config.output_dir:
            self._save_results(result)

        return result

    # =========================================================================
    # Step 1: Data Loading
    # =========================================================================

    def _load_vcf(self, vcf_path: str) -> Tuple[VariantDataset, ValidationReport]:
        """Load and validate VCF data."""
        self._vcf_loader = VCFLoader(
            min_quality=self.config.processing.min_quality,
            filter_pass_only=self.config.processing.filter_pass_only,
        )

        dataset = self._vcf_loader.load(vcf_path)
        validation_report = self._vcf_loader.validate(dataset)

        logger.info(
            f"Loaded {validation_report.n_variants} variants "
            f"from {validation_report.n_samples} samples"
        )

        return dataset, validation_report

    def _load_pathways(self) -> PathwayDatabase:
        """Load pathway database from configured sources."""
        if self.config.data.pathway_database is not None:
            self._pathway_db = self.config.data.pathway_database
            return self._pathway_db

        loader = PathwayLoader(gene_id_type=self.config.data.gene_id_type)

        # Load from GMT file (Reactome, KEGG, etc.)
        if self.config.data.pathway_gmt_path:
            self._pathway_db = loader.load_gmt(self.config.data.pathway_gmt_path)
            logger.info(f"Loaded {len(self._pathway_db.pathways)} pathways from GMT")

        # Load from GO files
        elif self.config.data.pathway_go_obo_path:
            go_data = loader.load_obo(self.config.data.pathway_go_obo_path)
            if self.config.data.pathway_go_gaf_path:
                self._pathway_db = loader.load_gaf(
                    self.config.data.pathway_go_gaf_path,
                    go_data
                )
            else:
                self._pathway_db = go_data
            logger.info(f"Loaded {len(self._pathway_db.pathways)} GO terms")

        else:
            raise ValueError(
                "No pathway source configured. Provide pathway_gmt_path, "
                "pathway_go_obo_path, or pathway_database"
            )

        # Filter pathways by size
        self._pathway_db = self._pathway_db.filter_by_size(
            min_size=self.config.pathway_scoring.min_pathway_size,
            max_size=self.config.pathway_scoring.max_pathway_size,
        )

        return self._pathway_db

    def _load_knowledge_graph(self) -> Optional[KnowledgeGraph]:
        """Load or build knowledge graph for network propagation."""
        if self.config.data.knowledge_graph is not None:
            self._knowledge_graph = self.config.data.knowledge_graph
            return self._knowledge_graph

        if self.config.data.knowledge_graph_path:
            self._knowledge_graph = KnowledgeGraph.load(
                self.config.data.knowledge_graph_path
            )
            return self._knowledge_graph

        # Build from pathway database if PPI network available
        if self.config.data.ppi_network_path and self._pathway_db:
            builder = KnowledgeGraphBuilder()
            builder.add_genes(list(self._pathway_db.get_all_genes()))
            builder.add_pathways(self._pathway_db)

            # Add PPI if available
            ppi = load_ppi_from_file(
                self.config.data.ppi_network_path,
                min_score=self.config.data.ppi_min_score,
            )
            builder.add_ppi(ppi)

            self._knowledge_graph = builder.build()
            logger.info(
                f"Built knowledge graph with {self._knowledge_graph.n_nodes} nodes, "
                f"{self._knowledge_graph.n_edges} edges"
            )
            return self._knowledge_graph

        return None

    def _load_constraints(self) -> Optional[GeneConstraints]:
        """Load gene constraint scores for variant weighting."""
        if self._gene_constraints is not None:
            return self._gene_constraints

        if not self.config.data.gnomad_constraints_path:
            return None

        loader = ConstraintLoader()
        self._gene_constraints = loader.load_gnomad_constraints(
            self.config.data.gnomad_constraints_path
        )
        logger.info(f"Loaded constraints for {len(self._gene_constraints.gene_ids)} genes")
        return self._gene_constraints

    # =========================================================================
    # Step 2: Variant Processing
    # =========================================================================

    def _apply_qc(self, dataset: VariantDataset) -> Tuple[VariantDataset, QCReport]:
        """Apply quality control filters to variants."""
        qc_config = QCConfig(
            min_quality=self.config.processing.min_quality,
            min_depth=self.config.processing.min_depth,
            filter_pass_only=self.config.processing.filter_pass_only,
            max_allele_freq=self.config.processing.max_allele_freq,
        )

        qc_filter = QCFilter(config=qc_config)
        filtered = qc_filter.filter(dataset)
        report = qc_filter.generate_report()

        logger.info(
            f"QC: {report.output_variants}/{report.input_variants} variants retained "
            f"({report.variant_retention_rate:.1%})"
        )

        return filtered, report

    def _annotate_variants(self, dataset: VariantDataset) -> List[AnnotatedVariant]:
        """Annotate variants with functional consequences."""
        annotator = VariantAnnotator()
        annotated = annotator.annotate_batch(dataset.variants)

        n_coding = sum(1 for v in annotated if v.is_coding)
        n_lof = sum(1 for v in annotated if v.is_lof)

        logger.info(
            f"Annotated {len(annotated)} variants: "
            f"{n_coding} coding, {n_lof} LoF"
        )

        return annotated

    def _compute_gene_burdens(
        self,
        variants: List[AnnotatedVariant],
        samples: List[str],
    ) -> GeneBurdenMatrix:
        """Compute gene burden scores from annotated variants."""
        # Configure weighting
        weight_config = WeightConfig(
            use_cadd_weighting=self.config.processing.use_cadd_weighting,
            cadd_threshold=self.config.processing.cadd_threshold,
            aggregation=self.config.processing.burden_aggregation,
        )

        calculator = GeneBurdenCalculator(config=weight_config)
        burdens = calculator.compute(variants, samples)

        # Apply constraint weighting if configured and available
        if self.config.processing.use_constraint_weighting:
            constraints = self._load_constraints()
            if constraints:
                burdens = self._apply_constraint_weights(burdens, constraints)

        logger.info(
            f"Computed burdens: {burdens.n_samples} samples x {burdens.n_genes} genes"
        )

        return burdens

    def _apply_constraint_weights(
        self,
        burdens: GeneBurdenMatrix,
        constraints: GeneConstraints,
    ) -> GeneBurdenMatrix:
        """Weight gene burdens by constraint scores (pLI)."""
        weighted_scores = burdens.scores.copy()

        for i, gene in enumerate(burdens.genes):
            if gene in constraints.pli_scores:
                pli = constraints.pli_scores[gene]
                # Weight by pLI (constrained genes get higher weight)
                weighted_scores[:, i] *= (1.0 + pli)

        return GeneBurdenMatrix(
            samples=burdens.samples,
            genes=burdens.genes,
            scores=weighted_scores,
            sample_index=burdens.sample_index,
            gene_index=burdens.gene_index,
            contributing_variants=burdens.contributing_variants,
        )

    # =========================================================================
    # Step 3: Pathway Scoring
    # =========================================================================

    def _compute_pathway_scores(
        self,
        gene_burdens: GeneBurdenMatrix,
        pathway_db: PathwayDatabase,
    ) -> PathwayScoreMatrix:
        """Aggregate gene burdens to pathway scores."""
        # Configure aggregation
        agg_config = AggregationConfig(
            method=self.config.pathway_scoring.aggregation_method,
            min_pathway_size=self.config.pathway_scoring.min_pathway_size,
            max_pathway_size=self.config.pathway_scoring.max_pathway_size,
            normalize_by_pathway_size=self.config.pathway_scoring.normalize_by_pathway_size,
        )

        aggregator = PathwayAggregator(config=agg_config)

        # Optionally use constraint weights
        gene_weights = None
        if self._gene_constraints and self.config.processing.use_constraint_weighting:
            gene_weights = {
                gene: self._gene_constraints.pli_scores.get(gene, 0.5)
                for gene in gene_burdens.genes
            }

        pathway_scores = aggregator.aggregate(
            gene_burdens, pathway_db, gene_weights=gene_weights
        )

        # Apply network propagation if configured
        if self.config.pathway_scoring.use_network_propagation:
            pathway_scores = self._apply_network_propagation(
                gene_burdens, pathway_scores
            )

        # Normalize scores
        normalizer = PathwayScoreNormalizer(
            config=NormalizationConfig(
                method=self.config.pathway_scoring.normalization_method
            )
        )
        pathway_scores = normalizer.normalize(pathway_scores)

        logger.info(
            f"Scored {pathway_scores.n_pathways} pathways for {pathway_scores.n_samples} samples"
        )

        return pathway_scores

    def _apply_network_propagation(
        self,
        gene_burdens: GeneBurdenMatrix,
        pathway_scores: PathwayScoreMatrix,
    ) -> PathwayScoreMatrix:
        """Apply network propagation to refine scores."""
        kg = self._load_knowledge_graph()
        if kg is None:
            logger.warning(
                "Network propagation requested but no knowledge graph available. "
                "Skipping propagation."
            )
            return pathway_scores

        propagator = NetworkPropagator(
            config=PropagationConfig(
                method=self.config.pathway_scoring.propagation_method,
                restart_prob=self.config.pathway_scoring.restart_prob,
            )
        )
        propagator.build_network(kg)

        result = propagator.propagate_gene_burdens(gene_burdens)
        logger.info(
            f"Network propagation converged: {result.converged} "
            f"in {result.n_iterations} iterations"
        )

        # Re-aggregate with propagated scores
        agg_config = AggregationConfig(
            method=self.config.pathway_scoring.aggregation_method,
            normalize_by_pathway_size=self.config.pathway_scoring.normalize_by_pathway_size,
        )
        aggregator = PathwayAggregator(config=agg_config)

        # Use propagated scores
        return aggregator.aggregate(
            result.propagated_scores, self._pathway_db
        )

    # =========================================================================
    # Step 4: Clustering
    # =========================================================================

    def _cluster_samples(self, pathway_scores: PathwayScoreMatrix) -> ClusteringResult:
        """Cluster samples into subtypes based on pathway scores."""
        cluster_config = ClusteringConfig(
            method=self.config.clustering.method,
            n_clusters=self.config.clustering.n_clusters,
            min_clusters=self.config.clustering.min_clusters,
            max_clusters=self.config.clustering.max_clusters,
            random_state=self.config.random_state,
        )

        clusterer = SubtypeClusterer(config=cluster_config)
        result = clusterer.fit(pathway_scores)

        logger.info(
            f"Clustering complete: {result.n_clusters} clusters identified "
            f"(silhouette: {result.metrics.get('silhouette_score', 0):.3f})"
        )

        return result

    # =========================================================================
    # Step 5: Stability Analysis
    # =========================================================================

    def _analyze_stability(
        self,
        pathway_scores: PathwayScoreMatrix,
    ) -> StabilityResult:
        """Analyze cluster stability through bootstrap resampling."""
        stability_config = StabilityConfig(
            n_bootstrap=self.config.clustering.n_bootstrap,
            random_state=self.config.random_state,
        )

        analyzer = StabilityAnalyzer(config=stability_config)
        result = analyzer.analyze(
            pathway_scores,
            n_bootstrap=self.config.clustering.n_bootstrap,
        )

        if hasattr(result, 'stability_rating'):
            logger.info(f"Stability rating: {result.stability_rating}")

        return result

    # =========================================================================
    # Step 6: Characterization
    # =========================================================================

    def _characterize_subtypes(
        self,
        clustering_result: ClusteringResult,
        pathway_scores: PathwayScoreMatrix,
    ) -> List[SubtypeProfile]:
        """Characterize each subtype by its pathway signatures."""
        char_config = CharacterizationConfig(
            n_top_pathways=self.config.clustering.n_top_pathways,
            min_fold_change=self.config.clustering.min_fold_change,
            use_fdr_correction=self.config.clustering.use_fdr_correction,
        )

        characterizer = SubtypeCharacterizer(config=char_config)
        profiles = characterizer.characterize(
            clustering_result=clustering_result,
            pathway_scores=pathway_scores.scores,
            pathway_ids=pathway_scores.pathways,
            pathway_names=pathway_scores.pathway_names,
            gene_contributions=pathway_scores.contributing_genes,
        )

        for profile in profiles:
            logger.info(
                f"Subtype {profile.subtype_id}: {profile.n_samples} samples, "
                f"top pathway: {profile.top_pathways[0].pathway_name if profile.top_pathways else 'N/A'}"
            )

        return profiles

    # =========================================================================
    # Step 7: Validation
    # =========================================================================

    def _analyze_confounds(
        self,
        labels: np.ndarray,
        confounds: Dict[str, np.ndarray],
    ) -> ConfoundReport:
        """Test for confounding variables affecting clusters."""
        analyzer = ConfoundAnalyzer(
            config=ConfoundAnalyzerConfig(
                use_bonferroni_correction=True,
            )
        )

        report = analyzer.test_cluster_confound_alignment(labels, confounds)

        if report.overall_risk != "low":
            logger.warning(
                f"Confound analysis warning: {report.overall_risk} risk level detected"
            )

        return report

    def _run_negative_controls(
        self,
        pathway_scores: PathwayScoreMatrix,
    ) -> NegativeControlReport:
        """Run negative control tests to validate clustering signal."""
        nc_config = NegativeControlConfig(
            n_permutations=self.config.clustering.n_permutations,
            random_state=self.config.random_state,
        )

        runner = NegativeControlRunner(config=nc_config)
        clusterer = SubtypeClusterer(
            config=ClusteringConfig(
                method=self.config.clustering.method,
                n_clusters=self.config.clustering.n_clusters,
            )
        )

        report = runner.run_full_negative_control(
            pathway_scores.scores,
            clusterer,
        )

        if hasattr(report, 'permutation_result') and report.permutation_result:
            logger.info(
                f"Negative control p-value: {report.permutation_result.p_value:.4f}"
            )

        return report

    # =========================================================================
    # Utilities
    # =========================================================================

    def _create_provenance_record(self) -> ProvenanceRecord:
        """Create a provenance record for reproducibility."""
        pathway_versions = {}
        if self._pathway_db:
            pathway_versions[self._pathway_db.source] = "loaded"

        return ProvenanceRecord(
            reference_genome="GRCh38",  # Default assumption
            annotation_version="VEP_v110",
            pathway_db_versions=pathway_versions,
            pipeline_version="1.0.0",
            timestamp=datetime.now(),
        )

    def _save_results(self, result: SubtypeDiscoveryResult) -> None:
        """Save pipeline results to output directory."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save pathway scores as DataFrame
        pathway_df = result.pathway_scores.to_dataframe()
        pathway_df.to_csv(output_dir / "pathway_scores.csv")

        # Save cluster assignments
        assignments = {
            "sample_id": result.pathway_scores.samples,
            "cluster": result.clustering_result.labels.tolist(),
        }
        import json
        with open(output_dir / "cluster_assignments.json", "w") as f:
            json.dump(assignments, f, indent=2)

        # Save summary
        with open(output_dir / "summary.txt", "w") as f:
            f.write(result.summary)

        logger.info(f"Results saved to {output_dir}")


# =============================================================================
# Convenience Functions
# =============================================================================

def run_subtype_discovery(
    vcf_path: str,
    pathway_path: str,
    n_clusters: Optional[int] = None,
    output_dir: Optional[str] = None,
    **kwargs: Any,
) -> SubtypeDiscoveryResult:
    """
    Convenience function to run subtype discovery with minimal configuration.

    Args:
        vcf_path: Path to VCF file
        pathway_path: Path to pathway GMT file
        n_clusters: Number of clusters (auto-detect if None)
        output_dir: Optional output directory
        **kwargs: Additional configuration options

    Returns:
        SubtypeDiscoveryResult with pipeline outputs
    """
    config = PipelineConfig(
        data=DataConfig(
            vcf_path=vcf_path,
            pathway_gmt_path=pathway_path,
        ),
        clustering=ClusteringPipelineConfig(
            n_clusters=n_clusters,
            run_stability=True,
        ),
        output_dir=output_dir,
        save_intermediate=output_dir is not None,
    )

    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config.processing, key):
            setattr(config.processing, key, value)
        elif hasattr(config.clustering, key):
            setattr(config.clustering, key, value)

    pipeline = SubtypeDiscoveryPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Run autism subtype discovery pipeline"
    )
    parser.add_argument("--vcf", required=True, help="Path to VCF file")
    parser.add_argument("--pathways", required=True, help="Path to pathway GMT file")
    parser.add_argument("--n-clusters", type=int, help="Number of clusters")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    result = run_subtype_discovery(
        vcf_path=args.vcf,
        pathway_path=args.pathways,
        n_clusters=args.n_clusters,
        output_dir=args.output,
    )

    print(result.summary)
