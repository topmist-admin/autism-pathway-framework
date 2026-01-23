# Module 09: Symbolic Rules

Rule-based biological inference for autism genetics analysis. This module implements curated biological rules (R1-R6) that encode domain knowledge about autism-relevant genetic mechanisms.

## Overview

The symbolic rules module provides:

1. **Condition Evaluators** - Predicates for testing biological conditions
2. **Biological Rules (R1-R6)** - Curated autism-specific inference rules
3. **Rule Engine** - Evaluation and inference system
4. **Explanation Generation** - Human-readable reasoning chains

## Rules Summary

| Rule | Name | Description |
|------|------|-------------|
| R1 | Constrained LoF in Developing Cortex | LoF in constrained gene with prenatal cortical expression → high-confidence pathway disruption |
| R2 | Pathway Convergence | Multiple hits (≥2 genes) in same pathway → strong convergence signal |
| R3 | CHD8 Chromatin Cascade | Disruption in CHD8 or targets → chromatin regulation cascade |
| R4 | Synaptic Excitatory Disruption | Synaptic gene + excitatory neuron expression → synaptic subtype indicator |
| R5 | Paralog Compensation | Intact expressed paralog → potential compensation (reduced penetrance) |
| R6 | Therapeutic Pathway Target | Drug targets disrupted pathway → therapeutic hypothesis candidate |

## Dependencies

- **Module 01**: `GeneConstraints`, `DevelopmentalExpression`, `SingleCellAtlas`, `SFARIGenes`
- **Module 02**: `AnnotatedVariant`, `GeneBurdenMatrix`
- **Module 07**: `PathwayScoreMatrix`

## Usage

### Basic Rule Evaluation

```python
from modules.09_symbolic_rules import (
    RuleEngine,
    BiologicalRules,
    BiologicalContext,
    IndividualData
)

# Load biological context (reference data)
context = BiologicalContext(
    gene_constraints=gene_constraints,
    developmental_expression=dev_expression,
    single_cell_atlas=cell_atlas,
    sfari_genes=sfari_genes,
    pathway_db=pathway_db,
)

# Create rule engine with all curated rules
rules = BiologicalRules.get_all_rules()
engine = RuleEngine(rules=rules, biological_context=context)

# Prepare individual data
individual = IndividualData(
    sample_id="SAMPLE_001",
    variants=annotated_variants,
    gene_burdens=burden_matrix.get_sample("SAMPLE_001"),
    pathway_scores=pathway_scores.get_sample("SAMPLE_001"),
)

# Evaluate rules
fired_rules = engine.evaluate(individual)

# Get reasoning chain with explanations
reasoning = engine.get_reasoning_chain(fired_rules)
print(reasoning.explanation_text)
```

### Batch Evaluation

```python
# Evaluate across cohort
cohort_results = engine.evaluate_batch(cohort_data)

# Get summary statistics
for sample_id, fired_rules in cohort_results.items():
    print(f"{sample_id}: {len(fired_rules)} rules fired")
```

### Custom Rule Creation

```python
from modules.09_symbolic_rules import Rule, Condition, Conclusion

custom_rule = Rule(
    id="CUSTOM_1",
    name="Custom Synaptic Rule",
    description="Custom rule for synaptic gene analysis",
    conditions=[
        Condition("has_variant", {"variant_type": "damaging"}),
        Condition("is_synaptic_gene", {"ontology": "SynGO"}),
    ],
    conclusion=Conclusion(
        type="subtype_indicator",
        attributes={"subtype": "synaptic_dysfunction"},
    ),
    base_confidence=0.75,
    evidence_sources=["SynGO database"],
)
```

## Interface Contract

### Data Structures

```python
@dataclass
class Condition:
    predicate: str  # e.g., "has_lof_variant", "is_constrained"
    arguments: Dict[str, Any]
    negated: bool = False

@dataclass
class Conclusion:
    type: str  # "pathway_disruption", "subtype_indicator", "therapeutic_candidate"
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
    evidence_sources: List[str]
    logic: str = "AND"  # "AND" or "OR" for combining conditions

@dataclass
class FiredRule:
    rule: Rule
    bindings: Dict[str, Any]  # Variable assignments
    confidence: float
    explanation: str
    evidence: Dict[str, Any]

@dataclass
class ReasoningChain:
    individual_id: str
    fired_rules: List[FiredRule]
    pathway_conclusions: Dict[str, float]
    subtype_indicators: List[str]
    therapeutic_hypotheses: List[Dict]
    explanation_text: str
```

## Testing

```bash
# Run module tests
python -m pytest modules/09_symbolic_rules/tests/ -v

# Run specific test
python -m pytest modules/09_symbolic_rules/tests/test_rules.py -v
```

## Architecture

```
09_symbolic_rules/
├── README.md
├── __init__.py
├── conditions.py          # Condition data structures and evaluators
├── biological_rules.py    # R1-R6 rule definitions
├── rule_engine.py         # Rule evaluation engine
├── explanation.py         # Reasoning chain generation
└── tests/
    ├── __init__.py
    ├── test_conditions.py
    ├── test_rules.py
    ├── test_rule_engine.py
    └── test_explanation.py
```

## Evidence Sources

Rules are backed by published research:

- **R1**: gnomAD constraint scores, BrainSpan developmental expression
- **R2**: Pathway databases (GO, Reactome, KEGG)
- **R3**: Cotney et al. 2015, Sugathan et al. 2014 (CHD8 targets)
- **R4**: SynGO synaptic ontology, single-cell expression atlases
- **R5**: Paralog databases (Ensembl), expression data
- **R6**: DrugBank, pathway-drug mappings

## Integration with Pipelines

This module is integrated into the framework's therapeutic hypothesis pipeline:

### TherapeuticHypothesisPipeline

Rules R1-R6 are automatically applied to each individual in the cohort:

```python
from pipelines import TherapeuticHypothesisPipeline, TherapeuticPipelineConfig, DataConfig

config = TherapeuticPipelineConfig(
    data=DataConfig(vcf_path="cohort.vcf.gz", pathway_gmt_path="reactome.gmt"),
    therapeutic=TherapeuticConfig(enable_rules=True),  # Enable symbolic rules
)
pipeline = TherapeuticHypothesisPipeline(config)
result = pipeline.run()

# Access per-individual rule analysis
for sample_id, analysis in result.individual_analyses.items():
    print(f"{sample_id}: {analysis.n_rules_fired} rules fired")
    for rule in analysis.fired_rules:
        print(f"  - {rule.rule.name}: {rule.explanation}")
```

See [pipelines/README.md](../../pipelines/README.md) for complete pipeline documentation
