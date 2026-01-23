# Module 11: Therapeutic Hypotheses

Drug-pathway mapping and therapeutic hypothesis generation for autism genetics analysis.

## Overview

This module provides:

1. **DrugTargetDatabase** - Curated drug-target-pathway relationships
2. **PathwayDrugMapper** - Maps disrupted pathways to potential drug candidates
3. **HypothesisRanker** - Ranks therapeutic hypotheses by evidence
4. **EvidenceScorer** - Scores hypotheses on biological plausibility and safety

## Architecture

```
                    ┌──────────────────────┐
                    │  Disrupted Pathways  │
                    │  (from Module 07/09) │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  PathwayDrugMapper   │
                    │                      │
                    │  - Drug-target DB    │
                    │  - Mechanism match   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Drug Candidates    │
                    │                      │
                    │  - Target genes      │
                    │  - Mechanisms        │
                    │  - Known indications │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   EvidenceScorer     │
                    │                      │
                    │  - Bio plausibility  │
                    │  - Literature        │
                    │  - Safety flags      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   HypothesisRanker   │
                    │                      │
                    │  - Score combination │
                    │  - Confidence calc   │
                    │  - Explanations      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ TherapeuticHypothesis│
                    │                      │
                    │  - Ranked list       │
                    │  - Evidence scores   │
                    │  - Explanations      │
                    │  - Safety warnings   │
                    └──────────────────────┘
```

## Dependencies

- **Module 07**: `PathwayScoreMatrix` for disrupted pathways
- **Module 09**: `FiredRule`, rule conclusions for evidence

## Usage

### Basic Usage

```python
from modules.11_therapeutic_hypotheses import (
    DrugTargetDatabase,
    PathwayDrugMapper,
    HypothesisRanker,
    EvidenceScorer,
    TherapeuticHypothesis,
)

# Load drug-target database
db = DrugTargetDatabase()
db.load_drugbank("path/to/drugbank.csv")
db.load_pathway_associations("path/to/drug_pathways.csv")

# Create mapper and scorer
mapper = PathwayDrugMapper(db)
scorer = EvidenceScorer()
ranker = HypothesisRanker(mapper, scorer)

# Get disrupted pathways (from Module 07)
disrupted_pathways = {
    "synaptic_transmission": 2.5,  # z-score
    "chromatin_remodeling": 1.8,
}

# Generate ranked hypotheses
hypotheses = ranker.rank(
    pathway_scores=disrupted_pathways,
    min_pathway_zscore=1.5,
)

# Review results
for hyp in hypotheses[:10]:
    print(f"{hyp.drug_name}: {hyp.score:.2f}")
    print(f"  Target: {hyp.target_pathway}")
    print(f"  Mechanism: {hyp.mechanism}")
    print(f"  Safety: {hyp.evidence.safety_flags}")
```

### With Rule-Based Evidence

```python
from modules.09_symbolic_rules import RuleEngine, BiologicalRules

# Get fired rules from rule engine
rule_engine = RuleEngine(BiologicalRules.get_all_rules(), context)
fired_rules = rule_engine.evaluate(individual)

# Include rule evidence in scoring
hypotheses = ranker.rank(
    pathway_scores=disrupted_pathways,
    fired_rules=fired_rules,  # Adds rule-based confidence
)
```

## Interface Contract

### Data Structures

```python
@dataclass
class DrugCandidate:
    drug_id: str
    drug_name: str
    target_genes: List[str]
    mechanism: str
    indications: List[str]
    contraindications: List[str]
    asd_relevance_score: float  # 0-1

@dataclass
class EvidenceScore:
    biological_plausibility: float  # 0-1
    mechanistic_alignment: float    # 0-1
    literature_support: float       # 0-1
    safety_flags: List[str]
    overall: float                  # Combined score

@dataclass
class TherapeuticHypothesis:
    drug_id: str
    drug_name: str
    target_pathway: str
    target_genes: List[str]
    mechanism: str
    score: float
    evidence: EvidenceScore
    explanation: str
    confidence: float
    requires_validation: bool = True  # ALWAYS True
```

### Key Classes

| Class | Description |
|-------|-------------|
| `DrugTargetDatabase` | Stores drug-target-pathway relationships |
| `PathwayDrugMapper` | Maps pathways to drug candidates |
| `EvidenceScorer` | Scores hypotheses on multiple criteria |
| `HypothesisRanker` | Ranks and explains hypotheses |

## Safety and Disclaimers

**CRITICAL**: All outputs from this module are **HYPOTHESES ONLY** and require:

1. **Clinical validation** before any therapeutic consideration
2. **Expert review** by domain specialists
3. **Safety assessment** by qualified professionals
4. **Regulatory approval** for any clinical application

The `requires_validation` flag is **always True** and cannot be changed.

## Testing

```bash
# Run module tests
source autismenv/bin/activate && python3 -m pytest modules/11_therapeutic_hypotheses/tests/ -v

# Run specific test
source autismenv/bin/activate && python3 -m pytest modules/11_therapeutic_hypotheses/tests/test_ranking.py -v
```

## Files

```
11_therapeutic_hypotheses/
├── README.md
├── __init__.py
├── pathway_drug_mapping.py  # DrugTargetDatabase, PathwayDrugMapper
├── evidence.py              # EvidenceScorer, EvidenceScore
├── ranking.py               # HypothesisRanker, TherapeuticHypothesis
└── tests/
    ├── __init__.py
    ├── test_mapping.py
    ├── test_evidence.py
    └── test_ranking.py
```
