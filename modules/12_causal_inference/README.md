# Module 12: Causal Inference

Causal reasoning framework for understanding genetic mechanisms and intervention effects in ASD.

## Overview

This module implements Pearl's causal inference framework adapted for autism genetics research. It enables:

- **Structural Causal Models (SCM)**: Encode causal relationships from genetic variants through pathways to phenotypes
- **Do-Calculus**: Reason about interventions (e.g., "What if we target this pathway?")
- **Counterfactual Queries**: Answer "what if" questions about individual cases
- **Effect Estimation**: Quantify direct, indirect, and mediated causal effects

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Causal Inference Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Knowledge Graph ──────┐                                                │
│                        │                                                │
│  Pathway Scores ───────┼──▶ StructuralCausalModel                       │
│                        │         │                                      │
│  Variant Data ─────────┘         │                                      │
│                                  ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Causal Query Engine                            │  │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │  │
│  │  │  DoCalculus     │  │ Counterfactual   │  │ Effect           │  │  │
│  │  │  Engine         │  │ Engine           │  │ Estimator        │  │  │
│  │  │                 │  │                  │  │                  │  │  │
│  │  │ • do(X=x)       │  │ • Abduction      │  │ • Total Effect   │  │  │
│  │  │ • P(Y|do(X))    │  │ • Action         │  │ • Direct Effect  │  │  │
│  │  │ • ATE/CATE      │  │ • Prediction     │  │ • Indirect Effect│  │  │
│  │  │                 │  │                  │  │ • Mediation      │  │  │
│  │  └─────────────────┘  └──────────────────┘  └──────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                  │                                      │
│                                  ▼                                      │
│              ┌───────────────────────────────────────┐                  │
│              │         Causal Query Results          │                  │
│              │  • Intervention effects               │                  │
│              │  • Counterfactual outcomes            │                  │
│              │  • Mediation proportions              │                  │
│              └───────────────────────────────────────┘                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Causal Chain Model

The module encodes the ASD causal chain:

```
Genetic Variants → Gene Function Disruption → Pathway Perturbation
                → Circuit-Level Effects → Behavioral Phenotype

With explicit confounders:
• Ancestry (population stratification)
• Batch effects (technical confounders)
• Ascertainment bias (diagnostic confounders)
```

## Components

### 1. causal_graph.py - Structural Causal Model

Core SCM implementation with:
- Node types: VARIANT, GENE_FUNCTION, PATHWAY, CIRCUIT, PHENOTYPE, CONFOUNDER
- Edge types: CAUSES, MEDIATES, CONFOUNDS, MODIFIES
- D-separation testing for conditional independence
- Backdoor path identification
- Valid adjustment set computation

### 2. do_calculus.py - Intervention Reasoning

Pearl's do-calculus implementation:
- `do(X=x)`: Apply intervention (graph surgery)
- `P(Y|do(X))`: Interventional probability queries
- Average Treatment Effect (ATE)
- Conditional Average Treatment Effect (CATE)

### 3. counterfactuals.py - Counterfactual Queries

Three-step counterfactual algorithm:
1. **Abduction**: Infer exogenous variables from evidence
2. **Action**: Apply counterfactual intervention
3. **Prediction**: Compute outcome in modified model

Also includes:
- Probability of Necessity: P(Y₀=0 | T=1, Y=1)
- Probability of Sufficiency: P(Y₁=1 | T=0, Y=0)

### 4. effect_estimation.py - Mediation Analysis

Causal effect decomposition:
- Total effect: Overall causal impact
- Natural Direct Effect: Effect not through mediator
- Natural Indirect Effect: Effect through mediator
- Proportion mediated: How much effect is mediated

## Usage Example

```python
from causal_graph import StructuralCausalModel, CausalNode, CausalEdge, CausalNodeType, CausalEdgeType
from do_calculus import DoCalculusEngine
from counterfactuals import CounterfactualEngine
from effect_estimation import CausalEffectEstimator

# Build SCM
scm = StructuralCausalModel()

# Add nodes representing the causal chain
scm.add_node(CausalNode("SHANK3_variant", CausalNodeType.VARIANT, observed=True))
scm.add_node(CausalNode("SHANK3_function", CausalNodeType.GENE_FUNCTION, observed=True))
scm.add_node(CausalNode("synaptic_pathway", CausalNodeType.PATHWAY, observed=True))
scm.add_node(CausalNode("asd_phenotype", CausalNodeType.PHENOTYPE, observed=True))

# Add causal edges
scm.add_edge(CausalEdge("SHANK3_variant", "SHANK3_function", CausalEdgeType.CAUSES, 0.9, "LoF"))
scm.add_edge(CausalEdge("SHANK3_function", "synaptic_pathway", CausalEdgeType.CAUSES, 0.8, "scaffold"))
scm.add_edge(CausalEdge("synaptic_pathway", "asd_phenotype", CausalEdgeType.CAUSES, 0.7, "transmission"))

# Do-calculus: What if we restore SHANK3 function?
do_engine = DoCalculusEngine(scm)
effect = do_engine.query(
    outcome="asd_phenotype",
    intervention={"SHANK3_function": 1.0}
)

# Counterfactual: For an individual with SHANK3 mutation and ASD,
# what would phenotype be if SHANK3 were intact?
cf_engine = CounterfactualEngine(scm)
cf_result = cf_engine.counterfactual(
    factual_evidence={"SHANK3_function": 0, "asd_phenotype": 1},
    counterfactual_intervention={"SHANK3_function": 1},
    query_variable="asd_phenotype"
)

# Mediation analysis: How much of the effect is through synaptic pathway?
estimator = CausalEffectEstimator(scm, do_engine)
mediation = estimator.mediation_analysis(
    treatment="SHANK3_function",
    outcome="asd_phenotype",
    mediator="synaptic_pathway"
)
print(f"Proportion mediated: {mediation.proportion_mediated:.2%}")
```

## Causal Query Builder

Fluent interface for building queries:

```python
from causal_graph import CausalQueryBuilder

query = (CausalQueryBuilder()
    .treatment("CHD8_function")
    .outcome("asd_phenotype")
    .mediated_by("chromatin_pathway")
    .do({"CHD8_function": 0})
    .build())
```

## Research Context

This module enables researchers to:

1. **Validate pathway hypotheses**: Test if pathway disruption causally affects phenotype
2. **Identify mediating mechanisms**: Determine how genetic effects are transmitted
3. **Guide therapeutic targets**: Find intervention points with largest causal effects
4. **Support personalized reasoning**: Answer individual "what if" questions

## Limitations

- Causal conclusions depend on the correctness of the assumed causal structure
- Effect estimates are approximate without individual-level experimental data
- Model assumes no unmeasured confounders (adjustable variables are explicit)
- Results are hypotheses requiring experimental validation

## Dependencies

- Module 03: Knowledge Graph (for constructing causal structure)
- Module 07: Pathway Scoring (for observed pathway states)
- Module 09: Symbolic Rules (for biological constraints)

## Testing

```bash
cd modules/12_causal_inference
python -m pytest tests/ -v
```
