# Module 10: Neurosymbolic Integration

Neural-symbolic integration combining graph neural network predictions with rule-based biological inference for autism genetics analysis.

## Overview

This module provides:

1. **NeuroSymbolicModel** - Unified model combining neural and symbolic components
2. **LearnedCombiner** - Learned strategies for combining neural and symbolic predictions
3. **Explanation Generation** - Combined explanations showing both contributions

## Architecture

```
                    ┌─────────────────┐
                    │ Individual Data │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  Neural Pathway │           │ Symbolic Pathway │
    │  (Module 06)    │           │  (Module 09)    │
    │                 │           │                 │
    │  GNN Embeddings │           │  Rule Engine    │
    │  → Gene Scores  │           │  → Fired Rules  │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             │    ┌─────────────────┐      │
             └───▶│ LearnedCombiner │◀─────┘
                  │                 │
                  │ - Attention     │
                  │ - Gating        │
                  │ - Learned Weights│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ NeuroSymbolic   │
                  │ Output          │
                  │                 │
                  │ - Predictions   │
                  │ - Contributions │
                  │ - Explanations  │
                  └─────────────────┘
```

## Dependencies

- **Module 06**: `OntologyAwareGNN`, `GNNConfig`, `GNNOutput`
- **Module 09**: `RuleEngine`, `BiologicalContext`, `FiredRule`, `IndividualData`

## Usage

### Basic Usage

```python
from modules.10_neurosymbolic import (
    NeuroSymbolicModel,
    NeuroSymbolicConfig,
    NeuroSymbolicOutput,
)

# Initialize with neural and symbolic components
model = NeuroSymbolicModel(
    neural_model=gnn_model,
    rule_engine=rule_engine,
    config=NeuroSymbolicConfig(
        combination_method="attention",
        neural_weight=0.6,
    ),
)

# Process individual
output = model.forward(
    individual_data=individual,
    graph_data=graph_data,
)

# Access combined predictions
print(output.predictions)
print(output.neural_contribution)
print(output.symbolic_contribution)
print(output.explanation)
```

### Learned Combination

```python
from modules.10_neurosymbolic import LearnedCombiner, CombinationMethod

# Create combiner with learned weights
combiner = LearnedCombiner(
    method=CombinationMethod.ATTENTION,
    neural_dim=128,
    symbolic_dim=64,
    output_dim=64,
)

# Combine predictions
combined = combiner(neural_scores, symbolic_scores)
```

### Training Mode

```python
# Train the combiner
combiner.train()
output = model.forward(individual_data, graph_data)

# Compute loss and backprop
loss = output.loss
loss.backward()
optimizer.step()
```

## Interface Contract

### Data Structures

```python
@dataclass
class NeuroSymbolicConfig:
    combination_method: str = "attention"  # "weighted_sum", "attention", "gating", "learned"
    neural_weight: float = 0.6
    symbolic_weight: float = 0.4
    temperature: float = 1.0
    use_rule_confidence: bool = True
    normalize_outputs: bool = True

@dataclass
class NeuroSymbolicOutput:
    predictions: Dict[str, float]  # Gene -> combined score
    neural_contribution: Dict[str, float]  # Gene -> neural contribution
    symbolic_contribution: Dict[str, float]  # Gene -> symbolic contribution
    fired_rules: List[FiredRule]  # Rules that fired
    explanation: str  # Combined explanation
    neural_embeddings: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
```

### Combination Methods

| Method | Description |
|--------|-------------|
| `weighted_sum` | Fixed-weight linear combination |
| `attention` | Attention-based combination using gene features |
| `gating` | Learned gating network |
| `learned` | Fully learned MLP combiner |
| `rule_guided` | Symbolic rules guide neural attention |

## Testing

```bash
# Run module tests
source autismenv/bin/activate && python3 -m pytest modules/10_neurosymbolic/tests/ -v

# Run specific test
source autismenv/bin/activate && python3 -m pytest modules/10_neurosymbolic/tests/test_integration.py -v
```

## Architecture Details

### Neural Pathway (from Module 06)

1. Takes graph data (node features, edges, biological priors)
2. Produces gene embeddings via GNN message passing
3. Gene classification scores via task head

### Symbolic Pathway (from Module 09)

1. Takes individual variant data
2. Evaluates biological rules (R1-R7)
3. Produces fired rules with confidence scores

### Combination Strategies

1. **Weighted Sum**: `output = α * neural + (1-α) * symbolic`
2. **Attention**: Learn attention weights based on features
3. **Gating**: Sigmoid gate: `output = g * neural + (1-g) * symbolic`
4. **Rule-Guided**: Use fired rules to guide neural attention

## Files

```
10_neurosymbolic/
├── README.md
├── __init__.py
├── integration.py      # NeuroSymbolicModel, NeuroSymbolicOutput
├── combiner.py         # LearnedCombiner, combination strategies
└── tests/
    ├── __init__.py
    └── test_integration.py
```
