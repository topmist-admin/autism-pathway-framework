"""
Module 09: Symbolic Rules

Rule-based biological inference for autism genetics analysis.

This module provides:
- Condition evaluators for biological predicates
- Curated biological rules (R1-R7) encoding autism genetics knowledge
- Rule engine for evaluating rules against individual data
- Explanation generation for reasoning chains

Example usage:
    from modules.09_symbolic_rules import (
        RuleEngine,
        BiologicalRules,
        BiologicalContext,
        IndividualData,
        ExplanationGenerator,
    )

    # Set up biological context
    context = BiologicalContext(
        gene_constraints=constraints,
        developmental_expression=expression,
        sfari_genes=sfari,
    )

    # Create rule engine
    rules = BiologicalRules.get_all_rules()
    engine = RuleEngine(rules, context)

    # Evaluate rules
    individual = IndividualData(sample_id="S1", variants=variants)
    fired_rules = engine.evaluate(individual)

    # Generate explanation
    generator = ExplanationGenerator()
    chain = generator.generate_reasoning_chain("S1", fired_rules)
    print(chain.explanation_text)
"""

import sys
from pathlib import Path

# Add module to path for imports
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

# Condition data structures and evaluators
from conditions import (
    Condition,
    ConditionType,
    ConditionResult,
    ConditionEvaluator,
)

# Rule definitions
from biological_rules import (
    Rule,
    Conclusion,
    ConclusionType,
    BiologicalRules,
)

# Rule engine
from rule_engine import (
    RuleEngine,
    FiredRule,
    IndividualData,
    BiologicalContext,
)

# Explanation generation
from explanation import (
    ReasoningChain,
    ReasoningStep,
    ExplanationGenerator,
    format_fired_rule,
    format_rule_summary,
)

__all__ = [
    # Conditions
    "Condition",
    "ConditionType",
    "ConditionResult",
    "ConditionEvaluator",
    # Rules
    "Rule",
    "Conclusion",
    "ConclusionType",
    "BiologicalRules",
    # Engine
    "RuleEngine",
    "FiredRule",
    "IndividualData",
    "BiologicalContext",
    # Explanations
    "ReasoningChain",
    "ReasoningStep",
    "ExplanationGenerator",
    "format_fired_rule",
    "format_rule_summary",
]

__version__ = "1.0.0"
