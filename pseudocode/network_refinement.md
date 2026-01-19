# Pseudocode: Network Refinement

## Overview

This module refines gene-level and pathway-level scores using gene-gene interaction networks. Network propagation allows signals to spread through connected genes, amplifying coherent biological signals while reducing noise.

---

## Input

```
gene_scores: Map[sample_id → Map[gene_id → score]]
    # Gene-level burden scores from variant_to_gene

network: Graph
    # Gene-gene interaction network
    # Nodes: genes
    # Edges: interactions with confidence weights

pathway_scores: Map[sample_id → Map[pathway_id → score]] (optional)
    # Pathway scores to refine
```

## Output

```
propagated_gene_scores: Map[sample_id → Map[gene_id → propagated_score]]
refined_pathway_scores: Map[sample_id → Map[pathway_id → refined_score]]
```

---

## Constants and Configuration

```python
# Network propagation parameters
RESTART_PROBABILITY = 0.5    # α: balance between prior and network
MAX_ITERATIONS = 100         # Convergence limit
CONVERGENCE_THRESHOLD = 1e-6 # Stop when change < threshold

# Network filtering
MIN_EDGE_CONFIDENCE = 0.4    # Minimum edge weight to include
MIN_NODE_DEGREE = 1          # Exclude isolated nodes

# Hub correction
APPLY_HUB_CORRECTION = True  # Reduce bias toward high-degree genes
HUB_PENALTY_POWER = 0.5      # Degree penalty exponent
```

---

## Algorithm

### Step 1: Load and Prepare Network

```python
function load_interaction_network(network_file):
    """
    Load gene-gene interaction network.

    Input format (edge list):
        GENE_A  GENE_B  CONFIDENCE
        SHANK3  NRXN1   0.9
        ...
    """
    edges = []

    for line in read_file(network_file):
        gene_a, gene_b, confidence = parse_line(line)

        # Filter low-confidence edges
        if confidence < MIN_EDGE_CONFIDENCE:
            continue

        edges.append((gene_a, gene_b, confidence))

    # Build graph
    G = Graph()
    for gene_a, gene_b, conf in edges:
        G.add_edge(gene_a, gene_b, weight=conf)

    return G


function filter_network(G):
    """
    Remove isolated nodes and apply quality filters.
    """
    # Remove nodes with degree below threshold
    nodes_to_remove = [
        node for node in G.nodes()
        if G.degree(node) < MIN_NODE_DEGREE
    ]
    G.remove_nodes(nodes_to_remove)

    return G
```

### Step 2: Build Normalized Adjacency Matrix

```python
function build_transition_matrix(G, apply_hub_correction=True):
    """
    Build row-normalized transition matrix for random walk.

    W[i,j] = probability of walking from gene i to gene j
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize adjacency matrix
    A = zeros((n, n))

    for edge in G.edges():
        i = node_to_idx[edge[0]]
        j = node_to_idx[edge[1]]
        weight = G.get_edge_weight(edge)

        A[i, j] = weight
        A[j, i] = weight  # Symmetric for undirected

    # Apply hub correction (reduce influence of high-degree nodes)
    if apply_hub_correction:
        degrees = A.sum(axis=1)
        for i in range(n):
            if degrees[i] > 0:
                # Penalize contribution from hubs
                penalty = degrees[i] ** (-HUB_PENALTY_POWER)
                A[i, :] *= penalty

    # Row-normalize to create transition probabilities
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = A / row_sums

    return W, nodes, node_to_idx
```

### Step 3: Random Walk with Restart

```python
function random_walk_with_restart(initial_scores, W, alpha, max_iter, tol):
    """
    Propagate scores through network using Random Walk with Restart.

    p(t+1) = (1 - α) * W * p(t) + α * p(0)

    Args:
        initial_scores: Vector of initial gene scores (p0)
        W: Transition matrix
        alpha: Restart probability (higher = stay closer to initial)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Propagated score vector
    """
    p0 = initial_scores.copy()
    p = p0.copy()

    for iteration in range(max_iter):
        # One step of random walk
        p_new = (1 - alpha) * (W @ p) + alpha * p0

        # Check convergence
        diff = norm(p_new - p)
        if diff < tol:
            break

        p = p_new

    return p
```

### Step 4: Propagate Gene Scores for One Sample

```python
function propagate_sample_scores(gene_scores_for_sample, W, nodes, node_to_idx):
    """
    Apply network propagation to one sample's gene scores.
    """
    n = len(nodes)

    # Build initial score vector
    p0 = zeros(n)
    for gene, score in gene_scores_for_sample.items():
        if gene in node_to_idx:
            idx = node_to_idx[gene]
            p0[idx] = score

    # Normalize initial scores (optional but recommended)
    if p0.sum() > 0:
        p0 = p0 / p0.sum()

    # Run propagation
    p_final = random_walk_with_restart(
        p0, W,
        alpha=RESTART_PROBABILITY,
        max_iter=MAX_ITERATIONS,
        tol=CONVERGENCE_THRESHOLD
    )

    # Convert back to gene-score map
    propagated_scores = {}
    for i, gene in enumerate(nodes):
        if p_final[i] > 0:
            propagated_scores[gene] = p_final[i]

    return propagated_scores
```

### Step 5: Propagate All Samples

```python
function propagate_all_samples(gene_scores, network):
    """
    Main entry point: propagate gene scores for all samples.
    """
    # Prepare network
    G = filter_network(network)
    W, nodes, node_to_idx = build_transition_matrix(G, APPLY_HUB_CORRECTION)

    propagated_scores = {}

    for sample_id, sample_gene_scores in gene_scores.items():
        propagated = propagate_sample_scores(
            sample_gene_scores, W, nodes, node_to_idx
        )
        propagated_scores[sample_id] = propagated

    return propagated_scores
```

### Step 6: Refine Pathway Scores

```python
function refine_pathway_scores(propagated_gene_scores, pathway_definitions):
    """
    Recompute pathway scores using propagated gene scores.

    This captures network-based signal spreading within and across pathways.
    """
    refined_pathway_scores = {}

    for sample_id, gene_scores in propagated_gene_scores.items():
        sample_pathway_scores = {}

        for pathway_id, pathway_genes in pathway_definitions.items():
            # Sum propagated scores for genes in pathway
            pathway_score = sum(
                gene_scores.get(gene, 0)
                for gene in pathway_genes
            )

            # Normalize by pathway size
            if len(pathway_genes) > 0:
                pathway_score /= math.sqrt(len(pathway_genes))

            sample_pathway_scores[pathway_id] = pathway_score

        refined_pathway_scores[sample_id] = sample_pathway_scores

    return refined_pathway_scores
```

---

## Alternative Propagation Methods

### Heat Diffusion

```python
function heat_diffusion(initial_scores, L, t):
    """
    Heat diffusion on graph Laplacian.

    p(t) = exp(-t * L) * p(0)

    Args:
        L: Graph Laplacian matrix
        t: Diffusion time parameter
    """
    # Compute matrix exponential (or approximate)
    diffusion_kernel = matrix_exp(-t * L)

    return diffusion_kernel @ initial_scores
```

### Personalized PageRank

```python
function personalized_pagerank(initial_scores, W, damping=0.85):
    """
    Personalized PageRank variant.
    Similar to RWR but with different formulation.
    """
    n = len(initial_scores)
    teleport = initial_scores / initial_scores.sum()

    p = ones(n) / n

    for _ in range(MAX_ITERATIONS):
        p_new = damping * (W.T @ p) + (1 - damping) * teleport

        if norm(p_new - p) < CONVERGENCE_THRESHOLD:
            break
        p = p_new

    return p
```

---

## Network Sources

```python
function build_composite_network(sources):
    """
    Combine multiple network sources into a single weighted network.
    """
    combined = Graph()

    for source_name, source_network, source_weight in sources:
        for edge in source_network.edges():
            gene_a, gene_b = edge
            edge_weight = source_network.get_edge_weight(edge) * source_weight

            # Add or update edge
            if combined.has_edge(gene_a, gene_b):
                current = combined.get_edge_weight((gene_a, gene_b))
                combined.set_edge_weight((gene_a, gene_b), current + edge_weight)
            else:
                combined.add_edge(gene_a, gene_b, weight=edge_weight)

    # Normalize combined weights to [0, 1]
    max_weight = max(combined.get_edge_weight(e) for e in combined.edges())
    for edge in combined.edges():
        w = combined.get_edge_weight(edge) / max_weight
        combined.set_edge_weight(edge, w)

    return combined


# Example network sources
NETWORK_SOURCES = [
    ("PPI", load_ppi_network(), 1.0),           # Protein-protein interactions
    ("coexpression", load_coexp_network(), 0.5), # Co-expression
    ("regulatory", load_regulatory_network(), 0.3), # TF-target
]
```

---

## Validation and Diagnostics

```python
function validate_propagation(original_scores, propagated_scores, network):
    """
    Sanity checks on propagation results.
    """
    diagnostics = {}

    # Check correlation with original
    original_vec = [original_scores.get(g, 0) for g in network.nodes()]
    propagated_vec = [propagated_scores.get(g, 0) for g in network.nodes()]

    correlation = pearson_correlation(original_vec, propagated_vec)
    diagnostics["original_correlation"] = correlation

    # Check that high-scoring genes remain high
    top_original = sorted(original_scores.items(), key=lambda x: -x[1])[:20]
    top_propagated = sorted(propagated_scores.items(), key=lambda x: -x[1])[:20]

    overlap = len(set(g for g, _ in top_original) & set(g for g, _ in top_propagated))
    diagnostics["top20_overlap"] = overlap

    # Check for hub dominance
    hub_genes = [g for g in network.nodes() if network.degree(g) > 100]
    hub_score_fraction = sum(propagated_scores.get(g, 0) for g in hub_genes) / sum(propagated_scores.values())
    diagnostics["hub_score_fraction"] = hub_score_fraction

    return diagnostics
```

---

## Output Schema

```python
# Propagated gene scores
PropagatedGeneScores = {
    "sample_id": string,
    "genes": {
        "GENE1": 0.05,
        "GENE2": 0.12,
        ...
    },
    "propagation_params": {
        "alpha": 0.5,
        "iterations": 45,
        "converged": True
    }
}

# Refined pathway scores
RefinedPathwayScores = {
    "sample_id": string,
    "pathways": {
        "GO:0007268": {
            "raw_score": 2.5,
            "propagated_score": 3.1,
            "network_amplification": 1.24
        },
        ...
    }
}
```

---

## Usage Example

```python
# Load inputs
gene_scores = load_gene_burdens("gene_burdens.json")
network = load_interaction_network("ppi_network.txt")
pathway_defs = load_pathway_definitions("pathways.gmt")

# Propagate gene scores
propagated_scores = propagate_all_samples(gene_scores, network)

# Validate
for sample_id in list(propagated_scores.keys())[:5]:
    diag = validate_propagation(
        gene_scores[sample_id],
        propagated_scores[sample_id],
        network
    )
    print(f"Sample {sample_id}: correlation={diag['original_correlation']:.3f}")

# Refine pathway scores
refined_pathways = refine_pathway_scores(propagated_scores, pathway_defs)

# Save outputs
save_propagated_scores(propagated_scores, "propagated_gene_scores.json")
save_refined_pathways(refined_pathways, "refined_pathway_scores.json")
```

---

## Tuning Guidelines

| Parameter | Low Value | High Value | Trade-off |
|-----------|-----------|------------|-----------|
| α (restart) | 0.3 | 0.8 | More network smoothing vs. more fidelity to original |
| Edge threshold | 0.2 | 0.7 | More connections vs. higher confidence |
| Hub penalty | 0.0 | 1.0 | More hub influence vs. less hub bias |

**Recommendation**: Tune parameters using cross-cohort stability as the objective.

---

## Next Steps

- [Subtype Clustering](subtype_clustering.md) - Cluster samples using refined pathway profiles
