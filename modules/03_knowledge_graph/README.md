# Module 03: Knowledge Graph

Provides tools for building and exporting heterogeneous biological knowledge graphs from pathway databases, GO terms, and protein-protein interaction networks.

## Overview

This module constructs a knowledge graph that integrates multiple biological data sources:

- **Gene Ontology (GO)**: Hierarchical functional annotations
- **Pathway databases**: Reactome, KEGG pathway definitions
- **Protein-protein interactions**: STRING network
- **Gene annotations**: SFARI genes, constraint scores

The graph uses NetworkX internally and can be exported to:
- DGL (Deep Graph Library) for GNN training
- PyTorch Geometric (PyG) for GNN training
- Neo4j Cypher for graph database import
- CSV/TSV for general use
- Adjacency matrices for network analysis

## Installation

The module requires core dependencies from the project:

```bash
pip install networkx numpy scipy
```

For GNN export, install optional dependencies:

```bash
# For DGL export
pip install dgl torch

# For PyG export
pip install torch torch-geometric
```

## Quick Start

### Building a Knowledge Graph

```python
from modules.03_knowledge_graph import (
    KnowledgeGraphBuilder,
    PPINetwork,
    NodeType,
    EdgeType,
)
from modules.01_data_loaders import PathwayLoader

# Load pathway data
pathway_loader = PathwayLoader()
go_db = pathway_loader.load_go("data/raw/go-basic.obo", "data/raw/goa_human.gaf")
reactome_db = pathway_loader.load_reactome("data/raw/ReactomePathways.gmt")

# Create PPI network
ppi = PPINetwork(
    interactions=[
        ("SHANK3", "NRXN1", 950),
        ("SHANK3", "NLGN1", 800),
        ("CHD8", "ADNP", 700),
    ],
    source="STRING",
)

# Build knowledge graph
builder = KnowledgeGraphBuilder()
kg = (
    builder
    .add_pathways(go_db)
    .add_pathways(reactome_db)
    .add_ppi(ppi)
    .build()
)

# Get statistics
stats = kg.get_stats()
print(f"Nodes: {stats.n_nodes}, Edges: {stats.n_edges}")
```

### Querying the Graph

```python
# Get neighbors of a gene
neighbors = kg.get_neighbors("SHANK3")

# Filter by edge type
ppi_partners = kg.get_neighbors("SHANK3", edge_type=EdgeType.GENE_INTERACTS)
pathways = kg.get_neighbors("SHANK3", edge_type=EdgeType.GENE_IN_PATHWAY)

# Get all genes
genes = kg.get_nodes_by_type(NodeType.GENE)

# Check node type
node_type = kg.get_node_type("SHANK3")  # NodeType.GENE
```

### Exporting to GNN Formats

```python
from modules.03_knowledge_graph import to_dgl, to_pyg

# Export to DGL
dgl_graph, node_mapping = to_dgl(kg)

# Export to PyG
pyg_data, node_mapping = to_pyg(kg)

# Save node mapping for later use
node_mapping.save("node_mapping.json")
```

### Exporting to Other Formats

```python
from modules.03_knowledge_graph import (
    to_csv,
    to_adjacency_matrix,
    to_neo4j_cypher,
)

# CSV export (compatible with Neo4j import)
files = to_csv(kg, "output/")

# Adjacency matrix
adj_matrix, mapping = to_adjacency_matrix(kg, sparse=True)

# Neo4j Cypher statements
to_neo4j_cypher(kg, "output/graph.cypher")
```

## Node Types

| Type | Description | Example |
|------|-------------|---------|
| `GENE` | Gene/protein | SHANK3, CHD8 |
| `PATHWAY` | Biological pathway | R-HSA-1266738 |
| `GO_TERM` | Gene Ontology term | GO:0007268 |
| `CELL_TYPE` | Cell type | Excitatory neuron |
| `DRUG` | Drug/compound | Risperidone |
| `PROTEIN` | Protein | ENSP00000269305 |
| `VARIANT` | Genetic variant | rs123456 |
| `PHENOTYPE` | Phenotype | Autism |

## Edge Types

| Type | Source → Target | Description |
|------|-----------------|-------------|
| `GENE_INTERACTS` | Gene → Gene | PPI interaction |
| `GENE_IN_PATHWAY` | Gene → Pathway | Pathway membership |
| `GENE_HAS_GO` | Gene → GO_TERM | GO annotation |
| `GO_IS_A` | GO_TERM → GO_TERM | Ontology hierarchy |
| `GO_PART_OF` | GO_TERM → GO_TERM | Part-of relation |
| `DRUG_TARGETS` | Drug → Gene | Drug targeting |
| `GENE_EXPRESSED_IN` | Gene → Cell_Type | Expression |

## API Reference

### KnowledgeGraphBuilder

```python
class KnowledgeGraphBuilder:
    def add_genes(self, gene_list: List[str]) -> "KnowledgeGraphBuilder"
    def add_pathways(self, pathway_db: PathwayDatabase) -> "KnowledgeGraphBuilder"
    def add_go_terms(self, go_terms: Dict, annotations: Dict) -> "KnowledgeGraphBuilder"
    def add_ppi(self, ppi_network: PPINetwork) -> "KnowledgeGraphBuilder"
    def build(self) -> KnowledgeGraph
```

### KnowledgeGraph

```python
class KnowledgeGraph:
    def add_node(self, node_id: str, node_type: NodeType, attributes: Dict = None) -> None
    def add_edge(self, source: str, target: str, edge_type: EdgeType, weight: float = 1.0) -> None
    def get_neighbors(self, node_id: str, edge_type: EdgeType = None) -> List[str]
    def get_nodes_by_type(self, node_type: NodeType) -> List[str]
    def get_stats(self) -> KnowledgeGraphStats
    def subgraph(self, node_ids: List[str]) -> "KnowledgeGraph"
    def save(self, path: str) -> None
    def load(cls, path: str) -> "KnowledgeGraph"
```

## Integration with BigQuery

For large-scale PPI data stored in BigQuery (see GCP setup), use:

```python
from google.cloud import bigquery

# Query STRING PPI from BigQuery
client = bigquery.Client()
query = """
    SELECT protein1, protein2, combined_score
    FROM `autism-pathway-framework.autism_genetics.string_ppi`
    WHERE combined_score >= 700
"""

df = client.query(query).to_dataframe()

# Add to builder
builder.add_ppi_from_dataframe(
    df,
    protein1_col="protein1",
    protein2_col="protein2",
    score_col="combined_score",
)
```

## File Structure

```
modules/03_knowledge_graph/
├── README.md
├── __init__.py
├── schema.py           # Node/edge type definitions
├── builder.py          # KnowledgeGraph and KnowledgeGraphBuilder
├── exporters.py        # Export functions (DGL, PyG, Neo4j, CSV)
└── tests/
    ├── __init__.py
    ├── test_builder.py
    └── test_exporters.py
```

## Testing

```bash
# Run all module tests
pytest modules/03_knowledge_graph/tests/ -v

# Run specific test file
pytest modules/03_knowledge_graph/tests/test_builder.py -v
```

## Dependencies

- **Required**: networkx, numpy, scipy
- **Optional (GNN export)**: dgl, torch, torch-geometric

## See Also

- [Module 01: Data Loaders](../01_data_loaders/README.md) - PathwayDatabase interface
- [Module 04: Graph Embeddings](../04_graph_embeddings/README.md) - Embedding training
- [Implementation Plan](../../docs/implementation_plan.md) - Full project roadmap
