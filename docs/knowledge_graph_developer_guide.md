# Knowledge Graph Developer Guide

> A comprehensive guide for developers building knowledge graphs in any domain, using the Autism Pathway Framework as a reference implementation.

---

## Table of Contents

1. [What is a Knowledge Graph?](#what-is-a-knowledge-graph)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Design Patterns](#design-patterns)
5. [API Design Decisions](#api-design-decisions)
6. [Storage Strategy](#storage-strategy)
7. [Adapting to Your Domain](#adapting-to-your-domain)
8. [Code Examples](#code-examples)
9. [Performance Considerations](#performance-considerations)

---

## What is a Knowledge Graph?

A **knowledge graph** is a structured representation of real-world entities and the relationships between them. Think of it as a sophisticated database that understands not just data, but the *meaning* and *connections* within that data.

### Real-World Analogies

| Domain | Entities (Nodes) | Relationships (Edges) |
|--------|------------------|----------------------|
| **Social Network** | Users, Posts, Groups | "follows", "likes", "member_of" |
| **E-commerce** | Products, Categories, Customers | "belongs_to", "purchased", "reviewed" |
| **Healthcare** | Patients, Diseases, Medications | "diagnosed_with", "treats", "interacts_with" |
| **Our Biology Domain** | Genes, Pathways, Proteins | "interacts_with", "belongs_to_pathway", "regulates" |

### Why Use a Knowledge Graph?

1. **Discover Hidden Connections**: Find relationships that aren't obvious in flat tables
2. **Flexible Schema**: Easily add new entity types without restructuring everything
3. **Natural Queries**: Ask questions like "What genes interact with SHANK3 and are also in the synaptic pathway?"
4. **Machine Learning Ready**: Feed directly into Graph Neural Networks (GNNs)

---

## Core Concepts

### Nodes (Vertices)

Nodes represent **entities** in your domain. Each node has:

- **ID**: Unique identifier (e.g., "SHANK3", "user_12345")
- **Type**: Category of entity (e.g., GENE, USER, PRODUCT)
- **Attributes**: Properties stored on the node (e.g., name, score, metadata)

```python
# In our implementation
from modules.03_knowledge_graph import NodeType, Node

# A node representing a gene
node = Node(
    id="SHANK3",
    node_type=NodeType.GENE,
    attributes={"pli_score": 0.99, "chromosome": "22"}
)
```

### Edges (Relationships)

Edges represent **relationships** between entities. Each edge has:

- **Source**: Starting node
- **Target**: Ending node
- **Type**: Kind of relationship
- **Weight**: Strength of relationship (optional)
- **Attributes**: Additional properties

```python
from modules.03_knowledge_graph import EdgeType, Edge

# SHANK3 interacts with NRXN1 with 95% confidence
edge = Edge(
    source="SHANK3",
    target="NRXN1",
    edge_type=EdgeType.GENE_INTERACTS,
    weight=0.95
)
```

### Heterogeneous vs Homogeneous Graphs

| Type | Description | Example |
|------|-------------|---------|
| **Homogeneous** | All nodes same type, all edges same type | Social network with only "follows" |
| **Heterogeneous** | Multiple node types, multiple edge types | Our biology graph with genes, pathways, GO terms |

Our implementation uses **heterogeneous graphs** because real-world domains have multiple entity types with different relationship semantics.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Knowledge Graph System                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐     │
│  │   Schema    │───▶│   Builder    │───▶│   KnowledgeGraph    │     │
│  │  (Types)    │    │  (Factory)   │    │   (Core Object)     │     │
│  └─────────────┘    └──────────────┘    └─────────────────────┘     │
│         │                  │                      │                  │
│         ▼                  ▼                      ▼                  │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐     │
│  │  NodeType   │    │ Data Sources │    │     Exporters       │     │
│  │  EdgeType   │    │  - Files     │    │  - DGL/PyG (GNN)    │     │
│  │ GraphSchema │    │  - BigQuery  │    │  - Neo4j (DB)       │     │
│  └─────────────┘    │  - APIs      │    │  - CSV (General)    │     │
│                     └──────────────┘    └─────────────────────┘     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         Storage Layer                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Local Storage              │         Cloud Storage                 │
│   ─────────────              │         ─────────────                 │
│   • NetworkX (in-memory)     │         • BigQuery (large tables)     │
│   • Pickle/JSON (serialize)  │         • GCS (file storage)          │
│   • CSV (interchange)        │         • Neo4j Aura (graph DB)       │
│                              │                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Design Patterns

### 1. Schema-First Design

**Why**: Define your types upfront to catch errors early and document your domain model.

```python
# schema.py - Define BEFORE building

class NodeType(Enum):
    """All possible entity types in your domain."""
    GENE = "gene"
    PATHWAY = "pathway"
    GO_TERM = "go_term"

class EdgeType(Enum):
    """All possible relationship types."""
    GENE_INTERACTS = "gene_interacts_gene"
    GENE_IN_PATHWAY = "gene_in_pathway"

class GraphSchema:
    """Defines which edges are valid between which node types."""
    valid_edges = {
        EdgeType.GENE_INTERACTS: (NodeType.GENE, NodeType.GENE),
        EdgeType.GENE_IN_PATHWAY: (NodeType.GENE, NodeType.PATHWAY),
    }
```

**Benefits**:
- Self-documenting code
- Validation at build time
- IDE autocomplete support
- Prevents invalid relationships

### 2. Builder Pattern

**Why**: Complex objects (graphs with millions of edges) need step-by-step construction.

```python
# Instead of this (hard to maintain):
graph = KnowledgeGraph()
graph.add_node(...)
graph.add_node(...)
# ... 1000 more lines

# Use this (fluent, readable):
graph = (
    KnowledgeGraphBuilder()
    .add_genes(gene_list)           # Step 1
    .add_pathways(pathway_db)       # Step 2
    .add_ppi(interaction_network)   # Step 3
    .build()                        # Final assembly
)
```

**Benefits**:
- Clear construction sequence
- Method chaining for readability
- Easy to add new data sources
- Separation of concerns

### 3. Adapter Pattern for Data Sources

**Why**: Different data sources have different formats. Adapters normalize them.

```python
# Our implementation handles multiple input formats:

# From a PathwayDatabase object (Module 01)
builder.add_pathways(pathway_db)

# From a PPINetwork object
builder.add_ppi(ppi_network)

# From a pandas DataFrame (BigQuery result)
builder.add_ppi_from_dataframe(df, protein1_col="p1", protein2_col="p2")

# The builder internally converts all formats to the same structure
```

### 4. Strategy Pattern for Export

**Why**: Same graph, multiple output formats for different use cases.

```python
from modules.03_knowledge_graph import to_dgl, to_pyg, to_csv, to_neo4j_cypher

# Same graph, different exports:
dgl_graph, mapping = to_dgl(kg)           # For DGL-based GNN training
pyg_data, mapping = to_pyg(kg)            # For PyTorch Geometric
to_csv(kg, "output/")                     # For general analysis
to_neo4j_cypher(kg, "graph.cypher")       # For graph database import
```

---

## API Design Decisions

### Why NetworkX as the Core?

| Option | Pros | Cons | Our Choice |
|--------|------|------|------------|
| **NetworkX** | Pure Python, no dependencies, mature, well-documented | Slower for very large graphs | Selected |
| **igraph** | Faster C backend | Less Pythonic API, extra compilation | Not selected |
| **graph-tool** | Very fast | Hard to install, GPL license | Not selected |
| **Neo4j Python driver** | Persistent, scalable | Requires Neo4j server running | Export supported |

**Reasoning**:
1. **Zero friction**: Works immediately with `pip install networkx`
2. **Prototyping speed**: Easy to iterate during development
3. **Sufficient scale**: Handles millions of edges for our use case
4. **Export flexibility**: Can export to optimized formats when needed

### Why Enums for Types (Not Strings)?

```python
# Bad: Strings are error-prone
graph.add_edge("gene1", "gene2", edge_type="gene_interacts")  # typo won't be caught
graph.add_edge("gene1", "gene2", edge_type="gene_interracts") # silent bug!

# Good: Enums catch errors at development time
graph.add_edge("gene1", "gene2", edge_type=EdgeType.GENE_INTERACTS)  # IDE helps
graph.add_edge("gene1", "gene2", edge_type=EdgeType.GENE_INTERRACTS) # NameError!
```

### Why Separate Builder from Graph?

```python
# The KnowledgeGraph is the final, queryable product
# The KnowledgeGraphBuilder is the factory that creates it

class KnowledgeGraph:
    """Immutable-ish, optimized for queries."""
    def get_neighbors(self, node_id): ...
    def get_nodes_by_type(self, node_type): ...
    def subgraph(self, nodes): ...

class KnowledgeGraphBuilder:
    """Mutable, optimized for construction."""
    def add_genes(self, genes): ...
    def add_pathways(self, pathways): ...
    def build(self) -> KnowledgeGraph: ...
```

**Separation benefits**:
- Clear lifecycle: build phase vs query phase
- Builder can validate incrementally
- Graph object is simpler (no construction logic)
- Easier testing of each component

---

## Storage Strategy

### Local vs Cloud: Decision Framework

```
                        ┌─────────────────────────┐
                        │   How big is your data? │
                        └───────────┬─────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            < 1 million edges               > 1 million edges
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │   Local Storage   │           │   Cloud Storage   │
        │   - NetworkX      │           │   - BigQuery      │
        │   - Pickle/JSON   │           │   - GCS + local   │
        └───────────────────┘           └───────────────────┘
```

### Our Hybrid Approach

| Data Type | Storage Location | Reasoning |
|-----------|------------------|-----------|
| **Small reference data** (GO terms, pathways) | Local files with GCS backup | Fast access, rarely changes |
| **Large interaction data** (13M PPI edges) | BigQuery | Too large for memory, needs filtering |
| **Built knowledge graph** | In-memory NetworkX | Fast queries during analysis |
| **Serialized graph** | GCS (.gpickle) | Sharing, versioning |

### BigQuery for Large Datasets

**Why BigQuery instead of local files?**

```python
# Problem: STRING PPI has 13.7 million interactions
# Loading all into memory = slow and wasteful

# Solution: Query only what you need
query = """
    SELECT protein1, protein2, combined_score
    FROM `autism-pathway-framework.autism_genetics.string_ppi`
    WHERE combined_score >= 700  -- High confidence only
    AND (protein1 LIKE '%SHANK%' OR protein2 LIKE '%SHANK%')
"""
# Returns ~1000 rows instead of 13 million
```

**BigQuery benefits**:
1. **Pay per query**: Only process data you need
2. **No infrastructure**: Serverless, always available
3. **SQL interface**: Familiar, powerful filtering
4. **Scales automatically**: Handles petabytes

### GCS for File Storage

**Why Google Cloud Storage?**

```yaml
# configs/gcp_config.yaml
storage:
  bucket: "autism-pathway-data"
  paths:
    raw: "raw/"           # Original downloaded files
    processed: "processed/"  # Cleaned/transformed data
    embeddings: "embeddings/"  # ML model outputs
    models: "models/"     # Trained model weights
```

**Benefits**:
1. **Versioning**: Track changes to data files
2. **Sharing**: Collaborators access same data
3. **Durability**: 99.999999999% (11 nines) durability
4. **Integration**: Works with BigQuery, Vertex AI, etc.

---

## Adapting to Your Domain

### Step 1: Define Your Schema

Replace our biology types with your domain:

```python
# E-commerce example
class NodeType(Enum):
    PRODUCT = "product"
    CATEGORY = "category"
    CUSTOMER = "customer"
    ORDER = "order"

class EdgeType(Enum):
    PRODUCT_IN_CATEGORY = "product_in_category"
    CUSTOMER_PURCHASED = "customer_purchased"
    PRODUCT_SIMILAR = "product_similar_to"
    CUSTOMER_VIEWED = "customer_viewed"
```

### Step 2: Define Valid Relationships

```python
# Which edges make sense between which nodes?
valid_edges = {
    EdgeType.PRODUCT_IN_CATEGORY: (NodeType.PRODUCT, NodeType.CATEGORY),
    EdgeType.CUSTOMER_PURCHASED: (NodeType.CUSTOMER, NodeType.PRODUCT),
    EdgeType.PRODUCT_SIMILAR: (NodeType.PRODUCT, NodeType.PRODUCT),
    EdgeType.CUSTOMER_VIEWED: (NodeType.CUSTOMER, NodeType.PRODUCT),
}
```

### Step 3: Create Domain-Specific Loaders

```python
class EcommerceGraphBuilder(KnowledgeGraphBuilder):
    """Builder specialized for e-commerce data."""

    def add_products(self, product_catalog: pd.DataFrame):
        """Add products from catalog DataFrame."""
        for _, row in product_catalog.iterrows():
            self._graph.add_node(
                row["product_id"],
                NodeType.PRODUCT,
                {"name": row["name"], "price": row["price"]}
            )
        return self

    def add_purchase_history(self, orders: pd.DataFrame):
        """Add customer-product edges from order history."""
        for _, row in orders.iterrows():
            self._edges.append((
                row["customer_id"],
                row["product_id"],
                EdgeType.CUSTOMER_PURCHASED,
                row["quantity"]  # weight = purchase count
            ))
        return self
```

### Step 4: Choose Your Storage Strategy

| Your Data Size | Recommended Approach |
|----------------|---------------------|
| < 100K edges | Local NetworkX + JSON export |
| 100K - 10M edges | Local NetworkX + Pickle, GCS backup |
| 10M - 100M edges | BigQuery for raw data, filtered load to NetworkX |
| > 100M edges | Neo4j or dedicated graph database |

---

## Code Examples

### Example 1: Basic Graph Construction

```python
from modules.03_knowledge_graph import (
    KnowledgeGraph,
    NodeType,
    EdgeType
)

# Create empty graph
kg = KnowledgeGraph()

# Add nodes
kg.add_node("product_1", NodeType.PRODUCT, {"name": "Widget", "price": 29.99})
kg.add_node("product_2", NodeType.PRODUCT, {"name": "Gadget", "price": 49.99})
kg.add_node("electronics", NodeType.CATEGORY, {"name": "Electronics"})

# Add edges
kg.add_edge("product_1", "electronics", EdgeType.PRODUCT_IN_CATEGORY)
kg.add_edge("product_2", "electronics", EdgeType.PRODUCT_IN_CATEGORY)
kg.add_edge("product_1", "product_2", EdgeType.PRODUCT_SIMILAR, weight=0.85)

# Query
neighbors = kg.get_neighbors("product_1")
print(f"Products related to product_1: {neighbors}")
```

### Example 2: Using the Builder Pattern

```python
from modules.03_knowledge_graph import KnowledgeGraphBuilder

# Fluent construction
kg = (
    KnowledgeGraphBuilder()
    .add_genes(["SHANK3", "CHD8", "SCN2A"])
    .add_pathways(reactome_db)
    .add_pathways(go_db)
    .add_ppi(string_network)
    .build()
)

# Get statistics
stats = kg.get_stats()
print(f"Nodes: {stats.n_nodes}")
print(f"Edges: {stats.n_edges}")
print(f"Node types: {stats.node_type_counts}")
```

### Example 3: Export for Machine Learning

```python
from modules.03_knowledge_graph import to_dgl, to_adjacency_matrix
import numpy as np

# Export to DGL for GNN training
dgl_graph, node_mapping = to_dgl(kg)

# Or get adjacency matrix for traditional ML
adj_matrix, mapping = to_adjacency_matrix(kg, sparse=True)

# Create node features
node_features = np.random.randn(len(mapping), 64)  # 64-dim embeddings

# Now ready for scikit-learn, PyTorch, etc.
```

### Example 4: BigQuery Integration

```python
from google.cloud import bigquery
from modules.03_knowledge_graph import KnowledgeGraphBuilder

client = bigquery.Client()

# Query only high-confidence interactions
query = """
    SELECT protein1, protein2, combined_score
    FROM `your-project.your_dataset.interactions`
    WHERE combined_score >= 700
"""

df = client.query(query).to_dataframe()

# Build graph from query results
kg = (
    KnowledgeGraphBuilder()
    .add_ppi_from_dataframe(df,
        protein1_col="protein1",
        protein2_col="protein2",
        score_col="combined_score"
    )
    .build()
)
```

---

## Performance Considerations

### Memory Estimation

```python
# Rule of thumb for NetworkX:
# - Each node: ~500 bytes base + attributes
# - Each edge: ~200 bytes base + attributes

nodes = 100_000
edges = 1_000_000

estimated_memory_mb = (nodes * 500 + edges * 200) / 1_000_000
print(f"Estimated memory: {estimated_memory_mb:.0f} MB")  # ~250 MB
```

### When to Move Beyond NetworkX

| Symptom | Solution |
|---------|----------|
| Graph doesn't fit in memory | Use BigQuery + filtered loading |
| Queries are slow (> 1 second) | Add indexes, use adjacency matrix for specific queries |
| Need persistence across restarts | Export to Neo4j or use pickle |
| Need distributed processing | Use Apache Spark GraphX or Neo4j cluster |

### Optimization Tips

1. **Filter early**: Load only the data you need from BigQuery
2. **Use sparse matrices**: For adjacency representations
3. **Batch operations**: Add many edges at once, not one by one
4. **Cache subgraphs**: If you query the same subset repeatedly

```python
# Good: Batch addition
kg.add_edges(edge_list, EdgeType.GENE_INTERACTS)  # One call

# Bad: One at a time
for src, tgt in edge_list:
    kg.add_edge(src, tgt, EdgeType.GENE_INTERACTS)  # Many calls
```

---

## Summary

### Key Takeaways

1. **Schema first**: Define your domain types before building
2. **Builder pattern**: Construct complex graphs step-by-step
3. **Hybrid storage**: Local for computation, cloud for scale and sharing
4. **Export flexibility**: Same graph, multiple output formats
5. **Start simple**: NetworkX is sufficient for most use cases

### Files Reference

| File | Purpose |
|------|---------|
| `schema.py` | Define your domain's node/edge types |
| `builder.py` | Construct graphs from various data sources |
| `exporters.py` | Export to GNN frameworks, databases, files |
| `gcp_config.yaml` | Cloud storage and BigQuery configuration |

### Next Steps for Your Project

1. Fork this module structure
2. Replace `NodeType` and `EdgeType` with your domain
3. Implement domain-specific loaders in the builder
4. Choose storage based on your data size
5. Export to your ML framework of choice

---

*This guide is part of the Autism Pathway Framework. For biology-specific details, see the [Module 03 README](../modules/03_knowledge_graph/README.md).*
