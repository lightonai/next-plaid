# Multi-Vector Search

Understanding how multi-vector search differs from traditional vector search.

## Traditional Vector Search

In traditional dense retrieval, each document and query is represented as a **single vector**:

```
Document: "Paris is the capital of France" → [0.1, 0.2, ..., 0.5]  (1 x dim)
Query:    "What is France's capital?"      → [0.3, 0.4, ..., 0.6]  (1 x dim)
```

Similarity is computed using cosine similarity or dot product between these single vectors.

### Limitations

- **Information loss**: Compressing an entire document into one vector loses nuance
- **Length bias**: Long documents may be poorly represented
- **Semantic dilution**: Key terms get averaged with less relevant content

---

## Multi-Vector Search

Multi-vector search represents documents and queries as **multiple vectors** - typically one per token:

```
Document: "Paris is the capital of France"
          → [[v_Paris], [v_is], [v_the], [v_capital], [v_of], [v_France]]
          (6 x dim)

Query:    "What is France's capital?"
          → [[v_What], [v_is], [v_France's], [v_capital]]
          (4 x dim)
```

### MaxSim Scoring

Relevance is computed using **MaxSim** (Maximum Similarity):

1. For each query token, find its maximum similarity to any document token
2. Sum these maximum similarities

```
score = Σ max(sim(q_i, d_j) for all j in document)
        for all i in query
```

This allows fine-grained matching where:

- "France" in the query matches "France" in the document strongly
- "capital" in the query matches "capital" in the document strongly
- Less relevant query terms contribute less to the final score

---

## ColBERT Architecture

NextPlaid uses embeddings from ColBERT (Contextualized Late Interaction over BERT) models.

### How ColBERT Works

1. **Encoding**: BERT produces contextualized embeddings for each token
2. **Projection**: Embeddings are projected to a lower dimension (e.g., 128)
3. **Normalization**: Vectors are L2-normalized

```python
# Simplified ColBERT encoding
def encode(text):
    tokens = tokenize(text)
    bert_embeddings = bert(tokens)        # [num_tokens, 768]
    projected = linear(bert_embeddings)    # [num_tokens, 128]
    normalized = l2_normalize(projected)   # [num_tokens, 128]
    return normalized
```

### Late Interaction

"Late interaction" means the query and document are encoded independently, then interact only at scoring time:

```
Query encoding  ───┐
                   ├──→ MaxSim scoring
Document encoding ─┘
```

This enables:

- **Pre-computed document embeddings**: Encode once, search many times
- **Efficient indexing**: Documents can be indexed offline
- **Scalable search**: Only queries need real-time encoding

---

## Comparison

| Aspect | Single-Vector | Multi-Vector |
|--------|---------------|--------------|
| **Representation** | 1 vector per document | N vectors per document |
| **Storage** | ~128 floats/doc | ~128 × N floats/doc |
| **Scoring** | Dot product | MaxSim |
| **Accuracy** | Good | Excellent |
| **Speed** | Very fast | Fast (with PLAID) |

### When to Use Multi-Vector

**Choose multi-vector (ColBERT/NextPlaid) when:**

- Accuracy is critical
- Documents vary in length
- Fine-grained matching matters
- You have the storage budget

**Choose single-vector when:**

- Latency is extremely critical
- Storage is very constrained
- Good enough accuracy is acceptable

---

## Practical Example

Consider searching for "machine learning applications in healthcare":

### Single-Vector

The entire query becomes one vector. It might match documents about:

- Machine learning (partial match)
- Healthcare (partial match)
- Generic AI documents (semantic similarity)

### Multi-Vector

Each concept gets its own vector:

| Query Token | Best Document Match | Score |
|-------------|---------------------|-------|
| machine | "machine" | 0.95 |
| learning | "learning" | 0.92 |
| applications | "applications" | 0.88 |
| healthcare | "medical" | 0.85 |

Documents must match multiple concepts well to score highly.

---

## Storage Considerations

Multi-vector search requires more storage:

```
Single-vector:  1,000,000 docs × 128 dims × 4 bytes = 512 MB
Multi-vector:   1,000,000 docs × 15 tokens × 128 dims × 4 bytes = 7.7 GB
```

### How NextPlaid Reduces Storage

NextPlaid uses the PLAID algorithm to compress embeddings:

1. **Quantization**: 2-bit or 4-bit storage instead of 32-bit floats
2. **Residual encoding**: Store difference from cluster centroids
3. **IVF indexing**: Only load relevant partitions

Result: **~90% storage reduction** compared to naive multi-vector storage.

---

## Further Reading

- [PLAID Algorithm](plaid.md) - How NextPlaid achieves efficient multi-vector search
- [ColBERT Paper](https://arxiv.org/abs/2004.12832) - Original ColBERT research
- [ColBERTv2 Paper](https://arxiv.org/abs/2112.01488) - Improved efficiency
