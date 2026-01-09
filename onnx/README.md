# ONNX ColBERT Encoder

A Rust-based ONNX Runtime implementation for computing ColBERT embeddings. This provides a pure-Rust alternative to PyTorch/Python for inference.

## Prerequisites

- [Rust](https://rustup.rs/) (1.70+)
- [uv](https://github.com/astral-sh/uv) (for Python export step)

## Quick Start

### 1. Export the Model to ONNX

First, export the ColBERT model from HuggingFace to ONNX format using the Python export script.

```bash
cd python

# Create a virtual environment with Python 3.12 (required for pylate compatibility)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Export the model
python export_onnx.py
```

This will create:
- `models/answerai-colbert-small-v1.onnx` - The ONNX model
- `models/tokenizer.json` - The tokenizer with [Q] and [D] special tokens

### 2. Build the Rust Encoder

```bash
cd ..  # Back to onnx/ directory
cargo build --release --bin encode_cli
```

### 3. Encode Texts

Create a JSON file with texts to encode:

```json
{
  "texts": [
    "What is machine learning?",
    "Neural networks are computational models."
  ]
}
```

Encode as documents:

```bash
./target/release/encode_cli encode \
  --input texts.json \
  --output-dir embeddings/ \
  --model models/answerai-colbert-small-v1.onnx \
  --tokenizer models/tokenizer.json
```

Encode as queries (uses [MASK] padding and fixed length 32):

```bash
./target/release/encode_cli encode \
  --input queries.json \
  --output-dir query_embeddings/ \
  --model models/answerai-colbert-small-v1.onnx \
  --tokenizer models/tokenizer.json \
  --is-query
```

## Running the SciFact Benchmark

The benchmark evaluates the full pipeline: ONNX encoding + Lategrep REST API indexing and search.

### Prerequisites

1. Export the ONNX model (see Quick Start above)
2. Install Python dependencies:

```bash
cd ..  # Go to lategrep root
cd docs && uv sync --extra eval
```

### Run the Benchmark (Recommended)

The easiest way to run the benchmark is using the Makefile target, which handles starting/stopping the API server:

```bash
# From the lategrep root directory
make benchmark-onnx-api
```

This will:
1. Start the Lategrep REST API server
2. Download the SciFact dataset (if not cached)
3. Build the ONNX encoder binary
4. Encode documents and queries using ONNX Runtime (or use cached embeddings)
5. Create a Lategrep index via the REST API
6. Run search queries
7. Evaluate with NDCG@10, MAP, and Recall metrics
8. Stop the API server

### Manual Run

If you prefer to run the benchmark manually:

```bash
# Terminal 1: Start the API server
cargo build --release -p lategrep-api --features accelerate
./target/release/lategrep-api -h 127.0.0.1 -p 8080 -d ./api/indices

# Terminal 2: Run the benchmark
cd docs
uv run python onnx_benchmark.py --skip-encoding
```

### Benchmark Options

```bash
# Skip encoding if embeddings already exist (default in make target)
python onnx_benchmark.py --skip-encoding

# Use a custom embeddings directory
python onnx_benchmark.py --embeddings-dir ./my_embeddings

# Custom API endpoint
python onnx_benchmark.py --host 127.0.0.1 --port 8080

# Custom batch size for API calls
python onnx_benchmark.py --batch-size 100
```

### Expected Output

```
======================================================================
  SciFact Benchmark: ONNX (Rust) + Lategrep REST API
======================================================================

Configuration:
  Top-k:             100
  n_ivf_probe:       8
  n_full_scores:     8192
  nbits:             4
  API endpoint:      http://127.0.0.1:8080

[1/4] Loading scifact dataset...
  Documents: 5183
  Queries:   300

[2/4] Building ONNX encoder...

[3/4] Encoding with ONNX (Rust)...

[4/4] Running Lategrep via REST API...

======================================================================
  RESULTS
======================================================================

  Retrieval Metrics:
  ----------------------------------------
  Metric               Score
  ----------------------------------------
  map                         0.7xxx
  ndcg@10                     0.7xxx
  recall@100                  0.9xxx
  ----------------------------------------
```

## CLI Reference

### encode_cli

```
ONNX ColBERT Encoder CLI

Usage:
    encode_cli encode [options]

Options:
    --input <path>       JSON file with texts ({"texts": ["...", "..."]})
    --output-dir <path>  Directory to write .npy embeddings
    --model <path>       Path to ONNX model (default: models/answerai-colbert-small-v1.onnx)
    --tokenizer <path>   Path to tokenizer.json (default: models/tokenizer.json)
    --is-query           Encode as queries (default: encode as documents)
    --threads <n>        Number of threads for ONNX Runtime (default: 4)

Output:
    For documents: doc_000000.npy, doc_000001.npy, ...
    For queries:   query_000000.npy, query_000001.npy, ...
```

## File Structure

```
onnx/
├── models/
│   ├── answerai-colbert-small-v1.onnx  # Exported ONNX model
│   └── tokenizer.json                   # Tokenizer with [Q]/[D] tokens
├── python/
│   ├── export_onnx.py                   # Model export script
│   └── requirements.txt                 # Python dependencies
├── src/
│   ├── main.rs                          # Comparison tool (Rust vs Python)
│   └── bin/
│       └── encode_cli.rs                # Encoding CLI
├── Cargo.toml
└── README.md
```

## Model Details

- **Base model**: `lightonai/answerai-colbert-small-v1`
- **Embedding dimension**: 96
- **Query max length**: 32 tokens (padded with [MASK])
- **Document max length**: 300 tokens (variable, no padding in output)
- **Special tokens**: [Q] for queries, [D] for documents
