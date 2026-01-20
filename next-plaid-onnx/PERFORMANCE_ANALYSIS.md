# Performance Analysis: PyLate vs next-plaid-onnx

## Executive Summary

This document presents a comprehensive performance analysis comparing PyLate (PyTorch) against next-plaid-onnx (ONNX Runtime) for ColBERT inference using the `lightonai/GTE-ModernColBERT-v1` model.

### Key Findings

| Implementation | Speed (docs/s) | Relative to MPS |
|---------------|----------------|-----------------|
| PyLate MPS (Apple GPU) | 273.5 | 1.00x (baseline) |
| **ONNX INT8 (CPU)** | **161.3** | **0.59x** |
| PyLate CPU (PyTorch) | 136.8 | 0.50x |
| ONNX FP32 (CPU) | 71.7 | 0.26x |

### Conclusions

1. **ONNX FP32 is slower than PyTorch CPU** - ONNX FP32 is ~1.9x slower than PyTorch on CPU
2. **ONNX INT8 is faster than PyTorch CPU** - ONNX INT8 is ~1.18x faster than PyTorch on CPU
3. **INT8 quantization provides 2.25x speedup** over FP32 with >99.5% accuracy preservation
4. **GPU acceleration (MPS) provides 2x speedup** over best CPU implementation

---

## Detailed Analysis

### 1. Correctness Verification

All implementations produce mathematically equivalent embeddings:

| Comparison | Cosine Similarity |
|------------|-------------------|
| PyLate ↔ ONNX FP32 | 1.000000 (identical) |
| PyLate ↔ ONNX INT8 | 0.9958 - 0.9978 |
| ONNX FP32 ↔ INT8 | 0.9958 - 0.9978 |

**Conclusion**: ONNX FP32 produces identical results. INT8 quantization introduces minimal error (<0.5%).

### 2. Where Time is Spent (ONNX Profiling)

| Operation | % of Total Time |
|-----------|-----------------|
| **MatMul** | **82.8%** |
| Slice | 3.2% |
| Transpose | 2.1% |
| Mul | 1.9% |
| LayerNorm | 1.6% |
| Gelu | 1.6% |
| Other | 6.8% |

**Conclusion**: MatMul operations dominate inference time. Optimizing matrix multiplication is the key to faster inference.

### 3. Threading Configuration

Tested with 16 documents on CPU:

| intra_op_threads | inter_op_threads | Speed (docs/s) |
|------------------|------------------|----------------|
| 0 (auto) | 0 (auto) | 68.5 |
| 1 | 1 | 20.2 |
| 2 | 1 | 39.1 |
| **4** | **1** | **71.6** |
| 4 | 2 | 71.9 |
| 8 | 1 | 54.0 |

**Conclusion**: Auto-detection (0, 0) works reasonably well. Explicit configuration of 4 intra-op threads provides marginal improvement.

### 4. Batch Size Impact

Tested on 64 documents with ONNX FP32:

| Batch Size | Speed (docs/s) |
|------------|----------------|
| 1 | 60.8 |
| 4 | 74.5 |
| 8 | 72.4 |
| **16** | **78.4** |
| 32 | 78.3 |
| 64 | 73.1 |

**Conclusion**: Batch size 16-32 is optimal for CPU. Current default of 32 is appropriate.

### 5. Session Options Impact

| Configuration | Speed (docs/s) |
|---------------|----------------|
| Default | 75.1 |
| Memory pattern disabled | **80.1** |
| CPU arena disabled | 79.4 |
| Sequential execution | 71.8 |
| Parallel execution | 45.1 (worse!) |

**Conclusion**: Disabling memory pattern optimization provides slight improvement. Parallel execution mode is counterproductive.

### 6. IO Binding Impact

| Batch Size | Standard | IO Binding | Speedup |
|------------|----------|------------|---------|
| 4 | 64.6 docs/s | 63.2 docs/s | 0.98x |
| 8 | 72.4 docs/s | 72.9 docs/s | 1.01x |
| 16 | 70.8 docs/s | 74.1 docs/s | 1.05x |
| 32 | 74.0 docs/s | 73.4 docs/s | 0.99x |

**Conclusion**: IO binding provides minimal benefit (~5% at best) for CPU execution.

---

## Current Rust Implementation Analysis

### What's Already Good

1. **Graph Optimization Level**: Uses `Level3` (maximum optimization) ✓
2. **Batch Size**: Default 32 for CPU is optimal ✓
3. **INT8 Support**: Already implemented via `with_quantized(true)` ✓
4. **Parallel Sessions**: Supports multiple ONNX sessions for throughput ✓

### Potential Optimizations

1. **Memory Pattern**: Consider disabling memory pattern optimization
   - Provides ~7% speedup based on benchmarks
   - Currently not exposed in the builder API

2. **Thread Configuration**: Auto-detection works well
   - Explicit 4 intra-op threads could provide marginal improvement
   - Current implementation is acceptable

3. **Default to INT8**: Consider making INT8 the default for CPU
   - 2.25x faster than FP32
   - >99.5% accuracy preservation

---

## Recommendations

### For Maximum CPU Performance

1. **Use INT8 quantization** (highest impact)
   ```rust
   let model = Colbert::builder("models/GTE-ModernColBERT-v1")
       .with_quantized(true)  // 2.25x speedup
       .build()?;
   ```

2. **Use batch size 16-32** (already default)

3. **Consider parallel sessions for high throughput**
   ```rust
   let model = Colbert::builder("models/GTE-ModernColBERT-v1")
       .with_quantized(true)
       .with_parallel(num_cpus::get())  // One session per core
       .with_batch_size(2)
       .build()?;
   ```

### For GPU Acceleration

- **Apple Silicon**: CoreML execution provider (requires fixing compatibility issues)
- **NVIDIA**: CUDA execution provider (already supported)
- **Windows**: DirectML execution provider (already supported)

### Implementation Priority

| Priority | Optimization | Expected Impact | Effort |
|----------|-------------|-----------------|--------|
| 1 | Use INT8 by default | 2.25x | Low |
| 2 | Fix CoreML support | 2-4x | Medium |
| 3 | Disable memory pattern | 7% | Low |
| 4 | IO binding for GPU | 5-10% | Medium |

---

## Appendix: Test Environment

- **Platform**: macOS Darwin 24.6.0 (Apple Silicon)
- **Model**: lightonai/GTE-ModernColBERT-v1 (149M parameters, 569MB)
- **ONNX Runtime**: 1.23.2
- **PyTorch**: 2.8.0
- **PyLate**: 1.3.4

---

## Conclusion

**The current next-plaid-onnx implementation is already well-optimized.** The main opportunity for improvement is:

1. **INT8 quantization** - Already supported, provides 2.25x speedup
2. **GPU execution providers** - Already supported via feature flags

There are no major algorithmic or implementation changes needed. The ONNX FP32 being slower than PyTorch CPU is expected due to:
- PyTorch's highly optimized CPU kernels (especially with MKL)
- ONNX Runtime's general-purpose nature

INT8 quantization compensates for this and provides faster inference than PyTorch CPU while maintaining >99.5% accuracy.
