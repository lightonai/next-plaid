"""Compare PyLate (Python) vs next-plaid-onnx (Rust) per-token embeddings."""

import json
from pathlib import Path

import numpy as np

BASE = Path("/Users/raphael/Documents/lighton/lategrep/try/parity")


def load(path: Path):
    """Return list of 2D arrays [num_tokens, dim]."""
    data = json.loads(path.read_text())
    return [np.asarray(e, dtype=np.float32) for e in data]


def compare(label: str, py_list, rs_list) -> bool:
    print(f"\n=== {label} ===")
    print(f"  #texts py={len(py_list)} rs={len(rs_list)}")
    if len(py_list) != len(rs_list):
        print("  MISMATCH: different count")
        return False
    ok = True
    for i, (p, r) in enumerate(zip(py_list, rs_list)):
        if p.shape != r.shape:
            print(f"  [text {i}] shape mismatch py={p.shape} rs={r.shape}")
            ok = False
            continue
        diff = np.abs(p - r)
        cos = (p * r).sum(axis=-1) / (
            np.linalg.norm(p, axis=-1) * np.linalg.norm(r, axis=-1) + 1e-12
        )
        print(
            f"  [text {i}] shape={p.shape}  max|diff|={diff.max():.6e}  "
            f"mean|diff|={diff.mean():.6e}  min_cos={cos.min():.8f}  mean_cos={cos.mean():.8f}"
        )
    return ok


def main():
    q_py = load(BASE / "py_query.json")
    q_rs = load(BASE / "rs_query.json")
    d_py = load(BASE / "py_doc.json")
    d_rs = load(BASE / "rs_doc.json")

    compare("QUERIES", q_py, q_rs)
    compare("DOCUMENTS", d_py, d_rs)


if __name__ == "__main__":
    main()
