#!/usr/bin/env python3

import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "fixtures" / "demo-bundle"
ARTIFACTS_DIR = BUNDLE_DIR / "artifacts"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack_f32(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


def pack_i64(values: list[int]) -> bytes:
    return b"".join(struct.pack("<q", value) for value in values)


def pack_i32(values: list[int]) -> bytes:
    return b"".join(struct.pack("<i", value) for value in values)


def pack_bucket_indices(indices: list[int], nbits: int) -> bytes:
    bit_idx = 0
    output = bytearray((len(indices) * nbits + 7) // 8)
    for bucket in indices:
        for bit in range(nbits):
            bit_value = (bucket >> bit) & 1
            byte_idx = bit_idx // 8
            bit_pos = 7 - (bit_idx % 8)
            output[byte_idx] |= bit_value << bit_pos
            bit_idx += 1
    return bytes(output)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    nbits = 2
    embedding_dim = 4

    centroids = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ]
    ivf = [0, 1]
    ivf_lengths = [1, 1]
    doc_lengths = [2, 2]
    merged_codes = [0, 0, 1, 1]
    bucket_weights = [-0.2, -0.05, 0.05, 0.2]
    merged_residuals = b"".join(
        [
            pack_bucket_indices([3, 1, 1, 0], nbits),
            pack_bucket_indices([2, 0, 0, 1], nbits),
            pack_bucket_indices([1, 3, 0, 0], nbits),
            pack_bucket_indices([0, 2, 1, 1], nbits),
        ]
    )
    metadata = {
        "documents": [
            {"id": 0, "title": "alpha"},
            {"id": 1, "title": "beta"},
        ]
    }

    files = {
        "centroids.bin": pack_f32(centroids),
        "ivf.bin": pack_i64(ivf),
        "ivf_lengths.bin": pack_i32(ivf_lengths),
        "doc_lengths.json": json.dumps(doc_lengths).encode("utf-8"),
        "merged_codes.bin": pack_i64(merged_codes),
        "merged_residuals.bin": merged_residuals,
        "bucket_weights.bin": pack_f32(bucket_weights),
        "metadata.json": json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
    }

    for name, data in files.items():
        (ARTIFACTS_DIR / name).write_bytes(data)

    manifest = {
        "format_version": 1,
        "index_id": "demo-bundle",
        "build_id": "build-demo-001",
        "embedding_dim": embedding_dim,
        "nbits": nbits,
        "document_count": len(doc_lengths),
        "metadata_mode": "inline_json",
        "artifacts": [
            {
                "kind": "centroids",
                "path": "artifacts/centroids.bin",
                "byte_size": len(files["centroids.bin"]),
                "sha256": sha256_hex(files["centroids.bin"]),
                "compression": "none",
            },
            {
                "kind": "ivf",
                "path": "artifacts/ivf.bin",
                "byte_size": len(files["ivf.bin"]),
                "sha256": sha256_hex(files["ivf.bin"]),
                "compression": "none",
            },
            {
                "kind": "ivf_lengths",
                "path": "artifacts/ivf_lengths.bin",
                "byte_size": len(files["ivf_lengths.bin"]),
                "sha256": sha256_hex(files["ivf_lengths.bin"]),
                "compression": "none",
            },
            {
                "kind": "doc_lengths",
                "path": "artifacts/doc_lengths.json",
                "byte_size": len(files["doc_lengths.json"]),
                "sha256": sha256_hex(files["doc_lengths.json"]),
                "compression": "none",
            },
            {
                "kind": "merged_codes",
                "path": "artifacts/merged_codes.bin",
                "byte_size": len(files["merged_codes.bin"]),
                "sha256": sha256_hex(files["merged_codes.bin"]),
                "compression": "none",
            },
            {
                "kind": "merged_residuals",
                "path": "artifacts/merged_residuals.bin",
                "byte_size": len(files["merged_residuals.bin"]),
                "sha256": sha256_hex(files["merged_residuals.bin"]),
                "compression": "none",
            },
            {
                "kind": "bucket_weights",
                "path": "artifacts/bucket_weights.bin",
                "byte_size": len(files["bucket_weights.bin"]),
                "sha256": sha256_hex(files["bucket_weights.bin"]),
                "compression": "none",
            },
            {
                "kind": "metadata_json",
                "path": "artifacts/metadata.json",
                "byte_size": len(files["metadata.json"]),
                "sha256": sha256_hex(files["metadata.json"]),
                "compression": "none",
            },
        ],
    }

    (BUNDLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
