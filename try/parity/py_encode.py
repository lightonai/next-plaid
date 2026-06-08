"""Encode texts with PyLate and dump per-token embeddings as JSON for parity check."""

import json
import sys
from pathlib import Path

import numpy as np
from pylate import models as pm

MODEL_DIR = Path("/Users/raphael/Documents/lighton/lategrep/models/LateOn")
INPUT = Path("/Users/raphael/Documents/lighton/lategrep/try/parity/texts.json")
OUT_QUERY = Path("/Users/raphael/Documents/lighton/lategrep/try/parity/py_query.json")
OUT_DOC = Path("/Users/raphael/Documents/lighton/lategrep/try/parity/py_doc.json")


def main() -> int:
    with open(INPUT) as f:
        texts = json.load(f)

    # PyLate with do_query_expansion=False matches the export-time setting.
    model = pm.ColBERT(
        model_name_or_path=str(MODEL_DIR),
        device="cpu",
        do_query_expansion=False,
    )

    # Query encoding (padded to query_length, no skiplist filter)
    q_embs = model.encode(
        texts,
        is_query=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=len(texts),
    )

    # Document encoding (skiplist filtering applied by PyLate)
    d_embs = model.encode(
        texts,
        is_query=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=len(texts),
    )

    q_out = [e.tolist() for e in q_embs]
    d_out = [e.tolist() for e in d_embs]

    OUT_QUERY.write_text(json.dumps(q_out))
    OUT_DOC.write_text(json.dumps(d_out))

    print("PyLate query shapes:", [np.array(e).shape for e in q_embs])
    print("PyLate doc shapes:  ", [np.array(e).shape for e in d_embs])
    return 0


if __name__ == "__main__":
    sys.exit(main())
