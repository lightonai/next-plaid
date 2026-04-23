import { expect, it } from "@effect/vitest";
import { Effect } from "effect";

import type { PreparedEncoderInput } from "./encoder-preprocessor.js";
import { buildEncodedQuery } from "./wasm-encoder-backend.js";

function preparedInput(activeLength: number): PreparedEncoderInput {
  return {
    inputIds: BigInt64Array.from([101n, 102n, 103n, 0n]),
    attentionMask: BigInt64Array.from([1n, 1n, 1n, 0n]),
    tokenTypeIds: null,
    inputIdValues: [101, 102, 103, 0],
    attentionMaskValues: [1, 1, 1, 0],
    tokenTypeIdValues: null,
    activeLength,
  };
}

const encoder = {
  encoder_id: "proof-encoder",
  encoder_build: "build-1",
  embedding_dim: 2,
  normalized: true,
};

it.effect("returns ragged query embeddings for non-expansion models", () =>
  Effect.sync(() => {
    const encoded = buildEncodedQuery(
      preparedInput(3),
      [[1, 0], [0, 1], [0.5, 0.5], [9, 9]],
      encoder,
      2,
      false,
      10,
      2,
      8,
    );

    expect(encoded.payload.layout).toBe("ragged");
    expect(encoded.payload.embeddings).toEqual([[1, 0], [0, 1], [0.5, 0.5]]);
    expect(encoded.input_ids).toEqual([101, 102, 103]);
    expect(encoded.attention_mask).toEqual([1, 1, 1]);
  }),
);

it.effect("keeps padded query embeddings when query expansion is enabled", () =>
  Effect.sync(() => {
    const encoded = buildEncodedQuery(
      preparedInput(3),
      [[1, 0], [0, 1], [0.5, 0.5], [9, 9]],
      encoder,
      2,
      true,
      10,
      2,
      8,
    );

    expect(encoded.payload.layout).toBe("padded_query_length");
    expect(encoded.payload.embeddings).toEqual([[1, 0], [0, 1], [0.5, 0.5], [9, 9]]);
    expect(encoded.input_ids).toEqual([101, 102, 103, 0]);
    expect(encoded.attention_mask).toEqual([1, 1, 1, 0]);
  }),
);
