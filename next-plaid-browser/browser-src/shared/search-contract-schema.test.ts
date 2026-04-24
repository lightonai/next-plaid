import { expect, it } from "@effect/vitest";
import { Effect } from "effect";

import type { QueryEmbeddingsPayload } from "./search-contract.js";
import {
  decodeDenseQueryEmbeddingsPayload,
  decodeInlineQueryEmbeddingsPayload,
  isBinaryQueryEmbeddingsPayload,
  isDenseQueryEmbeddingsPayload,
  isEncoderIdentity,
  isInlineQueryEmbeddingsPayload,
  isQueryEmbeddingsPayload,
} from "./search-contract-schema.js";

function proofPayload(): QueryEmbeddingsPayload {
  return {
    encoder: {
      encoder_id: "proof-encoder",
      encoder_build: "proof-build-1",
      embedding_dim: 4,
      normalized: true,
    },
    dtype: "f32_le",
    layout: "ragged",
    embeddings: [[0.1, 0.2, 0.3, 0.4]],
  };
}

it.effect("derives schema guards for encoder identity and query payload variants", () =>
  Effect.sync(() => {
    const inlinePayload = proofPayload();
    const binaryPayload: QueryEmbeddingsPayload = {
      encoder: inlinePayload.encoder,
      dtype: inlinePayload.dtype,
      layout: inlinePayload.layout,
      embeddings_b64: "AQIDBA==",
      shape: [1, 4],
    };

    expect(isEncoderIdentity(inlinePayload.encoder)).toBe(true);
    expect(isQueryEmbeddingsPayload(inlinePayload)).toBe(true);
    expect(isDenseQueryEmbeddingsPayload(inlinePayload)).toBe(true);
    expect(isInlineQueryEmbeddingsPayload(inlinePayload)).toBe(true);
    expect(isBinaryQueryEmbeddingsPayload(inlinePayload)).toBe(false);

    expect(isQueryEmbeddingsPayload(binaryPayload)).toBe(true);
    expect(isDenseQueryEmbeddingsPayload(binaryPayload)).toBe(true);
    expect(isInlineQueryEmbeddingsPayload(binaryPayload)).toBe(false);
    expect(isBinaryQueryEmbeddingsPayload(binaryPayload)).toBe(true);
  }),
);

it.effect("normalizes dense payload decodes back into the generated contract shape", () =>
  Effect.gen(function*() {
    const payload = yield* decodeDenseQueryEmbeddingsPayload({
      encoder: {
        encoder_id: "proof-encoder",
        encoder_build: "proof-build-1",
        embedding_dim: 4,
        normalized: true,
      },
      dtype: "f32_le",
      layout: "ragged",
      embeddings: [[0.1, 0.2, 0.3, 0.4]],
    });

    expect(payload).toEqual(proofPayload());
  }),
);

it.effect("keeps the encoder-side inline payload decoder aligned to the shared contract schema", () =>
  Effect.gen(function*() {
    const payload = yield* decodeInlineQueryEmbeddingsPayload({
      encoder: {
        encoder_id: "proof-encoder",
        encoder_build: "proof-build-1",
        embedding_dim: 4,
        normalized: true,
      },
      dtype: "f32_le",
      layout: "padded_query_length",
      embeddings: [[0.1, 0.2, 0.3, 0.4]],
    });

    expect(isInlineQueryEmbeddingsPayload(payload)).toBe(true);
    expect(payload.layout).toBe("padded_query_length");
  }),
);

it.effect("rejects ambiguous dense payloads at the shared schema layer", () =>
  Effect.gen(function*() {
    const result = yield* Effect.result(
      decodeDenseQueryEmbeddingsPayload({
        encoder: {
          encoder_id: "proof-encoder",
          encoder_build: "proof-build-1",
          embedding_dim: 4,
          normalized: true,
        },
        dtype: "f32_le",
        layout: "ragged",
        embeddings: [[0.1, 0.2, 0.3, 0.4]],
        embeddings_b64: "AQIDBA==",
        shape: [1, 4],
      }),
    );

    expect(result._tag).toBe("Failure");
  }),
);

it.effect("rejects non-finite dense embedding values at the shared schema layer", () =>
  Effect.gen(function*() {
    const result = yield* Effect.result(
      decodeDenseQueryEmbeddingsPayload({
        encoder: {
          encoder_id: "proof-encoder",
          encoder_build: "proof-build-1",
          embedding_dim: 4,
          normalized: true,
        },
        dtype: "f32_le",
        layout: "ragged",
        embeddings: [[0.1, Number.NaN, 0.3, 0.4]],
      }),
    );

    expect(result._tag).toBe("Failure");
  }),
);
